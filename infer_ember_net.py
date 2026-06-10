"""
infer_ember_net.py — Full-video inference with the trained EmberNet, followed
by the SAME tracking + physical gating stage as the rule-based labeler, so the
detector is the only changed variable.

Per chunk (ignition-aware, like vision_autolabel):
  1. median background / sigma / flame mask / fuel ROI (context for gates)
  2. EmberNet heatmap on 5-frame stacks (GPU, full frames)
  3. peaks >= threshold -> detections with the candidate fields the gates need
Then: ByteTrack + stitching + track gates -> tracks_model.json + events,
plus an overlap summary vs the rule-based labels.

Usage (Spartan):
  python infer_ember_net.py --checkpoint outputs/ember_net/best.pt \
      --frames-dir images/frames --out-dir outputs/ember_net
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ember_net import STACK, EmberNet, extract_peaks  # noqa: E402
from vision_autolabel import (  # noqa: E402
    FPS, build_background, build_chunks, compute_track_features,
    default_params, detect_ignition, flame_mask_and_dist, fuel_roi_and_dist,
    gate_track, list_frames, load_gray, run_tracking, stats_chunk,
    stitch_tracklets, write_rejected_jsonl)


def peak_to_detection(px, py, score, gray, diff, sigma, bg, flame_dist,
                      roi_dist, p, win=5):
    """Build a gate-compatible candidate dict around a heatmap peak."""
    h, w = gray.shape
    x0, x1 = max(0, px - win), min(w, px + win + 1)
    y0, y1 = max(0, py - win), min(h, py + win + 1)
    dwin = diff[y0:y1, x0:x1]
    thr = max(p["diff_threshold"], p["noise_k"] * float(sigma[py, px]))
    blob = dwin >= min(thr, max(float(dwin.max()), 1.0))
    if not blob.any():
        # zero-contrast peak (model fired on static texture): single-pixel
        # blob — downstream track gates will reject it on its merits
        blob = np.zeros_like(dwin, dtype=bool)
        blob[py - y0, px - x0] = True
    ys, xs = np.nonzero(blob)
    bx1, bx2 = x0 + int(xs.min()), x0 + int(xs.max()) + 1
    by1, by2 = y0 + int(ys.min()), y0 + int(ys.max()) + 1
    return {
        "centroid": [float(px), float(py)],
        "bbox": [bx1, by1, bx2, by2],
        "area": int(blob.sum()),
        "score": round(float(score), 4),
        "bright_max": int(gray[y0:y1, x0:x1].max()),
        "bright_mean": round(float(gray[y0:y1, x0:x1][blob].mean()), 1),
        "diff_max": int(dwin.max()),
        "diff_mean": round(float(dwin[blob].mean()), 1),
        "bg_val": int(bg[py, px]),
        "flame_dist": round(float(flame_dist[py, px]), 1),
        "roi_dist": round(float(roi_dist[py, px]), 1),
    }


@torch.no_grad()
def infer_chunk(model, device, chunk, p, thresh, batch_size=8):
    """One ignition-aware chunk -> {frame: [detections]}."""
    paths = [path for _, path in chunk]
    bg, sigma = build_background(paths, max_samples=p["bg_samples"])
    if bg is None:
        return {}
    flame_mask, flame_dist = flame_mask_and_dist(
        bg, p["flame_bg_thresh"], p["flame_buffer"])
    fuel_roi, roi_dist, _ = fuel_roi_and_dist(
        bg, flame_mask, p["warm_bg_thresh"], p["roi_buffer"])

    out = {}
    window = deque(maxlen=STACK)
    ids = deque(maxlen=STACK)
    batch, batch_meta = [], []

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).to(device)
        heat = torch.sigmoid(model(x)).cpu()
        for bi, (cidx, gray, diff) in enumerate(batch_meta):
            dets = []
            for px, py, sc in extract_peaks(heat[bi, 0], thresh):
                if flame_mask[py, px]:
                    continue
                dets.append(peak_to_detection(
                    px, py, sc, gray, diff, sigma, bg,
                    flame_dist, roi_dist, p))
            out[cidx] = dets
        batch.clear()
        batch_meta.clear()

    for idx, path in chunk:
        gray = load_gray(path)
        if gray is None:
            continue
        window.append(gray)
        ids.append(idx)
        if len(window) < STACK:
            continue
        cidx = ids[STACK // 2]
        stack = np.stack(window).astype(np.float32) / 255.0
        med = np.median(stack, axis=0)
        diff_n = np.clip(stack[STACK // 2] - med, 0, 1)
        x = np.concatenate([stack, diff_n[None]], axis=0)
        cgray = window[STACK // 2]
        cdiff = cv2.subtract(cgray, bg).astype(np.float32)
        batch.append(x)
        batch_meta.append((cidx, cgray, cdiff))
        if len(batch) >= batch_size:
            flush()
    flush()
    return out


def main():
    parser = argparse.ArgumentParser(description="EmberNet full-video inference")
    parser.add_argument("--checkpoint", default="outputs/ember_net/best.pt")
    parser.add_argument("--frames-dir", default="images/frames")
    parser.add_argument("--out-dir", default="outputs/ember_net")
    parser.add_argument("--rule-labels", default="outputs/vision_labels/tracks_vision.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--thresh", type=float, default=None,
                        help="default: best-F1 threshold stored in checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=600)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = EmberNet(base=ckpt.get("base_width", 16)).to(device).eval()
    model.load_state_dict(ckpt["model"])
    thresh = args.thresh if args.thresh is not None else ckpt.get("thresh", 0.2)
    print(f"[INFO] checkpoint epoch {ckpt.get('epoch')} valF1 {ckpt.get('f1')}"
          f" | thresh {thresh} | device {device}", flush=True)

    params = default_params()
    params["start"], params["end"] = args.start, args.end
    frames = list_frames(args.frames_dir, args.start, args.end)
    print(f"[INFO] {len(frames)} frames", flush=True)

    # ignition (cheap stats pass, single process is fine on GPU node)
    stats = []
    for i in range(0, len(frames), args.chunk_size):
        stats.extend(stats_chunk({"frames": frames[i:i + args.chunk_size],
                                  "params": params}))
    stats.sort(key=lambda s: s["frame_0based"])
    ignition = detect_ignition(stats, params["ignition_pixels"])
    print(f"[INFO] ignition frame: {ignition}", flush=True)

    chunks = build_chunks(frames, args.chunk_size, ignition)
    candidates = {}
    for ci, chunk in enumerate(chunks):
        candidates.update(infer_chunk(model, device, chunk, params, thresh,
                                      args.batch_size))
        n = sum(len(v) for v in candidates.values())
        print(f"[INFO] chunk {ci + 1}/{len(chunks)} | {n} detections so far",
              flush=True)

    # Stage D: identical tracking + gates as the rule labeler
    frame_indices = [idx for idx, _ in frames]
    raw = run_tracking(frame_indices, candidates, params)
    raw = stitch_tracklets(raw, params)
    accepted, rejected = {}, {}
    reject_counts = {}
    for tid, dets in raw.items():
        dets.sort(key=lambda d: d[0])
        feat = compute_track_features(dets)
        ok, reason = gate_track(feat, params)
        if ok and ignition is not None and feat["start_frame_0based"] < ignition:
            ok, reason = False, "pre_ignition"
        if ok:
            accepted[tid] = (dets, feat)
        else:
            rejected[tid] = (dets, feat, reason)
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
    n_labels = sum(len(d) for d, _ in accepted.values())
    print(f"[INFO] model tracks accepted: {len(accepted)} "
          f"({n_labels} detections) | rejected: {reject_counts}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = []
    for tid, (dets, feat) in accepted.items():
        t = dict(feat)
        t.update({
            "track_id": tid, "label": "firebrand",
            "frames_0based": [f for f, _ in dets],
            "centroids": [d["centroid"] for _, d in dets],
            "scores": [d["score"] for _, d in dets],
        })
        tracks.append(t)
    with open(out_dir / "tracks_model.json", "w") as f:
        json.dump({"params": params, "thresh": thresh,
                   "ignition_frame_0based": ignition,
                   "counts": {"tracks": len(tracks), "detections": n_labels},
                   "tracks": tracks}, f, indent=1)

    with open(out_dir / "detections_model.jsonl", "w") as f:
        per_frame = {}
        for tid, (dets, _) in accepted.items():
            for idx, d in dets:
                e = dict(d)
                e["track_id"] = tid
                per_frame.setdefault(idx, []).append(e)
        for idx, _ in frames:
            dets = per_frame.get(idx, [])
            f.write(json.dumps({"frame": idx + 1, "frame_0based": idx,
                                "n_candidates": len(dets),
                                "detections": dets}) + "\n")

    # Overlap vs rule-based labels (track level, 20px / >=3 frame hits)
    if Path(args.rule_labels).exists():
        rule = json.load(open(args.rule_labels))
        rule_fb = [t for t in rule["tracks"] if t["label"] == "firebrand"]
        rule_by_frame = {}
        for t in rule_fb:
            for fr, c in zip(t["frames_0based"], t["centroids"]):
                rule_by_frame.setdefault(fr, []).append((t["track_id"], c))
        matched_rule = set()
        model_matched = 0
        for tid, (dets, _) in accepted.items():
            hits = {}
            for fr, d in dets:
                cx, cy = d["centroid"]
                for rtid, (rx, ry) in rule_by_frame.get(fr, []):
                    if (rx - cx) ** 2 + (ry - cy) ** 2 <= 400:
                        hits[rtid] = hits.get(rtid, 0) + 1
            good = [r for r, h in hits.items() if h >= 3]
            if good:
                model_matched += 1
                matched_rule.update(good)
        summary = {
            "model_tracks": len(accepted),
            "model_matching_rule": model_matched,
            "model_new_tracks": len(accepted) - model_matched,
            "rule_tracks": len(rule_fb),
            "rule_recovered_by_model": len(matched_rule),
            "rule_recall": round(len(matched_rule) / max(len(rule_fb), 1), 4),
        }
        print("[INFO] overlap vs rule labels:", json.dumps(summary), flush=True)
        with open(out_dir / "model_vs_rule.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"[INFO] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
