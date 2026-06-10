"""
compare_vision_temp_labels.py — Agreement report: vision-only labels vs the
temperature pipeline. Comparison/evaluation only — temperature data never
feeds the vision labels.

Inputs (Spartan paths):
  outputs/vision_labels/detections_vision.jsonl  vision-only labels
  outputs/detections_refined_0-25467.jsonl       raw temp-threshold detections
  outputs/tracks_labeled.json                    physics-labeled temp tracks

Outputs:
  outputs/vision_labels/agreement_report.json    metrics
  outputs/vision_labels/qa_disagreements.png     vision-only dets (top) and
                                                 temp-firebrand dets missed by
                                                 vision (bottom)
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

MATCH_DIST = 8.0
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_per_frame_jsonl(path, swap_xy=False):
    """swap_xy: the temp pipeline stores centroids as (y, x) — refine_detections.py
    unpacks cv2 centroids in the wrong order — so they must be swapped to (x, y)
    before geometric comparison (verified against bboxes + actual frames)."""
    per_frame = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("n_candidates", 0) > 0:
                pts = []
                for d in e["detections"]:
                    a, b = float(d["centroid"][0]), float(d["centroid"][1])
                    pts.append((b, a) if swap_xy else (a, b))
                per_frame[e["frame_0based"]] = pts
    return per_frame


def load_labeled_tracks(path, swap_xy=False):
    """Per-frame centroids of physics-labeled tracks, by label; plus track list.
    Track centroids inherit the (y, x) swap from refine_detections.py."""
    with open(path) as f:
        data = json.load(f)
    per_frame = {"firebrand": {}, "not_firebrand": {}, "ambiguous": {}}
    fb_tracks = []
    for t in data["tracks"]:
        if swap_xy:
            t = dict(t)
            t["centroids"] = [[c[1], c[0]] for c in t["centroids"]]
        label = t.get("label", "ambiguous")
        if label == "firebrand":
            fb_tracks.append(t)
        store = per_frame[label]
        for i, fr in enumerate(t["frames_0based"]):
            store.setdefault(fr, []).append(
                (float(t["centroids"][i][0]), float(t["centroids"][i][1])))
    return per_frame, fb_tracks


def greedy_match(a_pts, b_pts, max_dist=MATCH_DIST):
    """Greedy nearest matching; returns set of matched indices in a and in b."""
    if not a_pts or not b_pts:
        return set(), set()
    a = np.array(a_pts)
    b = np.array(b_pts)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    matched_a, matched_b = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    for ai, bi in order:
        if d[ai, bi] > max_dist:
            break
        if ai in matched_a or bi in matched_b:
            continue
        matched_a.add(int(ai))
        matched_b.add(int(bi))
    return matched_a, matched_b


def crop_tile(frames_dir, frame_idx, cx, cy, text, size=64, color=(0, 255, 0)):
    path = Path(frames_dir) / f"frame_{frame_idx + 1:05d}.png"
    img = cv2.imread(str(path))
    if img is None:
        return None
    half = size // 2
    x1, y1 = max(0, int(cx) - half), max(0, int(cy) - half)
    x2, y2 = min(img.shape[1], int(cx) + half), min(img.shape[0], int(cy) + half)
    tile = img[y1:y2, x1:x2]
    tile = cv2.copyMakeBorder(tile, 0, max(0, size - tile.shape[0]),
                              0, max(0, size - tile.shape[1]),
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.circle(tile, (int(cx) - x1, int(cy) - y1), 5, color, 1)
    tile = cv2.resize(tile, (size * 2, size * 2), interpolation=cv2.INTER_NEAREST)
    cv2.putText(tile, text, (3, 12), FONT, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(tile, text, (3, 12), FONT, 0.35, color, 1, cv2.LINE_AA)
    return tile


def grid(tiles, cols=5):
    if not tiles:
        return None
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser(description="Vision vs temperature label agreement")
    parser.add_argument("--vision", default="outputs/vision_labels/detections_vision.jsonl")
    parser.add_argument("--temp-raw", default="outputs/detections_refined_0-25467.jsonl")
    parser.add_argument("--temp-tracks", default="outputs/tracks_labeled.json")
    parser.add_argument("--frames-dir", default="images/frames")
    parser.add_argument("--out", default="outputs/vision_labels/agreement_report.json")
    parser.add_argument("--match-dist", type=float, default=MATCH_DIST)
    parser.add_argument("--no-swap-temp", action="store_true",
                        help="Disable the (y,x)->(x,y) fix for temp centroids")
    args = parser.parse_args()
    swap = not args.no_swap_temp

    print("[INFO] Loading label sets...", flush=True)
    vision = load_per_frame_jsonl(args.vision)
    temp_raw = load_per_frame_jsonl(args.temp_raw, swap_xy=swap)
    temp_labeled, fb_tracks = load_labeled_tracks(args.temp_tracks, swap_xy=swap)
    temp_fb = temp_labeled["firebrand"]

    # Compare only over the frame range the vision run covered (pilot-safe)
    v_frames = sorted(vision)
    if not v_frames:
        print("[ERROR] No vision labels found")
        return
    lo, hi = v_frames[0], v_frames[-1]
    print(f"[INFO] Vision label range: frames {lo}..{hi}", flush=True)

    n_vision = n_temp_fb = 0
    vision_matched_fb = 0          # vision dets matching a physics-firebrand det
    vision_matched_any = 0         # vision dets matching ANY raw temp det
    temp_fb_recalled = 0           # physics-firebrand dets recovered by vision
    vision_only_samples = []       # (frame, cx, cy)
    temp_missed_samples = []

    all_frames = sorted(set(list(vision) +
                            [f for f in temp_fb if lo <= f <= hi]))
    for fr in all_frames:
        v = vision.get(fr, [])
        tfb = temp_fb.get(fr, [])
        traw = temp_raw.get(fr, [])
        n_vision += len(v)
        n_temp_fb += len(tfb)

        mv_fb, mt_fb = greedy_match(v, tfb, args.match_dist)
        vision_matched_fb += len(mv_fb)
        temp_fb_recalled += len(mt_fb)

        mv_any, _ = greedy_match(v, traw, args.match_dist)
        vision_matched_any += len(mv_any)

        for i, (cx, cy) in enumerate(v):
            if i not in mv_any:
                vision_only_samples.append((fr, cx, cy))
        for i, (cx, cy) in enumerate(tfb):
            if i not in mt_fb:
                temp_missed_samples.append((fr, cx, cy))

    # Track-level recall of physics-labeled firebrand tracks. Also computed for
    # the "confirmed flying" subset: temp tracks that genuinely exit the fuel
    # region (reference box from the auto-derived ROI) — the temp-FB set is
    # otherwise contaminated with plume fragments (coordinate-swap bug).
    BX1, BY1, BX2, BY2 = 270, 350, 500, 540

    def dist_outside_box(x, y):
        dx = max(BX1 - x, 0, x - BX2)
        dy = max(BY1 - y, 0, y - BY2)
        return (dx * dx + dy * dy) ** 0.5

    tracks_recovered = 0
    n_fb_tracks_in_range = 0
    flying_recovered = 0
    n_flying_in_range = 0
    for t in fb_tracks:
        pts = [(fr, c) for fr, c in zip(t["frames_0based"], t["centroids"])
               if lo <= fr <= hi]
        if len(pts) < 3:
            continue
        n_fb_tracks_in_range += 1
        is_flying = max(dist_outside_box(c[0], c[1]) for _, c in pts) >= 30
        if is_flying:
            n_flying_in_range += 1
        hits = 0
        for fr, (cx, cy) in pts:
            for vx, vy in vision.get(fr, []):
                if ((vx - cx) ** 2 + (vy - cy) ** 2) ** 0.5 <= args.match_dist:
                    hits += 1
                    break
        if hits >= max(3, len(pts) // 2):
            tracks_recovered += 1
            if is_flying:
                flying_recovered += 1

    report = {
        "frame_range": [lo, hi],
        "match_dist_px": args.match_dist,
        "temp_centroids_swapped_yx_to_xy": swap,
        "vision_detections": n_vision,
        "temp_firebrand_detections": n_temp_fb,
        "detection_level": {
            "temp_fb_recall_by_vision": round(temp_fb_recalled / n_temp_fb, 4)
            if n_temp_fb else None,
            "vision_matching_temp_fb": round(vision_matched_fb / n_vision, 4)
            if n_vision else None,
            "vision_matching_any_temp": round(vision_matched_any / n_vision, 4)
            if n_vision else None,
            "vision_only_fraction": round(1 - vision_matched_any / n_vision, 4)
            if n_vision else None,
        },
        "track_level": {
            "temp_fb_tracks_in_range": n_fb_tracks_in_range,
            "recovered_by_vision": tracks_recovered,
            "recall": round(tracks_recovered / n_fb_tracks_in_range, 4)
            if n_fb_tracks_in_range else None,
        },
        "confirmed_flying_temp_tracks": {
            "in_range": n_flying_in_range,
            "recovered_by_vision": flying_recovered,
            "recall": round(flying_recovered / n_flying_in_range, 4)
            if n_flying_in_range else None,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Vision vs Temperature Agreement ===")
    print(json.dumps(report, indent=2))

    # Disagreement grid: vision-only (top, yellow) / temp-FB missed (bottom, red)
    random.seed(42)
    random.shuffle(vision_only_samples)
    random.shuffle(temp_missed_samples)
    tiles = []
    for fr, cx, cy in vision_only_samples[:10]:
        t = crop_tile(args.frames_dir, fr, cx, cy, f"vis-only f{fr}",
                      color=(0, 255, 255))
        if t is not None:
            tiles.append(t)
    for fr, cx, cy in temp_missed_samples[:10]:
        t = crop_tile(args.frames_dir, fr, cx, cy, f"temp-missed f{fr}",
                      color=(0, 0, 255))
        if t is not None:
            tiles.append(t)
    g = grid(tiles)
    if g is not None:
        png = out_path.parent / "qa_disagreements.png"
        cv2.imwrite(str(png), g)
        print(f"[INFO] Disagreement grid: {png}")


if __name__ == "__main__":
    main()
