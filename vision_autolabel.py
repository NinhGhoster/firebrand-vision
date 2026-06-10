"""
vision_autolabel.py — Vision-only automatic firebrand labeling + scene events.

No temperature data is used anywhere: frames are treated as ordinary video
(grayscale intensity). This works because the camera is static and
extract_frames.py rendered all PNGs with a FIXED 20-600C -> inferno mapping,
so pixel intensities are temporally consistent across the whole recording.

Stages:
  A. Scene events — per-frame brightness stats -> ignition frame
  B. Background   — per-chunk temporal median (embers are transient -> erased)
                    + persistent-bright flame/burner mask (dilated)
  C. Candidates   — (frame - background) > diff threshold, outside flame mask,
                    connected components, area + peak-contrast filters
  D. Labeling     — ByteTrack association + track-level motion gates
                    (lifetime, displacement, speed, angle consistency,
                     start distance from flame) -> firebrand labels

Outputs (under --out-dir, default outputs/vision_labels):
  detections_vision.jsonl  per-frame accepted firebrand detections (same schema
                           family as detections_refined_*.jsonl)
  tracks_vision.json       track-level labels + motion features
  labels_coco.json         COCO export (training-ready)
  yolo/*.txt               YOLO export (labeled frames only)
  qa_onset_strip.png       frames around ignition + first firebrand
  qa_grid.png              sample labeled crops
  qa_clip_*.mp4            short annotated clips
  qa_flux.png              labels per second over time
  outputs/vision_events.json  ignition + firebrand onset + flux (--events-out)

Usage:
  python vision_autolabel.py --frames-dir images/frames --out-dir outputs/vision_labels
  python vision_autolabel.py --frames-dir images/frames --start 0 --end 4000   # pilot
  python vision_autolabel.py --self-test
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytetrack_tracker import ByteTrackTracker  # noqa: E402

FPS = 29.41


# ----------------------------------------------------------------------------
# Frame listing
# ----------------------------------------------------------------------------

def list_frames(frames_dir, start=0, end=-1):
    """Return sorted [(frame_0based, path)]. Filenames are 1-based (frame_00001.png)."""
    frames = []
    for name in os.listdir(frames_dir):
        if not name.endswith(".png"):
            continue
        nums = re.findall(r"\d+", name)
        if not nums:
            continue
        frames.append((int(nums[-1]) - 1, os.path.join(frames_dir, name)))
    frames.sort(key=lambda t: t[0])
    if end is not None and end >= 0:
        frames = [f for f in frames if start <= f[0] < end]
    else:
        frames = [f for f in frames if f[0] >= start]
    return frames


def load_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# ----------------------------------------------------------------------------
# Stage B/C: per-chunk background, flame mask, candidate detection
# ----------------------------------------------------------------------------

def build_background(paths, max_samples=30):
    """Temporal median + per-pixel noise sigma of sampled frames. Transient
    embers vanish in the median; sigma is large in flame/smoke flicker regions
    and ~1-2 in quiet sky, driving the adaptive detection threshold."""
    stride = max(1, len(paths) // max_samples)
    samples = [load_gray(p) for p in paths[::stride]]
    samples = [s for s in samples if s is not None]
    if not samples:
        return None, None
    stack = np.stack(samples).astype(np.float32)
    bg = np.median(stack, axis=0).astype(np.uint8)
    # sigma from the 16-84 percentile spread: a 2-3 frame ember transit cannot
    # inflate it (robust), but persistent smoke/flame flicker still widens it,
    # raising the threshold exactly where false transients live
    p16, p84 = np.percentile(stack, [16, 84], axis=0)
    sigma = ((p84 - p16) / 2).astype(np.float32)
    return bg, sigma


def flame_mask_and_dist(bg, flame_bg_thresh, buffer_px):
    """Persistent-bright region of the background = flame/burner/hot plate.
    Returns (dilated mask bool, distance-to-mask map float32)."""
    mask = (bg >= flame_bg_thresh).astype(np.uint8)
    if buffer_px > 0:
        mask = cv2.dilate(mask, np.ones((buffer_px, buffer_px), np.uint8))
    if mask.max() == 0:
        dist = np.full(bg.shape, 1e6, dtype=np.float32)
    else:
        dist = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)
    return mask.astype(bool), dist


def fuel_roi_and_dist(bg, flame_mask, warm_thresh, buffer_px):
    """Fuel/burner zone: persistently WARM background components that touch the
    flame region (the fuel sample, the burner metal, glowing bark/twigs), plus
    the flame mask itself, dilated. Firebrands must EXIT this zone — bright
    spots that stay inside are burning fuel or hot metal, not firebrands.
    Returns (roi bool, distance-outside-roi float32)."""
    warm = (bg >= warm_thresh).astype(np.uint8)
    roi = flame_mask.astype(np.uint8).copy()
    if warm.max() > 0 and flame_mask.any():
        n, lab = cv2.connectedComponents(warm, connectivity=8)
        touching = np.unique(lab[flame_mask])
        for comp in touching:
            if comp != 0:
                roi[lab == comp] = 1
    core_bbox = None
    if roi.max() > 0:
        ys, xs = np.nonzero(roi)
        core_bbox = [int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1]
    if buffer_px > 0 and roi.max() > 0:
        roi = cv2.dilate(roi, np.ones((buffer_px, buffer_px), np.uint8))
    if roi.max() == 0:
        dist = np.full(bg.shape, 1e6, dtype=np.float32)
    else:
        dist = cv2.distanceTransform((1 - roi).astype(np.uint8), cv2.DIST_L2, 3)
    return roi.astype(bool), dist, core_bbox


def detect_candidates(gray, bg, sigma, flame_mask, flame_dist, roi_dist, p):
    """Foreground blobs outside the flame region -> candidate detections.
    Threshold adapts to per-pixel noise: low in quiet sky (dim embers), high
    where flame/smoke flicker (false transients)."""
    diff = cv2.subtract(gray, bg).astype(np.float32)  # clamps at 0: brighter-than-bg only
    thr_map = np.maximum(p["diff_threshold"], p["noise_k"] * sigma)

    # Suppress large bright transients (flame flicker, hot gas plumes); the
    # largest flame-scale component is also the per-frame FLAME scene object
    bright = (gray >= p["flame_bright"]).astype(np.uint8)
    n_b, lab_b, st_b, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    big = np.zeros_like(bright)
    flame_bbox, flame_area = None, 0
    for i in range(1, n_b):
        a = int(st_b[i, cv2.CC_STAT_AREA])
        if a > p["big_bright_area"]:
            big[lab_b == i] = 1
            if a > flame_area:
                flame_area = a
                flame_bbox = [int(st_b[i, cv2.CC_STAT_LEFT]),
                              int(st_b[i, cv2.CC_STAT_TOP]),
                              int(st_b[i, cv2.CC_STAT_LEFT] + st_b[i, cv2.CC_STAT_WIDTH]),
                              int(st_b[i, cv2.CC_STAT_TOP] + st_b[i, cv2.CC_STAT_HEIGHT])]
    if big.max() > 0:
        big = cv2.dilate(big, np.ones((p["transient_buffer"], p["transient_buffer"]), np.uint8))

    fg = (diff >= thr_map).astype(np.uint8)
    fg[flame_mask] = 0
    fg[big.astype(bool)] = 0

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)

    # Large foreground transients (plume bursts, flame flicker) shed soft edge
    # slivers; exclude a dilated zone around them so slivers can't be candidates
    big_fg = np.zeros_like(fg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > p["max_area"]:
            big_fg[labels == i] = 1
    if big_fg.max() > 0:
        big_fg = cv2.dilate(big_fg, np.ones((p["transient_buffer"],
                                             p["transient_buffer"]), np.uint8))

    img_h, img_w = gray.shape
    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p["min_area"] or area > p["max_area"]:
            continue
        ci_x, ci_y = int(centroids[i][0]), int(centroids[i][1])
        if big_fg[ci_y, ci_x]:
            continue
        blob = labels == i
        if float(diff[blob].max()) < p["peak_diff"]:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        # blobs touching the frame border are clipped fragments of larger
        # objects (e.g. plume leaving the frame) — size/centroid unreliable
        if x <= 2 or y <= 2 or x + w >= img_w - 2 or y + h >= img_h - 2:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        cands.append({
            "centroid": [round(cx, 1), round(cy, 1)],
            "bbox": [x, y, x + w, y + h],
            "area": area,
            "bright_max": int(gray[blob].max()),
            "bright_mean": round(float(gray[blob].mean()), 1),
            "diff_max": int(diff[blob].max()),
            "diff_mean": round(float(diff[blob].mean()), 1),
            "bg_val": int(bg[int(cy), int(cx)]),
            "flame_dist": round(float(flame_dist[int(cy), int(cx)]), 1),
            "roi_dist": round(float(roi_dist[int(cy), int(cx)]), 1),
        })
    return cands, {"flame_bbox": flame_bbox, "flame_area": flame_area}


def stats_chunk(job):
    """Worker pass 1: lightweight per-frame brightness stats (for ignition)."""
    frames, p = job["frames"], job["params"]
    out = []
    for idx, path in frames:
        gray = load_gray(path)
        if gray is None:
            continue
        out.append({
            "frame_0based": idx,
            "mean": round(float(gray.mean()), 2),
            "p99": int(np.percentile(gray, 99)),
            "bright_count": int((gray >= p["bright_px_thresh"]).sum()),
        })
    return out


def process_chunk(job):
    """Worker pass 2: one chunk of frames -> per-frame candidates.

    Chunks never straddle the ignition frame (see build_chunks): a chunk's
    median background is either fully pre-ignition (no flame -> no candidates)
    or fully post-ignition (stable flame mask). Mixing the two dilutes the
    background and leaves the flame-spread transient unmasked."""
    frames, p = job["frames"], job["params"]
    paths = [path for _, path in frames]
    bg, sigma = build_background(paths, max_samples=p["bg_samples"])
    cands_out, scene_out = {}, {}
    if bg is None:
        return cands_out, scene_out
    flame_mask, flame_dist = flame_mask_and_dist(bg, p["flame_bg_thresh"], p["flame_buffer"])
    fuel_roi, roi_dist, fuel_bbox = fuel_roi_and_dist(
        bg, flame_mask, p["warm_bg_thresh"], p["roi_buffer"])
    for idx, path in frames:
        gray = load_gray(path)
        if gray is None:
            continue
        cands_out[idx], scene = detect_candidates(
            gray, bg, sigma, flame_mask, flame_dist, roi_dist, p)
        scene["fuel_bbox"] = fuel_bbox
        scene_out[idx] = scene
    return cands_out, scene_out


def build_chunks(frames, chunk_size, ignition):
    """Split frames into chunks that do not straddle ignition; the first
    post-ignition chunk is kept short so the background adapts to the
    rapidly growing flame."""
    if ignition is None:
        return [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]
    pre = [f for f in frames if f[0] < ignition]
    post = [f for f in frames if f[0] >= ignition]
    chunks = [pre[i:i + chunk_size] for i in range(0, len(pre), chunk_size)]
    first_len = min(max(chunk_size // 2, 150), chunk_size)
    if post:
        chunks.append(post[:first_len])
        rest = post[first_len:]
        chunks.extend(rest[i:i + chunk_size] for i in range(0, len(rest), chunk_size))
    return [c for c in chunks if c]


# ----------------------------------------------------------------------------
# Stage A: ignition from brightness statistics
# ----------------------------------------------------------------------------

def detect_ignition(stats, min_pixels, min_frames=3):
    """First sustained jump in bright-pixel count (same logic as detect_ignition.py)."""
    consecutive = 0
    for s in stats:
        if s["bright_count"] >= min_pixels:
            consecutive += 1
            if consecutive >= min_frames:
                return s["frame_0based"] - min_frames + 1
        else:
            consecutive = 0
    return None


def detect_states(stats, scenes, ignition, p):
    """Scene-state timeline: pre_ignition -> burning -> (burner_off:
    smoldering | extinguished). 'Smoldering' = gas flame gone but glowing
    spots remain on the fuel/charcoal — distinct from extinguished."""
    if ignition is None:
        last = stats[-1]["frame_0based"] if stats else 0
        return [{"state": "pre_ignition", "start_frame": 0,
                 "end_frame": last}], None

    glow = {s["frame_0based"]: s["bright_count"] > 0 for s in stats}
    ordered = sorted(s["frame_0based"] for s in stats if
                     s["frame_0based"] >= ignition)
    if not ordered:
        return [{"state": "pre_ignition", "start_frame": 0,
                 "end_frame": stats[-1]["frame_0based"]}], None

    raw_on = np.array([scenes.get(f, {}).get("flame_area", 0)
                       >= p["flame_on_area"] for f in ordered])

    # burner off = the flame goes away and NEVER comes back (a lull followed
    # by flame return is still 'burning' — e.g. right after the ignition
    # flare while the fuel catches). Anchor on the last SUSTAINED (>=3 frame)
    # flame run so an isolated spurious spike can't postpone it.
    burner_off = None
    if raw_on.any():
        last_on = int(np.nonzero(raw_on)[0][-1])
        for i in reversed(np.nonzero(raw_on)[0]):
            if i >= 2 and raw_on[i - 2:i + 1].all():
                last_on = int(i)
                break
        tail = len(ordered) - 1 - last_on
        if tail >= p["burner_off_min_frames"]:
            burner_off = ordered[last_on + 1]
    else:
        burner_off = ordered[0]

    states = []
    if ignition > 0:
        states.append({"state": "pre_ignition", "start_frame": 0,
                       "end_frame": ignition - 1})
    last = ordered[-1]
    if burner_off is None:
        states.append({"state": "burning", "start_frame": ignition,
                       "end_frame": last})
    else:
        if burner_off > ignition:
            states.append({"state": "burning", "start_frame": ignition,
                           "end_frame": burner_off - 1})
        post = [f for f in ordered if f >= burner_off]
        smold = any(glow.get(f, False) for f in post)
        states.append({"state": "smoldering" if smold else "extinguished",
                       "start_frame": burner_off, "end_frame": last})
    return states, burner_off


def derive_burner_bbox(frames, ignition, p, n_samples=60):
    """Static gas-burner box: the region that stays bright across the WHOLE
    post-ignition burn (flame flickers and moves; the hot metal base does
    not) in a global median background."""
    post = [f for f in frames if ignition is None or f[0] >= ignition]
    if not post:
        return None
    stride = max(1, len(post) // n_samples)
    bg, _ = build_background([p_ for _, p_ in post[::stride]],
                             max_samples=n_samples)
    if bg is None:
        return None
    mask = (bg >= p["flame_bg_thresh"]).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, area = None, 0
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a > area:
            area = a
            best = [int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                    int(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH]),
                    int(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT])]
    return best


# ----------------------------------------------------------------------------
# Stage D: tracking + track-level gates
# ----------------------------------------------------------------------------

def run_tracking(frame_indices, candidates, p):
    """ByteTrack association. Only detections matched in their own frame are
    recorded (coasting trackers keep a stale _last_det and must be skipped)."""
    tracker = ByteTrackTracker(
        max_distance=p["max_distance"],
        min_hits=3,
        max_age=p["max_age"],
        area_threshold=p["track_area_threshold"],
    )
    raw_tracks = {}
    for idx in frame_indices:
        dets = candidates.get(idx, [])
        active = tracker.update(dets)
        det_ids = {id(d) for d in dets}
        for t in active:
            if id(getattr(t, "_last_det", None)) in det_ids:
                raw_tracks.setdefault(t._id, []).append((idx, t._last_det))
    return raw_tracks


def stitch_tracklets(raw_tracks, p):
    """Re-link tracklets broken by missed detections (fast or briefly-dim
    embers): tracklet B starting a few frames after tracklet A ends, near A's
    extrapolated position, is the same ember."""
    tracks = {tid: list(dets) for tid, dets in raw_tracks.items()}
    by_start = {}
    for tid, dets in tracks.items():
        by_start.setdefault(dets[0][0], []).append(tid)
    consumed = set()
    out = {}
    for tid in sorted(tracks, key=lambda t: tracks[t][0][0]):
        if tid in consumed:
            continue
        dets = tracks[tid]
        while True:
            fa, da = dets[-1]
            ca = np.array(da["centroid"], dtype=float)
            k = max(0, len(dets) - 4)
            f0, d0 = dets[k]
            vel = ((ca - np.array(d0["centroid"], dtype=float)) / max(fa - f0, 1)
                   if fa > f0 else np.zeros(2))
            best, best_d = None, p["stitch_dist"]
            for gap in range(1, p["stitch_gap"] + 1):
                for tid_b in by_start.get(fa + gap, []):
                    if tid_b in consumed or tid_b == tid:
                        continue
                    cb = np.array(tracks[tid_b][0][1]["centroid"], dtype=float)
                    d = float(np.linalg.norm(cb - (ca + vel * gap)))
                    if d < best_d:
                        best, best_d = tid_b, d
            if best is None:
                break
            consumed.add(best)
            dets = dets + tracks[best]
        out[tid] = dets
    return out


def compute_track_features(dets):
    """Motion features for one track. dets = [(frame_0based, det_dict)] sorted."""
    frames = [f for f, _ in dets]
    cents = np.array([d["centroid"] for _, d in dets], dtype=np.float64)
    lifetime = frames[-1] - frames[0] + 1
    total_dist, avg_speed, consistency = 0.0, 0.0, 0.0
    if len(cents) > 1:
        vecs = np.diff(cents, axis=0)
        norms = np.linalg.norm(vecs, axis=1)
        total_dist = float(norms.sum())
        avg_speed = total_dist / max(lifetime - 1, 1)
        moving = vecs[norms > 1e-3]
        if len(moving) >= 2:
            angles = np.arctan2(moving[:, 1], moving[:, 0])
            d_ang = np.diff(angles)
            d_ang = (d_ang + np.pi) % (2 * np.pi) - np.pi
            consistency = max(0.0, 1.0 - float(np.std(d_ang) / np.pi))
    roi_dists = [d.get("roi_dist", 1e6) for _, d in dets]
    return {
        "lifetime_frames": int(lifetime),
        "lifetime_s": round(lifetime / FPS, 2),
        "start_frame_0based": frames[0],
        "end_frame_0based": frames[-1],
        "n_detections": len(dets),
        "total_distance_px": round(total_dist, 1),
        "avg_speed_px_per_frame": round(avg_speed, 2),
        "angle_consistency": round(consistency, 3),
        "start_flame_dist": dets[0][1].get("flame_dist", 1e6),
        "start_roi_dist": round(float(roi_dists[0]), 1),
        "max_roi_dist": round(float(max(roi_dists)), 1),
        "end_roi_dist": round(float(roi_dists[-1]), 1),
        "start_centroid": [round(cents[0][0], 1), round(cents[0][1], 1)],
        "end_centroid": [round(cents[-1][0], 1), round(cents[-1][1], 1)],
        "bright_start": dets[0][1]["bright_max"],
        "bright_end": dets[-1][1]["bright_max"],
        "bright_peak": max(d["bright_max"] for _, d in dets),
        # point sources (embers) have a sharp peak over their blob; diffuse
        # glow (reflections, illumination, haze) is flat
        "sharpness": round(float(np.median(
            [d["diff_max"] / max(d.get("diff_mean", d["diff_max"]), 1.0)
             for _, d in dets])), 2),
        # ember streaks are thin even when motion-blurred; glow patches are wide
        "min_dim_med": round(float(np.median(
            [min(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1])
             for _, d in dets])), 1),
        # flying embers cross open air (dark background); reflections and
        # illumination patterns live on the warm structures beneath them
        "bg_med": round(float(np.median(
            [d.get("bg_val", 0) for _, d in dets])), 1),
        "y_med": round(float(np.median([c[1] for c in cents])), 1),
    }


def gate_track(feat, p):
    """Return (accepted, reject_reason)."""
    if feat["lifetime_frames"] < p["min_lifetime"]:
        return False, "lifetime"
    if feat["total_distance_px"] < p["min_displacement"]:
        return False, "displacement"
    if feat["avg_speed_px_per_frame"] < p["min_speed"]:
        return False, "speed"
    if feat["angle_consistency"] < p["min_consistency"]:
        return False, "consistency"
    if feat["max_roi_dist"] < p["min_exit_dist"]:
        # never flew clear of the fuel/burner zone: burning fuel surface,
        # glowing burner metal, or flame crawling along the sample
        return False, "never_exits_fuel_roi"
    if feat["bright_peak"] < p["min_track_bright"]:
        # never even briefly bright enough to be burning material:
        # floor glints, heat shimmer, reflections
        return False, "too_dim"
    if (feat["sharpness"] < p["min_sharpness"]
            and feat["min_dim_med"] > p["max_thin_dim"]):
        # flat AND wide blob: firelight reflections / illumination patches /
        # haze. (Flat but THIN is a motion-blurred ember streak — keep it.)
        return False, "diffuse"
    if feat["bg_med"] > p["max_track_bg"]:
        # the track spends its life over warm structure, not open air:
        # reflections/illumination sliding on the table, floor, or rig
        return False, "on_structure"
    if feat["y_med"] > p["floor_y"]:
        # below the table plane: flame light and ember mirror-images playing
        # on the floor. Flying firebrands live above the horizon; an ember
        # that drops below it has landed.
        return False, "below_horizon"
    return True, None


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------

def write_outputs(out_dir, frames, accepted, stats, events, params, img_shape,
                  scenes=None, burner_bbox=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = img_shape

    # Per-frame accepted detections, keyed by frame
    per_frame = {}
    for tid, dets in accepted.items():
        for idx, d in dets:
            entry = dict(d)
            entry["track_id"] = tid
            per_frame.setdefault(idx, []).append(entry)

    jsonl_path = out_dir / "detections_vision.jsonl"
    with open(jsonl_path, "w") as f:
        for idx, _ in frames:
            dets = per_frame.get(idx, [])
            f.write(json.dumps({
                "frame": idx + 1,
                "frame_0based": idx,
                "n_candidates": len(dets),
                "detections": dets,
            }) + "\n")

    # COCO export — multi-class: firebrand + scene context objects
    scenes = scenes or {}
    images, annotations = [], []
    ann_id = 1
    coco_frames = sorted(set(per_frame) | set(scenes))
    for idx in coco_frames:
        images.append({
            "id": idx,
            "file_name": f"frame_{idx + 1:05d}.png",
            "width": w, "height": h,
        })
        for d in per_frame.get(idx, []):
            x1, y1, x2, y2 = d["bbox"]
            annotations.append({
                "id": ann_id, "image_id": idx, "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": d["area"], "iscrowd": 0,
                "attributes": {"track_id": d["track_id"]},
            })
            ann_id += 1
        sc = scenes.get(idx, {})
        for cat_id, box in [(2, sc.get("flame_bbox")),
                            (3, sc.get("fuel_bbox")),
                            (4, burner_bbox)]:
            if box:
                x1, y1, x2, y2 = box
                annotations.append({
                    "id": ann_id, "image_id": idx, "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                })
                ann_id += 1
    with open(out_dir / "labels_coco.json", "w") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "firebrand"},
                           {"id": 2, "name": "flame"},
                           {"id": 3, "name": "fuel_stringybark"},
                           {"id": 4, "name": "gas_burner"}],
        }, f)

    # per-frame scene objects + state (for renderers / training context)
    if scenes:
        with open(out_dir / "scene_objects.jsonl", "w") as f:
            for idx in sorted(scenes):
                sc = scenes[idx]
                f.write(json.dumps({
                    "frame_0based": idx,
                    "flame_bbox": sc.get("flame_bbox"),
                    "flame_area": sc.get("flame_area", 0),
                    "fuel_bbox": sc.get("fuel_bbox"),
                    "burner_bbox": burner_bbox,
                }) + "\n")

    # YOLO export (labeled frames only)
    yolo_dir = out_dir / "yolo"
    yolo_dir.mkdir(exist_ok=True)
    for idx in sorted(per_frame):
        with open(yolo_dir / f"frame_{idx + 1:05d}.txt", "w") as f:
            for d in per_frame[idx]:
                x1, y1, x2, y2 = d["bbox"]
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    return per_frame, jsonl_path


def write_rejected_jsonl(out_dir, rejected):
    """Flat dump of rejected-track detections: hard negatives for training the
    learned detector (attached glow, reflections, glints, shimmer...)."""
    path = Path(out_dir) / "rejected_detections.jsonl"
    with open(path, "w") as f:
        for tid, (dets, feat, reason) in rejected.items():
            for idx, d in dets:
                f.write(json.dumps({
                    "frame_0based": idx,
                    "centroid": d["centroid"],
                    "bbox": d["bbox"],
                    "track_id": tid,
                    "reason": reason,
                }) + "\n")
    return path


def write_tracks_json(out_dir, accepted, rejected, events, params):
    tracks = []
    for tid, (dets, feat) in accepted.items():
        t = dict(feat)
        t.update({
            "track_id": tid, "label": "firebrand",
            "frames_0based": [f for f, _ in dets],
            "centroids": [d["centroid"] for _, d in dets],
            "bboxes": [d["bbox"] for _, d in dets],
            "areas": [d["area"] for _, d in dets],
            "bright_max": [d["bright_max"] for _, d in dets],
        })
        tracks.append(t)
    for tid, (dets, feat, reason) in rejected.items():
        t = dict(feat)
        t.update({"track_id": tid, "label": "rejected", "reject_reason": reason})
        tracks.append(t)
    tracks.sort(key=lambda t: (t["label"] != "firebrand", -t.get("lifetime_frames", 0)))
    path = Path(out_dir) / "tracks_vision.json"
    with open(path, "w") as f:
        json.dump({
            "params": params, "events": events,
            "counts": {
                "firebrand_tracks": sum(1 for t in tracks if t["label"] == "firebrand"),
                "rejected_tracks": sum(1 for t in tracks if t["label"] == "rejected"),
                "firebrand_detections": sum(t["n_detections"] for t in tracks
                                            if t["label"] == "firebrand"),
            },
            "tracks": tracks,
        }, f, indent=1)
    return path


# ----------------------------------------------------------------------------
# QA artifacts
# ----------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _annotate_strip_frame(img, title):
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    cv2.putText(img, title, (6, 18), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, title, (6, 18), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def qa_roi_overlay(out_dir, frames, ignition, params):
    """Render the auto-derived fuel/burner exclusion zone for visual approval:
    background of a representative post-ignition chunk with the ROI (green) and
    flame mask (red) contours."""
    post = [f for f in frames if ignition is None or f[0] >= ignition]
    if not post:
        return
    mid = len(post) // 2
    chunk = post[max(0, mid - 300):mid + 300]
    bg, _ = build_background([p for _, p in chunk], max_samples=30)
    if bg is None:
        return
    flame_mask, _ = flame_mask_and_dist(bg, params["flame_bg_thresh"],
                                        params["flame_buffer"])
    fuel_roi, _, _ = fuel_roi_and_dist(bg, flame_mask, params["warm_bg_thresh"],
                                       params["roi_buffer"])
    vis = cv2.applyColorMap(bg, cv2.COLORMAP_INFERNO)
    for mask, color in [(fuel_roi, (0, 255, 0)), (flame_mask, (0, 0, 255))]:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)
    cv2.putText(vis, "fuel/burner exclusion zone (green) | flame mask (red)",
                (10, 24), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, "firebrand tracks must exit the green zone",
                (10, 46), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(Path(out_dir) / "qa_roi.png"), vis)


def qa_onset_strip(out_dir, frame_paths, events, first_fb_det):
    rows = []
    for key, label in [("ignition_frame_0based", "ignition"),
                       ("first_firebrand_frame_0based", "first firebrand")]:
        center = events.get(key)
        if center is None:
            continue
        tiles = []
        for idx in range(center - 3, center + 5):
            path = frame_paths.get(idx)
            if path is None:
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            if key == "first_firebrand_frame_0based" and first_fb_det and idx >= center:
                x1, y1, x2, y2 = first_fb_det["bbox"]
                cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 255, 0), 2)
            mark = " <<" if idx == center else ""
            tiles.append(_annotate_strip_frame(
                img, f"{label} f{idx} t={idx / FPS:.1f}s{mark}"))
        if tiles:
            rows.append(np.hstack(tiles))
    if rows:
        width = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1],
                                   cv2.BORDER_CONSTANT, value=(30, 30, 30)) for r in rows]
        cv2.imwrite(str(Path(out_dir) / "qa_onset_strip.png"), np.vstack(rows))


def qa_crop_grid(out_dir, frame_paths, accepted, n_samples=40, crop=64, cols=5):
    pool = []
    for tid, (dets, feat) in accepted.items():
        for idx, d in dets:
            pool.append((tid, idx, d))
    if not pool:
        return
    random.seed(42)
    random.shuffle(pool)
    seen_tracks, samples = set(), []
    for tid, idx, d in pool:  # prefer one crop per track for diversity
        if tid in seen_tracks:
            continue
        seen_tracks.add(tid)
        samples.append((tid, idx, d))
        if len(samples) >= n_samples:
            break
    for tid, idx, d in pool:  # fill up if few tracks
        if len(samples) >= n_samples:
            break
        if (tid, idx, d) not in samples:
            samples.append((tid, idx, d))

    tiles = []
    for tid, idx, d in samples[:n_samples]:
        path = frame_paths.get(idx)
        img = cv2.imread(path) if path else None
        if img is None:
            continue
        cx, cy = int(d["centroid"][0]), int(d["centroid"][1])
        half = crop // 2
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(img.shape[1], cx + half), min(img.shape[0], cy + half)
        tile = img[y1:y2, x1:x2]
        tile = cv2.copyMakeBorder(tile, 0, max(0, crop - tile.shape[0]),
                                  0, max(0, crop - tile.shape[1]),
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
        bx1, by1, bx2, by2 = d["bbox"]
        cv2.rectangle(tile, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), (0, 255, 0), 1)
        tile = cv2.resize(tile, (crop * 2, crop * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, f"T{tid} f{idx}", (3, 12), FONT, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(tile, f"T{tid} f{idx}", (3, 12), FONT, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(Path(out_dir) / "qa_grid.png"), np.vstack(rows))


def qa_clips(out_dir, frame_paths, per_frame, accepted, events, n_clips=3, clip_len=240):
    """Short annotated clips: around firebrand onset, peak flux, and a midpoint."""
    if not per_frame:
        return
    counts = {idx: len(v) for idx, v in per_frame.items()}
    peak = max(counts, key=counts.get)
    onset = events.get("first_firebrand_frame_0based") or min(per_frame)
    all_idx = sorted(frame_paths)
    mid = all_idx[len(all_idx) // 2]
    centers = []
    for c in [onset, peak, mid][:n_clips]:
        if all(abs(c - e) > clip_len for e in centers):
            centers.append(c)

    trails = {}  # track_id -> {frame: centroid} for trail rendering
    for tid, (dets, _) in accepted.items():
        trails[tid] = {idx: d["centroid"] for idx, d in dets}

    for ci, center in enumerate(centers):
        start = max(all_idx[0], center - clip_len // 4)
        idxs = [i for i in all_idx if start <= i < start + clip_len]
        if not idxs:
            continue
        first = cv2.imread(frame_paths[idxs[0]])
        if first is None:
            continue
        h, w = first.shape[:2]
        path = str(Path(out_dir) / f"qa_clip_{ci}_f{start}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for idx in idxs:
            img = cv2.imread(frame_paths[idx])
            if img is None:
                continue
            for d in per_frame.get(idx, []):
                tid = d["track_id"]
                pts = [trails[tid][f] for f in sorted(trails[tid]) if idx - 15 <= f <= idx]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                             (0, 0, 255), 1, cv2.LINE_AA)
                cx, cy = int(d["centroid"][0]), int(d["centroid"][1])
                cv2.circle(img, (cx, cy), 4, (0, 255, 0), 1)
            txt = f"f{idx} t={idx / FPS:.1f}s labels={len(per_frame.get(idx, []))}"
            cv2.putText(img, txt, (10, 24), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (10, 24), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(img)
        writer.release()


def qa_flux_plot(out_dir, per_frame, events, total_range):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib unavailable — skipping qa_flux.png")
        return
    start, end = total_range
    seconds = np.arange(start, end) / FPS
    counts = np.zeros(end - start)
    for idx, dets in per_frame.items():
        if start <= idx < end:
            counts[idx - start] = len(dets)
    # 1-second bins
    bins = np.arange(seconds[0], seconds[-1] + 1, 1.0)
    binned = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        m = (seconds >= bins[i]) & (seconds < bins[i + 1])
        binned[i] = counts[m].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(bins[:-1], binned, lw=0.8)
    for key, color, label in [("ignition_time_s", "red", "ignition"),
                              ("first_firebrand_time_s", "green", "first firebrand")]:
        if events.get(key) is not None:
            ax.axvline(events[key], color=color, ls="--", label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("firebrand labels / s")
    ax.set_title("Vision-only firebrand flux")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "qa_flux.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def default_params(args=None):
    p = {
        "diff_threshold": 7,        # FLOOR of the adaptive threshold (gray)
        "noise_k": 3.5,             # threshold = max(floor, k * pixel sigma)
        "peak_diff": 14,            # min peak contrast within a blob
        "min_area": 3,
        "max_area": 150,
        "bright_px_thresh": 80,     # ~202C equivalent; pre-ignition scene peaks
                                    # at gray ~31, so ample margin
        "ignition_pixels": 3,       # sustained bright pixels => ignition
                                    # (matches temp pipeline's 200C definition)
        "flame_bg_thresh": 120,     # persistent-bright background => flame mask
        "flame_buffer": 20,         # dilation of flame mask (px)
        "flame_bright": 110,        # per-frame bright transients (early flame is
                                    # ~100-145 gray; warm table stays below this)
        "big_bright_area": 400,     # transient blobs larger than this excluded
        "transient_buffer": 10,
        "bg_samples": 30,
        "max_distance": 80,         # tracker association gate (px; fast embers)
        "max_age": 10,
        "track_area_threshold": 5,  # ByteTrack high/low confidence split
        "warm_bg_thresh": 80,       # persistent warm bg => fuel/burner zone
        "roi_buffer": 25,           # dilation of fuel/burner zone (px)
        "stitch_gap": 6,            # tracklet re-linking: max frame gap
        "stitch_dist": 30,          # ...max error vs extrapolated position (px)
        "min_lifetime": 4,          # track gates
        "min_displacement": 15,
        "min_speed": 1.0,
        "min_consistency": 0.5,
        "min_exit_dist": 20,        # track must get this far outside the zone
        "min_track_bright": 50,     # peak brightness; cooler = glint/shimmer
        "min_sharpness": 1.5,       # peak/mean contrast; lower = diffuse glow
        "max_thin_dim": 7,          # blobs thinner than this are never "diffuse"
        "max_track_bg": 45,         # median background under track; brighter =
                                    # glow on structure, not flight through air
        "floor_y": 620,             # horizon: tracks living below this are
                                    # floor reflections / landed embers
        "flame_on_area": 300,       # flame-scale bright area => gas burner on
        "burner_off_min_frames": 90,  # sustained no-flame (~3 s) => burner off
    }
    if args is not None:
        for k in p:
            v = getattr(args, k, None)
            if v is not None:
                p[k] = v
    return p


def run_pipeline(frames_dir, out_dir, events_out, params, workers=4, chunk_size=600):
    t0 = time.time()
    frames = list_frames(frames_dir, params["start"], params["end"])
    if not frames:
        print(f"[ERROR] No frames found in {frames_dir}")
        return None
    print(f"[INFO] {len(frames)} frames "
          f"({frames[0][0]}..{frames[-1][0]}), workers={workers}", flush=True)
    frame_paths = dict(frames)

    sample = load_gray(frames[0][1])
    img_shape = sample.shape

    # Stage A: brightness stats pass -> ignition frame
    stat_chunks = [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]
    stat_jobs = [{"frames": c, "params": params} for c in stat_chunks]
    stats = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for s in ex.map(stats_chunk, stat_jobs):
                stats.extend(s)
    else:
        for job in stat_jobs:
            stats.extend(stats_chunk(job))
    stats.sort(key=lambda s: s["frame_0based"])

    ignition = detect_ignition(stats, params["ignition_pixels"])
    events = {
        "ignition_frame_0based": ignition,
        "ignition_time_s": round(ignition / FPS, 2) if ignition is not None else None,
    }
    print(f"[INFO] Stage A: ignition frame {ignition} "
          f"(t={events['ignition_time_s']}s) in {time.time() - t0:.0f}s"
          if ignition is not None else "[WARN] No ignition detected", flush=True)

    # Stage B/C: ignition-aware chunks -> candidates + per-frame scene objects
    chunks = build_chunks(frames, chunk_size, ignition)
    jobs = [{"frames": c, "params": params} for c in chunks]
    candidates, scenes = {}, {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (c, s) in enumerate(ex.map(process_chunk, jobs)):
                candidates.update(c)
                scenes.update(s)
                print(f"[INFO] chunk {i + 1}/{len(chunks)} done "
                      f"({sum(len(v) for v in c.values())} candidates)", flush=True)
    else:
        for i, job in enumerate(jobs):
            c, s = process_chunk(job)
            candidates.update(c)
            scenes.update(s)
            print(f"[INFO] chunk {i + 1}/{len(chunks)} done", flush=True)

    n_cand = sum(len(v) for v in candidates.values())
    print(f"[INFO] Stage B/C: {n_cand} raw candidates in "
          f"{time.time() - t0:.0f}s", flush=True)

    # Stage D: tracking + gates
    frame_indices = [idx for idx, _ in frames]
    raw_tracks = run_tracking(frame_indices, candidates, params)
    n_tracklets = len(raw_tracks)
    raw_tracks = stitch_tracklets(raw_tracks, params)
    print(f"[INFO] Stage D: {n_tracklets} tracklets -> "
          f"{len(raw_tracks)} after stitching", flush=True)

    accepted, rejected = {}, {}
    reject_counts = {}
    for tid, dets in raw_tracks.items():
        dets.sort(key=lambda d: d[0])
        feat = compute_track_features(dets)
        ok, reason = gate_track(feat, params)
        if ok and ignition is not None and feat["start_frame_0based"] < ignition:
            # moving hot objects before combustion = the igniter torch,
            # not firebrands
            ok, reason = False, "pre_ignition"
        if ok:
            accepted[tid] = (dets, feat)
        else:
            rejected[tid] = (dets, feat, reason)
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

    n_labels = sum(len(d) for d, _ in accepted.values())
    print(f"[INFO] Accepted {len(accepted)} firebrand tracks "
          f"({n_labels} labeled detections)", flush=True)
    print(f"[INFO] Rejected {len(rejected)} tracks: {reject_counts}", flush=True)

    # Events: firebrand onset + validity check
    if accepted:
        first_tid = min(accepted, key=lambda t: accepted[t][1]["start_frame_0based"])
        onset = accepted[first_tid][1]["start_frame_0based"]
        first_fb_det = accepted[first_tid][0][0][1]
        events["first_firebrand_frame_0based"] = onset
        events["first_firebrand_time_s"] = round(onset / FPS, 2)
        events["first_firebrand_track_id"] = first_tid
    else:
        first_fb_det = None
        events["first_firebrand_frame_0based"] = None
        events["first_firebrand_time_s"] = None

    pre_ignition = 0
    if ignition is not None:
        for tid, (dets, _) in accepted.items():
            pre_ignition += sum(1 for f, _ in dets if f < ignition)
    events["labels_before_ignition"] = pre_ignition
    if pre_ignition:
        print(f"[WARN] {pre_ignition} labels BEFORE ignition — "
              f"likely false positives, tune thresholds", flush=True)

    # Per-minute counts
    per_min = {}
    for tid, (dets, _) in accepted.items():
        for f, _ in dets:
            m = int(f / FPS / 60)
            per_min[m] = per_min.get(m, 0) + 1
    events["labels_per_minute"] = [per_min.get(m, 0)
                                   for m in range(max(per_min, default=0) + 1)]

    # Scene states (burning / burner-off / smoldering) + static burner box
    states, burner_off = detect_states(stats, scenes, ignition, params)
    events["states"] = states
    events["burner_off_frame_0based"] = burner_off
    events["burner_off_time_s"] = (round(burner_off / FPS, 2)
                                   if burner_off is not None else None)
    print(f"[INFO] states: " + ", ".join(
        f"{s['state']}[{s['start_frame']}-{s['end_frame']}]" for s in states),
        flush=True)
    burner_bbox = derive_burner_bbox(frames, ignition, params)
    print(f"[INFO] gas burner bbox: {burner_bbox}", flush=True)

    # Outputs
    accepted_dets = {tid: dets for tid, (dets, _) in accepted.items()}
    per_frame, jsonl_path = write_outputs(out_dir, frames, accepted_dets,
                                          stats, events, params, img_shape,
                                          scenes=scenes, burner_bbox=burner_bbox)
    write_tracks_json(out_dir, accepted, rejected, events, params)
    write_rejected_jsonl(out_dir, rejected)
    events_path = Path(events_out)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "w") as f:
        json.dump({**events, "params": params,
                   "frame_range": [frames[0][0], frames[-1][0] + 1]}, f, indent=2)

    # QA
    qa_roi_overlay(out_dir, frames, ignition, params)
    qa_onset_strip(out_dir, frame_paths, events, first_fb_det)
    qa_crop_grid(out_dir, frame_paths, accepted)
    qa_clips(out_dir, frame_paths, per_frame, accepted, events)
    qa_flux_plot(out_dir, per_frame, events, (frames[0][0], frames[-1][0] + 1))

    print(f"[INFO] Outputs in {out_dir}; events in {events_out}", flush=True)
    print(f"[INFO] Total time: {time.time() - t0:.0f}s", flush=True)
    return {
        "events": events,
        "n_tracks": len(accepted),
        "n_labels": n_labels,
        "n_rejected": len(rejected),
        "accepted": accepted,
    }


# ----------------------------------------------------------------------------
# Self-test: synthetic scene with known ground truth
# ----------------------------------------------------------------------------

def _draw_gauss(img, cx, cy, sigma, peak):
    """Additive Gaussian point source (realistic ember PSF)."""
    h, w = img.shape[:2]
    r = int(max(3, sigma * 4))
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], g)


def make_synthetic_scene(out_dir, n_frames=360, h=256, w=320, ignition=60,
                         flameoff=335):
    """Static background + flame from `ignition` to `flameoff` (then two
    static smoldering glow spots) + 7 GT embers + distractors."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (18 + 14 * yy / h).astype(np.float32)  # dark static gradient
    noise = rng.normal(0, 1.5, size=(h, w)).astype(np.float32)
    flame_cx, flame_cy = 160, 85

    # spawn just outside the column edge (inside the dilated fuel ROI) and fly
    # away — exercises the born-on-fuel-then-exits case
    embers = []  # (t0, x0, y0, vx, vy, life, radius, peak)
    specs = [(90, 190, 75, 1.8, -1.0, 60, 2.0, 170),
             (120, 195, 80, 2.4, -0.6, 50, 1.8, 150),
             (150, 188, 70, 1.4, -1.4, 70, 2.2, 180),
             (180, 196, 85, 2.0, -0.2, 55, 1.6, 140),
             (210, 192, 78, 1.2, -1.1, 65, 2.4, 160),
             (240, 194, 72, 2.6, -0.9, 45, 1.9, 155),
             (270, 192, 76, 1.6, -0.8, 50, 2.0, 70)]  # dim ember (adaptive thr)
    for s in specs:
        embers.append(s)

    jitter_center = (60.0, 200.0)  # (x, y) far from flame
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt_tracks = []
    for t0, x0, y0, vx, vy, life, rad, peak in embers:
        pts = []
        for k in range(life):
            pts.append((t0 + k, x0 + vx * k, y0 + vy * k))
        gt_tracks.append(pts)

    jit_x, jit_y = jitter_center
    for i in range(n_frames):
        img = base + noise * 0  # keep background deterministic
        img = img + rng.normal(0, 1.0, size=(h, w)).astype(np.float32)
        # fuel column under the flame: cold pre-ignition, warm after
        img[100:200, 145:175] = np.maximum(
            img[100:200, 145:175], 90.0 if i >= ignition else 35.0)
        # warm structure band (e.g. table edge), static in the background
        img[210:217, 40:140] = np.maximum(
            img[210:217, 40:140], 70.0 if i >= ignition else 30.0)
        if i >= flameoff:
            # burner off, fuel smoldering: two static glow spots on the column
            cv2.circle(img, (155, 150), 2, 100.0, -1)
            cv2.circle(img, (168, 175), 2, 95.0, -1)
        if ignition <= i < flameoff:
            # flame: bright flickering blob (~40x30) at burner
            fh, fw = 30, 40
            flicker = 200 + 40 * np.sin(i * 0.7) + rng.normal(0, 5)
            y0f, x0f = flame_cy - fh // 2, flame_cx - fw // 2
            img[y0f:y0f + fh, x0f:x0f + fw] = np.maximum(
                img[y0f:y0f + fh, x0f:x0f + fw], flicker)
            # plume: large soft blob rising above flame (must NOT be labeled)
            py = int(flame_cy - 30 - (i % 40))
            if py > 10:
                cv2.circle(img, (flame_cx, py), 22,
                           float(55 + 10 * np.sin(i)), -1)
            # jitter distractor: small bright dot random-walking (no net motion)
            jx = jit_x + rng.normal(0, 4)
            jy = jit_y + rng.normal(0, 4)
            cv2.circle(img, (int(jx), int(jy)), 2, 160.0, -1)
            # flame-crawl distractor: bright spot climbing the fuel column
            # consistently (passes motion gates; must be killed by the
            # fuel-ROI exit rule)
            if i >= 130:
                crawl_y = 190 - 1.6 * ((i - 130) % 50)
                cv2.circle(img, (160, int(crawl_y)), 2, 165.0, -1)
            # floor-glint distractor: dim spot sliding along the floor,
            # consistent motion but never bright (killed by brightness gate)
            if i >= 140:
                glint_x = 250 - 1.4 * ((i - 140) % 60)
                cv2.circle(img, (int(glint_x), 230), 2, 38.0, -1)
            # structure-reflection distractor: BRIGHT thin glint sliding along
            # the warm band — only the background-darkness gate can kill it
            if i >= 150:
                refl_x = 50 + 1.3 * ((i - 150) % 60)
                _draw_gauss(img, refl_x, 213, 1.5, 95.0)
            # GT embers: Gaussian point sources, fading as they cool
            for t0, x0, y0, vx, vy, life, rad, peak in embers:
                k = i - t0
                if 0 <= k < life:
                    fade = 1.0 - 0.7 * k / life
                    _draw_gauss(img, x0 + vx * k, y0 + vy * k,
                                rad, peak * fade)
        # stationary speck, below ignition brightness (present from frame 0;
        # bg should absorb it)
        cv2.circle(img, (40, 40), 2, 60.0, -1)

        frame = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f"frame_{i + 1:05d}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    return gt_tracks, ignition, jitter_center


def self_test():
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = os.path.join(tmp, "frames")
        out_dir = os.path.join(tmp, "labels")
        gt_tracks, ignition_gt, jitter_center = make_synthetic_scene(frames_dir)
        flameoff_gt = 335
        params = default_params()
        # synthetic scene is 256px tall; its floor zone starts ~y=222.
        # burner-off confirmation shortened to fit the 360-frame scene.
        params.update({"start": 0, "end": -1, "floor_y": 222,
                       "burner_off_min_frames": 12})
        result = run_pipeline(frames_dir, out_dir,
                              os.path.join(tmp, "vision_events.json"),
                              params, workers=2, chunk_size=120)

        def check(name, cond):
            nonlocal ok
            print(f"[{'PASS' if cond else 'FAIL'}] {name}")
            ok = ok and cond

        check("pipeline ran", result is not None)
        if result is None:
            sys.exit(1)
        ev = result["events"]
        ign = ev["ignition_frame_0based"]
        check(f"ignition detected near {ignition_gt} (got {ign})",
              ign is not None and abs(ign - ignition_gt) <= 5)
        onset = ev["first_firebrand_frame_0based"]
        check(f"firebrand onset near 90 (got {onset})",
              onset is not None and 85 <= onset <= 115)
        check(f"no labels before ignition (got {ev['labels_before_ignition']})",
              ev["labels_before_ignition"] == 0)

        # Burner-off + smoldering state detection
        boff = ev.get("burner_off_frame_0based")
        check(f"burner-off detected near {flameoff_gt} (got {boff})",
              boff is not None and abs(boff - flameoff_gt) <= 6)
        state_names = [s["state"] for s in ev.get("states", [])]
        check(f"states = pre_ignition/burning/smoldering (got {state_names})",
              state_names == ["pre_ignition", "burning", "smoldering"])
        post_labels = sum(1 for tid, (dets, _) in result["accepted"].items()
                          for f, _ in dets if f >= flameoff_gt)
        check(f"static smoldering glow not labeled (labels after burner-off: "
              f"{post_labels})", post_labels == 0)

        # Match accepted tracks to GT embers
        recovered = []
        for gi, pts in enumerate(gt_tracks):
            gt_by_frame = {f: (x, y) for f, x, y in pts}
            best = 0
            for tid, (dets, feat) in result["accepted"].items():
                hits = 0
                for f, d in dets:
                    if f in gt_by_frame:
                        gx, gy = gt_by_frame[f]
                        dd = ((d["centroid"][0] - gx) ** 2 +
                              (d["centroid"][1] - gy) ** 2) ** 0.5
                        if dd < 6:
                            hits += 1
                best = max(best, hits)
            if best >= 5:
                recovered.append(gi)
        check(f"recovered >=5/7 GT embers (got {len(recovered)})",
              len(recovered) >= 5)
        check("dim ember (peak 60 gray) recovered", len(gt_tracks) - 1 in recovered)

        # No accepted track parked at the jitter distractor
        jx, jy = jitter_center
        jitter_labeled = 0
        for tid, (dets, feat) in result["accepted"].items():
            sx, sy = feat["start_centroid"]
            ex_, ey = feat["end_centroid"]
            if (abs(sx - jx) < 12 and abs(sy - jy) < 12 and
                    abs(ex_ - jx) < 12 and abs(ey - jy) < 12):
                jitter_labeled += 1
        check(f"jitter distractor rejected (accepted near it: {jitter_labeled})",
              jitter_labeled == 0)

        # Stationary speck never labeled
        speck_labeled = 0
        for tid, (dets, _) in result["accepted"].items():
            for f, d in dets:
                if abs(d["centroid"][0] - 40) < 6 and abs(d["centroid"][1] - 40) < 6:
                    speck_labeled += 1
        check("stationary speck never labeled", speck_labeled == 0)

        # Flame-crawl distractor (climbs the fuel column, never leaves it)
        crawl_accepted = 0
        for tid, (dets, feat) in result["accepted"].items():
            xs = [d["centroid"][0] for _, d in dets]
            if all(140 <= x <= 180 for x in xs):
                crawl_accepted += 1
        check(f"flame-crawl on fuel column rejected (accepted: {crawl_accepted})",
              crawl_accepted == 0)

        # Floor-glint distractor (dim slider on the floor at y=230)
        glint_accepted = 0
        for tid, (dets, feat) in result["accepted"].items():
            ys_ = [d["centroid"][1] for _, d in dets]
            if all(y > 215 for y in ys_):
                glint_accepted += 1
        check(f"dim floor glint rejected (accepted: {glint_accepted})",
              glint_accepted == 0)

        # Structure-reflection distractor (bright glint on warm band y~213)
        refl_accepted = 0
        for tid, (dets, feat) in result["accepted"].items():
            ys_ = [d["centroid"][1] for _, d in dets]
            if all(205 <= y <= 220 for y in ys_):
                refl_accepted += 1
        check(f"bright glint on warm structure rejected (accepted: {refl_accepted})",
              refl_accepted == 0)

        # Output files exist and parse
        for fname in ["detections_vision.jsonl", "tracks_vision.json",
                      "labels_coco.json"]:
            path = os.path.join(out_dir, fname)
            exists = os.path.exists(path)
            if exists and fname.endswith(".json"):
                json.load(open(path))
            check(f"output exists: {fname}", exists)

    print(f"\n=== SELF-TEST {'PASSED' if ok else 'FAILED'} ===")
    sys.exit(0 if ok else 1)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vision-only automatic firebrand labeling (no temperature data)")
    parser.add_argument("--frames-dir", default="images/frames")
    parser.add_argument("--out-dir", default="outputs/vision_labels")
    parser.add_argument("--events-out", default="outputs/vision_events.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1, help="-1 = all frames")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--self-test", action="store_true")
    # tunables (None = use default_params)
    for key in ["diff_threshold", "peak_diff", "min_area", "max_area",
                "bright_px_thresh", "ignition_pixels", "flame_bg_thresh",
                "flame_buffer", "flame_bright", "big_bright_area",
                "transient_buffer", "bg_samples", "max_distance", "max_age",
                "track_area_threshold", "min_lifetime", "warm_bg_thresh",
                "roi_buffer", "stitch_gap", "stitch_dist", "min_track_bright",
                "max_thin_dim", "max_track_bg", "floor_y", "flame_on_area",
                "burner_off_min_frames"]:
        parser.add_argument(f"--{key.replace('_', '-')}", type=int, default=None)
    for key in ["min_displacement", "min_speed", "min_consistency",
                "min_exit_dist", "noise_k", "min_sharpness"]:
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=None)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    params = default_params(args)
    params["start"] = args.start
    params["end"] = args.end
    run_pipeline(args.frames_dir, args.out_dir, args.events_out, params,
                 workers=args.workers, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
