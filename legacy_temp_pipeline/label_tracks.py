"""
label_tracks.py — Physics-based labeling of ByteTrack tracks.

Labels:
  - firebrand: consistent motion + cooling + away from burner
  - not_firebrand: stationary, jittery, near burner, or noise
  - ambiguous: doesn't meet either threshold (excluded from classifier training)

Output:
  - tracks_labeled.json: tracks with label + label_confidence
  - metadata_labeled.jsonl: per-detection crops with corrected labels
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np


def label_track(t, thresholds):
    """Apply physics-based rules to classify a track."""
    speed = t["avg_speed_px_per_frame"]
    cons = t["angle_consistency"]
    slope = t["temp_slope_C_per_frame"]
    burner_dist = t["burner_distance_start"]
    lifetime = t["lifetime_frames"]

    # Firebrand conditions (ALL must be true)
    is_firebrand = (
        speed > thresholds["firebrand_min_speed"]
        and cons > thresholds["firebrand_min_consistency"]
        and (slope < thresholds["firebrand_max_temp_slope"] or
             t["total_distance_px"] > thresholds["firebrand_min_distance"])
        and burner_dist > thresholds["firebrand_min_burner_dist"]
        and lifetime >= thresholds["min_lifetime"]
    )

    # Not-firebrand conditions (ANY can trigger)
    is_not_firebrand = (
        speed < thresholds["not_firebrand_max_speed"]
        or cons < thresholds["not_firebrand_max_consistency"]
        or burner_dist < thresholds["not_firebrand_max_burner_dist"]
        or lifetime < thresholds["min_lifetime"]
    )

    if is_firebrand and not is_not_firebrand:
        return "firebrand", 1.0
    elif is_not_firebrand and not is_firebrand:
        return "not_firebrand", 1.0
    elif is_firebrand and is_not_firebrand:
        # Higher confidence wins
        fb_score = sum([
            min(speed / thresholds["firebrand_min_speed"], 3) / 3,
            cons,
            1.0 if slope < 0 else 0.5,
            min(burner_dist / 500, 1),
        ])
        nfb_score = sum([
            1.0 if speed < thresholds["not_firebrand_max_speed"] else 0,
            1.0 if cons < thresholds["not_firebrand_max_consistency"] else 0,
            1.0 if burner_dist < thresholds["not_firebrand_max_burner_dist"] else 0,
        ])
        if fb_score > nfb_score:
            return "firebrand", 0.7
        else:
            return "not_firebrand", 0.7
    else:
        return "ambiguous", 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Label ByteTrack tracks using physics-based rules"
    )
    parser.add_argument("--tracks", default="outputs/tracks_bytetrack.json")
    parser.add_argument("--out-tracks", default="outputs/tracks_labeled.json")
    parser.add_argument("--out-metadata", default="outputs/metadata_labeled.jsonl")
    parser.add_argument("--min-lifetime", type=int, default=10)
    parser.add_argument("--firebrand-min-speed", type=float, default=1.5)
    parser.add_argument("--firebrand-min-consistency", type=float, default=0.7)
    parser.add_argument("--firebrand-max-temp-slope", type=float, default=0.0)
    parser.add_argument("--firebrand-min-distance", type=float, default=30)
    parser.add_argument("--firebrand-min-burner-dist", type=float, default=100)
    parser.add_argument("--not-firebrand-max-speed", type=float, default=0.5)
    parser.add_argument("--not-firebrand-max-consistency", type=float, default=0.4)
    parser.add_argument("--not-firebrand-max-burner-dist", type=float, default=50)
    parser.add_argument("--max-crops-per-class", type=int, default=10000,
                        help="Max crops to extract per class for training")
    parser.add_argument("--frames-dir", default="images/frames")
    args = parser.parse_args()

    thresholds = {
        "min_lifetime": args.min_lifetime,
        "firebrand_min_speed": args.firebrand_min_speed,
        "firebrand_min_consistency": args.firebrand_min_consistency,
        "firebrand_max_temp_slope": args.firebrand_max_temp_slope,
        "firebrand_min_distance": args.firebrand_min_distance,
        "firebrand_min_burner_dist": args.firebrand_min_burner_dist,
        "not_firebrand_max_speed": args.not_firebrand_max_speed,
        "not_firebrand_max_consistency": args.not_firebrand_max_consistency,
        "not_firebrand_max_burner_dist": args.not_firebrand_max_burner_dist,
    }

    with open(args.tracks) as f:
        data = json.load(f)
    tracks = data["tracks"]
    print(f"[INFO] Loaded {len(tracks)} tracks", flush=True)

    # Label all tracks
    labeled = {"firebrand": [], "not_firebrand": [], "ambiguous": []}
    for t in tracks:
        label, confidence = label_track(t, thresholds)
        t["label"] = label
        t["label_confidence"] = confidence
        labeled[label].append(t)

    print(f"\n=== Physics-Based Labeling Results ===")
    print(f"Firebrand: {len(labeled['firebrand'])}")
    print(f"Not firebrand: {len(labeled['not_firebrand'])}")
    print(f"Ambiguous: {len(labeled['ambiguous'])}")

    # Summary stats
    firebrand = labeled["firebrand"]
    not_firebrand = labeled["not_firebrand"]
    print(f"\n--- Firebrand stats ---")
    if firebrand:
        fspeeds = [t["avg_speed_px_per_frame"] for t in firebrand]
        fcons = [t["angle_consistency"] for t in firebrand]
        fslopes = [t["temp_slope_C_per_frame"] for t in firebrand]
        print(f"  Speed: {np.mean(fspeeds):.2f} +/- {np.std(fspeeds):.2f}")
        print(f"  Consistency: {np.mean(fcons):.3f} +/- {np.std(fcons):.3f}")
        print(f"  Temp slope: {np.mean(fslopes):.3f} (cooling={sum(1 for s in fslopes if s<0)}, heating={sum(1 for s in fslopes if s>0)})")

    print(f"\n--- Not firebrand stats ---")
    if not_firebrand:
        nspeeds = [t["avg_speed_px_per_frame"] for t in not_firebrand]
        print(f"  Speed: {np.mean(nspeeds):.2f} +/- {np.std(nspeeds):.2f}")
        nslopes = [t["temp_slope_C_per_frame"] for t in not_firebrand]
        nburner = [t["burner_distance_start"] for t in not_firebrand]
        print(f"  Within 100px of burner: {sum(1 for d in nburner if d<100)}")

    # Save labeled tracks
    out_data = {
        "thresholds": thresholds,
        "counts": {k: len(v) for k, v in labeled.items()},
        "tracks": tracks,  # all tracks with label added
    }
    Path(args.out_tracks).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tracks, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[INFO] Labeled tracks saved to {args.out_tracks}")

    # Extract crops for training (optional, --frames-dir must exist)
    import os
    import cv2
    frames_dir = Path(args.frames_dir)
    if frames_dir.exists():
        print(f"\n[INFO] Extracting crops for classifier training...")
        crop_dir = Path("crops_labeled")
        fb_dir = crop_dir / "firebrand"
        nfb_dir = crop_dir / "not_firebrand"
        fb_dir.mkdir(parents=True, exist_ok=True)
        nfb_dir.mkdir(parents=True, exist_ok=True)

        random.seed(42)
        metadata = []
        CROP_SIZE = 64

        for label_name, track_list in [("firebrand", firebrand), ("not_firebrand", not_firebrand)]:
            out_dir = fb_dir if label_name == "firebrand" else nfb_dir
            # Collect all detections
            all_dets = []
            for t in track_list:
                frames_list = t["frames_0based"]
                centroids_list = t["centroids"]
                temps_list = t["temps"]
                for i in range(len(frames_list)):
                    all_dets.append((frames_list[i], centroids_list[i],
                                     temps_list[i] if i < len(temps_list) else 0))

            random.shuffle(all_dets)
            count = 0
            for frame_idx, centroid, temp in all_dets:
                if count >= args.max_crops_per_class:
                    break
                cx, cy = centroid
                fname = f"frame_{frame_idx + 1:05d}.png"
                frame_path = frames_dir / fname
                if not frame_path.exists():
                    continue
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                half = CROP_SIZE // 2
                x1 = max(0, int(cx) - half)
                y1 = max(0, int(cy) - half)
                x2 = min(w, int(cx) + half)
                y2 = min(h, int(cy) + half)
                crop = img[y1:y2, x1:x2]
                if crop.shape[0] < CROP_SIZE or crop.shape[1] < CROP_SIZE:
                    crop = cv2.copyMakeBorder(
                        crop,
                        0, max(0, CROP_SIZE - crop.shape[0]),
                        0, max(0, CROP_SIZE - crop.shape[1]),
                        cv2.BORDER_CONSTANT, value=(0, 0, 0),
                    )
                if crop.shape[:2] != (CROP_SIZE, CROP_SIZE):
                    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

                crop_name = f"{label_name}_{count:06d}.png"
                crop_path = out_dir / crop_name
                cv2.imwrite(str(crop_path), crop)
                metadata.append({
                    "crop": str(crop_path),
                    "label": label_name,
                    "frame_0based": frame_idx,
                    "centroid": [round(cx, 1), round(cy, 1)],
                    "temp": round(temp, 1),
                })
                count += 1

        # Save metadata
        meta_path = args.out_metadata
        with open(meta_path, "w") as f:
            for m in metadata:
                f.write(json.dumps(m) + "\n")
        print(f"[INFO] {len(metadata)} crops saved")
        print(f"[INFO] Metadata saved to {meta_path}")
    else:
        print(f"[WARN] Frames dir {frames_dir} not found — skipping crop extraction")


if __name__ == "__main__":
    main()
