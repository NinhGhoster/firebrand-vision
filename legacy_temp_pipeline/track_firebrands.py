#!/usr/bin/env python3
"""
track_firebrands.py — Phase 3: Associate firebrand detections into tracks.

Simple proximity-based tracking:
  1. Read Phase 2 (or Phase 1) detections
  2. For each frame, match centroids to closest centroid in previous frame
  3. If distance < match_distance_px, same track; otherwise new track
  4. Output: track summaries and per-frame assignments

Usage:
  python track_firebrands.py --detections outputs/verified.jsonl --out outputs/tracks.json
  python track_firebrands.py --detections outputs/detections_refined_0-25467.jsonl --out outputs/tracks.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def match_tracks(prev_centroids, curr_centroids, max_distance=50):
    """Greedy matching of current centroids to previous ones by distance."""
    if not prev_centroids:
        return [-1] * len(curr_centroids)

    assignments = [-1] * len(curr_centroids)
    available_prev = list(range(len(prev_centroids)))

    for ci, (cx, cy) in enumerate(curr_centroids):
        best_dist = max_distance + 1
        best_pi = -1
        for pi in available_prev:
            px, py = prev_centroids[pi]
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_pi = pi
        if best_pi >= 0:
            assignments[ci] = best_pi
            available_prev.remove(best_pi)

    return assignments


def track_detections(detections_path: str, output_path: str, max_distance: float = 50.0,
                     min_track_length: int = 1, verbose: bool = True):
    print(f"[INFO] Loading detections from {detections_path}", flush=True)
    t0 = time.time()

    # Load all detections sorted by frame
    with open(detections_path) as f:
        lines = f.readlines()

    entries = []
    for line in lines:
        d = json.loads(line)
        entries.append(d)

    entries.sort(key=lambda e: e.get("frame_0based", e.get("frame", 0)))

    tracks = {}          # track_id -> list of detection dicts
    next_track_id = 0
    prev_centroids = []  # (cx, cy) for each active track in prev frame
    prev_track_ids = []  # track_id corresponding to each centroid
    total_frames = len(entries)
    n_total_detections = sum(e.get("n_candidates", len(e.get("detections", [e] if "centroid" in e else []))) for e in entries)

    if verbose:
        print(f"[INFO] {total_frames} frames, ~{n_total_detections} detections", flush=True)

    # Track per-frame counts
    frame_detections = 0
    t_start = time.time()

    for frame_idx, entry in enumerate(entries):
        candidates = entry.get("detections", [entry] if "centroid" in entry else [])
        verified = entry.get("verified", None)

        if not candidates:
            prev_centroids = []
            prev_track_ids = []
            continue

        curr_centroids = [c["centroid"] if "centroid" in c else c.get("centroid", (0, 0)) for c in candidates]
        curr_verified = [c.get("verified", None) if isinstance(c, dict) else None for c in candidates]
        curr_boxes = [c.get("bbox", None) if isinstance(c, dict) else None for c in candidates]

        assignments = match_tracks(prev_centroids, curr_centroids, max_distance)

        frame_track_ids = []
        for ci, assign in enumerate(assignments):
            if assign >= 0 and assign < len(prev_track_ids):
                tid = prev_track_ids[assign]
            else:
                tid = next_track_id
                next_track_id += 1

            detection = {
                "frame": entry.get("frame", entry.get("frame_0based", frame_idx) + 1),
                "frame_0based": entry.get("frame_0based", frame_idx),
                "track_id": tid,
                "centroid": curr_centroids[ci],
                "bbox": curr_boxes[ci],
            }
            if isinstance(candidates[ci], dict):
                if "temp_max" in candidates[ci]:
                    detection["temp_max"] = candidates[ci]["temp_max"]
                if "temp_mean" in candidates[ci]:
                    detection["temp_mean"] = candidates[ci]["temp_mean"]
            if verified is not None:
                detection["verified"] = verified
            if curr_verified[ci] is not None:
                detection["verified"] = curr_verified[ci]

            if tid not in tracks:
                tracks[tid] = []
            tracks[tid].append(detection)
            frame_track_ids.append(tid)

        prev_centroids = curr_centroids
        prev_track_ids = frame_track_ids
        frame_detections += len(candidates)

        if verbose and (frame_idx + 1) % 2000 == 0:
            elapsed = time.time() - t_start
            rate = (frame_idx + 1) / elapsed
            print(f"[INFO] {frame_idx+1}/{total_frames} frames | "
                  f"{len(tracks)} tracks | {rate:.1f} fps", flush=True)

    # Build track summaries
    track_summaries = []
    for tid, detections in tracks.items():
        detections.sort(key=lambda d: d["frame_0based"])
        start_frame = detections[0]["frame_0based"]
        end_frame = detections[-1]["frame_0based"]
        lifetime = end_frame - start_frame + 1
        centroids = [d["centroid"] for d in detections]
        positions = np.array(centroids)
        if len(positions) > 1:
            displacements = np.diff(positions, axis=0)
            total_distance = float(np.sum(np.linalg.norm(displacements, axis=1)))
            avg_speed = total_distance / lifetime if lifetime > 0 else 0
            direction = displacements.mean(axis=0).tolist() if len(displacements) > 0 else [0, 0]
        else:
            total_distance = 0
            avg_speed = 0
            direction = [0, 0]

        verified_count = sum(1 for d in detections if d.get("verified", True))

        temps = [d.get("temp_max", 0) for d in detections]

        track_summaries.append({
            "track_id": tid,
            "start_frame": start_frame + 1,
            "end_frame": end_frame + 1,
            "lifetime_frames": lifetime,
            "lifetime_s": lifetime / 29.41,
            "n_detections": len(detections),
            "total_distance_px": round(total_distance, 1),
            "avg_speed_px_per_frame": round(avg_speed, 2),
            "direction": [round(direction[0], 1), round(direction[1], 1)],
            "verified_count": verified_count,
            "start_centroid": [round(centroids[0][0], 1), round(centroids[0][1], 1)],
            "end_centroid": [round(centroids[-1][0], 1), round(centroids[-1][1], 1)],
            "frames_0based": [d["frame_0based"] for d in detections],
            "centroids": [[round(c[0], 1), round(c[1], 1)] for c in centroids],
            "temps": [round(t, 1) for t in temps],
        })

    track_summaries.sort(key=lambda t: t["lifetime_frames"], reverse=True)

    result = {
        "total_tracks": len(track_summaries),
        "total_detections": sum(t["n_detections"] for t in track_summaries),
        "max_distance_px": max_distance,
        "total_frames": total_frames,
        "tracks": track_summaries,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    elapsed = time.time() - t0
    print(f"[INFO] Done: {len(track_summaries)} tracks in {elapsed:.1f}s", flush=True)
    print(f"[INFO] Output: {out_path}", flush=True)

    # Summary stats
    lifetimes = [t["lifetime_frames"] for t in track_summaries]
    speeds = [t["avg_speed_px_per_frame"] for t in track_summaries]
    distances = [t["total_distance_px"] for t in track_summaries]

    print(f"\n=== Tracking Summary ===")
    print(f"Total tracks: {len(track_summaries)}")
    print(f"Lifetime: min={min(lifetimes)} max={max(lifetimes)} mean={np.mean(lifetimes):.1f} frames")
    print(f"  ({min(lifetimes)/29.41:.2f}s - {max(lifetimes)/29.41:.2f}s)")
    print(f"Speed: min={min(speeds):.2f} max={max(speeds):.2f} mean={np.mean(speeds):.2f} px/frame")
    print(f"Distance: min={min(distances):.1f} max={max(distances):.1f} mean={np.mean(distances):.1f} px")

    # Track length distribution
    short_tracks = sum(1 for t in lifetimes if t < 10)
    med_tracks = sum(1 for t in lifetimes if 10 <= t < 100)
    long_tracks = sum(1 for t in lifetimes if t >= 100)
    print(f"\nTrack length: short(<10)={short_tracks} med(10-100)={med_tracks} long(>=100)={long_tracks}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Firebrand tracking across frames"
    )
    parser.add_argument("--detections", required=True, help="Phase 1 or 2 JSONL")
    parser.add_argument("--out", default="outputs/tracks.json", help="Output JSON")
    parser.add_argument("--max_distance", type=float, default=50.0,
                        help="Max centroid distance for matching (px)")
    parser.add_argument("--min_track_length", type=int, default=1,
                        help="Min frames for a valid track")
    args = parser.parse_args()

    track_detections(
        detections_path=args.detections,
        output_path=args.out,
        max_distance=args.max_distance,
        min_track_length=args.min_track_length,
    )


if __name__ == "__main__":
    main()
