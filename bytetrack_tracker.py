"""
bytetrack_tracker.py — Kalman filter + ByteTrack-style tracking for firebrand detections.

Input:  detections_refined_0-25467.jsonl (148K candidates from temp thresholding)
Output: tracks with Kalman-filtered velocity, trajectory smoothness, cooling trends

Key features per track (for physics-based labeling):
  - velocity_mean, velocity_std   → consistency of motion
  - temp_slope                    → cooling rate (negative = firebrand)
  - path_curvature                → straight vs jittery
  - burner_proximity_start        → distance from burner at first detection
  - total_distance, avg_speed     → basic motion stats
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def iou(bbox1, bbox2):
    """Intersection over Union of two boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def bbox_format(bbox):
    """Convert centroid+area to [x1, y1, x2, y2] box if needed."""
    if len(bbox) == 4:
        return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    cx, cy = bbox
    s = 10
    return [int(cx - s), int(cy - s), int(cx + s), int(cy + s)]


class KalmanTracker:
    """Constant-velocity Kalman filter for a single track.

    State: [cx, cy, vx, vy]
    Observation: [cx, cy]
    """
    def __init__(self, cx, cy, dt=1.0):
        self.state = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0  # cov
        self.F = np.array([  # transition matrix (constant velocity)
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([  # observation matrix
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self.Q = np.eye(4, dtype=np.float64) * 0.01  # process noise
        self.R = np.eye(2, dtype=np.float64) * 5.0   # measurement noise
        self.hit_streak = 1
        self.age = 0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        return self.state[:2].copy()

    def update(self, cx, cy):
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self.H @ self.state  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.hit_streak += 1

    @property
    def centroid(self):
        return self.state[:2].copy()

    @property
    def velocity(self):
        return self.state[2:4].copy()


class ByteTrackTracker:
    """ByteTrack-style tracker using Kalman filters + Hungarian matching.

    Uses detection area as proxy for confidence (ByteTrack uses model score).
    Two-stage matching: high-area detections first, then low-area.
    """

    def __init__(self, max_distance=50, min_hits=3, max_age=10, area_threshold=15):
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.max_age = max_age
        self.area_threshold = area_threshold  # proxy for "confidence"
        self.trackers = []
        self.next_id = 0
        self.frame_count = 0

    def _match(self, detections, track_indices):
        """Match detections to trackers using Hungarian on distance matrix."""
        if not track_indices or not detections:
            return [], [], list(track_indices), list(range(len(detections)))

        dets = [(d["centroid"], d.get("area", 10)) for d in detections]
        cost = np.zeros((len(track_indices), len(detections)), dtype=np.float64)
        for ti, tidx in enumerate(track_indices):
            pred = self.trackers[tidx].centroid
            for di, (cent, _) in enumerate(dets):
                d = np.linalg.norm(pred - np.array(cent))
                cost[ti, di] = d

        try:
            from scipy.optimize import linear_sum_assignment
            row_idx, col_idx = linear_sum_assignment(cost)
        except ImportError:
            # greedy fallback
            row_idx, col_idx = [], []
            used_d = set()
            for ti in range(len(track_indices)):
                best_d, best_di = self.max_distance + 1, -1
                for di in range(len(detections)):
                    if di in used_d:
                        continue
                    if cost[ti, di] < best_d:
                        best_d = cost[ti, di]
                        best_di = di
                if best_di >= 0:
                    row_idx.append(ti)
                    col_idx.append(best_di)
                    used_d.add(best_di)

        matches = []
        for ti, di in zip(row_idx, col_idx):
            if cost[ti, di] <= self.max_distance:
                matches.append((track_indices[ti], di))

        unmatched_tracks = [t for t in track_indices
                            if t not in [m[0] for m in matches]]
        unmatched_dets = [d for d in range(len(detections))
                          if d not in [m[1] for m in matches]]

        return matches, [], unmatched_tracks, unmatched_dets

    def update(self, detections):
        """Process one frame of detections. Returns active tracks."""
        self.frame_count += 1

        # Predict all trackers
        for t in self.trackers:
            t.predict()

        # Split detections by confidence (area proxy)
        high_dets = [(i, d) for i, d in enumerate(detections)
                     if d.get("area", 10) >= self.area_threshold]
        low_dets = [(i, d) for i, d in enumerate(detections)
                    if d.get("area", 10) < self.area_threshold]

        # Stage 1: match high-confidence detections
        track_indices = [i for i, t in enumerate(self.trackers)
                         if t.hit_streak > 0]
        matches, _, unmatched_tracks, unmatched_high = self._match(
            [d for _, d in high_dets], track_indices
        )
        for tidx, di in matches:
            d = high_dets[di][1]
            cx, cy = d["centroid"]
            self.trackers[tidx].update(cx, cy)
            self.trackers[tidx]._last_det = d

        # Stage 2: match low-confidence detections to remaining tracks
        unmatched_track_indices = [t for t in unmatched_tracks]
        low_matches, _, _, unmatched_low = self._match(
            [d for _, d in low_dets], unmatched_track_indices
        )
        for tidx, di in low_matches:
            d = low_dets[di][1]
            cx, cy = d["centroid"]
            self.trackers[tidx].update(cx, cy)
            self.trackers[tidx]._last_det = d

        # Mark unmatched tracks as lost
        for tidx in unmatched_track_indices:
            self.trackers[tidx].hit_streak = 0

        # Create new trackers for remaining high-confidence detections
        for di in unmatched_high:
            d = high_dets[di][1]
            cx, cy = d["centroid"]
            tracker = KalmanTracker(cx, cy)
            tracker._last_det = d
            tracker._id = self.next_id
            self.next_id += 1
            self.trackers.append(tracker)

        # Remove old trackers
        self.trackers = [t for t in self.trackers
                         if t.age - t.hit_streak < self.max_age or t.hit_streak > 0]

        return self.trackers


def load_detections(path):
    """Load detections JSONL, return list of dicts per frame."""
    frames = []
    with open(path) as f:
        for line in f:
            frames.append(json.loads(line))
    return sorted(frames, key=lambda e: e["frame_0based"])


def main():
    parser = argparse.ArgumentParser(
        description="ByteTrack-style firebrand tracker with Kalman filter"
    )
    parser.add_argument("--detections", default="outputs/detections_refined_0-25467.jsonl")
    parser.add_argument("--out", default="outputs/tracks_bytetrack.json")
    parser.add_argument("--max-distance", type=float, default=50)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--area-threshold", type=float, default=15,
                        help="High/low confidence split point (pixels)")
    args = parser.parse_args()

    t0 = time.time()
    frames = load_detections(args.detections)
    print(f"[INFO] Loaded {len(frames)} frames", flush=True)

    tracker = ByteTrackTracker(
        max_distance=args.max_distance,
        min_hits=args.min_hits,
        max_age=args.max_age,
        area_threshold=args.area_threshold,
    )

    # Collect raw track data: track_id -> list of detections
    raw_tracks = {}

    for fi, frame in enumerate(frames):
        dets = frame.get("detections", [])
        if not dets:
            continue

        active = tracker.update(dets)

        for t in active:
            tid = t._id
            det = t._last_det
            if tid not in raw_tracks:
                raw_tracks[tid] = []
            raw_tracks[tid].append({
                "frame_0based": frame["frame_0based"],
                "frame": frame["frame"],
                "centroid": det["centroid"],
                "bbox": det.get("bbox"),
                "area": det.get("area", 0),
                "temp_max": det.get("temp_max", 0),
                "temp_mean": det.get("temp_mean", 0),
                "kalman_cx": float(t.centroid[0]),
                "kalman_cy": float(t.centroid[1]),
                "kalman_vx": float(t.velocity[0]),
                "kalman_vy": float(t.velocity[1]),
            })

        if (fi + 1) % 5000 == 0:
            print(f"[INFO] Frame {fi+1}/{len(frames)} | "
                  f"{len(raw_tracks)} tracks so far", flush=True)

    print(f"[INFO] Raw tracks collected: {len(raw_tracks)}", flush=True)

    # Build track summaries with rich features
    BURNER_CX, BURNER_CY = 485.0, 658.0
    track_summaries = []

    for tid, detections in raw_tracks.items():
        detections.sort(key=lambda d: d["frame_0based"])
        start_frame = detections[0]["frame_0based"]
        end_frame = detections[-1]["frame_0based"]
        lifetime = end_frame - start_frame + 1

        if lifetime < 3:
            continue

        centroids = np.array([d["centroid"] for d in detections], dtype=np.float64)
        temps = np.array([d["temp_max"] for d in detections], dtype=np.float64)
        areas = np.array([d["area"] for d in detections], dtype=np.float64)
        kalman_v = np.array([[d["kalman_vx"], d["kalman_vy"]] for d in detections])

        # Velocity features from Kalman filter
        vel_mag = np.linalg.norm(kalman_v, axis=1)
        avg_speed = float(np.mean(vel_mag))
        speed_std = float(np.std(vel_mag))

        # Direction consistency
        if len(kalman_v) > 1:
            angles = np.arctan2(kalman_v[:, 1], kalman_v[:, 0])
            angle_diff = np.diff(angles)
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # wrap
            angle_consistency = 1.0 - float(np.std(angle_diff) / np.pi)
        else:
            angle_consistency = 0.0

        # Temperature trend (cooling rate)
        if len(temps) > 2:
            coeffs = np.polyfit(np.arange(len(temps)), temps, 1)
            temp_slope = float(coeffs[0])
        else:
            temp_slope = 0.0

        # Path curvature (sum of angle changes / path length)
        if len(centroids) > 2:
            vecs = np.diff(centroids, axis=0)
            vec_norms = np.linalg.norm(vecs, axis=1)
            total_dist = float(np.sum(vec_norms))
            if len(vecs) > 1 and total_dist > 0:
                dot = np.sum(vecs[:-1] * vecs[1:], axis=1)
                denom = vec_norms[:-1] * vec_norms[1:]
                cos_angles = np.clip(dot / np.clip(denom, 1e-8, None), -1, 1)
                turn_angles = np.arccos(cos_angles)
                path_curvature = float(np.sum(turn_angles) / total_dist)
            else:
                path_curvature = 0.0
        else:
            total_dist = 0.0
            path_curvature = 0.0

        # Burner proximity at start of track
        start_dist = float(np.linalg.norm(centroids[0] - np.array([BURNER_CX, BURNER_CY])))

        # Position summary
        y_mean = float(np.mean(centroids[:, 1]))
        y_std = float(np.std(centroids[:, 1]))

        track_summaries.append({
            "track_id": tid,
            "lifetime_frames": lifetime,
            "lifetime_s": round(lifetime / 29.41, 2),
            "start_frame": start_frame + 1,
            "end_frame": end_frame + 1,
            "n_detections": len(detections),
            "avg_speed_px_per_frame": round(avg_speed, 2),
            "speed_std": round(speed_std, 2),
            "total_distance_px": round(total_dist, 1),
            "angle_consistency": round(angle_consistency, 3),
            "path_curvature": round(path_curvature, 6),
            "temp_slope_C_per_frame": round(temp_slope, 3),
            "temp_start": round(float(temps[0]), 1),
            "temp_end": round(float(temps[-1]), 1),
            "burner_distance_start": round(start_dist, 1),
            "y_mean": round(y_mean, 1),
            "y_std": round(y_std, 1),
            "start_centroid": [round(centroids[0][0], 1), round(centroids[0][1], 1)],
            "end_centroid": [round(centroids[-1][0], 1), round(centroids[-1][1], 1)],
            "frames_0based": [d["frame_0based"] for d in detections],
            "centroids": [[round(c[0], 1), round(c[1], 1)] for c in centroids],
            "temps": [round(float(t), 1) for t in temps],
            "areas": [int(a) for a in areas],
        })

    track_summaries.sort(key=lambda t: t["lifetime_frames"], reverse=True)

    result = {
        "total_tracks": len(track_summaries),
        "total_detections": sum(t["n_detections"] for t in track_summaries),
        "params": {
            "max_distance": args.max_distance,
            "min_hits": args.min_hits,
            "max_age": args.max_age,
            "area_threshold": args.area_threshold,
        },
        "tracks": track_summaries,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[INFO] Done: {len(track_summaries)} tracks in {elapsed:.0f}s", flush=True)
    print(f"[INFO] Output: {out_path}", flush=True)

    # Summary
    lifetimes = [t["lifetime_frames"] for t in track_summaries]
    speeds = [t["avg_speed_px_per_frame"] for t in track_summaries]
    slopes = [t["temp_slope_C_per_frame"] for t in track_summaries]
    consistencies = [t["angle_consistency"] for t in track_summaries]
    print(f"\n=== Tracking Summary ===")
    print(f"Tracks: {len(track_summaries)}")
    print(f"Lifetime: mean={np.mean(lifetimes):.0f} +/- {np.std(lifetimes):.0f} frames")
    print(f"Speed: mean={np.mean(speeds):.3f} +/- {np.std(speeds):.3f} px/f")
    print(f"Temp slope: mean={np.mean(slopes):.3f} C/f (cooling={sum(1 for s in slopes if s<0)}, heating={sum(1 for s in slopes if s>0)})")
    print(f"Angle consistency: mean={np.mean(consistencies):.3f}")

    # Track length distribution
    short_tracks = sum(1 for t in lifetimes if t < 10)
    med_tracks = sum(1 for t in lifetimes if 10 <= t < 100)
    long_tracks = sum(1 for t in lifetimes if t >= 100)
    print(f"\nLength: short(<10)={short_tracks} med(10-100)={med_tracks} long(>=100)={long_tracks}")


if __name__ == "__main__":
    main()
