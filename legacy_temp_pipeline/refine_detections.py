#!/usr/bin/env python3
"""
refine_detections.py — Phase 1: Refined firebrand detection with dynamic flame masking.

Per frame:
  1. Threshold NC temperature data at firebrand_temp °C
  2. Connected components
  3. Largest component = main flame → EXCLUDE it with a buffer
  4. Small components (min_area to max_area) = firebrand candidates
  5. Filter by min distance from flame edge
  6. Output JSONL with candidate metadata (no model verification yet)

Usage:
  python refine_detections.py --nc S12.nc --out_dir outputs/
  python refine_detections.py --nc S12.nc --out_dir outputs/ --start 0 --end 1000
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import xarray as xr


def refine_frame(
    frame_data: np.ndarray,
    threshold: float = 299.0,
    min_area: int = 5,
    max_area: int = 100,
    flame_buffer_px: int = 20,
    min_distance_from_flame: float = 0,
) -> list[dict]:
    mask = (frame_data >= threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return []

    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cy, cx = centroids[i]
        blob_mask = labels == i
        region_temps = frame_data[blob_mask]
        components.append({
            "comp_id": int(i),
            "area": int(area),
            "centroid": (float(cx), float(cy)),
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
            "width": int(w),
            "height": int(h),
            "temp_min": float(region_temps.min()),
            "temp_max": float(region_temps.max()),
            "temp_mean": float(region_temps.mean()),
        })

    components.sort(key=lambda c: c["area"], reverse=True)

    if not components:
        return []

    flame = components[0]
    flame_mask = labels == flame["comp_id"]

    flame_dilated = cv2.dilate(
        flame_mask.astype(np.uint8),
        np.ones((flame_buffer_px, flame_buffer_px), np.uint8),
        iterations=1,
    ).astype(bool)

    candidates = []
    for c in components[1:]:
        if c["area"] < min_area or c["area"] > max_area:
            continue

        cx, cy = c["centroid"]
        if int(cy) < flame_mask.shape[0] and int(cx) < flame_mask.shape[1]:
            if flame_dilated[int(cy), int(cx)]:
                continue

        candidates.append(c)

    return candidates


def process_all_frames(
    nc_path: str,
    out_dir: str,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    threshold: float = 299.0,
    min_area: int = 5,
    max_area: int = 100,
    flame_buffer_px: int = 20,
    min_distance_from_flame: float = 0,
):
    ds = xr.open_dataset(nc_path, engine="netcdf4", chunks={"frame": 1})
    temp = ds.temperature
    total_frames = temp.shape[0]
    if end is None or end > total_frames:
        end = total_frames

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / f"detections_refined_{start}-{end}.jsonl"

    print(f"[INFO] Processing frames {start}-{end} (step={step})", flush=True)
    print(f"[INFO] threshold={threshold}C min_area={min_area} max_area={max_area}", flush=True)
    print(f"[INFO] flame_buffer={flame_buffer_px}px min_dist={min_distance_from_flame}px", flush=True)
    print(f"[INFO] Output: {output_file}", flush=True)

    total_candidates = 0
    n_frames = 0
    t0 = time.time()

    with open(output_file, "w") as f:
        for i in range(start, end, step):
            frame_data = temp[i].values
            candidates = refine_frame(
                frame_data,
                threshold=threshold,
                min_area=min_area,
                max_area=max_area,
                flame_buffer_px=flame_buffer_px,
                min_distance_from_flame=min_distance_from_flame,
            )

            entry = {
                "frame": i + 1,
                "frame_0based": i,
                "n_candidates": len(candidates),
                "detections": candidates,
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            total_candidates += len(candidates)
            n_frames += 1

            if n_frames % 500 == 0:
                elapsed = time.time() - t0
                rate = n_frames / elapsed
                eta = (end - start - n_frames * step) / (rate * step) if rate > 0 else 0
                print(
                    f"[INFO] {n_frames} frames | {total_candidates} candidates "
                    f"| {rate:.1f} fps | ETA {eta:.0f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"[INFO] Done: {n_frames} frames, {total_candidates} candidates", flush=True)
    print(f"[INFO] Time: {elapsed:.1f}s ({n_frames/elapsed:.1f} fps)", flush=True)
    ds.close()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Refined firebrand detection with dynamic flame masking"
    )
    parser.add_argument("--nc", required=True, help="Path to NC file")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=299.0)
    parser.add_argument("--min_area", type=int, default=5)
    parser.add_argument("--max_area", type=int, default=100)
    parser.add_argument("--flame_buffer", type=int, default=20)
    parser.add_argument("--min_distance", type=float, default=0)
    args = parser.parse_args()

    process_all_frames(
        nc_path=args.nc,
        out_dir=args.out_dir,
        start=args.start,
        end=args.end,
        step=args.step,
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        flame_buffer_px=args.flame_buffer,
        min_distance_from_flame=args.min_distance,
    )


if __name__ == "__main__":
    main()
