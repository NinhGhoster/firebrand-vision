#!/usr/bin/env python3
"""
extract_frames.py — Extract thermal frames from NC file as RGB PNGs with colormap.

Usage:
  # Extract all 25467 frames
  python extract_frames.py --nc S12.nc --out_dir frames/

  # Extract a range (for SLURM array jobs)
  python extract_frames.py --nc S12.nc --out_dir frames/ --start 0 --end 500

  # Custom temperature range for normalization
  python extract_frames.py --nc S12.nc --out_dir frames/ --t_min 20 --t_max 600
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import xarray as xr

NORM_COLORMAP = cv2.COLORMAP_INFERNO


def extract_frames(
    nc_path: str,
    out_dir: str,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    t_min: float = 20.0,
    t_max: float = 600.0,
    task_id: str = "",
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc_path, engine="netcdf4", chunks={"frame": 1})
    temp = ds.temperature
    total_frames = temp.shape[0]
    if end is None or end > total_frames:
        end = total_frames

    n_digits = len(str(total_frames))
    n_processed = 0

    print(f"[INFO] Extracting frames {start} to {end} (step={step})", flush=True)
    print(f"[INFO] Temperature range: {t_min:.0f} to {t_max:.0f} °C", flush=True)
    t0 = time.time()

    for i in range(start, end, step):
        frame_data = temp[i].values
        frame_clipped = np.clip(frame_data, t_min, t_max)
        frame_normalized = ((frame_clipped - t_min) / (t_max - t_min) * 255).astype(
            np.uint8
        )
        frame_colored = cv2.applyColorMap(frame_normalized, NORM_COLORMAP)
        frame_colored_rgb = cv2.cvtColor(frame_colored, cv2.COLOR_BGR2RGB)

        fname = f"frame_{i+1:0{n_digits}d}.png"
        fpath = out_path / fname
        cv2.imwrite(str(fpath), cv2.cvtColor(frame_colored_rgb, cv2.COLOR_RGB2BGR))

        n_processed += 1
        if n_processed % 100 == 0:
            elapsed = time.time() - t0
            rate = n_processed / elapsed
            eta = (end - start - n_processed) / rate if rate > 0 else 0
            print(
                f"[{task_id}] {n_processed} frames ({i+1}/{end}) "
                f"| {rate:.1f} fps | ETA {eta:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"[INFO] Done: {n_processed} frames in {elapsed:.1f}s ({n_processed/elapsed:.1f} fps)",
        flush=True,
    )
    ds.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract thermal NC frames as colormap PNGs"
    )
    parser.add_argument("--nc", required=True, help="Path to NC file")
    parser.add_argument(
        "--out_dir", default="frames", help="Output directory for PNGs"
    )
    parser.add_argument("--start", type=int, default=0, help="First frame (0-based)")
    parser.add_argument("--end", type=int, default=None, help="Last frame (exclusive)")
    parser.add_argument("--step", type=int, default=1, help="Frame step")
    parser.add_argument(
        "--t_min", type=float, default=20.0, help="Min temperature for normalization"
    )
    parser.add_argument(
        "--t_max", type=float, default=600.0, help="Max temperature for normalization"
    )
    parser.add_argument(
        "--task_id", type=str, default="", help="Task ID for logging"
    )
    args = parser.parse_args()

    extract_frames(
        nc_path=args.nc,
        out_dir=args.out_dir,
        start=args.start,
        end=args.end,
        step=args.step,
        t_min=args.t_min,
        t_max=args.t_max,
        task_id=args.task_id,
    )


if __name__ == "__main__":
    main()
