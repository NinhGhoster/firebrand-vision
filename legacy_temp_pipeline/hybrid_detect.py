#!/usr/bin/env python3
"""
hybrid_detect.py — Hybrid firebrand detection: temperature thresholding
for candidate finding + LocateAnything visual verification on ROIs.

Pipeline per frame:
  1. Read NC temperature data
  2. Threshold at firebrand_temp °C → binary mask
  3. Connected components → candidate blobs
  4. Filter by size (min_area, max_area)
  5. For each candidate, crop ROI from colormap image
  6. Run LocateAnything on ROI for visual verification
  7. Record results with frame#, centroid, bbox, temp stats

Usage:
  # Temperature-only detection (fast, no GPU needed)
  python hybrid_detect.py --nc S12.nc --out_dir outputs/ --start 0 --end 100

  # Full hybrid with model verification
  python hybrid_detect.py --nc S12.nc --frames_dir frames/ --model nvidia/LocateAnything-3B
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import xarray as xr

NORM_COLORMAP = cv2.COLORMAP_INFERNO


def temperature_to_colormap(frame_data, t_min=20, t_max=600):
    clipped = np.clip(frame_data, t_min, t_max)
    normalized = ((clipped - t_min) / (t_max - t_min) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, NORM_COLORMAP)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def find_candidates(temp_frame, threshold=299, min_area=5, max_area=5000):
    mask = (temp_frame >= threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cy, cx = centroids[i]
        blob_mask = labels == i
        region_temps = temp_frame[blob_mask]
        candidates.append({
            "blob_id": i,
            "area": int(area),
            "centroid": (float(cx), float(cy)),
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
            "temp_min": float(region_temps.min()),
            "temp_max": float(region_temps.max()),
            "temp_mean": float(region_temps.mean()),
        })
    return candidates


class HybridDetector:
    def __init__(
        self,
        nc_path: str,
        model_path: str | None = None,
        threshold: float = 299.0,
        min_area: int = 5,
        max_area: int = 5000,
        t_min: float = 20.0,
        t_max: float = 600.0,
    ):
        self.nc_path = nc_path
        self.ds = xr.open_dataset(nc_path, engine="netcdf4", chunks={"frame": 1})
        self.temp = self.ds.temperature
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.t_min = t_min
        self.t_max = t_max

        self.model = None
        self.processor = None
        self.tokenizer = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        print(f"[INFO] Loading model {model_path} ...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            .to("cuda")
            .eval()
        )
        print(f"[INFO] Model loaded in {time.time() - t0:.1f}s", flush=True)

    def verify_roi(self, roi_image: np.ndarray) -> bool:
        if self.model is None:
            return True
        import torch
        from PIL import Image

        img_pil = Image.fromarray(roi_image)
        question = "Locate all the instances that matches the following description: firebrand."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_pil},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, _ = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt"
        ).to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                use_cache=True,
                generation_mode="fast",
                temperature=0.0,
                do_sample=False,
            )
        answer = response[0] if isinstance(response, tuple) else response
        has_boxes = "<box>" in answer
        return has_boxes

    def process_frame(self, frame_idx: int) -> list[dict]:
        frame_data = self.temp[frame_idx].values
        candidates = find_candidates(
            frame_data,
            threshold=self.threshold,
            min_area=self.min_area,
            max_area=self.max_area,
        )

        if not candidates:
            return []

        if self.model is not None:
            vis = temperature_to_colormap(
                frame_data, self.t_min, self.t_max
            )
            verified = []
            for c in candidates:
                x1, y1, x2, y2 = c["bbox"]
                x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
                x2, y2 = min(vis.shape[1], x2 + 10), min(vis.shape[0], y2 + 10)
                roi = vis[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                is_firebrand = self.verify_roi(roi)
                c["verified"] = is_firebrand
                if is_firebrand:
                    verified.append(c)
            return verified
        else:
            return candidates

    def close(self):
        self.ds.close()


def process_range(
    nc_path: str,
    frames_dir: str | None,
    model_path: str | None,
    output_dir: str,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    threshold: float = 299.0,
    min_area: int = 5,
    max_area: int = 5000,
    t_min: float = 20.0,
    t_max: float = 600.0,
):
    detector = HybridDetector(
        nc_path=nc_path,
        model_path=model_path,
        threshold=threshold,
        min_area=min_area,
        max_area=max_area,
        t_min=t_min,
        t_max=t_max,
    )

    total_frames = detector.temp.shape[0]
    if end is None or end > total_frames:
        end = total_frames

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    use_model = model_path is not None
    mode_str = "hybrid" if use_model else "temp-only"
    output_file = out_path / f"detections_{mode_str}_{start}-{end}.jsonl"

    print(f"[INFO] Processing frames {start}-{end} (step={step})", flush=True)
    print(f"[INFO] Mode: {mode_str}, threshold={threshold}°C", flush=True)
    print(f"[INFO] Output: {output_file}", flush=True)

    n_frames = 0
    n_detections = 0
    t0 = time.time()

    with open(output_file, "w") as f:
        for i in range(start, end, step):
            frame_t0 = time.time()
            results = detector.process_frame(i)
            frame_time = time.time() - frame_t0

            entry = {
                "frame": i + 1,
                "frame_0based": i,
                "n_candidates": len(results),
                "time_s": round(frame_time, 3),
                "detections": results,
            }
            f.write(json.dumps(entry) + "\n")
            n_frames += 1
            n_detections += len(results)

            if n_frames % 50 == 0:
                elapsed = time.time() - t0
                rate = n_frames / elapsed
                eta = (end - start - n_frames * step) / (rate * step) if rate > 0 else 0
                print(
                    f"[INFO] {n_frames} frames | "
                    f"{n_detections} detections | "
                    f"{rate:.1f} fps | ETA {eta:.0f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"[INFO] Done: {n_frames} frames, {n_detections} detections", flush=True)
    print(f"[INFO] Time: {elapsed:.1f}s ({n_frames/elapsed:.1f} fps)", flush=True)

    detector.close()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid firebrand detection (temperature + visual verification)"
    )
    parser.add_argument("--nc", required=True, help="Path to NC file")
    parser.add_argument("--frames_dir", default=None, help="Colormap PNG frames (needed for model verification)")
    parser.add_argument("--model", default=None, help="Model path (None = temperature-only)")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="First frame (0-based)")
    parser.add_argument("--end", type=int, default=None, help="Last frame (exclusive)")
    parser.add_argument("--step", type=int, default=1, help="Frame step")
    parser.add_argument("--threshold", type=float, default=299.0, help="Temperature threshold °C")
    parser.add_argument("--min_area", type=int, default=5, help="Min blob area (pixels)")
    parser.add_argument("--max_area", type=int, default=5000, help="Max blob area (pixels)")
    parser.add_argument("--t_min", type=float, default=20.0, help="Visualization min temp")
    parser.add_argument("--t_max", type=float, default=600.0, help="Visualization max temp")
    args = parser.parse_args()

    process_range(
        nc_path=args.nc,
        frames_dir=args.frames_dir,
        model_path=args.model,
        output_dir=args.out_dir,
        start=args.start,
        end=args.end,
        step=args.step,
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        t_min=args.t_min,
        t_max=args.t_max,
    )


if __name__ == "__main__":
    main()
