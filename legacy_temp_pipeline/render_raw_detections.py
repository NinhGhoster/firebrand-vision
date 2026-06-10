import json
import os
import re
import sys
import cv2
import numpy as np

FRAMES_DIR = "images/frames"
DETECTIONS_FILE = "outputs/detections_refined_0-25467.jsonl"
OUTPUT_FILE = "outputs/raw_detections.mp4"
FPS = 29.41


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def build_detection_lookup(detections_path):
    lookup = {}
    with open(detections_path) as f:
        for line in f:
            entry = json.loads(line)
            frame_0based = entry["frame_0based"]
            dets = entry.get("detections", [])
            for d in dets:
                if frame_0based not in lookup:
                    lookup[frame_0based] = []
                lookup[frame_0based].append(d)
    return lookup


def main():
    if not os.path.isdir(FRAMES_DIR):
        print(f"[ERROR] Frames directory not found: {FRAMES_DIR}")
        sys.exit(1)

    frame_files = sorted(
        [f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")],
        key=natural_sort_key,
    )
    if not frame_files:
        print(f"[ERROR] No PNG files in {FRAMES_DIR}")
        sys.exit(1)
    print(f"[INFO] Found {len(frame_files)} frame files")

    first = cv2.imread(os.path.join(FRAMES_DIR, frame_files[0]))
    if first is None:
        print(f"[ERROR] Cannot read first frame")
        sys.exit(1)
    frame_h, frame_w = first.shape[:2]
    print(f"[INFO] Frame dimensions: {frame_w}x{frame_h}")

    print("[INFO] Loading detections...")
    lookup = build_detection_lookup(DETECTIONS_FILE)
    print(f"[INFO] Frames with detections: {len(lookup)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (frame_w, frame_h))
    if not writer.isOpened():
        print("[ERROR] Could not open VideoWriter")
        sys.exit(1)

    t0 = cv2.getTickCount()
    total = len(frame_files)
    report_interval = max(1, total // 20)
    loaded = 0
    failed = 0

    for fname in frame_files:
        nums = re.findall(r"\d+", fname)
        frame_0based = int(nums[-1]) - 1 if nums else 0

        frame_path = os.path.join(FRAMES_DIR, fname)
        img = cv2.imread(frame_path)
        if img is None:
            failed += 1
            continue
        loaded += 1

        dets = lookup.get(frame_0based, [])

        for d in dets:
            cx, cy = int(d["centroid"][0]), int(d["centroid"][1])
            temp = d.get("temp_max", 0)
            # Blue dot for raw detection
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)
            cv2.circle(img, (cx, cy), 3, (200, 0, 0), 1)

        time_s = frame_0based / FPS
        info = f"Frame {frame_0based} | t={time_s:.1f}s | Raw: {len(dets)} detections"
        cv2.putText(img, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(img)

        if loaded % report_interval == 0:
            elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
            rate = loaded / elapsed if elapsed > 0 else 0
            eta = (total - loaded) / rate if rate > 0 else 0
            print(f"[INFO] {loaded}/{total} ({100*loaded/total:.1f}%) | "
                  f"{rate:.1f} fps | ETA {eta:.0f}s", flush=True)

    writer.release()
    elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
    print(f"[INFO] Done: {loaded} frames in {elapsed:.0f}s ({loaded/elapsed:.1f} fps)")
    print(f"[INFO] Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
