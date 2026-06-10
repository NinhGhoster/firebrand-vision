import json
import os
import re
import sys
import cv2
import numpy as np

FRAMES_DIR = "images/frames"
TRACKS_FILE = "outputs/tracks_refined.json"
OUTPUT_FILE = "outputs/firebrand_tracks.mp4"
FPS = 29.41

TRAIL_LENGTH = 15
FONT = cv2.FONT_HERSHEY_SIMPLEX


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def load_tracks(path):
    with open(path) as f:
        data = json.load(f)
    return data["tracks"]


def build_frame_lookup(tracks):
    lookup = {}
    for t in tracks:
        tid = t["track_id"]
        frames = t.get("frames_0based")
        centroids = t.get("centroids")
        temps = t.get("temps")
        if not frames:
            continue
        for i, fnum in enumerate(frames):
            if fnum not in lookup:
                lookup[fnum] = []
            lookup[fnum].append({
                "track_id": tid,
                "centroid": centroids[i] if centroids else None,
                "temp": temps[i] if temps else None,
            })
    return lookup


def build_track_metadata(tracks):
    meta = {}
    for t in tracks:
        tid = t["track_id"]
        meta[tid] = {
            "lifetime": t["lifetime_frames"],
            "frames": t.get("frames_0based", []),
            "centroids": t.get("centroids", []),
        }
    return meta


def draw_text(img, text, pos, color=(255, 255, 255), size=0.5, thickness=1):
    x, y = pos
    shadow = (0, 0, 0)
    cv2.putText(img, text, (x + 1, y + 1), FONT, size, shadow, thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, size, color, thickness, cv2.LINE_AA)


def main():
    # List frame files
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

    # Determine dimensions from first frame
    first_frame = cv2.imread(os.path.join(FRAMES_DIR, frame_files[0]))
    if first_frame is None:
        print(f"[ERROR] Cannot read first frame: {frame_files[0]}")
        sys.exit(1)

    frame_h, frame_w = first_frame.shape[:2]
    print(f"[INFO] Frame dimensions: {frame_w}x{frame_h}")

    # Load tracks
    print("[INFO] Loading tracks...")
    tracks = load_tracks(TRACKS_FILE)
    print(f"[INFO] {len(tracks)} tracks loaded. Building lookup...")
    lookup = build_frame_lookup(tracks)
    track_meta = build_track_metadata(tracks)
    print(f"[INFO] Frames with detections: {len(lookup)}")

    # Setup video writer
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
    skipped_frame = 0

    for fname in frame_files:
        # Extract frame index from filename (files are 1-based, convert to 0-based)
        nums = re.findall(r"\d+", fname)
        lookup_key = int(nums[-1]) - 1 if nums else 0

        # Load frame
        frame_path = os.path.join(FRAMES_DIR, fname)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"[WARN] Failed to load frame: {fname}")
            failed += 1
            continue
        loaded += 1
        actual_h, actual_w = img.shape[:2]
        if actual_w != frame_w or actual_h != frame_h:
            print(f"[WARN] Frame {fname} has unexpected size: {actual_w}x{actual_h}")
            img = cv2.resize(img, (frame_w, frame_h))

        active_dets = lookup.get(lookup_key, [])

        # Draw trajectory tails
        for det in active_dets:
            c = det["centroid"]
            if c is None:
                continue
            cx, cy = int(c[0]), int(c[1])
            tid = det["track_id"]

            meta = track_meta.get(tid)
            if not meta or meta["lifetime"] < 5:
                continue

            frames_list = meta["frames"]
            centroids_list = meta["centroids"]
            try:
                cur_pos = frames_list.index(lookup_key)
            except ValueError:
                continue

            start = max(0, cur_pos - TRAIL_LENGTH + 1)
            trail_points = []
            for j in range(start, cur_pos + 1):
                cpt = centroids_list[j]
                trail_points.append((int(cpt[0]), int(cpt[1])))

            if len(trail_points) >= 2:
                for j in range(len(trail_points) - 1):
                    alpha = (j - start) / (cur_pos - start) if cur_pos > start else 1.0
                    thickness = max(1, int(3 * alpha))
                    cv2.line(img, trail_points[j], trail_points[j + 1],
                             (0, 0, 255), thickness, cv2.LINE_AA)

        # Draw firebrands
        for det in active_dets:
            c = det["centroid"]
            if c is None:
                continue
            cx, cy = int(c[0]), int(c[1])
            temp = det["temp"]
            tid = det["track_id"]
            meta = track_meta.get(tid, {})
            lifetime = meta.get("lifetime", 0)

            # Green dot with outline
            cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
            cv2.circle(img, (cx, cy), 4, (0, 200, 0), 2)

            # Temperature label
            if temp is not None:
                label = f"{temp:.0f}C"
                tw, _ = cv2.getTextSize(label, FONT, 0.35, 1)[0]
                draw_text(img, label, (cx - tw // 2, cy - 10),
                          color=(255, 255, 255), size=0.35, thickness=1)

            # Track ID for long-lived tracks
            if lifetime >= 30:
                tid_label = f"#{tid}"
                tw, _ = cv2.getTextSize(tid_label, FONT, 0.3, 1)[0]
                draw_text(img, tid_label, (cx - tw // 2, cy + 12),
                          color=(255, 255, 0), size=0.3, thickness=1)

        # Overlay text
        time_s = lookup_key / FPS
        info_line = f"Frame {lookup_key} | t={time_s:.1f}s | Active: {len(active_dets)}"
        draw_text(img, info_line, (12, 28), color=(255, 255, 255), size=0.6, thickness=2)

        writer.write(img)

        if (loaded) % report_interval == 0:
            elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
            rate = loaded / elapsed if elapsed > 0 else 0
            eta = (total - loaded) / rate if rate > 0 else 0
            print(
                f"[INFO] {loaded}/{total} frames ({100*loaded/total:.1f}%) | "
                f"{rate:.1f} fps | ETA {eta:.0f}s | failed={failed}",
                flush=True,
            )

    writer.release()
    elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
    print(f"[INFO] Done: {loaded} frames in {elapsed:.0f}s ({loaded/elapsed:.1f} fps)")
    print(f"[INFO] Skipped (no file): {total - loaded - failed}")
    print(f"[INFO] Failed to load: {failed}")
    print(f"[INFO] Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
