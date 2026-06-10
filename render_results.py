"""
render_results.py — Review video rendering, per researcher conventions:

  - rectangular bounding boxes (see-through center — never filled dots)
  - NO trajectory trails
  - status label above each box: 'detect' when a track is first seen,
    'tracking' once it has been associated for >= --confirm-frames frames
  - optional scene state in the header (from vision_events.json "states")

Works with rule labels (tracks_vision.json) or model output (tracks_model.json).

Usage:
  python render_results.py --tracks outputs/vision_labels/tracks_vision.json \
      --frames-dir images/frames --out outputs/vision_labels/review.mp4
  python render_results.py --start 200 --end 700 --out review_ignition.mp4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_autolabel import FPS, list_frames  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_DETECT = (0, 255, 255)    # yellow
COL_TRACK = (0, 255, 0)       # green


def load_track_boxes(tracks_path, confirm_frames):
    """Per-frame draw list: (track_id, bbox, status)."""
    data = json.load(open(tracks_path))
    per_frame = {}
    for t in data["tracks"]:
        if t.get("label", "firebrand") != "firebrand":
            continue
        frames = t["frames_0based"]
        boxes = t.get("bboxes")
        cents = t.get("centroids")
        for k, fr in enumerate(frames):
            if boxes and k < len(boxes):
                x1, y1, x2, y2 = boxes[k]
            else:  # model tracks may store centroids only
                cx, cy = cents[k]
                x1, y1, x2, y2 = cx - 4, cy - 4, cx + 4, cy + 4
            status = "tracking" if k >= confirm_frames else "detect"
            per_frame.setdefault(fr, []).append(
                (t["track_id"], (x1, y1, x2, y2), status))
    return per_frame, data


def load_states(events_path):
    if not events_path or not Path(events_path).exists():
        return None
    ev = json.load(open(events_path))
    return ev.get("states")


def state_at(states, frame):
    if not states:
        return None
    for s in states:
        if s["start_frame"] <= frame <= s["end_frame"]:
            return s["state"]
    return None


def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    y = max(th + 4, y)
    cv2.putText(img, text, (x, y - 3), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y - 3), FONT, 0.45, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Render review video")
    parser.add_argument("--tracks", default="outputs/vision_labels/tracks_vision.json")
    parser.add_argument("--events", default="outputs/vision_events.json")
    parser.add_argument("--frames-dir", default="images/frames")
    parser.add_argument("--out", default="outputs/vision_labels/review.mp4")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--confirm-frames", type=int, default=3,
                        help="detections before a track counts as 'tracking'")
    parser.add_argument("--inflate", type=int, default=6,
                        help="box inflation (px) so the object stays visible")
    args = parser.parse_args()

    per_frame, data = load_track_boxes(args.tracks, args.confirm_frames)
    states = load_states(args.events)
    frames = list_frames(args.frames_dir, args.start, args.end)
    if not frames:
        print("[ERROR] no frames")
        sys.exit(1)
    print(f"[INFO] {len(frames)} frames | {len(per_frame)} frames with boxes",
          flush=True)

    first = cv2.imread(frames[0][1])
    h, w = first.shape[:2]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (w, h))
    t0 = time.time()
    inf = args.inflate
    for n, (idx, path) in enumerate(frames):
        img = cv2.imread(path)
        if img is None:
            continue
        n_track = n_detect = 0
        for tid, (x1, y1, x2, y2), status in per_frame.get(idx, []):
            x1, y1 = int(x1) - inf, int(y1) - inf
            x2, y2 = int(x2) + inf, int(y2) + inf
            color = COL_TRACK if status == "tracking" else COL_DETECT
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            draw_label(img, f"{status} #{tid}", x1, y1, color)
            if status == "tracking":
                n_track += 1
            else:
                n_detect += 1
        head = f"f{idx}  t={idx / FPS:.1f}s  detect:{n_detect} tracking:{n_track}"
        st = state_at(states, idx)
        if st:
            head += f"  state:{st}"
        cv2.putText(img, head, (12, 26), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, head, (12, 26), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)
        if (n + 1) % 2000 == 0:
            r = (n + 1) / (time.time() - t0)
            print(f"[INFO] {n + 1}/{len(frames)} | {r:.0f} fps | "
                  f"ETA {(len(frames) - n - 1) / r:.0f}s", flush=True)
    writer.release()
    print(f"[INFO] wrote {out_path} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
