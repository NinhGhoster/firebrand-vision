import argparse
import json
from pathlib import Path


def convert_fullframe_to_tracking(input_path, output_path):
    """Convert full-frame model detections to tracking-compatible JSONL format.
    
    Tracking format expects per-line JSON with:
    {"frame": N, "frame_0based": N-1, "detections": [{"centroid": [x,y], "area": A, ...}]}
    """
    with open(input_path) as f:
        lines = f.readlines()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    with open(output_path, "w") as out:
        for line in lines:
            entry = json.loads(line)
            boxes = entry.get("boxes", [])
            frame_0based = entry["frame_0based"]

            detections = []
            for box in boxes:
                cx = (box["x1"] + box["x2"]) / 2
                cy = (box["y1"] + box["y2"]) / 2
                w = box["x2"] - box["x1"]
                h = box["y2"] - box["y1"]
                detections.append({
                    "centroid": [round(cx, 1), round(cy, 1)],
                    "bbox": [int(box["x1"]), int(box["y1"]),
                             int(box["x2"]), int(box["y2"])],
                    "area": int(w * h),
                    "width": int(w),
                    "height": int(h),
                })
                total_boxes += 1

            out_line = {
                "frame": frame_0based + 1,
                "frame_0based": frame_0based,
                "n_candidates": len(detections),
                "detections": detections,
            }
            out.write(json.dumps(out_line) + "\n")

    print(f"[INFO] Converted {total_boxes} detections from {len(lines)} frames")
    print(f"[INFO] Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert full-frame detections to tracking format"
    )
    parser.add_argument("input", help="Full-frame detections JSONL")
    parser.add_argument("--out", default="outputs/model_detections_for_tracking.jsonl")
    args = parser.parse_args()

    convert_fullframe_to_tracking(args.input, args.out)


if __name__ == "__main__":
    main()
