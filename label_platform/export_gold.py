"""
export_gold.py — Turn the human ops log into the gold label set.

Only frames explicitly marked DONE become gold: their boxes are the accepted
proposals (with any geometry adjustments) plus human-drawn boxes (keyframes +
interpolation). Rejected proposals on done frames are recorded as verified
negatives (hard-negative training signal).

Outputs (outputs/human_labels/):
  gold_frames.json   per-frame gold boxes + negatives + stats
  gold_coco.json     COCO export (category: firebrand, attribute source)
"""

import json
from pathlib import Path


def build_gold(state, proposals, segments, out_dir):
    out_dir = Path(out_dir)
    frames = {}
    n_boxes = n_neg = 0
    for fr in sorted(state.frames_done):
        boxes, negatives = [], []
        for pb in proposals.get(fr, []):
            s, tid = pb["source"], pb["track_id"]
            status = state.status_of(s, tid, fr)
            bbox = state.box_geom.get((s, tid, fr), pb["bbox"])
            if status == "accepted":
                boxes.append({"bbox": [round(v, 1) for v in bbox],
                              "source": s, "track_id": tid})
            elif status == "rejected":
                negatives.append({"bbox": [round(v, 1) for v in bbox],
                                  "source": s, "track_id": tid})
        for hb in state.human_boxes_at(fr):
            boxes.append({"bbox": [round(v, 1) for v in hb["bbox"]],
                          "source": "human", "track_id": f"h{hb['hid']}"})
        frames[fr] = {"boxes": boxes, "negatives": negatives}
        n_boxes += len(boxes)
        n_neg += len(negatives)

    summary = {
        "done_frames": len(frames),
        "gold_boxes": n_boxes,
        "verified_negatives": n_neg,
        "human_tracks": len(state.human),
    }
    with open(out_dir / "gold_frames.json", "w") as f:
        json.dump({"segments": segments, "summary": summary,
                   "frames": {str(k): v for k, v in frames.items()}}, f, indent=1)

    images, annotations = [], []
    ann_id = 1
    for fr, data in frames.items():
        images.append({"id": fr, "file_name": f"frame_{fr + 1:05d}.png",
                       "width": 1024, "height": 768})
        for b in data["boxes"]:
            x1, y1, x2, y2 = b["bbox"]
            annotations.append({
                "id": ann_id, "image_id": fr, "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                "attributes": {"source": b["source"]},
            })
            ann_id += 1
    with open(out_dir / "gold_coco.json", "w") as f:
        json.dump({"images": images, "annotations": annotations,
                   "categories": [{"id": 1, "name": "firebrand"}]}, f)
    return summary
