"""
evaluate_gold.py — The honest numbers: rule pipeline and EmberNet evaluated
against the human gold set (only frames the human marked done).

Matching: center distance <= --match-dist px (boxes are tiny; IoU is brittle).

Usage:
  python label_platform/evaluate_gold.py \
      --gold outputs/human_labels/gold_frames.json \
      --rules outputs/tracks_vision.json \
      --model outputs/tracks_model.json
"""

import argparse
import json
from pathlib import Path


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def load_system_tracks(path, default_half=6):
    """frame -> [centers] from a tracks json (rule or model format)."""
    per_frame = {}
    data = json.load(open(path))
    for t in data["tracks"]:
        if t.get("label", "firebrand") != "firebrand":
            continue
        boxes = t.get("bboxes")
        for k, fr in enumerate(t["frames_0based"]):
            if boxes and k < len(boxes):
                c = center(boxes[k])
            else:
                c = tuple(t["centroids"][k])
            per_frame.setdefault(fr, []).append(c)
    return per_frame


def evaluate(gold_frames, system, match_dist):
    tp = fp = fn = 0
    md2 = match_dist ** 2
    for fr_s, data in gold_frames.items():
        fr = int(fr_s)
        gold = [center(b["bbox"]) for b in data["boxes"]]
        preds = list(system.get(fr, []))
        used = set()
        for px, py in preds:
            hit = None
            for gi, (gx, gy) in enumerate(gold):
                if gi not in used and (px - gx) ** 2 + (py - gy) ** 2 <= md2:
                    hit = gi
                    break
            if hit is None:
                fp += 1
            else:
                used.add(hit)
                tp += 1
        fn += len(gold) - len(used)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def main():
    ap = argparse.ArgumentParser(description="Evaluate systems vs human gold")
    ap.add_argument("--gold", default="outputs/human_labels/gold_frames.json")
    ap.add_argument("--rules", default="outputs/tracks_vision.json")
    ap.add_argument("--model", default="outputs/tracks_model.json")
    ap.add_argument("--match-dist", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/human_labels/gold_eval.json")
    args = ap.parse_args()

    gold = json.load(open(args.gold))
    frames = gold["frames"]
    if not frames:
        print("[ERROR] gold set is empty — mark frames done and export first")
        return
    print(f"[INFO] gold: {gold['summary']}")

    report = {"summary": gold["summary"], "match_dist": args.match_dist,
              "systems": {}}
    for name, path in [("rules", args.rules), ("model", args.model)]:
        if not Path(path).exists():
            continue
        res = evaluate(frames, load_system_tracks(path), args.match_dist)
        report["systems"][name] = res
        print(f"{name:>6}: P {res['precision']:.3f}  R {res['recall']:.3f}  "
              f"F1 {res['f1']:.3f}  (tp {res['tp']} fp {res['fp']} fn {res['fn']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] wrote {args.out}")


if __name__ == "__main__":
    main()
