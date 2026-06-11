"""
label_platform/server.py — Local assisted-labeling server.

Serves the labeling UI, frames, and merged label state (rule + model
proposals + human edits). Every human action is appended to an append-only
ops log (crash-proof autosave); state is rebuilt by replaying the log.

Run:
  .venv/bin/python label_platform/server.py
  -> open http://localhost:8754
"""

import json
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "images" / "frames"
OUT_DIR = ROOT / "outputs" / "human_labels"
OPS_PATH = OUT_DIR / "ops.jsonl"
RULE_TRACKS = ROOT / "outputs" / "tracks_vision.json"
MODEL_TRACKS = ROOT / "outputs" / "tracks_model.json"
SCENE_OBJECTS = ROOT / "outputs" / "scene_objects.jsonl"

FPS = 29.41
SEGMENTS = [
    {"name": "pre-ignition control", "start": 0, "end": 250},
    {"name": "ignition + onset", "start": 250, "end": 700},
    {"name": "early peak", "start": 1300, "end": 1750},
    {"name": "early burn", "start": 2800, "end": 3250},
    {"name": "t~170s", "start": 5000, "end": 5450},
    {"name": "t~272s", "start": 8000, "end": 8450},
    {"name": "t~374s", "start": 11000, "end": 11450},
    {"name": "t~476s", "start": 14000, "end": 14450},
    {"name": "t~595s", "start": 17500, "end": 17950},
    {"name": "t~714s", "start": 21000, "end": 21450},
    {"name": "late burn", "start": 24500, "end": 24950},
]

app = Flask(__name__, static_folder="static")


# ----------------------------------------------------------------------------
# Proposals (read-only inputs)
# ----------------------------------------------------------------------------

def load_proposals():
    """frame -> [{source, track_id, bbox}]; track -> frames map for track ops."""
    by_frame, track_frames = {}, {}
    if RULE_TRACKS.exists():
        data = json.load(open(RULE_TRACKS))
        for t in data["tracks"]:
            if t.get("label") != "firebrand":
                continue
            tid = t["track_id"]
            boxes = t.get("bboxes") or []
            for k, fr in enumerate(t["frames_0based"]):
                if k < len(boxes):
                    bbox = [float(v) for v in boxes[k]]
                else:
                    cx, cy = t["centroids"][k]
                    bbox = [cx - 5, cy - 5, cx + 5, cy + 5]
                by_frame.setdefault(fr, []).append(
                    {"source": "rule", "track_id": tid, "bbox": bbox})
                track_frames.setdefault(("rule", tid), []).append(fr)
    if MODEL_TRACKS.exists():
        data = json.load(open(MODEL_TRACKS))
        for t in data["tracks"]:
            tid = t["track_id"]
            for k, fr in enumerate(t["frames_0based"]):
                cx, cy = t["centroids"][k]
                bbox = [cx - 6, cy - 6, cx + 6, cy + 6]
                by_frame.setdefault(fr, []).append(
                    {"source": "model", "track_id": tid, "bbox": bbox})
                track_frames.setdefault(("model", tid), []).append(fr)
    return by_frame, track_frames


def load_scene():
    scene = {}
    if SCENE_OBJECTS.exists():
        with open(SCENE_OBJECTS) as f:
            for line in f:
                e = json.loads(line)
                scene[e["frame_0based"]] = {
                    "flame": e.get("flame_bbox"),
                    "fuel": e.get("fuel_bbox"),
                    "burner": e.get("burner_bbox"),
                }
    return scene


PROPOSALS, TRACK_FRAMES = load_proposals()
SCENE = load_scene()


# ----------------------------------------------------------------------------
# Human state (derived from the ops log)
# ----------------------------------------------------------------------------

class State:
    def __init__(self):
        self.track_status = {}   # (source, tid) -> accepted|rejected
        self.box_status = {}     # (source, tid, frame) -> accepted|rejected
        self.box_geom = {}       # (source, tid, frame) -> bbox override
        self.human = {}          # hid -> {frame(int): bbox}
        self.next_hid = 1
        self.frames_done = set()

    def apply(self, op):
        kind, p = op["op"], op.get("payload", {})
        key_t = (p.get("source"), p.get("track_id"))
        key_b = (p.get("source"), p.get("track_id"), p.get("frame"))
        if kind == "accept_box":
            self.box_status[key_b] = "accepted"
        elif kind == "reject_box":
            self.box_status[key_b] = "rejected"
        elif kind == "accept_track":
            self.track_status[key_t] = "accepted"
        elif kind == "reject_track":
            self.track_status[key_t] = "rejected"
        elif kind == "move_box":
            self.box_geom[key_b] = p["bbox"]
            self.box_status[key_b] = "accepted"
        elif kind == "add_keyframe":
            hid = p.get("hid") or self.next_hid
            self.human.setdefault(hid, {})[int(p["frame"])] = p["bbox"]
            self.next_hid = max(self.next_hid, hid + 1)
        elif kind == "delete_keyframe":
            self.human.get(p["hid"], {}).pop(int(p["frame"]), None)
            if not self.human.get(p["hid"]):
                self.human.pop(p["hid"], None)
        elif kind == "delete_human_track":
            self.human.pop(p["hid"], None)
        elif kind == "mark_frame_done":
            if p.get("done", True):
                self.frames_done.add(int(p["frame"]))
            else:
                self.frames_done.discard(int(p["frame"]))

    def human_boxes_at(self, frame):
        """Keyframes + linear interpolation between consecutive keyframes."""
        out = []
        for hid, kfs in self.human.items():
            fr = sorted(kfs)
            if not fr:
                continue
            if frame in kfs:
                out.append({"hid": hid, "bbox": kfs[frame], "keyframe": True})
            elif fr[0] < frame < fr[-1]:
                prev = max(f for f in fr if f < frame)
                nxt = min(f for f in fr if f > frame)
                a = (frame - prev) / (nxt - prev)
                b0, b1 = kfs[prev], kfs[nxt]
                bbox = [b0[i] + a * (b1[i] - b0[i]) for i in range(4)]
                out.append({"hid": hid, "bbox": bbox, "keyframe": False})
        return out

    def status_of(self, source, tid, frame):
        return (self.box_status.get((source, tid, frame))
                or self.track_status.get((source, tid))
                or "pending")


STATE = State()
EFFECTIVE_OPS = []  # ops with undos resolved; replayed to rebuild STATE


def rebuild_state():
    global STATE
    STATE = State()
    for op in EFFECTIVE_OPS:
        STATE.apply(op)


def ingest(op):
    """Apply one logged op, honouring 'undo' (cancels the previous op)."""
    if op.get("op") == "undo":
        if EFFECTIVE_OPS:
            EFFECTIVE_OPS.pop()
            rebuild_state()
    else:
        EFFECTIVE_OPS.append(op)
        STATE.apply(op)


OUT_DIR.mkdir(parents=True, exist_ok=True)
if OPS_PATH.exists():
    with open(OPS_PATH) as f:
        for line in f:
            try:
                ingest(json.loads(line))
            except Exception:
                pass
    print(f"[INFO] replayed ops log: {OPS_PATH} "
          f"({len(EFFECTIVE_OPS)} effective ops)")


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/config")
def config():
    return jsonify({"segments": SEGMENTS, "fps": FPS,
                    "frame_width": 1024, "frame_height": 768})


@app.get("/frame/<int:idx>.jpg")
def frame(idx):
    path = FRAMES_DIR / f"frame_{idx + 1:05d}.png"
    if not path.exists():
        return ("missing", 404)
    return send_file(path, mimetype="image/png", max_age=3600)


@app.get("/api/labels/<int:idx>")
def labels(idx):
    props = []
    for pb in PROPOSALS.get(idx, []):
        s, tid = pb["source"], pb["track_id"]
        props.append({
            "source": s, "track_id": tid,
            "bbox": STATE.box_geom.get((s, tid, idx), pb["bbox"]),
            "status": STATE.status_of(s, tid, idx),
        })
    return jsonify({
        "frame": idx,
        "proposals": props,
        "human": STATE.human_boxes_at(idx),
        "scene": SCENE.get(idx, {}),
        "done": idx in STATE.frames_done,
        "next_hid": STATE.next_hid,
    })


@app.post("/api/op")
def post_op():
    op = request.get_json(force=True)
    op["ts"] = time.time()
    with open(OPS_PATH, "a") as f:
        f.write(json.dumps(op) + "\n")
    ingest(op)
    return jsonify({"ok": True, "next_hid": STATE.next_hid})


@app.get("/api/progress")
def progress():
    out = []
    for seg in SEGMENTS:
        total = seg["end"] - seg["start"]
        done = sum(1 for f in range(seg["start"], seg["end"])
                   if f in STATE.frames_done)
        out.append({**seg, "done": done, "total": total})
    return jsonify({"segments": out,
                    "human_tracks": len(STATE.human),
                    "ops": sum(1 for _ in open(OPS_PATH)) if OPS_PATH.exists() else 0})


@app.post("/api/export")
def export():
    from export_gold import build_gold
    summary = build_gold(STATE, PROPOSALS, SEGMENTS, OUT_DIR)
    return jsonify(summary)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    print(f"[INFO] proposals: {sum(len(v) for v in PROPOSALS.values())} boxes "
          f"on {len(PROPOSALS)} frames")
    app.run(host="127.0.0.1", port=8754, debug=False)
