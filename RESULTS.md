# Results File Map — Vision-Only Firebrand Pipeline

All HPC paths are under `/data/gpfs/projects/punim1990/locateanything/` on Spartan.
Local copies (fetched for review) are under `/Users/firecaster/locateanything-firebrand/outputs/`.

## Main deliverables — labels (v9, current) — Spartan `outputs/vision_labels/`

| File | Contents |
|---|---|
| `detections_vision.jsonl` | **The label set**: per-frame flying-firebrand detections, all 25,467 frames (1,625 tracks / 9,600 detections) |
| `tracks_vision.json` | Track-level: motion features, accept/reject + reason per track |
| `rejected_detections.jsonl` | All rejected detections with reasons (= hard negatives for model training) |
| `labels_coco.json` | COCO-format export (detector training) |
| `yolo/*.txt` | YOLO-format export (labeled frames only) |
| `agreement_report.json` | Vision-vs-temperature agreement metrics |

## Events — Spartan `outputs/vision_events.json`

Ignition frame/time (303 / 10.30 s), first flying firebrand (323 / 10.98 s),
labels-per-minute flux, validity counters (`labels_before_ignition` = 0).

## QA / review images & clips — Spartan `outputs/vision_labels/`

| File | What to look at |
|---|---|
| `qa_roi.png` | Auto-derived fuel/burner exclusion zone (green) + flame mask (red) |
| `qa_grid.png` | Sample labeled crops (green boxes on embers) |
| `qa_onset_strip.png` | Frames around ignition + first firebrand |
| `qa_clip_*.mp4` | Annotated clips (ignition window, peak flux, midpoint) |
| `qa_disagreements.png` | Vision-only vs temp-missed example crops |

Local copies of the latest review set: `outputs/final_qa/` (incl. `qa_flux.png`
flux curve, plus diagnostic grids `burst_check.png`, `floor_check.png`,
`bottom_check.png`, `lateral_check.png`).

## Learned detector (EmberNet) — Spartan `outputs/ember_net/`

| File | Contents |
|---|---|
| `best.pt`, `final.pt` | Trained checkpoints (best = highest val F1 + its threshold) |
| `history.json` | Per-epoch training/val metrics |
| `cache_v2/` | Training patch cache (memmap + meta) |
| `detections_model.jsonl` | (after inference) model detections per frame |
| `tracks_model.json` | (after inference) model tracks after identical gating |
| `model_vs_rule.json` | (after inference) overlap summary vs rule labels |

## Logs — Spartan `logs/`

`vision_label_<jobid>.out`, `ember_net_<jobid>.out`, `ember_infer_<jobid>.out`.
Final label run: job 25950017. Training v2: job 25960754.

## Code (repo root, deployed to Spartan)

| Script | Role |
|---|---|
| `vision_autolabel.py` (+ `.slurm`) | Vision-only auto-labeler (events, detection, tracking, gates, QA, `--self-test`) |
| `compare_vision_temp_labels.py` | Vision-vs-temperature agreement report (un-swaps temp centroids) |
| `train_ember_net.py` (+ `ember_net.slurm`) | EmberNet training (5-frame stacks + motion channel, hard negatives, ignore regions) |
| `infer_ember_net.py` (+ `.slurm`) | Full-video model inference + identical tracking/gating + rule-overlap report |

## Known temp-pipeline issues found during this work (see AGENTS.md)

- `refine_detections.py:50`: centroids stored (y, x)-swapped — invalidated all
  crop-based training (ResNet/VLM) and all video overlays; physics labels'
  burner-distance gates corrupted.
- `bytetrack_tracker.py`: unmatched young trackers linger as "zombies",
  emitting stale phantom detections (vision pipeline filters these; temp
  tracks did not).
