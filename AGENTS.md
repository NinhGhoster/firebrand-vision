# AGENTS.md — Firebrand Vision working notes

## Project definition (from the researcher)

- A **firebrand** is only material that DETACHES and FLIES AWAY from the fuel.
  Glowing bark/twigs on the sample, flame crawling along the fuel, hot burner
  metal, reflections, smoke: NOT firebrands.
- Slow shutter ⇒ moving firebrands smear into glowing **tails**; the tail is
  part of the firebrand (thin elongated streaks must be kept).
- Scene: fuel sample inside a metal **gas burner** on a table. Fuel taxonomy:
  (a) stringy bark (trunk — the current S12 video), (b) candle bark (strip),
  (c) twig/branchlet. Only (a) exists in current data.
- Scene states to detect: pre-ignition (camera rolls before ignition),
  ignition, burning, and **burner-off vs smoldering** — when the gas flame
  stops there may still be glowing spots on fuel/charcoal; these are distinct
  states and must not be confused.
- Vision-only constraint: never read `S12.nc` or °C values in the vision
  pipeline; treat the PNGs as ordinary video. (Comparison/evaluation against
  the temperature pipeline is allowed and expected.)

## Researcher preferences (hard requirements)

- Review rendering: **rectangular bounding boxes** (see-through, never filled
  dots), **no trajectory trails**, status label above each box: `detect` when
  first seen, `tracking` once the track is confirmed/moving.
- **Maximize HPC throughput**: submit independent SLURM jobs in parallel;
  use `gpu-h100` (non-preempt) for GPU work; don't serialize what can run
  concurrently. Epochs/compute are not a constraint.
- Repo must stay GitHub-ready (this directory is the git repo).

## Data

- Source: `S12.nc` (5.9 GB, 25,467 frames, 768×1024 FLIR T1040) — Spartan only.
- Frames: `images/frames/frame_00001.png … frame_25467.png` (1-based names,
  0-based indices in code), rendered by `extract_frames.py` with FIXED
  normalization 20–600 °C → inferno (so intensities are comparable across
  frames; ~32 °C per 14 gray levels, pre-ignition scene peaks at gray ~31).
- FPS = 29.41. Static camera.

## Active pipeline (repo root)

| Script | Role |
|---|---|
| `vision_autolabel.py` (+ `.slurm`) | auto-labeler: events/states, adaptive detection, ByteTrack+gates, JSONL/COCO/YOLO labels, QA, `--self-test` (synthetic GT + 6 distractor classes) |
| `bytetrack_tracker.py` | tracker library (KalmanTracker, ByteTrackTracker) |
| `train_ember_net.py` + `ember_net.slurm` | EmberNet (5-frame + motion channel U-Net heatmap); confuser-only hard negatives; kinematic rejections = ignore regions; `--base-width` |
| `infer_ember_net.py` (+ `.slurm`) | full-video inference + identical Stage D gating + `model_vs_rule.json` |
| `render_results.py` (+ `.slurm`) | review videos per researcher conventions |
| `compare_vision_temp_labels.py` | vision-vs-temp agreement (un-swaps temp centroids!) |
| `label_platform/` | local assisted-labeling web app (Flask, port 8754): verify/correct auto-labels + add missed over stratified segments; append-only ops log `outputs/human_labels/ops.jsonl`; `export_gold.py` → gold set; `evaluate_gold.py` → rules/model vs gold |

Run `python3 vision_autolabel.py --self-test` after ANY labeler change.
Outputs map: see `RESULTS.md`. Current labels: v9 (1,625 tracks / 9,600 dets);
events: ignition f303 (10.30 s), first firebrand f323 (10.98 s).

## Deployment / SLURM (Spartan, project punim1990)

- Working dir: `/data/gpfs/projects/punim1990/locateanything/` (flat copies of
  the root scripts). Transfer: `scp <files> spartan:$LOCATE_DIR/`.
- Submit: `ssh spartan "cd $LOCATE_DIR && sbatch <script>.slurm [args]"`;
  poll `sacct -j <id> --format=State --noheader -X`; logs in `logs/`.
- Partitions: CPU default; GPU `gpu-h100` (16 nodes, non-preempt). A100
  partitions exist but preempt jobs can be killed.
- Conda env `locateanything`: set
  `PATH=/data/gpfs/projects/punim1990/anaconda3/envs/locateanything/bin:$PATH`.
  No matplotlib on Spartan (plots are rendered locally).

## CRITICAL BUG in legacy temperature pipeline (kept for the comparison study)

- `legacy_temp_pipeline/refine_detections.py:50` unpacks cv2 centroids as
  `cy, cx` → every stored temp centroid is **(y, x)-swapped** (bboxes are
  fine). Blast radius: all crop training (ResNet/VLM) and every video overlay
  in the old pipeline were misplaced; physics labels' burner-distance gates
  were computed in mixed coordinates (plume contamination in temp "firebrand"
  labels). Swap before any geometric use (`compare_vision_temp_labels.py`
  does).
- `bytetrack_tracker.py` keeps unmatched young trackers as zombies emitting
  stale detections (never re-matched). The vision pipeline filters these via
  the `id(t._last_det)` check in `run_tracking`; legacy temp tracks did not.

## Label classes (COCO)

1 firebrand (flying ember; includes motion-blur tail) — per-detection boxes
2 flame (active combustion region) — per-frame
3 fuel_stringybark (burning fuel sample) — per-chunk static-ish box
4 gas_burner (metal burner; reflects heat — context class) — static box

## Memory aids

- Local venv: `.venv/` (numpy, cv2, scipy, matplotlib, torch-cpu).
- Local review copies of outputs: `outputs/final_qa/`.
- The mp4s in repo root are legacy renders from the broken-overlay era;
  gitignored, superseded by `render_results.py` output.
