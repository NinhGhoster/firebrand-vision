# Firebrand Vision — vision-only firebrand detection, tracking & labeling

Detect and track **firebrands** (flying embers) in fixed-camera FLIR thermal
recordings of fuel combustion experiments — using **vision only** (the
colormapped video as ordinary footage; no radiometric temperature data) — so
the result can be compared against a temperature-threshold pipeline.

A firebrand here is strictly *material that detaches and flies away from the
fuel*. Glowing bark on the sample, flame crawling along the fuel, hot burner
metal, reflections, and smoke are all explicitly **not** firebrands and are
rejected by physical track gates.

## Pipeline

```
frames (PNG, fixed 20–600°C → inferno rendering, static camera)
  │
  ├─ vision_autolabel.py        rule-based auto-labeler (CPU, ~4 min full video)
  │    Stage A  brightness stats → ignition / burner-off / smoldering states
  │    Stage B  per-chunk temporal-median background + percentile noise σ
  │    Stage C  adaptive-threshold small-blob detection (+ flame/transient masks)
  │    Stage D  ByteTrack + tracklet stitching + physical gates:
  │             lifetime, displacement, speed, straightness,
  │             exits-fuel-zone, peak brightness, background darkness,
  │             horizon (floor) cut, thin-streak-aware diffuse rejection
  │    → labels (JSONL / COCO / YOLO) + events + QA images/clips
  │
  ├─ train_ember_net.py         EmberNet: spatio-temporal U-Net (5-frame stack
  │                             + motion channel → ember heatmap), trained on
  │                             the auto-labels with confuser hard negatives
  │                             and ignore-region masking (GPU)
  │
  ├─ infer_ember_net.py         full-video model inference + the same tracking
  │                             and gating stage + model-vs-rules report (GPU)
  │
  ├─ render_results.py          review videos: rectangular boxes (see-through),
  │                             "detect"/"tracking" status labels, no trails
  │
  └─ compare_vision_temp_labels.py   agreement vs the temperature pipeline
```

Motion blur from the slow shutter (glowing tails) is treated as part of the
firebrand: elongated thin streaks are explicitly kept.

## Key results (S12 recording, 25,467 frames, 768×1024 @ 29.41 fps)

- Ignition detected at frame 303 (t = 10.30 s); first flying firebrand at
  frame 323 (t = 10.98 s); zero pre-ignition labels.
- 1,625 flying-firebrand tracks / 9,600 labeled detections (v9 labels).
- ~90% of vision-detected embers are below the temperature pipeline's 299 °C
  threshold (sub-pixel hot objects read cold radiometrically) — vision and
  radiometric detection are complementary.

## Repo layout

| Path | Contents |
|---|---|
| `vision_autolabel.py` (+ `.slurm`) | auto-labeler, events/states, QA, `--self-test` |
| `bytetrack_tracker.py` | Kalman + ByteTrack association (library) |
| `train_ember_net.py`, `ember_net.slurm` | EmberNet training |
| `infer_ember_net.py` (+ `.slurm`) | model inference + gating + comparison |
| `render_results.py` (+ `.slurm`) | review video rendering |
| `compare_vision_temp_labels.py` | vision vs temperature agreement |
| `extract_frames.py` | NC → PNG frame rendering (fixed normalization) |
| `legacy_temp_pipeline/` | archived temperature-threshold pipeline (baseline) |
| `assets/` | sample frames |
| `AGENTS.md` | working notes: HPC workflow, conventions, known bugs |
| `RESULTS.md` | where every result file lives |

Data, frames, and outputs live on the Spartan HPC
(`/data/gpfs/projects/punim1990/locateanything/`); see `RESULTS.md`.

## Running (Spartan)

```bash
sbatch vision_autolabel.slurm 0 -1      # full-video labeling (CPU)
sbatch ember_net.slurm                  # train EmberNet (H100)
sbatch infer_ember_net.slurm            # model inference + comparison (H100)
sbatch render_results.slurm             # review video (CPU)
python3 vision_autolabel.py --self-test # synthetic ground-truth self-test
```

## Fuel taxonomy

(a) stringy bark (trunk) — the S12 recording, labeled now;
(b) candle bark (strip) and (c) twig/branchlet — future recordings.
