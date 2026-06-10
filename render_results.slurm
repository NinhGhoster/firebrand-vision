#!/bin/bash
#SBATCH --job-name=render_review
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/render_review_%j.out
#SBATCH --error=logs/render_review_%j.err

set -euo pipefail
cd /data/gpfs/projects/punim1990/locateanything
source config.sh 2>/dev/null || true
export CONDA_ENVS_PATH=/data/gpfs/projects/punim1990/anaconda3/envs
export PATH=/data/gpfs/projects/punim1990/anaconda3/envs/locateanything/bin:$PATH
export OMP_NUM_THREADS=4

# Usage: sbatch render_results.slurm [start] [end] [out] [tracks]
START=${1:-0}
END=${2:--1}
OUT=${3:-outputs/vision_labels/review.mp4}
TRACKS=${4:-outputs/vision_labels/tracks_vision.json}

echo "=== Review render ==="
echo "Start: $(date) | frames $START..$END -> $OUT"

python3 render_results.py \
  --tracks "$TRACKS" \
  --events outputs/vision_events.json \
  --frames-dir images/frames \
  --out "$OUT" \
  --start "$START" --end "$END"

echo "End: $(date)"
