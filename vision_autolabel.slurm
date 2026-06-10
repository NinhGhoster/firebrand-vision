#!/bin/bash
#SBATCH --job-name=vision_label
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vision_label_%j.out
#SBATCH --error=logs/vision_label_%j.err

set -euo pipefail
cd /data/gpfs/projects/punim1990/locateanything
source config.sh 2>/dev/null || true
export CONDA_ENVS_PATH=/data/gpfs/projects/punim1990/anaconda3/envs
export PATH=/data/gpfs/projects/punim1990/anaconda3/envs/locateanything/bin:$PATH
export OMP_NUM_THREADS=1

# Usage: sbatch vision_autolabel.slurm [start_frame] [end_frame]
START=${1:-0}
END=${2:--1}

echo "=== Vision-only firebrand auto-labeling ==="
echo "Start: $(date) | frames $START..$END"

python3 vision_autolabel.py \
  --frames-dir images/frames \
  --out-dir outputs/vision_labels \
  --events-out outputs/vision_events.json \
  --start "$START" --end "$END" \
  --workers 8

echo "End: $(date)"
