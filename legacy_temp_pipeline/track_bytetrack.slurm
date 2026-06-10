#!/bin/bash
#SBATCH --job-name=bytetrack
#SBATCH --partition=gpu-a100-preempt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bytetrack_%j.out
#SBATCH --error=logs/bytetrack_%j.err

set -euo pipefail

cd /data/gpfs/projects/punim1990/locateanything
source config.sh 2>/dev/null || true

export CONDA_ENVS_PATH=/data/gpfs/projects/punim1990/anaconda3/envs
export PATH=/data/gpfs/projects/punim1990/anaconda3/envs/locateanything/bin:$PATH
export OMP_NUM_THREADS=4

echo "Start: $(date)"

python3 bytetrack_tracker.py \
  --detections outputs/detections_refined_0-25467.jsonl \
  --out outputs/tracks_bytetrack.json \
  --max-distance 50 --min-hits 3 --max-age 10 --area-threshold 15

echo "End: $(date)"
