#!/bin/bash
#SBATCH --job-name=render_raw
#SBATCH --partition=gpu-a100-preempt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/render_raw_%j.out
#SBATCH --error=logs/render_raw_%j.err

set -euo pipefail

cd /data/gpfs/projects/punim1990/locateanything
source config.sh 2>/dev/null || true

export CONDA_ENVS_PATH=/data/gpfs/projects/punim1990/anaconda3/envs
export PATH=/data/gpfs/projects/punim1990/anaconda3/envs/locateanything/bin:$PATH
export OMP_NUM_THREADS=4

echo "Start: $(date)"

python3 render_raw_detections.py

echo "End: $(date)"
