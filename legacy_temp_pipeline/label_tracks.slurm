#!/bin/bash
#SBATCH --job-name=label_tracks
#SBATCH --partition=gpu-a100-preempt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/label_tracks_%j.out
#SBATCH --error=logs/label_tracks_%j.err

set -euo pipefail

cd /data/gpfs/projects/punim1990/locateanything
source config.sh 2>/dev/null || true

export CONDA_ENVS_PATH=/data/gpfs/projects/punim1990/anaconda3/envs
export PATH=/data/gpfs/projects/punim1990/anaconda3/envs/locateanything/bin:$PATH
export OMP_NUM_THREADS=4

echo "Start: $(date)"

python3 label_tracks.py \
  --tracks outputs/tracks_bytetrack.json \
  --out-tracks outputs/tracks_labeled.json \
  --out-metadata outputs/metadata_labeled.jsonl \
  --frames-dir images/frames \
  --firebrand-min-speed 1.5 \
  --firebrand-min-consistency 0.7 \
  --firebrand-min-burner-dist 100 \
  --not-firebrand-max-speed 0.5 \
  --not-firebrand-max-consistency 0.4 \
  --not-firebrand-max-burner-dist 50 \
  --min-lifetime 10

echo "End: $(date)"
