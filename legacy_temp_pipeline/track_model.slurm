#!/bin/bash
#SBATCH --job-name=track_model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/track_model_%j.out
#SBATCH --error=logs/track_model_%j.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"
source config.sh 2>/dev/null || true

export CONDA_ENVS_PATH=/data/gpfs/projects/punim1990/anaconda3/envs
export PATH=/data/gpfs/projects/punim1990/anaconda3/envs/locateanything/bin:$PATH

echo "=== Post-inference tracking ==="
echo "Start: $(date)"

# Step 1: Convert detections to tracking format
echo "--- Step 1: Convert detections ---"
python3 convert_detections.py \
  outputs/fullframe_detections.jsonl \
  --out outputs/model_dets_for_tracking.jsonl

# Step 2: Track
echo "--- Step 2: Tracking ---"
python3 track_firebrands.py \
  --detections outputs/model_dets_for_tracking.jsonl \
  --out outputs/model_tracks.json \
  --max_distance 30

# Step 3: Render video
echo "--- Step 3: Render video ---"
python3 render_video.py

echo "End: $(date)"
echo "=== Done ==="
