#!/bin/bash
#SBATCH --job-name=render
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/render_%j.out
#SBATCH --error=logs/render_%j.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"
source config.sh 2>/dev/null || true

export CONDA_ENVS_PATH=/data/gpfs/projects/punim1990/anaconda3/envs
export PATH=/data/gpfs/projects/punim1990/anaconda3/envs/locateanything/bin:$PATH

echo "=== Starting video render ==="
python3 render_video.py
echo "=== Done ==="
