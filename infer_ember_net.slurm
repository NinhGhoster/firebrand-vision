#!/bin/bash
#SBATCH --job-name=ember_infer
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ember_infer_%j.out
#SBATCH --error=logs/ember_infer_%j.err

set -euo pipefail
cd /data/gpfs/projects/punim1990/locateanything
source config.sh 2>/dev/null || true
export CONDA_ENVS_PATH=/data/gpfs/projects/punim1990/anaconda3/envs
export PATH=/data/gpfs/projects/punim1990/anaconda3/envs/locateanything/bin:$PATH
export OMP_NUM_THREADS=4

echo "=== EmberNet full-video inference ==="
echo "Start: $(date)"

python3 infer_ember_net.py \
  --checkpoint outputs/ember_net/best.pt \
  --frames-dir images/frames \
  --out-dir outputs/ember_net \
  --batch-size 8

echo "End: $(date)"
