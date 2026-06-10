#!/bin/bash
#SBATCH --job-name=ember_net
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ember_net_%j.out
#SBATCH --error=logs/ember_net_%j.err

set -euo pipefail
cd /data/gpfs/projects/punim1990/locateanything
source config.sh 2>/dev/null || true
export CONDA_ENVS_PATH=/data/gpfs/projects/punim1990/anaconda3/envs
export PATH=/data/gpfs/projects/punim1990/anaconda3/envs/locateanything/bin:$PATH
export HF_HOME=/data/gpfs/projects/punim1990/hf_cache
export OMP_NUM_THREADS=4

echo "=== EmberNet training ==="
echo "Start: $(date)"

python3 train_ember_net.py \
  --frames-dir images/frames \
  --labels-dir outputs/vision_labels \
  --out outputs/ember_net_h100 \
  --cache-dir outputs/ember_net/cache_v2 \
  --epochs 60 \
  --batch-size 128 \
  --base-width 32 \
  --workers 12

echo "End: $(date)"
