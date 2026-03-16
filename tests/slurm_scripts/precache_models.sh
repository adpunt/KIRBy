#!/bin/bash
#SBATCH --job-name=precache_models
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-precache-%j.out
#
# Pre-download all HuggingFace model weights so compute-node jobs
# don't stall on network downloads.
#
# Usage:
#   sbatch tests/slurm_scripts/precache_models.sh
#   # or on login node directly (if login has enough RAM):
#   bash tests/slurm_scripts/precache_models.sh

cd /data/stat-cadd/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh

export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

echo "============================================"
echo "Pre-caching all model weights"
echo "Started: $(date)"
echo "============================================"

python tests/slurm_scripts/precache_models.py all

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
