#!/bin/bash
#SBATCH --job-name=nz_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Run all noise experiments for a single dataset.
# Relies on rep caching (~/.cache/kirby/rep_cache/) and hybrid caching.
#
# Usage:
#   sbatch submit_noise_dataset.sh herg
#   sbatch submit_noise_dataset.sh flab_thermo_tresanco
#
#   # Run specific experiments only:
#   sbatch submit_noise_dataset.sh herg "n01 n02"

DATASET=${1:?Usage: sbatch submit_noise_dataset.sh DATASET [EXPERIMENTS]}
EXPERIMENTS=${2:-}

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

echo "=== Noise experiments: ${DATASET} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

if [ -n "$EXPERIMENTS" ]; then
    python tests/experiments/noise/${DATASET}.py \
        --experiments ${EXPERIMENTS} \
        --output tests/results/noise
else
    python tests/experiments/noise/${DATASET}.py \
        --output tests/results/noise
fi

echo "Done:   $(date)"
