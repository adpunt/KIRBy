#!/bin/bash
#SBATCH --job-name=cx_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Run all complexity experiments for a single dataset.
# Relies on rep caching (~/.cache/kirby/rep_cache/) and hybrid caching.
#
# Usage:
#   sbatch submit_complexity_dataset.sh herg
#   sbatch submit_complexity_dataset.sh flab_thermo_tresanco
#   sbatch submit_complexity_dataset.sh tcga_luad_vs_lusc
#
#   # Run specific experiments only:
#   sbatch submit_complexity_dataset.sh herg "c01 c02 c05"

DATASET=${1:?Usage: sbatch submit_complexity_dataset.sh DATASET [EXPERIMENTS]}
EXPERIMENTS=${2:-}

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

echo "=== Complexity experiments: ${DATASET} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

if [ -n "$EXPERIMENTS" ]; then
    python tests/experiments/complexity/${DATASET}.py \
        --experiments ${EXPERIMENTS} \
        --output tests/results/complexity
else
    python tests/experiments/complexity/${DATASET}.py \
        --output tests/results/complexity
fi

echo "Done:   $(date)"
