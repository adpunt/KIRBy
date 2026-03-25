#!/bin/bash
#SBATCH --job-name=sim_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Run all similarity experiments for a single dataset.
# Relies on rep caching (~/.cache/kirby/rep_cache/) and hybrid caching.
#
# Usage:
#   sbatch submit_similarity_dataset.sh herg
#   sbatch submit_similarity_dataset.sh flab_thermo_tresanco
#
#   # Run specific experiments only:
#   sbatch submit_similarity_dataset.sh herg "s01 s02 s05"

DATASET=${1:?Usage: sbatch submit_similarity_dataset.sh DATASET [EXPERIMENTS]}
EXPERIMENTS=${2:-}

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

echo "=== Similarity experiments: ${DATASET} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

if [ -n "$EXPERIMENTS" ]; then
    python tests/experiments/similarity/${DATASET}.py \
        --experiments ${EXPERIMENTS} \
        --output tests/results/similarity
else
    python tests/experiments/similarity/${DATASET}.py \
        --output tests/results/similarity
fi

echo "Done:   $(date)"
