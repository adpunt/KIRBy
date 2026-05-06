#!/bin/bash
#SBATCH --job-name=exp_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

# Run experiments for one (category, dataset) pair.
#
# Usage:
#   sbatch submit_dataset.sh complexity herg
#   sbatch submit_dataset.sh noise flab_thermo_tresanco
#   sbatch submit_dataset.sh similarity logd "s01 s02 s05"
#
# Pruning mode (default: aggressive_pruning):
#   sbatch --export=PRUNING_MODE=default_pruning submit_dataset.sh complexity herg
#
# SLURM defaults: long, 7d, 128G, 8 cpus.
# Override at sbatch time if needed:
#   sbatch --time=2-00:00:00 --mem=64G submit_dataset.sh complexity caco2

CATEGORY=${1:?Usage: sbatch submit_dataset.sh CATEGORY DATASET [EXPERIMENTS]}
DATASET=${2:?Usage: sbatch submit_dataset.sh CATEGORY DATASET [EXPERIMENTS]}
EXPERIMENTS=${3:-}
PRUNING_MODE="${PRUNING_MODE:-aggressive_pruning}"

cd /data/stat-ecr/scat9264/KIRBy

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

. setup.sh

echo "=== ${CATEGORY} experiments: ${DATASET} ==="
echo "Branch: ${BRANCH} (${COMMIT})"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Pruning mode: ${PRUNING_MODE}"
echo "Start:  $(date)"

if [ -n "$EXPERIMENTS" ]; then
    python tests/experiments/run_dataset.py ${CATEGORY} ${DATASET} \
        --pruning-mode "${PRUNING_MODE}" \
        --experiments ${EXPERIMENTS}
else
    python tests/experiments/run_dataset.py ${CATEGORY} ${DATASET} \
        --pruning-mode "${PRUNING_MODE}"
fi

echo "Done:   $(date)"
