#!/bin/bash
#SBATCH --job-name=exp_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Run experiments for one (category, dataset) pair.
# Override SLURM resources at sbatch time:
#   sbatch --time=96:00:00 --mem=128G submit_dataset.sh federated herg
#
# Usage:
#   sbatch submit_dataset.sh complexity herg
#   sbatch submit_dataset.sh noise flab_thermo_tresanco
#   sbatch submit_dataset.sh similarity logd "s01 s02 s05"
#   sbatch submit_dataset.sh federated herg

CATEGORY=${1:?Usage: sbatch submit_dataset.sh CATEGORY DATASET [EXPERIMENTS]}
DATASET=${2:?Usage: sbatch submit_dataset.sh CATEGORY DATASET [EXPERIMENTS]}
EXPERIMENTS=${3:-}

cd /data/stat-cadd/scat9264/KIRBy

# Checkout the right branch and log it
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

. setup.sh

echo "=== ${CATEGORY} experiments: ${DATASET} ==="
echo "Branch: ${BRANCH} (${COMMIT})"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

if [ -n "$EXPERIMENTS" ]; then
    python tests/experiments/run_dataset.py ${CATEGORY} ${DATASET} \
        --experiments ${EXPERIMENTS}
else
    python tests/experiments/run_dataset.py ${CATEGORY} ${DATASET}
fi

echo "Done:   $(date)"
