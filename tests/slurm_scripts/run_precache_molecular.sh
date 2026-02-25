#!/bin/bash
#SBATCH --job-name=kirby_precache_mol
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# =============================================================================
# Precache molecular representations (subprocess-isolated)
#
# Each rep runs in its own subprocess with a timeout, so crashes (SIGBUS),
# hangs, and import errors in one rep can't take down the rest.
# Re-runnable: skips reps that already have cached .npy files.
#
# Usage:
#   sbatch run_precache_molecular.sh              # all datasets, all reps, GPU
#   sbatch run_precache_molecular.sh logd          # single dataset
#   sbatch run_precache_molecular.sh logd chemberta,molformer  # specific reps
# =============================================================================

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

DATASET=${1:-}
REPS=${2:-}

echo "============================================================"
echo "KIRBy Molecular Precache (subprocess-isolated)"
echo "  Dataset:  ${DATASET:-all}"
echo "  Reps:     ${REPS:-all (16)}"
echo "  GPU:      yes"
echo "  Started:  $(date)"
echo "============================================================"

CMD="python scripts/precache_molecular_reps.py --gpu"

if [ -n "$DATASET" ]; then
    CMD="$CMD $DATASET"
else
    CMD="$CMD --all-datasets"
fi

if [ -n "$REPS" ]; then
    CMD="$CMD --reps $REPS"
fi

echo "Running: $CMD"
eval $CMD

echo ""
echo "Finished: $(date)"
