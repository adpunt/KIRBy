#!/bin/bash
#SBATCH --job-name=kirby_c11c12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# C11 (allocation strategies) + C12 (adaptive pruning) for one dataset.
#
# Usage:
#   sbatch tests/slurm_scripts/run_c11_c12.sh [mode] [dataset] [reps] [extra_flags]
#
# Examples:
#   sbatch tests/slurm_scripts/run_c11_c12.sh intermediary logd fast
#   sbatch tests/slurm_scripts/run_c11_c12.sh intermediary tcga_brca_pam50 fast

MODE=${1:-intermediary}
DATASET=${2:-logd}
REPS=${3:-fast}
EXTRA=${4:-}

COMMON="--mode $MODE --dataset $DATASET --reps $REPS --output tests/results $EXTRA"

echo "============================================================"
echo "C11 + C12: Allocation Strategies + Adaptive Pruning"
echo "  Mode:    $MODE"
echo "  Dataset: $DATASET"
echo "  Reps:    $REPS"
echo "  Extra:   ${EXTRA:-none}"
echo "============================================================"

set -e

echo ""
echo ">>> C11: Allocation Strategies"
python tests/experiments/complexity/c11_allocation_strategies.py $COMMON

echo ""
echo ">>> C12: Adaptive Pruning Analysis"
python tests/experiments/complexity/c12_adaptive_pruning.py $COMMON

echo ""
echo ">>> C13: Greedy Step Size & Patience"
python tests/experiments/complexity/c13_greedy_step_size.py $COMMON

echo ""
echo "============================================================"
echo "C11 + C12 + C13 COMPLETE"
echo "Results in: tests/results"
echo "============================================================"
