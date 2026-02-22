#!/bin/bash
#SBATCH --job-name=kirby_c11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# C11: Allocation strategy comparison
#
# Usage:
#   sbatch tests/slurm_scripts/run_allocation_experiment.sh [mode] [dataset] [reps] [extra_flags]
#
# Examples:
#   sbatch tests/slurm_scripts/run_allocation_experiment.sh intermediary logd fast
#   sbatch tests/slurm_scripts/run_allocation_experiment.sh intermediary flab_expression_koenig mutation_fast "--delta"
#   sbatch tests/slurm_scripts/run_allocation_experiment.sh intermediary tcga_brca_pam50 fast

MODE=${1:-intermediary}
DATASET=${2:-logd}
REPS=${3:-fast}
EXTRA=${4:-}

echo "============================================================"
echo "C11: Allocation Strategy Comparison"
echo "  Mode:    $MODE"
echo "  Dataset: $DATASET"
echo "  Reps:    $REPS"
echo "  Extra:   ${EXTRA:-none}"
echo "============================================================"

python tests/experiments/complexity/c11_allocation_strategies.py \
    --mode $MODE --dataset $DATASET --reps $REPS --output tests/results $EXTRA

echo ""
echo "C11 COMPLETE"
