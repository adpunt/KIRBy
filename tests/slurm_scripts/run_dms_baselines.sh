#!/bin/bash
#SBATCH --job-name=dms_baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# =============================================================================
# DMS PER-REP BASELINES: 9 datasets × 20 reps × 3 models
# =============================================================================
#
# Evaluates each DMS rep individually to build the BOM pruning matrix.
# Output: tests/results/dms_baselines/dms_baselines_<dataset>_<timestamp>.json
#
# Usage:
#   sbatch run_dms_baselines.sh              # all 9 datasets
#   sbatch run_dms_baselines.sh single flab_expression_koenig  # one dataset

MODE=${1:-all}
DATASET=${2:-}

echo "============================================================"
echo "DMS Per-Rep Baseline Survey"
echo "  Mode:    $MODE"
echo "  Dataset: ${DATASET:-all}"
echo "  Started: $(date)"
echo "============================================================"

if [ "$MODE" = "single" ] && [ -n "$DATASET" ]; then
    python tests/run_dms_baselines.py "$DATASET" --folds 5
else
    python tests/run_dms_baselines.py all --folds 5
fi

echo ""
echo "============================================================"
echo "DMS Baselines COMPLETE: $(date)"
echo "============================================================"
