#!/bin/bash
#SBATCH --job-name=c02_verify_fresh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --mem=96G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Run c02 on a dataset whose precompute is fully populated in cache.
# Default: flab_aggregation_jain_hicrt (Y/Y/C search caches, balanced
# none/uncapped, small n => fast to recompute if cache misses anyway).
#
# Override the dataset with:
#   sbatch --export=DATASET=ginkgo_pr_cho tests/slurm_scripts/verify_c02_fresh_dataset.sh

DATASET="${DATASET:-flab_aggregation_jain_hicrt}"

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh

export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

echo "=================================================================="
echo "BEFORE RUN: cache state for $DATASET"
echo "=================================================================="
python3 tests/scripts/diagnose_dataset_cache.py "$DATASET"

echo
echo "=================================================================="
echo "RUNNING c02 on $DATASET"
echo "=================================================================="
time python3 tests/experiments/complexity/c02_complexity_grid.py \
    --dataset "$DATASET" \
    --mode full \
    --augment \
    --output "tests/results/c02_verify_fresh/"

echo
echo "=================================================================="
echo "AFTER RUN: cache state for $DATASET"
echo "=================================================================="
python3 tests/scripts/diagnose_dataset_cache.py "$DATASET"

echo
echo "=================================================================="
echo "KEY METRIC — look for [cache] lines above. Every search emits one."
echo "If hybrid_eval shows mostly HIT, the cache is working."
echo "If mostly MISS or STALE, we have a bug to chase."
echo "=================================================================="
