#!/bin/bash
#SBATCH --job-name=kirby_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Run the full per-dataset experiment suite for one dataset.
# Covers all four registered categories: complexity, similarity, noise, federated.
# OPIG task-specific experiments (opig_*.py) are standalone files — not in
# the per-dataset runner — and are NOT submitted here. Launch those
# separately if you want them.
#
# Usage (submit one dataset):
#   sbatch --export=DATASET=flab_thermo_tresanco tests/slurm_scripts/run_full_suite.sh
#
# Override which categories to run:
#   sbatch --export=DATASET=caco2,CATEGORIES="complexity similarity" \
#          tests/slurm_scripts/run_full_suite.sh
#
# Exits non-zero if any category's run_all fails. Per-category stdout is
# prefixed so you can grep the log by category.

set -u

if [[ -z "${DATASET:-}" ]]; then
    echo "ERROR: DATASET not set. Pass via --export=DATASET=<name>" >&2
    exit 2
fi

CATEGORIES="${CATEGORIES:-complexity similarity noise federated}"

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh

export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

echo "=================================================================="
echo "KIRBy full suite for dataset: $DATASET"
echo "Categories: $CATEGORIES"
echo "=================================================================="
echo

FAIL=0
for CAT in $CATEGORIES; do
    echo "=================================================================="
    echo "[$CAT] starting on $DATASET"
    echo "=================================================================="
    if ! time python3 tests/experiments/run_dataset.py "$CAT" "$DATASET"; then
        echo "[$CAT] FAILED on $DATASET (exit code $?)" >&2
        FAIL=1
    fi
    echo
done

echo "=================================================================="
if [[ $FAIL -eq 0 ]]; then
    echo "ALL categories completed successfully on $DATASET"
else
    echo "ONE OR MORE categories failed on $DATASET — see logs above"
fi
echo "=================================================================="
exit $FAIL
