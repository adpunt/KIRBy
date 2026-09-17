#!/bin/bash
#SBATCH --job-name=pcall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#
# EVERY precompute pass for ONE dataset, in one job, resumable.
#
# WHY ONE JOB PER DATASET RATHER THAN ONE PER (CONDITION, PASS).
# The full grid is 8,160 configurations and no dataset finishes it in 12 hours:
# measured on the f33 record and scaled to eight budgets, logd is 79 hours
# unpruned and 44 with calibrated pruning, herg_ki 42/35, tcga_brca_pam50 42/30,
# flab_thermo_tresanco 32/24, humanness_therapeutic 15/13. Splitting by
# condition does not fix that -- it just makes more jobs that each run out.
#
# What fixes it is that a search RESUMES. Rows flush to result_db every 25
# configurations (find_best_hybrid.py:_DB_FLUSH_INTERVAL) and the next run skips
# every configuration already stored (:7144 and :7153). So this script walks the
# whole list for one dataset, gets as far as the wall allows, and the NEXT link
# in the chain picks up exactly where it stopped. A link with nothing left to do
# costs minutes, not hours, which is why over-chaining is cheap insurance and
# under-chaining is not.
#
# ORDER. Cheapest and most-read first, so a chain that never completes still
# leaves the most useful things done:
#   1. general_none          predictive + similarity   every experiment reads it
#   2. general_distance_none the distance lineage       c03, s04-s13
#   3. general_uncapped      predictive + similarity
#   4. general_capped        predictive + similarity
#   5. general_distance_uncapped
#   6. knn passes            s02 only
#   7. ridge passes          c16 only
# A dataset with no augmentation representations skips 3, 4 and 5 in seconds.
#
# SUBMIT IT THROUGH submit_everything.sh, which builds the chain and the
# experiment dependencies. By hand, for one dataset:
#   sbatch --account=stat-ecr --partition=short,medium,long \
#          tests/slurm_scripts/run_precompute_dataset.sh caco2
#
# ARGUMENTS
#   $1              dataset name (required)
#   KIRBY_SPEED     full (default) | fast
#   KIRBY_PRUNING   calibrated (default) | no_pruning | default_pruning | aggressive_pruning
#   KIRBY_PASSES    space-separated subset of: sim dist knn ridge  (default: all four)

set -uo pipefail
DS=${1:?usage: sbatch run_precompute_dataset.sh <dataset>}
SPEED="${KIRBY_SPEED:-full}"
PRUNING="${KIRBY_PRUNING:-calibrated}"
PASSES="${KIRBY_PASSES:-sim dist knn ridge}"

cd /data/stat-ecr/scat9264/KIRBy || exit 2

# The jobs do NOT git pull. Hundreds of them start over days on one shared
# checkout; pulling would race, and a mid-flight code change is worse than a
# known one. Pull once by hand before submitting. The commit is recorded here so
# a result can be traced to the code that made it.
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

export KIRBY_SKIP_INSTALLS=1
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

echo "=========================================================="
echo " precompute  dataset=${DS}"
echo " speed=${SPEED}  pruning=${PRUNING}  passes=${PASSES}"
echo " branch=${BRANCH} (${COMMIT})   job=${SLURM_JOB_ID:-none}"
echo " cache=${KIRBY_CACHE_DIR}"
echo " start=$(date)"
echo "=========================================================="

run_one() {   # condition, then the pass's own flags
    local cond="$1"; shift
    echo ""
    echo "---- ${DS} / ${cond} / $* ----"
    echo "---- $(date) ----"
    python -u tests/precompute_hybrids.py "${DS}" \
        --condition "${cond}" --speed "${SPEED}" \
        --pruning-mode "${PRUNING}" "$@"
    echo "---- exit=$? ----"
}

has_pass() { [[ " ${PASSES} " == *" $1 "* ]]; }

# 1-2: what every experiment reads first.
has_pass sim  && run_one general_none          --objectives predictive,similarity
has_pass dist && run_one general_distance_none --distance

# 3-5: the augmented conditions. A dataset with no augmentation representations
# runs no condition here and returns in seconds.
has_pass sim  && run_one general_uncapped           --objectives predictive,similarity
has_pass sim  && run_one general_capped             --objectives predictive,similarity
has_pass dist && run_one general_distance_uncapped  --distance

# 6-7: the two single-experiment passes, last because they block only s02 and c16.
for c in general_none general_uncapped general_capped; do
    has_pass knn && run_one "$c" --with-knn --objectives similarity
done
for c in general_none general_uncapped general_capped; do
    has_pass ridge && run_one "$c" --with-ridge
done

echo ""
echo "=========================================================="
echo " end=$(date)"
python -u tests/precompute_coverage.py "${DS}" 2>&1 | tail -20
echo "=========================================================="
# Always exit 0: a link that ran out of wall is not a failure, and the chain
# uses afterany so the next link starts regardless. Real failures are visible
# in the per-pass exit codes above and in the coverage table.
exit 0
