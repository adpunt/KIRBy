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
#   KIRBY_SPEED     force a speed. Unset: from the dataset's class below.
#   KIRBY_PRUNING   force a pruning mode. Unset: from the class.
#   KIRBY_SMALL_DATASETS / KIRBY_LARGE_DATASETS  override the class lists
#   KIRBY_PRUNING   calibrated (default) | no_pruning | default_pruning | aggressive_pruning
#   KIRBY_PASSES    space-separated subset of: sim dist knn ridge  (default: all four)

set -uo pipefail
DS=${1:?usage: sbatch run_precompute_dataset.sh <dataset>}

# ── THE THREE CLASSES ───────────────────────────────────────────────────
#
# Cost is driven by ROWS far more than by features. Calibrated against a real
# run -- caco2, 2,158 rows, 1,811 of 8,160 configurations in four hours, so
# about 18 h for the grid -- and scaled by rows^1.5 x features.
#
# ⛔ NOT BECAUSE OF svm. I claimed that on 2026-09-17 without measuring it.
# From the per-configuration seconds the engine stores on every row, over five
# datasets already computed: xgboost or lightgbm is the dearest model EVERY
# time (23.8% to 56.5%) and svm is 5.8% to 23.8%, the cheapest of the five on
# two of them. svm is already skipped above 10,000 rows
# (DEFAULT_MODELS['svm']['_max_samples']), which covers the five nucleotide and
# genomic sets; below that it is not worth banning and it wins herg_ki outright
# (FRAMEWORK_FINALIZATION 7.2).
#
#   SMALL   full grid, NO pruning. Finishes in about 1 to 9 hours, so there is
#           no reason to accept even the calibrated schedule's risk. Under
#           ~1,500 rows, or measured under 10 h.
#             tcga_brca_pam50 ~1 h   flab_thermo_tresanco ~3 h
#             humanness_therapeutic ~3 h   herg_ki ~9 h
#             plus every dataset under 700 rows: the five ginkgo sets, abcov_kd,
#             the three abagym sets, ddg_combined, cdr_flexibility
#
#   MEDIUM  full grid, CALIBRATED pruning. 10 to 40 hours, which four 12-hour
#           links cover. Pruning is measured to lose no winner on any of the
#           five recorded datasets while saving 13% to 44%.
#             caco2 ~18 h   buchwald_hartwig ~36 h   tcga_luad_vs_lusc
#             and everything whose size is unknown, which is the safe default
#
#   LARGE   FAST tier. Past what four links can reach at full, or past 10,000
#           rows where no full grid is sensible.
#             nt_H3K27me3 33,000 rows   nt_H3K9me3 28,288
#             gb_human_enhancers_cohn 27,791   nt_H3K9ac 24,278
#             nt_H3K4me3 18,244   qm9 ~103 h   flab_expression_koenig ~64 h
#             logd ~62 h   nt_promoter_tata ~39 h   herg 7,772 rows
#             tz_her2_binary and tz_her2_multiclass 5,000 rows each
#
# ⛔ FAST IS NOT FREE: 3 allocation methods instead of 5, one importance method
# instead of 3, 3 models instead of 5, 2 representation counts instead of 4 --
# 432 configurations against 8,160. It answers a smaller question. Everything it
# computes is reused by a later full pass, so a dataset can be upgraded after
# the deadline without redoing anything.
#
# Move a dataset by editing one of these three lines. KIRBY_SPEED and
# KIRBY_PRUNING override the class entirely.
SMALL_DATASETS="${KIRBY_SMALL_DATASETS:-tcga_brca_pam50 flab_thermo_tresanco humanness_therapeutic herg_ki ginkgo_acsins ginkgo_colloidal ginkgo_hic ginkgo_titer ginkgo_pr_cho abcov_kd abagym_Ang2 abagym_VEGF abagym_EGFR ddg_combined cdr_flexibility}"
LARGE_DATASETS="${KIRBY_LARGE_DATASETS:-nt_H3K27me3 nt_H3K9me3 nt_H3K9ac nt_H3K4me3 gb_human_enhancers_cohn nt_promoter_tata qm9 herg logd flab_expression_koenig tz_her2_binary tz_her2_multiclass}"

if [[ " ${LARGE_DATASETS} " == *" ${DS} "* ]]; then
    CLASS="LARGE";  DEF_SPEED="fast"; DEF_PRUNING="calibrated"
elif [[ " ${SMALL_DATASETS} " == *" ${DS} "* ]]; then
    CLASS="SMALL";  DEF_SPEED="full"; DEF_PRUNING="no_pruning"
else
    CLASS="MEDIUM"; DEF_SPEED="full"; DEF_PRUNING="calibrated"
fi
SPEED="${KIRBY_SPEED:-$DEF_SPEED}"
PRUNING="${KIRBY_PRUNING:-$DEF_PRUNING}"
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
echo " class=${CLASS}  speed=${SPEED}  pruning=${PRUNING}"
echo " passes=${PASSES}"
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
