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
# instead of 3, 2 representation counts instead of 4 -- 816 configurations
# against 8,160. It answers a smaller question. Everything it computes is reused
# by a later full pass, so a dataset can be upgraded after the deadline without
# redoing anything.
#
# It used to cut the downstream models from 5 to 3 as well, which took it to 432
# configurations and made two of the five models unreadable to every experiment.
# The MODELS block below names them back. Changed 2026-09-18.
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

# ── THE MODEL LIST IS ITS OWN DIAL, NOT PART OF THE SPEED ───────────────
#
# WHAT WAS WRONG. `--speed fast` shortens four lists at once, and one of them is
# the downstream model list: rf, lightgbm and dnn are kept, xgboost and svm are
# dropped. That happens at find_best_hybrid.py:8221-8232, inside `if models is
# None`. Every experiment reads a recipe per model name in DOWNSTREAM_MODELS
# (_shared.py:539), which is all five. A name the search never scored comes back
# as (None, None) from get_best_search_for_downstream, and the experiment then
# writes no rows for that model without saying so -- the comment at
# _shared.py:544-548 describes exactly this failure. So on the twelve LARGE
# datasets, two of the five downstream models were unreadable.
#
# THE FIX. Name the models explicitly. An explicit list skips the `models is
# None` branch entirely, so `--speed fast` still shortens the allocation,
# importance and representation-count lists and no longer touches the models.
# Per condition the grid goes from 432 configurations to 816, against 8,160 at
# full speed. Counted with find_best_hybrid._expected_config_count on a
# 10-representation pool and the eight FULL_BUDGETS entries.
#
# WHY THE FAST PASS IS STILL NOT WASTED. find_best_hybrid puts the RESOLVED
# model list in the search key (find_best_hybrid.py:8566), so a fast run naming
# all five writes the same `model_names` tuple a full-speed read asks for. The
# other three lists still differ, so the search pickle is still a miss -- but
# the per-configuration rows are keyed by result_db_identity, which carries the
# data, the fold split and the tier and NEVER the swept lists
# (find_best_hybrid.py:8608-8611). Every row a fast pass writes is read by a
# later full pass. Nothing here is redone.
#
# svm drops itself where it cannot run: _passes_max_samples applies to an
# explicit list too (find_best_hybrid.py:8236-8241), and svm carries a
# 10,000-row ceiling, which covers the five nucleotide sets and
# gb_human_enhancers_cohn. Naming it there costs nothing.
#
# ⛔ FULL-SPEED DATASETS PASS NOTHING. On a full run `models=None` already
# resolves to the same five names, so naming them changes nothing about what is
# computed -- but None and the tuple are different search keys, so every
# full-speed search pickle already on disk would stop matching. SMALL and MEDIUM
# send no flag, on purpose.
#
# ⛔ THE knn AND ridge PASSES MUST NOT GET THIS FLAG. At
# precompute_hybrids.py:682-693 an explicit --models wins over --with-knn and
# --with-ridge, so passing it there would drop knn from the pass s02 reads and
# ridge from the pass c16 reads. Below, the flag goes to the three
# predictive/similarity passes only. The distance passes fix their own list
# (DISTANCE_MODELS) and ignore it either way.
#
#   KIRBY_MODELS   comma-separated names, or the word 'default' to send no flag
#                  and take whatever the speed resolves to.
if [ "${KIRBY_MODELS:-}" = "default" ]; then
    MODELS=""
elif [ -n "${KIRBY_MODELS:-}" ]; then
    MODELS="${KIRBY_MODELS}"
elif [ "${SPEED}" = "fast" ]; then
    MODELS="rf,lightgbm,dnn,svm,xgboost"
else
    MODELS=""
fi

# A plain string, deliberately expanded unquoted at the call sites below. Model
# names hold no spaces, and an empty string expands to no argument at all --
# which an empty bash array does not, under `set -u` on bash 3.2.
MODEL_FLAG=""
if [ -n "${MODELS}" ]; then
    MODEL_FLAG="--models ${MODELS}"
fi

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

# ── THE FAST LIST IS OWNED BY dataset_policy.py, NOT BY THIS FILE ───────
# LARGE_DATASETS above is a copy kept so the class can be resolved before
# setup.sh has been sourced. The list the EXPERIMENTS read is
# dataset_policy.FAST_SPEED_DATASETS, and if the two ever disagree the
# precompute searches one grid while every experiment asks for another --
# which is exactly the state this repository was in until 2026-09-21, and it
# made every experiment on all twelve datasets refuse to start.
#
# So they are compared here, once setup.sh has given us the interpreter, and a
# disagreement stops the job. It does not cost a fit and it cannot be silent.
# KIRBY_LARGE_DATASETS deliberately skips the check: an override is a
# one-off by hand, and the experiments will not know about it.
if [ -z "${KIRBY_LARGE_DATASETS:-}" ]; then
    python - "$DS" "$CLASS" <<'EOF'
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'tests')
ds, cls = sys.argv[1], sys.argv[2]
try:
    from experiments.dataset_policy import FAST_SPEED_DATASETS
except Exception as e:
    print(f"  ⛔ could not read FAST_SPEED_DATASETS ({type(e).__name__}: {e}).")
    print("     Refusing to precompute: the speed this job would use cannot be")
    print("     checked against the speed the experiments will ask for.")
    sys.exit(2)
wants_fast = ds in FAST_SPEED_DATASETS
if wants_fast != (cls == 'LARGE'):
    print(f"  ⛔ {ds}: this script says class={cls} and dataset_policy says "
          f"{'fast' if wants_fast else 'full'}.")
    print("     One of the two lists has been edited without the other.")
    print("     Fix dataset_policy.FAST_SPEED_DATASETS and LARGE_DATASETS in")
    print("     this file so they name the same datasets, then resubmit.")
    sys.exit(3)
EOF
    _rc=$?
    if [ "$_rc" != "0" ]; then
        echo "  precompute aborted before any work (exit ${_rc})."
        exit "$_rc"
    fi
fi

echo "=========================================================="
echo " precompute  dataset=${DS}"
echo " class=${CLASS}  speed=${SPEED}  pruning=${PRUNING}"
echo " models=${MODELS:-<whatever the speed resolves to>}"
echo "   the knn, ridge and distance passes keep their own lists"
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

# ── NUCLEOTIDE DATASETS DO NOT NEED THE knn PASS ────────────────────────
# The knn pass exists for s02 and nothing else -- it scores the six-model list
# under the similarity objective so s02 can read a recipe chosen for
# five-nearest-neighbour prediction. On a nucleotide dataset s02 never runs:
# _shared.NUCLEOTIDE_SKIP_EXPERIMENTS skips ALL of s01-s13 there, on the stated
# ground that "the similarity premise is unreliable for sequence embeddings".
# So the pass computes searches that no experiment will ever open. Degenerate,
# which is one of the three reasons a cut is allowed.
#
# The other three passes STAY, and each for a reason:
#   sim    n13 sets POOL_SUBSET='predictive_and_similarity' and it is NOT in the
#          nucleotide skip list, so it does read the similarity searches.
#
#          ⛔ KEPT BY THE OWNER, 2026-09-18. DO NOT CUT IT. It was put to her
#          that the June skip list ("the similarity premise is unreliable for
#          sequence embeddings", 2026-06-10) predates n13's similarity arm
#          (2026-09-12) by three months, so n13's absence from that list is an
#          omission rather than a judgement -- and that the arm is the same
#          nearest-neighbour construct the list exists to exclude. Her answer
#          was to keep it. Do not re-raise this and do not drop `sim` from the
#          nucleotide passes.
#   dist   c03 reads the distance pool and complexity is not skipped at all --
#          all 20 of its experiments run on nucleotide.
#   ridge  c16 reads it, same reason.
case "$DS" in
    nt_*|gb_*)
        if has_pass knn; then
            PASSES="${PASSES// knn/}"; PASSES="${PASSES//knn /}"; PASSES="${PASSES//knn/}"
            echo "  ${DS} is a nucleotide dataset: dropping the knn pass."
            echo "  It is read by s02 alone, and every s01-s13 experiment is"
            echo "  skipped on this domain (_shared.NUCLEOTIDE_SKIP_EXPERIMENTS)."
        fi
        ;;
esac

# ── DATASETS THE SIMILARITY OBJECTIVE CANNOT BE SCORED ON ───────────────
# objective='similarity' predicts each sample's label from its five nearest
# neighbours in the space the features define. Three datasets cannot produce
# that number at all, for two different reasons, and on all three every
# representation comes back unscored on every fold and the search dies with
# "not one of the N representations could be scored".
#
#   tcga_brca_pam50 (4 classes), tz_her2_multiclass (3)
#       scoring a classification dataset uses AUROC, which is defined for 2.
#   tz_her2_binary
#       two classes, so AUROC is fine. Its FOLDS are the problem: they are cut
#       on edit-distance shells from the wildtype CDRH3, and a held-out block
#       can hold a single shell, leaving no other shell to find neighbours in.
#
# UNDEFINED, which is one of the three reasons a cut is allowed. This is NOT a
# claim that the objective would lose on any of them. Defining the multiclass
# score -- AUROC of each class against the rest, averaged -- would release the
# first two. tz_her2_binary's grouping is the leakage control and is not
# negotiable, so that one stays cut. Owner's call, 2026-09-20.
#
# Two passes are affected and they are affected differently:
#   sim  asks for predictive AND similarity. The predictive half is fine, so the
#        pass still runs, with similarity taken off it.
#   knn  asks for similarity ALONE, so it would write nothing. Dropped.
#
# THE LIST LIVES IN ONE PLACE: tests/experiments/dataset_policy.py. It is read
# from there rather than copied, because this directory already carries the scar
# of three scripts holding their own copies of the pass flags -- see the header
# of _precompute_passes.sh. precompute_hybrids.py drops the objective AGAIN from
# the dataset's own labels, so a dataset missing from that list costs a wasted
# pass here, never a wrong search.
SIM_OBJ="predictive,similarity"
python - "$DS" <<'EOF'
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'tests')
try:
    from experiments.dataset_policy import SIMILARITY_OBJECTIVE_SKIP_DATASETS as M
except Exception as e:
    print(f"  could not read the similarity skip list ({type(e).__name__}: {e})")
    sys.exit(2)
sys.exit(0 if sys.argv[1] in M else 1)
EOF
case $? in
    0)
        SIM_OBJ="predictive"
        if has_pass knn; then
            PASSES="${PASSES// knn/}"; PASSES="${PASSES//knn /}"; PASSES="${PASSES//knn/}"
        fi
        echo "  ${DS} cannot be scored under objective='similarity'."
        echo "  The sim pass runs PREDICTIVE ONLY, and the knn pass is dropped"
        echo "  -- it asks for similarity alone, so it would write nothing."
        echo "  The reason for this dataset is in dataset_policy.py; precompute"
        echo "  prints it in full when it drops the objective."
        ;;
    2)
        echo "  WARNING: could not check the skip list for ${DS}, so the sim"
        echo "  pass keeps the similarity objective. precompute_hybrids.py still"
        echo "  drops it from the labels if it does not apply."
        ;;
esac

# 1-2: what every experiment reads first.
has_pass sim  && run_one general_none          --objectives ${SIM_OBJ} ${MODEL_FLAG}
has_pass dist && run_one general_distance_none --distance

# 3-5: the augmented conditions. A dataset with no augmentation representations
# runs no condition here and returns in seconds.
has_pass sim  && run_one general_uncapped           --objectives ${SIM_OBJ} ${MODEL_FLAG}
has_pass sim  && run_one general_capped             --objectives ${SIM_OBJ} ${MODEL_FLAG}
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
