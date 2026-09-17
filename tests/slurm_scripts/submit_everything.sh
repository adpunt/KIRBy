#!/bin/bash
# Queue the whole remaining PhD: every dataset's precompute, then every
# experiment that depends on it. One command, one dry-run, no babysitting.
#
# ─────────────────────────────────────────────────────────────────────────
# WHAT IT QUEUES, PER DATASET
#
#   precompute   a CHAIN of N jobs (default 4), each 12 hours, each starting
#                when the previous one ends. Link 1 does as much as it can;
#                link 2 resumes exactly where it stopped; a link with nothing
#                left exits in minutes.
#
#   experiments  one job per category (complexity, similarity, noise), each
#                depending on the LAST link of that dataset's chain.
#
# WHY A CHAIN AND NOT A LONGER WALL. No dataset finishes the 8,160-configuration
# grid in 12 hours -- measured on the f33 record, scaled to eight budgets: logd
# 79 h unpruned and 44 h with calibrated pruning, herg_ki 42/35,
# tcga_brca_pam50 42/30, flab_thermo_tresanco 32/24, humanness_therapeutic
# 15/13. A longer wall would cover it and would also stop the job ever being
# scheduled: a 2-day request cannot backfill into a gap a 12-hour request fits.
# Chaining gets the same total compute out of slots that actually start.
#
# WHY afterany AND NOT afterok. A precompute link that hits its wall is killed,
# which SLURM reports as failure. That is the NORMAL case here, not an error --
# its rows are on disk. afterok would stall the whole chain on the first wall.
#
# ─────────────────────────────────────────────────────────────────────────
# ⛔ THE EXPERIMENT JOBS READ YOUR CODE WHEN THEY START, NOT NOW.
#
# The experiment jobs are submitted WITHOUT naming experiments, so each one runs
# whatever SINGLE_DATASET_EXPERIMENTS holds in noise_helpers.py /
# complexity_helpers.py / similarity_helpers.py at the moment it starts. Add an
# experiment to that list, or delete one, and every queued job picks the change
# up. NO RESUBMISSION. The same is true of the experiment files themselves --
# they are read from disk at run time.
#
# What you must NOT change while these are queued:
#   - the dataset rosters, which decide what was queued in the first place
#   - anything under src/kirby/ that alters a stored result's identity (the
#     metric, the fold split, the representation pool), because that orphans
#     every row the precompute already wrote
#
# ─────────────────────────────────────────────────────────────────────────
# USAGE
#   bash tests/slurm_scripts/submit_everything.sh --dry-run
#   bash tests/slurm_scripts/submit_everything.sh --domain all
#   bash tests/slurm_scripts/submit_everything.sh --datasets "herg_ki caco2"
#   bash tests/slurm_scripts/submit_everything.sh --precompute-only
#   bash tests/slurm_scripts/submit_everything.sh --experiments-only
#
#   --links N            chain depth, default 4 (48 h of wall per dataset)
#   --speed fast|full    default full
#   --pruning MODE       default calibrated
#   --skip "A B"         datasets to leave out (e.g. ones you did by hand)
#   --no-cold-refusal    let an experiment build a missing search instead of
#                        failing fast. Default is to fail fast.
#
# Run the checks in an srun first (they import rdkit and stat every
# representation file):
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh sim --domain all --checks-only
set -uo pipefail

REPO="${KIRBY_REPO:-/data/stat-ecr/scat9264/KIRBy}"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
PARTITION="${PARTITION:-short,medium,long}"
ACCOUNT="${ACCOUNT:-}"
WALL="12:00:00"
LINKS=4
SPEED="full"
PRUNING="calibrated"
DRY_RUN=false
DOMAIN="all"
EXPLICIT=""
SKIP=""
DO_PRE=true
DO_EXP=true
COLD_REFUSAL=1
CATEGORIES="complexity similarity noise"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)          DRY_RUN=true; shift ;;
        --domain)           DOMAIN=$2; shift 2 ;;
        --datasets)         EXPLICIT=$2; shift 2 ;;
        --skip)             SKIP=$2; shift 2 ;;
        --links)            LINKS=$2; shift 2 ;;
        --speed)            SPEED=$2; shift 2 ;;
        --pruning)          PRUNING=$2; shift 2 ;;
        --wall)             WALL=$2; shift 2 ;;
        --partition)        PARTITION=$2; shift 2 ;;
        --account)          ACCOUNT=$2; shift 2 ;;
        --categories)       CATEGORIES=$2; shift 2 ;;
        --precompute-only)  DO_EXP=false; shift ;;
        --experiments-only) DO_PRE=false; shift ;;
        --no-cold-refusal)  COLD_REFUSAL=0; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── account ─────────────────────────────────────────────────────────────
if [ -z "$ACCOUNT" ]; then
    ACCOUNT=$(bash "$REPO/tests/slurm_scripts/where_to_submit.sh" --emit 2>/dev/null | awk '{print $1}')
fi
[ -n "$ACCOUNT" ] || { echo "Could not resolve an account. Pass --account."; exit 1; }

# ── which datasets ──────────────────────────────────────────────────────
if [ -n "$EXPLICIT" ]; then
    DATASETS="$EXPLICIT"
else
    DATASETS=$(cd "$REPO" && python - <<'PY'
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'tests')
from experiments.complexity.complexity_helpers import DATASET_REPS
# chembl_qsar has no load_data path; davis is parked. Both are excluded in
# submit_precompute_extra_passes.sh for the same stated reasons.
print(' '.join(sorted(set(DATASET_REPS) - {'chembl_qsar', 'davis'})))
PY
)
fi
for S in $SKIP; do
    DATASETS=$(echo "$DATASETS" | tr ' ' '\n' | grep -vx "$S" | tr '\n' ' ')
done
DATASETS=$(echo "$DATASETS" | xargs)
N_DS=$(echo "$DATASETS" | wc -w)

# ── the plan ────────────────────────────────────────────────────────────
N_PRE=$([ "$DO_PRE" = true ] && echo $((N_DS * LINKS)) || echo 0)
N_EXP=$([ "$DO_EXP" = true ] && echo $((N_DS * $(echo "$CATEGORIES" | wc -w))) || echo 0)
echo ""
echo "=========================================================="
echo " datasets       $N_DS"
echo " precompute     $LINKS links x $N_DS = $N_PRE jobs, ${WALL} each"
echo " experiments    $(echo "$CATEGORIES" | wc -w) categories x $N_DS = $N_EXP jobs, each after that dataset's last link"
echo " TOTAL          $((N_PRE + N_EXP)) jobs"
echo ""
echo " account        $ACCOUNT"
echo " partition      $PARTITION"
echo " speed          $SPEED       pruning  $PRUNING"
echo " cold refusal   $([ "$COLD_REFUSAL" = 1 ] && echo 'ON — an experiment with no precompute fails in minutes' || echo 'off — it will build the search itself')"
echo "=========================================================="
echo " $DATASETS" | fold -s -w 74 | sed 's/^/   /'
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "--dry-run: nothing submitted."
    echo ""
    echo "One dataset's chain would look like:"
    D=$(echo "$DATASETS" | awk '{print $1}')
    echo "   sbatch --account=$ACCOUNT --partition=$PARTITION --time=$WALL \\"
    echo "          --job-name=pc_${D}_1 run_precompute_dataset.sh $D"
    echo "   sbatch ... --dependency=afterany:<link1> --job-name=pc_${D}_2 ... (x$((LINKS-1)))"
    for C in $CATEGORIES; do
        echo "   sbatch ... --dependency=afterany:<link$LINKS> --job-name=x_${C}_${D} submit_dataset.sh $C $D"
    done
    exit 0
fi

read -r -p "Submit $((N_PRE + N_EXP)) jobs? [y/N] " REPLY
[[ "$REPLY" == "y" || "$REPLY" == "Y" ]] || { echo "Aborted."; exit 0; }

# ── submit ──────────────────────────────────────────────────────────────
N_OK=0
for DS in $DATASETS; do
    MEM=64G
    case "$DS" in nt_*|gb_*|tcga_*|kiba|bindingdb_kd|qm9) MEM=128G ;; esac

    LAST=""
    if [ "$DO_PRE" = true ]; then
        for L in $(seq 1 "$LINKS"); do
            DEP=""
            [ -n "$LAST" ] && DEP="--dependency=afterany:${LAST}"
            JID=$(sbatch --parsable \
                --account="$ACCOUNT" --partition="$PARTITION" --time="$WALL" \
                --mem="$MEM" --cpus-per-task=8 \
                --job-name="pc_${DS}_${L}" \
                --mail-user="$EMAIL" --mail-type=FAIL \
                --export="ALL,KIRBY_SPEED=${SPEED},KIRBY_PRUNING=${PRUNING}" \
                $DEP \
                "$REPO/tests/slurm_scripts/run_precompute_dataset.sh" "$DS" 2>/dev/null)
            [ -n "$JID" ] || { echo "  FAILED pc_${DS}_${L}"; continue; }
            LAST="$JID"; N_OK=$((N_OK+1))
        done
    fi

    if [ "$DO_EXP" = true ]; then
        for C in $CATEGORIES; do
            DEP=""
            [ -n "$LAST" ] && DEP="--dependency=afterany:${LAST}"
            # The CALIBRATED schedule, not aggressive_pruning. The experiment
            # jobs must prune the same way the precompute did, or they ask for
            # configurations it deleted. aggressive_pruning is the April
            # setting: it drops at budgets 25 and 50 as well as 100, and on the
            # f33 record it deletes the two allocation methods that win most of
            # tcga_brca_pam50's conditions, damaging 17 of its 25.
            EXPORTS="ALL,PRUNING_MODE=${PRUNING}"
            [ "$COLD_REFUSAL" = 1 ] && EXPORTS="${EXPORTS},KIRBY_NO_COLD_SEARCH=1"
            JID=$(sbatch --parsable \
                --account="$ACCOUNT" --partition="$PARTITION" \
                --job-name="x_${C}_${DS}" \
                --mail-user="$EMAIL" --mail-type=FAIL \
                --export="$EXPORTS" \
                $DEP \
                "$REPO/tests/slurm_scripts/submit_dataset.sh" "$C" "$DS" 2>/dev/null)
            [ -n "$JID" ] || { echo "  FAILED x_${C}_${DS}"; continue; }
            N_OK=$((N_OK+1))
        done
    fi
    echo "  $DS: queued"
done

echo ""
echo "Submitted $N_OK jobs."
echo ""
echo "Watch:    squeue -u \$USER -o '%.10i %.26j %.20P %.2t %.10M %.10L %R' | head -40"
echo "Progress: python tests/precompute_coverage.py --trial     (in an srun)"
echo ""
echo "⛔ The experiment jobs read SINGLE_DATASET_EXPERIMENTS and every experiment"
echo "   file WHEN THEY START. Add or remove an experiment and the queued jobs"
echo "   pick it up. Do NOT change anything that alters a stored result's"
echo "   identity — the metric, the fold split, the representation pool —"
echo "   or the rows already written are orphaned."
