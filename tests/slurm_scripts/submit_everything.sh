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
#   --speed fast|full    force one speed. Default: from the dataset class in
#                        run_precompute_dataset.sh (SMALL / MEDIUM / LARGE)
#   --pruning MODE       default calibrated
#   --skip "A B"         datasets to leave out (e.g. ones you did by hand)
#   --no-cold-refusal    let an experiment build a missing search instead of
#                        failing fast. Default is to fail fast.
#   --begin TIME         no job may START before this. It still enters the queue
#                        now and still accrues its place; it simply is not
#                        eligible until then, and after that it waits its turn
#                        like anything else. Slurm's own format:
#                          --begin 2026-09-26T19:00:00
#                          --begin now+7days
#                        Use it when the code these jobs will read is going to
#                        change before then -- they read it WHEN THEY START.
#
# EXISTING JOBS ARE NEVER CANCELLED. A dataset that already has jobs queued
# keeps them, and its new links wait for them -- so no queue position is lost
# and two processes never write one dataset's database at the same time.
#
# Run the checks in an srun first (they import rdkit and stat every
# representation file):
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh sim --domain all --checks-only
set -uo pipefail

REPO="${KIRBY_REPO:-/data/stat-ecr/scat9264/KIRBy}"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
PARTITION="${PARTITION:-short,medium,long}"
ACCOUNT="${ACCOUNT:-}"
CLUSTERS=""       # empty = the login node's cluster (arc); "htc" sends every job there
WALL="12:00:00"
LINKS=4
SPEED=""          # empty = let run_precompute_dataset.sh choose per dataset
PRUNING="calibrated"
DRY_RUN=false
DOMAIN="all"
EXPLICIT=""
SKIP=""
DO_PRE=true
DO_EXP=true
BEGIN=""          # --begin=<time> passed to every sbatch, or empty
COLD_REFUSAL=1
CATEGORIES="complexity similarity noise"

# federated is a fourth category of run_dataset.py, but only these three
# datasets have silo data. Mirrors FEDERATED_DATASETS in submit_all_datasets.sh.
FEDERATED_DATASETS="lhasa_herg caco2 logd"

# The opig suite is 13 standalone scripts, not a per-dataset category. Each line
# is: <name>|<memory>|<script + arguments>|<datasets it reads>
# The datasets decide what it waits for -- an opig job starts when the last
# precompute link of every dataset it names has ended.
#
# Five of these were in no submitter before today: cdr_flexibility,
# cgr_comparison, kingsmith, scaffold_hybrid, scaffold_similarity.
#
# shared_vectors_census names no dataset and activity_cliff_qsar reads
# chembl_qsar, which builds its own representations, so neither waits for
# anything.
# ⛔ CUT 2026-09-23, both degenerate at the fold level, not underpowered searches.
#
# cdr_flexibility — 216 CDRH3 sequences, 41 of them rigid, and 11 IGHV families
#   that collapse to 5 held-out groups once the seven families under 10
#   antibodies are pooled. Matthews correlation is 0.0000 on 11 of the 25
#   (representation, model) rows: those models call the majority class for every
#   loop and only the ranking carries signal. No search size fixes 41 positives.
#   submit_opig_experiments.sh had already dropped it on the same grounds.
#
# kingsmith — leave-one-component-out gives 4 held-out ligands and 3 held-out
#   bases. Measured on the ligand axis: fold spreads of 8.1 to 15.9 RMSE in
#   yield units against hybrid-minus-baseline deltas of 1.2 to 7.2. The delta is
#   inside the spread on every model, and with 3 or 4 folds no grid makes it
#   measurable. The additive axis has 23 components and would be readable; the
#   file runs all four axes together, so it goes as a whole.
OPIG_JOBS='
humanness|64G|opig_humanness.py --datasets humanness_therapeutic PM|humanness_therapeutic
trastuzumab|64G|opig_trastuzumab_enrichment.py PM|tz_her2_binary
ginkgo|128G|opig_ginkgo_developability.py PM|ginkgo_acsins ginkgo_colloidal ginkgo_hic ginkgo_pr_cho ginkgo_titer
graphinity|128G|opig_graphinity_ddg.py --max-foldx 5000 PM|ddg_combined
qsar|64G|opig_activity_cliff_qsar.py PM|
blueprint_transfer|64G|opig_blueprint_transfer.py --source logd --target herg_ki --mode full PM|logd herg_ki
cgr_comparison|64G|opig_cgr_comparison.py PM|bh_doyle_plates buchwald_hartwig suzuki
shared_vectors_census|64G|opig_shared_vectors_census.py --check-freshness 200|
shared_vectors_remedy|64G|opig_shared_vectors_remedy.py|herg_ki
scaffold_hybrid|64G|opig_scaffold_hybrid.py|flab_binding_adams flab_binding_koenig flab_expression_adams flab_expression_koenig
scaffold_similarity|64G|opig_scaffold_similarity.py|ddg_combined flab_binding_adams flab_binding_koenig flab_expression_adams flab_expression_koenig
'
DO_OPIG=true
DO_FED=true

# Which precompute link each dataset ended on, so the opig jobs can wait for the
# right ones. A newline table and a lookup, not an associative array: this has
# to dry-run on a mac, where bash is 3.2 and `declare -A` does not exist.
LINK_TABLE=""
last_link() { printf '%s\n' "$LINK_TABLE" | awk -v d="$1" '$1==d {print $2; exit}'; }

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
        --clusters)         CLUSTERS=$2; shift 2 ;;
        --categories)       CATEGORIES=$2; shift 2 ;;
        --begin)            BEGIN=$2; shift 2 ;;
        --precompute-only)  DO_EXP=false; shift ;;
        --experiments-only) DO_PRE=false; shift ;;
        --no-cold-refusal)  COLD_REFUSAL=0; shift ;;
        --no-opig)          DO_OPIG=false; shift ;;
        --no-federated)     DO_FED=false; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Empty expands to no argument at all under `set -u` on bash 3.2, which an
# empty array would not. Same reason as MODEL_FLAG in run_precompute_dataset.sh.
BEGIN_FLAG=""
[ -n "$BEGIN" ] && BEGIN_FLAG="--begin=${BEGIN}"

# ── account ─────────────────────────────────────────────────────────────
# ── cluster ─────────────────────────────────────────────────────────────
# --clusters htc sends every job to the htc cluster. The login node picks the
# cluster otherwise, and --partition cannot cross it. No --account is passed
# on htc: where_to_submit.sh reads arc's associations only, and the standing
# rule is to let SLURM resolve the default (project_arc_vs_htc_cluster).
# SLURM dependencies cannot cross clusters, so a dataset with precompute
# already queued on the OTHER cluster is skipped below, never double-queued.
CL_FLAG=""; SQ_FLAG=""
if [ -n "$CLUSTERS" ]; then
    CL_FLAG="--clusters=${CLUSTERS}"; SQ_FLAG="-M ${CLUSTERS}"
fi
ACCT_FLAG=""
if [ -z "$CLUSTERS" ]; then
    if [ -z "$ACCOUNT" ]; then
        ACCOUNT=$(bash "$REPO/tests/slurm_scripts/where_to_submit.sh" --emit 2>/dev/null | awk '{print $1}')
    fi
    [ -n "$ACCOUNT" ] || { echo "Could not resolve an account. Pass --account."; exit 1; }
fi
[ -n "$ACCOUNT" ] && ACCT_FLAG="--account=${ACCOUNT}"

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
N_FED=0
if [ "$DO_EXP" = true ] && [ "$DO_FED" = true ]; then
    for D in $DATASETS; do
        [[ " $FEDERATED_DATASETS " == *" $D "* ]] && N_FED=$((N_FED+1))
    done
fi
N_OPIG=$([ "$DO_OPIG" = true ] && printf '%s\n' "$OPIG_JOBS" | grep -c '|' || echo 0)
echo ""
echo "=========================================================="
echo " datasets       $N_DS"
echo " precompute     $LINKS links x $N_DS = $N_PRE jobs, ${WALL} each"
echo " experiments    $(echo "$CATEGORIES" | wc -w) categories x $N_DS = $N_EXP jobs, each after that dataset's last link"
echo " federated      $N_FED jobs (lhasa_herg, caco2, logd only)"
echo " opig           $N_OPIG jobs, each after the datasets it reads"
echo " TOTAL          $((N_PRE + N_EXP + N_FED + N_OPIG)) jobs"
echo ""
echo " begin          ${BEGIN:-as soon as the queue allows}"
echo " account        ${ACCOUNT:-<SLURM default>}"
echo " cluster        ${CLUSTERS:-login node default (arc)}"
echo " partition      $PARTITION"
echo " speed          ${SPEED:-per dataset class: 15 SMALL full+unpruned, 14 MEDIUM full+pruned, 12 LARGE fast}"
echo " pruning        $PRUNING"
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

read -r -p "Submit $((N_PRE + N_EXP + N_FED + N_OPIG)) jobs? [y/N] " REPLY
[[ "$REPLY" == "y" || "$REPLY" == "Y" ]] || { echo "Aborted."; exit 0; }

# ── submit ──────────────────────────────────────────────────────────────
N_OK=0
for DS in $DATASETS; do
    MEM=64G
    case "$DS" in nt_*|gb_*|tcga_*|kiba|bindingdb_kd|qm9) MEM=128G ;; esac

    # ── DO NOT STRAND AN EXISTING JOB ───────────────────────────────────
    # Jobs already queued for this dataset carry days of accrued age -- worth
    # 235 to 344 priority points on 2026-09-18, against jobs getting slots at
    # 1088 to 1541. Cancelling them to resubmit throws that away.
    #
    # So they are KEPT, and this dataset's first new link waits for them. Two
    # processes on one dataset both write result_db/<dataset>.db; SQLite waits
    # 120 s for the other writer and then the search dies AFTER paying for its
    # baselines and its per-fold scores (result_cache.py:187-200). Depending on
    # them costs nothing -- the new link would have waited in the queue anyway --
    # and removes that entirely.
    #
    # Matches all three namings in use: the old pc_<dataset>_general_..., this
    # script's own pc_<dataset>_<link>, and the hand-queued pc_<dataset>_x<n>.
    # It matched only the first until 2026-09-24, so a second submission for a
    # dataset whose chain was already queued found nothing, waited for nothing,
    # and two precompute processes wrote one result_db at once. Anchored at the
    # end, so pc_herg_ still cannot match pc_herg_ki_1.
    EXISTING=$(squeue -u "$USER" -h $SQ_FLAG -o "%i %j" 2>/dev/null \
               | awk -v d="$DS" '$2 ~ "^pc_" d "_(general_|[0-9]+$|x[0-9]+$)" {printf "%s:", $1}' \
               | sed 's/:$//')
    # Precompute for this dataset on ANY cluster. Anything here that is not in
    # EXISTING is on the other cluster, which a dependency cannot reach.
    ANYWHERE=$(squeue -u "$USER" -h -M all -o "%i %j" 2>/dev/null \
               | awk -v d="$DS" '$2 ~ "^pc_" d "_(general_|[0-9]+$|x[0-9]+$)" {print $1}' | sort -u)
    OTHER=$(comm -23 <(printf '%s\n' $ANYWHERE | sort -u) <(printf '%s\n' $(echo "$EXISTING" | tr ':' ' ') | sort -u) | grep -v '^$')
    if [ -n "$OTHER" ] && [ "$DO_PRE" = true ]; then
        echo "  $DS: SKIPPED -- precompute already queued on the other cluster ($(echo $OTHER | tr '\n' ' ')). Two chains would write one database at once."
        continue
    fi
    if [ -n "$EXISTING" ]; then
        echo "  $DS: $(echo "$EXISTING" | tr ':' '\n' | wc -l | tr -d ' ') job(s) already queued — new links wait for them"
    fi

    LAST=""
    if [ "$DO_PRE" = true ]; then
        for L in $(seq 1 "$LINKS"); do
            DEP=""
            if [ -n "$LAST" ]; then
                DEP="--dependency=afterany:${LAST}"
            elif [ -n "$EXISTING" ]; then
                DEP="--dependency=afterany:${EXISTING}"
            fi
            JID=$(sbatch --parsable $BEGIN_FLAG $CL_FLAG $ACCT_FLAG \
                --partition="$PARTITION" --time="$WALL" \
                --mem="$MEM" --cpus-per-task=8 \
                --job-name="pc_${DS}_${L}" \
                --mail-user="$EMAIL" --mail-type=FAIL \
                --export="ALL${SPEED:+,KIRBY_SPEED=${SPEED}},KIRBY_PRUNING=${PRUNING}" \
                $DEP \
                "$REPO/tests/slurm_scripts/run_precompute_dataset.sh" "$DS" 2>/dev/null)
            JID="${JID%%;*}"
            [ -n "$JID" ] || { echo "  FAILED pc_${DS}_${L}"; continue; }
            LAST="$JID"; N_OK=$((N_OK+1))
        done
    fi
    # Remember what this dataset ended on -- the opig jobs below wait on it.
    # When no chain was submitted (--experiments-only) fall back to whatever was
    # already queued, so an opig job still waits rather than racing it.
    LINK_TABLE="${LINK_TABLE}${DS} ${LAST:-$EXISTING}
"

    if [ "$DO_EXP" = true ]; then
        for C in $CATEGORIES; do
            DEP=""
            if [ -n "$LAST" ]; then
                DEP="--dependency=afterany:${LAST}"
            elif [ -n "$EXISTING" ]; then
                DEP="--dependency=afterany:${EXISTING}"
            fi
            # The CALIBRATED schedule, not aggressive_pruning. The experiment
            # jobs must prune the same way the precompute did, or they ask for
            # configurations it deleted. aggressive_pruning is the April
            # setting: it drops at budgets 25 and 50 as well as 100, and on the
            # f33 record it deletes the two allocation methods that win most of
            # tcga_brca_pam50's conditions, damaging 17 of its 25.
            EXPORTS="ALL,PRUNING_MODE=${PRUNING}"
            [ "$COLD_REFUSAL" = 1 ] && EXPORTS="${EXPORTS},KIRBY_NO_COLD_SEARCH=1"
            JID=$(sbatch --parsable $BEGIN_FLAG $CL_FLAG $ACCT_FLAG \
                --partition="$PARTITION" \
                --job-name="x_${C}_${DS}" \
                --mail-user="$EMAIL" --mail-type=FAIL \
                --export="$EXPORTS" \
                $DEP \
                "$REPO/tests/slurm_scripts/submit_dataset.sh" "$C" "$DS" 2>/dev/null)
            [ -n "$JID" ] || { echo "  FAILED x_${C}_${DS}"; continue; }
            N_OK=$((N_OK+1))
        done

        # federated is a fourth category of the same runner, but only the three
        # datasets with silo data have anything to run.
        if [ "$DO_FED" = true ] && [[ " $FEDERATED_DATASETS " == *" $DS "* ]]; then
            DEP=""
            if [ -n "$LAST" ]; then DEP="--dependency=afterany:${LAST}"
            elif [ -n "$EXISTING" ]; then DEP="--dependency=afterany:${EXISTING}"; fi
            EXPORTS="ALL,PRUNING_MODE=${PRUNING}"
            [ "$COLD_REFUSAL" = 1 ] && EXPORTS="${EXPORTS},KIRBY_NO_COLD_SEARCH=1"
            JID=$(sbatch --parsable $BEGIN_FLAG $CL_FLAG $ACCT_FLAG \
                --partition="$PARTITION" \
                --job-name="x_federated_${DS}" \
                --mail-user="$EMAIL" --mail-type=FAIL \
                --export="$EXPORTS" \
                $DEP \
                "$REPO/tests/slurm_scripts/submit_dataset.sh" federated "$DS" 2>/dev/null)
            [ -n "$JID" ] && N_OK=$((N_OK+1)) || echo "  FAILED x_federated_${DS}"
        fi
    fi
    echo "  $DS: queued"
done

# ── the opig suite ──────────────────────────────────────────────────────
# Standalone scripts, not a per-dataset category. Each waits for the last
# precompute link of every dataset it reads, so it cannot start before the
# searches it looks up exist -- and cannot write a dataset's database while a
# precompute job is still writing it.
if [ "$DO_OPIG" = true ]; then
    echo ""
    printf '%s\n' "$OPIG_JOBS" | while IFS='|' read -r NAME MEM CMD DSLIST; do
        [ -n "$NAME" ] || continue
        DEPIDS=""
        for D in $DSLIST; do
            _l=$(last_link "$D")
            [ -n "$_l" ] && DEPIDS="${DEPIDS}${_l}:"
        done
        DEPIDS="${DEPIDS%:}"
        DEP=""
        [ -n "$DEPIDS" ] && DEP="--dependency=afterany:${DEPIDS}"
        EXPORTS="ALL,PRUNING_MODE=${PRUNING}"
        [ "$COLD_REFUSAL" = 1 ] && EXPORTS="${EXPORTS},KIRBY_NO_COLD_SEARCH=1"
        # PM in the table marks the scripts that HAVE a --pruning-mode argument.
        # Four do not -- scaffold_hybrid, scaffold_similarity and the two
        # shared_vectors scripts -- and passing it to them is an immediate
        # argparse error, which is why it is not appended blindly.
        CMD=${CMD/ PM/ --pruning-mode ${PRUNING}}
        WRAP="cd ${REPO} && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && python -u tests/experiments/${CMD}"
        JID=$(sbatch --parsable $BEGIN_FLAG $CL_FLAG $ACCT_FLAG \
            --partition="$PARTITION" \
            --job-name="opig_${NAME}" \
            --nodes=1 --ntasks=1 --cpus-per-task=8 \
            --time=47:59:00 --mem="$MEM" \
            --mail-user="$EMAIL" --mail-type=FAIL \
            --export="$EXPORTS" \
            $DEP \
            --wrap="$WRAP" 2>/dev/null)
        if [ -n "$JID" ]; then
            echo "  opig_${NAME}: queued${DSLIST:+ after $DSLIST}"
        else
            echo "  opig_${NAME}: FAILED"
        fi
    done
fi

echo ""
echo "Submitted $N_OK jobs (opig counted separately above)."
echo ""
echo "Watch:    squeue -u \$USER -o '%.10i %.26j %.20P %.2t %.10M %.10L %R' | head -40"
echo "Progress: python tests/precompute_coverage.py --trial     (in an srun)"
echo ""
echo "⛔ The experiment jobs read SINGLE_DATASET_EXPERIMENTS and every experiment"
echo "   file WHEN THEY START. Add or remove an experiment and the queued jobs"
echo "   pick it up. Do NOT change anything that alters a stored result's"
echo "   identity — the metric, the fold split, the representation pool —"
echo "   or the rows already written are orphaned."
