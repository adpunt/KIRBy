#!/bin/bash
# Add more precompute links to datasets that already have jobs queued, without
# cancelling anything and without two jobs writing one dataset's database.
#
# WHY THIS EXISTS AND submit_everything.sh DOES NOT DO IT.
# submit_everything.sh only recognises jobs named pc_<dataset>_general_ as
# already queued (its EXISTING check). The links it submits itself are named
# pc_<dataset>_<link>, which that pattern does not match. So running it a second
# time submits links that wait for nothing, and two precompute jobs then write
# result_db/<dataset>.db at once. SQLite waits 120 s for the other writer and
# then the search dies AFTER paying for its baselines and its per-fold scores
# (result_cache.py:187-200).
#
# This script takes the opposite approach: a dataset's new link waits for EVERY
# job currently queued for that dataset, whatever it is named and whatever state
# it is in. One writer at a time, always.
#
# THE WALL. Default 24 hours, not the 12 in run_precompute_dataset.sh. Each link
# re-pays the setup a pass does before it scores anything -- every
# representation's own baseline, then the per-fold importance scores, then the
# per-fold allocation scores once per internal evaluator. Measured on caco2 on
# 2026-09-14: 11 minutes of a 12.7-minute run was that setup. Fewer, longer
# links pay it fewer times. The cost is scheduling: a 12-hour request backfills
# into gaps a 24-hour one cannot, and a 24-hour request cannot run in a 12-hour
# partition at all. Check the limits before choosing:  sinfo -o "%P %l"
#
# ⛔ A RUNNING OR QUEUED JOB'S WALL CANNOT BE RAISED. Slurm lets a normal user
# only LOWER a time limit; raising one needs operator rights. This script
# changes nothing about jobs that already exist except, with --repoint, what
# they wait for.
#
# USAGE
#   bash tests/slurm_scripts/extend_precompute_chain.sh                  # dry run
#   bash tests/slurm_scripts/extend_precompute_chain.sh --go
#   bash tests/slurm_scripts/extend_precompute_chain.sh --links 2 --go
#   bash tests/slurm_scripts/extend_precompute_chain.sh --datasets "logd caco2" --go
#   bash tests/slurm_scripts/extend_precompute_chain.sh --no-repoint --go
#
#   --links N       extra links per dataset, default 1
#   --wall T        default 24:00:00
#   --speed S       fast | full. Default: the dataset's class in
#                   run_precompute_dataset.sh. The twelve LARGE datasets --
#                   logd and flab_expression_koenig among them -- default to
#                   fast, which evaluates 816 configurations per condition
#                   rather than 8,160. Pass full to make a link work the whole
#                   grid instead.
#   --partition P   default medium,long
#   --datasets "A B"  only these. Default: every dataset with a pc_ job queued
#   --no-repoint    leave the experiment jobs pointed at the link they were
#                   submitted against. BY DEFAULT they are repointed at the new
#                   last link, so none can start while a precompute still runs.
#   --go            actually submit. Without it, nothing is sent.
set -uo pipefail

REPO="${KIRBY_REPO:-/data/stat-ecr/scat9264/KIRBy}"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WALL="24:00:00"
PARTITION="medium,long"
LINKS=1
PRUNING="${KIRBY_PRUNING:-calibrated}"
SPEED=""          # empty = the dataset's class in run_precompute_dataset.sh
EXPLICIT=""
REPOINT=true
GO=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --links)     LINKS=$2; shift 2 ;;
        --wall)      WALL=$2; shift 2 ;;
        --partition) PARTITION=$2; shift 2 ;;
        --datasets)  EXPLICIT=$2; shift 2 ;;
        --pruning)   PRUNING=$2; shift 2 ;;
        --speed)     SPEED=$2; shift 2 ;;
        --repoint)   REPOINT=true; shift ;;
        --no-repoint) REPOINT=false; shift ;;
        --go)        GO=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── the queue, read once ────────────────────────────────────────────────
QUEUE=$(squeue -u "$USER" -h -o "%i|%j|%T|%a" 2>/dev/null)
[ -n "$QUEUE" ] || { echo "No jobs queued for $USER. Nothing to extend."; exit 0; }

# The account every precompute was submitted under. Taken from the queue rather
# than guessed, so the new links land in the same fair-share pool.
ACCOUNT=$(printf '%s\n' "$QUEUE" | awk -F'|' '$2 ~ /^pc_/ {print $4; exit}')
[ -n "$ACCOUNT" ] || { echo "No pc_ job queued, so no account to copy. Aborting."; exit 1; }

# ── which datasets ──────────────────────────────────────────────────────
# A precompute job is pc_<dataset>_<link> (this year) or pc_<dataset>_general_...
# (the older naming). Strip either tail to get the dataset back.
if [ -n "$EXPLICIT" ]; then
    DATASETS="$EXPLICIT"
else
    DATASETS=$(printf '%s\n' "$QUEUE" | awk -F'|' '$2 ~ /^pc_/ {print $2}' \
               | sed -E 's/^pc_//; s/_general_.*$//; s/_x?[0-9]+$//' \
               | sort -u | tr '\n' ' ')
fi
DATASETS=$(echo "$DATASETS" | xargs)
N_DS=$(echo "$DATASETS" | wc -w | tr -d ' ')

echo ""
echo "=========================================================="
echo " datasets     $N_DS"
echo " extra links  $LINKS each, ${WALL} per link"
echo " partition    $PARTITION      account $ACCOUNT"
echo " pruning      $PRUNING"
echo " speed        ${SPEED:-per dataset class — LARGE datasets run fast, 816 configurations per condition, not 8,160}"
echo " repoint      $([ "$REPOINT" = true ] && echo 'YES — every experiment job moves onto the new last link' || echo 'no — experiment jobs keep waiting for the link they have')"
echo " mode         $([ "$GO" = true ] && echo 'SUBMITTING' || echo 'dry run, nothing sent')"
echo "=========================================================="
echo ""

for DS in $DATASETS; do
    # Every job queued for this dataset, whatever its state. The new link waits
    # for all of them, so it can never write the database beside one.
    #
    # ⛔ THE NAME IS MATCHED WHOLE, NOT AS A PREFIX. A precompute job is
    # pc_<dataset>_<link>, pc_<dataset>_x<link> from this script, or the older
    # pc_<dataset>_general_<condition>. Matching "^pc_herg_" instead gave herg
    # every job belonging to herg_ki, because that name starts the same way.
    _PAT="^pc_${DS}_(x?[0-9]+|general_.*)$"
    WAIT_ON=$(printf '%s\n' "$QUEUE" \
              | awk -F'|' -v p="$_PAT" '$2 ~ p {printf "%s:", $1}' | sed 's/:$//')
    N_WAIT=$([ -n "$WAIT_ON" ] && echo "$WAIT_ON" | tr ':' '\n' | wc -l | tr -d ' ' || echo 0)

    MEM=64G
    case "$DS" in nt_*|gb_*|tcga_*|kiba|bindingdb_kd|qm9) MEM=128G ;; esac

    echo "$DS   ($MEM)"
    # Name and state of each one, so the dependency list is readable rather
    # than a row of job numbers.
    printf '%s\n' "$QUEUE" | awk -F'|' -v p="$_PAT" \
        '$2 ~ p {printf "   waits for %-12s %-34s %s\n", $1, $2, $3}'
    [ "$N_WAIT" = "0" ] && echo "   waits for nothing — no precompute queued for it"

    LAST="$WAIT_ON"
    for L in $(seq 1 "$LINKS"); do
        DEP=""
        [ -n "$LAST" ] && DEP="--dependency=afterany:${LAST}"
        if [ "$GO" = true ]; then
            JID=$(sbatch --parsable \
                --account="$ACCOUNT" --partition="$PARTITION" --time="$WALL" \
                --mem="$MEM" --cpus-per-task=8 \
                --job-name="pc_${DS}_x${L}" \
                --mail-user="$EMAIL" --mail-type=FAIL \
                --export="ALL${SPEED:+,KIRBY_SPEED=${SPEED}},KIRBY_PRUNING=${PRUNING}" \
                $DEP \
                "$REPO/tests/slurm_scripts/run_precompute_dataset.sh" "$DS" 2>/dev/null)
            [ -n "$JID" ] || { echo "   FAILED pc_${DS}_x${L}"; continue; }
            echo "   submitted $JID  pc_${DS}_x${L}"
            LAST="$JID"
        else
            echo "   would submit  pc_${DS}_x${L}  ${DEP:-(no dependency — nothing queued for it)}"
            LAST="<new_x${L}>"
        fi
    done

    # ── repoint the experiment jobs ──────────────────────────────────────
    # x_<category>_<dataset>, submitted by submit_everything.sh with
    # --dependency=afterany:<old last link>. Pointing them at the new last link
    # is the only thing that stops one starting while the precompute still runs.
    # Only PENDING jobs can be changed; a running one is left alone.
    if [ "$REPOINT" = true ]; then
        # Matched on the WHOLE name, x_<category>_<dataset>, not on a suffix.
        # A suffix match would give herg every job belonging to lhasa_herg.
        # The opig jobs are NOT touched: each one waits on several datasets at
        # once and its name carries none of them, so repointing it from here
        # would drop the other datasets it is waiting for.
        XJOBS=$(printf '%s\n' "$QUEUE" \
                | awk -F'|' -v d="$DS" \
                  '$3 == "PENDING" && $2 ~ "^x_(complexity|similarity|noise|federated)_" d "$" {print $1}')
        for XJ in $XJOBS; do
            if [ "$GO" = true ]; then
                scontrol update jobid="$XJ" dependency=afterany:"$LAST" \
                    && echo "   repointed $XJ -> $LAST" \
                    || echo "   COULD NOT repoint $XJ"
            else
                echo "   would repoint $XJ -> $LAST"
            fi
        done
    fi
done

# ── datasets whose chain has already ended ──────────────────────────────
# A dataset only appears above if it still has a precompute job queued. One
# whose four links have all finished has none, so it is invisible here -- and
# it is the DANGEROUS case, because its experiment jobs are already past their
# dependency and free to start on whatever the precompute managed to store.
# Name them, and say how to include them.
ORPHANS=""
for X in $(printf '%s\n' "$QUEUE" \
           | awk -F'|' '$2 ~ /^x_(complexity|similarity|noise|federated)_/ {n=$2;
               sub(/^x_(complexity|similarity|noise|federated)_/, "", n); print n}' \
           | sort -u); do
    if ! printf '%s\n' "$QUEUE" | awk -F'|' -v p="^pc_${X}_(x?[0-9]+|general_.*)$" \
         '$2 ~ p {found=1} END {exit !found}'; then
        ORPHANS="$ORPHANS $X"
    fi
done
ORPHANS=$(echo "$ORPHANS" | xargs)
if [ -n "$ORPHANS" ]; then
    echo ""
    echo "⛔ THESE HAVE EXPERIMENT JOBS QUEUED AND NO PRECOMPUTE LEFT TO WAIT FOR."
    echo "   Their chains have ended. Nothing above touched them, because there"
    echo "   was nothing to chain onto. Check what each one stored before you"
    echo "   decide -- under half and the experiment quits in minutes:"
    echo ""
    echo "     python tests/precompute_coverage.py $ORPHANS"
    echo ""
    echo "   To give them links anyway (they wait for nothing, which is safe:"
    echo "   no other job is writing their database):"
    echo ""
    echo "     bash $0 --datasets \"$ORPHANS\" --links 2 --go"
    echo ""
fi

echo ""
if [ "$GO" = true ]; then
    echo "Done. Check:  squeue -u \$USER -o '%.10i %.30j %.2t %.10M %.10L %R'"
else
    echo "Dry run. Add --go to submit."
fi
echo ""
