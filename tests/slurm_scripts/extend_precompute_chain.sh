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
#   bash tests/slurm_scripts/extend_precompute_chain.sh --repoint --go
#
#   --links N       extra links per dataset, default 1
#   --wall T        default 24:00:00
#   --partition P   default medium,long
#   --datasets "A B"  only these. Default: every dataset with a pc_ job queued
#   --repoint       also point each pending x_*_<dataset> job at the new last
#                   link, so it cannot start on a precompute that is still
#                   going. Without this they keep waiting for the old last link.
#   --go            actually submit. Without it, nothing is sent.
set -uo pipefail

REPO="${KIRBY_REPO:-/data/stat-ecr/scat9264/KIRBy}"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WALL="24:00:00"
PARTITION="medium,long"
LINKS=1
PRUNING="${KIRBY_PRUNING:-calibrated}"
EXPLICIT=""
REPOINT=false
GO=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --links)     LINKS=$2; shift 2 ;;
        --wall)      WALL=$2; shift 2 ;;
        --partition) PARTITION=$2; shift 2 ;;
        --datasets)  EXPLICIT=$2; shift 2 ;;
        --pruning)   PRUNING=$2; shift 2 ;;
        --repoint)   REPOINT=true; shift ;;
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
               | sed -E 's/^pc_//; s/_general_.*$//; s/_[0-9]+$//' \
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
echo " repoint      $([ "$REPOINT" = true ] && echo 'yes — experiment jobs will wait for the new last link' || echo 'no — experiment jobs keep waiting for the old one')"
echo " mode         $([ "$GO" = true ] && echo 'SUBMITTING' || echo 'dry run, nothing sent')"
echo "=========================================================="
echo ""

for DS in $DATASETS; do
    # Every job queued for this dataset, whatever its name or state. The new
    # link waits for all of them, so it can never write the database beside one.
    WAIT_ON=$(printf '%s\n' "$QUEUE" \
              | awk -F'|' -v d="$DS" '$2 == "pc_" d || $2 ~ "^pc_" d "_" {printf "%s:", $1}' \
              | sed 's/:$//')
    N_WAIT=$([ -n "$WAIT_ON" ] && echo "$WAIT_ON" | tr ':' '\n' | wc -l | tr -d ' ' || echo 0)

    MEM=64G
    case "$DS" in nt_*|gb_*|tcga_*|kiba|bindingdb_kd|qm9) MEM=128G ;; esac

    echo "$DS  —  waits for $N_WAIT queued job(s), ${MEM}"

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
                --export="ALL,KIRBY_PRUNING=${PRUNING}" \
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

echo ""
if [ "$GO" = true ]; then
    echo "Done. Check:  squeue -u \$USER -o '%.10i %.30j %.2t %.10M %.10L %R'"
else
    echo "Dry run. Add --go to submit."
fi
echo ""
