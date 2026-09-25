#!/bin/bash
# STAGE 2: proof-of-concept runs of complexity, similarity, federated and opig
# on two trial datasets, before the full suite. Written 2026-09-25.
#
#   bash tests/slurm_scripts/runbook/stage2_trials.sh --datasets "herg_ki herg" --fed-datasets "caco2"
#   bash tests/slurm_scripts/runbook/stage2_trials.sh --datasets "herg_ki herg" --fed-datasets "caco2" --sbatch --go
#   bash tests/slurm_scripts/runbook/stage2_trials.sh --status
#
# WHAT IT MAKES. One command per (category, dataset) and one per OPIG script.
#   complexity, similarity: submit_dataset.sh <category> <dataset>, which runs
#       each of the category's experiments in its own python process.
#   federated: the same, on --fed-datasets. Only lhasa_herg, caco2 and logd
#       have silo data (FEDERATED_DATASETS, submit_everything.sh:99); a trial
#       dataset outside those three has nothing to run.
#   opig: runbook/opig_one.sh <name> for every name in OPIG_JOBS
#       (submit_everything.sh:127-139), each with its own flags from that
#       table. Each script reads its own datasets; --datasets does not change
#       them.
#
# WHERE IT WRITES. Results to tests/results/trial_<category>/, never the real
# results tree. One log per experiment: logs/trial/<category>_<dataset>_<experiment>.log
# (opig: logs/trial/opig_<name>.log). Exit codes per experiment:
# tests/results/trial_<category>/_job_summaries/<jobid>.tsv.
#
# TWO MODES.
#   default (srun): prints one srun line per command, to paste into a shell on
#       the login node. srun blocks, so this script does not run them; paste
#       several into separate tmux windows to run them side by side.
#       Partition interactive, 12 h (ARC docs give interactive 24 h maximum;
#       where_to_submit.sh §6 says 4 h -- `sinfo -M htc -p interactive` has
#       the live limit).
#   --sbatch: one batch job per command, partition short, 11:59:00. Prints the
#       sbatch lines; --go submits them.
#
# CLUSTER AND ACCOUNT. Default htc, and the account with the best fair share
# on it (where_to_submit.sh --best-cluster-account htc). --clusters best picks
# the cluster too; --account wins when given; --account none passes none.
# If srun on arc-login refuses --clusters=htc, log in to htc-login.arc.ox.ac.uk
# and pass --clusters "" (noise_prelim/check_one_srun.sh has the same note).
#
# Every command carries KIRBY_NO_PULL=1: pull once by hand before starting.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KIRBY_REPO:-$(cd "$HERE/../../.." && pwd)}"
SLURM_DIR="$REPO/tests/slurm_scripts"

DATASETS=""
FED_DATASETS=""
CATEGORIES="complexity similarity federated opig"
MODE="srun"
GO=false
CLUSTERS="htc"
ACCOUNT=""
PARTITION=""
WALL=""
CPUS=8
MEM="64G"
PRUNING="calibrated"
STATUS=false
FEDERATED_OK="lhasa_herg caco2 logd"

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)     DATASETS=$2; shift 2 ;;
        --fed-datasets) FED_DATASETS=$2; shift 2 ;;
        --categories)   CATEGORIES=$2; shift 2 ;;
        --sbatch)       MODE="sbatch"; shift ;;
        --go)           GO=true; shift ;;
        --clusters)     CLUSTERS=$2; shift 2 ;;
        --account)      ACCOUNT=$2; shift 2 ;;
        --partition)    PARTITION=$2; shift 2 ;;
        --wall)         WALL=$2; shift 2 ;;
        --cpus)         CPUS=$2; shift 2 ;;
        --mem)          MEM=$2; shift 2 ;;
        --pruning)      PRUNING=$2; shift 2 ;;
        --status)       STATUS=true; shift ;;
        -h|--help)      sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "$STATUS" = true ]; then
    cd "$REPO" || exit 2
    DIRS=()
    for C in complexity similarity federated opig; do DIRS+=("tests/results/trial_${C}"); done
    exec bash "$HERE/progress.sh" "${DIRS[@]}" logs/trial
fi

[ -n "$DATASETS" ] || { echo "Pass --datasets \"A B\" (e.g. herg_ki herg). The owner picks them after checking the precompute."; exit 1; }
[ -n "$PARTITION" ] || { [ "$MODE" = "srun" ] && PARTITION="interactive" || PARTITION="short"; }
[ -n "$WALL" ] || { [ "$MODE" = "srun" ] && WALL="12:00:00" || WALL="11:59:00"; }

# federated: the trial datasets that have silo data, unless named explicitly.
if [[ " $CATEGORIES " == *" federated "* ]] && [ -z "$FED_DATASETS" ]; then
    for D in $DATASETS; do
        [[ " $FEDERATED_OK " == *" $D "* ]] && FED_DATASETS="$FED_DATASETS $D"
    done
    FED_DATASETS=$(echo "$FED_DATASETS" | xargs)
    if [ -z "$FED_DATASETS" ]; then
        echo "federated: none of '$DATASETS' has silo data. Pass --fed-datasets with"
        echo "  one or more of: $FEDERATED_OK   (or drop federated from --categories)"
        exit 1
    fi
fi

# cluster and account
CL_FLAG=""; ACCT_FLAG=""
if [ -n "$CLUSTERS" ]; then
    if [ "$CLUSTERS" = "best" ] || [ -z "$ACCOUNT" ]; then
        ARG=""; [ "$CLUSTERS" != "best" ] && ARG="$CLUSTERS"
        PAIR=$(bash "$SLURM_DIR/where_to_submit.sh" --best-cluster-account $ARG 2>/dev/null)
        if [ -n "$PAIR" ]; then
            [ "$CLUSTERS" = "best" ] && CLUSTERS=${PAIR%% *}
            [ -n "$ACCOUNT" ] || ACCOUNT=${PAIR##* }
        elif [ "$CLUSTERS" = "best" ]; then
            echo "Could not resolve the best cluster (sshare unavailable?). Pass --clusters."; exit 1
        fi
    fi
    CL_FLAG="--clusters=${CLUSTERS}"
fi
[ -n "$ACCOUNT" ] && [ "$ACCOUNT" != "none" ] && ACCT_FLAG="--account=${ACCOUNT}"

ENV_BASE="KIRBY_NO_PULL=1 KIRBY_REPO=${REPO} KIRBY_EXP_LOG_DIR=logs/trial PRUNING_MODE=${PRUNING} KIRBY_NO_COLD_SEARCH=1"

echo "stage 2 trial runs   mode ${MODE}   cluster ${CLUSTERS:-login node}   account ${ACCOUNT:-<SLURM default>}"
echo "partition ${PARTITION}   wall ${WALL}   ${CPUS} CPUs   ${MEM}"
echo "results tests/results/trial_<category>/   logs logs/trial/"
echo "pull once first:  cd $REPO && git pull --ff-only origin \$(git rev-parse --abbrev-ref HEAD)"
echo

CMDS=()
add() { CMDS+=("$1"); }
for C in $CATEGORIES; do
    case "$C" in
        complexity|similarity|federated)
            LIST="$DATASETS"; [ "$C" = "federated" ] && LIST="$FED_DATASETS"
            for D in $LIST; do
                NAME="t2_${C}_${D}"
                ENVV="${ENV_BASE} KIRBY_OUTPUT_DIR=tests/results/trial_${C}"
                if [ "$MODE" = "srun" ]; then
                    add "cd $REPO && env ${ENVV} srun ${CL_FLAG} ${ACCT_FLAG} -p ${PARTITION} --cpus-per-task=${CPUS} --mem=${MEM} --time=${WALL} --job-name=${NAME} bash tests/slurm_scripts/submit_dataset.sh ${C} ${D}"
                else
                    add "cd $REPO && sbatch --parsable ${CL_FLAG} ${ACCT_FLAG} --partition=${PARTITION} --time=${WALL} --cpus-per-task=${CPUS} --mem=${MEM} --job-name=${NAME} --export=ALL,${ENVV// /,} tests/slurm_scripts/submit_dataset.sh ${C} ${D}"
                fi
            done ;;
        opig)
            for N in $(bash "$HERE/opig_one.sh" --list | awk '{print $1}'); do
                OM=$(bash "$HERE/opig_one.sh" --list | awk -v n="$N" '$1==n {print $2}')
                NAME="t2_opig_${N}"
                ENVV="${ENV_BASE} KIRBY_OUTPUT_DIR=tests/results/trial_opig"
                if [ "$MODE" = "srun" ]; then
                    add "cd $REPO && env ${ENVV} srun ${CL_FLAG} ${ACCT_FLAG} -p ${PARTITION} --cpus-per-task=${CPUS} --mem=${OM} --time=${WALL} --job-name=${NAME} bash tests/slurm_scripts/runbook/opig_one.sh ${N}"
                else
                    add "cd $REPO && sbatch --parsable ${CL_FLAG} ${ACCT_FLAG} --partition=${PARTITION} --time=${WALL} --cpus-per-task=${CPUS} --mem=${OM} --job-name=${NAME} --export=ALL,${ENVV// /,} tests/slurm_scripts/runbook/opig_one.sh ${N}"
                fi
            done ;;
        noise) echo "noise is stage 1 (stage1_prelim_noise.py), not stage 2; skipped here" ;;
        *) echo "Unknown category: $C"; exit 1 ;;
    esac
done

printf '%s\n' "${CMDS[@]}" | sed 's/  */ /g'
echo
echo "${#CMDS[@]} commands."

if [ "$MODE" = "sbatch" ] && [ "$GO" = true ]; then
    echo "submitting..."
    for L in "${CMDS[@]}"; do
        OUT=$(bash -c "$L" 2>&1)
        echo "  ${OUT}   <- ${L##*--job-name=}"
    done
elif [ "$MODE" = "sbatch" ]; then
    echo "Printed only. Add --go to submit them."
else
    echo "Paste each line into its own shell (tmux). Check progress with:"
    echo "  bash tests/slurm_scripts/runbook/stage2_trials.sh --status"
fi
