#!/bin/bash
# STAGE 3: the full suite except noise -- complexity, similarity, federated and
# opig -- on every dataset whose precompute has finished. Written 2026-09-25.
#
#   bash tests/slurm_scripts/runbook/stage3_full.sh --dry-run
#   bash tests/slurm_scripts/runbook/stage3_full.sh
#   bash tests/slurm_scripts/runbook/stage3_full.sh --datasets "herg_ki caco2" --yes
#
# WHAT IT DOES.
#   1. Picks the cluster and account: where_to_submit.sh --best-cluster-account
#      (the highest FairShare over arc and htc), unless --clusters/--account
#      are given. An explicit --account wins; --account none passes none.
#   2. Reads tests/tools/precompute_status.py. A dataset goes forward only when
#      it says DONE AND nothing pc_ is still queued or running for it on either
#      cluster. Everything else is listed with its reason and left out.
#   3. Calls submit_everything.sh --experiments-only --categories
#      "complexity similarity" with those datasets. submit_everything.sh adds
#      federated on lhasa_herg, caco2 and logd when they are in the list, and
#      the opig jobs.
#
# WHY THE GATE AND NOT A DEPENDENCY. A SLURM dependency cannot name a job on
# another cluster. submit_everything.sh looks for pc_ jobs with squeue on the
# cluster it submits to only (its EXISTING, with -M <cluster>), and with
# --experiments-only it does not skip a dataset whose precompute is on the
# other cluster. An htc experiment job would then start while an arc precompute
# is still writing that dataset's searches. precompute_status.py reads both
# clusters' queues and the precompute's own logs (its docstring), so the gate
# is that instead.
#
# MEMORY. submit_everything.sh passes no --mem for the x_* jobs, so they take
# submit_dataset.sh's 64G. nt_*, gb_*, tcga_*, kiba, bindingdb_kd and qm9 had
# 128G from that header until 2026-09-25 (queue_noise_conditions.py:80 keeps
# the same split). So those datasets go in a second call with
# SBATCH_MEM_PER_NODE=128G, which sbatch reads as --mem when the command line
# does not give one (sbatch's input environment variables), and with
# --no-opig --no-federated so neither is queued twice.
#
# OPIG. submit_everything.sh submits all eleven opig scripts, and cannot pick
# some. They are included only when every dataset any of them reads is through
# the gate; otherwise --no-opig, the waiting ones are listed, and a later
# rerun of this script with --opig-only queues them. --include-opig overrides.
#
# WALL. The x_* jobs take submit_dataset.sh's 47:59:00; the partition default
# here is medium,long (short's 12 h maximum is below it).
#
# Other arguments pass through to submit_everything.sh (e.g. --begin).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KIRBY_REPO:-$(cd "$HERE/../../.." && pwd)}"
SLURM_DIR="$REPO/tests/slurm_scripts"
BIG_RE='^(nt_|gb_|tcga_|kiba$|bindingdb_kd$|qm9$)'

DATASETS=""
CLUSTERS=""
ACCOUNT=""
PARTITION="medium,long"
INCLUDE_NOT_DONE=false
INCLUDE_OPIG=false
OPIG_ONLY=false
DRY=false
PASS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)          DATASETS=$2; shift 2 ;;
        --clusters)          CLUSTERS=$2; shift 2 ;;
        --account)           ACCOUNT=$2; shift 2 ;;
        --partition)         PARTITION=$2; shift 2 ;;
        --include-not-done)  INCLUDE_NOT_DONE=true; shift ;;
        --include-opig)      INCLUDE_OPIG=true; shift ;;
        --opig-only)         OPIG_ONLY=true; shift ;;
        --dry-run)           DRY=true; PASS+=(--dry-run); shift ;;
        -h|--help)           sed -n '2,50p' "$0"; exit 0 ;;
        *)                   PASS+=("$1"); shift ;;
    esac
done

# ── 1. cluster and account ──
if [ -z "$CLUSTERS" ] || [ -z "$ACCOUNT" ]; then
    PAIR=$(bash "$SLURM_DIR/where_to_submit.sh" --best-cluster-account $CLUSTERS 2>/dev/null)
    if [ -n "$PAIR" ]; then
        [ -n "$CLUSTERS" ] || CLUSTERS=${PAIR%% *}
        [ -n "$ACCOUNT" ] || ACCOUNT=${PAIR##* }
    elif [ "$DRY" != true ]; then
        echo "Could not read fair share (where_to_submit.sh --best-cluster-account). Pass --clusters and --account."
        exit 1
    fi
fi
ACCT_ARGS=()
[ -n "$ACCOUNT" ] && [ "$ACCOUNT" != "none" ] && ACCT_ARGS=(--account "$ACCOUNT")
CL_ARGS=()
[ -n "$CLUSTERS" ] && CL_ARGS=(--clusters "$CLUSTERS")
echo "cluster ${CLUSTERS:-<unresolved>}   account ${ACCOUNT:-<unresolved>}   partition ${PARTITION}"

# ── 2. the precompute gate ──
cd "$REPO" || exit 2
STATUS_OUT=$(python tests/tools/precompute_status.py $DATASETS 2>&1)
echo "$STATUS_OUT" | sed 's/^/  | /'
# precompute_status.py prints "<mark> <dataset>" then two indented lines, the
# second "queued: <jobs>" or "queued: nothing queued".
GATE=$(printf '%s\n' "$STATUS_OUT" | awk '
    /^DONE /                 {ds=$2; st="done"; next}
    /^NOT DONE /             {ds=$3; st="not done"; next}
    /^⛔ NOT DONE /          {ds=$4; st="not done, nothing queued"; next}
    /^ +queued: / && ds!=""  {q=$0; sub(/^ +queued: /, "", q)
                              if (st=="done" && q!="nothing queued") st="done, but pc_ still queued: " q
                              print ds "\t" st; ds=""}')
PASSED=$(printf '%s\n' "$GATE" | awk -F'\t' '$2=="done" {print $1}' | xargs)
HELD=$(printf '%s\n' "$GATE" | awk -F'\t' '$2!="done" && $1!=""')
if [ "$INCLUDE_NOT_DONE" = true ]; then
    PASSED=$(printf '%s\n' "$GATE" | awk -F'\t' '$1!="" {print $1}' | xargs)
    [ -n "$DATASETS" ] && [ -z "$PASSED" ] && PASSED="$DATASETS"
    echo "--include-not-done: the gate is OFF; every dataset goes forward."
    HELD=""
fi
echo
echo "through the gate ($(echo "$PASSED" | wc -w | tr -d ' ')): ${PASSED:-none}"
if [ -n "$HELD" ]; then
    echo "held back:"
    printf '%s\n' "$HELD" | awk -F'\t' '{printf "  %-28s %s\n", $1, $2}'
fi
[ -n "$PASSED" ] || { echo "Nothing to submit."; exit 0; }

# ── opig: only when every dataset it reads is through ──
OPIG_FLAG=(--no-opig)
OPIG_WAIT=""
while IFS='|' read -r NAME _MEM _CMD DSLIST; do
    [ -n "$NAME" ] || continue
    for D in $DSLIST; do
        [[ " $PASSED " == *" $D "* ]] || OPIG_WAIT="${OPIG_WAIT}  opig_${NAME} waits for ${D}
"
    done
done < <(awk "/^OPIG_JOBS='/{on=1; next} on && /^'/{exit} on && NF" "$SLURM_DIR/submit_everything.sh")
if [ -z "$OPIG_WAIT" ] || [ "$INCLUDE_OPIG" = true ]; then
    OPIG_FLAG=()
    echo "opig: included"
else
    echo "opig: NOT included this time (--no-opig); rerun with --opig-only once these pass:"
    printf '%s' "$OPIG_WAIT" | sort -u
fi

SMALL=""; BIG=""
for D in $PASSED; do
    if [[ "$D" =~ $BIG_RE ]]; then BIG="$BIG $D"; else SMALL="$SMALL $D"; fi
done
SMALL=$(echo "$SMALL" | xargs); BIG=$(echo "$BIG" | xargs)

SE=(bash "$SLURM_DIR/submit_everything.sh" --experiments-only
    --categories "complexity similarity" --partition "$PARTITION"
    "${CL_ARGS[@]}" "${ACCT_ARGS[@]}")

run() { echo; echo ">>> $*"; "$@"; }

if [ "$OPIG_ONLY" = true ]; then
    # No category and no federated: submit_everything.sh then queues only the
    # opig block. One dataset is named so it has a list to walk.
    FIRST=$(echo "$PASSED" | awk '{print $1}')
    run "${SE[@]}" --categories "" --no-federated --datasets "$FIRST" "${PASS[@]}"
    exit $?
fi

RC=0
if [ -n "$SMALL" ]; then
    run "${SE[@]}" --datasets "$SMALL" "${OPIG_FLAG[@]}" "${PASS[@]}" || RC=$?
fi
if [ -n "$BIG" ]; then
    echo; echo "(128G for: $BIG)"
    run env SBATCH_MEM_PER_NODE=128G "${SE[@]}" --datasets "$BIG" --no-opig --no-federated "${PASS[@]}" || RC=$?
fi
echo
echo "Progress: bash tests/slurm_scripts/runbook/progress.sh tests/results/complexity tests/results/similarity tests/results/federated"
exit $RC
