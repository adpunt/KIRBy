#!/bin/bash
# The one job script that serves both ledgers. It names no dataset. It runs the
# Papers 2 and 3 worker and the screening worker one after the other. Each takes
# whatever its ledger has runnable and returns when nothing is left for this job.
#
# Even job numbers start with Papers, odd job numbers with screening, so neither
# ledger waits for the other to finish.
#
# Submitted by queue_ledgers.py. The old queued jobs (x_*, opig_*, pc_*) reach this
# script through setup.sh. Their environment was frozen when they were submitted,
# weeks ago, so every KIRBY_ variable is dropped first: a stale one would change
# what a search does. setup.sh and each ledger set the ones they need again.
set -uo pipefail
REPO="${KIRBY_REPO:-/data/stat-ecr/scat9264/KIRBy}"
[ -d "$REPO" ] || REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 2
for v in $(compgen -e | grep '^KIRBY_'); do unset "$v"; done
export KIRBY_REPO="$REPO"
export KIRBY_SKIP_INSTALLS=1
export KIRBY_LEDGER_MANAGED=1      # setup.sh must not send this job here a second time
if (( ${SLURM_JOB_ID:-0} % 2 )); then ORDER="screen papers"; else ORDER="papers screen"; fi
echo "ledgers job ${SLURM_JOB_ID:-local} name ${SLURM_JOB_NAME:-?} order: $ORDER  $(date)"
for L in $ORDER; do
    echo "=== $L ledger  $(date)"
    bash "$REPO/tests/slurm_scripts/${L}_worker.sh"
    echo "=== $L ledger ended with exit code $?  $(date)"
    export KW_SLEEP=0
done
