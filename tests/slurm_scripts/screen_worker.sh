#!/bin/bash
# The one screening job script (dataset screening ledger, runbook/SCREEN.md).
# Same shape as worker.sh: the sbatch line gives cluster, partition, time,
# CPUs and memory; any arguments go to tests/experiments/screen/worker.py.
#
#   sbatch --clusters=htc -p medium,long --time=47:59:00 --cpus-per-task=8 --mem=64G \
#          --job-name=ks tests/slurm_scripts/screen_worker.sh
#
# KW_SLEEP      seconds to wait before the first claim (submit_screen_workers.sh sets it)
# KIRBY_SCREEN  the screen run root; default tests/results/CURRENT_SCREEN
# The job name starts with ks, never x_/opig_/pc_, so setup.sh never redirects
# it to the experiments' worker. No git pull here: pull on the login node.
set -uo pipefail
REPO="${KIRBY_REPO:-/data/stat-ecr/scat9264/KIRBy}"
[ -d "$REPO" ] || REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 2
export KIRBY_SKIP_INSTALLS="${KIRBY_SKIP_INSTALLS:-1}"
. ./setup.sh
export PYTHONWARNINGS=ignore
[ -f tests/results/SCREEN_ENV ] && . tests/results/SCREEN_ENV    # interpreter paths, written once (SCREEN.md)
echo "screen job ${SLURM_JOB_ID:-local} cluster ${SLURM_CLUSTER_NAME:-?} node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} mem ${SLURM_MEM_PER_NODE:-?}MB  $(date)"
echo "commit $(git rev-parse --short HEAD)  screen ${KIRBY_SCREEN:-$(cat tests/results/CURRENT_SCREEN 2>/dev/null)}"
exec python -u tests/experiments/screen/worker.py --sleep "${KW_SLEEP:-0}" "$@"
