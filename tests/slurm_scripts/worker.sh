#!/bin/bash
# The one job script. Every server job is this file; the sbatch line gives
# the cluster, partition, time, CPUs and memory (see submit_workers.sh), and
# any arguments are passed to worker.py (--dataset X, --experiment E, ...).
#
#   sbatch --clusters=htc -p medium,long --time=47:59:00 --cpus-per-task=8 --mem=64G \
#          --job-name=kw tests/slurm_scripts/worker.sh
#   sbatch ... --export=ALL,KW_SLEEP=90 tests/slurm_scripts/worker.sh --dataset qm9
#
# KW_SLEEP   seconds to wait before the first claim (submit_workers.sh sets
#            i*90 so a batch does not open the ledger at the same second; the
#            lock is what actually keeps them apart)
# KIRBY_RUN  the run root; default tests/results/CURRENT_RUN
# No #SBATCH resource lines on purpose, and no git pull: pull on the login
# node before submitting.
set -uo pipefail
REPO="${KIRBY_REPO:-/data/stat-ecr/scat9264/KIRBy}"
[ -d "$REPO" ] || REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 2
export KIRBY_SKIP_INSTALLS="${KIRBY_SKIP_INSTALLS:-1}"
. ./setup.sh
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"
export PYTHONWARNINGS=ignore
echo "worker job ${SLURM_JOB_ID:-local} cluster ${SLURM_CLUSTER_NAME:-?} node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} mem ${SLURM_MEM_PER_NODE:-?}MB  $(date)"
echo "commit $(git rev-parse --short HEAD)  run ${KIRBY_RUN:-$(cat tests/results/CURRENT_RUN 2>/dev/null)}"
exec python -u tests/experiments/worker.py --sleep "${KW_SLEEP:-0}" "$@"
