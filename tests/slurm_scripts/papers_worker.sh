#!/bin/bash
# The one Papers 2 and 3 job script (Papers 2 and 3 ledger, runbook/PAPERS.md).
# Same shape as worker.sh: the sbatch line gives cluster, partition, time,
# CPUs and memory; any arguments go to tests/experiments/papers/worker.py.
#
#   sbatch --clusters=htc -p medium,long --time=47:59:00 --cpus-per-task=8 --mem=64G \
#          --job-name=kp tests/slurm_scripts/papers_worker.sh
#
# KW_SLEEP      seconds to wait before the first claim (submit_papers_workers.sh sets it)
# KIRBY_PAPERS_RUN  the run root; default tests/results/CURRENT_PAPERS
# The job name starts with kp, never x_/opig_/pc_, so setup.sh never redirects
# it to the old experiments' worker. No git pull here: pull on the login node.
set -uo pipefail
REPO="${KIRBY_REPO:-/data/stat-ecr/scat9264/KIRBy}"
[ -d "$REPO" ] || REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 2
export KIRBY_SKIP_INSTALLS="${KIRBY_SKIP_INSTALLS:-1}"
. ./setup.sh
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"
export PYTHONWARNINGS=ignore
echo "papers job ${SLURM_JOB_ID:-local} cluster ${SLURM_CLUSTER_NAME:-?} node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} mem ${SLURM_MEM_PER_NODE:-?}MB  $(date)"
echo "commit $(git rev-parse --short HEAD)  run ${KIRBY_PAPERS_RUN:-$(cat tests/results/CURRENT_PAPERS 2>/dev/null)}"
exec python -u tests/experiments/papers/worker.py --sleep "${KW_SLEEP:-0}" "$@"
