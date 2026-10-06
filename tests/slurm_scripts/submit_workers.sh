#!/bin/bash
# Submit N identical worker jobs, staggered by 90 s each.
#
#   bash tests/slurm_scripts/submit_workers.sh N [worker args] -- [sbatch flags]
#
#   bash tests/slurm_scripts/submit_workers.sh 10 -- --clusters=htc -p medium,long
#   bash tests/slurm_scripts/submit_workers.sh 3 --dataset qm9 -- --clusters=htc -p medium,long --mem=128G --cpus-per-task=12
#   bash tests/slurm_scripts/submit_workers.sh 2 --dataset herg_ki --stage noise -- --clusters=arc -p medium
#
# Defaults when the sbatch flags leave them out: --time=47:59:00
# --cpus-per-task=8 --mem=64G --job-name=kw. Pass --mem=128G (and
# --cpus-per-task=12 on htc at 10 GB/core, 16 on arc at 8 GB/core) for a
# batch meant for the big datasets; a 64G worker never takes one of those.
# No dependencies. Pass --account=<the higher fair share in where_to_submit.sh §2>
# or SLURM uses the default association. Read where_to_submit.sh --fits first.
# DRY=1 prints the sbatch lines and submits nothing.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:?usage: submit_workers.sh N [worker args] -- [sbatch flags]}"; shift
WARGS=(); SFLAGS=()
seen=0
for x in "$@"; do
    if [ "$x" = "--" ]; then seen=1; continue; fi
    if [ $seen = 0 ]; then WARGS+=("$x"); else SFLAGS+=("$x"); fi
done
has() { for f in "${SFLAGS[@]:-}"; do case "$f" in "$1"*) return 0;; esac; done; return 1; }
has --time || SFLAGS+=(--time=47:59:00)
has --cpus-per-task || has -c || SFLAGS+=(--cpus-per-task=8)
has --mem || SFLAGS+=(--mem=64G)
NAME=kw
set +u; NW=${#WARGS[@]}; set -u        # an empty array trips set -u on older bash
for ((i = 0; i < NW; i++)); do
    [ "${WARGS[$i]}" = "--dataset" ] && NAME="kw_${WARGS[$((i+1))]//:/_}"
done
has --job-name || has -J || SFLAGS+=(--job-name="$NAME")
mkdir -p logs/workers
has --output || SFLAGS+=(--output="logs/workers/%x_%j.out")
for ((i = 0; i < N; i++)); do
    SLEEP=$((i * 90))
    CMD=(sbatch --parsable "${SFLAGS[@]}" --export=ALL,KW_SLEEP=$SLEEP "$HERE/worker.sh" ${WARGS[@]+"${WARGS[@]}"})
    if [ "${DRY:-0}" = 1 ]; then echo "${CMD[*]}"; continue; fi
    JID=$("${CMD[@]}") && echo "submitted $JID  sleep ${SLEEP}s  ${WARGS[*]+"${WARGS[*]}"}" || echo "FAILED: ${CMD[*]}"
done
