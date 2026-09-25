#!/bin/bash
# One array task of stage 1 or stage 4: one noise experiment on one dataset
# under one condition, in its own python process. Written 2026-09-25.
#
# Submitted by stage1_prelim_noise.py, which passes every resource on the
# sbatch line (cluster, account, partition, wall, CPUs, memory, array), so this
# file has no #SBATCH lines of its own.
#
# Reads from the environment (sbatch --export):
#   TASKS        task file; line k (blank lines and # lines not counted) is
#                "<dataset> <experiment> <condition>", '-' meaning none
#   RUNBOOK_ENV  env.sh written beside it: KIRBY_REPO, KIRBY_OUTPUT_DIR,
#                KIRBY_EXP_LOG_DIR, PRUNING_MODE, KIRBY_NO_COLD_SEARCH=1,
#                KIRBY_NO_PULL=1, KIRBY_SKIP_INSTALLS=1, KIRBY_NOISE_CONTROLS,
#                and optionally KIRBY_SIMILARITY_SETTINGS and EXP_TIMEOUT
#
# Runs submit_dataset.sh noise <dataset> <experiment> with NOISE_CONDITION
# set. That script records the exit code in
# <KIRBY_OUTPUT_DIR>/_job_summaries/<array>_<task>.tsv. This task's noise cache
# hits and misses go to <KIRBY_OUTPUT_DIR>/_cache_reports/ (noise_cache.py
# writes KIRBY_NOISE_CACHE_REPORT at exit).
set -uo pipefail

: "${TASKS:?TASKS must name the task file}"
: "${RUNBOOK_ENV:?RUNBOOK_ENV must name the env.sh written with the task file}"
K="${SLURM_ARRAY_TASK_ID:?this is an array task: submit with --array=1-N}"

# shellcheck disable=SC1090
. "$RUNBOOK_ENV"

LINE=$(grep -vE '^[[:space:]]*(#|$)' "$TASKS" | sed -n "${K}p")
[ -n "$LINE" ] || { echo "task ${K}: no line ${K} in ${TASKS}"; exit 2; }
read -r DATASET EXP COND _rest <<< "$LINE"
[ "${COND:-}" = "-" ] && COND=""

export NOISE_CONDITION="${COND}"
mkdir -p "${KIRBY_OUTPUT_DIR}/_cache_reports"
export KIRBY_NOISE_CACHE_REPORT="${KIRBY_OUTPUT_DIR}/_cache_reports/${DATASET}__${EXP}__${COND:-nocondition}__${SLURM_ARRAY_JOB_ID:-local}_${K}.json"

echo "=== runbook task ${K} of ${TASKS}: ${DATASET} ${EXP} condition=${COND:-none}"
echo "  controls KIRBY_NOISE_CONTROLS=${KIRBY_NOISE_CONTROLS:-unset}   similarity ${KIRBY_SIMILARITY_SETTINGS:-default (all)}"
echo "  cache report ${KIRBY_NOISE_CACHE_REPORT}"

exec bash "${KIRBY_REPO}/tests/slurm_scripts/submit_dataset.sh" noise "$DATASET" "$EXP"
