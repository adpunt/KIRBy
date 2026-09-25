#!/bin/bash
# SAMPLE. An array job: one task per (dataset, experiment, condition) line of a
# text file.
#
#   TASKS=tests/slurm_scripts/noise_prelim/prelim_tasks.example.txt
#   N=$(grep -cvE '^\s*(#|$)' "$TASKS")
#   sbatch --array=1-${N}%20 --export=ALL,TASKS=$TASKS \
#          tests/slurm_scripts/noise_prelim/prelim_array.sbatch
#
# TASK FILE. One task per line, whitespace-separated:
#     <dataset> <experiment> [condition]
# '-' or nothing in the condition column runs the experiment's own condition.
# Blank lines and lines starting with # are skipped and not counted: task k is
# the k-th remaining line, so --array=1-N covers the file.
#
# WHAT ONE TASK RUNS. run_one.py once, with the same cuts as check_one.sbatch
# (fold 0 of the 5-fold split, one level) unless PRELIM_CUT=none, which runs
# every fold and every level. A PRELIM_CUT=none task is a whole experiment for
# one condition; queue_noise_conditions.py gives those 12 h and resumes over
# several links, so use --partition=medium (48 h maximum) for it.
# Each task writes its own cache report beside its results, so hits and misses
# can be read per task afterwards.
#
# WHERE IT WRITES. tests/results/noise_prelim/array_<array job id>/<dataset>__<experiment>__<condition>/
#
# HTC, from https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html
# and .../slurm-reference.html:
#   --clusters=htc selects htc. NEVER pass --account.
#   short: 12 h maximum, 1 h default. medium: 48 h maximum. long: 30 days.
#   Default memory 8000 MB per CPU. htc nodes "typically have 16, 20 or 48 cores".
#   --array and SLURM_ARRAY_TASK_ID as below. The ARC pages state no limit on
#   array size or jobs per user; `scontrol show config | grep -i MaxArraySize`
#   on the login node gives the real one. %20 caps tasks running at once.
# Logs: SLURM's default slurm-<array id>_<task>.out in the submit directory;
# no --output here, by convention.
#
#SBATCH --job-name=np_prelim
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

TASKS="${TASKS:-tests/slurm_scripts/noise_prelim/prelim_tasks.example.txt}"
PRUNING_MODE="${PRUNING_MODE:-calibrated}"
PRELIM_CUT="${PRELIM_CUT:-one}"

export KIRBY_NO_COLD_SEARCH=1
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
unset KIRBY_NOISE_CACHE KIRBY_SMOKE_N

K="${SLURM_ARRAY_TASK_ID:?this is an array job: submit with --array=1-N}"
LINE=$(grep -vE '^\s*(#|$)' "$TASKS" | sed -n "${K}p")
[ -n "$LINE" ] || { echo "task ${K}: no line ${K} in ${TASKS}"; exit 2; }
read -r DATASET EXP COND _rest <<< "$LINE"
[ "${COND:-}" = "-" ] && COND=""

P="${PRELIM_DIR:-$PWD/tests/results/noise_prelim}/array_${SLURM_ARRAY_JOB_ID:-local}/${DATASET}__${EXP}__${COND:-default}"
mkdir -p "$P"
CUT_ARGS=()
[ "$PRELIM_CUT" = "none" ] && CUT_ARGS=(--all-folds --all-levels)
COND_ARG=()
[ -n "$COND" ] && COND_ARG=(--condition "$COND")

echo "=== task ${K}: ${DATASET} ${EXP} condition=${COND:-experiment default} cut=${PRELIM_CUT}"
echo "  branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ($(git rev-parse --short HEAD 2>/dev/null))"
echo "  node $(hostname)  cluster ${SLURM_CLUSTER_NAME:-?}  partition ${SLURM_JOB_PARTITION:-?}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}  KIRBY_LGBM_THREADS=${KIRBY_LGBM_THREADS}  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  pruning mode ${PRUNING_MODE}   output ${P}   start $(date)"

KIRBY_NOISE_CACHE_REPORT="$P/cache.json" \
  python -u tests/slurm_scripts/noise_prelim/run_one.py "$DATASET" "$EXP" \
    "${COND_ARG[@]}" "${CUT_ARGS[@]}" --pruning-mode "$PRUNING_MODE" --output "$P"
RC=$?
echo "=== task ${K} done $(date)  exit ${RC}"
exit "$RC"
