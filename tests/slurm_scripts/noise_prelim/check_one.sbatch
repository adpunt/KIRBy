#!/bin/bash
# SAMPLE. One noise experiment on one dataset, run TWICE, then checked.
#
#   sbatch tests/slurm_scripts/noise_prelim/check_one.sbatch <dataset> <experiment> [condition]
#   sbatch tests/slurm_scripts/noise_prelim/check_one.sbatch caco2 n07 outlier_p10
#   sbatch tests/slurm_scripts/noise_prelim/check_one.sbatch caco2 n18
#
# WHAT RUNS. run_one.py (next to this file) runs the experiment through
# noise_helpers.run_all with one fold (fold 0 of the same 5-fold split the full
# run makes), one condition and one level. It runs twice with identical
# settings; only the output directory and KIRBY_NOISE_CACHE_REPORT differ. Then
# tests/experiments/noise/check_noise_run.py reads both and exits 1 on any FAIL:
#   (a) both runs finished, rows on disk for the fold that ran, run 2's rows
#       equal run 1's;
#   (b) cache hits against misses per kind; run 2 must be all hits;
#   (c) recipes, representations, budget, pruning mode, tier, metric and
#       direction the result file records, beside what the precompute stored.
#       Read-only: every model fit in the search is blocked and counted.
#
# WHERE IT WRITES. tests/results/noise_prelim/<dataset>__<experiment>__<condition>__<jobid>/
# inside the repo checkout. Never tests/results/noise: the result files say
# n_splits=5 while holding one fold, and a full run would resume on them.
#
# CLUSTER. htc, by ARC's own division of labour: one node, 8 CPUs, no GPU.
# --clusters=htc is in the header. NEVER pass --account.
#   https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html
#
# WALL. short: 1 h default, 12 h maximum (ARC docs, same page). A one-fold,
# one-level run twice plus a read of the stored searches is expected to fit;
# nothing here has been timed on the cluster.
#
# MEMORY. ARC's default is 8000 MB per CPU (same page), so 8 CPUs x 8 GB = 64G
# asks for no more memory than the cores carry. queue_noise_conditions.py gives
# nt_*, gb_*, tcga_*, kiba, bindingdb_kd and qm9 128G: pass --mem=128G on the
# sbatch line for those.
#
# ENV. KIRBY_NO_COLD_SEARCH=1: an experiment asking for a search the precompute
# never stored raises in minutes instead of searching. KIRBY_LGBM_THREADS is in
# every noise_cache key (as lgbm_n_jobs), so a run that should reuse these
# cache entries must use the same value. OMP_NUM_THREADS is left at setup.sh's
# value (the CPU count) and printed; it is not in any cache key.
#
#SBATCH --job-name=np_check
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -uo pipefail

DATASET="${1:?usage: sbatch check_one.sbatch <dataset> <experiment> [condition]}"
EXP="${2:?usage: sbatch check_one.sbatch <dataset> <experiment> [condition]}"
COND="${3:-}"
PRUNING_MODE="${PRUNING_MODE:-calibrated}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

export KIRBY_NO_COLD_SEARCH=1
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
unset KIRBY_NOISE_CACHE          # the cache must be ON for check (b)
unset KIRBY_SMOKE_N              # a row cap, never wanted here

P="${PRELIM_DIR:-$PWD/tests/results/noise_prelim}/${DATASET}__${EXP}__${COND:-default}__${SLURM_JOB_ID:-local}"
mkdir -p "$P"
COND_ARG=()
[ -n "$COND" ] && COND_ARG=(--condition "$COND")

echo "=== noise prelim check: ${DATASET} ${EXP} condition=${COND:-experiment default}"
echo "  branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ($(git rev-parse --short HEAD 2>/dev/null))"
echo "  node $(hostname)  cluster ${SLURM_CLUSTER_NAME:-?}  partition ${SLURM_JOB_PARTITION:-?}  cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}  KIRBY_LGBM_THREADS=${KIRBY_LGBM_THREADS}  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  pruning mode ${PRUNING_MODE}   output ${P}"
echo "  start $(date)"

echo; echo "--- run 1 ---"
KIRBY_NOISE_CACHE_REPORT="$P/cache_run1.json" \
  python -u tests/slurm_scripts/noise_prelim/run_one.py "$DATASET" "$EXP" \
    "${COND_ARG[@]}" --pruning-mode "$PRUNING_MODE" --output "$P/run1"
RC1=$?
echo "--- run 1 exit ${RC1}  $(date)"

echo; echo "--- run 2 (identical settings) ---"
KIRBY_NOISE_CACHE_REPORT="$P/cache_run2.json" \
  python -u tests/slurm_scripts/noise_prelim/run_one.py "$DATASET" "$EXP" \
    "${COND_ARG[@]}" --pruning-mode "$PRUNING_MODE" --output "$P/run2"
RC2=$?
echo "--- run 2 exit ${RC2}  $(date)"

echo; echo "--- check ---"
python -u tests/experiments/noise/check_noise_run.py \
  --dataset "$DATASET" --experiment "$EXP" ${COND:+--condition "$COND"} \
  --out1 "$P/run1" --out2 "$P/run2" \
  --report1 "$P/cache_run1.json" --report2 "$P/cache_run2.json" \
  --rc1 "$RC1" --rc2 "$RC2" --pruning-mode "$PRUNING_MODE" \
  | tee "$P/check.txt"
RC=${PIPESTATUS[0]}
echo "=== done $(date)  check exit ${RC}  (printout also in $P/check.txt)"
exit "$RC"
