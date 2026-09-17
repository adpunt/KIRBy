#!/bin/bash
# The kirby_hybrid arm, one dataset per array task, as a BATCH job.
#
#   sbatch --array=0-7 tests/slurm_scripts/run_hybrid_arm.sh
#
# No interactive session. Submit it and close the laptop.
#
# RESOURCES ARE MEASURED, NOT DERIVED. Peak RSS on abcov_kd with ten
# representations loaded was 0.38 GB; /usr/bin/time -v reported 391 MB. 8G and
# 2 cores is generous. Earlier geometry jobs asked 256G and 48h and pended for
# a week -- at ~10 GB per core that reserved 26 cores' worth of memory and had
# to wait for a whole node. Nothing here needs that, and nothing here needs a
# GPU: the structures are cached and so are the representations.
#
# WALL. Scoring the arm without leakage refits the feature selection inside
# every fold, so it runs one find_best_hybrid per fold per model -- ten
# searches per dataset. short allows 12h, which is above the slowest dataset's
# measured full-survey time by a wide margin. If a task hits the wall, its
# result file is simply absent and resubmitting the array picks it up:
# --skip-existing continues rather than repeats.
#
#SBATCH --job-name=hybrid_arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --output=logs/hybrid_arm_%A_%a.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
mkdir -p logs

DATASETS=(abcov_kd ddg_combined flab_thermo_tresanco ginkgo_acsins
          ginkgo_colloidal ginkgo_hic ginkgo_pr_cho ginkgo_titer)
ds="${DATASETS[${SLURM_ARRAY_TASK_ID:-0}]}"

OUTPUT="${OUTPUT:-tests/results/geometry_survey}"
BASE_REPS="${BASE_REPS:-pruned}"
MODELS="${MODELS:-rf,mlp}"

{ git pull --ff-only origin similarity-metrics-study \
    || echo "[job] GIT PULL FAILED — running the checked-out tree, may be stale"; }
git log -1 --format='[job] running commit %h %ad %s' --date=short

. ./setup.sh || exit 1
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export OMP_NUM_THREADS=2
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"

echo "[job] dataset=${ds}  base-reps=${BASE_REPS}  models=${MODELS}"
mkdir -p "$OUTPUT"

TIME_V=""
[ -x /usr/bin/time ] && TIME_V="/usr/bin/time -v" || true

${TIME_V} python -u tests/experiments/structural_geometry_survey.py "$ds" \
    --base-reps "$BASE_REPS" --models "$MODELS" \
    --hybrid-only --skip-existing \
    --output "$OUTPUT/"
