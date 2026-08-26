#!/bin/bash
# F24 — the augmentation admission grid. Which way of admitting extra features
# works, per dataset. Five methods x donor source x per-donor cap x candidate
# ranking x size rule x margin, with a matched random control arm and two
# planted decoy donors on every configuration.
#
# SPLIT BY DATASET *AND* BY PLACEMENT-AND-BUDGET, never by axis.
# 14 datasets x 4 placement-and-budget combinations = 56 array tasks.
#
#   task % 4 == 0  extras onto the best single representation at its full width
#   task % 4 == 1  extras onto the multi-representation hybrid, budget 500
#   task % 4 == 2  same, budget 1000
#   task % 4 == 3  same, budget 2000
#
# WHY THE SPLIT. Counted from the engine with nothing fitted (--count-only):
# herg_ki is 2,260 configurations and 446,760 model fits, tresanco 2,164 and
# 374,040. The only timing so far is 10.82 s per fit, measured on tresanco with
# rf on a LOCAL machine at load average 43 on 8 cores -- an upper bound, not the
# server's number. At that rate one dataset is 47 days as a single job and 11.7
# days per placement-and-budget, so even this split may not be enough. Run
# run_f24_augmentation_cost.sh FIRST and read the real seconds-per-fit off it.
#
# IF IT STILL OVERRUNS: split further within a dataset by admission method
# (--methods), which is five more sub-jobs. Do not drop a dataset, a budget, a
# donor source or an axis to make it fit.
#
# CACHE. f24 constructs recipes directly and calls create_hybrid. It runs no
# search and writes to no result_db, so nothing needs deleting before or after
# and no stored result can be evicted.
#
# --account and --partition are LIVE STATE. Confirm with where_to_submit.sh
# first, then:
#   sbatch --account=<acct> --partition=<part> --array=0-55%8 \
#          tests/slurm_scripts/run_f24_augmentation_grid.sh
#SBATCH --job-name=f24_aug
#SBATCH --time=47:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# The framework ten, plus the four widest measured gaps between the best and
# second-best solo representation. Same roster as run_f24_augmentation_cost.sh
# -- keep the two lists identical, or the cost numbers stop matching the run.
DATASETS=(
  logd  herg_ki  buchwald_hartwig  caco2  flab_thermo_tresanco
  flab_pk_jain_fcrnrt  flab_pk_jain_heprt  ginkgo_hic  tz_her2_binary
  tcga_luad_vs_lusc
  abagym_VEGF  abcov_kd  ginkgo_acsins  ddg_combined
)

t="${SLURM_ARRAY_TASK_ID}"
ds="${DATASETS[$((t / 4))]}"
slot=$((t % 4))

case "${slot}" in
  0) ARGS="--placement single" ;;
  1) ARGS="--placement hybrid --hybrid-budgets 500" ;;
  2) ARGS="--placement hybrid --hybrid-budgets 1000" ;;
  3) ARGS="--placement hybrid --hybrid-budgets 2000" ;;
esac

echo "f24 array task ${t} -> ds=${ds}  ${ARGS}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

# What this task is about to cost, printed before it starts, so a job that dies
# on the wall clock still leaves the count in its log.
python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --count-only ${ARGS}

python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --mode full ${ARGS}

echo "=== ${ds} slot ${slot}: done"
