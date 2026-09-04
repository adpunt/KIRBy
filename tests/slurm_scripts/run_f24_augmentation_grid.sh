#!/bin/bash
# F24 — the augmentation admission grid. STAGE 3.
#
# Which way of admitting extra features works, per dataset. Four admission
# methods x donor source x per-donor cap x candidate ranking x size rule x
# margin, with a matched random control arm and two planted decoy donors on
# every configuration.
#
# SPLIT BY DATASET, ALLOCATION, ADMISSION METHOD AND DOWNSTREAM MODEL.
# Never by an axis of the study. Every axis still runs, on every dataset.
#
# WHY THIS SPLIT, from measured numbers. Job 8707403 timed one fold of the most
# expensive configuration on each dataset at budget 250 (AUGMENTATION.md 9.17):
#
#   dataset                 s/fit   fits/task   whole task   capped, one model
#   caco2                   13.00      18,240      65.9 h       16.4 h
#   tcga_brca_pam50         22.15      11,352      69.8 h       17.1 h
#   buchwald_hartwig        31.31      10,464      91.0 h       21.8 h
#   flab_expression_koenig  66.41       9,768     180.2 h       43.6 h
#
# All four overrun a 48-hour wall as one task per allocation. Splitting by
# admission method alone fits three of them; flab_expression_koenig at 87.2 h
# does not. Splitting by method AND model fits all four.
#
# flab_expression_koenig IS THE ONE TO WATCH. 43.6 h is the counted FLOOR. The
# margin ladder runs once per configuration and every margin multiple is read
# off its rung scores; a multiple that picks a different block size gets its own
# run, so capped and uncapped can be up to twice the counted fits. At that
# ceiling koenig is 87.2 h. Its tasks therefore go to a partition with room:
#
#   sbatch --clusters=htc --partition=medium --time=47:59:00 \
#          --array=0-31,48-63 tests/slurm_scripts/run_f24_augmentation_grid.sh
#   sbatch --clusters=htc --partition=long --time=4-00:00:00 \
#          --array=32-47   tests/slurm_scripts/run_f24_augmentation_grid.sh
#
# f24 writes its result ONCE, at the end. A task killed on the wall leaves
# nothing, which is why koenig gets the longer wall rather than a tight one.
#
# MEMORY. Job 8707403 used 0.44G to 3.97G at 16 cores. 64G is ample and keeps
# the request under arc's 8 GB per core, so it backfills instead of waiting for
# a whole node. The old 256G request is what made these sit in the queue.
#
# CACHE. f24 constructs recipes directly and calls create_hybrid. It runs no
# search and writes to no result_db, so nothing needs deleting before or after
# and no stored result can be evicted.
#
# NEVER PASS --account. 123 of the 125 scripts in this directory omit it and let
# SLURM resolve the default association. Adding it cost three failed
# submissions on 2026-07-11 and moved nothing.
#
# WHICH CLUSTER is set by the LOGIN NODE, not by --partition. CPU only, no GPU.
# --partition is LIVE STATE: confirm with where_to_submit.sh before submitting.
#
#SBATCH --job-name=f24_aug
#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# FOUR datasets, chosen by the user on 2026-09-02. One reaction, one molecular,
# one antibody, one TCGA. RMSE, Spearman and f1_weighted. tcga_brca_pam50 is the
# first classification dataset this study has run.
#
# ORDER MATTERS: the array ranges in the header are computed from it.
#   index 0  buchwald_hartwig       tasks  0-15
#   index 1  caco2                  tasks 16-31
#   index 2  flab_expression_koenig tasks 32-47   <- the long ones
#   index 3  tcga_brca_pam50        tasks 48-63
DATASETS=(
  buchwald_hartwig        # reaction,  RMSE         n=3,955
  caco2                   # molecular, RMSE         n=2,158
  flab_expression_koenig  # antibody,  Spearman     n=4,275
  tcga_brca_pam50         # TCGA,      f1_weighted  n=900
)

# Three quota allocation methods, chosen by the user on 2026-09-02: linear,
# power-law squared, rank-exponential. All three resolve to
# hybrid.allocate_performance_weighted with a different weighting scheme.
# rank_exp_d0.3 is the decay in the current find_best_hybrid
# DEFAULT_ALLOCATION_METHODS; rank_exp_d0.1 and rank_exp_d0.5 also exist.
ALLOCS=(performance_weighted power_2 rank_exp_d0.3)

# One budget, 250, chosen by the user on 2026-09-02, down from 500/1000/2000.
# Measured on buchwald_hartwig: 31.31 s per fit at 250 against 42.74 at 500,
# 27% cheaper. The grid can no longer say whether the winning admission method
# depends on the budget. That is what the cut costs.
BUDGET=250

# 16 tasks per dataset:
#   slot 0        the single-representation placement, all methods, both models.
#                 Allocation is inert with one representation -- it takes 100%
#                 of the budget whatever the allocation says -- so this runs
#                 once and the experiment prints that it collapsed the axis.
#   slots 1-15    the hybrid placement: 3 allocations x 5 method-and-model
#                 groups. none and all are cheap and share one task; capped and
#                 uncapped each get one task per downstream model.
PER_DS=16
ds="${DATASETS[$(( SLURM_ARRAY_TASK_ID / PER_DS ))]}"
slot=$(( SLURM_ARRAY_TASK_ID % PER_DS ))

if [ "${slot}" -eq 0 ]; then
  ARGS="--placement single"
else
  i=$(( slot - 1 ))
  alloc="${ALLOCS[$(( i / 5 ))]}"
  case $(( i % 5 )) in
    0) grp="--methods none,all" ;;
    1) grp="--methods capped   --models lightgbm" ;;
    2) grp="--methods capped   --models rf" ;;
    3) grp="--methods uncapped --models lightgbm" ;;
    4) grp="--methods uncapped --models rf" ;;
  esac
  ARGS="--placement hybrid --hybrid-budgets ${BUDGET} --allocations ${alloc} ${grp}"
fi

echo "f24 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}  ${ARGS}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

# What this task is about to cost, printed before it starts, so a job that dies
# on the wall clock still leaves the count in its log.
python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --count-only ${ARGS}

python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --mode full ${ARGS}

echo "=== ${ds} slot ${slot}: done"
