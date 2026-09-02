#!/bin/bash
# F26 — does penalising redundancy in the allocation step help, and which
# measurement of redundancy?
#
# Twelve allocation arms (an equal split that scores nothing, no penalty on both
# bases, the existing entropy_*
# row-norm penalty, the new pairdist_* pairwise-distance penalty on both bases)
# x 2 objectives (predictive, and activity read-across) x the full budget list,
# through nested_cv_evaluate. Reports delta_mean_of_folds per dataset. Never
# averaged across datasets.
#
# ONE ARRAY TASK PER DATASET. Each task writes its own file
# tests/experiments/results/f26_redundancy_penalty_<ds>.json — no write races.
#
# CACHE. Runs under its own tier 'f26_redund', and every arm gets its own
# condition_suffix, so no arm's search file evicts another's. The
# per-(representation, model) baseline files are deliberately SHARED across the
# twelve arms — a single representation's baseline does not depend on the
# allocation method. Touches no existing result DB. Nothing needs deleting
# before or after.
#
# WALL TIME. Twelve arms is about 1.3x the config count of one default search
# (the default allocation list has 9 entries), times 2 objectives, times 5 outer
# folds. Estimated 9-30h per dataset from the config count — that is a
# structural estimate, not a measured time. Wall and memory are the house
# defaults (47:59:00 / 256G); do not tune them per job.
#
# NEVER PASS --account. 123 of the 125 scripts in this directory omit it and let
# SLURM resolve the default association. Adding it cost three failed
# submissions on 2026-07-11 and moved nothing.
#
# WHICH CLUSTER is set by the LOGIN NODE, not by --partition. arc-login sends
# everything to arc (multi-node parallel); htc is the high-throughput,
# low-core-count cluster. This job is 16 CPUs, CPU only, no GPU, five
# independent tasks — that is an htc job by ARC's own division of labour.
# where_to_submit.sh reports the CURRENT cluster only and never surfaces this
# choice. To run it on htc, submit from htc-login (or add --clusters=htc).
#
# --partition is LIVE STATE. Confirm with where_to_submit.sh, then:
#   sbatch --partition=short --array=0-219%40 \
#          tests/slurm_scripts/run_f26_redundancy_penalty.sh
#
# 220 TASKS = 10 datasets x 2 objectives x 11 allocation methods. ONE
# nested_cv_evaluate per task, so a task that dies loses one allocation method
# on one dataset, not 22 of them. Each writes its own file
# f26_redundancy_penalty_<ds>__<objective>__<allocation>.json -- no write races.
# The per-(representation, model) baseline files ARE shared across tasks, which
# is correct and safe: a baseline does not depend on the allocation method, and
# every cache write goes to a temporary file and is renamed over the target.
#
# WHY 220 AND NOT 10. The first submission (12907696) put 22 cells in one task
# and every one of the five tasks hit its wall having finished ONE outer fold of
# ONE allocation method. logd was 468 configurations per outer fold: 6 budgets x
# 3 importance methods x 5 internal evaluators x 6 models. Holding importance at
# lightgbm_gain and the internal evaluator at lightgbm takes that to 36.
#
# BEFORE QUEUEING TEN LONG TASKS, prove it runs on real data. devel has a
# 10-minute limit and is usually empty. setup.sh MUST be sourced first —
# without it KIRBY_CACHE_DIR is unset, so the representation cache resolves to
# the home quota, every representation is recomputed from SMILES, and the venv
# at /data/stat-cadd/scat9264/py311-kirby is never activated. Skip mode is what
# makes it fit in ten minutes; full mode spends the window on pip checks.
#
#   srun -p devel -c 8 --mem=32G --time=10:00 --pty /bin/bash
#   # then, from the repo root:
#   export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
#   python -u tests/experiments/framework/f26_redundancy_penalty.py \
#       --dataset herg_ki --framework --redundancy-only
#
# That loads the real representations and computes both redundancy matrices,
# in seconds IF the representation cache is already warm for that dataset. It
# does NOT exercise the search; for that, add --smoke instead.
#SBATCH --job-name=f26_redund
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# TEN datasets. Every dataset with a curated pool in
# experiments._shared.FRAMEWORK_TASKS, plus caco2.
DATASETS=(
  logd                    # molecular ADMET, RMSE
  herg_ki                 # molecular, RMSE
  buchwald_hartwig        # reaction, RMSE
  caco2                   # molecular ADMET, RMSE -- NOT in FRAMEWORK_TASKS, so
                          # --framework falls back to the 'full' preset for it
  flab_thermo_tresanco    # antibody, Spearman
  flab_pk_jain_fcrnrt     # antibody, Spearman
  flab_pk_jain_heprt      # antibody, Spearman
  ginkgo_hic              # antibody developability, Spearman
  tz_her2_binary          # antibody binding, AUROC
  tcga_luad_vs_lusc       # TCGA, AUROC
)
OBJECTIVES=(predictive similarity)
ARMS=(
  linear rank_linear
  entropy_0.5 entropy_1.0 entropy_2.0
  pairdist_0.5 pairdist_1.0 pairdist_2.0
  pairdist_score_0.5 pairdist_score_1.0 pairdist_score_2.0
)

n_arms=${#ARMS[@]}                       # 11
n_obj=${#OBJECTIVES[@]}                  # 2
per_ds=$(( n_obj * n_arms ))             # 22

ds_idx=$(( SLURM_ARRAY_TASK_ID / per_ds ))
rem=$(( SLURM_ARRAY_TASK_ID % per_ds ))
obj_idx=$(( rem / n_arms ))
arm_idx=$(( rem % n_arms ))

ds="${DATASETS[$ds_idx]}"
obj="${OBJECTIVES[$obj_idx]}"
arm="${ARMS[$arm_idx]}"

echo "f26 task ${SLURM_ARRAY_TASK_ID} -> ds=${ds} objective=${obj} allocation=${arm}"

python -u tests/experiments/framework/f26_redundancy_penalty.py \
    --dataset "${ds}" --framework --mode full \
    --objectives "${obj}" --arms "${arm}"
