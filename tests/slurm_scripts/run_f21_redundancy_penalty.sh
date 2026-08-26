#!/bin/bash
# F21 — does penalising redundancy in the allocation step help, and which
# measurement of redundancy?
#
# Eleven allocation arms (no penalty on both bases, the existing entropy_*
# row-norm penalty, the new pairdist_* pairwise-distance penalty on both bases)
# x 3 objectives (predictive, similarity_sac, similarity) x the full budget list,
# through nested_cv_evaluate. Reports delta_mean_of_folds per dataset. Never
# averaged across datasets.
#
# ONE ARRAY TASK PER DATASET. Each task writes its own file
# tests/experiments/results/f21_redundancy_penalty_<ds>.json — no write races.
#
# CACHE. Runs under its own tier 'f21_redund', and every (objective, arm) gets
# its own cache_namespace, so this touches no existing result DB and no arm
# evicts another. Nothing needs deleting before or after.
#
# WALL TIME. Eleven arms is about 1.2x the config count of one default search
# (the default allocation list has 9 entries), times 3 objectives, times 5 outer
# folds. Estimated 12-40h per dataset from the config count — that is a
# structural estimate, not a measured time. --time is set to the 47h ceiling.
#
# --account and --partition are LIVE STATE. Confirm with where_to_submit.sh
# first, then:
#   sbatch --account=<acct> --partition=<part> --array=0-9%5 \
#          tests/slurm_scripts/run_f21_redundancy_penalty.sh
#SBATCH --job-name=f21_redund
#SBATCH --time=47:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# Every dataset with a curated pool in experiments._shared.FRAMEWORK_TASKS —
# molecular, reaction, antibody and TCGA, regression and classification.
DATASETS=(
  logd                    # molecular ADMET, RMSE
  herg_ki                 # molecular, RMSE
  buchwald_hartwig        # reaction, RMSE
  caco2                   # molecular ADMET, RMSE -- NOT in FRAMEWORK_TASKS, so
                          # --framework falls back to the 'full' preset for it
                          # and says so in the log
  flab_thermo_tresanco    # antibody, Spearman
  flab_pk_jain_fcrnrt     # antibody, Spearman
  flab_pk_jain_heprt      # antibody, Spearman
  ginkgo_hic              # antibody developability, Spearman
  tz_her2_binary          # antibody binding, AUROC
  tcga_luad_vs_lusc       # TCGA, AUROC
)

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "f21 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

python -u tests/experiments/framework/f21_redundancy_penalty.py \
    --dataset "${ds}" --framework --mode full
