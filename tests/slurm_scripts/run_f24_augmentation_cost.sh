#!/bin/bash
# F24/F25 COST MEASUREMENT — seconds per model fit, per dataset.
#
# THIS IS NOT THE GRID. It runs ONE fold of ONE cell per dataset and reports how
# long a model fit takes there. The grid's own scripts print the fit COUNT
# without fitting anything (--count-only); multiplying the two is the cost
# estimate stage 2 owes. Neither number is guessed.
#
# Why it has to run here and not locally: only four of the fourteen datasets
# have their representations on the local drive, and the machine those four
# could be timed on was at load average 57 on 8 cores.
#
# ONE ARRAY TASK PER DATASET. Each writes its own file, so there are no write
# races: tests/experiments/results/framework/f24_augmentation_admission_<ds>__timing.json
#
# CACHE. Both experiments construct recipes directly and call create_hybrid.
# Neither runs a search and neither writes to result_db, so nothing needs
# deleting before or after and no stored result can be evicted.
#
# WALL TIME. One fold, one cell, one model per dataset. Minutes, not hours.
# --time is set well above that so a slow representation load cannot kill it.
#
# --account and --partition are LIVE STATE. Confirm with where_to_submit.sh
# first, then:
#   sbatch --account=<acct> --partition=<part> --array=0-13%7 \
#          tests/slurm_scripts/run_f24_augmentation_cost.sh
#SBATCH --job-name=f24_cost
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# The framework ten (the same roster run_f26_redundancy_penalty.sh submits),
# plus the four datasets with the widest measured gap between the best and
# second-best solo representation. The framework ten are all flat-topped: their
# gaps run 0.0005 to 0.0279, so none of them can answer whether extras help a
# DOMINANT representation. The four added ones can.
DATASETS=(
  # --- the framework ten -------------------------------------------------
  logd                    # molecular ADMET, RMSE      gap 0.0018
  herg_ki                 # molecular, RMSE            gap 0.0118
  buchwald_hartwig        # reaction, RMSE             gap 0.0202
  caco2                   # molecular ADMET, RMSE      gap 0.0279
  flab_thermo_tresanco    # antibody, Spearman         gap 0.0167
  flab_pk_jain_fcrnrt     # antibody, Spearman         gap not measured
  flab_pk_jain_heprt      # antibody, Spearman         gap not measured
  ginkgo_hic              # antibody developability    gap 0.0254
  tz_her2_binary          # antibody binding, AUROC    gap 0.0085
  tcga_luad_vs_lusc       # TCGA, AUROC                gap 0.0005
  # --- widest gaps, added for a dominant-representation testbed -----------
  abagym_VEGF             # antibody DMS               gap 0.0965
  abcov_kd                # antibody binding, n=141    gap 0.0632
  ginkgo_acsins           # antibody developability    gap 0.0530
  ddg_combined            # antibody stability         gap 0.0519
)
# Gaps recomputed 2026-08-26 from <cache>/rep_prune/<ds>/scores.json, the same
# file DATASET_STATUS.md's "best solo" column is built from. The four added
# datasets have an empty curated augmentation list in DATASET_REPS, so only the
# borrow-from-the-other-representations donor source runs on them.

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "f24 cost array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

# 1. The count. Fits nothing; prints configurations and model fits per dataset.
echo "=== ${ds}: counting"
python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --count-only

python -u tests/experiments/framework/f25_candidate_cutoff.py \
    --dataset "${ds}" --count-only

# 2. The timing. One fold of the most expensive cell: capped, both donor
#    sources, no per-donor cap, residual_correlation, the ladder.
echo "=== ${ds}: timing one fold of the most expensive configuration"
python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --time-only --mode intermediary --models lightgbm

echo "=== ${ds}: done"
