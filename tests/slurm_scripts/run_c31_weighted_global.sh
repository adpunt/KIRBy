#!/bin/bash
# c31 — performance-weighted global feature selection, one array task per dataset.
# Sweeps the FULL allocation weighting system (linear, power_2/3, softmax T*,
# rank_linear, rank_exp*, thresholded, score_gap, entropy*, top_3) to size a
# global candidate pool, then one global lightgbm_gain ranking picks the budget.
# Writes $KIRBY_CACHE_DIR/c31_weighted_global/<ds>.json (one file per task — no
# cross-task write races). Never calls find_best_hybrid, so NO result_db / cache
# is touched and there is nothing to delete before or after.
#
# --account and --partition are LIVE STATE — pass them at submit time. This is a
# CPU job (lightgbm), no GPU. Example (10 datasets in the list below -> 0-9):
#   sbatch --account=stat-cadd --partition=long --array=0-9%8 \
#          tests/slurm_scripts/run_c31_weighted_global.sh
# Confirm the right account/partition/cluster first with where_to_submit.sh.
#SBATCH --job-name=c31_wglobal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# High-signal testbed (best base-rep score from the Phase-2.1 validity gate,
# spanning molecular / reaction / antibody). Edit freely. herg_ki, logd and
# buchwald_hartwig were already run locally — comment them out to skip the
# re-compute, or keep them for one uniform server-produced result set.
DATASETS=(
  qm9                     # 0.96  molecular (quantum) — NB: large n, may be slow
  logd                    # 0.90  molecular ADMET
  buchwald_hartwig        # 0.83  reaction
  bh_doyle_plates         # 0.81  reaction
  herg_ki                 # 0.76  molecular
  adme_mdr1               # 0.74  molecular ADMET
  adme_rlm                # 0.70  molecular ADMET
  flab_expression_koenig  # 0.70  antibody
  caco2                   # 0.69  molecular ADMET
  adme_hlm                # 0.68  molecular ADMET
)

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
outdir="${KIRBY_CACHE_DIR}/c31_weighted_global"
mkdir -p "${outdir}"
export C31_OUT="${outdir}/${ds}.json"

echo "c31 array task ${SLURM_ARRAY_TASK_ID} -> ${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR}"
echo "  C31_OUT=${C31_OUT}"
python -u tests/experiments/complexity/c31_perf_weighted_global_selection.py "${ds}"
