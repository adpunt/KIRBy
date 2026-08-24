#!/bin/bash
# c32 — allocation strategy CROSS (weighting x mechanism x N).
#
# Fills the (weighting x N) grid interior that f01/f12/c31 never ran: for each
# (dataset x model x budget) CELL it scores all 32 strategies
#   {2,3,5,10} N  x  {linear,power_2,softmax_T1,rank_exp_d0.3} W  x  {quota,global} M
# vs the best-single-rep target, paired per fold, with a bootstrap noise floor.
# Deliverable = per-full-strategy win-counts across the 60 cells.
#
# MODEL is an ARRAY dimension (not an inner loop): datasets(10) x models(3) = 30
# tasks. Task idx maps to
#   ds    = DATASETS[idx / nmodels]
#   model = MODELS[idx % nmodels]
# and each task passes C32_MODELS=<one model> and writes its OWN file
# <ds>__<model>.json (one file per task — no cross-task write races).
#
# Never calls find_best_hybrid, so NO result_db / cache is touched — nothing to
# delete before or after. Uses the engine's own _compute_allocation_weights
# (import-only); never edits hybrid.py / find_best_hybrid.py.
#
# --account and --partition are LIVE STATE — pass them at submit time. CPU job
# (lightgbm/rf/catboost are all CPU here), no GPU. Confirm the right
# account/partition/cluster first with where_to_submit.sh, then:
#   sbatch --account=stat-cadd --partition=medium --array=0-29%8 \
#          tests/slurm_scripts/run_c32_allocation_cross.sh
#   (10 datasets x 3 models = 30 tasks -> 0-29; drop to 20 tasks / 0-19 if you
#    cut catboost. The bash index math uses ${#MODELS[@]}, so it auto-adjusts.)
#SBATCH --job-name=c32_alloc_cross
#SBATCH --time=47:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# High-signal testbed (spanning molecular / reaction / antibody) — same 10 as c31.
# All have exactly 10 staged reps wired in complexity_helpers.DATASET_REPS.
DATASETS=(
  qm9                     # molecular (quantum) — large n, may be slow
  logd                    # molecular ADMET
  buchwald_hartwig        # reaction
  bh_doyle_plates         # reaction
  herg_ki                 # molecular
  adme_mdr1               # molecular ADMET
  adme_rlm                # molecular ADMET
  flab_expression_koenig  # antibody
  caco2                   # molecular ADMET
  adme_hlm                # molecular ADMET
)
# MODEL is the inner array dimension. 3 CPU tree/boosting models. Cut catboost
# for a 2-model run (array 0-19).
MODELS=(lightgbm rf catboost)

nmodels=${#MODELS[@]}
ds_idx=$(( SLURM_ARRAY_TASK_ID / nmodels ))
m_idx=$(( SLURM_ARRAY_TASK_ID % nmodels ))
ds="${DATASETS[$ds_idx]}"
model="${MODELS[$m_idx]}"

outdir="${KIRBY_CACHE_DIR}/c32_allocation_cross"
mkdir -p "${outdir}"
# One file per (dataset, model) task — no write races.
export C32_OUT="${outdir}/${ds}__${model}.json"
export C32_MODELS="${model}"

echo "c32 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds} model=${model}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR}"
echo "  C32_OUT=${C32_OUT}  C32_MODELS=${C32_MODELS}"
python -u tests/experiments/complexity/c32_allocation_strategy_cross.py "${ds}"
