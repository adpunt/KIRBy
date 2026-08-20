#!/bin/bash
# c31 — performance-weighted global feature selection.
#
# EXTENDED 2026-08-20 (items 1-4): the experiment now loops the DOWNSTREAM MODEL
# (lightgbm, rf, catboost), sweeps the GLOBAL-ranking IMPORTANCE (lightgbm_gain,
# treeshap — 2 of the engine's main 3), sizes each rep's pool slot PER FOLD
# (train-only, no leak), and emits a group-block bootstrap noise floor per cell.
#
# To keep per-task wall time bounded, MODEL is an ARRAY DIMENSION (not an inner
# loop): the array is datasets(10) x models(3) = 30 tasks. Task idx maps to
#   ds    = DATASETS[idx / nmodels]
#   model = MODELS[idx % nmodels]
# and each task passes C31_MODELS=<one model> and writes its OWN file
# <ds>__<model>.json (one file per task — no cross-task write races).
#
# SCHEME CUT: the experiment's 15-scheme weighting sweep is cut to the round-3
# winners linear + power_2 + top_2 (see SCHEMES / CAVEATS in the .py). This
# offsets the model x importance multiplier so the 24h task shape still holds.
# Restore the fuller SCHEMES dict from git history for a scheme-only re-sweep.
#
# Never calls find_best_hybrid, so NO result_db / cache is touched — nothing to
# delete before or after.
#
# --account and --partition are LIVE STATE — pass them at submit time. CPU job
# (lightgbm/rf/catboost/mlp are all CPU here), no GPU. Confirm the right
# account/partition/cluster first with where_to_submit.sh, then:
#   sbatch --account=stat-cadd --partition=medium --array=0-29%8 \
#          tests/slurm_scripts/run_c31_weighted_global.sh
#   (10 datasets x 3 models = 30 tasks -> 0-29; drop to 20 tasks / 0-19 if you
#    cut catboost. The bash index math uses ${#MODELS[@]}, so it auto-adjusts.)
#SBATCH --job-name=c31_wglobal
#SBATCH --time=47:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# High-signal testbed (best base-rep score from the Phase-2.1 validity gate,
# spanning molecular / reaction / antibody). Edit freely.
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
# MODEL is the inner array dimension (item 1). Trimmed 2026-08-20: 3 CPU tree/
# boosting models, consistent coverage across all datasets. mlp dropped (it
# auto-skips large-n datasets -> ragged coverage). Cut catboost for a 2-model run.
MODELS=(lightgbm rf catboost)

nmodels=${#MODELS[@]}
ds_idx=$(( SLURM_ARRAY_TASK_ID / nmodels ))
m_idx=$(( SLURM_ARRAY_TASK_ID % nmodels ))
ds="${DATASETS[$ds_idx]}"
model="${MODELS[$m_idx]}"

outdir="${KIRBY_CACHE_DIR}/c31_weighted_global"
mkdir -p "${outdir}"
# One file per (dataset, model) task — no write races; the reader merges the
# per-model files (each has by_model={<model>: ...}).
export C31_OUT="${outdir}/${ds}__${model}.json"
export C31_MODELS="${model}"

echo "c31 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds} model=${model}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR}"
echo "  C31_OUT=${C31_OUT}  C31_MODELS=${C31_MODELS}"
python -u tests/experiments/complexity/c31_perf_weighted_global_selection.py "${ds}"
