#!/bin/bash
# f30 — feature importance scoring methods: do new ones beat the current ones?
#
# Varies ONLY the importance scoring method. Held fixed: budget 100, 5 group-aware
# folds, the pruned 10 representations per dataset, allocation in
# {rank_exp_d0.3, power_2} on the quota mechanism, internal evaluator and
# downstream model each in {lightgbm, rf}.
#
# Never calls find_best_hybrid — it imports the engine's own
# _compute_allocation_weights / _sanitize_baseline_scores and reproduces the quota
# mechanism, exactly as c32 does. So NO result_db and NO search/baseline cache is
# touched: there is NOTHING TO DELETE before or after this job, on any dataset.
#
# TWO STAGES. The per-representation baseline scores depend only on
# (dataset, fold, model), so they are computed once and every arm reuses them.
#
#   stage 1  baselines : 4 tasks, one per dataset. Metric-direction gate, the
#                        10-representation assertion, folds, inner-CV scores and
#                        the best-single-representation target per model.
#                        MUST FINISH BEFORE STAGE 2.
#   stage 2  arms      : 4 datasets x 31 arms = 124 tasks. Each scores its one
#                        method across 10 representations x 5 folds, builds both
#                        allocations, evaluates both downstream models, and writes
#                        <ds>__<arm>__full.json plus a compressed .npz of the score
#                        vectors (the agreement analysis reads those).
#
# One file per task — no cross-task write races.
#
# --account / --partition / --clusters are LIVE STATE. Run where_to_submit.sh
# first (it counts memory as well as cores) and pass what it tells you:
#
#   bash tests/slurm_scripts/where_to_submit.sh --fits 16 96
#
# Account: stat-cadd — its FairShare (0.053237) is higher than stat-ecr's
# (0.037705) as of 2026-09-03. Re-check before submitting; fair-share moves.
#
#   sbatch --account=stat-cadd --partition=short --array=0-3 \
#          tests/slurm_scripts/run_f30_importance_methods.sh baselines
#
#   sbatch --account=stat-cadd --partition=short --array=0-123%8 \
#          tests/slurm_scripts/run_f30_importance_methods.sh arms
#
# The 11:59 wall fits `short` (12h limit) and backfills. Three arms are expected to
# need longer than that and will hit the wall: kernelshap (MEASURED 96-118s on SIXTY
# features), boruta_hits and rf_mdi. Re-run just those on medium (2-day limit) with
# a longer --time and --arms naming them, once the short run shows which timed out.
#
# Then, locally or on the login node once the files are back:
#   python -u tests/experiments/framework/f30_analyse.py
# Resources sized against the cluster state of 2026-09-03: 8 GB/core, so 16 cores
# allows up to 128G before a task starts waiting for a whole node. 96G stays under
# that ceiling deliberately, so these tasks backfill instead of queueing behind the
# 25 pending 128G jobs already in the queue.
#SBATCH --job-name=f30_importance
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

STAGE="${1:-arms}"

DATASETS=(caco2 buchwald_hartwig flab_expression_koenig tcga_brca_pam50)

# Must match f30_scorers.SCORERS. Checked at runtime below, so a drift fails the
# task loudly instead of silently running the wrong arm.
ARMS=(
  lightgbm_gain lightgbm_gain_nocap lightgbm_gain_deep treeshap kernelshap
  gradientshap deeplift integrated_gradients permutation
  catboost_importance rf_mdi treeshap_interventional
  f_test spearman_marginal mutual_info
  mrmr jmi cmim hsic_lasso dcor relieff multisurf
  conditional_permutation boruta_hits stability_selection
  lasso elastic_net matching_pursuit
  variance laplacian random
)

# Fail loudly if this list has drifted from f30_scorers.SCORERS.
python - "${ARMS[@]}" <<'PYCHK'
import sys, os
sys.path.insert(0, os.path.join("tests", "experiments", "framework"))
import f30_scorers as S
here, there = set(sys.argv[1:]), set(S.SCORERS)
if here != there:
    raise SystemExit(
        f"ARMS in the SLURM script has drifted from f30_scorers.SCORERS.\n"
        f"  only in script:  {sorted(here - there)}\n"
        f"  only in scorers: {sorted(there - here)}")
print(f"  [arms] {len(here)} arms, script and registry agree")
PYCHK

outdir="${KIRBY_CACHE_DIR}/f30_importance_methods"
mkdir -p "${outdir}"
export F30_OUT="${outdir}"
# macOS-only guard, harmless here; keeps local and cluster runs identical.
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"

DRIVER=tests/experiments/framework/f30_importance_methods.py

if [ "${STAGE}" = "baselines" ]; then
  ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
  echo "f30 baselines task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}"
  echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR}  F30_OUT=${F30_OUT}"
  python -u "${DRIVER}" "${ds}" --stage baselines --n-jobs "${SLURM_CPUS_PER_TASK:-1}"
else
  narms=${#ARMS[@]}
  ds_idx=$(( SLURM_ARRAY_TASK_ID / narms ))
  a_idx=$(( SLURM_ARRAY_TASK_ID % narms ))
  ds="${DATASETS[$ds_idx]}"
  arm="${ARMS[$a_idx]}"
  echo "f30 arms task ${SLURM_ARRAY_TASK_ID} -> ds=${ds} arm=${arm} (${narms} arms)"
  echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR}  F30_OUT=${F30_OUT}"
  if [ ! -f "${outdir}/${ds}__baselines.json" ]; then
    echo "ERROR: ${outdir}/${ds}__baselines.json missing — run the baselines stage first" >&2
    exit 2
  fi
  python -u "${DRIVER}" "${ds}" --stage arms --arms "${arm}" \
         --n-jobs "${SLURM_CPUS_PER_TASK:-1}"
fi
