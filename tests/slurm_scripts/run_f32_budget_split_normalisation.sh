#!/bin/bash
# f32 — how a representation's score is normalised before it sizes its share of
# the budget.
#
# Varies ONLY the transform applied to the per-representation score vector before
# create_hybrid divides the budget by it. Held fixed: 5 group-aware folds, the
# pruned representation pool per dataset, importance lightgbm_gain, internal
# evaluator lightgbm, downstream model lightgbm, seed 42.
#
# Never calls find_best_hybrid. It goes through create_hybrid and passes the
# transformed score vector as `precomputed_alloc_scores`, so NO result_db and NO
# search or baseline cache is touched: there is NOTHING TO DELETE before or after
# this job, on any dataset.
#
# WHY IT IS CHEAP. Fitting the internal evaluator once per representation is the
# expensive part of a build and it does not depend on the arm, the weighting or
# the budget. create_hybrid hands those scores back RAW, so each task pays for
# ONE scoring pass and every cell in that task reuses it. Arms that end up
# selecting the same features share one downstream fit.
#
# ONE TASK PER (DATASET, FOLD) — 9 datasets x 5 folds = 45 tasks, one file each,
# no cross-task write races, and one dataset hitting the wall costs one fold.
#
# --account / --partition / --clusters are LIVE STATE. Run where_to_submit.sh
# first (it counts memory as well as cores) and pass what it tells you:
#
#   bash tests/slurm_scripts/where_to_submit.sh --fits 16 96
#
#   # 1. gate: the pools must exist and the metric pairs must still hold
#   bash tests/slurm_scripts/run_f32_budget_split_normalisation.sh check
#
#   # 2. time one fold before committing to the array
#   sbatch --account=stat-cadd --partition=short --array=0 \
#          tests/slurm_scripts/run_f32_budget_split_normalisation.sh grid
#
#   # 3. the grid
#   sbatch --account=stat-cadd --partition=short --array=0-44%12 \
#          tests/slurm_scripts/run_f32_budget_split_normalisation.sh grid
#
# The 11:59 wall fits `short` (12h limit) and backfills. gb_human_enhancers_cohn
# is the largest of the nine at 27,791 rows and is the one to watch for the wall.
# Resources sized against the cluster state of 2026-09-03: 8 GB/core, so 16 cores
# allows up to 128G before a task waits for a whole node. 96G stays under that
# ceiling deliberately.
#SBATCH --job-name=f32_budget_norm
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

STAGE="${1:-grid}"

# Two datasets per metric so "the metric decides" can be told apart from "the
# dataset decides", plus tcga_brca_pam50 alone for the f1_weighted null. This
# list must match DATASETS in the driver — checked below.
DATASETS=(
  herg_ki caco2
  flab_thermo_tresanco flab_expression_koenig
  humanness_therapeutic tcga_luad_vs_lusc
  nt_promoter_tata gb_human_enhancers_cohn
  tcga_brca_pam50
)
N_FOLDS=5

DRIVER=tests/experiments/framework/f32_budget_split_normalisation.py

# Fail loudly if this list has drifted from the driver's own.
python - "${DATASETS[@]}" <<'PYCHK'
import sys, os
sys.path.insert(0, os.path.join("tests", "experiments", "framework"))
import f32_budget_split_normalisation as F
here, there = set(sys.argv[1:]), set(F.DATASETS)
if here != there:
    raise SystemExit(
        f"DATASETS in the SLURM script has drifted from the driver.\n"
        f"  only in script: {sorted(here - there)}\n"
        f"  only in driver: {sorted(there - here)}")
print(f"  [datasets] {len(here)} datasets, script and driver agree")
PYCHK

outdir="${KIRBY_CACHE_DIR}/f32_budget_split_normalisation"
mkdir -p "${outdir}"
export F32_OUT="${outdir}"
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"

if [ "${STAGE}" = "check" ]; then
  # Prints which metric each dataset reports and which metrics have only one
  # dataset, then runs ONE fold of the first dataset with both controls. Run
  # this before the array: tcga_luad_vs_lusc and gb_human_enhancers_cohn are
  # the two whose representation pools were never staged locally, so this is
  # where a missing pool shows up.
  echo "f32 check: metric pairing, pools and controls"
  echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR}  F32_OUT=${F32_OUT}"
  python -u "${DRIVER}" --check --output "${F32_OUT}"
else
  nf="${N_FOLDS}"
  ds_idx=$(( SLURM_ARRAY_TASK_ID / nf ))
  fold=$(( SLURM_ARRAY_TASK_ID % nf ))
  ds="${DATASETS[$ds_idx]}"
  echo "f32 task ${SLURM_ARRAY_TASK_ID} -> ds=${ds} fold=${fold} (of ${nf})"
  echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR}  F32_OUT=${F32_OUT}"
  python -u "${DRIVER}" --datasets "${ds}" --fold "${fold}" \
         --folds "${nf}" --output "${F32_OUT}"
fi
