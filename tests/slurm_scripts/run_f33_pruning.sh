#!/bin/bash
# f33 — record every configuration on every fold, unpruned, so the two prunes can
# be replayed off the record instead of re-fitted.
#
# WHY. FRAMEWORK_FINALIZATION.md 3.4 leaves two answers open: whether the
# end-of-fold-0 allocation prune stays on by default, and whether a two-fold
# version gets built. Every pruning number in circulation was fitted in April by
# f15/f16 on 13 allocation methods, of which two survive, and f16's own
# calibration file records datasets_evaluated: ["caco2"]. The brief is
# PRUNING_RECALIBRATION_HANDOFF.md.
#
# ONE TASK = ONE DATASET = ONE FULL GRID. 5 allocation methods x 3 importance
# methods x 5 budgets x 17 (model, internal evaluator) pairs = 1275
# configurations, each scored on 5 folds. nt_promoter_tata drops xgboost under
# the nucleotide policy in tests/experiments/dataset_policy.py, so it is 13 pairs
# and 975 configurations.
#
# NOTHING IS PRUNED. adaptive_pruning=False. That is the point: the replay needs
# every allocation method on every fold.
#
# BUDGETS 25, 50, 100, 150, 300. The default search runs [50, 150, 300]. The
# other two are the aggressive_pruning checkpoints in tests/experiments/_shared.py,
# and a checkpoint fires only on an exact budget match, so without them two thirds
# of that schedule can never be tested. The replay restricts to any subset.
#
# WALL TIME, MEASURED LOCALLY on an 8-core Mac (contended), 2026-09-10/11:
#   logd            5028 molecules  15.86 h
#   flab_thermo     672  molecules   4.1 h
#   humanness       564  molecules   3.0 h
#   tcga_brca_pam50 821  molecules   2.78 h
# One allocation method dominates: wglobal_power2 with treeshap was 8.80 h of
# logd's 14.45 h of fitting. Budget for the large sets accordingly and read the
# real numbers off the logs before any resubmit:
#     sacct -j <jobid> --format=JobID,JobName,MaxRSS,Elapsed,State
#
# MEMORY IS DERIVED, NOT MEASURED. buchwald_hartwig's pool is 52,404 features
# across 3,955 molecules. 32G at 8 cores matches the usual node ratio; read
# MaxRSS off the first run and resize.
#
# TORCH AND LIGHTGBM SHARE THIS PROCESS. dnn and lightgbm are both downstream
# models and both internal evaluators here, so they cannot be separated into
# different tasks. The python file sets KIRBY_LGBM_THREADS=1 and OMP_NUM_THREADS=1
# at import, which is what f31 and f32 ran under without the 22-minute deadlock
# f29 recorded. It also sets USE_TF=0 and TRANSFORMERS_NO_TF=1: without those,
# treeshap importance failed on all five folds of logd because shap imports
# transformers, transformers imports its TensorFlow half, and that half does not
# survive the protobuf version TensorFlow has already loaded for deeplift.
#
# NEVER PASS --account. SLURM resolves the default association.
#
# CHECK THE POOLS ARE PRESENT BEFORE SUBMITTING — NOT ON THE LOGIN NODE. The
# check loads all nine datasets, and the login node has neither the memory nor
# the CUDA libraries for that. Run on 2026-09-12 from arc-login02 it returned
# "libcufft.so.11: failed to map segment from shared object" on the molecular
# sets and "Memory mapping file failed: Cannot allocate memory" on
# nt_promoter_tata. Neither says anything about the pools. Run it in a job:
#   srun --partition=devel --cpus-per-task=8 --mem=32G --time=00:20:00 \
#       python tests/experiments/framework/f33_pruning_recalibration.py --check
#
# SUBMIT:
#   bash tests/slurm_scripts/where_to_submit.sh --fits 8 32G
#   sbatch --partition=medium --array=0-8 tests/slurm_scripts/run_f33_pruning.sh
#
# RESUBMIT just the datasets that did not finish:
#   sbatch --partition=medium --array=<ids> tests/slurm_scripts/run_f33_pruning.sh
#
# READ THE RESULTS:
#   python tests/experiments/framework/f33_analyse.py
#   python tests/experiments/framework/f33_schedule_calibration.py
#   python tests/experiments/framework/f33_schedule_calibration.py --vote-off
#SBATCH --job-name=f33_pruning
#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# Fail loudly rather than build every representation from scratch into whatever
# ~/.cache/kirby resolves to on a compute node.
: "${KIRBY_CACHE_DIR:?KIRBY_CACHE_DIR is unset after setup.sh — set it to the server cache root}"

DATASETS=(
  humanness_therapeutic     # antibody,   auc,          564 molecules
  flab_thermo_tresanco      # antibody,   spearman,     672 molecules
  tcga_brca_pam50           # tcga,       f1_weighted,  821 molecules
  herg_ki                   # molecular,  rmse,       1,412 molecules
  caco2                     # molecular,  rmse,       2,158 molecules
  buchwald_hartwig          # reaction,   rmse,       3,955 molecules
  flab_expression_koenig    # antibody,   spearman,   4,275 molecules
  logd                      # molecular,  rmse,       5,028 molecules
  nt_promoter_tata          # nucleotide, mcc,        5,274 molecules, 9 representations
)

total=${#DATASETS[@]}
id="${SLURM_ARRAY_TASK_ID:?set --array=0-$(( total - 1 ))}"
if [ "$id" -ge "$total" ]; then
  echo "task ${id} is past the last dataset; the array is 0-$(( total - 1 ))" >&2
  exit 1
fi
ds="${DATASETS[$id]}"

echo "f33 task ${id} -> ${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR}"

TIME_V=""
[ -x /usr/bin/time ] && TIME_V="/usr/bin/time -v" || true

${TIME_V} python -u tests/experiments/framework/f33_pruning_recalibration.py \
    --datasets "${ds}"
