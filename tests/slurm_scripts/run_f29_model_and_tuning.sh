#!/bin/bash
# f29 — which models show signal on the BASE representations, and does
# hyperparameter tuning pay per-representation or per-model?
#
# No hybrid, no allocation, no budget. Each dataset's 10 general representations
# from DATASET_REPS go into each model whole, 5 group-aware outer folds. Nothing
# in result_db, searches/ or the hybrid caches is read or written, so there is
# NOTHING TO DELETE before or after this job on any dataset.
#
# THREE ARMS
#   default          the DEFAULT_MODELS parameters, untouched.
#   tuned_per_rep    20 draws from find_best_hybrid._HP_SEARCH_SPACES, 3 inner
#                    folds, inside each outer TRAIN fold, on that
#                    representation's own features. One parameter set per
#                    (representation, model).
#   tuned_per_model  the same search run once per fold on all 10
#                    representations concatenated, then reused for every
#                    representation. One parameter set per model.
# Tuning never sees the outer test fold, so the reported score stays honest.
#
# TWO MODELS CANNOT BE TUNED. _HP_SEARCH_SPACES has no entry for dnn or cnn —
# the engine's comment says RandomizedSearchCV is too slow for them, and they
# carry internal early stopping instead. They therefore run the default arm
# only, and the tuning question is answered for eight models, not ten. That is
# an undefinedness cut, not a compute cut.
#
# RIDGE'S SEARCH SPACE IS NEW. The engine defines none (its ridge is
# nucleotide-only). f29 carries an alpha grid of its own; see RIDGE_SEARCH_SPACE
# in the python file.
#
# GP RUNS ON ONE DATASET OUT OF FIVE. DEFAULT_MODELS['gp']['_max_samples'] is
# 1000 and get_model skips above it. tcga_brca_pam50 is n=821; caco2 2158,
# buchwald_hartwig 3955, flab_expression_koenig 4275, logd 5028. Raising that
# cap is a code change in find_best_hybrid, not a flag here.
#
# TORCH AND LIGHTGBM MUST NOT SHARE A PROCESS. Measured on the local machine:
# one DNN fit after LightGBM and CatBoost had run in the same process sat at 0%
# CPU for 22 minutes; alone in a fresh process the same fit took 44 seconds.
# Every array task is its own process and dnn tasks name no other model, so the
# two runtimes never meet. Do not merge dnn into a multi-model task.
#
# WALL TIME. Derived from a measured one-fold probe (seconds per fit at the
# narrowest and widest representation of each dataset, fitted linearly in the
# feature count and summed over the 10 representations). Worst default task is
# buchwald_hartwig catboost at 17 min per fold; worst tuned task is the same
# cell at 8.9 h per fold (20 draws x 3 inner folds on ~53% of the rows). Both
# fit 11:59. These are DERIVED, not measured end-to-end — read MaxRSS and
# Elapsed off the logs and resize the next submit from those:
#     sacct -j <jobid> --format=JobID,JobName,MaxRSS,Elapsed,State
#
# CNN IS NOT IN THIS ARRAY. Its cost is 13.6 h per fold on
# flab_expression_koenig alone, past short's limit, and its one-fold probe
# returns the label standard deviation (logd RMSE 1.1752 against std 1.1753)
# and the same score at 50D and 1152D on tcga_brca_pam50. Submit it separately
# if you want the five-fold confirmation:
#     sbatch --partition=medium --array=0-24 --time=47:59:00 \
#         --export=ALL,F29_MODELS=cnn,F29_ARMS=default \
#         tests/slurm_scripts/run_f29_model_and_tuning.sh
#
# NEVER PASS --account. SLURM resolves the default association.
#
# SUBMIT (confirm the shape fits first — where_to_submit counts memory now):
#   bash tests/slurm_scripts/where_to_submit.sh --fits 8 32G
#   sbatch --partition=short --array=0-624 tests/slurm_scripts/run_f29_model_and_tuning.sh
#
# RESUBMIT just the tasks that did not finish:
#   sbatch --partition=short --array=<ids> tests/slurm_scripts/run_f29_model_and_tuning.sh
#SBATCH --job-name=f29_models
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

DATASETS=(
  buchwald_hartwig          # reaction, RMSE, leave-additive-out, n=3955
  caco2                     # molecular ADMET, RMSE, scaffold, n=2158
  flab_expression_koenig    # antibody, Spearman, identity, n=4275
  tcga_brca_pam50           # TCGA multiclass, f1_weighted, TSS, n=821
  logd                      # molecular ADMET, RMSE, scaffold, n=5028
)

# Eight tunable models x three arms, plus dnn on the default arm only.
# F29_MODELS / F29_ARMS override the pair lists for the CNN submit above.
if [ -n "${F29_MODELS:-}" ]; then
  PAIRS=()
  for m in ${F29_MODELS//,/ }; do
    for a in ${F29_ARMS:-default}; do PAIRS+=("${m}:${a}"); done
  done
else
  PAIRS=(
    ridge:default    ridge:tuned_per_rep    ridge:tuned_per_model
    rf:default       rf:tuned_per_rep       rf:tuned_per_model
    lightgbm:default lightgbm:tuned_per_rep lightgbm:tuned_per_model
    catboost:default catboost:tuned_per_rep catboost:tuned_per_model
    xgboost:default  xgboost:tuned_per_rep  xgboost:tuned_per_model
    mlp:default      mlp:tuned_per_rep      mlp:tuned_per_model
    gp:default       gp:tuned_per_rep       gp:tuned_per_model
    knn:default      knn:tuned_per_rep      knn:tuned_per_model
    dnn:default
  )
fi

N_FOLDS=5
n_ds=${#DATASETS[@]}
n_pair=${#PAIRS[@]}
total=$(( n_ds * n_pair * N_FOLDS ))

id="${SLURM_ARRAY_TASK_ID:?set --array=0-$(( total - 1 ))}"
if [ "$id" -ge "$total" ]; then
  echo "task ${id} is past the last cell; the array is 0-$(( total - 1 ))" >&2
  exit 1
fi

# Consecutive ids land on different datasets, so no two neighbouring tasks
# parse the same loader at the same moment.
ds="${DATASETS[$(( id % n_ds ))]}"
rest=$(( id / n_ds ))
pair="${PAIRS[$(( rest % n_pair ))]}"
fold=$(( rest / n_pair ))
model="${pair%%:*}"
arm="${pair##*:}"
tag="${model}_${arm}_f${fold}"

echo "f29 task ${id} -> ds=${ds} model=${model} arm=${arm} fold=${fold}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

TIME_V=""
[ -x /usr/bin/time ] && TIME_V="/usr/bin/time -v" || true

${TIME_V} python -u tests/experiments/framework/f29_model_and_tuning_survey.py \
    --datasets "${ds}" --models "${model}" --arms "${arm}" \
    --folds "${N_FOLDS}" --fold "${fold}" --tag "${tag}"
