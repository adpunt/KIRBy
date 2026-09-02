#!/bin/bash
# F28 on the SHORT partition — one CELL per array task instead of one dataset.
#
# WHY THIS EXISTS. run_f28_objective_weight_space.sh asks 16 cores, 256 GB and
# 48 h per task. Job 12923011 sat PENDING for a week on that shape. Both arc
# node types are 8 GB per core (arc-c046 48c/384G, arc-c312 288c/2266G), so
# 256 GB is 33 cores' worth of memory: one task per node, eight nodes held for
# two days. Check any shape before submitting it:
#     bash tests/slurm_scripts/where_to_submit.sh --fits 16 128G
#
# WHAT CHANGED, and nothing else did. Same experiment, same 15 cells, same 8
# datasets, same --max-reps 3 and --outer-folds 3. No axis is cut.
#   wall    47:59 -> 11:59   short's limit, and a short wall backfills into
#                            gaps a 48 h request can never fit.
#   memory  256G  -> 128G    16 cores x 8 GB/core, so the task stops reserving
#                            cores it will not use. NOT a measurement — no run
#                            of f28 has ever reported its peak memory. If tasks
#                            come back OOM-killed, that is the measurement.
#   tasks   8     -> 120     one (dataset, cell) each, 15 x 8.
#
# INDEXING. ds = ID % 8, cell = ID / 8. So consecutive tasks are on DIFFERENT
# datasets, and the first task to want MCS pairs for a dataset is alone in
# doing so. get_mcs_pairs now writes its cache through a temp file and renames,
# so a second task arriving mid-write reads the old file or the new one, never
# half of one.
#
# OUTPUT. Each task writes its own file via --tag:
#   tests/experiments/results/f28_objective_weight_space_<ds>__c<NN>.json
# read_f28_objective_weight_space.py merges them back into one table per
# dataset, so nothing downstream changes.
#
# NEVER PASS --account. SLURM resolves the default association.
#
# SUBMIT (confirm the partition is live first):
#   bash tests/slurm_scripts/where_to_submit.sh
#   sbatch --partition=short --array=0-119%20 tests/slurm_scripts/run_f28_short.sh
#
# RESUBMIT just the tasks that did not finish:
#   sbatch --partition=short --array=<ids> tests/slurm_scripts/run_f28_short.sh
#SBATCH --job-name=f28_short
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# Ten of the fifteen cells score MCS fidelity, which needs SMILES, so only
# molecular datasets can run this grid. A definedness cut, not a compute cut.
DATASETS=(
  herg          # molecular classification (FLuID, Lhasa), AUROC
  herg_ki       # molecular, RMSE
  caco2         # molecular ADMET, RMSE
  logd          # molecular ADMET, RMSE
  qm9           # molecular quantum, RMSE — loader is the known risk here
  adme_hlm      # molecular ADMET, RMSE
  adme_rlm      # molecular ADMET, RMSE
  adme_mdr1     # molecular ADMET, RMSE
)

# The 15 cells, in the order f28_objective_weight_space.build_grid() defines
# them. Confirm with:  python .../f28_objective_weight_space.py --list-grid
# Cells 0 and 1 need no MCS pairs, so the array's first pass over the datasets
# runs before any MCS cache is touched.
CELLS=(
  "predict"
  "neighbour"
  "MCS"
  "predict.75+neighbour.25"
  "predict.75+MCS.25"
  "predict.50+neighbour.50"
  "predict.50+MCS.50"
  "predict.25+neighbour.75"
  "predict.25+MCS.75"
  "neighbour.75+MCS.25"
  "neighbour.50+MCS.50"
  "neighbour.25+MCS.75"
  "predict.50+neighbour.25+MCS.25"
  "predict.25+neighbour.50+MCS.25"
  "predict.25+neighbour.25+MCS.50"
)

n_ds=${#DATASETS[@]}
n_cell=${#CELLS[@]}
total=$(( n_ds * n_cell ))
id="${SLURM_ARRAY_TASK_ID:?set --array=0-$(( total - 1 ))}"
if [ "$id" -ge "$total" ]; then
  echo "task ${id} is past the last (dataset, cell) pair; the array is 0-$(( total - 1 ))" >&2
  exit 1
fi

ds="${DATASETS[$(( id % n_ds ))]}"
ci=$(( id / n_ds ))
cell="${CELLS[$ci]}"
tag=$(printf 'c%02d' "$ci")

echo "f28 task ${id} -> ds=${ds} cell=${cell} tag=${tag}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

python -u tests/experiments/framework/f28_objective_weight_space.py \
    --dataset "${ds}" --mode full --max-reps 3 --outer-folds 3 \
    --cells "${cell}" --tag "${tag}"
