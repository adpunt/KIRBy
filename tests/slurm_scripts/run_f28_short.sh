#!/bin/bash
# F28 on the SHORT partition — five CELLS per array task instead of a whole dataset.
#
# WHY THIS EXISTS. run_f28_objective_weight_space.sh asks 16 cores, 256 GB and
# 48 h per task. Job 12923011 sat PENDING for a week on that shape. Both arc
# node types are 8 GB per core (arc-c046 48c/384G, arc-c312 288c/2266G), so
# 256 GB is 33 cores' worth of memory: one task per node, eight nodes held for
# two days. Check any shape before submitting it:
#     bash tests/slurm_scripts/where_to_submit.sh --fits 16 128G
#
# WHAT CHANGED, and nothing else did. Same experiment, same 8
# datasets, same --max-reps 3 and --outer-folds 3. No axis is cut.
#   wall    47:59 -> 11:59   short's limit, and a short wall backfills into
#                            gaps a 48 h request can never fit.
#   memory  256G  -> 128G    16 cores x 8 GB/core, so the task stops reserving
#                            cores it will not use. NOT a measurement — no run
#                            of f28 has ever reported its peak memory. If tasks
#                            come back OOM-killed, that is the measurement.
#   tasks   8     -> 32      one (dataset, 4-cell group) each, 4 x 8.
#
# INDEXING. ds = ID % 8, cell group = ID / 8. So consecutive tasks are on
# DIFFERENT datasets, and the first task to want MCS pairs for a dataset is
# alone in doing so. get_mcs_pairs now writes its cache through a temp file and
# renames, so a second task arriving mid-write reads the old file or the new
# one, never half of one.
#
# OUTPUT. Each task writes its own file via --tag:
#   tests/experiments/results/f28_objective_weight_space_<ds>__g<NN>.json
# read_f28_objective_weight_space.py merges them back into one table per
# dataset, so nothing downstream changes.
#
# NEVER PASS --account. SLURM resolves the default association.
#
# SUBMIT (confirm the partition is live first):
#   bash tests/slurm_scripts/where_to_submit.sh
#   sbatch --partition=short --array=0-31 tests/slurm_scripts/run_f28_short.sh
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

# Ten of the sixteen cells score MCS fidelity, which needs SMILES, so only
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

# The 16 cells, in the order f28_objective_weight_space.build_grid() defines
# them. Confirm with:  python .../f28_objective_weight_space.py --list-grid
# Order matters only for which cells share a task. Every task calls
# get_mcs_pairs for its dataset before the grid starts, whether or not its
# cells need MCS -- so the pair cache is built on the first task to reach a
# dataset either way.
#
# predict_corr was added 2026-09-03. 'predict' maximises the dataset's own
# metric; every cell that mixes prediction with anything maximises prediction
# as a CORRELATION. So the prediction corner was not the limit of its own edge.
# predict_corr is that corner on the mixtures' scale. Both are kept.
CELLS=(
  "predict"
  "predict_corr"
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

# How many of the 16 cells one task runs. This is the ONLY knob that trades
# queue entries against wall time per task, and it cuts nothing:
#   4 cells -> 32 tasks, 1.9-5.9 h each   (the default; 4 even groups of 4)
#   3 cells -> 48 tasks, 1.4-4.4 h each   (starts sooner, more entries)
#   5 cells -> 32 tasks, 2.3-7.3 h each   (same task count, last group holds 1)
#  16 cells ->  8 tasks, 7.5-23 h each    (does NOT fit short's 12 h limit)
# Those hours are 28-88 min per cell, which is arithmetic from configuration
# counts and has never been measured. Results are written after EVERY cell, so
# a task killed at the wall loses only the cell it was inside.
#
# Was 5 while the grid held 15 cells. At 16 cells a group of 5 leaves the last
# group holding one cell for the same 32 array entries, so 4 is the even split.
CELLS_PER_TASK="${CELLS_PER_TASK:-4}"
n_grp=$(( (n_cell + CELLS_PER_TASK - 1) / CELLS_PER_TASK ))
total=$(( n_ds * n_grp ))

id="${SLURM_ARRAY_TASK_ID:?set --array=0-$(( total - 1 ))}"
if [ "$id" -ge "$total" ]; then
  echo "task ${id} is past the last (dataset, cell-group) pair; the array is 0-$(( total - 1 ))" >&2
  exit 1
fi

ds="${DATASETS[$(( id % n_ds ))]}"
grp=$(( id / n_ds ))
first=$(( grp * CELLS_PER_TASK ))

cells=""
for (( k = first; k < first + CELLS_PER_TASK && k < n_cell; k++ )); do
  cells="${cells:+$cells,}${CELLS[$k]}"
done
tag=$(printf 'g%02d' "$grp")

echo "f28 task ${id} -> ds=${ds} group=${tag} cells=${cells}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

# Report peak memory into the log, so the NEXT sizing is measured instead of
# derived from the node ratio. 128G below is 16 cores x 8 GB/core, not a
# measurement -- no f28 run has ever reported what a cell actually uses.
# sacct gives the same number afterwards:
#   sacct -j <jobid> --format=JobID,MaxRSS,Elapsed,State
TIME_V=""
[ -x /usr/bin/time ] && TIME_V="/usr/bin/time -v" || true

${TIME_V} python -u tests/experiments/framework/f28_objective_weight_space.py \
    --dataset "${ds}" --mode full --max-reps 3 --outer-folds 3 \
    --cells "${cells}" --tag "${tag}"
