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
#   sbatch --partition=short --array=0-15 tests/slurm_scripts/run_f28_short.sh
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

# The cells, READ FROM THE EXPERIMENT rather than copied here. A hardcoded list
# went stale the moment the grid grew from 15 rows to 16 (`predict_corr`), and a
# stale copy silently drops a row from the whole run. --list-cells loads no data
# and runs no search.
#
# Order matters only for which cells share a task. Every task calls
# get_mcs_pairs for its dataset before the grid starts, whether or not its cells
# need MCS, so the pair cache is built on the first task to reach a dataset
# either way.
# read, not mapfile: mapfile is bash 4+ and this has to be testable on a mac.
CELLS=()
while IFS= read -r _c; do
  [ -n "$_c" ] && CELLS+=("$_c")
done < <(python -u tests/experiments/framework/f28_objective_weight_space.py \
           --dataset "${DATASETS[0]}" --list-cells)
if [ "${#CELLS[@]}" -eq 0 ]; then
  echo "could not read the grid from f28_objective_weight_space.py --list-cells" >&2
  exit 1
fi
echo "grid has ${#CELLS[@]} cells"

n_ds=${#DATASETS[@]}
n_cell=${#CELLS[@]}

# How many of the cells one task runs. The ONLY knob that trades queue entries
# against wall time per task, and it cuts nothing.
#
# From the MEASURED cost. Job 12923011 timed a cell at 3.4-15 h under the old
# axes (herg 15.0, logd 9.7, herg_ki 5.5, adme_* ~5.3, caco2 3.4). The reduced
# axes above take roughly 19x off that, so 11-50 min per cell:
#   8 cells -> 16 tasks, 1.5-6.7 h each   (the default; comfortable in 11:59)
#  16 cells ->  8 tasks, 2.9-13.3 h each  (fewest entries; herg may hit the wall)
#   4 cells -> 32 tasks, 0.7-3.3 h each   (starts soonest, most entries)
# Results are written after EVERY cell, so a task killed at the wall loses only
# the cell it was inside. Resubmit those by name with --cells.
CELLS_PER_TASK="${CELLS_PER_TASK:-8}"
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
    --dataset "${ds}" --mode full --outer-folds 3 \
    --allocations power_2,softmax_T1 \
    --importance lightgbm_gain \
    --evaluators match,rf \
    --models rf,dnn \
    --cells "${cells}" --tag "${tag}"
