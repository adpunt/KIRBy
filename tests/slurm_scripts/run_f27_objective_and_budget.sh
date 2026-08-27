#!/bin/bash
# F27 — which objective builds the best hybrid, out of which representations,
# at which budget?
#
# Five objectives (predict; each molecule's property from its five nearest
# neighbours; MCS structural fidelity; and those last two blended at 0.75/0.25
# and 0.50/0.50) x three representation pools (the dataset's signed-off pool;
# the 11 structure-tuned sim_ recipes alone; all 21) x the full budget list,
# through nested_cv_evaluate. Reported per dataset x model x budget, never
# averaged across datasets.
#
# THIS REPLACES f19 AND f21. Both of those scored feature subsets written by
# hand inside the experiment, over five fingerprints — neither called
# create_hybrid, neither used an allocation method, neither used a sim_ recipe.
# They stay on disk as the example of what not to do. Do not resurrect their
# design.
#
# ONE ARRAY TASK PER (DATASET, POOL). Each task writes its own file
# tests/experiments/results/f27_objective_and_budget_<ds>__<pool>.json — no
# write races. 8 datasets x 3 pools = 24 tasks, five objectives each.
#
# MCS PAIRS. Every task needs them, and the first task to touch a dataset builds
# them, cached in $KIRBY_CACHE_DIR/mcs_pairs and free afterwards. MEASURED on
# herg_ki (n=1412) on arc-c295, 2026-08-27: 4000 pairs in 97 s, so 0.024 s/pair
# and about 3 minutes for the default 8000. The 0.28 s/pair this comment used to
# claim was never measured and is eleven times too slow. The three tasks for one
# dataset will race to build the same file the first time; that costs a few
# duplicated minutes, not a wrong answer, because get_mcs_pairs is keyed by a
# content hash of that dataset's own SMILES.
#
# ★ The .npz files in tests/experiments/similarity_metrics/ are the JULY study's
# row ordering and MUST NOT be used here. Row 900 is a different molecule.
# get_mcs_pairs refuses them by hash; nothing needs deleting.
#
# CACHE. Runs under its own tier 'f27_objbud', and every cell gets its own
# condition_suffix (objective + pool), so no cell's search file evicts another's.
# The per-(representation, model) baseline files are shared across the five
# objectives within one pool — a single representation's baseline does not
# depend on the objective. Touches no existing result DB. Nothing needs deleting
# before or after.
#
# WALL TIME. Five objectives per task. f26's structural estimate for one
# nested_cv_evaluate is 3.5-11h, so 17-55h per task — a task at the top of that
# range will NOT finish inside the house 47:59:00 wall. That is survivable: the
# result file is rewritten after EVERY cell, and 'progress' inside it says how
# many of the five ran. A killed task loses its last cell, not the run. Resubmit
# with --objectives naming the ones that did not finish. Wall and memory are the
# house defaults; do not tune them per job.
#
# NEVER PASS --account. 123 of the 125 scripts in this directory omit it and let
# SLURM resolve the default association. Adding it cost three failed
# submissions on 2026-07-11 and moved nothing.
#
# WHICH CLUSTER is set by the LOGIN NODE, not by --partition. arc-login sends
# everything to arc (multi-node parallel); htc is the high-throughput,
# low-core-count cluster. This job is 16 CPUs, CPU only, no GPU, 24 independent
# tasks — that is an htc job by ARC's own division of labour.
# where_to_submit.sh reports the CURRENT cluster only and never surfaces this
# choice. To run it on htc, submit from htc-login (or add --clusters=htc).
#
# --partition is LIVE STATE. Confirm with where_to_submit.sh, then:
#   sbatch --partition=<part> --array=0-23%12 \
#          tests/slurm_scripts/run_f27_objective_and_budget.sh
#
# BEFORE QUEUEING 24 LONG TASKS, prove it runs on real data. devel has a
# 10-minute limit and is usually empty. setup.sh MUST be sourced first —
# without it KIRBY_CACHE_DIR is unset, so the representation cache resolves to
# the home quota, every representation is recomputed from SMILES, and the venv
# at /data/stat-cadd/scat9264/py311-kirby is never activated.
#
#   srun -p devel -c 8 --mem=32G --time=10:00 --pty /bin/bash
#   # then, from the repo root:
#   export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
#   python -u tests/experiments/framework/f27_objective_and_budget.py \
#       --dataset herg_ki --smoke --outer-folds 2 --models rf \
#       --mcs-pairs 4000
#
# --mcs-pairs 4000, not the 400 a smoke would otherwise want: MCS fidelity is
# NaN when fewer than 10 sampled pairs have BOTH molecules inside the fold being
# scored, and an inner validation fold holds about n_pairs x (its fraction of
# the dataset) squared. At --outer-folds 2 that fraction is about 0.1, so 400
# pairs leaves ~4 and every struct_fid cell refuses to run. The engine now says
# so instead of reporting a winner picked by nothing.
#SBATCH --job-name=f27_objbud
#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# The eight molecular datasets that have a 'similarity' recipe list registered
# in complexity_helpers.DATASET_REPS. The sim_ recipes need SMILES, 3D
# conformers and MCS pairs, so no non-molecular dataset can run the recipe pools
# and none is listed here. That is a definedness cut, not a compute cut.
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

# 'general' first so that row of the array builds each dataset's MCS pairs once
# and the other two rows read them from cache.
POOLS=(general recipes both)

n_ds=${#DATASETS[@]}
pool_i=$(( SLURM_ARRAY_TASK_ID / n_ds ))
ds_i=$(( SLURM_ARRAY_TASK_ID % n_ds ))
ds="${DATASETS[$ds_i]}"
pool="${POOLS[$pool_i]}"

echo "f27 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds} pool=${pool}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

python -u tests/experiments/framework/f27_objective_and_budget.py \
    --dataset "${ds}" --pools "${pool}" --mode full
