#!/bin/bash
# F28 — every objective on its own, then every blend of them.
#
# The search maximises ONE number and which number is a free choice. Three are
# available: a fitted model's held-out score; each molecule's property predicted
# from its five nearest neighbours; and MCS structural fidelity. F28 runs all
# three on their own and every mixture of them in steps of 0.25 — 15 cells per
# dataset, through nested_cv_evaluate, per dataset x model x budget.
#
# ALL THREE ON ONE SCALE. A weighted mean needs its parts on one scale, and the
# dataset's own metric is not one: RMSE on seven of the eight molecular sets
# against two Spearman correlations. So INSIDE A BLEND ONLY, prediction is
# scored as a correlation between predicted and true on the same held-out rows —
# Spearman for regression, 2*(AUROC - 0.5) for classification. The plain
# predictive objective is untouched. Both numbers are recorded for every winner.
#
# WHAT MAKES IT AFFORDABLE, AND WHAT THAT COSTS
#   --max-reps 3   runs top_2 and top_3 only, so a hybrid draws on at most 3
#                  representations instead of all 10. top_3 takes the three
#                  BEST-SCORING ones; it does not try all 120 trios. A cap, not
#                  an enumeration.
#   --outer-folds 3 instead of 5.
# Together 4.5x fewer allocation methods and 1.7x fewer outer folds = 7.5x off a
# default cell. Everything else is identical to find_best_hybrid.
#
# ★ The cap is a real restriction on the answer. An objective whose advantage
# only appears when it can combine six representations cannot show it here.
# A null result from F28 means "not within 3 representations", not "never".
#
# ONE POOL — the dataset's signed-off 'general' pool. The representation-pool
# axis is what F27 is for.
#
# ONE ARRAY TASK PER DATASET. Each task writes its own file
# tests/experiments/results/f28_objective_weight_space_<ds>.json — no write
# races. Eight datasets, 15 cells each.
#
# MCS PAIRS. Ten of the fifteen cells contain MCS, so every task needs them. The
# first task to touch a dataset builds them into $KIRBY_CACHE_DIR/mcs_pairs and
# every later run reads them. Measured 0.024 s/pair on arc, so about 3 minutes
# for the default 8000, once per dataset.
#
# ★ The .npz files in tests/experiments/similarity_metrics/ are the JULY study's
# row ordering and MUST NOT be used here. Row 900 is a different molecule.
# get_mcs_pairs refuses them by hash; nothing needs deleting.
#
# ★ MCS fidelity is NaN when fewer than 10 sampled pairs have BOTH molecules
# inside the fold being scored, and a fold holds about n_pairs x (its fraction
# of the dataset) squared. At 3 outer folds an inner validation fold is about
# 13% of the set, so 8000 pairs leaves roughly 140 — comfortable. Drop the pair
# count and the MCS cells stop running; the engine raises rather than reporting
# a winner picked by nothing.
#
# CACHE. Runs under its own tier 'f28_objweight', and every cell gets its own
# condition_suffix, so no cell's search file evicts another's. The
# per-(representation, model) baseline files are shared across the fifteen cells
# — a single representation's baseline does not depend on the objective.
# Touches no existing result DB. Nothing needs deleting before or after.
#
# WALL TIME. 15 cells at roughly 28-88 minutes each is 7-22 h per dataset. That
# is a structural estimate from configuration counts, NOT a measured time; the
# smoke below sets the real one. The result file is rewritten after EVERY cell
# and 'progress' says how many ran, so a killed task loses one cell. Resubmit
# with --cells naming the ones that did not finish.
#
# NEVER PASS --account. 123 of the 125 scripts in this directory omit it and let
# SLURM resolve the default association. Adding it cost three failed
# submissions on 2026-07-11 and moved nothing.
#
# WHICH CLUSTER is set by the LOGIN NODE, not by --partition. arc-login sends
# everything to arc (multi-node parallel); htc is the high-throughput,
# low-core-count cluster. This job is 16 CPUs, CPU only, no GPU, eight
# independent tasks — that is an htc job by ARC's own division of labour.
# where_to_submit.sh reports the CURRENT cluster only and never surfaces this
# choice. To run it on htc, submit from htc-login (or add --clusters=htc).
#
# --partition is LIVE STATE. Confirm with where_to_submit.sh, then:
#   sbatch --partition=<part> --array=0-7 \
#          tests/slurm_scripts/run_f28_objective_weight_space.sh
#
# SEE THE GRID WITHOUT RUNNING ANYTHING — loads no data, no search:
#   python tests/experiments/framework/f28_objective_weight_space.py \
#       --dataset herg_ki --list-grid
#
# BEFORE QUEUEING EIGHT TASKS, prove it runs on real data. devel has a
# 10-minute limit and is usually empty. setup.sh MUST be sourced first —
# without it KIRBY_CACHE_DIR is unset, so the representation cache resolves to
# the home quota, every representation is recomputed from SMILES, and the venv
# at /data/stat-cadd/scat9264/py311-kirby is never activated.
#
#   srun -p devel -c 8 --mem=32G --time=10:00 --pty /bin/bash
#   # then, from the repo root:
#   export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
#   python -u tests/experiments/framework/f28_objective_weight_space.py \
#       --dataset herg_ki --smoke --outer-folds 2 --models rf \
#       --mcs-pairs 4000 --cells predict,neighbour,MCS
#SBATCH --job-name=f28_objweight
#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# The eight molecular datasets. Ten of the fifteen cells score MCS fidelity,
# which needs SMILES, so no non-molecular dataset can run this grid. That is a
# definedness cut, not a compute cut.
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

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "f28 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

python -u tests/experiments/framework/f28_objective_weight_space.py \
    --dataset "${ds}" --mode full --max-reps 3 --outer-folds 3
