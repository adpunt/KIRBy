#!/bin/bash
# F24 — the augmentation admission grid. Which way of admitting extra features
# works, per dataset. Five methods x donor source x per-donor cap x candidate
# ranking x size rule x margin, with a matched random control arm and two
# planted decoy donors on every configuration.
#
# SPLIT BY DATASET *AND* BY PLACEMENT-AND-BUDGET, never by axis.
# 14 datasets x 4 placement-and-budget combinations = 56 array tasks.
#
#   task % 4 == 0  extras onto the best single representation at its full width
#   task % 4 == 1  extras onto the multi-representation hybrid, budget 500
#   task % 4 == 2  same, budget 1000
#   task % 4 == 3  same, budget 2000
#
# WHY THE SPLIT. Counted from the engine with nothing fitted (--count-only):
# herg_ki is 2,260 configurations and 446,760 model fits, tresanco 2,164 and
# 374,040. The only timing so far is 10.82 s per fit, measured on tresanco with
# rf on a LOCAL machine at load average 43 on 8 cores -- an upper bound, not the
# server's number. At that rate one dataset is 47 days as a single job and 11.7
# days per placement-and-budget, so even this split may not be enough. Run
# run_f24_augmentation_cost.sh FIRST and read the real seconds-per-fit off it.
#
# IF IT STILL OVERRUNS: split further within a dataset by admission method
# (--methods), which is five more sub-jobs. Do not drop a dataset, a budget, a
# donor source or an axis to make it fit.
#
# CACHE. f24 constructs recipes directly and calls create_hybrid. It runs no
# search and writes to no result_db, so nothing needs deleting before or after
# and no stored result can be evicted.
#
# NEVER PASS --account. 123 of the 125 scripts in this directory omit it and let
# SLURM resolve the default association. Adding it cost three failed
# submissions on 2026-07-11 and moved nothing.
#
# WHICH CLUSTER is set by the LOGIN NODE, not by --partition. This job is
# 16 CPUs, CPU only, no GPU, 24 independent tasks -- an htc job by ARC's own
# division of labour. Submit from htc-login, or add --clusters=htc.
#
# --partition is LIVE STATE. Confirm with where_to_submit.sh, then:
#   sbatch --partition=<part> --array=0-167%12 \
#          tests/slurm_scripts/run_f24_augmentation_grid.sh
#
# BEFORE QUEUEING, prove it runs on real data. devel has a 10-minute limit and
# is usually empty. setup.sh MUST be sourced first -- without it
# KIRBY_CACHE_DIR is unset, the representation cache resolves to the home quota,
# every representation is recomputed from SMILES, and the venv is never
# activated. Skip mode is what makes it fit in ten minutes.
#
#   srun -p devel -c 8 --mem=32G --time=10:00 --pty /bin/bash
#   export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
#   python -u tests/experiments/framework/f24_augmentation_admission.py \
#       --dataset flab_thermo_tresanco --smoke --placement single
#
# --smoke is one fold, one model, one seed on the REAL data. It is the
# cheapest thing that actually exercises the admission methods.
#
#SBATCH --job-name=f24_aug
#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# SIX datasets x 28 = 168 array tasks (--array=0-167).
DATASETS=(
  # SIX, cut from fourteen by the user on 2026-08-26 for time. That is the
  # reason. It is not a scientific one and must not be written up as one.
  #
  # What was kept, and why these six: two molecular, one reaction, three
  # antibody; RMSE and Spearman both present; flat-topped and dominant both
  # present. All four with representations on the local drive are here, so the
  # counts and timings can be checked without the cluster.
  herg_ki                 # molecular, RMSE       gap 0.0118  flat-topped
  logd                    # molecular, RMSE       gap 0.0018  flattest of all
  buchwald_hartwig        # reaction, RMSE        gap 0.0202
  flab_thermo_tresanco    # antibody, Spearman    gap 0.0167
  abagym_VEGF             # antibody DMS          gap 0.0965  DOMINANT
  abcov_kd                # antibody, n=141       gap 0.0632  DOMINANT
)
# DROPPED, and what that costs in coverage. caco2, flab_pk_jain_fcrnrt,
# flab_pk_jain_heprt, ginkgo_hic, tz_her2_binary, tcga_luad_vs_lusc,
# ginkgo_acsins, ddg_combined. Two whole domains go with them: TCGA and
# nucleotide are now unrepresented, and nucleotide was the expensive one at
# 18,000 to 33,000 rows. So this run cannot say whether an admission method
# that wins on molecules also wins on gene expression or on DNA.
# Gaps from <cache>/rep_prune/<ds>/scores.json, recomputed 2026-08-26.

# The nine allocation methods find_best_hybrid actually sweeps
# (DEFAULT_ALLOCATION_METHODS), not the 39 names in NAMED_ALLOCATIONS. Added
# 2026-08-27: "which admission method wins" is only useful if the answer
# survives the allocation methods the search uses, and until today this file
# ran one -- weighted_global -- so it could not have said.
#
# Allocation is inert on the single-representation placement: one
# representation takes 100% of the budget whatever the allocation says. Task 0
# therefore passes none and the experiment prints that it collapsed the axis.
ALLOCS=(top_2 top_3 top_5 performance_weighted entropy_1.0 power_2 rank_linear entropy_2.0 rank_exp_d0.5)
BUDGETS=(500 1000 2000)

# 28 tasks per dataset: 1 single-representation + 3 budgets x 9 allocations.
#
# WHY SPLIT THIS FINELY. Measured 2026-08-27 on flab_thermo_tresanco: one
# configuration takes 44.6s after the precompute fix, down from 336s. One
# dataset at one budget is 1,467 configurations x 3 folds x 2 models, so 109
# hours -- over the 48h wall. Split by allocation it is 12.1 hours per task.
# Split by dataset and by allocation, never by axis: every allocation still
# runs, on every dataset.
PER_DS=28
ds="${DATASETS[$(( SLURM_ARRAY_TASK_ID / PER_DS ))]}"
slot=$(( SLURM_ARRAY_TASK_ID % PER_DS ))

if [ "${slot}" -eq 0 ]; then
  # Allocation is inert with one representation: it takes 100% of the budget
  # whatever the allocation says. One task, no allocation sweep.
  ARGS="--placement single"
else
  i=$(( slot - 1 ))
  ARGS="--placement hybrid --hybrid-budgets ${BUDGETS[$(( i / 9 ))]} --allocations ${ALLOCS[$(( i % 9 ))]}"
fi

echo "f24 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}  ${ARGS}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

# What this task is about to cost, printed before it starts, so a job that dies
# on the wall clock still leaves the count in its log.
python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --count-only ${ARGS}

python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --mode full ${ARGS}

echo "=== ${ds} slot ${slot}: done"
