#!/bin/bash
# F26 — does penalising redundancy in the allocation step help, and which
# measurement of redundancy?
#
# Eleven allocation arms (no penalty on both bases, the existing entropy_*
# row-norm penalty, the new pairdist_* pairwise-distance penalty on both bases)
# x 2 objectives (predictive, and activity read-across) x the full budget list,
# through nested_cv_evaluate. Reports delta_mean_of_folds per dataset. Never
# averaged across datasets.
#
# ONE ARRAY TASK PER DATASET. Each task writes its own file
# tests/experiments/results/f26_redundancy_penalty_<ds>.json — no write races.
#
# CACHE. Runs under its own tier 'f26_redund', and every arm gets its own
# condition_suffix, so no arm's search file evicts another's. The
# per-(representation, model) baseline files are deliberately SHARED across the
# eleven arms — a single representation's baseline does not depend on the
# allocation method. Touches no existing result DB. Nothing needs deleting
# before or after.
#
# WALL TIME. Eleven arms is about 1.2x the config count of one default search
# (the default allocation list has 9 entries), times 2 objectives, times 5 outer
# folds. Estimated 8-27h per dataset from the config count — that is a
# structural estimate, not a measured time. Wall and memory are the house
# defaults (47:59:00 / 256G); do not tune them per job.
#
# NEVER PASS --account. 123 of the 125 scripts in this directory omit it and let
# SLURM resolve the default association. Adding it cost three failed
# submissions on 2026-07-11 and moved nothing.
#
# WHICH CLUSTER is set by the LOGIN NODE, not by --partition. arc-login sends
# everything to arc (multi-node parallel); htc is the high-throughput,
# low-core-count cluster. This job is 16 CPUs, CPU only, no GPU, ten
# independent tasks — that is an htc job by ARC's own division of labour.
# where_to_submit.sh reports the CURRENT cluster only and never surfaces this
# choice. To run it on htc, submit from htc-login (or add --clusters=htc).
#
# --partition is LIVE STATE. Confirm with where_to_submit.sh, then:
#   sbatch --partition=<part> --array=0-4%5 \
#          tests/slurm_scripts/run_f26_redundancy_penalty.sh
#
# BEFORE QUEUEING TEN LONG TASKS, prove it runs on real data. devel has a
# 10-minute limit and is usually empty. setup.sh MUST be sourced first —
# without it KIRBY_CACHE_DIR is unset, so the representation cache resolves to
# the home quota, every representation is recomputed from SMILES, and the venv
# at /data/stat-cadd/scat9264/py311-kirby is never activated. Skip mode is what
# makes it fit in ten minutes; full mode spends the window on pip checks.
#
#   srun -p devel -c 8 --mem=32G --time=10:00 --pty /bin/bash
#   # then, from the repo root:
#   export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
#   python -u tests/experiments/framework/f26_redundancy_penalty.py \
#       --dataset herg_ki --framework --redundancy-only
#
# That loads the real representations and computes both redundancy matrices,
# in seconds IF the representation cache is already warm for that dataset. It
# does NOT exercise the search; for that, add --smoke instead.
#SBATCH --job-name=f26_redund
#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# Every dataset with a curated pool in experiments._shared.FRAMEWORK_TASKS —
# molecular, reaction, antibody and TCGA, regression and classification.
DATASETS=(
  # ONE DATASET PER DOMAIN, TWO FOR MOLECULAR (owner, 2026-08-26). This is a
  # split by dataset, not a cut of any axis: all 11 arms, both objectives, the
  # full budget list and 5 outer folds still run on every dataset below.
  # The five dropped datasets are the SECOND and later member of a domain that
  # is already represented, never a domain left unmeasured.
  logd                    # molecular ADMET, RMSE
  herg_ki                 # molecular, RMSE
  buchwald_hartwig        # reaction, RMSE -- the only reaction set in the roster
  ginkgo_hic              # antibody developability, Spearman -- regression, so
                          # the read-across objective gets a continuous label
  tcga_luad_vs_lusc       # TCGA, AUROC -- the only TCGA set in the roster
)

# Dropped, each because its domain is already covered above, with the reason:
#   caco2                 second molecular ADMET set, and it is NOT in
#                         FRAMEWORK_TASKS, so --framework falls back to the
#                         'full' rep preset for it
#   flab_thermo_tresanco  second antibody set
#   flab_pk_jain_fcrnrt   third antibody set
#   flab_pk_jain_heprt    fourth antibody set
#   tz_her2_binary        fifth antibody set, and classification
#
# To run the full ten again, restore them here and submit --array=0-9%10.

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "f26 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

python -u tests/experiments/framework/f26_redundancy_penalty.py \
    --dataset "${ds}" --framework --mode full
