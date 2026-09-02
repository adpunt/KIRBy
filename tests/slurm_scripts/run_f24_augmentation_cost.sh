#!/bin/bash
# F24/F25 COST MEASUREMENT — seconds per model fit, per dataset.
#
# THIS IS NOT THE GRID. It runs ONE fold of ONE cell per dataset and reports how
# long a model fit takes there. The grid's own scripts print the fit COUNT
# without fitting anything (--count-only); multiplying the two is the cost
# estimate stage 2 owes. Neither number is guessed.
#
# Why it has to run here and not locally: only four of the fourteen datasets
# have their representations on the local drive, and the machine those four
# could be timed on was at load average 57 on 8 cores.
#
# ONE ARRAY TASK PER DATASET. Each writes its own file, so there are no write
# races: tests/experiments/results/framework/f24_augmentation_admission_<ds>__timing.json
#
# CACHE. Both experiments construct recipes directly and call create_hybrid.
# Neither runs a search and neither writes to result_db, so nothing needs
# deleting before or after and no stored result can be evicted.
#
# WALL TIME. One fold, one cell, one model per dataset. Minutes, not hours.
# --time is set well above that so a slow representation load cannot kill it.
#
# NEVER PASS --account. 123 of the 125 scripts in this directory omit it and let
# SLURM resolve the default association. Adding it cost three failed
# submissions on 2026-07-11 and moved nothing.
#
# WHICH CLUSTER is set by the LOGIN NODE, not by --partition. This job is
# 16 CPUs, CPU only, no GPU, four independent tasks -- an htc job by ARC's own
# division of labour. Submit from htc-login, or add --clusters=htc.
#
# --partition is LIVE STATE. Confirm with where_to_submit.sh, then:
#   sbatch --partition=<part> --array=0-3 \
#          tests/slurm_scripts/run_f24_augmentation_cost.sh
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
#       --dataset herg_ki --count-only
#
# Loads the real representations and prints the configuration and
# model-fit counts. Fits NOTHING, so it finishes in the window.
#
#SBATCH --job-name=f24_cost
#SBATCH --time=01:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# WALL AND MEMORY ARE MEASURED, from job 12913493 on 2026-09-02.
# Elapsed 8 to 26 minutes across the six tasks. MaxRSS 0.44G to 3.97G, the
# worst being logd at 5,028 rows. The old request was 08:00:00 and 256G --
# 64x the memory ever used. At 8 GB per core on arc a 16-core 256G task ties
# up 32 cores' worth of memory and waits for a whole node, which is why these
# sat in the queue. 64G is 16x the worst measured and well under the ratio.
#
# CORES STAY AT 16. Seconds per fit is what this job measures, and it depends
# on the core count. Dropping to 8 would make the new numbers incomparable
# with buchwald_hartwig's 42.74, which was measured on 16.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# FOUR datasets = 4 array tasks (--array=0-3).
DATASETS=(
  # FOUR, chosen by the user on 2026-09-02, replacing the six of 26 August.
  #
  # What this roster covers: one reaction, one molecular, one antibody, one
  # TCGA. RMSE, Spearman and f1_weighted. tcga_brca_pam50 is the first
  # CLASSIFICATION dataset this study has run.
  buchwald_hartwig        # reaction,  RMSE         n=3,955  timed 42.74 s/fit
  caco2                   # molecular, RMSE         n=2,158  not timed
  flab_expression_koenig  # antibody,  Spearman     n=4,275  not timed
  tcga_brca_pam50         # TCGA,      f1_weighted  n=900    not timed
)
# DROPPED from the 26 August six, and what that costs in coverage: herg_ki,
# logd, flab_thermo_tresanco, abagym_VEGF, abcov_kd. Antibody DMS and the
# small-n antibody set go with them, so this run cannot say whether an
# admission method that wins elsewhere also wins at n=141. caco2 and
# tcga_brca_pam50 were both in the eight dropped on 26 August and are back.
#
# humanness_therapeutic was proposed and dropped by the user on 2026-09-02.
# It has an empty curated augmentation list in DATASET_REPS, and its metric is
# AUC while f24 scores every fit from hard class predictions (m.predict, not
# predict_proba) -- AUROC computed that way is not AUROC. f1_weighted, which
# tcga_brca_pam50 uses, IS correct off hard predictions, so that dataset is
# unaffected.
#
# Row counts from DATASET_STATUS.md and dataset_policy.DATASET_REGISTRY.

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "f24 cost array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

# 1. The count. Fits nothing; prints configurations and model fits per dataset.
echo "=== ${ds}: counting"
python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --count-only

python -u tests/experiments/framework/f25_candidate_cutoff.py \
    --dataset "${ds}" --count-only

# 2. The timing. One fold of the most expensive configuration, on the HYBRID
#    placement: capped, both donor sources, no per-donor cap,
#    residual_correlation, the ladder. The hybrid placement is the one that
#    costs -- create_hybrid scores all ten representations and computes feature
#    importance on every call, measured at 336s per configuration locally.
#    The wall for run_f24_augmentation_grid.sh follows from this number.
echo "=== ${ds}: timing one fold of the most expensive configuration"
python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --time-only --mode intermediary --models lightgbm \
    --hybrid-budgets 250

echo "=== ${ds}: done"
