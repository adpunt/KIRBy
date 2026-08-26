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
# 16 CPUs, CPU only, no GPU, six independent tasks -- an htc job by ARC's own
# division of labour. Submit from htc-login, or add --clusters=htc.
#
# --partition is LIVE STATE. Confirm with where_to_submit.sh, then:
#   sbatch --partition=<part> --array=0-5 \
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
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# SIX datasets = 6 array tasks (--array=0-5).
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
# Gaps recomputed 2026-08-26 from <cache>/rep_prune/<ds>/scores.json, the same
# file DATASET_STATUS.md's "best solo" column is built from. The four added
# datasets have an empty curated augmentation list in DATASET_REPS, so only the
# borrow-from-the-other-representations donor source runs on them.

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "f24 cost array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

# 1. The count. Fits nothing; prints configurations and model fits per dataset.
echo "=== ${ds}: counting"
python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --count-only

python -u tests/experiments/framework/f25_candidate_cutoff.py \
    --dataset "${ds}" --count-only

# 2. The timing. One fold of the most expensive cell: capped, both donor
#    sources, no per-donor cap, residual_correlation, the ladder.
echo "=== ${ds}: timing one fold of the most expensive configuration"
python -u tests/experiments/framework/f24_augmentation_admission.py \
    --dataset "${ds}" --time-only --mode intermediary --models lightgbm

echo "=== ${ds}: done"
