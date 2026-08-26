#!/bin/bash
# F25 — how long should the ranked candidate list be? 50, 100, 250.
#
# One axis. Everything else pinned: capped and uncapped (the only two methods
# that read a ranked list), all three donor sources, per-donor caps of 64 and
# off, the out-of-fold residual ranking, the ladder and each fraction, margin 0.
#
# A per-donor cap of 5 is EXCLUDED by degeneracy, not by cost: it pools to 15-50
# features, which is at most the smallest cut-off, so no value of the cut-off
# can change anything there. The script asserts that from the measured width and
# prints what it skipped.
#
# WHY THIS EXISTS. The number has been 50 since the code was written and no
# caller has ever set it. At 50, every augmentation_fraction of a 1000 budget
# asks for 50, 100 or 250 extras and all three clamp to 50 -- so the size rule
# is not an axis at all until this moves. RUN THIS BEFORE THE F24 GRID and set
# f24's --max-candidates from the answer.
#
# SMALL. 144 configurations and 22,440 model fits per dataset, against 446,760
# for the f24 grid. One array task per dataset, no further split expected.
#
# CACHE. Constructs recipes directly, runs no search, writes to no result_db.
# The cut-off is deliberately NOT a key column: adding one would force every
# result_db/*.db to be deleted for a value the search never varies.
#
# --account and --partition are LIVE STATE. Confirm with where_to_submit.sh
# first, then:
#   sbatch --account=<acct> --partition=<part> --array=0-13%7 \
#          tests/slurm_scripts/run_f25_candidate_cutoff.sh
#SBATCH --job-name=f25_cutoff
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# Same fourteen as run_f24_augmentation_grid.sh. Keep the lists identical.
DATASETS=(
  logd  herg_ki  buchwald_hartwig  caco2  flab_thermo_tresanco
  flab_pk_jain_fcrnrt  flab_pk_jain_heprt  ginkgo_hic  tz_her2_binary
  tcga_luad_vs_lusc
  abagym_VEGF  abcov_kd  ginkgo_acsins  ddg_combined
)

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "f25 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

python -u tests/experiments/framework/f25_candidate_cutoff.py \
    --dataset "${ds}" --count-only

python -u tests/experiments/framework/f25_candidate_cutoff.py \
    --dataset "${ds}" --mode full

echo "=== ${ds}: done"
