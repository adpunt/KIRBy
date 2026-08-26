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
#          tests/slurm_scripts/run_f25_candidate_cutoff.sh
#
# BEFORE QUEUEING, prove it runs on real data. devel has a 10-minute limit and
# is usually empty. setup.sh MUST be sourced first -- without it
# KIRBY_CACHE_DIR is unset, the representation cache resolves to the home quota,
# every representation is recomputed from SMILES, and the venv is never
# activated. Skip mode is what makes it fit in ten minutes.
#
#   srun -p devel -c 8 --mem=32G --time=10:00 --pty /bin/bash
#   export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
#   python -u tests/experiments/framework/f25_candidate_cutoff.py \
#       --dataset herg_ki --smoke
#
# --smoke is one fold, one model, one seed, curated donors, cap 64, ladder
# only: 57 model fits, measured with --count-only, so it fits the window. Use
# herg_ki and NOT flab_thermo_tresanco -- tresanco's curated donor is 48
# features, under the smallest cut-off of 50, so every configuration there is
# correctly skipped as degenerate and nothing runs.
#
#SBATCH --job-name=f25_cutoff
#SBATCH --time=24:00:00
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

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "f25 array task ${SLURM_ARRAY_TASK_ID} -> ds=${ds}"
echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"

python -u tests/experiments/framework/f25_candidate_cutoff.py \
    --dataset "${ds}" --count-only

python -u tests/experiments/framework/f25_candidate_cutoff.py \
    --dataset "${ds}" --mode full

echo "=== ${ds}: done"
