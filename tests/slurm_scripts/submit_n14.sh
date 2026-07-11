#!/bin/bash
#SBATCH --job-name=n14_ac_landscape
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --account=stat-ecr
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# N14 activity-cliff landscape survey — the whole molecular-regression set in
# ONE small job. n14 trains NO downstream models: an n x n ECFP4-Tanimoto matrix
# + one LightGBM stage-0 diagnostic per dataset. Minutes per dataset, CPU-only,
# low memory.
#
# Account/partition mirror the PROVEN scripts (submit_dataset.sh,
# submit_arc_nogpu.sh): account=stat-ecr, partition matched to the wall.
# stat-ecr is the account whose association covers the arc CPU nodes and holds
# the data/cache on /data/stat-ecr — do NOT bill this to stat-cadd on a raw
# fair-share number: that association routes to a node set that is currently
# drained (temperature), which is what produced ReqNodeNotAvail.
# `short` (12h wall) is the idiomatic partition for a <12h CPU job on ARC
# (see submit_arc_nogpu.sh's reaction job) and is separate hardware from
# medium/long, so it isn't gated behind their backlog.
#
# We call n14 DIRECTLY (not via run_dataset.py/run_all): NO hybrid precompute,
# which n14 does not use. --reps fast loads a couple of cheap RDKit reps; the
# cliff landscape (prevalence/geometry/feasibility) uses only SMILES + labels,
# and only stage-0 uses the reps.
#
#   sbatch tests/slurm_scripts/submit_n14.sh

set -uo pipefail

cd /data/stat-ecr/scat9264/KIRBy
git pull --ff-only origin main 2>&1 || echo "WARNING: git pull failed; running on-disk code"
. ./setup.sh

export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

OUT="tests/results/noise"
DATASETS=(herg_ki herg caco2 logd adme_hlm adme_rlm adme_mdr1)

echo "=== N14 activity-cliff landscape survey ==="
echo "Branch: $(git rev-parse --abbrev-ref HEAD) ($(git rev-parse --short HEAD))"
echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Start: $(date)"
echo "Datasets: ${DATASETS[*]}"

WORST_RC=0
for DS in "${DATASETS[@]}"; do
    echo ""
    echo "--- ${DS} ---"
    python -u tests/experiments/noise/n14_activity_cliff_landscape.py \
        --dataset "${DS}" --reps fast --output "${OUT}"
    RC=$?
    [ ${RC} -gt ${WORST_RC} ] && WORST_RC=${RC}
    echo "--- ${DS} done (exit=${RC}) ---"
done

echo ""
echo "Done: $(date)  (worst exit=${WORST_RC})"
# Results at ${OUT}/n14_activity_cliff_landscape/<dataset>.json
exit ${WORST_RC}
