#!/bin/bash
#SBATCH --job-name=n14_ac_landscape
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# N14 activity-cliff landscape survey — the whole molecular-regression set in
# ONE small job. n14 trains NO downstream models: it builds an n x n
# ECFP4-Tanimoto matrix + a single LightGBM stage-0 diagnostic per dataset, so
# it is minutes-per-dataset, CPU-only, low memory. RIGHT-SIZED accordingly
# (1h wall, 32G, 8 cpus) — do NOT run this through submit_dataset.sh, which
# requests a 7-day/128G/long job AND triggers the full hybrid precompute n14
# never uses.
#
# We call n14 DIRECTLY (not via run_dataset.py/run_all), so there is NO
# precompute step. --reps fast loads a couple of cheap RDKit reps; the cliff
# landscape (prevalence/geometry/feasibility — the primary output) uses only
# SMILES + labels, and only stage-0 uses the reps.
#
# NO partition / account directive here ON PURPOSE. Both are LIVE-STATE. Run
# tests/slurm_scripts/where_to_submit.sh first and pass them at submit time:
#
#   cd /data/stat-ecr/scat9264/KIRBy && git pull --ff-only origin main \
#     && bash tests/slurm_scripts/where_to_submit.sh
#   sbatch --partition=<PICK> --account=<PICK> tests/slurm_scripts/submit_n14.sh
#
# A 1h job backfills into idle gaps a 7-day job cannot — so it should start even
# when long/medium are backlogged.

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
