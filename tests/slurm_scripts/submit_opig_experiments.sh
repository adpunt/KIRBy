#!/bin/bash
#
# Submit the OPIG-derived experiments to SLURM.
#
# Usage:
#   bash tests/slurm_scripts/submit_opig_experiments.sh          # all 6
#   bash tests/slurm_scripts/submit_opig_experiments.sh humanness # single
#   bash tests/slurm_scripts/submit_opig_experiments.sh antibody  # Ab group (humanness + tz + ginkgo + cdr)
#   bash tests/slurm_scripts/submit_opig_experiments.sh molecular # Mol group (graphinity + qsar)
#   bash tests/slurm_scripts/submit_opig_experiments.sh shared_vectors  # the census only

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/data/stat-ecr/scat9264/KIRBy"

submit() {
    local name=$1
    local mem=$2
    local timelimit=$3
    local partition=$4
    shift 4
    local cmd="$@"

    J=$(sbatch --parsable \
        --job-name=opig_${name} \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=${timelimit} --partition=${partition} \
        --mem=${mem} \
        --mail-user=adelaide.punt@stcatz.ox.ac.uk \
        --mail-type=END,FAIL \
        --wrap="cd ${PROJECT_DIR} && git pull 2>/dev/null && . ./setup.sh && export PYTHONWARNINGS=ignore && ${cmd}")
    echo "  $J  opig_${name}  (${mem}, ${timelimit}, ${partition})"
}

FILTER="${1:-all}"
PRUNING_MODE="${PRUNING_MODE:-aggressive_pruning}"
PM="--pruning-mode ${PRUNING_MODE}"

echo "OPIG Experiment Submission (filter: ${FILTER}, pruning=${PRUNING_MODE})"
echo "================================================"

# --- Antibody experiments ---
if [[ "$FILTER" == "all" || "$FILTER" == "antibody" || "$FILTER" == "humanness" ]]; then
    submit humanness 64G 47:59:00 long \
        "python tests/experiments/opig_humanness.py ${PM} --datasets humanness_therapeutic"
fi

if [[ "$FILTER" == "all" || "$FILTER" == "antibody" || "$FILTER" == "trastuzumab" ]]; then
    submit trastuzumab 64G 47:59:00 long \
        "python tests/experiments/opig_trastuzumab_enrichment.py ${PM}"
fi

if [[ "$FILTER" == "all" || "$FILTER" == "antibody" || "$FILTER" == "ginkgo" ]]; then
    submit ginkgo 128G 47:59:00 long \
        "python tests/experiments/opig_ginkgo_developability.py ${PM}"
fi

# CDR flexibility DROPPED — only ~116 unique CDR sequences across all types,
# too small for meaningful 5-fold CV. Script still exists if needed later.

# --- DDG / Molecular experiments ---
if [[ "$FILTER" == "all" || "$FILTER" == "molecular" || "$FILTER" == "graphinity" ]]; then
    submit graphinity 128G 7-00:00:00 long \
        "python tests/experiments/opig_graphinity_ddg.py ${PM} --max-foldx 5000"
fi

if [[ "$FILTER" == "all" || "$FILTER" == "molecular" || "$FILTER" == "qsar" ]]; then
    submit qsar 64G 47:59:00 long \
        "python tests/experiments/opig_activity_cliff_qsar.py ${PM}"
fi

# --- Cross-dataset blueprint transfer (OPIG C5) ---
if [[ "$FILTER" == "all" || "$FILTER" == "molecular" || "$FILTER" == "blueprint" ]]; then
    submit blueprint_transfer 64G 47:59:00 long \
        "python tests/experiments/opig_blueprint_transfer.py ${PM} --source logd --target herg_ki --mode full"
fi

# --- How many molecules share a feature vector (2026-09-16) ---
# The census only. It reads the saved matrices and counts; it fits no model, so
# it belongs in this batch like any other single job.
#
# ⛔ ITS COMPANION IS NOT HERE. opig_shared_vectors_remedy.py is about 218
# million model fits per dataset at --speed full, which is an array across three
# stages, not a --wrap. This line said 948 million until 2026-09-18: it doubled
# the configuration count (the file pins rep_counts to [None]) and it counted
# 516 sweep settings, 288 of which were duplicates. It has its own submitter:
#     bash tests/slurm_scripts/submit_opig_shared_vectors.sh {census|spread|sweep}
if [[ "$FILTER" == "all" || "$FILTER" == "shared_vectors" ]]; then
    submit shared_vectors_census 64G 04:00:00 short \
        "python tests/experiments/opig_shared_vectors_census.py --check-freshness 200"
fi

echo ""
echo "Results will be in tests/results/comparisons/opig_*.json"
echo ""
echo "NOT submitted by this script: opig_shared_vectors_remedy.py — too large for"
echo "a --wrap job. See tests/slurm_scripts/submit_opig_shared_vectors.sh"
