#!/bin/bash
#
# Submit all 6 OPIG-derived experiments to SLURM.
#
# Usage:
#   bash tests/slurm_scripts/submit_opig_experiments.sh          # all 6
#   bash tests/slurm_scripts/submit_opig_experiments.sh humanness # single
#   bash tests/slurm_scripts/submit_opig_experiments.sh antibody  # Ab group (humanness + tz + ginkgo + cdr)
#   bash tests/slurm_scripts/submit_opig_experiments.sh molecular # Mol group (graphinity + qsar)

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
        "python tests/experiments/opig_humanness.py ${PM} --datasets humanness_species,humanness_therapeutic"
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
        "python tests/experiments/opig_graphinity_ddg.py ${PM} --max-samples 5000"
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

echo ""
echo "Results will be in tests/results/comparisons/opig_*.json"
