#!/bin/bash
# Phase 5 baseline jobs for ARC (CPU-only, no GPU).
# Run from ARC login node.
#
# Usage:
#   bash tests/slurm_scripts/submit_phase5_arc.sh          # all CPU jobs
#   bash tests/slurm_scripts/submit_phase5_arc.sh molecular
#   bash tests/slurm_scripts/submit_phase5_arc.sh reaction
#   bash tests/slurm_scripts/submit_phase5_arc.sh tcga

set -euo pipefail

MODE=${1:-all}
BASE_OUTPUT="tests/results/phase5_baselines"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}

    J=$(sbatch \
        --job-name="p5_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=47:59:00 --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, 48h)"
}

echo "============================================"
echo "Phase 5: ARC CPU Jobs"
echo "============================================"

run_molecular() {
    local OUT="${BASE_OUTPUT}/molecular"
    echo "--- Molecular (8 datasets, ~35 reps) ---"
    submit mol_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py molecular --n-folds 5 --augment --output ${OUT}/"
    echo ""
}

run_reaction() {
    local OUT="${BASE_OUTPUT}/reaction"
    echo "--- Reaction (3 datasets, 8 reps) ---"
    submit rxn_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py reaction --n-folds 5 --augment --output ${OUT}/"
    echo ""
}

run_tcga() {
    local OUT="${BASE_OUTPUT}/tcga"
    echo "--- TCGA (3 datasets, 57 reps) ---"
    submit tcga_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py tcga --n-folds 5 --augment --output ${OUT}/"
    echo ""
}

run_reaction_comparison() {
    local OUT="tests/results/comparisons"
    echo "--- Reaction Rep Comparison (8 reps × 3 models) ---"
    submit cmp_reaction \
        "mkdir -p ${OUT} && python tests/experiments/reaction_rep_comparison.py --n-folds 5 --output ${OUT}"
    echo ""
}

case "$MODE" in
    molecular) run_molecular ;;
    reaction)  run_reaction ;;
    tcga)      run_tcga ;;
    cmp_reaction) run_reaction_comparison ;;
    all)
        run_molecular
        run_reaction
        run_tcga
        run_reaction_comparison
        echo "Total: 4 CPU jobs"
        ;;
    *) echo "Usage: bash submit_phase5_arc.sh [all|molecular|reaction|tcga|cmp_reaction]"; exit 1 ;;
esac
