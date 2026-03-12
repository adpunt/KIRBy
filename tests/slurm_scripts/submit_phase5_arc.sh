#!/bin/bash
# Phase 5 baseline jobs for ARC (CPU-only, no GPU).
# Run from ARC login node.
#
# Usage:
#   bash tests/slurm_scripts/submit_phase5_arc.sh          # all CPU jobs
#   bash tests/slurm_scripts/submit_phase5_arc.sh molecular
#   bash tests/slurm_scripts/submit_phase5_arc.sh reaction
#   bash tests/slurm_scripts/submit_phase5_arc.sh antibody
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
    local mem=${3:-64G}
    local wall=${4:-47:59:00}

    J=$(sbatch \
        --job-name="p5_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall})"
}

echo "============================================"
echo "Phase 5: ARC CPU Jobs"
echo "============================================"

run_molecular() {
    local OUT="${BASE_OUTPUT}/molecular"
    echo "--- Molecular (8 datasets, ~35 reps) ---"
    submit mol_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py molecular --n-folds 5 --augment --output ${OUT}/" \
        64G 23:59:00
    echo ""
}

run_reaction() {
    local OUT="${BASE_OUTPUT}/reaction"
    echo "--- Reaction (3 datasets, 8 reps) ---"
    submit rxn_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py reaction --n-folds 5 --augment --output ${OUT}/" \
        32G 23:59:00
    echo ""
}

run_antibody_fast() {
    local OUT="${BASE_OUTPUT}/antibody"
    echo "--- Antibody fast reps (CPU: onehot, blosum, aac, paratope) ---"
    submit ab_fast_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps onehot,blosum,aac,paratope" \
        32G 23:59:00
    submit ab_fast_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps onehot,blosum,aac,paratope" \
        32G 23:59:00
    echo ""
}

run_tcga() {
    local OUT="${BASE_OUTPUT}/tcga"
    echo "--- TCGA (3 datasets, 57 reps) ---"
    submit tcga_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py tcga --n-folds 5 --augment --output ${OUT}/" \
        256G 23:59:00
    echo ""
}

run_reaction_comparison() {
    local OUT="tests/results/comparisons"
    echo "--- Reaction Rep Comparison (8 reps × 3 models) ---"
    submit cmp_reaction \
        "mkdir -p ${OUT} && python tests/experiments/reaction_rep_comparison.py --n-folds 5 --output ${OUT}/reaction_rep_comparison.json" \
        32G 06:00:00
    echo ""
}

case "$MODE" in
    molecular) run_molecular ;;
    reaction)  run_reaction ;;
    antibody)  run_antibody_fast ;;
    tcga)      run_tcga ;;
    cmp_reaction) run_reaction_comparison ;;
    all)
        run_molecular
        run_reaction
        run_antibody_fast
        run_tcga
        run_reaction_comparison
        echo "Total: 6 CPU jobs (mol, rxn, 2× ab_fast, tcga, cmp_reaction)"
        ;;
    *) echo "Usage: bash submit_phase5_arc.sh [all|molecular|reaction|antibody|tcga|cmp_reaction]"; exit 1 ;;
esac
