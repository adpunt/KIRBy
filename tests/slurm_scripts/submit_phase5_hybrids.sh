#!/bin/bash
# Phase 5B: Run hybrid search per domain after baselines complete.
#
# Runs find_best_hybrid_*.py with all reps per domain. This is where
# KIRBy proves its value: hybridizing representations improves on
# individual baselines.
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_phase5_hybrids.sh              # all domains
#   bash tests/slurm_scripts/submit_phase5_hybrids.sh molecular    # one domain
#   bash tests/slurm_scripts/submit_phase5_hybrids.sh antibody
#   bash tests/slurm_scripts/submit_phase5_hybrids.sh dta
#   bash tests/slurm_scripts/submit_phase5_hybrids.sh tcga
#
# Output: tests/results/phase5_hybrids/<domain>/<dataset>.json

set -euo pipefail

MODE=${1:-all}
BASE_OUTPUT="tests/results/phase5_hybrids"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-64G}
    local wall=${4:-23:59:00}
    local gpu=${5:-}

    J=$(sbatch \
        --job-name="p5h_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        ${gpu:+--gres=gpu:1} \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}${gpu:+, GPU})"
}

echo "============================================"
echo "Phase 5B: Hybrid Search (all domains)"
echo "============================================"
echo "  Output: ${BASE_OUTPUT}/"
echo ""

# ---------------------------------------------------------------
# MOLECULAR — 8 datasets × all reps → hybrid search
# ---------------------------------------------------------------
run_molecular() {
    local OUT="${BASE_OUTPUT}/molecular"
    echo "--- Molecular Hybrid Search ---"

    for ds in logd caco2 herg qm9 adme_hlm adme_rlm adme_mdr1 adme_solubility; do
        submit "mol_${ds}" \
            "mkdir -p ${OUT} && python tests/find_best_hybrid_molecular.py ${ds} --reps all --search fast --n-folds 5 --tune --output ${OUT}/${ds}.json" \
            64G 23:59:00
    done
    echo ""
}

# ---------------------------------------------------------------
# TCGA — 3 datasets × 57 reps → hybrid search
# ---------------------------------------------------------------
run_tcga() {
    local OUT="${BASE_OUTPUT}/tcga"
    echo "--- TCGA Hybrid Search ---"

    for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        submit "tcga_${ds#tcga_}" \
            "mkdir -p ${OUT} && python tests/find_best_hybrid_tcga.py ${ds} --reps all --search fast --n-folds 5 --tune --output ${OUT}/${ds}.json" \
            256G 23:59:00
    done
    echo ""
}

# ---------------------------------------------------------------
# DTA — davis, kiba, bindingdb_kd × 37 reps → hybrid search
# ---------------------------------------------------------------
run_dta() {
    local OUT="${BASE_OUTPUT}/dta"
    echo "--- DTA Hybrid Search ---"

    submit dta_davis \
        "mkdir -p ${OUT} && python tests/find_best_hybrid_dta.py davis --reps full --search fast --n-folds 5 --tune --output ${OUT}/davis.json" \
        128G 47:59:00 gpu
    submit dta_kiba \
        "mkdir -p ${OUT} && python tests/find_best_hybrid_dta.py kiba --reps full --search fast --n-folds 5 --max-pairs 30000 --tune --output ${OUT}/kiba.json" \
        128G 47:59:00 gpu
    submit dta_bindingdb \
        "mkdir -p ${OUT} && python tests/find_best_hybrid_dta.py bindingdb_kd --reps full --search fast --n-folds 5 --max-pairs 30000 --tune --output ${OUT}/bindingdb_kd.json" \
        128G 47:59:00 gpu
    echo ""
}

# ---------------------------------------------------------------
# ANTIBODY — augmentation test via antibody_baseline_survey.py --augment
# No standalone find_best_hybrid_antibody.py exists; the --augment flag
# on antibody_baseline_survey.py tests concat of each small rep with the
# top base rep, which demonstrates KIRBy's hybrid augmentation value.
# ---------------------------------------------------------------
run_antibody() {
    local OUT="${BASE_OUTPUT}/antibody"
    echo "--- Antibody Augmentation/Hybrid (--augment on baseline survey) ---"

    # DMS datasets with all reps + augmentation
    submit ab_dms_augment \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment" \
        96G 23:59:00 gpu

    # Inter-family datasets with all reps + augmentation
    submit ab_inter_augment \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment" \
        96G 23:59:00 gpu
    echo ""
}

# ---------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------

case "$MODE" in
    molecular)
        run_molecular
        ;;
    tcga)
        run_tcga
        ;;
    dta)
        run_dta
        ;;
    antibody)
        run_antibody
        ;;
    all)
        run_molecular
        run_tcga
        run_dta
        run_antibody
        echo "============================================"
        echo "Total: ~15 hybrid search jobs"
        echo "  Molecular:  8"
        echo "  TCGA:       3"
        echo "  DTA:        3"
        echo "  Antibody:   2 (augmentation via baseline survey)"
        echo "============================================"
        ;;
    *)
        echo "Phase 5B: Hybrid Search"
        echo ""
        echo "Usage: bash submit_phase5_hybrids.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all         All domains (~39 jobs)"
        echo "  molecular   8 datasets, all reps"
        echo "  tcga        3 datasets, 57 reps"
        echo "  dta         3 datasets, 37 reps (GPU, 48h)"
        echo "  antibody    25 datasets (GPU)"
        echo ""
        echo "Output: ${BASE_OUTPUT}/<domain>/<dataset>.json"
        echo ""
        echo "Run AFTER baselines complete:"
        echo "  bash tests/slurm_scripts/submit_phase5_baselines.sh all"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${BASE_OUTPUT}/"
