#!/bin/bash
# Phase 5B: Run hybrid search per domain after baselines complete.
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_phase5_hybrids.sh              # all domains
#   bash tests/slurm_scripts/submit_phase5_hybrids.sh molecular
#   bash tests/slurm_scripts/submit_phase5_hybrids.sh antibody
#   bash tests/slurm_scripts/submit_phase5_hybrids.sh dta
#   bash tests/slurm_scripts/submit_phase5_hybrids.sh tcga
#
# Output: tests/results/phase5_hybrids/<domain>/<dataset>.json

set -euo pipefail

MODE=${1:-all}
BASE_OUTPUT="tests/results/phase5_hybrids"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}
    local gpu=${4:-}

    local gpu_flag=""
    if [ "$gpu" = "gpu" ]; then
        gpu_flag="--gres=gpu:1"
    fi

    J=$(sbatch \
        --job-name="p5h_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=47:59:00 --partition=long \
        --mem="${mem}" \
        ${gpu_flag} \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, 48h${gpu_flag:+, GPU})"
}

echo "============================================"
echo "Phase 5B: Hybrid Search (all domains)"
echo "============================================"
echo "  Output: ${BASE_OUTPUT}/"
echo ""

run_molecular() {
    local OUT="${BASE_OUTPUT}/molecular"
    echo "--- Molecular Hybrid Search ---"

    for ds in logd caco2 herg qm9 adme_hlm adme_rlm adme_mdr1 adme_solubility; do
        submit "mol_${ds}" \
            "mkdir -p ${OUT} && python tests/find_best_hybrid_molecular.py ${ds} --reps all --search fast --n-folds 5 --tune --output ${OUT}/${ds}.json"
    done
    echo ""
}

run_tcga() {
    local OUT="${BASE_OUTPUT}/tcga"
    echo "--- TCGA Hybrid Search ---"

    for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        submit "tcga_${ds#tcga_}" \
            "mkdir -p ${OUT} && python tests/find_best_hybrid_tcga.py ${ds} --reps all --search fast --n-folds 5 --tune --output ${OUT}/${ds}.json"
    done
    echo ""
}

run_dta() {
    local OUT="${BASE_OUTPUT}/dta"
    echo "--- DTA Hybrid Search ---"

    submit dta_davis \
        "mkdir -p ${OUT} && python tests/find_best_hybrid_dta.py davis --reps full --search fast --n-folds 5 --tune --output ${OUT}/davis.json" \
        256G gpu
    submit dta_kiba \
        "mkdir -p ${OUT} && python tests/find_best_hybrid_dta.py kiba --reps full --search fast --n-folds 5 --max-pairs 30000 --tune --output ${OUT}/kiba.json" \
        256G gpu
    submit dta_bindingdb \
        "mkdir -p ${OUT} && python tests/find_best_hybrid_dta.py bindingdb_kd --reps full --search fast --n-folds 5 --max-pairs 30000 --tune --output ${OUT}/bindingdb_kd.json" \
        256G gpu
    echo ""
}

run_antibody() {
    local OUT="${BASE_OUTPUT}/antibody"
    echo "--- Antibody Augmentation/Hybrid ---"

    submit ab_dms_augment \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment" \
        256G gpu
    submit ab_inter_augment \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment" \
        256G gpu
    echo ""
}

case "$MODE" in
    molecular) run_molecular ;;
    tcga)      run_tcga ;;
    dta)       run_dta ;;
    antibody)  run_antibody ;;
    all)
        run_molecular
        run_tcga
        run_dta
        run_antibody
        echo "============================================"
        echo "Total: 16 hybrid search jobs"
        echo "  Molecular:  8"
        echo "  TCGA:       3"
        echo "  DTA:        3"
        echo "  Antibody:   2"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash submit_phase5_hybrids.sh [all|molecular|tcga|dta|antibody]"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${BASE_OUTPUT}/"
