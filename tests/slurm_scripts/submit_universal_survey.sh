#!/bin/bash
# Submit universal baseline survey as parallel SLURM jobs (one per domain).
# Run on the login node: bash tests/slurm_scripts/submit_universal_survey.sh
#
# Covers: molecular (4 datasets), TCGA (3 datasets), DTA (2 datasets).
# Antibody datasets use submit_antibody_survey.sh instead.
#
# Usage:
#   bash tests/slurm_scripts/submit_universal_survey.sh           # all domains
#   bash tests/slurm_scripts/submit_universal_survey.sh molecular # single domain
#   bash tests/slurm_scripts/submit_universal_survey.sh tcga_ft   # TCGA with fine-tuning (GPU)

MODE=${1:-all}
OUTPUT="tests/results/universal_survey"
COMMON="--n-folds 3 --output ${OUTPUT}/"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

submit() {
    local domain=$1
    local mem=${2:-64G}
    local wall=${3:-23:59:00}
    local extra=${4:-}
    local gpu=${5:-}

    J=$(sbatch \
        --job-name="survey_${domain}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        ${gpu:+--gres=gpu:1} \
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python tests/run_baseline_survey.py ${domain} ${COMMON} ${extra}" \
        | awk '{print $4}')
    echo "  $J  ${domain}  (mem=${mem}, time=${wall}${gpu:+, GPU})"
}

echo "Submitting universal baseline survey..."
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    quick|molecular)
        submit molecular
        ;;
    tcga)
        submit tcga 96G
        ;;
    tcga_ft)
        submit tcga 96G 23:59:00 "--finetune" gpu
        ;;
    dta)
        submit dta 96G
        ;;
    all)
        submit molecular
        submit tcga 96G
        submit dta 96G
        echo ""
        echo "  Total: 3 jobs, 9 datasets"
        ;;
    all_ft)
        submit molecular
        submit tcga 96G 23:59:00 "--finetune" gpu
        submit dta 96G
        echo ""
        echo "  Total: 3 jobs, 9 datasets (TCGA with fine-tuning)"
        ;;
    *)
        echo "Usage: bash submit_universal_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all         All 3 domains in parallel (default)"
        echo "  all_ft      All domains, TCGA with fine-tuning (GPU)"
        echo "  quick       Molecular only (fastest)"
        echo "  molecular   Molecular domain (4 datasets, 15 reps)"
        echo "  tcga        TCGA domain (3 datasets, 3 reps)"
        echo "  tcga_ft     TCGA with fine-tuning (3 datasets, 3+2 reps, GPU)"
        echo "  dta         DTA domain (2 datasets, 5 reps)"
        echo ""
        echo "For antibody datasets: bash submit_antibody_survey.sh"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
