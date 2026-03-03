#!/bin/bash
# Submit universal baseline survey as parallel SLURM jobs (one per domain).
# Run on the login node: bash tests/slurm_scripts/submit_universal_survey.sh
#
# Each domain runs as a separate SLURM job. The Python script iterates over
# all datasets in the domain. Default + tuned HPs in one pass.
#
# Usage:
#   bash tests/slurm_scripts/submit_universal_survey.sh           # all domains
#   bash tests/slurm_scripts/submit_universal_survey.sh quick     # molecular only
#   bash tests/slurm_scripts/submit_universal_survey.sh molecular # single domain
#   bash tests/slurm_scripts/submit_universal_survey.sh tcga_ft   # TCGA with fine-tuning (GPU)

MODE=${1:-all}
OUTPUT="tests/results/universal_survey"
COMMON="--tune --n-folds 3 --output ${OUTPUT}/"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

submit() {
    local domain=$1
    local mem=${2:-64G}
    local wall=${3:-23:59:00}
    local extra=${4:-}
    local gpu=${5:-}

    local gpu_flag=""
    if [ -n "$gpu" ]; then
        gpu_flag="#SBATCH --gres=gpu:1"
    fi

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
        # Molecular: 4 datasets x 15 reps x 5 models = 300 cells
        submit molecular
        ;;
    ab_dms)
        # Antibody DMS: 7 datasets x 10 reps x 5 models = 350 cells
        submit ab_dms
        ;;
    ab_inter)
        # Antibody inter: 5 datasets x 6 reps x 5 models = 150 cells
        submit ab_inter
        ;;
    tcga)
        # TCGA: 3 datasets x 3 reps x 5 models = 45 cells
        submit tcga 96G
        ;;
    tcga_ft)
        # TCGA with fine-tuning: 3 datasets x (3+2 FT) reps x 5 models = 75 cells
        submit tcga 96G 23:59:00 "--finetune" gpu
        ;;
    dta)
        # DTA: 2 datasets x 5 reps x 5 models = 50 cells
        submit dta 96G
        ;;
    all)
        submit molecular
        submit ab_dms
        submit ab_inter
        submit tcga 96G
        submit dta 96G
        echo ""
        echo "  Total: 5 jobs, 21 datasets"
        ;;
    all_ft)
        submit molecular
        submit ab_dms
        submit ab_inter
        submit tcga 96G 23:59:00 "--finetune" gpu
        submit dta 96G
        echo ""
        echo "  Total: 5 jobs, 21 datasets (TCGA with fine-tuning)"
        ;;
    *)
        echo "Usage: bash submit_universal_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all         All 5 domains in parallel (default)"
        echo "  all_ft      All domains, TCGA with fine-tuning (GPU)"
        echo "  quick       Molecular only (fastest)"
        echo "  molecular   Molecular domain (4 datasets, 15 reps)"
        echo "  ab_dms      Antibody DMS domain (7 datasets, 10 reps)"
        echo "  ab_inter    Antibody inter-family (5 datasets, 6 reps)"
        echo "  tcga        TCGA domain (3 datasets, 3 reps)"
        echo "  tcga_ft     TCGA with fine-tuning (3 datasets, 3+2 reps, GPU)"
        echo "  dta         DTA domain (2 datasets, 5 reps)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
