#!/bin/bash
# Submit universal baseline survey as parallel SLURM jobs (one per domain).
# Run on the login node: bash tests/slurm_scripts/submit_universal_survey.sh
#
# Covers: molecular (4), TCGA (3), DTA (4), reaction (2) datasets.
# Antibody datasets use submit_antibody_survey.sh instead.
#
# Usage:
#   bash tests/slurm_scripts/submit_universal_survey.sh           # all domains
#   bash tests/slurm_scripts/submit_universal_survey.sh molecular # single domain
#   bash tests/slurm_scripts/submit_universal_survey.sh all_ft    # all with fine-tuning (GPU)

MODE=${1:-all}
OUTPUT="tests/results/universal_survey"
COMMON="--n-folds 3 --augment --output ${OUTPUT}/"
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
    molecular_ft)
        submit molecular 64G 23:59:00 "--finetune" gpu
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
    dta_ft)
        submit dta 96G 23:59:00 "--finetune" gpu
        ;;
    reaction)
        submit reaction 32G 06:00:00
        ;;
    all)
        submit molecular
        submit tcga 96G
        submit dta 96G
        submit reaction 32G 06:00:00
        echo ""
        echo "  Total: 4 jobs, 13 datasets (with --augment)"
        ;;
    all_ft)
        submit molecular 64G 23:59:00 "--finetune" gpu
        submit tcga 96G 23:59:00 "--finetune" gpu
        submit dta 96G 23:59:00 "--finetune" gpu
        submit reaction 32G 06:00:00
        echo ""
        echo "  Total: 4 jobs, 13 datasets (with --finetune --augment, GPU)"
        ;;
    *)
        echo "Usage: bash submit_universal_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all           All domains, frozen reps + augment (default)"
        echo "  all_ft        All domains with fine-tuning + augment (GPU)"
        echo "  quick         Molecular only (fastest)"
        echo "  molecular     Molecular domain (4 datasets)"
        echo "  molecular_ft  Molecular + ChemBERTa/MolFormer LoRA (GPU)"
        echo "  tcga          TCGA domain (3 datasets)"
        echo "  tcga_ft       TCGA + Geneformer/scFoundation FT (GPU)"
        echo "  dta           DTA domain (4 datasets)"
        echo "  dta_ft        DTA + ESM-2 protein LoRA (GPU)"
        echo "  reaction      Reaction yield (2 datasets, needs: pip install drfp)"
        echo ""
        echo "All modes include --augment (concat small reps with top base)."
        echo "For antibody datasets: bash submit_antibody_survey.sh"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
