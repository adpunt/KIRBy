#!/bin/bash
# Submit DTA (Drug-Target Affinity) survey jobs.
# Run on the login node: bash tests/slurm_scripts/run_dta_scenarios.sh [MODE]
#
# Usage:
#   bash tests/slurm_scripts/run_dta_scenarios.sh              # all: davis + kiba frozen + kiba FT
#   bash tests/slurm_scripts/run_dta_scenarios.sh davis         # davis frozen only
#   bash tests/slurm_scripts/run_dta_scenarios.sh kiba          # kiba frozen only
#   bash tests/slurm_scripts/run_dta_scenarios.sh kiba_ft       # kiba with ESM-2 LoRA fine-tuning
#   bash tests/slurm_scripts/run_dta_scenarios.sh all           # davis + kiba + kiba_ft

MODE=${1:-all}
OUTPUT="tests/results/dta_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

submit() {
    local name=$1
    local dataset=$2
    local extra=$3
    local mem=${4:-64G}
    local wall=${5:-23:59:00}
    local gpu=${6:-}

    J=$(sbatch \
        --job-name="dta_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        ${gpu:+--gres=gpu:1} \
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && mkdir -p ${OUTPUT} && python tests/find_best_hybrid_dta.py ${dataset} --reps full --search fast --n-folds 5 --output ${OUTPUT}/${name}.json ${extra}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}${gpu:+, GPU})"
}

echo "Submitting DTA survey..."
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    davis)
        submit davis_frozen davis "" 64G 47:59:00 gpu
        echo ""
        echo "  Total: 1 job (davis frozen, 27 reps)"
        ;;
    kiba)
        submit kiba_frozen kiba "--max-pairs 30000" 64G 47:59:00 gpu
        echo ""
        echo "  Total: 1 job (kiba frozen, 27 reps)"
        ;;
    kiba_ft)
        submit kiba_ft kiba "--max-pairs 30000 --finetune protein_esm2_lora --finetune-epochs 5" 96G 47:59:00 gpu
        echo ""
        echo "  Total: 1 job (kiba with ESM-2 LoRA fine-tuning, GPU)"
        ;;
    all)
        submit davis_frozen davis "" 64G 47:59:00 gpu
        submit kiba_frozen kiba "--max-pairs 30000" 64G 47:59:00 gpu
        submit kiba_ft kiba "--max-pairs 30000 --finetune protein_esm2_lora --finetune-epochs 5" 96G 47:59:00 gpu
        echo ""
        echo "  Total: 3 jobs (davis frozen + kiba frozen + kiba FT)"
        ;;
    *)
        echo "Usage: bash run_dta_scenarios.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all       Davis frozen + KIBA frozen + KIBA FT (3 jobs)"
        echo "  davis     Davis frozen (27 reps, 1 job)"
        echo "  kiba      KIBA frozen (27 reps, 30K pairs, 1 job)"
        echo "  kiba_ft   KIBA with ESM-2 protein LoRA fine-tuning (GPU)"
        echo ""
        echo "KIBA: DeepDTA E2E beats frozen by +0.056 R²; fine-tuning recommended."
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
