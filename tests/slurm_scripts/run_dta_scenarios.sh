#!/bin/bash
# Submit DTA (Drug-Target Affinity) survey jobs.
# Run on the login node: bash tests/slurm_scripts/run_dta_scenarios.sh [MODE]
#
# Usage:
#   bash tests/slurm_scripts/run_dta_scenarios.sh              # all: davis + kiba + bindingdb + cold-drug
#   bash tests/slurm_scripts/run_dta_scenarios.sh davis         # davis frozen only
#   bash tests/slurm_scripts/run_dta_scenarios.sh kiba          # kiba frozen only
#   bash tests/slurm_scripts/run_dta_scenarios.sh kiba_ft       # kiba with ESM-2 LoRA fine-tuning
#   bash tests/slurm_scripts/run_dta_scenarios.sh bindingdb_kd  # BindingDB_Kd (via TDC)
#   bash tests/slurm_scripts/run_dta_scenarios.sh davis_cold_drug  # davis cold-drug split
#   bash tests/slurm_scripts/run_dta_scenarios.sh all           # all modes

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
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error && mkdir -p ${OUTPUT} && python tests/find_best_hybrid_dta.py ${dataset} --reps full --search fast --n-folds 5 --output ${OUTPUT}/${name}.json ${extra} 2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)" \
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
    bindingdb_kd)
        submit bindingdb_kd_frozen bindingdb_kd "--max-pairs 30000" 64G 47:59:00 gpu
        echo ""
        echo "  Total: 1 job (BindingDB_Kd frozen, 30K pairs)"
        ;;
    davis_cold_drug)
        submit davis_cold_drug davis "--cold-setting drug" 64G 47:59:00 gpu
        echo ""
        echo "  Total: 1 job (davis cold-drug, 68 drug groups)"
        ;;
    kiba_cold_drug)
        submit kiba_cold_drug kiba "--max-pairs 30000 --cold-setting drug" 64G 47:59:00 gpu
        echo ""
        echo "  Total: 1 job (kiba cold-drug, 30K pairs)"
        ;;
    all)
        submit davis_frozen davis "" 64G 47:59:00 gpu
        submit kiba_frozen kiba "--max-pairs 30000" 64G 47:59:00 gpu
        submit kiba_ft kiba "--max-pairs 30000 --finetune protein_esm2_lora --finetune-epochs 5" 96G 47:59:00 gpu
        submit bindingdb_kd_frozen bindingdb_kd "--max-pairs 30000" 64G 47:59:00 gpu
        submit davis_cold_drug davis "--cold-setting drug" 64G 47:59:00 gpu
        submit kiba_cold_drug kiba "--max-pairs 30000 --cold-setting drug" 64G 47:59:00 gpu
        echo ""
        echo "  Total: 6 jobs (davis + kiba + kiba_ft + bindingdb + cold-drug × 2)"
        ;;
    *)
        echo "Usage: bash run_dta_scenarios.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all              All modes (6 jobs)"
        echo "  davis            Davis frozen (27 reps, 1 job)"
        echo "  kiba             KIBA frozen (27 reps, 30K pairs, 1 job)"
        echo "  kiba_ft          KIBA with ESM-2 protein LoRA fine-tuning (GPU)"
        echo "  bindingdb_kd     BindingDB_Kd frozen (via TDC, 30K pairs)"
        echo "  davis_cold_drug  Davis cold-drug split (68 drug groups)"
        echo "  kiba_cold_drug   KIBA cold-drug split (30K pairs)"
        echo ""
        echo "KIBA: DeepDTA E2E beats frozen by +0.056 R²; fine-tuning recommended."
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
