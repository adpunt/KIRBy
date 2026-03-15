#!/bin/bash
# Structural Protein Reps Survey
#
# Runs just the 5 new structural protein representations through the DTA
# baseline survey: contact_graph (80D), usr_shape (60D), pocket (55D),
# esm2_pocket (1280D), pocket_composition (20D).
#
# These reps reuse ESMFold/AlphaFold cached coordinates. First run will
# predict structures (~0.5s/protein with ESMFold on GPU); subsequent runs
# hit the cache.
#
# Run from HTC login node (needs GPU for ESMFold structure prediction).
#
# Usage:
#   bash tests/slurm_scripts/submit_structural_protein_survey.sh          # all datasets
#   bash tests/slurm_scripts/submit_structural_protein_survey.sh davis    # davis only
#   bash tests/slurm_scripts/submit_structural_protein_survey.sh kiba     # kiba only

set -euo pipefail

MODE=${1:-all}
OUTPUT="tests/results/structural_protein_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

# The 5 structural reps + drug_ecfp4 (needed as minimal drug partner for paired eval)
STRUCT_REPS="protein_contact_graph,protein_usr_shape,protein_pocket,protein_esm2_pocket,protein_pocket_composition,drug_ecfp4"

submit() {
    local name=$1
    local dataset=$2
    local extra=${3:-}
    local mem=${4:-128G}
    local wall=${5:-47:59:00}

    J=$(sbatch \
        --job-name="struct_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error && mkdir -p ${OUTPUT} && python tests/dta_baseline_survey.py ${dataset} --reps ${STRUCT_REPS} --n-folds 5 --output ${OUTPUT}/${name}.json ${extra} 2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}, GPU)"
}

echo "============================================"
echo "Structural Protein Reps Survey"
echo "============================================"
echo "  Reps: contact_graph (80D), usr_shape (60D), pocket (55D),"
echo "        esm2_pocket (1280D), pocket_composition (20D)"
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    davis)
        submit davis davis
        echo ""
        echo "  Total: 1 job (davis, 5 structural reps + drug_ecfp4)"
        ;;
    kiba)
        submit kiba kiba "--max-pairs 30000"
        echo ""
        echo "  Total: 1 job (kiba, 30K pairs, 5 structural reps + drug_ecfp4)"
        ;;
    bindingdb)
        submit bindingdb bindingdb_kd "--max-pairs 30000"
        echo ""
        echo "  Total: 1 job (bindingdb_kd, 30K pairs)"
        ;;
    all)
        submit davis davis "" 128G 47:59:00
        submit kiba kiba "--max-pairs 30000" 128G 47:59:00
        submit bindingdb bindingdb_kd "--max-pairs 30000" 128G 47:59:00
        echo ""
        echo "  Total: 3 jobs (davis + kiba + bindingdb_kd)"
        ;;
    *)
        echo "Usage: bash submit_structural_protein_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all        All 3 DTA datasets (3 jobs)"
        echo "  davis      Davis only (442 kinases)"
        echo "  kiba       KIBA only (30K subsample)"
        echo "  bindingdb  BindingDB_Kd only (30K subsample)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
