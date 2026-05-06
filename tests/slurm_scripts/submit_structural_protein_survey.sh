#!/bin/bash
# Structural Protein Reps Survey
#
# Runs the 5 structural protein representations through the DTA baseline survey:
# contact_graph (80D), usr_shape (60D), pocket (55D), esm2_pocket (1280D),
# pocket_composition (20D).
#
# Usage:
#   bash tests/slurm_scripts/submit_structural_protein_survey.sh          # all datasets
#   bash tests/slurm_scripts/submit_structural_protein_survey.sh davis    # davis only

set -euo pipefail

MODE=${1:-all}
OUTPUT="tests/results/structural_protein_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

STRUCT_REPS="protein_contact_graph,protein_usr_shape,protein_pocket,protein_esm2_pocket,protein_pocket_composition,drug_ecfp4"

submit() {
    local name=$1
    local dataset=$2
    local extra=${3:-}

    J=$(sbatch \
        --job-name="struct_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=47:59:00 --partition=long \
        --mem=256G \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/dta_baseline_survey.py ${dataset} --reps ${STRUCT_REPS} --n-folds 5 --output ${OUTPUT}/${name}.json ${extra} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (256G, 48h, GPU)"
}

echo "============================================"
echo "Structural Protein Reps Survey"
echo "============================================"

case "$MODE" in
    davis)
        submit davis davis
        ;;
    kiba)
        submit kiba kiba "--max-pairs 30000"
        ;;
    bindingdb)
        submit bindingdb bindingdb_kd "--max-pairs 30000"
        ;;
    all)
        submit davis davis
        submit kiba kiba "--max-pairs 30000"
        submit bindingdb bindingdb_kd "--max-pairs 30000"
        echo "  Total: 3 jobs"
        ;;
    *)
        echo "Usage: bash submit_structural_protein_survey.sh [all|davis|kiba|bindingdb]"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
