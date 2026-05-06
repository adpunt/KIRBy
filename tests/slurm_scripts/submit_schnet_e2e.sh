#!/bin/bash
# E2E SchNet generalization test: logd, herg, caco2.
# Currently confirmed top tier on qm9 only (R²=0.95).
# Question: does E2E SchNet generalize beyond quantum properties?
#
# Runs --finetune which adds: ft_schnet, lora_chemberta, lora_molformer.
# Baselines computed too (unavoidable) but screening mode keeps it fast.
# GPU needed for: SchNet E2E training + LM rep generation.
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_schnet_e2e.sh          # all 3 datasets
#   bash tests/slurm_scripts/submit_schnet_e2e.sh logd     # single dataset

MODE=${1:-all}
OUTPUT="tests/results/schnet_e2e"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# Screening: 3 models, 3 folds
MODELS="ridge,rf,xgboost"
FOLDS=3

submit() {
    local name=$1
    local dataset=$2

    J=$(sbatch \
        --job-name="schnet_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=23:59:00 --partition=long \
        --mem=128G \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/run_baseline_survey.py ${dataset} --finetune --models ${MODELS} --n-folds ${FOLDS} --output ${OUTPUT}/ ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  schnet_${name}  (long, 128G, 24h, GPU, ${FOLDS} folds)"
}

echo "Submitting E2E SchNet generalization test..."
echo "  Output: ${OUTPUT}/"
echo "  Models: ${MODELS} (screening)"
echo "  Folds: ${FOLDS}"
echo "  FT reps: ft_schnet, lora_chemberta, lora_molformer"
echo ""

case "$MODE" in
    all)
        submit logd logd
        submit herg herg
        submit caco2 caco2
        echo ""
        echo "  Total: 3 jobs (baselines + 3 FT reps × 3 models × 3 folds)"
        ;;
    logd|herg|caco2)
        submit "$MODE" "$MODE"
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_schnet_e2e.sh [MODE]"
        echo ""
        echo "Modes: all (default), logd, herg, caco2"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
echo ""
echo "Compare ft_schnet scores vs frozen baselines to determine"
echo "if SchNet E2E generalizes beyond quantum properties (qm9)."
