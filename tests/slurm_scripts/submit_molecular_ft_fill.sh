#!/bin/bash
# Fill missing molecular FT results on datasets without FT data.
#
# Already have FT results: logd, caco2, herg (from molecular_finetune_comparison.json)
# Missing: qm9, adme_hlm, adme_rlm, adme_mdr1
#
# Runs: lora_molformer, contrastive_molformer, gnn_graphmvp
# Uses find_best_hybrid_molecular.py (not run_baseline_survey.py) because
# run_baseline_survey.py doesn't support contrastive or graphmvp FT.
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_molecular_ft_fill.sh          # all 4 datasets
#   bash tests/slurm_scripts/submit_molecular_ft_fill.sh qm9      # single dataset

MODE=${1:-all}
OUTPUT="tests/results/molecular_ft_fill"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

FT_METHODS="lora_molformer,contrastive_molformer,gnn_graphmvp"

submit() {
    local name=$1
    local dataset=$2
    local wall=${3:-23:59:00}

    J=$(sbatch \
        --job-name="molft_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem=128G \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && pip install --quiet 'urllib3<2' 'peft>=0.7' 'accelerate>=0.25' bitsandbytes 2>/dev/null && mkdir -p ${OUTPUT} && python tests/find_best_hybrid_molecular.py ${dataset} --reps fast --search fast --finetune ${FT_METHODS} --output ${OUTPUT}/${dataset}_ft.json ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  molft_${name}  (long, 128G, ${wall}, GPU)"
}

echo "Submitting molecular FT fill..."
echo "  Output: ${OUTPUT}/"
echo "  FT methods: ${FT_METHODS}"
echo ""

case "$MODE" in
    all)
        submit qm9 qm9 47:59:00
        submit adme_hlm adme_hlm
        submit adme_rlm adme_rlm
        submit adme_mdr1 adme_mdr1
        echo ""
        echo "  Total: 4 jobs"
        ;;
    qm9|adme_hlm|adme_rlm|adme_mdr1)
        local_wall=23:59:00
        [ "$MODE" = "qm9" ] && local_wall=47:59:00
        submit "$MODE" "$MODE" "$local_wall"
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_molecular_ft_fill.sh [MODE]"
        echo ""
        echo "Modes: all (default), qm9, adme_hlm, adme_rlm, adme_mdr1"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
