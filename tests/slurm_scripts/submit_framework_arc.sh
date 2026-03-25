#!/bin/bash
# Framework f01/f03/f04 on ARC (CPU-only, no GPU).
# Run from ARC login node: bash tests/slurm_scripts/submit_framework_arc.sh
#
# All reps should be cached — these experiments only run sklearn models.

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
OUTPUT="tests/results/framework"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

ALL_TASKS=(logd herg flab_thermo_tresanco flab_pk_jain_fcrnrt tcga_luad_vs_lusc tcga_brca_pam50 buchwald_hartwig)
declare -A SHORT_NAMES=(
    [logd]="logd"
    [herg]="herg"
    [flab_thermo_tresanco]="tres"
    [flab_pk_jain_fcrnrt]="fcrn"
    [tcga_luad_vs_lusc]="luad"
    [tcga_brca_pam50]="brca"
    [buchwald_hartwig]="buch"
)

TOTAL=0

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-128G}
    local wall=${4:-23:59:00}

    J=$(sbatch \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (long, ${mem}, ${wall}, CPU)"
    TOTAL=$((TOTAL + 1))
}

echo "============================================"
echo "Framework f01/f03/f04 on ARC (CPU-only)"
echo "============================================"
echo ""

echo "--- f01: Allocation Strategies ---"
for task in "${ALL_TASKS[@]}"; do
    submit "f01_${SHORT_NAMES[$task]}" \
        "python tests/experiments/framework/f01_allocation_strategies.py --dataset ${task} --framework --mode full --models ridge,rf,xgboost,lightgbm --output ${OUTPUT}"
done
echo ""

echo "--- f03: Greedy Step Size & Patience ---"
for task in "${ALL_TASKS[@]}"; do
    submit "f03_${SHORT_NAMES[$task]}" \
        "python tests/experiments/framework/f03_greedy_step_size.py --dataset ${task} --framework --mode full --models ridge,rf,xgboost,lightgbm --budgets 50,150,300 --max-folds 1 --output ${OUTPUT}"
done
echo ""

echo "--- f04: Two-Phase Search ---"
for task in "${ALL_TASKS[@]}"; do
    submit "f04_${SHORT_NAMES[$task]}" \
        "python tests/experiments/framework/f04_two_phase_search.py --dataset ${task} --framework --mode full --models ridge,rf,xgboost,lightgbm --budgets 50,150,300 --max-folds 1 --output ${OUTPUT}"
done
echo ""

echo "============================================"
echo "Total: ${TOTAL} jobs submitted (cancel HTC jobs first!)"
echo "============================================"
