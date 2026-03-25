#!/bin/bash
# Framework f01/f03/f04 on ARC (CPU-only, no GPU).
# Run from ARC login node:
#   bash tests/slurm_scripts/submit_framework_arc.sh              # all framework tasks
#   bash tests/slurm_scripts/submit_framework_arc.sh rep          # 7 representative datasets
#   bash tests/slurm_scripts/submit_framework_arc.sh logd         # single task

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
OUTPUT="tests/results/framework"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

ALL_TASKS=(logd herg flab_thermo_tresanco flab_pk_jain_fcrnrt flab_pk_jain_heprt abagym_EGFR tz_her2_binary ginkgo_hic tcga_luad_vs_lusc tcga_brca_pam50 buchwald_hartwig)

# Representative 7: one per domain type (from EXPERIMENT_DESIGN_DISCUSSION.md §7)
REP_TASKS=(herg flab_thermo_tresanco abagym_EGFR tz_her2_binary ginkgo_hic tcga_luad_vs_lusc flab_pk_jain_heprt)

declare -A SHORT_NAMES=(
    [logd]="logd"
    [herg]="herg"
    [flab_thermo_tresanco]="tres"
    [flab_pk_jain_fcrnrt]="fcrn"
    [flab_pk_jain_heprt]="hprt"
    [abagym_EGFR]="egfr"
    [tz_her2_binary]="tzh2"
    [ginkgo_hic]="ghic"
    [tcga_luad_vs_lusc]="luad"
    [tcga_brca_pam50]="brca"
    [buchwald_hartwig]="buch"
)

FILTER=${1:-all}
case "$FILTER" in
    all) TASKS=("${ALL_TASKS[@]}") ;;
    rep) TASKS=("${REP_TASKS[@]}") ;;
    *)   TASKS=("$FILTER") ;;
esac

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
echo "  Tasks: ${TASKS[*]}"
echo "============================================"
echo ""

echo "--- f01: Allocation Strategies (tiered) ---"
for task in "${TASKS[@]}"; do
    submit "f01_${SHORT_NAMES[$task]}" \
        "python tests/experiments/framework/f01_allocation_strategies.py --dataset ${task} --framework --mode full --models ridge,rf,xgboost,lightgbm --budgets 50,150,300 --max-folds 1 --output ${OUTPUT}"
done
echo ""

echo "--- f03: Greedy Step Size & Patience (tiered) ---"
for task in "${TASKS[@]}"; do
    submit "f03_${SHORT_NAMES[$task]}" \
        "python tests/experiments/framework/f03_greedy_step_size.py --dataset ${task} --framework --mode full --models ridge,rf,xgboost,lightgbm --budgets 50,150,300 --max-folds 1 --output ${OUTPUT}"
done
echo ""

echo "--- f04: Two-Phase Search ---"
for task in "${TASKS[@]}"; do
    submit "f04_${SHORT_NAMES[$task]}" \
        "python tests/experiments/framework/f04_two_phase_search.py --dataset ${task} --framework --mode full --models ridge,rf,xgboost,lightgbm --budgets 50,150,300 --max-folds 1 --output ${OUTPUT}"
done
echo ""

echo "============================================"
echo "Total: ${TOTAL} jobs submitted"
echo "============================================"
