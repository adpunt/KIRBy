#!/bin/bash
# Submit framework experiments (f00-f08) as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_framework.sh [mode]
#
# Wave 1: f00, f01, f03, f04, f05, f06, f07, f08  (all independent)
# Wave 2: f02 consuming f01+f03 JSON output        (run after wave 1)
# Local:  f09 cross-domain synthesis                (run locally after wave 2)
#
# Usage:
#   bash tests/slurm_scripts/submit_framework.sh wave1     # all wave 1 (56 jobs)
#   bash tests/slurm_scripts/submit_framework.sh wave2     # f02 from JSON (14 jobs)
#   bash tests/slurm_scripts/submit_framework.sh f01       # just f01 (7 jobs)
#   bash tests/slurm_scripts/submit_framework.sh f01 logd  # just f01 on logd (1 job)
#
# Resources:
#   f00, f05, f06, f07:  128G, 12h, short partition
#   f01, f03, f04, f08:  256G, 24h, long partition
#   f02 (wave 2):         64G,  2h, short partition

set -euo pipefail

MODE=${1:-wave1}
TASK_FILTER=${2:-all}

OUTPUT="tests/results/framework"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# All 7 framework tasks
ALL_TASKS=(logd herg flab_thermo_tresanco flab_pk_jain_fcrnrt tcga_luad_vs_lusc tcga_brca_pam50 buchwald_hartwig)

# Short names for SLURM job names
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
    local jobname=$1
    local mem=$2
    local timelimit=$3
    local cmd=$4

    # Pick partition based on time limit (short ≤12h, long >12h)
    local hours=${timelimit%%:*}
    local partition="long"
    if [ "$hours" -le 12 ]; then
        partition="short"
    fi

    J=$(sbatch \
        --job-name="${jobname}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${timelimit}" --partition="${partition}" \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${jobname}  (${mem}, ${timelimit}, ${partition})"
    TOTAL=$((TOTAL + 1))
}

get_tasks() {
    if [ "$TASK_FILTER" = "all" ]; then
        echo "${ALL_TASKS[@]}"
    else
        echo "$TASK_FILTER"
    fi
}

# =========================================================================
# WAVE 1: Independent experiments (f00, f01, f03, f04, f05, f06, f07, f08)
# =========================================================================

run_f00() {
    echo "--- f00: Rep Screening ---"
    for task in $(get_tasks); do
        submit "f00_${SHORT_NAMES[$task]}" 128G 11:59:00 \
            "python tests/experiments/framework/f00_rep_screening.py --dataset ${task} --framework --mode full --output ${OUTPUT}"
    done
    echo ""
}

run_f01() {
    echo "--- f01: Allocation Strategies ---"
    for task in $(get_tasks); do
        submit "f01_${SHORT_NAMES[$task]}" 128G 11:59:00 \
            "python tests/experiments/framework/f01_allocation_strategies.py --dataset ${task} --framework --mode full --models ridge,rf,xgboost,lightgbm --budgets 50,150,300 --max-folds 1 --output ${OUTPUT}"
    done
    echo ""
}

run_f03() {
    echo "--- f03: Greedy Step Size & Patience ---"
    for task in $(get_tasks); do
        submit "f03_${SHORT_NAMES[$task]}" 128G 11:59:00 \
            "python tests/experiments/framework/f03_greedy_step_size.py --dataset ${task} --framework --mode full --models ridge,rf,xgboost,lightgbm --budgets 50,150,300 --max-folds 1 --output ${OUTPUT}"
    done
    echo ""
}

run_f04() {
    echo "--- f04: Two-Phase Search ---"
    for task in $(get_tasks); do
        submit "f04_${SHORT_NAMES[$task]}" 128G 11:59:00 \
            "python tests/experiments/framework/f04_two_phase_search.py --dataset ${task} --framework --mode full --models ridge,rf,xgboost,lightgbm --budgets 50,150,300 --max-folds 1 --output ${OUTPUT}"
    done
    echo ""
}

run_f05() {
    echo "--- f05: Importance Correlation ---"
    for task in $(get_tasks); do
        submit "f05_${SHORT_NAMES[$task]}" 128G 11:59:00 \
            "python tests/experiments/framework/f05_importance_correlation.py --dataset ${task} --framework --mode full --output ${OUTPUT}"
    done
    echo ""
}

run_f06() {
    echo "--- f06: Augmentation Test ---"
    for task in $(get_tasks); do
        submit "f06_${SHORT_NAMES[$task]}" 128G 11:59:00 \
            "python tests/experiments/framework/f06_augmentation_test.py --dataset ${task} --framework --mode full --output ${OUTPUT}"
    done
    echo ""
}

run_f07() {
    echo "--- f07: Internal Evaluator ---"
    for task in $(get_tasks); do
        submit "f07_${SHORT_NAMES[$task]}" 128G 11:59:00 \
            "python tests/experiments/framework/f07_internal_evaluator.py --dataset ${task} --framework --mode full --output ${OUTPUT}"
    done
    echo ""
}

run_f08() {
    echo "--- f08: Iterative Augmentation ---"
    for task in $(get_tasks); do
        submit "f08_${SHORT_NAMES[$task]}" 256G 23:59:00 \
            "python tests/experiments/framework/f08_iterative_augmentation.py --dataset ${task} --framework --mode full --output ${OUTPUT}"
    done
    echo ""
}

# =========================================================================
# WAVE 2: f02 adaptive pruning (consumes f01 + f03 JSON)
# =========================================================================

run_f02() {
    echo "--- f02: Adaptive Pruning (from f01 + f03 JSON) ---"
    for task in $(get_tasks); do
        local f01_json="${OUTPUT}/f01_allocation_strategies/${task}.json"
        local f03_json="${OUTPUT}/f03_greedy_step_size/${task}.json"

        # One job consuming f01, one consuming f03
        submit "f02f1_${SHORT_NAMES[$task]}" 64G 01:59:00 \
            "python tests/experiments/framework/f02_adaptive_pruning.py --from-json ${f01_json} --output ${OUTPUT}"
        submit "f02f3_${SHORT_NAMES[$task]}" 64G 01:59:00 \
            "python tests/experiments/framework/f02_adaptive_pruning.py --from-json ${f03_json} --output ${OUTPUT}"
    done
    echo ""
}

# =========================================================================
# Dispatch
# =========================================================================

echo "============================================"
echo "Framework Experiments"
echo "  Output: ${OUTPUT}/"
echo "  Tasks: $(get_tasks)"
echo "============================================"
echo ""

case "$MODE" in
    wave1)
        run_f00
        run_f01
        run_f03
        run_f04
        run_f05
        run_f06
        run_f07
        run_f08
        ;;
    wave2)
        run_f02
        ;;
    f00) run_f00 ;;
    f01) run_f01 ;;
    f02) run_f02 ;;
    f03) run_f03 ;;
    f04) run_f04 ;;
    f05) run_f05 ;;
    f06) run_f06 ;;
    f07) run_f07 ;;
    f08) run_f08 ;;
    all)
        run_f00
        run_f01
        run_f03
        run_f04
        run_f05
        run_f06
        run_f07
        run_f08
        echo ">>> Wave 1 submitted. Run 'bash $0 wave2' after wave 1 completes."
        echo ""
        ;;
    *)
        echo "Usage: bash $0 [wave1|wave2|all|f00|f01|f02|f03|f04|f05|f06|f07|f08] [task]"
        echo ""
        echo "Modes:"
        echo "  wave1   Submit f00,f01,f03,f04,f05,f06,f07,f08 (56 jobs)"
        echo "  wave2   Submit f02 from f01+f03 JSON (14 jobs)"
        echo "  all     Same as wave1 (reminds you to run wave2 after)"
        echo "  f00-f08 Submit a single experiment (7 jobs)"
        echo ""
        echo "Optional task filter (default: all 7):"
        echo "  logd, herg, flab_thermo_tresanco, flab_pk_jain_fcrnrt,"
        echo "  tcga_luad_vs_lusc, tcga_brca_pam50, buchwald_hartwig"
        exit 1
        ;;
esac

echo "============================================"
echo "Total: ${TOTAL} jobs submitted"
echo "============================================"
