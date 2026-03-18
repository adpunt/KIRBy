#!/bin/bash
# Re-run ONLY the C_relative screening option from f00 across all 7 tasks.
# Tests the sign-bug fix (gap-based threshold) with expanded fractions.
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_f00_crelative.sh
#
# C_relative is lightweight (no hybrid search for screened-out fractions),
# so we can use short partition with modest resources.

set -euo pipefail

OUTPUT="tests/results/framework/f00_crelative_retest"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
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

echo "============================================"
echo "F00 C_relative retest (sign-bug fix)"
echo "  Output: ${OUTPUT}/"
echo "============================================"
echo ""

for task in "${ALL_TASKS[@]}"; do
    J=$(sbatch \
        --job-name="f00c_${SHORT_NAMES[$task]}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=05:59:00 --partition=short \
        --mem=128G \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/experiments/framework/f00_rep_screening.py --dataset ${task} --framework --mode full --output ${OUTPUT} --only-option C ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  f00c_${SHORT_NAMES[$task]}  (128G, 6h, short)"
    TOTAL=$((TOTAL + 1))
done

echo ""
echo "============================================"
echo "Total: ${TOTAL} jobs submitted"
echo "============================================"
