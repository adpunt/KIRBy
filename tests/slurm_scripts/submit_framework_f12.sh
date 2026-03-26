#!/bin/bash
# Submit F12 allocation strategy v2 as SLURM jobs.
# Run from ARC login node: bash tests/slurm_scripts/submit_framework_f12.sh
#
# 6 datasets, ~4464 configs each, ~1-5h per dataset.
# CPU-only (no GPU needed).
#
# Usage:
#   bash tests/slurm_scripts/submit_framework_f12.sh          # all 6 datasets
#   bash tests/slurm_scripts/submit_framework_f12.sh herg     # single dataset

set -euo pipefail

MODE=${1:-all}
OUTPUT="tests/results/framework"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local dataset=$2

    J=$(sbatch \
        --job-name="f12_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=23:59:00 --partition=long \
        --mem=128G \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/experiments/framework/f12_allocation_v2.py --dataset ${dataset} --reps all --mode full --max-folds 1 --budgets 50,100,200,500 --output ${OUTPUT} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  f12_${name}  (long, 128G, 24h, CPU)"
}

echo "Submitting F12: Allocation Strategy v2 (budget floor)"
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    herg)       submit herg herg ;;
    buchwald)   submit buch buchwald_hartwig ;;
    tresanco)   submit tres flab_thermo_tresanco ;;
    heprt)      submit hprt flab_pk_jain_heprt ;;
    vegf)       submit vegf abagym_VEGF ;;
    brca)       submit brca tcga_brca_pam50 ;;
    all)
        submit herg herg
        submit buch buchwald_hartwig
        submit tres flab_thermo_tresanco
        submit hprt flab_pk_jain_heprt
        submit vegf abagym_VEGF
        submit brca tcga_brca_pam50
        echo ""
        echo "  Total: 6 jobs"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [all|herg|buchwald|tresanco|heprt|vegf|brca]"
        exit 1
        ;;
esac
