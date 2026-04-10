#!/bin/bash
# Submit CGR vs SMILES comparison experiment (Phase 3 of CGR_EMBEDDING_RESEARCH.md).
# One GPU job per dataset — GPU needed for chytorch_graphormer, rxnfp, reactiont5.
#
# Tests whether CGR representations add unique signal beyond SMILES-based reps
# for reaction yield prediction. Two families:
#   Fingerprint: cgr_morgan vs diff_ecfp4 vs drfp_2048
#   Neural:      chytorch_graphormer vs rxnfp vs reactiont5
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_cgr_comparison.sh           # all 3 datasets
#   bash tests/slurm_scripts/submit_cgr_comparison.sh bh        # buchwald_hartwig only
#   bash tests/slurm_scripts/submit_cgr_comparison.sh suzuki    # suzuki only
#   bash tests/slurm_scripts/submit_cgr_comparison.sh doyle     # bh_doyle_plates only

MODE=${1:-all}
OUTPUT="tests/results/comparisons"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local dataset=$2

    J=$(sbatch \
        --job-name="cgr_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=23:59:00 --partition=long \
        --mem=256G \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/experiments/opig_cgr_comparison.py --datasets ${dataset} --n-folds 5 --output ${OUTPUT}/opig_cgr_comparison_${dataset}.json ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  cgr_${name}  (long, 256G, 24h)"
}

echo "Submitting CGR vs SMILES comparison experiment..."
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    all)
        submit bh buchwald_hartwig
        submit doyle bh_doyle_plates
        submit suzuki suzuki
        echo ""
        echo "  Total: 3 jobs (all reps × 6 models × 5-fold + hybrid comparison)"
        ;;
    bh)
        submit bh buchwald_hartwig
        ;;
    suzuki)
        submit suzuki suzuki
        ;;
    doyle)
        submit doyle bh_doyle_plates
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_cgr_comparison.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all      All 3 reaction datasets (default)"
        echo "  bh       Buchwald-Hartwig only"
        echo "  suzuki   Suzuki only"
        echo "  doyle    BH Doyle plates only"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
