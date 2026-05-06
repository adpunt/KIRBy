#!/bin/bash
# CGR comparison resubmit — buchwald_hartwig + bh_doyle_plates timed out at
# the 24h wall in jobs 7548433/7548434. Bumped to 4-day wall (long partition max).
# Suzuki (7548435) completed in 9h18m — do NOT resubmit it.
#
# Usage (from /data/stat-ecr/scat9264/KIRBy on HTC login):
#   bash tests/slurm_scripts/submit_cgr_comparison_resubmit.sh           # both
#   bash tests/slurm_scripts/submit_cgr_comparison_resubmit.sh bh        # BH only
#   bash tests/slurm_scripts/submit_cgr_comparison_resubmit.sh doyle     # Doyle only

set -euo pipefail

MODE=${1:-all}
OUTPUT="tests/results/comparisons"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"

submit() {
    local name=$1
    local dataset=$2

    JOB=$(sbatch --parsable \
        --job-name="cgr_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="3-23:59:00" --partition=long \
        --mem="256G" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/experiments/opig_cgr_comparison.py --datasets ${dataset} --n-folds 5 --output ${OUTPUT}/opig_cgr_comparison_${dataset}.json")
    echo "  ${JOB}  cgr_${name}  (long, 256G, 4d)"
}

echo "=== CGR comparison resubmit — extended wall (4d vs 24h) ==="
echo ""

case "$MODE" in
    all)
        submit bh buchwald_hartwig
        submit doyle bh_doyle_plates
        ;;
    bh)    submit bh buchwald_hartwig ;;
    doyle) submit doyle bh_doyle_plates ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_cgr_comparison_resubmit.sh [all|bh|doyle]"
        exit 1
        ;;
esac

echo ""
echo "Outputs: ${OUTPUT}/opig_cgr_comparison_<dataset>.json"
