#!/bin/bash
# F15: Adaptive Pruning Validation
# Full search tier on 6 datasets, no augmentation, per-fold scores.
# Used to validate pruning rules before implementing in find_best_hybrid.

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP="cd /data/stat-cadd/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && export PYTHONUNBUFFERED=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"

echo "=== F15 Adaptive Pruning Validation (6 datasets) ==="

for ds in herg_ki caco2 logd abagym_EGFR buchwald_hartwig flab_pk_jain_fcrnrt; do
    J=$(sbatch \
        --job-name="f15_${ds}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time=24:00:00 --partition=medium \
        --mem=64G \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP} && sleep \$((RANDOM % 60)) && python tests/experiments/framework/f15_adaptive_pruning_validation.py ${ds}" \
        | awk '{print $4}')
    echo "  $J  f15_${ds}"
done

echo ""
echo "Monitor: squeue -u \$USER | grep f15_"
