#!/bin/bash
# F15: Adaptive Pruning Validation — one job per dataset × model
# 36 jobs total. Dynamic budget capping keeps each job under ~20h.
# Results saved every 100 configs (no data lost on timeout).
#
# Output: tests/results/framework/f15_adaptive_pruning/{dataset}_{model}.json

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP="cd /data/stat-cadd/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"

echo "=== F15 Adaptive Pruning Validation (6 datasets × 6 models = 36 jobs) ==="

for ds in abagym_EGFR buchwald_hartwig caco2 flab_pk_jain_fcrnrt herg_ki logd; do
    for model in rf xgboost lightgbm ridge svm mlp; do
        J=$(sbatch \
            --job-name="f15_${ds}_${model}" \
            --nodes=1 --ntasks=1 --cpus-per-task=4 \
            --time=24:00:00 --partition=medium \
            --mem=64G \
            --mail-user="${EMAIL}" --mail-type=FAIL \
            --wrap="${WRAP} && python tests/experiments/framework/f15_adaptive_pruning_validation.py ${ds} --model ${model}" \
            | awk '{print $4}')
        echo "  $J  f15_${ds}_${model}"
    done
done

echo ""
echo "Monitor: squeue -u \$USER | grep f15_"
