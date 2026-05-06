#!/bin/bash
# DTA baseline re-survey under the new per-fold importance filter
# (commit 506e4bb) — high-dim reps (mordred, ESM-family) reduced to top-200
# features per fold to fix R² blowups (drug_mordred+ridge R²=-40,
# drug_mordred+mlp R²=-2247 on bindingdb_kd).
#
# Re-runs the same 3 DTA datasets the FT precompute v2 jobs covered:
#   - davis (was 7521193, 9h04m)
#   - kiba  (was 7521194, 13h54m)
#   - bindingdb_kd (was 7521196, 16h55m)
#
# Output overwrites the JSONs at tests/results/phase5_baselines/dta/
# (FT cache from the v2 jobs is reused — no re-fine-tuning needed).
#
# Usage (from /data/stat-ecr/scat9264/KIRBy on HTC login):
#   bash tests/slurm_scripts/submit_dta_baseline_resurvey.sh

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
DTA_OUT="tests/results/phase5_baselines/dta"

submit_gpu() {
    local name=$1
    local cmd=$2

    JOB=$(sbatch --parsable \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="3-23:59:00" --partition=long \
        --mem="256G" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}")
    echo "  ${JOB}  ${name}  (long, 256G, 4d, GPU)"
}

echo "=== DTA Baseline Re-Survey — importance-filter fix (commit 506e4bb) ==="
echo ""

mkdir -p "${DTA_OUT}"
for ds in davis kiba bindingdb_kd; do
    submit_gpu "dtaresurvey_${ds}" \
        "mkdir -p ${DTA_OUT} && python tests/run_baseline_survey.py ${ds} --finetune --n-folds 5 --output ${DTA_OUT}/"
done

echo ""
echo "Total: 3 GPU jobs"
echo "Output: ${DTA_OUT}/{davis,kiba,bindingdb_kd}.json (overwrite)"
echo "FT cache reused from v2 (\$KIRBY_CACHE_DIR/finetuned_reps/) — no refit."
echo "Expected runtime: 9-17h per dataset (similar to v2 timings)."
