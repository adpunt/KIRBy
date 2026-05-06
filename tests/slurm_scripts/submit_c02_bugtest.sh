#!/bin/bash
# C02 bug test: verify the frozen+FT search pass wires up correctly.
#
# Target: caco2 (smallest FT-having dataset, 1 FT rep = lora_molformer).
# Mode: intermediary (1-fold) — cheap smoke test.
# Expected: output JSON has conditions.frozen_plus_ft populated.
#
# Run AFTER the FT cache is fully warm:
#   - 7516559 (molecular general) ✅
#   - 7528901 ftpc_adme_hlm
#   - 7528902 ftpc_adme_rlm
# (caco2's FT cache is from 7516559 — already warm, this script can run now.)
#
# Usage (from /data/stat-ecr/scat9264/KIRBy on HTC login):
#   bash tests/slurm_scripts/submit_c02_bugtest.sh

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
OUT_DIR="tests/results/complexity_c02/bugtest"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error && mkdir -p ${OUT_DIR}"

JOB=$(sbatch --parsable \
    --job-name="c02_bugtest_caco2" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --time="02:00:00" --partition=long \
    --mem="128G" \
    --gres=gpu:1 \
    --mail-user="${EMAIL}" --mail-type=END,FAIL \
    --wrap="${WRAP_PREFIX} && python tests/experiments/complexity/c02_complexity_grid.py --dataset caco2 --mode intermediary --reps all --output ${OUT_DIR}")

echo "${JOB}  c02_bugtest_caco2  (long, 128G, 2h, GPU)"
echo ""
echo "When done, verify with:"
echo "  python -c \"import json; d=json.load(open('${OUT_DIR}/caco2_c02_complexity_matching_grid.json')); print(list(d.get('conditions', {}).keys()))\""
echo ""
echo "Expected output: ['no_aug', 'uncapped', 'capped', 'top_tier_only', 'frozen_plus_ft']"
echo "If 'frozen_plus_ft' is missing → bug in wiring."
