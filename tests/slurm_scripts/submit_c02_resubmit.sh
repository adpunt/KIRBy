#!/bin/bash
# C02 resubmit — 8 datasets that didn't complete in the 2026-04-23 batch.
#
# Bugs fixed since previous run (must be on HEAD):
#   - fb0acb0: UnboundLocalError '_prune_frac' on checkpoint-resume
#   - <local c02 fix>: defensive baselines[model_name] indexing in c02_complexity_grid.py
#
# Resubmit list (10 originally → 8 here, after dropping the 4 completes):
#   timeouts (7): caco2, logd, qm9, adme_mdr1, adme_hlm, adme_rlm,
#                 flab_binding_phillips_h3
#   failed   (1): abagym_Ang2
#
# Skipped (already complete, do NOT resubmit):
#   herg_ki, abcov_kd, abagym_EGFR, abagym_HER2
#
# Usage (from /data/stat-ecr/scat9264/KIRBy on HTC login):
#   bash tests/slurm_scripts/submit_c02_resubmit.sh

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
OUT_DIR="tests/results/complexity_c02/full"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error && mkdir -p ${OUT_DIR}"

DATASETS=(
    # Molecular timeouts
    caco2
    logd
    qm9
    adme_mdr1
    adme_hlm
    adme_rlm
    # Antibody timeout
    flab_binding_phillips_h3
    # Antibody fail (Ang2 hit the _prune_frac bug at fold-0 resume)
    abagym_Ang2
)

echo "=== C02 Resubmit — ${#DATASETS[@]} datasets (post-bugfixes) ==="
echo ""

for ds in "${DATASETS[@]}"; do
    JOB=$(sbatch --parsable \
        --job-name="c02_${ds}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="3-23:59:00" --partition=long \
        --mem="128G" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && python tests/experiments/complexity/c02_complexity_grid.py --dataset ${ds} --mode full --reps all --output ${OUT_DIR}")
    echo "  ${JOB}  c02_${ds}  (long, 128G, 4d, GPU)"
done

echo ""
echo "Outputs: ${OUT_DIR}/c02_complexity_matching_grid/<dataset>.json"
echo ""
echo "Note: existing checkpoint files in \$KIRBY_CACHE_DIR/searches/ will be"
echo "reused — _prune_frac fix means resume now works. If anything looks off"
echo "after a few hours, check the new slurm logs for the 8 jobs above."
