#!/bin/bash
# C02 full submission: frozen vs frozen+FT comparison on all FT-having datasets.
#
# 12 datasets × 1 job = 12 GPU jobs.
# Mode: full (5-fold CV). Reps: all.
#
# Pre-requisites:
#   - Stage 2 bug test PASSED (tests/slurm_scripts/submit_c02_bugtest.sh)
#   - FT cache fully warm (7516559 + 7528901/2 complete)
#   - Antibody FT cache warm (7520844 complete)
#
# Usage (from /data/stat-cadd/scat9264/KIRBy on HTC login):
#   bash tests/slurm_scripts/submit_c02_full.sh

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
OUT_DIR="tests/results/complexity_c02/full"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error && mkdir -p ${OUT_DIR}"

# 7 molecular + 5 antibody = 12 datasets with ft_reps configured.
DATASETS=(
    # Molecular
    caco2
    logd
    qm9
    herg_ki
    adme_mdr1
    adme_hlm
    adme_rlm
    # Antibody
    abcov_kd
    flab_binding_phillips_h3
    abagym_Ang2
    abagym_EGFR
    abagym_HER2
)

echo "=== C02 Full — frozen vs frozen+FT on ${#DATASETS[@]} datasets ==="
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
echo "Outputs: ${OUT_DIR}/<dataset>_c02_complexity_matching_grid.json"
echo "Each JSON includes conditions.no_aug (frozen hybrid) and"
echo "conditions.frozen_plus_ft (frozen+FT hybrid) per downstream model."
