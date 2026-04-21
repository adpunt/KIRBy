#!/bin/bash
# FT precompute fill-in for adme_hlm + adme_rlm.
#
# Context: 7516559 covered 5 molecular datasets (herg_ki, caco2, logd, qm9,
# adme_mdr1). adme_hlm / adme_rlm had no ft_reps configured. adme_solubility
# was dropped (R²=0.32 ceiling, not viable).
#
# This script populates the per-fold .npz cache for the 3 successful FT reps
# (lora_molformer, contrastive_molformer, gnn_graphmvp) on adme_hlm + adme_rlm.
#
# Usage (from /data/stat-cadd/scat9264/KIRBy on HTC login):
#   bash tests/slurm_scripts/submit_ft_precompute_adme.sh

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"

submit_gpu() {
    local name=$1
    local cmd=$2

    JOB=$(sbatch --parsable \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="23:59:00" --partition=long \
        --mem="128G" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}")
    echo "  ${JOB}  ${name}  (long, 128G, 24h, GPU)"
}

echo "=== FT precompute fill-in: adme_hlm + adme_rlm ==="
echo ""

submit_gpu "ftpc_adme_hlm" \
    "python tests/precompute_ft_reps.py adme_hlm --speed full"

submit_gpu "ftpc_adme_rlm" \
    "python tests/precompute_ft_reps.py adme_rlm --speed full"

echo ""
echo "Cache target: /data/stat-cadd/scat9264/.cache/kirby/finetuned_reps/"
echo "Monitor: squeue -u \$USER"
