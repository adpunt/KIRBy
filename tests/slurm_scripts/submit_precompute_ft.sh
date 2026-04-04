#!/bin/bash
# Precompute fine-tuned representations — one SLURM job per (dataset × rep).
# Each job needs a GPU for fine-tuning. Cache files land in
# ~/.cache/kirby/finetuned_reps/ (or $KIRBY_CACHE_DIR/finetuned_reps/).
#
# After these complete, run precompute_hybrids.py to include FT reps
# in hybrid search results (cache hit, no GPU needed).
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute_ft.sh              # all 12 jobs
#   bash tests/slurm_scripts/submit_precompute_ft.sh molecular     # 7 molecular
#   bash tests/slurm_scripts/submit_precompute_ft.sh antibody      # 5 antibody

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
COUNT=0

submit() {
    local name=$1
    local dataset=$2
    local rep=$3
    local mem=${4:-128G}
    local time_limit=${5:-12:00:00}

    J=$(sbatch \
        --job-name="ft_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --gres=gpu:1 \
        --time="${time_limit}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && python tests/precompute_ft_reps.py ${dataset} --rep ${rep}" \
        | awk '{print $4}')
    echo "  $J  ft_${name}"
    COUNT=$((COUNT + 1))
}

# ── Molecular (7 jobs) ────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "molecular" ]]; then
    echo "=== Molecular FT ==="
    submit "logd_lora_mf"       logd      lora_molformer
    submit "logd_contr_mf"      logd      contrastive_molformer
    submit "logd_gnn_gmvp"      logd      gnn_graphmvp
    submit "caco2_lora_mf"      caco2     lora_molformer
    submit "qm9_lora_mf"        qm9       lora_molformer
    submit "herg_gnn_gmvp"      herg_ki   gnn_graphmvp
    submit "mdr1_gnn_gmvp"      adme_mdr1 gnn_graphmvp
fi

# ── Antibody (5 jobs) ─────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "antibody" ]]; then
    echo "=== Antibody FT ==="
    submit "phil_h3_lora"       flab_binding_phillips_h3 antibody_lora_esm2
    submit "ang2_lora"          abagym_Ang2              antibody_lora_esm2
    submit "egfr_lora"          abagym_EGFR              antibody_lora_esm2
    submit "her2_lora"          abagym_HER2              antibody_lora_esm2
    submit "abcov_lora"         abcov_kd                 antibody_lora_esm2
fi

echo ""
echo "Submitted $COUNT FT precompute jobs"
