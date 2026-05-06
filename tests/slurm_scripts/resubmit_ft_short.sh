#!/bin/bash
# Resubmit GPU fine-tuning jobs targeting SHORT partition (12h, idle nodes).
# Split by estimated runtime: small datasets → short, large → medium.
#
# Usage:
#   bash tests/slurm_scripts/resubmit_ft_short.sh

set -euo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"
PIP_DEPS="true"  # deps should already be in venv; inline pip install was breaking torch

AB_OUTPUT="tests/results/antibody_ft_fill"
AB_MODELS="ridge,rf,xgboost"
AB_FOLDS=3

submit() {
    local name=$1
    local cmd=$2
    local mem=$3
    local wall=$4
    local partition=$5

    J=$(sbatch --parsable \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition="${partition}" \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}")
    echo "  $J  ${name}  (${partition}, ${mem}, ${wall})"
}

ab_ft() {
    local ds=$1
    local mem=$2
    local wall=$3
    local partition=$4
    submit "abft_${ds}" \
        "${PIP_DEPS} && mkdir -p ${AB_OUTPUT} && python tests/antibody_baseline_survey.py ${ds} --finetune --models ${AB_MODELS} --n-folds ${AB_FOLDS} --output ${AB_OUTPUT}/" \
        "${mem}" "${wall}" "${partition}"
}

echo "============================================"
echo "GPU FT Resubmission — targeting short partition"
echo "============================================"
echo ""

# ── Ginkgo (6 datasets, n=94-242) → SHORT, ~1-2h each ──
echo "--- Ginkgo FT (6 jobs, short, ~1-2h each) ---"
for ds in ginkgo_hic ginkgo_acsins ginkgo_colloidal ginkgo_heparin ginkgo_pr_cho ginkgo_titer; do
    ab_ft "$ds" 64G 11:59:00 short
done
echo ""

# ── AbAgym DMS (5 datasets, n=200-700) → SHORT, ~2-4h each ──
echo "--- AbAgym DMS FT (5 jobs, short, ~2-4h each) ---"
for ds in abagym_Ang2 abagym_EGFR abagym_lysozyme abagym_VEGF abagym_HER2; do
    ab_ft "$ds" 64G 11:59:00 short
done
echo ""

# ── Small inter-family (8 datasets, n=130-1000) → SHORT, ~3-6h each ──
echo "--- Small inter-family FT (8 jobs, short, ~3-6h each) ---"
for ds in flab_thermo_tresanco flab_thermo_jain flab_aggregation_jain \
          flab_aggregation_jain_hicrt flab_immunogenicity_marks \
          flab_polyreact_jain_bvp flab_polyreact_jain_cic abcov_kd; do
    ab_ft "$ds" 64G 11:59:00 short
done
echo ""

# ── CLF humanness (1 dataset, n~10K but binary → fast) → SHORT ──
echo "--- CLF FT (1 job, short) ---"
ab_ft humanness_therapeutic 64G 11:59:00 short
echo ""

# ── SchNet small datasets → SHORT ──
echo "--- SchNet FT (herg + caco2, short, ~4-6h each) ---"
SCHNET_OUTPUT="tests/results/schnet_e2e"
submit schnet_herg \
    "mkdir -p ${SCHNET_OUTPUT} && python tests/run_baseline_survey.py herg --finetune --models ridge,rf,xgboost --n-folds 3 --output ${SCHNET_OUTPUT}/" \
    64G 11:59:00 short
submit schnet_caco2 \
    "mkdir -p ${SCHNET_OUTPUT} && python tests/run_baseline_survey.py caco2 --finetune --models ridge,rf,xgboost --n-folds 3 --output ${SCHNET_OUTPUT}/" \
    64G 11:59:00 short
echo ""

echo "=== SHORT TOTAL: 22 jobs (can start on idle nodes) ==="
echo ""

# ── Larger datasets → MEDIUM (48h) ──
echo "--- Larger FT jobs (medium partition) ---"

# FLAb PK (n~1500-2000, more complex)
for ds in flab_pk_jain_fcrnrt flab_pk_jain_heprt; do
    ab_ft "$ds" 128G 1-23:59:00 medium
done

# DDG combined (n~1500, grouped CV)
ab_ft ddg_combined 128G 1-23:59:00 medium

# FLAb large DMS (n~2000-5000)
for ds in flab_expression_koenig flab_binding_koenig flab_binding_phillips_h3 \
          flab_binding_adams flab_expression_adams; do
    ab_ft "$ds" 128G 1-23:59:00 medium
done

# Trastuzumab binary (n=5000)
ab_ft tz_her2_binary 128G 1-23:59:00 medium

# SchNet logd (n=4200)
submit schnet_logd \
    "mkdir -p ${SCHNET_OUTPUT} && python tests/run_baseline_survey.py logd --finetune --models ridge,rf,xgboost --n-folds 3 --output ${SCHNET_OUTPUT}/" \
    128G 1-23:59:00 medium

# Molecular FT (all 4)
MOLFT_OUTPUT="tests/results/molecular_ft_fill"
MOLFT_METHODS="lora_molformer,contrastive_molformer,gnn_graphmvp"
for ds in adme_hlm adme_rlm adme_mdr1; do
    submit "molft_${ds}" \
        "${PIP_DEPS} && mkdir -p ${MOLFT_OUTPUT} && python tests/find_best_hybrid_molecular.py ${ds} --reps fast --search fast --finetune ${MOLFT_METHODS} --output ${MOLFT_OUTPUT}/${ds}_ft.json" \
        128G 1-23:59:00 medium
done
submit molft_qm9 \
    "${PIP_DEPS} && mkdir -p ${MOLFT_OUTPUT} && python tests/find_best_hybrid_molecular.py qm9 --reps fast --search fast --finetune ${MOLFT_METHODS} --output ${MOLFT_OUTPUT}/qm9_ft.json" \
    128G 1-23:59:00 medium

echo ""
echo "=== MEDIUM TOTAL: 14 jobs ==="
echo ""
echo "=== GRAND TOTAL: 36 GPU jobs (22 short + 14 medium) ==="
