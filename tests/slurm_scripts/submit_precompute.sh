#!/bin/bash
# Precompute hybrid searches — one SLURM job per (dataset × condition).
# 6 conditions per dataset = 6 parallel jobs per dataset.
#
# Conditions:
#   general_none       — general reps, no augmentation
#   general_uncapped   — general reps, uncapped augmentation
#   general_capped     — general reps, capped augmentation
#   top_none           — top_tier reps, no augmentation
#   top_uncapped       — top_tier reps, uncapped augmentation
#   top_capped         — top_tier reps, capped augmentation
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute.sh              # all
#   bash tests/slurm_scripts/submit_precompute.sh molecular     # one domain
#   bash tests/slurm_scripts/submit_precompute.sh antibody

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
COUNT=0

CONDITIONS="general_none general_uncapped general_capped top_none top_uncapped top_capped"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}
    local time_limit=${4:-12:00:00}

    J=$(sbatch \
        --job-name="pc_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${time_limit}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}" \
        | awk '{print $4}')
    echo "  $J  pc_${name}"
    COUNT=$((COUNT + 1))
}

submit_dataset() {
    local ds=$1
    local mem=${2:-256G}
    local time_limit=${3:-12:00:00}

    for cond in ${CONDITIONS}; do
        submit "${ds}_${cond}" \
            "python tests/precompute_hybrids.py ${ds} --condition ${cond} --speed fast" \
            "${mem}" "${time_limit}"
    done
}

echo "============================================"
echo "Hybrid Precompute (1 job per dataset×condition)"
echo "============================================"
echo ""

run_molecular() {
    echo "--- Molecular (7 × 6 = 42 jobs) ---"
    for ds in herg_ki caco2 logd qm9 adme_hlm adme_rlm adme_mdr1; do
        submit_dataset "${ds}" 256G 12:00:00
    done
    echo ""
}

run_antibody() {
    echo "--- Antibody Inter-Family (11 × 6 = 66 jobs) ---"
    for ds in flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt \
              flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain \
              abcov_kd flab_aggregation_jain flab_immunogenicity_marks \
              flab_polyreact_jain_bvp ddg_combined; do
        submit_dataset "${ds}" 256G 12:00:00
    done

    echo "--- Antibody DMS (5 × 6 = 30 jobs) ---"
    for ds in flab_binding_koenig flab_expression_koenig flab_binding_adams \
              flab_expression_adams flab_binding_phillips_h3; do
        submit_dataset "${ds}" 256G 12:00:00
    done

    echo "--- AbAgym (5 × 6 = 30 jobs) ---"
    for ds in abagym_Ang2 abagym_EGFR abagym_HER2 abagym_lysozyme abagym_VEGF; do
        submit_dataset "${ds}" 256G 8:00:00
    done

    echo "--- Antibody CLF + Ginkgo (8 × 6 = 48 jobs) ---"
    for ds in tz_her2_binary humanness_therapeutic \
              ginkgo_acsins ginkgo_colloidal ginkgo_heparin \
              ginkgo_hic ginkgo_pr_cho ginkgo_titer; do
        submit_dataset "${ds}" 128G 8:00:00
    done
    echo ""
}

run_tcga() {
    echo "--- TCGA (2 × 6 = 12 jobs) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50; do
        submit_dataset "${ds}" 256G 12:00:00
    done
    echo ""
}

run_reaction() {
    echo "--- Reaction (2 × 6 = 12 jobs) ---"
    for ds in buchwald_hartwig bh_doyle_plates; do
        submit_dataset "${ds}" 128G 8:00:00
    done
    echo ""
}

case "$MODE" in
    molecular) run_molecular ;;
    antibody)  run_antibody ;;
    tcga)      run_tcga ;;
    reaction)  run_reaction ;;
    all)
        run_molecular
        run_antibody
        run_tcga
        run_reaction
        echo "============================================"
        echo "Total: ${COUNT} SLURM jobs (40 datasets × 6 conditions)"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash submit_precompute.sh [all|molecular|antibody|tcga|reaction]"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER | grep pc_"
echo "Cache:   ls -lt ~/.cache/kirby/hybrid_search/ | head"
