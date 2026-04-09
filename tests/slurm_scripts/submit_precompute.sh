#!/bin/bash
# Precompute hybrid searches — one SLURM job per (dataset × condition).
# 4 conditions per dataset = 4 parallel jobs per dataset.
#
# Conditions:
#   general_none       — general reps, no augmentation
#   general_uncapped   — general reps, uncapped augmentation
#   top_none           — top_tier reps, no augmentation
#   top_uncapped       — top_tier reps, uncapped augmentation
#
# Capped removed: iterative A/B testing makes it 10x slower. _shared.py
# already skips capped in precompute. Experiments needing capped call
# create_hybrid directly.
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute.sh              # all
#   bash tests/slurm_scripts/submit_precompute.sh molecular     # one domain
#   bash tests/slurm_scripts/submit_precompute.sh antibody

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
COUNT=0

CONDITIONS="general_none general_uncapped top_none top_uncapped"
# NOTE: capped conditions removed — _shared.py already skips capped in precompute.
# Experiments that need capped (C02/C05/C06/C07/C09) call create_hybrid directly.
# For on-demand capping of uncapped results, use cap_hybrid() (TODO).

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}
    local time_limit=${4:-6-23:59:00}

    J=$(sbatch \
        --job-name="pc_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time="${time_limit}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && sleep \$((RANDOM % 30)) && ${cmd}" \
        | awk '{print $4}')
    echo "  $J  pc_${name}"
    COUNT=$((COUNT + 1))
}

submit_dataset() {
    local ds=$1
    local mem=${2:-256G}
    local time_limit=${3:-6-23:59:00}

    for cond in ${CONDITIONS}; do
        submit "${ds}_${cond}" \
            "python tests/precompute_hybrids.py ${ds} --condition ${cond} --speed full" \
            "${mem}" "${time_limit}"
    done
}

echo "============================================"
echo "Hybrid Precompute (1 job per dataset×condition)"
echo "============================================"
echo ""

run_molecular() {
    echo "--- Molecular (7 × 4 = 28 jobs) ---"
    for ds in herg_ki caco2 logd qm9 adme_hlm adme_rlm adme_mdr1; do
        submit_dataset "${ds}" 64G 6-23:59:00
    done
    echo ""
}

run_antibody() {
    echo "--- Antibody Inter-Family (11 × 4 = 44 jobs) ---"
    for ds in flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt \
              flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain \
              abcov_kd flab_aggregation_jain flab_immunogenicity_marks \
              flab_polyreact_jain_bvp ddg_combined; do
        submit_dataset "${ds}" 64G 6-23:59:00
    done

    echo "--- Antibody DMS (5 × 4 = 20 jobs) ---"
    for ds in flab_binding_koenig flab_expression_koenig flab_binding_adams \
              flab_expression_adams flab_binding_phillips_h3; do
        submit_dataset "${ds}" 64G 6-23:59:00
    done

    echo "--- AbAgym (5 × 4 = 20 jobs) ---"
    for ds in abagym_Ang2 abagym_EGFR abagym_HER2 abagym_lysozyme abagym_VEGF; do
        submit_dataset "${ds}" 64G 6-23:59:00
    done

    echo "--- Antibody CLF (2 × 4 = 8 jobs) ---"
    for ds in tz_her2_binary humanness_therapeutic; do
        submit_dataset "${ds}" 64G 6-23:59:00
    done

    echo "--- Ginkgo (6 × 4 = 24 jobs) ---"
    for ds in ginkgo_acsins ginkgo_colloidal ginkgo_heparin \
              ginkgo_hic ginkgo_pr_cho ginkgo_titer; do
        submit_dataset "${ds}" 64G 6-23:59:00
    done
    echo ""
}

run_tcga() {
    echo "--- TCGA (2 × 4 = 8 jobs) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50; do
        submit_dataset "${ds}" 64G 6-23:59:00
    done
    echo ""
}

case "$MODE" in
    molecular) run_molecular ;;
    antibody)  run_antibody ;;
    tcga)      run_tcga ;;
    all)
        run_molecular
        run_antibody
        run_tcga
        echo "============================================"
        echo "Total: ${COUNT} SLURM jobs (38 datasets × 4 conditions)"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash submit_precompute.sh [all|molecular|antibody|tcga]"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER | grep pc_"
echo "Cache:   ls -lt ~/.cache/kirby/hybrid_search/ | head"
