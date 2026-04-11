#!/bin/bash
# Precompute hybrid searches — one SLURM job per (dataset × condition).
#
# Standard conditions (6 per dataset):
#   general_none       — general reps, no augmentation
#   general_uncapped   — general reps, uncapped augmentation
#   general_capped     — general reps, capped augmentation
#   top_none           — top_tier reps, no augmentation
#   top_uncapped       — top_tier reps, uncapped augmentation
#   top_capped         — top_tier reps, capped augmentation
#
# Distance conditions (4 per dataset, for similarity/noise experiments):
#   general_distance_none     — DISTANCE_EVALUATORS + DISTANCE_BUDGETS
#   general_distance_uncapped
#   top_distance_none
#   top_distance_uncapped
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute.sh              # all standard
#   bash tests/slurm_scripts/submit_precompute.sh molecular     # one domain
#   bash tests/slurm_scripts/submit_precompute.sh distance      # all distance
#   bash tests/slurm_scripts/submit_precompute.sh distance molecular  # distance + domain

set -euo pipefail

MODE=${1:-all}
DOMAIN=${2:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
COUNT=0

STANDARD_CONDITIONS="general_none general_uncapped top_none top_uncapped"
CAPPED_CONDITIONS="general_capped top_capped"
DISTANCE_CONDITIONS="general_distance_none general_distance_uncapped top_distance_none top_distance_uncapped"

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

    # Standard conditions (none + uncapped)
    for cond in ${STANDARD_CONDITIONS}; do
        submit "${ds}_${cond}" \
            "python tests/precompute_hybrids.py ${ds} --condition ${cond} --speed full" \
            "${mem}" "${time_limit}"
    done
    # Capped conditions — more aggressive pruning (0.25 threshold)
    for cond in ${CAPPED_CONDITIONS}; do
        submit "${ds}_${cond}" \
            "python tests/precompute_hybrids.py ${ds} --condition ${cond} --speed full --pruning 0.25" \
            "${mem}" "${time_limit}"
    done
}

submit_dataset_distance() {
    local ds=$1
    local mem=${2:-256G}
    local time_limit=${3:-6-23:59:00}

    for cond in ${DISTANCE_CONDITIONS}; do
        submit "${ds}_${cond}" \
            "python tests/precompute_hybrids.py ${ds} --condition ${cond} --speed full --distance" \
            "${mem}" "${time_limit}"
    done
}

# --- Dataset lists ---
MOLECULAR="herg_ki caco2 logd qm9 adme_hlm adme_rlm adme_mdr1"
ANTIBODY_IF="flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain abcov_kd flab_aggregation_jain flab_immunogenicity_marks flab_polyreact_jain_bvp ddg_combined"
ANTIBODY_DMS="flab_binding_koenig flab_expression_koenig flab_binding_adams flab_expression_adams flab_binding_phillips_h3"
ABAGYM="abagym_Ang2 abagym_EGFR abagym_HER2 abagym_lysozyme abagym_VEGF"
ANTIBODY_CLF="tz_her2_binary humanness_therapeutic"
GINKGO="ginkgo_acsins ginkgo_colloidal ginkgo_heparin ginkgo_hic ginkgo_pr_cho ginkgo_titer"
TCGA="tcga_luad_vs_lusc tcga_brca_pam50"
REACTION="buchwald_hartwig bh_doyle_plates"

echo "============================================"
echo "Hybrid Precompute (v14 — normalized cache keys)"
echo "============================================"
echo ""

run_standard() {
    local filter_domain=${1:-all}

    if [[ "$filter_domain" == "all" || "$filter_domain" == "molecular" ]]; then
        echo "--- Molecular (7 × 6 = 42 jobs) ---"
        for ds in ${MOLECULAR}; do submit_dataset "${ds}" 64G; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "antibody" ]]; then
        echo "--- Antibody Inter-Family (11 × 6 = 66 jobs) ---"
        for ds in ${ANTIBODY_IF}; do submit_dataset "${ds}" 64G; done

        echo "--- Antibody DMS (5 × 6 = 30 jobs) ---"
        for ds in ${ANTIBODY_DMS}; do submit_dataset "${ds}" 64G; done

        echo "--- AbAgym (5 × 6 = 30 jobs) ---"
        for ds in ${ABAGYM}; do submit_dataset "${ds}" 64G; done

        echo "--- Antibody CLF (2 × 6 = 12 jobs) ---"
        for ds in ${ANTIBODY_CLF}; do submit_dataset "${ds}" 64G; done

        echo "--- Ginkgo (6 × 6 = 36 jobs) ---"
        for ds in ${GINKGO}; do submit_dataset "${ds}" 64G; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "tcga" ]]; then
        echo "--- TCGA (2 × 6 = 12 jobs) ---"
        for ds in ${TCGA}; do submit_dataset "${ds}" 64G; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "reaction" ]]; then
        echo "--- Reaction (2 × 6 = 12 jobs) ---"
        for ds in ${REACTION}; do submit_dataset "${ds}" 64G; done
        echo ""
    fi
}

run_distance() {
    local filter_domain=${1:-all}

    if [[ "$filter_domain" == "all" || "$filter_domain" == "molecular" ]]; then
        echo "--- Distance: Molecular (7 × 4 = 28 jobs) ---"
        for ds in ${MOLECULAR}; do submit_dataset_distance "${ds}" 64G; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "antibody" ]]; then
        echo "--- Distance: Antibody (29 × 4 = 116 jobs) ---"
        for ds in ${ANTIBODY_IF} ${ANTIBODY_DMS} ${ABAGYM} ${ANTIBODY_CLF} ${GINKGO}; do
            submit_dataset_distance "${ds}" 64G
        done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "tcga" ]]; then
        echo "--- Distance: TCGA (2 × 4 = 8 jobs) ---"
        for ds in ${TCGA}; do submit_dataset_distance "${ds}" 64G; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "reaction" ]]; then
        echo "--- Distance: Reaction (2 × 4 = 8 jobs) ---"
        for ds in ${REACTION}; do submit_dataset_distance "${ds}" 64G; done
        echo ""
    fi
}

case "$MODE" in
    molecular|antibody|tcga|reaction)
        run_standard "$MODE"
        ;;
    distance)
        run_distance "$DOMAIN"
        ;;
    all)
        run_standard all
        echo "============================================"
        echo "Standard: ${COUNT} SLURM jobs (40 datasets × 6 conditions)"
        echo "============================================"
        ;;
    all+distance)
        run_standard all
        STD_COUNT=$COUNT
        run_distance all
        DIST_COUNT=$((COUNT - STD_COUNT))
        echo "============================================"
        echo "Standard: ${STD_COUNT} jobs, Distance: ${DIST_COUNT} jobs"
        echo "Total: ${COUNT} SLURM jobs"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash submit_precompute.sh [all|all+distance|distance|molecular|antibody|tcga|reaction] [domain]"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER | grep pc_"
echo "Cache:   ls -lt ~/.cache/kirby/hybrid_search/ | head"
