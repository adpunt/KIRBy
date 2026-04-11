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

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}
    local time_limit=${4:-6-23:59:00}
    local partition=${5:-long}

    J=$(sbatch \
        --job-name="pc_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time="${time_limit}" --partition="${partition}" \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && sleep \$((RANDOM % 30)) && ${cmd}" \
        | awk '{print $4}')
    echo "  $J  pc_${name}  (${partition})"
    COUNT=$((COUNT + 1))
}

# Submit standard + capped conditions for one dataset.
# has_aug: pass "yes" if dataset has augmentation reps, "no" to skip aug conditions.
submit_dataset() {
    local ds=$1
    local has_aug=${2:-yes}
    local mem=${3:-256G}
    local time_limit=${4:-6-23:59:00}
    local partition=${5:-long}

    # Always: general_none + top_none
    submit "${ds}_general_none" \
        "python tests/precompute_hybrids.py ${ds} --condition general_none --speed full" \
        "${mem}" "${time_limit}" "${partition}"
    submit "${ds}_top_none" \
        "python tests/precompute_hybrids.py ${ds} --condition top_none --speed full" \
        "${mem}" "${time_limit}" "${partition}"

    # Only if dataset has augmentation reps
    if [[ "$has_aug" == "yes" ]]; then
        submit "${ds}_general_uncapped" \
            "python tests/precompute_hybrids.py ${ds} --condition general_uncapped --speed full" \
            "${mem}" "${time_limit}" "${partition}"
        submit "${ds}_top_uncapped" \
            "python tests/precompute_hybrids.py ${ds} --condition top_uncapped --speed full" \
            "${mem}" "${time_limit}" "${partition}"
        # Capped — more aggressive pruning (0.25 threshold)
        submit "${ds}_general_capped" \
            "python tests/precompute_hybrids.py ${ds} --condition general_capped --speed full --pruning 0.25" \
            "${mem}" "${time_limit}" "${partition}"
        submit "${ds}_top_capped" \
            "python tests/precompute_hybrids.py ${ds} --condition top_capped --speed full --pruning 0.25" \
            "${mem}" "${time_limit}" "${partition}"
    fi
}

# Distance conditions — general only (no experiment uses top-tier distance search).
submit_dataset_distance() {
    local ds=$1
    local has_aug=${2:-yes}
    local mem=${3:-256G}
    local time_limit=${4:-6-23:59:00}
    local partition=${5:-long}

    submit "${ds}_general_distance_none" \
        "python tests/precompute_hybrids.py ${ds} --condition general_distance_none --speed full --distance" \
        "${mem}" "${time_limit}" "${partition}"

    if [[ "$has_aug" == "yes" ]]; then
        submit "${ds}_general_distance_uncapped" \
            "python tests/precompute_hybrids.py ${ds} --condition general_distance_uncapped --speed full --distance" \
            "${mem}" "${time_limit}" "${partition}"
    fi
}

# --- Dataset lists (has_aug flag: yes/no based on DATASET_REPS) ---
# Datasets with NO augmentation reps get 2 jobs instead of 6 (standard)
# and 1 job instead of 2 (distance). Saves ~70 no-op SLURM submissions.

# Molecular: all have augmentation
MOLECULAR="herg_ki caco2 logd qm9 adme_hlm adme_rlm adme_mdr1"

# Antibody inter-family: mixed
ANTIBODY_IF_AUG="flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt flab_aggregation_jain_hicrt flab_thermo_jain flab_aggregation_jain ddg_combined"
ANTIBODY_IF_NOAUG="flab_polyreact_jain_cic abcov_kd flab_immunogenicity_marks flab_polyreact_jain_bvp"

# DMS: all have augmentation
ANTIBODY_DMS="flab_binding_koenig flab_expression_koenig flab_binding_adams flab_expression_adams flab_binding_phillips_h3"

# AbAgym: none have augmentation
ABAGYM="abagym_Ang2 abagym_EGFR abagym_HER2 abagym_lysozyme abagym_VEGF"

# CLF: none have augmentation
ANTIBODY_CLF="tz_her2_binary humanness_therapeutic"

# Ginkgo: all have augmentation
GINKGO="ginkgo_acsins ginkgo_colloidal ginkgo_heparin ginkgo_hic ginkgo_pr_cho ginkgo_titer"

TCGA="tcga_luad_vs_lusc tcga_brca_pam50"

# Reaction: buchwald has no aug, bh_doyle has aug
REACTION_AUG="bh_doyle_plates"
REACTION_NOAUG="buchwald_hartwig"

# Nucleotide: all have augmentation
NUCLEOTIDE="nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 nt_H3K4me1"

echo "============================================"
echo "Hybrid Precompute (v14 — normalized cache keys)"
echo "  Standard: ~224 jobs (6 per aug dataset, 2 per no-aug)"
echo "  Distance: ~79 jobs (general only, no top_distance)"
echo "  Total: ~303 jobs"
echo "============================================"
echo ""

run_standard() {
    local filter_domain=${1:-all}
    local P=${2:-long}

    if [[ "$filter_domain" == "all" || "$filter_domain" == "molecular" ]]; then
        echo "--- Molecular (7 datasets, all have aug) ---"
        for ds in ${MOLECULAR}; do submit_dataset "${ds}" yes 64G 6-23:59:00 "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "antibody" ]]; then
        echo "--- Antibody IF with aug (7 datasets) ---"
        for ds in ${ANTIBODY_IF_AUG}; do submit_dataset "${ds}" yes 64G 6-23:59:00 "${P}"; done
        echo "--- Antibody IF no aug (4 datasets) ---"
        for ds in ${ANTIBODY_IF_NOAUG}; do submit_dataset "${ds}" no 64G 6-23:59:00 "${P}"; done

        echo "--- Antibody DMS (5 datasets, all have aug) ---"
        for ds in ${ANTIBODY_DMS}; do submit_dataset "${ds}" yes 64G 6-23:59:00 "${P}"; done

        echo "--- AbAgym (5 datasets, no aug) ---"
        for ds in ${ABAGYM}; do submit_dataset "${ds}" no 64G 6-23:59:00 "${P}"; done

        echo "--- Antibody CLF (2 datasets, no aug) ---"
        for ds in ${ANTIBODY_CLF}; do submit_dataset "${ds}" no 64G 6-23:59:00 "${P}"; done

        echo "--- Ginkgo (6 datasets, all have aug) ---"
        for ds in ${GINKGO}; do submit_dataset "${ds}" yes 64G 6-23:59:00 "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "tcga" ]]; then
        echo "--- TCGA (2 datasets, all have aug) ---"
        for ds in ${TCGA}; do submit_dataset "${ds}" yes 64G 6-23:59:00 "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "reaction" ]]; then
        echo "--- Reaction ---"
        submit_dataset buchwald_hartwig no 64G 6-23:59:00 "${P}"
        submit_dataset bh_doyle_plates yes 64G 6-23:59:00 "${P}"
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "nucleotide" ]]; then
        echo "--- Nucleotide (5 datasets, all have aug, 128G) ---"
        for ds in ${NUCLEOTIDE}; do submit_dataset "${ds}" yes 128G 6-23:59:00 "${P}"; done
        echo ""
    fi
}

# Distance: general-tier only (no experiment uses top-tier distance search)
run_distance() {
    local filter_domain=${1:-all}
    local P=${2:-long}

    if [[ "$filter_domain" == "all" || "$filter_domain" == "molecular" ]]; then
        echo "--- Distance: Molecular (7 datasets) ---"
        for ds in ${MOLECULAR}; do submit_dataset_distance "${ds}" yes 64G 6-23:59:00 "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "antibody" ]]; then
        echo "--- Distance: Antibody ---"
        for ds in ${ANTIBODY_IF_AUG}; do submit_dataset_distance "${ds}" yes 64G 6-23:59:00 "${P}"; done
        for ds in ${ANTIBODY_IF_NOAUG}; do submit_dataset_distance "${ds}" no 64G 6-23:59:00 "${P}"; done
        for ds in ${ANTIBODY_DMS}; do submit_dataset_distance "${ds}" yes 64G 6-23:59:00 "${P}"; done
        for ds in ${ABAGYM}; do submit_dataset_distance "${ds}" no 64G 6-23:59:00 "${P}"; done
        for ds in ${ANTIBODY_CLF}; do submit_dataset_distance "${ds}" no 64G 6-23:59:00 "${P}"; done
        for ds in ${GINKGO}; do submit_dataset_distance "${ds}" yes 64G 6-23:59:00 "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "tcga" ]]; then
        echo "--- Distance: TCGA ---"
        for ds in ${TCGA}; do submit_dataset_distance "${ds}" yes 64G 6-23:59:00 "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "reaction" ]]; then
        echo "--- Distance: Reaction ---"
        submit_dataset_distance buchwald_hartwig no 64G 6-23:59:00 "${P}"
        submit_dataset_distance bh_doyle_plates yes 64G 6-23:59:00 "${P}"
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "nucleotide" ]]; then
        echo "--- Distance: Nucleotide (128G) ---"
        for ds in ${NUCLEOTIDE}; do submit_dataset_distance "${ds}" yes 128G 6-23:59:00 "${P}"; done
        echo ""
    fi
}

PARTITION=${3:-long}

case "$MODE" in
    molecular|antibody|tcga|reaction|nucleotide)
        run_standard "$MODE" "$PARTITION"
        ;;
    distance)
        run_distance "$DOMAIN" "$PARTITION"
        ;;
    all)
        run_standard all "$PARTITION"
        echo "============================================"
        echo "Standard: ${COUNT} SLURM jobs (~224 needed)"
        echo "============================================"
        ;;
    all+distance)
        run_standard all "$PARTITION"
        STD_COUNT=$COUNT
        run_distance all "$PARTITION"
        DIST_COUNT=$((COUNT - STD_COUNT))
        echo "============================================"
        echo "Standard: ${STD_COUNT} jobs, Distance: ${DIST_COUNT} jobs"
        echo "Total: ${COUNT} SLURM jobs (~303 needed)"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash submit_precompute.sh <mode> [domain] [partition]"
        echo ""
        echo "Modes:"
        echo "  all            Standard conditions only (~224 jobs)"
        echo "  all+distance   Standard + distance (~303 jobs)"
        echo "  distance       Distance conditions only (~79 jobs)"
        echo "  molecular|antibody|tcga|reaction|nucleotide  One domain"
        echo ""
        echo "Partition (default: long):"
        echo "  bash submit_precompute.sh all+distance all short"
        echo "  bash submit_precompute.sh molecular all medium"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER | grep pc_"
echo "Cache:   ls -lt ~/.cache/kirby/hybrid_search/ | head"
