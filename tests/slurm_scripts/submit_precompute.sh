#!/bin/bash
# Precompute hybrid searches — ONE SLURM job per dataset (bundled).
#
# Each job runs every applicable condition in a single Python process so
# representations (incl. expensive ESM3/SaProt/IGT5/AntiBERTy stacks) are
# loaded ONCE per dataset. Previously each (dataset × condition) pair was
# its own job, paying the load cost 5-7 times per dataset.
#
# Standard conditions run per job (auto-selected by precompute_hybrids.py):
#   general_none, general_uncapped, top_none, top_uncapped, top_capped
#   (general/capped dropped 2026-05-22; top_capped uses default pruning here.)
#
# Distance conditions run per job (with --distance):
#   general_distance_none, general_distance_uncapped,
#   top_distance_none, top_distance_uncapped
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute.sh              # all standard
#   bash tests/slurm_scripts/submit_precompute.sh molecular     # one domain
#   bash tests/slurm_scripts/submit_precompute.sh distance      # all distance
#   bash tests/slurm_scripts/submit_precompute.sh distance molecular  # distance + domain
#   bash tests/slurm_scripts/submit_precompute.sh --datasets "qm9 caco2 logd"  # explicit list
#   bash tests/slurm_scripts/submit_precompute.sh distance --datasets "kiba"   # distance + list
#
# Flags (must come AFTER positional args):
#   --datasets "A B C"  Submit only these datasets (overrides domain). Each is
#                       matched against the known dataset names; aug/no-aug is
#                       auto-detected from membership in the *_NOAUG/_AUG lists.

set -euo pipefail

MODE="all"
DOMAIN="all"
EXPLICIT_DATASETS=""

# First positional: MODE (or domain shortcut if not 'distance'/'all')
if [[ $# -gt 0 && "$1" != --* ]]; then
    MODE="$1"; shift
fi
# Second positional: DOMAIN (only if MODE was 'distance' or similar)
if [[ $# -gt 0 && "$1" != --* ]]; then
    DOMAIN="$1"; shift
fi

# Then parse remaining flags.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) EXPLICIT_DATASETS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
COUNT=0

_time_for_partition() {
    # Use partition max limits (leave no headroom to waste)
    case "$1" in
        short)   echo "12:00:00" ;;
        medium)  echo "2-00:00:00" ;;
        *)       echo "7-00:00:00" ;;
    esac
}

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}
    local time_limit=${4:-}
    local partition=${5:-long}

    # Auto-resolve time limit from partition if not explicitly set
    if [[ -z "$time_limit" ]]; then
        time_limit=$(_time_for_partition "${partition}")
    fi

    J=$(sbatch \
        --job-name="pc_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time="${time_limit}" --partition="${partition}" \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && sleep \$((RANDOM % 30)) && ${cmd}" \
        | awk '{print $4}')
    echo "  $J  pc_${name}  (${partition}, ${time_limit})"
    COUNT=$((COUNT + 1))
}

# Submit ONE job that runs all standard conditions for one dataset
# (reps load once, conditions iterated inside the Python process).
# has_aug is kept in the signature for backwards-compat with callers; the
# Python script auto-selects which conditions to run based on dataset_reps
# (datasets without augmentation reps simply skip the aug conditions).
submit_dataset() {
    local ds=$1
    local has_aug=${2:-yes}   # unused — Python decides aug applicability
    local mem=${3:-256G}
    local partition=${4:-long}

    submit "${ds}_standard" \
        "python tests/precompute_hybrids.py ${ds} --speed full" \
        "${mem}" "" "${partition}"
}

# Submit ONE job that runs all distance conditions for one dataset.
submit_dataset_distance() {
    local ds=$1
    local has_aug=${2:-yes}   # unused — Python decides aug applicability
    local mem=${3:-256G}
    local partition=${4:-long}

    submit "${ds}_distance" \
        "python tests/precompute_hybrids.py ${ds} --speed full --distance" \
        "${mem}" "" "${partition}"
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
echo "Hybrid Precompute (filename-based cache)"
echo "  Cache: ~/.cache/kirby/searches/{dataset}__{tier}__{aug}__{speed}.pkl"
echo "  Nucleotides excluded (handled by survey scripts, not precompute)"
echo "============================================"
echo ""

run_standard() {
    local filter_domain=${1:-all}
    local P=${2:-long}

    if [[ "$filter_domain" == "all" || "$filter_domain" == "molecular" ]]; then
        echo "--- Molecular (7 datasets, all have aug) ---"
        for ds in ${MOLECULAR}; do submit_dataset "${ds}" yes 64G "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "antibody" ]]; then
        echo "--- Antibody IF with aug (7 datasets) ---"
        for ds in ${ANTIBODY_IF_AUG}; do submit_dataset "${ds}" yes 64G "${P}"; done
        echo "--- Antibody IF no aug (4 datasets) ---"
        for ds in ${ANTIBODY_IF_NOAUG}; do submit_dataset "${ds}" no 64G "${P}"; done

        echo "--- Antibody DMS (5 datasets, all have aug) ---"
        for ds in ${ANTIBODY_DMS}; do submit_dataset "${ds}" yes 64G "${P}"; done

        echo "--- AbAgym (5 datasets, no aug) ---"
        for ds in ${ABAGYM}; do submit_dataset "${ds}" no 64G "${P}"; done

        echo "--- Antibody CLF (2 datasets, no aug) ---"
        for ds in ${ANTIBODY_CLF}; do submit_dataset "${ds}" no 64G "${P}"; done

        echo "--- Ginkgo (6 datasets, all have aug) ---"
        for ds in ${GINKGO}; do submit_dataset "${ds}" yes 64G "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "tcga" ]]; then
        echo "--- TCGA (2 datasets, all have aug) ---"
        for ds in ${TCGA}; do submit_dataset "${ds}" yes 64G "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "reaction" ]]; then
        echo "--- Reaction ---"
        submit_dataset buchwald_hartwig no 64G "${P}"
        submit_dataset bh_doyle_plates yes 64G "${P}"
        echo ""
    fi

    # Nucleotides are handled by survey scripts (tests/survey_nucleotide_*.py)
    # and intentionally NOT run through precompute. Use --nucleotide to override.
    if [[ "$filter_domain" == "nucleotide" ]]; then
        echo "--- Nucleotide (5 datasets, all have aug, 128G) ---"
        echo "    WARNING: nucleotides usually skip precompute. Proceeding anyway."
        for ds in ${NUCLEOTIDE}; do submit_dataset "${ds}" yes 128G "${P}"; done
        echo ""
    fi
}

# Distance: general-tier only (no experiment uses top-tier distance search)
run_distance() {
    local filter_domain=${1:-all}
    local P=${2:-long}

    if [[ "$filter_domain" == "all" || "$filter_domain" == "molecular" ]]; then
        echo "--- Distance: Molecular (7 datasets) ---"
        for ds in ${MOLECULAR}; do submit_dataset_distance "${ds}" yes 64G "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "antibody" ]]; then
        echo "--- Distance: Antibody ---"
        for ds in ${ANTIBODY_IF_AUG}; do submit_dataset_distance "${ds}" yes 64G "${P}"; done
        for ds in ${ANTIBODY_IF_NOAUG}; do submit_dataset_distance "${ds}" no 64G "${P}"; done
        for ds in ${ANTIBODY_DMS}; do submit_dataset_distance "${ds}" yes 64G "${P}"; done
        for ds in ${ABAGYM}; do submit_dataset_distance "${ds}" no 64G "${P}"; done
        for ds in ${ANTIBODY_CLF}; do submit_dataset_distance "${ds}" no 64G "${P}"; done
        for ds in ${GINKGO}; do submit_dataset_distance "${ds}" yes 64G "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "tcga" ]]; then
        echo "--- Distance: TCGA ---"
        for ds in ${TCGA}; do submit_dataset_distance "${ds}" yes 64G "${P}"; done
        echo ""
    fi

    if [[ "$filter_domain" == "all" || "$filter_domain" == "reaction" ]]; then
        echo "--- Distance: Reaction ---"
        submit_dataset_distance buchwald_hartwig no 64G "${P}"
        submit_dataset_distance bh_doyle_plates yes 64G "${P}"
        echo ""
    fi

    # Nucleotides skip precompute by default (survey scripts own those).
    if [[ "$filter_domain" == "nucleotide" ]]; then
        echo "--- Distance: Nucleotide (128G) ---"
        echo "    WARNING: nucleotides usually skip precompute. Proceeding anyway."
        for ds in ${NUCLEOTIDE}; do submit_dataset_distance "${ds}" yes 128G "${P}"; done
        echo ""
    fi
}

PARTITION=${3:-long}

# Auto-detect (has_aug, mem) for one dataset name by checking the known lists.
# Returns "<has_aug> <mem>" — e.g. "yes 64G", "no 128G".
# Falls back to "yes 64G" with a warning for unknown datasets.
_lookup_dataset_meta() {
    local ds=$1
    local has_aug="yes"
    local mem="64G"
    # Has-aug membership
    local NOAUG_LIST="${ANTIBODY_IF_NOAUG} ${ABAGYM} ${ANTIBODY_CLF} ${REACTION_NOAUG}"
    for n in ${NOAUG_LIST}; do
        if [[ "$ds" == "$n" ]]; then has_aug="no"; break; fi
    done
    # Memory tier (matches the 128G overrides in submit_all_datasets.sh)
    case "$ds" in
        qm9|flab_binding_adams|flab_expression_adams|flab_binding_phillips_h3|\
        tz_her2_binary|tz_her2_multiclass|bh_doyle_plates|flab_binding_koenig|\
        flab_expression_koenig|flab_immunogenicity_marks|tcga_luad_vs_lusc|\
        tcga_brca_pam50|kiba|bindingdb_kd) mem="128G" ;;
        nt_*|gb_*) mem="128G" ;;
    esac
    echo "${has_aug} ${mem}"
}

# Dispatch an explicit dataset list (overrides MODE-driven enumeration).
# Args: $1 = "standard" or "distance"
run_explicit_datasets() {
    local kind=$1
    if [[ -z "${EXPLICIT_DATASETS}" ]]; then return; fi
    echo "--- Explicit datasets (${kind}): ${EXPLICIT_DATASETS} ---"
    for ds in ${EXPLICIT_DATASETS}; do
        read has_aug mem <<< "$(_lookup_dataset_meta "$ds")"
        if [[ "$kind" == "distance" ]]; then
            submit_dataset_distance "$ds" "$has_aug" "$mem" "$PARTITION"
        else
            submit_dataset "$ds" "$has_aug" "$mem" "$PARTITION"
        fi
    done
}

# When --datasets is provided, it short-circuits the MODE-based enumeration.
if [[ -n "${EXPLICIT_DATASETS}" ]]; then
    case "$MODE" in
        distance)       run_explicit_datasets distance ;;
        all+distance)   run_explicit_datasets standard
                        run_explicit_datasets distance ;;
        *)              run_explicit_datasets standard ;;
    esac
    echo ""
    echo "Total: ${COUNT} SLURM jobs submitted (explicit dataset list)"
    echo "Monitor: squeue -u \$USER | grep pc_"
    echo "Status:  python tests/check_precompute_status.py"
    exit 0
fi

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
        echo "Standard: ${COUNT} SLURM jobs submitted"
        echo "Monitor: squeue -u \$USER | grep pc_"
        echo "Status:  python tests/check_precompute_status.py"
        echo "============================================"
        ;;
    all+distance)
        run_standard all "$PARTITION"
        STD_COUNT=$COUNT
        run_distance all "$PARTITION"
        DIST_COUNT=$((COUNT - STD_COUNT))
        echo "============================================"
        echo "Standard: ${STD_COUNT} jobs, Distance: ${DIST_COUNT} jobs"
        echo "Total: ${COUNT} SLURM jobs submitted"
        echo "Monitor: squeue -u \$USER | grep pc_"
        echo "Status:  python tests/check_precompute_status.py --distance"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash submit_precompute.sh <mode> [domain] [partition] [--datasets \"A B C\"]"
        echo ""
        echo "Modes:"
        echo "  all            Standard conditions, all domains except nucleotide"
        echo "  all+distance   Standard + distance conditions"
        echo "  distance       Distance conditions only"
        echo "  molecular|antibody|tcga|reaction  One domain (nucleotide skipped)"
        echo "  nucleotide     Override — normally handled by survey scripts"
        echo ""
        echo "Partition (default: long):"
        echo "  bash submit_precompute.sh all all short"
        echo "  bash submit_precompute.sh molecular all medium"
        echo ""
        echo "After submission, check readiness with:"
        echo "  python tests/check_precompute_status.py --verbose"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER | grep pc_"
echo "Cache:   ls -lt ~/.cache/kirby/searches/ | head"
echo "Status:  python tests/check_precompute_status.py --verbose"
