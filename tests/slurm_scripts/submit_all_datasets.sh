#!/bin/bash
# Submit all dataset jobs for a given category to SLURM.
# Each dataset runs as an independent job (parallelised over datasets).
#
# Usage:
#   bash tests/slurm_scripts/submit_all_datasets.sh complexity
#   bash tests/slurm_scripts/submit_all_datasets.sh noise --dry-run
#   bash tests/slurm_scripts/submit_all_datasets.sh similarity --domain molecular
#   bash tests/slurm_scripts/submit_all_datasets.sh federated

CATEGORY=${1:?Usage: bash submit_all_datasets.sh CATEGORY [--dry-run] [--domain DOMAIN]}
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_dataset.sh"

# ── Dataset lists by domain (matches DATASET_REPS keys) ─────────────────
MOLECULAR="herg caco2 logd qm9 adme_hlm adme_rlm adme_mdr1"
ANTIBODY="flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain abcov_kd flab_aggregation_jain flab_immunogenicity_marks flab_polyreact_jain_bvp ddg_combined flab_binding_koenig flab_expression_koenig flab_binding_adams flab_expression_adams flab_binding_phillips_h3"
ABAGYM="abagym_Ang2 abagym_EGFR abagym_HER2 abagym_lysozyme abagym_VEGF"
AB_CLF="tz_her2_binary humanness_therapeutic"
GINKGO="ginkgo_acsins ginkgo_colloidal ginkgo_heparin ginkgo_hic ginkgo_pr_cho ginkgo_titer"
REACTION="buchwald_hartwig bh_doyle_plates"
TCGA="tcga_luad_vs_lusc tcga_brca_pam50"

# Federated-eligible datasets (grows as fed02-fed06 add support)
FEDERATED_DATASETS="herg caco2 logd"

# ── Job name prefix ────────────────────────────────────────────────────
case $CATEGORY in
    complexity) PREFIX="cx" ;;
    similarity) PREFIX="sim" ;;
    noise)      PREFIX="nz" ;;
    federated)  PREFIX="fed" ;;
    *)          PREFIX="exp" ;;
esac

# ── Parse args ──────────────────────────────────────────────────────────
DRY_RUN=false
DOMAIN="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --domain)  DOMAIN=$2; shift 2 ;;
        molecular|antibody|abagym|ab_clf|ginkgo|reaction|tcga) DOMAIN=$1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Build dataset list ──────────────────────────────────────────────────
if [ "$CATEGORY" = "federated" ]; then
    DATASETS="$FEDERATED_DATASETS"
else
    case $DOMAIN in
        molecular) DATASETS="$MOLECULAR" ;;
        antibody)  DATASETS="$ANTIBODY" ;;
        abagym)    DATASETS="$ABAGYM" ;;
        ab_clf)    DATASETS="$AB_CLF" ;;
        ginkgo)    DATASETS="$GINKGO" ;;
        reaction)  DATASETS="$REACTION" ;;
        tcga)      DATASETS="$TCGA" ;;
        all)       DATASETS="$MOLECULAR $ANTIBODY $ABAGYM $AB_CLF $GINKGO $REACTION $TCGA" ;;
        *)         echo "Unknown domain: $DOMAIN"; exit 1 ;;
    esac
fi

# ── Submit ──────────────────────────────────────────────────────────────
COUNT=0
for DS in $DATASETS; do
    CMD="sbatch --job-name=${PREFIX}_${DS} ${SUBMIT_SCRIPT} ${CATEGORY} ${DS}"
    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] $CMD"
    else
        echo "Submitting: ${CATEGORY}/${DS}"
        $CMD
    fi
    COUNT=$((COUNT + 1))
done

echo ""
echo "Total: ${COUNT} ${CATEGORY} jobs${DRY_RUN:+ (dry run)}"
