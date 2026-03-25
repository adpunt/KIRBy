#!/bin/bash
# Submit all 33 complexity dataset jobs to SLURM.
# Each dataset runs as an independent job (parallelised over datasets).
#
# Usage:
#   bash tests/slurm_scripts/submit_all_complexity.sh
#
#   # Dry run (print commands without submitting):
#   bash tests/slurm_scripts/submit_all_complexity.sh --dry-run
#
#   # Submit only molecular datasets:
#   bash tests/slurm_scripts/submit_all_complexity.sh --domain molecular

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_complexity_dataset.sh"

# All 33 DONE datasets grouped by domain
MOLECULAR="herg caco2 logd qm9 adme_hlm adme_rlm adme_mdr1"
ANTIBODY="flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain abcov_kd flab_aggregation_jain flab_immunogenicity_marks flab_polyreact_jain_bvp ddg_combined"
ABAGYM="abagym_Ang2 abagym_EGFR abagym_HER2 abagym_lysozyme abagym_VEGF"
AB_CLF="tz_her2_binary humanness_therapeutic"
GINKGO="ginkgo_acsins ginkgo_colloidal ginkgo_heparin ginkgo_hic ginkgo_pr_cho ginkgo_titer"
TCGA="tcga_luad_vs_lusc tcga_brca_pam50"

DRY_RUN=false
DOMAIN="all"

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --domain)  shift; DOMAIN=$1 ;;
        molecular|antibody|abagym|ab_clf|ginkgo|tcga) DOMAIN=$arg ;;
    esac
done

case $DOMAIN in
    molecular) DATASETS="$MOLECULAR" ;;
    antibody)  DATASETS="$ANTIBODY" ;;
    abagym)    DATASETS="$ABAGYM" ;;
    ab_clf)    DATASETS="$AB_CLF" ;;
    ginkgo)    DATASETS="$GINKGO" ;;
    tcga)      DATASETS="$TCGA" ;;
    all)       DATASETS="$MOLECULAR $ANTIBODY $ABAGYM $AB_CLF $GINKGO $TCGA" ;;
    *)         echo "Unknown domain: $DOMAIN"; exit 1 ;;
esac

COUNT=0
for DS in $DATASETS; do
    CMD="sbatch --job-name=cx_${DS} ${SUBMIT_SCRIPT} ${DS}"
    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] $CMD"
    else
        echo "Submitting: $DS"
        $CMD
    fi
    COUNT=$((COUNT + 1))
done

echo ""
echo "Total: ${COUNT} jobs ${DRY_RUN:+(dry run)}"
