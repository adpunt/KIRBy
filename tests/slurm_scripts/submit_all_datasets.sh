#!/bin/bash
# Submit all dataset jobs for a given category to SLURM.
# Each dataset runs as an independent job (parallelised over datasets).
#
# Usage:
#   bash tests/slurm_scripts/submit_all_datasets.sh complexity
#   bash tests/slurm_scripts/submit_all_datasets.sh noise --dry-run
#   bash tests/slurm_scripts/submit_all_datasets.sh similarity --domain molecular
#   bash tests/slurm_scripts/submit_all_datasets.sh federated
#
# Domains: molecular, molecular_clf, dta, antibody, abagym, ab_clf,
#          ginkgo, reaction, tcga, nucleotide, all (default)

CATEGORY=${1:?Usage: bash submit_all_datasets.sh CATEGORY [--dry-run] [--domain DOMAIN]}
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_dataset.sh"

# ── Dataset lists by domain (matches DATASET_REPS keys) ─────────────────
MOLECULAR="herg_ki caco2 logd qm9 adme_hlm adme_rlm adme_mdr1"
MOLECULAR_CLF="herg"
DTA="kiba bindingdb_kd"
ANTIBODY="flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain abcov_kd flab_aggregation_jain flab_immunogenicity_marks flab_polyreact_jain_bvp ddg_combined flab_binding_koenig flab_expression_koenig flab_binding_adams flab_expression_adams flab_binding_phillips_h3"
ABAGYM="abagym_Ang2 abagym_EGFR abagym_HER2 abagym_lysozyme abagym_VEGF"
AB_CLF="tz_her2_binary tz_her2_multiclass humanness_therapeutic"
GINKGO="ginkgo_acsins ginkgo_colloidal ginkgo_heparin ginkgo_hic ginkgo_pr_cho ginkgo_titer"
REACTION="buchwald_hartwig bh_doyle_plates"
TCGA="tcga_luad_vs_lusc tcga_brca_pam50"
NUCLEOTIDE="nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 nt_H3K4me1 nt_H3K4me2 nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_H4K20me1 nt_enhancers nt_enhancers_types nt_promoter_all nt_promoter_no_tata nt_promoter_tata nt_splice_sites_acceptors nt_splice_sites_donors gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm gb_human_enhancers_cohn gb_human_nontata_promoters"

# Federated-eligible datasets (grows as fed02-fed05 add support)
# lhasa_herg uses Lhasa's curated hERG data (binary classification).
FEDERATED_DATASETS="lhasa_herg caco2 logd"

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
        molecular|molecular_clf|dta|antibody|abagym|ab_clf|ginkgo|reaction|tcga|nucleotide) DOMAIN=$1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Build dataset list ──────────────────────────────────────────────────
if [ "$CATEGORY" = "federated" ]; then
    DATASETS="$FEDERATED_DATASETS"
else
    case $DOMAIN in
        molecular)     DATASETS="$MOLECULAR" ;;
        molecular_clf) DATASETS="$MOLECULAR_CLF" ;;
        dta)           DATASETS="$DTA" ;;
        antibody)      DATASETS="$ANTIBODY" ;;
        abagym)        DATASETS="$ABAGYM" ;;
        ab_clf)        DATASETS="$AB_CLF" ;;
        ginkgo)        DATASETS="$GINKGO" ;;
        reaction)      DATASETS="$REACTION" ;;
        tcga)          DATASETS="$TCGA" ;;
        nucleotide)    DATASETS="$NUCLEOTIDE" ;;
        all)           DATASETS="$MOLECULAR $MOLECULAR_CLF $DTA $ANTIBODY $ABAGYM $AB_CLF $GINKGO $REACTION $TCGA $NUCLEOTIDE" ;;
        *)             echo "Unknown domain: $DOMAIN"; exit 1 ;;
    esac
fi

# ── Per-dataset resource overrides ─────────────────────────────────────
# Large datasets (>4K samples or many high-dim reps) need more memory.
# TCGA datasets with methyl_hvg (5000D) need more time for importance.
get_mem_override() {
    case $1 in
        qm9|flab_binding_adams|flab_expression_adams|flab_binding_phillips_h3)
            echo "128G" ;;
        tz_her2_binary|tz_her2_multiclass|bh_doyle_plates|flab_binding_koenig|flab_expression_koenig)
            echo "128G" ;;
        flab_immunogenicity_marks)
            echo "128G" ;;
        tcga_luad_vs_lusc|tcga_brca_pam50)
            echo "128G" ;;
        # DTA: kiba (n=30K, 229 protein groups) and bindingdb_kd (n=30K, 1219 groups).
        kiba|bindingdb_kd)
            echo "128G" ;;
        # Nucleotide: n=30K (nt_*) to 100K (gb_demo_*), high-dim reps need RAM
        nt_*|gb_*)
            echo "128G" ;;
        *) echo "" ;;
    esac
}

# Default pruning mode for the bulk submission. Override at the env-var
# level: PRUNING_MODE=default_pruning bash submit_all_datasets.sh complexity
PRUNING_MODE="${PRUNING_MODE:-aggressive_pruning}"
EXPORT_FLAG="--export=ALL,PRUNING_MODE=${PRUNING_MODE}"

# ── Submit ──────────────────────────────────────────────────────────────
COUNT=0
for DS in $DATASETS; do
    MEM_OVERRIDE=$(get_mem_override $DS)
    if [ -n "$MEM_OVERRIDE" ]; then
        CMD="sbatch ${EXPORT_FLAG} --job-name=${PREFIX}_${DS} --mem=${MEM_OVERRIDE} ${SUBMIT_SCRIPT} ${CATEGORY} ${DS}"
    else
        CMD="sbatch ${EXPORT_FLAG} --job-name=${PREFIX}_${DS} ${SUBMIT_SCRIPT} ${CATEGORY} ${DS}"
    fi
    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] $CMD"
    else
        echo "Submitting: ${CATEGORY}/${DS}${MEM_OVERRIDE:+ (mem=${MEM_OVERRIDE})}  pruning=${PRUNING_MODE}"
        $CMD
    fi
    COUNT=$((COUNT + 1))
done

echo ""
echo "Total: ${COUNT} ${CATEGORY} jobs${DRY_RUN:+ (dry run)}  pruning_mode=${PRUNING_MODE}"
