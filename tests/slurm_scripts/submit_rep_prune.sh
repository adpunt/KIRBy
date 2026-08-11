#!/bin/bash
# Submit rep-pool cleanup as ONE job PER DOMAIN — each job runs that domain's
# datasets successively (rep_prune_evidence.py loops over them). Keeps the queue
# to ~10 jobs instead of 40+ per-dataset ones. The reps are precomputed, so each
# dataset is a quick lightgbm pass (~5-25 min); a domain runs in ~1-3h.
#
# --account and --partition are LIVE STATE — pass them at submit time (get them
# from where_to_submit.sh). Usage:
#   bash tests/slurm_scripts/submit_rep_prune.sh --account stat-cadd --partition medium
#   bash tests/slurm_scripts/submit_rep_prune.sh --account X --partition Y --domain antibody
#   bash tests/slurm_scripts/submit_rep_prune.sh --account X --partition Y --dry-run
#
# Domain lists MIRROR submit_all_datasets.sh (the roster source of truth) as of
# 2026-08-11 — keep in sync if the roster changes. herg (molecular_clf) and
# cdr_flexibility (ab_clf) are the two that never got a per-rep score; both are
# included here so this run covers them too.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/run_rep_prune.sh"

# ── Dataset lists by domain (mirrors submit_all_datasets.sh) ────────────
MOLECULAR="herg_ki caco2 logd qm9 adme_hlm adme_rlm adme_mdr1"
MOLECULAR_CLF="herg"
DTA="kiba bindingdb_kd"
ANTIBODY="flab_thermo_tresanco abcov_kd ddg_combined flab_binding_koenig flab_expression_koenig flab_binding_adams flab_expression_adams"
ABAGYM="abagym_Ang2 abagym_EGFR abagym_VEGF"
AB_CLF="tz_her2_binary tz_her2_multiclass humanness_therapeutic cdr_flexibility"
GINKGO="ginkgo_acsins ginkgo_colloidal ginkgo_hic ginkgo_pr_cho ginkgo_titer ginkgo_tm2 ginkgo_aggregation ginkgo_purity"
REACTION="buchwald_hartwig bh_doyle_plates"
TCGA="tcga_luad_vs_lusc tcga_brca_pam50"
NUCLEOTIDE="nt_H3K27me3 nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_promoter_tata gb_human_enhancers_cohn"

ALL_DOMAINS="molecular molecular_clf dta antibody abagym ab_clf ginkgo reaction tcga nucleotide"

# Per-domain memory — a domain job must fit its biggest dataset (mirrors the
# get_mem_override sizing in submit_all_datasets.sh).
domain_mem() {
    case $1 in
        nucleotide)                 echo "256G" ;;
        dta|tcga|antibody|ab_clf)   echo "128G" ;;
        *)                          echo "64G"  ;;
    esac
}
domain_datasets() {
    case $1 in
        molecular) echo "$MOLECULAR" ;; molecular_clf) echo "$MOLECULAR_CLF" ;;
        dta) echo "$DTA" ;; antibody) echo "$ANTIBODY" ;; abagym) echo "$ABAGYM" ;;
        ab_clf) echo "$AB_CLF" ;; ginkgo) echo "$GINKGO" ;; reaction) echo "$REACTION" ;;
        tcga) echo "$TCGA" ;; nucleotide) echo "$NUCLEOTIDE" ;;
        *) echo "" ;;
    esac
}

# ── Parse args ──────────────────────────────────────────────────────────
ACCOUNT=""; PARTITION=""; DOMAIN="all"; DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --account)   ACCOUNT=$2; shift 2 ;;
        --partition) PARTITION=$2; shift 2 ;;
        --domain)    DOMAIN=$2; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
if [ -z "$ACCOUNT" ] || [ -z "$PARTITION" ]; then
    echo "ERROR: --account and --partition are required (live state; see where_to_submit.sh)." >&2
    exit 1
fi

if [ "$DOMAIN" = "all" ]; then DOMAINS="$ALL_DOMAINS"; else DOMAINS="$DOMAIN"; fi

# ── Submit one job per domain ───────────────────────────────────────────
EXPORT_FLAG="--export=ALL,KIRBY_PRUNE_FOLDS=3,KIRBY_PRUNE_CAP=0"
COUNT=0
for D in $DOMAINS; do
    DS=$(domain_datasets "$D")
    if [ -z "$DS" ]; then echo "Unknown domain: $D" >&2; exit 1; fi
    MEM=$(domain_mem "$D")
    CMD="sbatch --account=${ACCOUNT} --partition=${PARTITION} ${EXPORT_FLAG} \
--job-name=repprune_${D} --mem=${MEM} ${JOB_SCRIPT} ${DS}"
    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] ${D} (mem=${MEM}): $CMD"
    else
        echo "Submitting rep_prune/${D} (mem=${MEM}): ${DS}"
        $CMD
    fi
    COUNT=$((COUNT + 1))
done
echo ""
echo "Total: ${COUNT} domain job(s)${DRY_RUN:+ (dry run)}"
