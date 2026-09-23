#!/bin/bash
# Submit all dataset jobs for a given category to SLURM.
# Each dataset runs as an independent job (parallelised over datasets).
#
# Usage:
#   bash tests/slurm_scripts/submit_all_datasets.sh complexity
#   bash tests/slurm_scripts/submit_all_datasets.sh noise --dry-run
#   bash tests/slurm_scripts/submit_all_datasets.sh similarity --domain molecular
#   bash tests/slurm_scripts/submit_all_datasets.sh complexity --datasets "qm9 caco2 logd"
#   bash tests/slurm_scripts/submit_all_datasets.sh similarity --skip "abagym_Ang2 abagym_HER2"
#   bash tests/slurm_scripts/submit_all_datasets.sh federated
#
# (The KIRBY_PRECOMPUTE_REVERSE_ORDER two-wave option was removed 2026-06-10:
# runs are single-writer now, so there is no second wave to parallelise against
# and the reverse order was the O3 cache-clobber source. Run categories
# sequentially, or as independent per-dataset jobs that never share a dataset.)
#
# Flags:
#   --domain DOMAIN     Limit to a domain (default 'all'). See list below.
#   --datasets "A B C"  Explicit dataset list (overrides --domain).
#   --skip "A B C"      Drop these dataset names from the final list (after --datasets/--domain).
#   --dry-run           Print sbatch commands without submitting.
#
# Domains: molecular, molecular_clf, dta, antibody, abagym, ab_clf,
#          ginkgo, reaction, tcga, nucleotide, all (default)

CATEGORY=${1:?Usage: bash submit_all_datasets.sh CATEGORY [--dry-run] [--domain DOMAIN] [--datasets "A B C"] [--skip "A B C"]}
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_dataset.sh"

# ── Dataset lists by domain (matches DATASET_REPS keys) ─────────────────
MOLECULAR="herg_ki caco2 logd qm9 adme_hlm adme_rlm adme_mdr1"
MOLECULAR_CLF="herg"
# davis is in the roster but PARKED (per-drug-mean degeneracy, pending job
# 7568428) — deliberately omitted from the submit list. Add when unparked.
DTA="kiba bindingdb_kd"
# Roster curated 2026-07-18 (see REP_PRUNING_LOG.md). CUT from ANTIBODY:
# flab_immunogenicity_marks, flab_binding_phillips_h3.
ANTIBODY="flab_thermo_tresanco abcov_kd ddg_combined flab_binding_koenig flab_expression_koenig flab_binding_adams flab_expression_adams"
ABAGYM="abagym_Ang2 abagym_EGFR abagym_VEGF"
# cdr_flexibility ADDED 2026-09-12. The "pending a load_data path + split
# decision" note this replaces was stale: _shared.load_data has handled it since
# the option-A call (cluster on the CDR3 loop sequences at 80% identity, not the
# whole Fv). opig_cdr_flexibility is the only experiment that reads it, and it
# had no precompute path until now.
AB_CLF="tz_her2_binary tz_her2_multiclass humanness_therapeutic cdr_flexibility"
# ginkgo_tm2/aggregation/purity CUT 2026-08-20 (near-dead targets, see DATASET_STATUS.md)
GINKGO="ginkgo_acsins ginkgo_colloidal ginkgo_hic ginkgo_pr_cho ginkgo_titer"
# suzuki ADDED 2026-09-12: opig_cgr_comparison loads all three reaction datasets
# and only two were being precomputed. Its representation pool is BORROWED from
# the other two, not measured on it.
REACTION="buchwald_hartwig bh_doyle_plates suzuki"
TCGA="tcga_luad_vs_lusc tcga_brca_pam50"
# Nucleotide curated 21->6 survivors on 2026-07-18 (weak signal / coverage).
NUCLEOTIDE="nt_H3K27me3 nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_promoter_tata gb_human_enhancers_cohn"

# Every dataset the OPIG experiments load, for `--domain opig`. All of these
# also appear in a domain list above; this group exists so the OPIG suite can be
# precomputed in one submission. Mirrors OPIG_DATASETS in
# tests/check_precompute_status.py. chembl_qsar is deliberately absent:
# opig_activity_cliff_qsar builds its own representations through
# kirby.datasets.chembl_qsar and _shared.load_data cannot load it, so precompute
# has no path to it.
OPIG="logd herg_ki cdr_flexibility ddg_combined tz_her2_binary humanness_therapeutic ginkgo_hic ginkgo_titer ginkgo_pr_cho ginkgo_acsins ginkgo_colloidal buchwald_hartwig bh_doyle_plates suzuki flab_binding_koenig flab_expression_koenig flab_binding_adams flab_expression_adams"

# Federated-eligible datasets (grows as fed02-fed05 add support)
# lhasa_herg uses Lhasa's curated hERG data (binary classification).
FEDERATED_DATASETS="lhasa_herg caco2 logd"

# ── Job name prefix ────────────────────────────────────────────────────
case $CATEGORY in
    complexity)                       PREFIX="cx" ;;
    similarity)                       PREFIX="sim" ;;
    noise)                            PREFIX="nz" ;;
    federated)                        PREFIX="fed" ;;
    all|complexity,similarity,noise)  PREFIX="all" ;;
    *,*)                              PREFIX="bundle" ;;
    *)                                PREFIX="exp" ;;
esac

# ── Parse args ──────────────────────────────────────────────────────────
# The five datasets the trial pass runs on. Kept identical to TRIAL in
# submit_precompute_extra_passes.sh and TRIAL_DATASETS in
# tests/check_rep_pools.py -- an experiment trial on a dataset the precompute
# trial did not cover is a cold build, which is the thing
# KIRBY_NO_COLD_SEARCH exists to stop.
TRIAL="buchwald_hartwig caco2 flab_expression_koenig tcga_brca_pam50 logd"

DRY_RUN=false
DOMAIN="all"
EXPLICIT_DATASETS=""
SKIP_DATASETS=""
# Which noise conditions to submit, space separated. Empty submits one job per
# dataset with no condition named, which is what every submission before
# 2026-09-23 did -- each noise experiment then runs its own default, gaussian
# for n09/n11/n12 and the outlier condition for n07.
#
# Set, it submits one job per (dataset, condition) and names each job after
# both, so two conditions on one dataset are two jobs and two result files
# rather than one overwriting the other.
NOISE_CONDITIONS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=true; shift ;;
        --trial)    EXPLICIT_DATASETS="$TRIAL"; shift ;;
        --domain)   DOMAIN=$2; shift 2 ;;
        --noise-conditions) NOISE_CONDITIONS=$2; shift 2 ;;
        --datasets) EXPLICIT_DATASETS=$2; shift 2 ;;
        --skip)     SKIP_DATASETS=$2; shift 2 ;;
        molecular|molecular_clf|dta|antibody|abagym|ab_clf|ginkgo|reaction|tcga|nucleotide) DOMAIN=$1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Build dataset list ──────────────────────────────────────────────────
# Precedence: --datasets (explicit) > federated (special) > --domain (default 'all').
if [ -n "$EXPLICIT_DATASETS" ]; then
    DATASETS="$EXPLICIT_DATASETS"
elif [ "$CATEGORY" = "federated" ]; then
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
        opig)          DATASETS="$OPIG" ;;
        reaction)      DATASETS="$REACTION" ;;
        tcga)          DATASETS="$TCGA" ;;
        nucleotide)    DATASETS="$NUCLEOTIDE" ;;
        all)           DATASETS="$MOLECULAR $MOLECULAR_CLF $DTA $ANTIBODY $ABAGYM $AB_CLF $GINKGO $REACTION $TCGA $NUCLEOTIDE" ;;
        *)             echo "Unknown domain: $DOMAIN"; exit 1 ;;
    esac
fi

# Apply --skip filter (drop named datasets from the list).
if [ -n "$SKIP_DATASETS" ]; then
    FILTERED=""
    for D in $DATASETS; do
        DROP=false
        for S in $SKIP_DATASETS; do
            if [ "$D" = "$S" ]; then DROP=true; break; fi
        done
        if [ "$DROP" = false ]; then FILTERED="$FILTERED $D"; fi
    done
    DATASETS=$(echo "$FILTERED" | xargs)  # trim whitespace
fi

# ── Per-dataset resource overrides ─────────────────────────────────────
# Large datasets (>4K samples or many high-dim reps) need more memory.
# TCGA datasets with methyl_hvg (5000D) need more time for importance.
get_mem_override() {
    case $1 in
        qm9|flab_binding_adams|flab_expression_adams)
            echo "128G" ;;
        tz_her2_binary|tz_her2_multiclass|bh_doyle_plates|flab_binding_koenig|flab_expression_koenig)
            echo "128G" ;;
        tcga_luad_vs_lusc|tcga_brca_pam50)
            echo "128G" ;;
        # DTA: kiba (n=30K, 229 protein groups) and bindingdb_kd (n=30K, 1219 groups).
        kiba|bindingdb_kd)
            echo "128G" ;;
        # Nucleotide: n=30K (nt_*) to 100K (gb_demo_*), high-dim reps need RAM.
        # Bumped 128G → 256G 2026-06-02 alongside cpus-per-task=32 in
        # submit_dataset.sh: more parallel workers via joblib need more RAM
        # to avoid OOM under find_best_hybrid's per-fold importance compute.
        nt_*|gb_*)
            echo "256G" ;;
        *) echo "" ;;
    esac
}

# Default pruning mode for the bulk submission. Override at the env-var
# level: PRUNING_MODE=default_pruning bash submit_all_datasets.sh complexity
PRUNING_MODE="${PRUNING_MODE:-aggressive_pruning}"

# Two switches that have to reach the compute node, both off unless asked for.
#
#   KIRBY_NO_COLD_SEARCH  An experiment that asks for a search the precompute
#       never wrote does not fail today: it builds the whole thing inside its
#       own wall, 8,160 configurations x 5 folds on a ten-representation pool,
#       and everything queued behind it on that dataset waits. With this set it
#       raises in minutes instead, naming the precompute command that would
#       fill it. It only fires on the tiers the precompute owns, so an
#       experiment-local tier, a label-noise run and a tuned run are untouched.
#       Set it once the precompute for these datasets has finished.
#
#   KIRBY_TUNE  Tune every model per fold, for the whole process. ⛔ NOT a
#       setting on a row: `tune` is an IDENTITY column of the result database,
#       so a tuned run shares NO row with an untuned one and needs its own
#       precompute (tests/precompute_hybrids.py --tune). It is global on
#       purpose -- an experiment reaches its hybrid and its single-
#       representation baseline through different calls, so tuning some of them
#       and not others would compare a tuned hybrid against an untuned baseline
#       with nothing raising. About 42x the fits; see the table --tune prints.
#
#   KIRBY_NO_COLD_SEARCH=1 bash submit_all_datasets.sh complexity --trial
#   KIRBY_TUNE=1 KIRBY_NO_COLD_SEARCH=1 bash submit_all_datasets.sh complexity --trial
_EXPORTS="ALL,PRUNING_MODE=${PRUNING_MODE}"
[ -n "${KIRBY_NO_COLD_SEARCH:-}" ] && _EXPORTS="${_EXPORTS},KIRBY_NO_COLD_SEARCH=${KIRBY_NO_COLD_SEARCH}"
[ -n "${KIRBY_TUNE:-}" ] && _EXPORTS="${_EXPORTS},KIRBY_TUNE=${KIRBY_TUNE}"
EXPORT_FLAG="--export=${_EXPORTS}"

# ── Check the condition names before submitting anything ────────────────
# A typo here costs the whole submission: every job reaches the compute node,
# pulls, imports, and only then raises on a name noiseInject does not carry.
# The list it is checked against is the one the experiments read.
if [ -n "${NOISE_CONDITIONS}" ]; then
    if [ "$CATEGORY" != "noise" ] && [[ "$CATEGORY" != *noise* ]]; then
        echo "--noise-conditions is only meaningful for the noise category; got '${CATEGORY}'." >&2
        exit 1
    fi
    _BAD=$(python -c '
import sys
sys.path.insert(0, "src")
from kirby.noise_spec import REGRESSION_STRATEGIES, CLASSIFICATION_STRATEGIES
known = set(REGRESSION_STRATEGIES) | set(CLASSIFICATION_STRATEGIES)
print(" ".join(c for c in sys.argv[1].split() if c not in known))
' "${NOISE_CONDITIONS}" 2>/dev/null)
    if [ -n "${_BAD}" ]; then
        echo "Unknown noise condition(s): ${_BAD}" >&2
        echo "Known names are in src/kirby/noise_spec.py (REGRESSION_STRATEGIES, CLASSIFICATION_STRATEGIES)." >&2
        exit 1
    fi
fi

# ── Submit ──────────────────────────────────────────────────────────────
COUNT=0
# One entry when no condition was asked for, so the loop body is one code path.
# The empty string leaves NOISE_CONDITION unset in submit_dataset.sh.
_COND_LIST="${NOISE_CONDITIONS:-__none__}"
for DS in $DATASETS; do
    MEM_OVERRIDE=$(get_mem_override $DS)
    for COND in $_COND_LIST; do
        if [ "$COND" = "__none__" ]; then
            JOB_NAME="${PREFIX}_${DS}"
            COND_EXPORT=""
            COND_NOTE=""
        else
            JOB_NAME="${PREFIX}_${DS}_${COND}"
            COND_EXPORT=",NOISE_CONDITION=${COND}"
            COND_NOTE="  condition=${COND}"
        fi
        if [ -n "$MEM_OVERRIDE" ]; then
            CMD="sbatch ${EXPORT_FLAG}${COND_EXPORT} --job-name=${JOB_NAME} --mem=${MEM_OVERRIDE} ${SUBMIT_SCRIPT} ${CATEGORY} ${DS}"
        else
            CMD="sbatch ${EXPORT_FLAG}${COND_EXPORT} --job-name=${JOB_NAME} ${SUBMIT_SCRIPT} ${CATEGORY} ${DS}"
        fi
        if [ "$DRY_RUN" = true ]; then
            echo "[dry-run] $CMD"
        else
            echo "Submitting: ${CATEGORY}/${DS}${MEM_OVERRIDE:+ (mem=${MEM_OVERRIDE})}  pruning=${PRUNING_MODE}${COND_NOTE}"
            $CMD
        fi
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "Total: ${COUNT} ${CATEGORY} jobs${DRY_RUN:+ (dry run)}  pruning_mode=${PRUNING_MODE}"
echo "  cold-search refusal: ${KIRBY_NO_COLD_SEARCH:-off}    tuning: ${KIRBY_TUNE:-off}"
if [ -n "${KIRBY_TUNE:-}" ]; then
    echo "  ⛔ TUNED ARM. These experiments read rows written by"
    echo "     'tests/precompute_hybrids.py ... --tune'. If that pass has not"
    echo "     run for these datasets and conditions, every one of these jobs"
    echo "     finds nothing" 
    echo "     — and with KIRBY_NO_COLD_SEARCH unset it will build it live."
fi
