#!/bin/bash
# Run the augmentation showcase (c21) alongside the hybrid precompute.
#
# Your decision, 2026-09-10: "Augmentation already exists, it just needs to
# showcase that... showcasing each base rep with the different augmented
# versions (capped/uncapped, features coming from curated or full pools) is
# already being done. This actually belongs on some level of precompute. While
# the hybrids are being precomputed so should the augmentation experiments."
#
# c21 already is that showcase, and it builds no search of its own -- it builds
# features and fits models. So this runs c21, rather than moving four hundred
# lines of it into precompute_hybrids.py where the logic would exist twice.
#
# What each dataset runs, after the 2026-09-12 defaults:
#   every base representation in the pool          (base_reps=None)
#   x capped and uncapped                          (DEFAULT_STRATEGIES)
#   x donors from the curated augmentation pool (condition B), from the other
#     nine base representations (condition C), and from both (condition D),
#     plus the two random-donor controls           (DEFAULT_TOP_N = 9)
#   x 3 downstream models, rf / xgboost / lightgbm (knn is out: it is not tested
#     as a standard predictive model, and every arm here is a predictive one)
#   x 5 folds
#
# Usage:
#   bash tests/slurm_scripts/submit_augmentation_showcase.sh --dry-run
#   bash tests/slurm_scripts/submit_augmentation_showcase.sh --domain molecular
#   bash tests/slurm_scripts/submit_augmentation_showcase.sh --datasets "herg_ki logd"
#
# Nothing is submitted until you review the plan and type `y`, unless --yes.

set -uo pipefail

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
PARTITION="medium"
TIME_LIMIT="1-00:00:00"
ACCOUNT="stat-cadd"

# Rosters copied from submit_all_datasets.sh. That file is the roster source of
# truth; this one only runs one more experiment on the same datasets.
MOLECULAR="herg_ki caco2 logd qm9 adme_hlm adme_rlm adme_mdr1"
MOLECULAR_CLF="herg"
DTA="kiba bindingdb_kd"
ANTIBODY="flab_thermo_tresanco abcov_kd ddg_combined flab_binding_koenig flab_expression_koenig flab_binding_adams flab_expression_adams"
ABAGYM="abagym_Ang2 abagym_EGFR abagym_VEGF"
AB_CLF="tz_her2_binary tz_her2_multiclass humanness_therapeutic"
GINKGO="ginkgo_acsins ginkgo_colloidal ginkgo_hic ginkgo_pr_cho ginkgo_titer"
REACTION="buchwald_hartwig bh_doyle_plates"
TCGA="tcga_luad_vs_lusc tcga_brca_pam50"
NUCLEOTIDE="nt_H3K27me3 nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_promoter_tata gb_human_enhancers_cohn"

DRY_RUN=false
ASSUME_YES=false
DOMAIN="all"
EXPLICIT_DATASETS=""
SKIP_DATASETS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)   DRY_RUN=true; shift ;;
        --yes)       ASSUME_YES=true; shift ;;
        --domain)    DOMAIN=$2; shift 2 ;;
        --datasets)  EXPLICIT_DATASETS=$2; shift 2 ;;
        --skip)      SKIP_DATASETS=$2; shift 2 ;;
        --partition) PARTITION=$2; shift 2 ;;
        --time)      TIME_LIMIT=$2; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -n "$EXPLICIT_DATASETS" ]; then
    DATASETS="$EXPLICIT_DATASETS"
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

if [ -n "$SKIP_DATASETS" ]; then
    FILTERED=""
    for D in $DATASETS; do
        DROP=false
        for S in $SKIP_DATASETS; do
            if [ "$D" = "$S" ]; then DROP=true; break; fi
        done
        if [ "$DROP" = false ]; then FILTERED="$FILTERED $D"; fi
    done
    DATASETS=$(echo "$FILTERED" | xargs)
fi

N=$(echo $DATASETS | wc -w)
echo ""
echo "=========================================="
echo "Augmentation showcase (c21), one job per dataset"
echo "  datasets:  $N"
echo "  partition: $PARTITION   time: $TIME_LIMIT"
echo "=========================================="
for DS in $DATASETS; do
    MEM="64G"
    [[ "$DS" == nt_* ]] && MEM="128G"
    printf "  %-26s %s\n" "$DS" "$MEM"
done

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "--dry-run: nothing submitted."
    exit 0
fi

if [ "$ASSUME_YES" = false ]; then
    echo ""
    read -r -p "Submit these $N jobs? [y/N] " REPLY
    [[ "$REPLY" == "y" || "$REPLY" == "Y" ]] || { echo "Aborted."; exit 0; }
fi

N_OK=0
N_FAIL=0
for DS in $DATASETS; do
    MEM="64G"
    [[ "$DS" == nt_* ]] && MEM="128G"
    cmd="python tests/experiments/run_dataset.py complexity ${DS} --experiments c21"
    jid=$(sbatch \
        --account="${ACCOUNT}" \
        --job-name="aug_${DS}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time="${TIME_LIMIT}" --partition="${PARTITION}" \
        --mem="${MEM}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}" 2>&1 | awk '/Submitted/ {print $4}')
    if [[ -n "$jid" ]]; then
        echo "  SUBMIT  $jid  aug_${DS}  ($MEM)"
        N_OK=$((N_OK + 1))
    else
        echo "  SUBMIT  FAILED  aug_${DS}"
        N_FAIL=$((N_FAIL + 1))
    fi
done

echo ""
echo "Submitted $N_OK, failed $N_FAIL."
echo "Monitor with: squeue -u \$USER -o '%.10i %.8P %.30j %.2t %.10M %.10L' | grep aug_"
