#!/bin/bash
# Submit the EXTRA precompute passes that the experiment suite needs and that
# nothing launched before 2026-09-12.
#
# The plain precompute pass searches the five default downstream models under the
# predictive objective. Three decisions in EXPERIMENTS_FINALIZED.md need searches
# the plain pass does not write:
#
#   knn    s02 asks for a recipe chosen for five-nearest-neighbour prediction.
#          k-nearest-neighbours is model tier 'extra', so models=None never scores
#          it and the request falls through to the five prediction models. This
#          pass names the six-model list explicitly -- and runs it under the
#          similarity objective ONLY.
#          (--with-knn --objectives similarity)
#
#          Owner's rule, 2026-09-12: k-nearest-neighbours is not tested as a
#          standard predictive model, but it has to be available and precomputed
#          for the similarity work. So it is never added to the default model set
#          (it stays tier 'extra', and models=None still never scores it), and it
#          is scored only where nearest-neighbour prediction is the thing being
#          measured: this pass, and the distance lineage below. There is no
#          six-model PREDICTIVE search on the general tier. When s02 reads the
#          pool, the predictive family has no k-nearest-neighbours entry, so that
#          family contributes nothing and the similarity family answers.

#   dist   The distance lineage -- its own evaluators, its own budget list (10 to
#          500 features), and DISTANCE_MODELS, which is the five default models
#          plus k-nearest-neighbours. s04, s05, s06, s08, s10, s13 and c03 read
#          it. This is the other place k-nearest-neighbours is scored.
#          (--distance)
#
#   ridge  c16 compares feature selection against Ridge regularisation and needs a
#          recipe chosen for Ridge. Ridge stays tier 'extra' and stays banned as a
#          downstream evaluator everywhere else; this pass names it explicitly.
#          (--with-ridge)
#
#   sim    Fourteen files draw from a pool that includes hybrids built to predict
#          each molecule's activity from its five nearest neighbours. Those are
#          stored under objective='similarity', which the plain pass never runs.
#          (--objectives predictive,similarity)
#
# Each pass writes an EXTRA search file beside the plain one and replaces nothing:
# the sorted model names are part of the search cache key, and the objective is
# namespaced into the tier in both the cache filename and the registry key.
# Nothing already precomputed is invalidated by running any of this.
#
# Job names carry the pass as a __knn / __ridge / __sim suffix so that
# migrate_short_to_medium.sh and migrate_medium_to_long.sh rebuild the right
# command if the job is moved between partitions. Without that suffix a migrated
# job comes back as the plain pass with no message.
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh knn --dry-run
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh sim --domain molecular
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh ridge --datasets "herg_ki logd"
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh knn,sim
#
# Nothing is submitted until you review the plan and type `y`, unless --yes.

set -uo pipefail

PASSES=${1:?Usage: bash submit_precompute_extra_passes.sh knn|ridge|sim|dist[,...] [--dry-run] [--domain D] [--datasets "A B C"] [--skip "A B C"] [--conditions "c1 c2"] [--yes]}
shift

EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
PARTITION="medium"
TIME_LIMIT="2-00:00:00"
ACCOUNT="stat-cadd"

# Rosters copied from submit_all_datasets.sh. Keep them in step: that file is the
# roster source of truth and this one only adds passes to the same datasets.
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

# The augmentation conditions the plain pass writes. A pass runs the same three
# unless --conditions narrows it.
DEFAULT_CONDITIONS="general_none general_uncapped general_capped"

DRY_RUN=false
ASSUME_YES=false
DOMAIN="molecular"
EXPLICIT_DATASETS=""
SKIP_DATASETS=""
CONDITIONS="$DEFAULT_CONDITIONS"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --yes)        ASSUME_YES=true; shift ;;
        --domain)     DOMAIN=$2; shift 2 ;;
        --datasets)   EXPLICIT_DATASETS=$2; shift 2 ;;
        --skip)       SKIP_DATASETS=$2; shift 2 ;;
        --conditions) CONDITIONS=$2; shift 2 ;;
        --partition)  PARTITION=$2; shift 2 ;;
        --time)       TIME_LIMIT=$2; shift 2 ;;
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

pass_flags() {
    case "$1" in
        knn)   echo "--with-knn --objectives similarity" ;;
        ridge) echo "--with-ridge" ;;
        sim)   echo "--objectives predictive,similarity" ;;
        dist)  echo "--distance" ;;
        *)     echo "__INVALID__" ;;
    esac
}

# ── Build the plan ──────────────────────────────────────────────────────
PLAN=()
for P in ${PASSES//,/ }; do
    FLAGS=$(pass_flags "$P")
    if [ "$FLAGS" = "__INVALID__" ]; then
        echo "Unknown pass: $P (expected knn, ridge, sim or dist)"; exit 1
    fi
    # The distance lineage has its own two conditions; it has no capped one.
    if [ "$P" = "dist" ]; then
        CONDS_FOR_PASS="general_distance_none general_distance_uncapped"
    else
        CONDS_FOR_PASS="$CONDITIONS"
    fi
    for DS in $DATASETS; do
        for COND in $CONDS_FOR_PASS; do
            # No --pruning on capped. The flag could never take effect: the
            # pruning mode defaults to no_pruning, which always yields a
            # schedule, and an explicit schedule wins over the legacy threshold
            # inside the engine. The number reached the search metadata and
            # nothing else, and it read as though capped was pruned, which the
            # 2026-09-13 decision says it is not. Removed 2026-09-13.
            PRUNING=""
            MEM="64G"
            [[ "$DS" == nt_* ]] && MEM="128G"
            PLAN+=("$P|$DS|$COND|$MEM|$PRUNING|$FLAGS")
        done
    done
done

echo ""
echo "=========================================="
echo "Extra precompute passes"
echo "  passes:     $PASSES"
echo "  datasets:   $(echo $DATASETS | wc -w)"
echo "  conditions: $CONDITIONS"
echo "  partition:  $PARTITION   time: $TIME_LIMIT"
echo "  jobs:       ${#PLAN[@]}"
echo "=========================================="
for row in "${PLAN[@]}"; do
    IFS='|' read -r p ds cond mem pruning flags <<< "$row"
    printf "  %-6s %-26s %-18s %-6s %s\n" "$p" "$ds" "$cond" "$mem" "$flags"
done

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "--dry-run: nothing submitted."
    exit 0
fi

if [ "$ASSUME_YES" = false ]; then
    echo ""
    read -r -p "Submit these ${#PLAN[@]} jobs? [y/N] " REPLY
    [[ "$REPLY" == "y" || "$REPLY" == "Y" ]] || { echo "Aborted."; exit 0; }
fi

# ── Submit ──────────────────────────────────────────────────────────────
N_OK=0
N_FAIL=0
for row in "${PLAN[@]}"; do
    IFS='|' read -r p ds cond mem pruning flags <<< "$row"
    cmd="python tests/precompute_hybrids.py ${ds} --condition ${cond} --speed full ${pruning} ${flags}"
    jid=$(sbatch \
        --account="${ACCOUNT}" \
        --job-name="pc_${ds}_${cond}__${p}" \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time="${TIME_LIMIT}" --partition="${PARTITION}" \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}" 2>&1 | awk '/Submitted/ {print $4}')
    if [[ -n "$jid" ]]; then
        echo "  SUBMIT  $jid  pc_${ds}_${cond}__${p}  ($mem)"
        N_OK=$((N_OK + 1))
    else
        echo "  SUBMIT  FAILED  pc_${ds}_${cond}__${p}"
        N_FAIL=$((N_FAIL + 1))
    fi
done

echo ""
echo "Submitted $N_OK, failed $N_FAIL."
echo "Monitor with: squeue -u \$USER -o '%.10i %.8P %.44j %.2t %.10M %.10L' | grep pc_"
