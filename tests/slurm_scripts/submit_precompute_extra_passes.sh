#!/bin/bash
# Submit the precompute passes, including the plain one. THE stage-1 submitter.
#
# Five passes. `main` is the plain predictive search over the five default
# downstream models -- what every complexity, noise and opig experiment reads
# first, and what nothing launched from a script before 2026-09-14. The other
# four write searches `main` does not, and each is an EXTRA file beside it:
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
#   main   The plain predictive search: the five default downstream models, the
#          full budget list, the full allocation / importance / evaluator lists
#          and the full representation-count list. Every other pass is measured
#          against it. Its job name carries NO pass suffix, which is what
#          migrate_short_to_medium.sh already parses as the plain pass.
#          (no extra flags)
#
#   sim    Fourteen files draw from a pool that includes hybrids built to predict
#          each molecule's activity from its five nearest neighbours. Those are
#          stored under objective='similarity', which the plain pass never runs.
#          (--objectives predictive,similarity)
#
#          ⛔ sim IS A SUPERSET OF main. It names both objectives, so it writes
#          the predictive search as well. Naming main,sim runs the predictive
#          half twice -- the second is a cache hit on the search file and cheap,
#          but it is a queue slot for nothing. Run `sim` alone when you want
#          both, and `main` alone when you want predictive only.
#
#          The similarity half is a QUARTER of the predictive half, not an equal
#          partner: its representation-count axis is one value (the whole pool)
#          instead of four, because a count below the pool size needs per-fold
#          allocation scores and a geometry objective computes none. 2,040
#          configurations against 8,160. Measured 2026-09-14.
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
# THREE THINGS IT DOES BEFORE IT SUBMITS ANYTHING, added 2026-09-14:
#
#   1. Asks where_to_submit.sh --emit for the account and partition rather than
#      hardcoding them. The rule lives in that one file: account = whichever
#      allowed account has the higher fair share. --account / --partition still
#      override.
#   2. Runs tests/check_rep_pools.py over the dataset list and DROPS any dataset
#      whose representation pool is incomplete. precompute_hybrids.py refuses
#      such a dataset anyway (its rep_pool is part of the result database's
#      identity, so a search over 7 of 10 is a search nothing will ever read) --
#      but it refuses AFTER the queue wait and the data load. This refuses
#      before submission. --no-gate skips it.
#   3. Checks the checkout matches its remote branch, because the jobs do NOT
#      git pull. A job runs whatever is on disk when it starts, and thirty jobs
#      pulling one shared checkout at once is worse than none of them pulling.
#      Pull once by hand, then submit.
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh main --trial --dry-run
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh main,sim,dist --trial
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh knn --dry-run
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh sim --domain molecular
#   bash tests/slurm_scripts/submit_precompute_extra_passes.sh ridge --datasets "herg_ki logd"
#
# Nothing is submitted until you review the plan and type `y`, unless --yes.

set -uo pipefail

PASSES=${1:?Usage: bash submit_precompute_extra_passes.sh main|knn|ridge|sim|dist[,...] [--trial] [--dry-run] [--domain D] [--datasets "A B C"] [--skip "A B C"] [--conditions "c1 c2"] [--yes]}
shift

# KIRBY_REPO lets this be dry-run from a checkout elsewhere.
REPO="${KIRBY_REPO:-/data/stat-ecr/scat9264/KIRBy}"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
# -u so the log shows progress. Without it Python buffers stdout in ~8 KB
# blocks, which on a cache-hit job can hide hours of work and makes a running
# job indistinguishable from a hung one.
WRAP_PREFIX="cd ${REPO} && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
PARTITION=""          # from where_to_submit.sh --emit unless --partition given
ACCOUNT=""            # same
TIME_LIMIT="2-00:00:00"
CPUS=8
# ARC nodes are 8 GB per core (where_to_submit.sh section 3 prints the live
# ratio). Asking above the ratio reserves cores the job never uses and waits for
# a whole node -- that is what left f28 job 12923011 pending for a week at 16
# cores and 256 GB. 8 cores x 8 GB is exactly the ratio.
MEM_DEFAULT="64G"
MEM_LARGE="128G"      # nt_* / gb_* / tcga_* / the two DTA sets
GATE=true
STAGGER=5             # seconds between submissions, to spread the job starts

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

# The five datasets stage 0 and stage 1 trial on, before the other 37.
TRIAL="buchwald_hartwig caco2 flab_expression_koenig tcga_brca_pam50 logd"

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
        --account)    ACCOUNT=$2; shift 2 ;;
        --time)       TIME_LIMIT=$2; shift 2 ;;
        --cpus)       CPUS=$2; shift 2 ;;
        --mem)        MEM_DEFAULT=$2; MEM_LARGE=$2; shift 2 ;;
        --trial)      EXPLICIT_DATASETS="$TRIAL"; shift ;;
        --no-gate)    GATE=false; shift ;;
        --stagger)    STAGGER=$2; shift 2 ;;
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

# ── BEFORE ANYTHING IS PLANNED ──────────────────────────────────────────

# 1. Is this checkout what the remote has? The jobs do not git pull: a job runs
#    whatever is on disk when it starts, and thirty jobs pulling one shared
#    checkout at once is worse than none of them pulling. So the pull is yours,
#    once, and this only reports whether you did it.
BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
git -C "$REPO" fetch -q origin "$BRANCH" 2>/dev/null
BEHIND=$(git -C "$REPO" rev-list --count "HEAD..origin/$BRANCH" 2>/dev/null || echo 0)
echo ""
echo "CHECKOUT   $BRANCH ($COMMIT) at $REPO"
if [ "${BEHIND:-0}" -gt 0 ]; then
    echo "  ⛔ $BEHIND commit(s) behind origin/$BRANCH. Every job below would run"
    echo "     the OLDER code, with no message saying so. Run:"
    echo "         git -C $REPO pull --ff-only origin $BRANCH"
    echo "     then submit again."
    exit 1
fi

# 2. Ask the one file that owns the account/partition rule.
if [ -z "$ACCOUNT" ] || [ -z "$PARTITION" ]; then
    EMITTED=$(EMIT_PARTITION="${PARTITION:-medium}" \
              bash "$REPO/tests/slurm_scripts/where_to_submit.sh" --emit 2>/dev/null)
    if [ -n "$EMITTED" ]; then
        [ -z "$ACCOUNT" ]   && ACCOUNT=$(echo "$EMITTED" | awk '{print $1}')
        [ -z "$PARTITION" ] && PARTITION=$(echo "$EMITTED" | awk '{print $2}')
        echo "WHERE      account=$ACCOUNT partition=$PARTITION (where_to_submit.sh --emit)"
    else
        echo "  ⛔ where_to_submit.sh --emit said nothing. Name them yourself:"
        echo "     --account <name> --partition <name>"
        exit 1
    fi
else
    echo "WHERE      account=$ACCOUNT partition=$PARTITION (named on the command line)"
fi
echo "           --emit does NOT consider memory. Check the shape before you trust it:"
echo "               bash $REPO/tests/slurm_scripts/where_to_submit.sh --fits $CPUS $MEM_DEFAULT"

pass_flags() {
    case "$1" in
        main)  echo "" ;;
        knn)   echo "--with-knn --objectives similarity" ;;
        ridge) echo "--with-ridge" ;;
        sim)   echo "--objectives predictive,similarity" ;;
        dist)  echo "--distance" ;;
        *)     echo "__INVALID__" ;;
    esac
}

# The job-name suffix that puts the pass back when a job is migrated between
# partitions. The plain pass has NONE -- migrate_short_to_medium.sh:44 already
# reads an unsuffixed pc_<dataset>_<condition> as the plain pass, so `main` must
# not invent one or a migrated job would come back as an unknown pass.
pass_suffix() {
    case "$1" in
        main) echo "" ;;
        *)    echo "__$1" ;;
    esac
}

# 3. Ask check_rep_pools.py which conditions each dataset can ACTUALLY run,
#    and plan only those.
#
#    Two different gaps, and they are not the same thing:
#      - an incomplete POOL means precompute_hybrids.py refuses the dataset
#        outright (rep_pool is part of the result database's identity, so a
#        search over 7 of 10 representations is one no experiment asking for the
#        full pool will ever read). The dataset is dropped entirely.
#      - absent AUGMENTATION representations mean the job keeps 'none' and drops
#        'uncapped' and 'capped'. Queuing those two anyway buys a node, loads the
#        data, finds no condition to run and exits 0 having written nothing.
#        7 of the 42 datasets have no augmentation representations at all
#        (the three abagym sets, cdr_flexibility, humanness_therapeutic and the
#        two tz_her2 sets), and qm9's four are absent -- so with --all that is 16
#        empty jobs.
#
#    --no-gate plans every condition for every dataset regardless.
#
#    Held as a newline-delimited "<dataset> <conds>" string rather than an
#    associative array: `declare -A` is bash 4+, and this script has to be
#    dry-runnable on a mac (bash 3.2) as well as run on ARC.
GATE_TABLE=""
allowed_conds() {
    printf '%s\n' "$GATE_TABLE" | awk -v d="$1" '$1==d {print $2; exit}'
}
if [ "$GATE" = true ]; then
    echo ""
    echo "REPRESENTATION GATE"
    GATE_OUT=$(cd "$REPO" && python tests/check_rep_pools.py $DATASETS --emit-conditions 2>&1 >/tmp/kirby_gate_$$.txt)
    GATE_RC=$?
    if [ "$GATE_RC" -eq 2 ]; then
        echo "  ⛔ $GATE_OUT"
        rm -f /tmp/kirby_gate_$$.txt
        exit 1
    fi
    KEPT=""
    while read -r g_ds g_conds; do
        [ -z "$g_ds" ] && continue
        GATE_TABLE="${GATE_TABLE}${g_ds} ${g_conds}
"
        KEPT="$KEPT $g_ds"
    done < /tmp/kirby_gate_$$.txt
    rm -f /tmp/kirby_gate_$$.txt

    for D in $DATASETS; do
        _c=$(allowed_conds "$D")
        if [ -z "$_c" ]; then
            echo "  DROPPED   $D — representation pool incomplete, the job would refuse it"
        elif [ "$_c" = "none" ]; then
            echo "  none only $D — augmentation representations absent"
        else
            echo "  ok        $D — $_c"
        fi
    done
    DATASETS=$(echo "$KEPT" | xargs)
    if [ -z "$DATASETS" ]; then
        echo "  Nothing left to submit."
        exit 1
    fi
fi

# ── Build the plan ──────────────────────────────────────────────────────
PLAN=()
for P in ${PASSES//,/ }; do
    FLAGS=$(pass_flags "$P")
    if [ "$FLAGS" = "__INVALID__" ]; then
        echo "Unknown pass: $P (expected main, knn, ridge, sim or dist)"; exit 1
    fi
    # The distance lineage has its own two conditions; it has no capped one.
    if [ "$P" = "dist" ]; then
        CONDS_FOR_PASS="general_distance_none general_distance_uncapped"
    else
        CONDS_FOR_PASS="$CONDITIONS"
    fi
    for DS in $DATASETS; do
        for COND in $CONDS_FOR_PASS; do
            # Skip a condition this dataset cannot run. The distance lineage has
            # its own two conditions and carries no augmentation beyond
            # 'uncapped', so map both families onto the same none/uncapped/capped
            # words the gate speaks.
            if [ "$GATE" = true ]; then
                _want="${COND#general_distance_}"; _want="${_want#general_}"
                case ",$(allowed_conds "$DS")," in
                    *",$_want,"*) : ;;
                    *) continue ;;
                esac
            fi
            # No --pruning on capped. The flag could never take effect: the
            # pruning mode defaults to no_pruning, which always yields a
            # schedule, and an explicit schedule wins over the legacy threshold
            # inside the engine. The number reached the search metadata and
            # nothing else, and it read as though capped was pruned, which the
            # 2026-09-13 decision says it is not. Removed 2026-09-13.
            PRUNING=""
            # Wide representations or many rows. tcga carries five whole-slide
            # image embeddings; nt_/gb_ carry the DNA language models; the two
            # DTA sets are ~30k rows with protein embeddings beside them.
            MEM="$MEM_DEFAULT"
            case "$DS" in
                nt_*|gb_*|tcga_*|kiba|bindingdb_kd|davis|qm9) MEM="$MEM_LARGE" ;;
            esac
            PLAN+=("$P|$DS|$COND|$MEM|$PRUNING|$FLAGS")
        done
    done
done

# main and sim overlap: sim names both objectives.
case ",${PASSES//,/,}," in
    *,main,*) case ",${PASSES//,/,}," in
        *,sim,*)
            echo ""
            echo "  ⛔ main AND sim both named. sim runs --objectives"
            echo "     predictive,similarity, so the predictive search is"
            echo "     planned twice. The second is a cache hit and cheap, but"
            echo "     it is a queue slot for nothing. Drop one:"
            echo "       sim  alone -> predictive AND similarity"
            echo "       main alone -> predictive only"
            ;;
    esac ;;
esac

echo ""
echo "=========================================="
echo "Precompute passes"
echo "  passes:     $PASSES"
echo "  datasets:   $(echo $DATASETS | wc -w)  ($DATASETS)"
echo "  conditions: $CONDITIONS"
echo "  account:    $ACCOUNT   partition: $PARTITION   time: $TIME_LIMIT"
echo "  shape:      ${CPUS} cores, ${MEM_DEFAULT} (${MEM_LARGE} for the wide/large sets)"
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
    jobname="pc_${ds}_${cond}$(pass_suffix "$p")"
    cmd="python -u tests/precompute_hybrids.py ${ds} --condition ${cond} --speed full ${pruning} ${flags}"
    jid=$(sbatch \
        --account="${ACCOUNT}" \
        --job-name="${jobname}" \
        --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
        --time="${TIME_LIMIT}" --partition="${PARTITION}" \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}" 2>&1 | awk '/Submitted/ {print $4}')
    if [[ -n "$jid" ]]; then
        echo "  SUBMIT  $jid  ${jobname}  (${CPUS}c ${mem})"
        N_OK=$((N_OK + 1))
    else
        echo "  SUBMIT  FAILED  ${jobname}"
        N_FAIL=$((N_FAIL + 1))
    fi
    # Spread the starts. Every condition of one dataset writes to the one
    # result_db/<dataset>.db; ResultCache waits 120 s for another writer's lock
    # and then the search dies AFTER paying for its baselines and its per-fold
    # scores (result_cache.py:187-200). Staggering costs nothing and makes a
    # simultaneous pile-up on one file much less likely.
    [ "${STAGGER:-0}" -gt 0 ] && sleep "$STAGGER"
done

echo ""
echo "Submitted $N_OK, failed $N_FAIL."
echo "Watch the queue:"
echo "    squeue -u \$USER -o '%.10i %.8P %.44j %.2t %.10M %.10L' | grep pc_"
echo "Watch the work (configurations stored against configurations asked for):"
echo "    python tests/precompute_coverage.py $DATASETS"
echo "Read the real memory and wall off the first one that finishes, and size the rest from it:"
echo "    sacct -j <jobid> --format=JobID,JobName%40,MaxRSS,Elapsed,State"
