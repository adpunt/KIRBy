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
TUNE_FLAG=""          # --tune: a SECOND precompute, see the note at the plan
CHECKS_ONLY=false     # run the three checks and submit nothing
SKIP_CHECKS=false     # submit without them, because they were run in an srun
STAGGER=5             # seconds between submissions, to spread the job starts

# Rosters copied from submit_all_datasets.sh. Keep them in step: that file is the
# roster source of truth and this one only adds passes to the same datasets.
MOLECULAR="herg_ki caco2 logd qm9 adme_hlm adme_rlm adme_mdr1"
# lhasa_herg joined DATASET_REPS 2026-09-17 with the same pool as herg. It is
# the federated suite's third dataset and had no precompute path before that.
MOLECULAR_CLF="herg lhasa_herg"
DTA="kiba bindingdb_kd"
ANTIBODY="flab_thermo_tresanco abcov_kd ddg_combined flab_binding_koenig flab_expression_koenig flab_binding_adams flab_expression_adams"
ABAGYM="abagym_Ang2 abagym_EGFR abagym_VEGF"
# cdr_flexibility and suzuki were added to submit_all_datasets.sh on 2026-09-12
# and were never copied here, so --domain all gave 38 of the 42 names in
# DATASET_REPS until 2026-09-14. The check below now catches that.
AB_CLF="tz_her2_binary tz_her2_multiclass humanness_therapeutic cdr_flexibility"
GINKGO="ginkgo_acsins ginkgo_colloidal ginkgo_hic ginkgo_pr_cho ginkgo_titer"
REACTION="buchwald_hartwig bh_doyle_plates suzuki"
TCGA="tcga_luad_vs_lusc tcga_brca_pam50"
NUCLEOTIDE="nt_H3K27me3 nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_promoter_tata gb_human_enhancers_cohn"

# The five datasets the trial pass runs on, before the other 37.
TRIAL="buchwald_hartwig caco2 flab_expression_koenig tcga_brca_pam50 logd"

# Names in DATASET_REPS that are DELIBERATELY not in any domain list above.
# Each needs a stated reason; without this list the consistency check below
# cannot tell a deliberate omission from a stale copy, which is exactly how
# cdr_flexibility and suzuki went missing for two days.
#
#   chembl_qsar  the name appears nowhere in tests/experiments/_shared.py, so
#                load_data has no path to it and precompute_hybrids.py cannot
#                run it at all. opig_activity_cliff_qsar builds its own
#                representations through kirby.datasets.chembl_qsar.
#   davis        PARKED — per-drug-mean degeneracy. Already absent from the
#                roster in submit_all_datasets.sh for the same reason.
EXCLUDED="chembl_qsar davis"

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
        --tune)       TUNE_FLAG="--tune"; shift ;;
        --checks-only) CHECKS_ONLY=true; shift ;;
        --skip-checks) SKIP_CHECKS=true; shift ;;
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

# ── Is the roster above still the whole roster? ─────────────────────────
# The domain lists are COPIES of the ones in submit_all_datasets.sh, and a copy
# goes stale silently: a dataset added there is simply never precomputed here,
# and the only symptom is an experiment finding nothing months later. Compare
# the union against DATASET_REPS and say so. Costs one rdkit import, so it runs
# only where the checks are allowed to run.
if [ "$SKIP_CHECKS" = false ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    _ALL_LISTED="$MOLECULAR $MOLECULAR_CLF $DTA $ANTIBODY $ABAGYM $AB_CLF $GINKGO $REACTION $TCGA $NUCLEOTIDE $EXCLUDED"
    _DRIFT=$(cd "$REPO" && KIRBY_LISTED="$_ALL_LISTED" python - <<'PY'
import os, sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'tests')
from experiments.complexity.complexity_helpers import DATASET_REPS
listed = set(os.environ.get('KIRBY_LISTED', '').split())
known = set(DATASET_REPS)
missing = sorted(known - listed)
unknown = sorted(listed - known)
if missing:
    print("MISSING from every domain list and from EXCLUDED: " + " ".join(missing))
if unknown:
    print("LISTED but not in DATASET_REPS: " + " ".join(unknown))
PY
)
    if [ -n "$_DRIFT" ]; then
        echo ""
        echo "⛔ ROSTER DRIFT — the domain lists in this script no longer match"
        echo "   DATASET_REPS. A name missing below is never precomputed by any"
        echo "   --domain, and nothing else reports that."
        echo "$_DRIFT" | sed 's/^/   /'
        echo "   Add it to the right domain list, or to EXCLUDED with a reason."
    fi
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
#
# ⛔ THE CHECKS RUN IN AN ALLOCATION, NOT ON A LOGIN NODE. Owner's rule,
# 2026-09-14. They are not free: check_rep_pools.py imports rdkit and numpy and
# then stats every representation file of every named dataset over NFS, and the
# checkout check does a git fetch. A login node is shared by everyone.
#
# So there are two ways to run this:
#
#   in an srun   bash ... --checks-only      then read the output
#                bash ...                    (checks + submit, both inside)
#   on a login   bash ... --skip-checks      after the checks passed in the srun
#
# --checks-only WRITES what the gate decided to $KIRBY_CACHE_DIR/precompute_gate.txt,
# and --skip-checks READS it. So the gate still applies when the submission
# happens on a login node -- it is not re-derived there, and it is not thrown
# away either. --skip-checks with no such file is refused rather than quietly
# planning every condition of every dataset, which is the failure the gate
# exists to stop.
if [ "$SKIP_CHECKS" = false ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    echo ""
    echo "⛔ REFUSING TO RUN THE CHECKS ON A LOGIN NODE."
    echo "   They import rdkit, stat every representation file over NFS and run"
    echo "   a git fetch. Get an allocation first — take the line"
    echo "   where_to_submit.sh section 6 prints, e.g."
    echo ""
    echo "     srun -p interactive --cpus-per-task=2 --mem=8G --time=0:30:00 --pty /bin/bash"
    echo "     cd $REPO && export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh"
    echo "     bash $0 $PASSES <your flags> --checks-only"
    echo ""
    echo "   Then either submit from inside that shell, or from the login node with"
    echo "   --skip-checks once the checks have passed."
    exit 1
fi

# 1. Is this checkout what the remote has? The jobs do not git pull: a job runs
#    whatever is on disk when it starts, and thirty jobs pulling one shared
#    checkout at once is worse than none of them pulling. So the pull is yours,
#    once, and this only reports whether you did it.
if [ "$SKIP_CHECKS" = false ]; then
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
fi   # end SKIP_CHECKS

# Account and partition still have to be known when the checks were skipped.
if [ -z "$ACCOUNT" ] || [ -z "$PARTITION" ]; then
    echo ""
    echo "⛔ --skip-checks means where_to_submit.sh --emit was not consulted."
    echo "   Name them: --account <name> --partition <name>"
    echo "   (the srun that ran --checks-only printed the pair it resolved)"
    exit 1
fi

# The pass definitions live in ONE file, shared with the two migrate scripts,
# because three copies of them drifted and two of the copies were wrong. See
# the note at the top of that file.
. "$(dirname "$0")/_precompute_passes.sh"

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
GATE_FILE="${KIRBY_CACHE_DIR:-$HOME/.cache/kirby}/precompute_gate.txt"
allowed_conds() {
    printf '%s\n' "$GATE_TABLE" | awk -v d="$1" '$1==d {print $2; exit}'
}

# Submitting from a login node after the checks ran in an srun: read back what
# the gate decided rather than re-deriving it here or discarding it.
if [ "$SKIP_CHECKS" = true ] && [ "$GATE" = true ]; then
    if [ ! -f "$GATE_FILE" ]; then
        echo ""
        echo "⛔ --skip-checks, but no gate result at $GATE_FILE."
        echo "   Run the checks in an srun first:"
        echo "       bash $0 $PASSES <your flags> --checks-only"
        echo "   or pass --no-gate to plan every condition of every dataset"
        echo "   regardless of which representations are on disk."
        exit 1
    fi
    _age=$(( $(date +%s) - $(date -r "$GATE_FILE" +%s 2>/dev/null || echo 0) ))
    echo ""
    echo "REPRESENTATION GATE  (read from $GATE_FILE, written $(( _age / 60 )) min ago)"
    KEPT=""
    while read -r g_ds g_conds; do
        [ -z "$g_ds" ] && continue
        GATE_TABLE="${GATE_TABLE}${g_ds} ${g_conds}
"
        KEPT="$KEPT $g_ds"
    done < "$GATE_FILE"
    _FILTERED=""
    for D in $DATASETS; do
        _c=$(allowed_conds "$D")
        if [ -z "$_c" ]; then
            echo "  DROPPED   $D — not in the gate result (pool incomplete, or not checked)"
        else
            echo "  ok        $D — $_c"
            _FILTERED="$_FILTERED $D"
        fi
    done
    DATASETS=$(echo "$_FILTERED" | xargs)
    if [ -z "$DATASETS" ]; then
        echo "  Nothing left to submit."
        exit 1
    fi
fi

if [ "$GATE" = true ] && [ "$SKIP_CHECKS" = false ]; then
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
    mkdir -p "$(dirname "$GATE_FILE")" 2>/dev/null
    printf '%s' "$GATE_TABLE" > "$GATE_FILE" \
        && echo "  written to $GATE_FILE for a later --skip-checks submission"
fi

if [ "$CHECKS_ONLY" = true ]; then
    echo ""
    echo "--checks-only: nothing planned, nothing submitted."
    echo "Submit with the pair the checks resolved:"
    echo "    bash $0 $PASSES <your flags> --account $ACCOUNT --partition $PARTITION --skip-checks"
    exit 0
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
echo "  tuning:     ${TUNE_FLAG:-off}"
if [ -n "$TUNE_FLAG" ]; then
    echo ""
    echo "  ⛔ TUNED PASS. \`tune\` is an IDENTITY column of the result database,"
    echo "     so these rows are a SEPARATE SET -- nothing untuned is reused and"
    echo "     no untuned experiment reads them. Measured 2026-09-16, the whole"
    echo "     grid is 29.6x the untuned wall, and xgboost and rf are 97% of"
    echo "     that. At --speed full one dataset and one condition is 6 to 19"
    echo "     days against a 7-day maximum: it does NOT fit. See section 10.4"
    echo "     of PRECOMPUTE_RUN_PLAN_2026_09_14.md before submitting this."
fi
echo "=========================================="
if [ "${#PLAN[@]}" -eq 0 ]; then
    echo ""
    echo "  Nothing to submit. Every (dataset, condition) the passes asked for"
    echo "  was dropped by the representation gate above. --no-gate plans them"
    echo "  anyway."
    exit 1
fi
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
    jobname="$(job_name "$ds" "$cond" "$p" "$TUNE_FLAG")"
    cmd="python -u tests/precompute_hybrids.py ${ds} --condition ${cond} --speed full ${pruning} ${flags} ${TUNE_FLAG}"
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
