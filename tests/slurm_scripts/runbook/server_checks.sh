#!/bin/bash
# SERVER CHECKS: one short srun per (dataset, experiment), many at once, in the
# background. Written 2026-09-25. Dirty checks, not results: see README.md,
# section "Server checks".
#
#   bash tests/slurm_scripts/runbook/server_checks.sh --dry-run
#   bash tests/slurm_scripts/runbook/server_checks.sh
#   bash tests/slurm_scripts/runbook/server_checks_status.sh
#
# WHAT IT LAUNCHES. One srun per line below. Each runs on a compute node and
# sources setup.sh with KIRBY_SKIP_INSTALLS=1, so every job reads the one cache
# setup.sh names (KIRBY_CACHE_DIR=/data/stat-ecr/scat9264/.cache/kirby).
#   noise       every experiment in noise_helpers.SINGLE_DATASET_EXPERIMENTS
#               (n42 included), on each of --datasets. noise_prelim/run_one.py:
#               fold 0, one condition, one level. It runs TWICE, each run with
#               its own output directory and its own KIRBY_NOISE_CACHE_REPORT,
#               then tests/experiments/noise/check_noise_run.py reads both.
#               Condition: --cond-reg (default outlier_p10) on a regression
#               dataset, --cond-clf (default: the first classification
#               condition) on a classification one; only for an experiment
#               that takes a condition (HONOURS_NOISE_CONDITION |
#               SWEEPS_ALL_CONDITIONS). n21 and n25 get a condition from
#               CONDITION_LIMITED_TO when the default is outside it.
#   similarity  every experiment in similarity_helpers.SINGLE_DATASET_EXPERIMENTS
#               on each of --datasets: submit_dataset.sh similarity <ds> <exp>,
#               which runs run_dataset.py --experiments <exp>, results to
#               tests/results/trial_similarity/.
#   federated   the same for federated_helpers.SINGLE_DATASET_EXPERIMENTS on
#               --fed-datasets (silo data exists for lhasa_herg, caco2, logd
#               only), results to tests/results/trial_federated/.
#   opig        runbook/opig_one.sh <name> for every row of OPIG_JOBS in
#               submit_everything.sh, with that row's flags and memory,
#               results to tests/results/trial_opig/.
#   complexity  only when named in --categories. Not in the default set.
# similarity, federated and opig run with KIRBY_FORCE_RERUN=1 by default, so a
# result file left in trial_<category>/ by stage 2 is rebuilt, not skipped.
#
# WHERE IT WRITES. Never tests/results/noise and never tests/results/<category>.
#   logs/server_checks/<category>_<dataset>_<experiment>.log   srun output
#   logs/server_checks/<...>.rc        exit code of the srun (absent = not done)
#   logs/server_checks/<...>.started   launch time
#   logs/server_checks/<...>.steps     noise only: run 1, run 2 and check exit codes
#   logs/server_checks/commands.tsv    every planned command, one per line
#   tests/results/server_checks/noise/<dataset>__<experiment>/{run1,run2,check.txt,cache_run*.json}
# A previous launch's logs/server_checks/ and tests/results/server_checks/ are
# moved to logs/server_checks_previous/<time>/ and
# tests/results/server_checks_previous/<time>/ first. Nothing is deleted.
#
# BEFORE LAUNCHING, ONCE: git pull --ff-only of this checkout's branch, then
# `. setup.sh` with KIRBY_SKIP_INSTALLS=1 (needed to list the experiments).
# Every srun carries KIRBY_NO_PULL=1, so no job pulls again.
#
# THROTTLE. A background dispatcher (nohup, setsid) keeps at most
# --max-parallel srun commands alive at once and starts the next when one ends.
# It survives logging out. Its own log is logs/server_checks/_dispatcher.log
# and its pid is in logs/server_checks/_dispatcher.pid. To stop it:
#   kill $(cat logs/server_checks/_dispatcher.pid)
# (the srun commands already started keep running; scancel them by job name
# prefix sc_ if needed, one at a time).
#
# CLUSTER, ACCOUNT. Default --clusters htc; the account is the best fair share
# on htc from where_to_submit.sh --best-cluster-account htc, falling back to
# stat-cadd (the highest on 2026-09-24: htc stat-cadd 0.0491 > htc stat-ecr
# 0.0360 > arc stat-cadd 0.0326 > arc stat-ecr 0.0195). --account wins when
# given. Before launching, one 1-minute srun is tried; if srun rejects
# --clusters on this login node, the launch stops and says to log in to
# htc-login.arc.ox.ac.uk and pass --clusters "".
#
# OPTIONS
#   --datasets "A B"        default "herg_ki herg" (one regression, one classification)
#   --fed-datasets "A B"    default: those of --datasets with silo data, else caco2
#   --categories "..."      default "noise similarity federated opig"
#   --experiments "..."     only these ids (any category); default all
#   --cond-reg C            default outlier_p10
#   --cond-clf C            default first classification condition
#   --clusters C            default htc; "" for none (on htc-login)
#   --account A             default best fair share on the cluster
#   --partition P           default short (12 h maximum); interactive also works
#   --time HH:MM:SS         default 02:00:00
#   --cpus N                default 8
#   --mem M / --mem-big M   default 64G / 128G (nt_*, gb_*, tcga_*, kiba, bindingdb_kd, qm9)
#   --max-parallel N        default 20
#   --pruning-mode M        default calibrated
#   --similarity-settings S KIRBY_SIMILARITY_SETTINGS for the noise runs; default unset ('all')
#   --controls              KIRBY_NOISE_CONTROLS=1 for the noise runs
#   --no-force              do not set KIRBY_FORCE_RERUN=1 on similarity/federated/opig
#   --no-pull               skip the git pull
#   --no-preflight          skip the 1-minute srun test
#   --dry-run               print every srun line and the count; launch nothing, move nothing
#
# INTERNAL MODES (called by the script itself, not by hand):
#   --dispatch <commands.tsv> <max-parallel>
#   --task-noise <dataset> <experiment> <condition or -> <output dir>
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
REPO="${KIRBY_REPO:-$(cd "$HERE/../../.." && pwd)}"
SLURM_DIR="$REPO/tests/slurm_scripts"
LOGD="${SC_LOGD:-$REPO/logs/server_checks}"
RESD="$REPO/tests/results/server_checks"

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: one noise check, on the compute node.
# ─────────────────────────────────────────────────────────────────────────────
if [ "${1:-}" = "--task-noise" ]; then
    DS="$2"; EXP="$3"; COND="$4"; P="$5"
    NAME="${SC_NAME:-noise_${DS}_${EXP}}"
    STEPS="$LOGD/${NAME}.steps"
    PRUNING_MODE="${PRUNING_MODE:-calibrated}"
    cd "$REPO" || exit 2
    export KIRBY_SKIP_INSTALLS=1
    . ./setup.sh
    export KIRBY_NO_COLD_SEARCH=1
    export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"
    unset KIRBY_NOISE_CACHE      # the cache must be on for the second-run check
    unset KIRBY_SMOKE_N          # a molecule cap, never wanted here
    mkdir -p "$P"
    : > "$STEPS"
    COND_ARG=(); CHECK_COND=()
    if [ "$COND" != "-" ]; then COND_ARG=(--condition "$COND"); CHECK_COND=(--condition "$COND"); fi

    echo "=== server check: noise ${DS} ${EXP} condition=${COND}"
    echo "  branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ($(git rev-parse --short HEAD 2>/dev/null))"
    echo "  node $(hostname)  cluster ${SLURM_CLUSTER_NAME:-?}  partition ${SLURM_JOB_PARTITION:-?}  job ${SLURM_JOB_ID:-?}"
    echo "  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}  KIRBY_LGBM_THREADS=${KIRBY_LGBM_THREADS}  OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
    echo "  KIRBY_SIMILARITY_SETTINGS=${KIRBY_SIMILARITY_SETTINGS:-unset (all)}  KIRBY_NOISE_CONTROLS=${KIRBY_NOISE_CONTROLS:-unset}"
    echo "  output ${P}   start $(date)"

    echo; echo "--- run 1 ---"
    KIRBY_NOISE_CACHE_REPORT="$P/cache_run1.json" \
      python -u tests/slurm_scripts/noise_prelim/run_one.py "$DS" "$EXP" \
        "${COND_ARG[@]}" --pruning-mode "$PRUNING_MODE" --output "$P/run1" < /dev/null
    RC1=$?
    echo "run1 $RC1 $(date -Iseconds)" >> "$STEPS"
    echo "--- run 1 exit ${RC1}  $(date)"
    if [ "$RC1" -ne 0 ]; then
        echo "run2 skipped_run1_failed" >> "$STEPS"
        echo "=== run 1 failed; run 2 and the check were not started"
        exit "$RC1"
    fi

    echo; echo "--- run 2 (identical settings, new output directory) ---"
    KIRBY_NOISE_CACHE_REPORT="$P/cache_run2.json" \
      python -u tests/slurm_scripts/noise_prelim/run_one.py "$DS" "$EXP" \
        "${COND_ARG[@]}" --pruning-mode "$PRUNING_MODE" --output "$P/run2" < /dev/null
    RC2=$?
    echo "run2 $RC2 $(date -Iseconds)" >> "$STEPS"
    echo "--- run 2 exit ${RC2}  $(date)"

    echo; echo "--- check_noise_run.py ---"
    python -u tests/experiments/noise/check_noise_run.py \
        --dataset "$DS" --experiment "$EXP" "${CHECK_COND[@]}" \
        --out1 "$P/run1" --out2 "$P/run2" \
        --report1 "$P/cache_run1.json" --report2 "$P/cache_run2.json" \
        --rc1 "$RC1" --rc2 "$RC2" --pruning-mode "$PRUNING_MODE" < /dev/null \
        | tee "$P/check.txt"
    RCC=${PIPESTATUS[0]}
    echo "check $RCC $(date -Iseconds)" >> "$STEPS"
    echo "=== done $(date)  run1=${RC1} run2=${RC2} check=${RCC}"
    [ "$RC2" -ne 0 ] && exit "$RC2"
    exit "$RCC"
fi

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: the dispatcher. Reads commands.tsv, keeps <= N alive.
# ─────────────────────────────────────────────────────────────────────────────
if [ "${1:-}" = "--dispatch" ]; then
    FILE="$2"; MAXP="$3"
    cd "$REPO" || exit 2
    echo "$$" > "$LOGD/_dispatcher.pid"
    echo "dispatcher $$ start $(date)  max ${MAXP} at once  $(wc -l < "$FILE") commands"
    N=0
    while IFS=$'\t' read -r NAME CAT DS EXP COND OUT CMD <&3; do
        [ -n "${NAME:-}" ] || continue
        while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do
            wait -n 2>/dev/null || sleep 10
        done
        date -Iseconds > "$LOGD/${NAME}.started"
        (
            bash -c "$CMD" > "$LOGD/${NAME}.log" 2>&1 < /dev/null
            echo $? > "$LOGD/${NAME}.rc"
        ) &
        N=$((N + 1))
        echo "$(date -Iseconds) started ${N}: ${NAME}"
        sleep 2   # stagger: each job sources setup.sh and loads a dataset
    done 3< "$FILE"
    echo "all ${N} started $(date); waiting for the last to end"
    wait
    echo "dispatcher done $(date)"
    rm -f "$LOGD/_dispatcher.pid"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
DATASETS="herg_ki herg"
FED_DATASETS=""
CATEGORIES="noise similarity federated opig"
ONLY_EXPS=""
COND_REG="outlier_p10"
COND_CLF=""
CLUSTERS="htc"
ACCOUNT=""
PARTITION="short"
WALL="02:00:00"
CPUS=8
MEM="64G"
MEM_BIG="128G"
MAXP=20
PRUNING="calibrated"
SIM_SETTINGS=""
CONTROLS=false
FORCE=true
PULL=true
PREFLIGHT=true
DRY=false
FEDERATED_OK="lhasa_herg caco2 logd"
BIG_MEM_RE='^(nt_|gb_|tcga_|kiba|bindingdb_kd|qm9)'

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)            DATASETS=$2; shift 2 ;;
        --fed-datasets)        FED_DATASETS=$2; shift 2 ;;
        --categories)          CATEGORIES=$2; shift 2 ;;
        --experiments)         ONLY_EXPS=$2; shift 2 ;;
        --cond-reg)            COND_REG=$2; shift 2 ;;
        --cond-clf)            COND_CLF=$2; shift 2 ;;
        --clusters)            CLUSTERS=$2; shift 2 ;;
        --account)             ACCOUNT=$2; shift 2 ;;
        --partition)           PARTITION=$2; shift 2 ;;
        --time)                WALL=$2; shift 2 ;;
        --cpus)                CPUS=$2; shift 2 ;;
        --mem)                 MEM=$2; shift 2 ;;
        --mem-big)             MEM_BIG=$2; shift 2 ;;
        --max-parallel)        MAXP=$2; shift 2 ;;
        --pruning-mode)        PRUNING=$2; shift 2 ;;
        --similarity-settings) SIM_SETTINGS=$2; shift 2 ;;
        --controls)            CONTROLS=true; shift ;;
        --no-force)            FORCE=false; shift ;;
        --no-pull)             PULL=false; shift ;;
        --no-preflight)        PREFLIGHT=false; shift ;;
        --dry-run)             DRY=true; shift ;;
        -h|--help)             sed -n '2,/^set -o pipefail/p' "$SELF" | sed '$d'; exit 0 ;;
        *) echo "Unknown argument: $1 (see --help)"; exit 1 ;;
    esac
done

cd "$REPO" || { echo "cannot cd to $REPO"; exit 2; }

# A dispatcher from an earlier launch that is still alive would race this one.
if [ -f "$LOGD/_dispatcher.pid" ] && kill -0 "$(cat "$LOGD/_dispatcher.pid")" 2>/dev/null; then
    echo "A dispatcher is still running (pid $(cat "$LOGD/_dispatcher.pid")), from an earlier launch."
    echo "Check it with server_checks_status.sh. Stop it with: kill $(cat "$LOGD/_dispatcher.pid")"
    [ "$DRY" = true ] || exit 1
fi

# ── pull once ──
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
if [ "$DRY" = true ]; then
    echo "dry run: no git pull"
elif [ "$PULL" = true ]; then
    echo "git pull --ff-only origin ${BRANCH}  (in ${REPO})"
    git pull --ff-only origin "$BRANCH" || { echo "git pull failed; fix it or pass --no-pull"; exit 1; }
fi
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

# ── setup.sh once, skip mode: the venv, the cache paths ──
export KIRBY_SKIP_INSTALLS=1
. ./setup.sh > /dev/null
echo "KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}   python $(command -v python)"

# ── account ──
if [ -z "$ACCOUNT" ] && [ -n "$CLUSTERS" ]; then
    PAIR=$(bash "$SLURM_DIR/where_to_submit.sh" --best-cluster-account "$CLUSTERS" 2>/dev/null)
    if [ -n "$PAIR" ]; then
        ACCOUNT=${PAIR##* }
        echo "account ${ACCOUNT}: best fair share on ${CLUSTERS} (where_to_submit.sh --best-cluster-account ${CLUSTERS})"
    else
        ACCOUNT="stat-cadd"
        echo "account stat-cadd: where_to_submit.sh gave no answer; using the 2026-09-24 best"
    fi
fi
CL_FLAG=""; [ -n "$CLUSTERS" ] && CL_FLAG="--clusters=${CLUSTERS}"
ACCT_FLAG=""; [ -n "$ACCOUNT" ] && [ "$ACCOUNT" != "none" ] && ACCT_FLAG="--account=${ACCOUNT}"

# ── federated datasets ──
if [[ " $CATEGORIES " == *" federated "* ]] && [ -z "$FED_DATASETS" ]; then
    for D in $DATASETS; do
        [[ " $FEDERATED_OK " == *" $D "* ]] && FED_DATASETS="$FED_DATASETS $D"
    done
    FED_DATASETS=$(echo "$FED_DATASETS" | xargs)
    if [ -z "$FED_DATASETS" ]; then
        FED_DATASETS="caco2"
        echo "federated: none of '${DATASETS}' has silo data (only ${FEDERATED_OK}); using caco2. Pass --fed-datasets to change."
    fi
fi

# ── list the experiments (python, repo environment) ──
PLAN=$(python - "$CATEGORIES" "$DATASETS" "$FED_DATASETS" "$COND_REG" "$COND_CLF" "$ONLY_EXPS" <<'PY'
import sys, importlib
sys.path[:0] = ['src', 'tests', 'tests/experiments']
cats, dss, fed, creg, cclf, only = sys.argv[1:7]
cats, dss, fed, only = cats.split(), dss.split(), fed.split(), set(only.split())
from experiments._shared import registry_lookup
mods = {'noise': 'experiments.noise.noise_helpers',
        'similarity': 'experiments.similarity.similarity_helpers',
        'federated': 'experiments.federated.federated_helpers',
        'complexity': 'experiments.complexity.complexity_helpers'}
task = {}
for d in set(dss) | set(fed):
    try:
        task[d] = getattr(registry_lookup(d), 'task_type', None)
    except Exception as e:
        sys.exit(f"unknown dataset {d!r}: {type(e).__name__}: {e}")
for cat in cats:
    if cat == 'opig':
        continue
    if cat not in mods:
        sys.exit(f"unknown category {cat!r}")
    H = importlib.import_module(mods[cat])
    exps = [e for e in H.SINGLE_DATASET_EXPERIMENTS if not only or e in only]
    for d in (fed if cat == 'federated' else dss):
        for e in exps:
            cond = '-'
            if cat == 'noise' and e in (H.HONOURS_NOISE_CONDITION | H.SWEEPS_ALL_CONDITIONS):
                conds = H.conditions_for_task({'task_type': task[d]})
                lim = getattr(H, 'CONDITION_LIMITED_TO', {}).get(e)
                want = cclf if task[d] == 'classification' else creg
                if not want:
                    want = conds[0]
                ok = [c for c in conds if lim is None or c in lim]
                if want in ok or not ok:
                    cond = want      # not ok at all: run_all writes the skip
                else:
                    cond = ok[0]
            print(f"{cat}\t{d}\t{e}\t{cond}")
PY
)
[ $? -eq 0 ] && [ -n "$PLAN" ] || [[ "$CATEGORIES" == "opig" ]] || { echo "Could not list the experiments (python error above)."; exit 1; }

# ── build the commands ──
q() { printf '%q' "$1"; }
CMDS_FILE_TMP=$(mktemp)
COUNT=0
add() {  # NAME CAT DS EXP COND OUT CMD
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$CMDS_FILE_TMP"
    COUNT=$((COUNT + 1))
}
SRUN_BASE="srun ${CL_FLAG} ${ACCT_FLAG} -p ${PARTITION} --cpus-per-task=${CPUS} --time=${WALL}"
ENV_COMMON="KIRBY_REPO=${REPO} KIRBY_NO_PULL=1 KIRBY_SKIP_INSTALLS=1 KIRBY_NO_COLD_SEARCH=1 PRUNING_MODE=${PRUNING}"
NOISE_ENV=""
[ -n "$SIM_SETTINGS" ] && NOISE_ENV="KIRBY_SIMILARITY_SETTINGS=$(q "$SIM_SETTINGS")"
[ "$CONTROLS" = true ] && NOISE_ENV="${NOISE_ENV} KIRBY_NOISE_CONTROLS=1"
FORCE_ENV=""; [ "$FORCE" = true ] && FORCE_ENV="KIRBY_FORCE_RERUN=1"

while IFS=$'\t' read -r CAT DS EXP COND; do
    [ -n "${CAT:-}" ] || continue
    M="$MEM"; [[ "$DS" =~ $BIG_MEM_RE ]] && M="$MEM_BIG"
    NAME="${CAT}_${DS}_${EXP}"
    JOB="sc_${CAT:0:3}_${DS}_${EXP}"
    if [ "$CAT" = "noise" ]; then
        OUT="${RESD}/noise/${DS}__${EXP}"
        CMD="cd $(q "$REPO") && env ${ENV_COMMON} ${NOISE_ENV} SC_LOGD=$(q "$LOGD") SC_NAME=${NAME} ${SRUN_BASE} --mem=${M} --job-name=${JOB} bash $(q "$SELF") --task-noise ${DS} ${EXP} ${COND} $(q "$OUT")"
    else
        OUT="${REPO}/tests/results/trial_${CAT}"
        CMD="cd $(q "$REPO") && env ${ENV_COMMON} ${FORCE_ENV} KIRBY_OUTPUT_DIR=tests/results/trial_${CAT} ${SRUN_BASE} --mem=${M} --job-name=${JOB} bash tests/slurm_scripts/submit_dataset.sh ${CAT} ${DS} ${EXP}"
    fi
    add "$NAME" "$CAT" "$DS" "$EXP" "$COND" "$OUT" "$(echo "$CMD" | tr -s ' ')"
done <<< "$PLAN"

if [[ " $CATEGORIES " == *" opig "* ]]; then
    while read -r ON OM _REST; do
        [ -n "${ON:-}" ] || continue
        [ -z "$ONLY_EXPS" ] || [[ " $ONLY_EXPS " == *" $ON "* ]] || continue
        NAME="opig_own_${ON}"
        OUT="${REPO}/tests/results/trial_opig/${ON}"
        CMD="cd $(q "$REPO") && env ${ENV_COMMON} ${FORCE_ENV} KIRBY_OUTPUT_DIR=tests/results/trial_opig KIRBY_EXP_LOG_DIR=$(q "$LOGD/_opig_detail") ${SRUN_BASE} --mem=${OM} --job-name=sc_opig_${ON} bash tests/slurm_scripts/runbook/opig_one.sh ${ON}"
        add "$NAME" opig own "$ON" - "$OUT" "$(echo "$CMD" | tr -s ' ')"
    done < <(bash "$HERE/opig_one.sh" --list)
fi

echo
echo "server checks   branch ${BRANCH} (${COMMIT})   repo ${REPO}"
echo "cluster ${CLUSTERS:-login node}   account ${ACCOUNT:-<SLURM default>}   partition ${PARTITION}   time ${WALL}   ${CPUS} CPUs   ${MEM} (${MEM_BIG} for big datasets)"
echo "datasets ${DATASETS}   federated datasets ${FED_DATASETS:--}   categories ${CATEGORIES}"
echo "noise condition: regression ${COND_REG}, classification ${COND_CLF:-first classification condition}   KIRBY_SIMILARITY_SETTINGS=${SIM_SETTINGS:-unset (all)}   controls ${CONTROLS}"
echo "at most ${MAXP} at once"
echo

if [ "$DRY" = true ]; then
    awk -F'\t' '{print $7}' "$CMDS_FILE_TMP"
    echo
    echo "${COUNT} srun commands (dry run: nothing launched, nothing moved)."
    awk -F'\t' '{print $2}' "$CMDS_FILE_TMP" | sort | uniq -c | sed 's/^/  /'
    rm -f "$CMDS_FILE_TMP"
    exit 0
fi

# ── preflight: does srun take these flags here? ──
if [ "$PREFLIGHT" = true ]; then
    echo "preflight: a 1-minute srun with the same cluster, account and partition (up to 90 s)..."
    PF=$(timeout 90 srun ${CL_FLAG} ${ACCT_FLAG} -p "${PARTITION}" --cpus-per-task=1 --mem=1G --time=00:01:00 --job-name=sc_preflight hostname 2>&1 < /dev/null)
    PRC=$?
    if [ "$PRC" -eq 0 ]; then
        echo "  ran on $(echo "$PF" | tail -n 1)"
    elif [ "$PRC" -eq 124 ]; then
        echo "  still queued after 90 s; the flags were accepted. Carrying on."
    else
        echo "  srun refused (exit ${PRC}):"
        echo "$PF" | tail -n 5 | sed 's/^/    /'
        if echo "$PF" | grep -qiE 'unrecognized option|invalid option|clusters'; then
            echo "  If it is --clusters: log in to htc-login.arc.ox.ac.uk and rerun with --clusters \"\"."
        fi
        echo "Nothing launched. Fix the flags, or pass --no-preflight to launch anyway."
        rm -f "$CMDS_FILE_TMP"
        exit 1
    fi
fi

# ── move the previous launch aside (never delete) ──
STAMP=$(date +%Y%m%d_%H%M%S)
if [ -d "$LOGD" ] && [ -n "$(ls -A "$LOGD" 2>/dev/null)" ]; then
    mkdir -p "$REPO/logs/server_checks_previous"
    mv "$LOGD" "$REPO/logs/server_checks_previous/$STAMP"
    echo "previous logs moved to logs/server_checks_previous/$STAMP"
fi
if [ -d "$RESD" ] && [ -n "$(ls -A "$RESD" 2>/dev/null)" ]; then
    mkdir -p "$REPO/tests/results/server_checks_previous"
    mv "$RESD" "$REPO/tests/results/server_checks_previous/$STAMP"
    echo "previous noise check outputs moved to tests/results/server_checks_previous/$STAMP"
fi
mkdir -p "$LOGD" "$RESD/noise" "$LOGD/_opig_detail"
mv "$CMDS_FILE_TMP" "$LOGD/commands.tsv"
{
    echo "launched $(date -Iseconds)  branch ${BRANCH} (${COMMIT})"
    echo "cluster ${CLUSTERS}  account ${ACCOUNT}  partition ${PARTITION}  time ${WALL}  max ${MAXP}"
    echo "datasets ${DATASETS}  federated ${FED_DATASETS}  categories ${CATEGORIES}"
} > "$LOGD/launch.txt"

SC_LOGD="$LOGD" setsid nohup bash "$SELF" --dispatch "$LOGD/commands.tsv" "$MAXP" \
    > "$LOGD/_dispatcher.log" 2>&1 < /dev/null &
sleep 1
echo "${COUNT} srun commands handed to the dispatcher (pid $(cat "$LOGD/_dispatcher.pid" 2>/dev/null || echo '?')), at most ${MAXP} at once."
echo "It keeps running after you log out."
echo "  status:  bash tests/slurm_scripts/runbook/server_checks_status.sh"
echo "  one log: less logs/server_checks/<category>_<dataset>_<experiment>.log"
