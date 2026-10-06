#!/bin/bash
#SBATCH --job-name=exp_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=medium
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

# SINCE 2026-10-06 jobs are submitted with tests/slurm_scripts/submit_workers.sh
# and tracked in the ledger (runbook/WORKERS.md). This file is kept because jobs
# submitted before then run a frozen copy of it; do not submit new ones with it.
# Run experiments for one (category, dataset) pair — or multiple categories
# bundled into a single SLURM job for one dataset.
#
# Usage:
#   sbatch submit_dataset.sh complexity herg
#   sbatch submit_dataset.sh noise flab_thermo_tresanco
#   sbatch submit_dataset.sh similarity logd "s01 s02 s05"
#   # Bundle: run complexity + similarity + noise sequentially in ONE job:
#   sbatch submit_dataset.sh all humanness_therapeutic
#   sbatch submit_dataset.sh complexity,similarity,noise humanness_therapeutic
#   # Also runs without SLURM, e.g. inside an srun or on a laptop checkout:
#   KIRBY_REPO=$PWD KIRBY_NO_PULL=1 bash tests/slurm_scripts/submit_dataset.sh noise caco2 n07
#
# Pruning mode (default: aggressive_pruning):
#   sbatch --export=ALL,PRUNING_MODE=calibrated submit_dataset.sh complexity herg
#
# ── RESOURCES. CHANGED 2026-09-25. ──────────────────────────────────────────
# Defaults are now 8 CPUs, 64G, 47:59:00, partition medium. They were 32 CPUs,
# 128G, 7 days, long. The old shape fits neither cluster's medium partition
# (48 h maximum) nor htc's 16- and 20-core nodes (ARC docs, job-scheduling.html,
# quoted in noise_prelim/README.md). 64G is 8 CPUs at ARC's 8000 MB default per
# CPU, so the request ties up no more memory than its cores carry.
# Callers override on the sbatch line, which wins over these lines:
#   sbatch --time=11:59:00 --partition=short --mem=128G submit_dataset.sh noise qm9
# ⛔ submit_everything.sh passes no --mem or --time for its x_* jobs, so they
# now get 64G where they got 128G. queue_noise_conditions.py and
# runbook/stage3_full.sh give nt_*, gb_*, tcga_*, kiba, bindingdb_kd and qm9
# 128G themselves.
#
# ── ACCOUNT. CHANGED 2026-09-25. ────────────────────────────────────────────
# There is no #SBATCH --account line any more. It said stat-ecr, which on
# 2026-09-24 had the LOWEST fair share of the four (cluster, account) pairs:
# htc stat-cadd 0.0491, htc stat-ecr 0.0360, arc stat-cadd 0.0326, arc stat-ecr
# 0.0195 (extend_precompute_chain.sh:48-56). The account is chosen at submit
# time, with the cluster, from:
#   read CL ACCT < <(bash tests/slurm_scripts/where_to_submit.sh --best-cluster-account)
#   sbatch --clusters="$CL" --account="$ACCT" submit_dataset.sh noise caco2
# Submitted with no --account, SLURM bills the user's default association.
# The /data/stat-ecr paths below are a filesystem; they do not bill anything.
#
# ── ONE PYTHON PROCESS PER EXPERIMENT. ADDED 2026-09-25. ────────────────────
# Each experiment runs in its OWN python process, one after another. The order
# is quickest first for complexity, similarity and federated (2026-10-03; see
# resolve_experiments below) and the list order for noise. Before this, one
# run_dataset.py process ran them all: a segfault (exit 139, LightGBM and torch
# each bringing an OpenMP runtime) or an out-of-memory kill in one experiment
# killed every experiment after it in the same job. Now it kills one.
#
# The cost: every process loads the dataset and reads the stored searches
# again (run_all does both before its first experiment). Those reads are cache
# hits; their time per process has not been measured.
#
# Order matters where one experiment reads another's result file: noise n06
# reads the rankings n11 and n13 wrote in the same output directory, complexity
# c08 reads c02's and c09 reads c02's and c06's. resolve_experiments keeps each
# of those after its inputs.
#
# A RESUBMITTED JOB (2026-10-03) skips, without starting python, every
# experiment whose latest row in <output>/_job_summaries/*.tsv says ok. See
# "FINISHED IN AN EARLIER JOB" below.
#
# Every experiment's exit code goes to <output>/_job_summaries/<jobid>.tsv, one
# row each. The job exits non-zero if any experiment failed. An experiment that
# exits 0 but wrote a file under <output>/_errors/ during its run is recorded as
# failed too: run_dataset.py exited 0 on a caught exception until it was changed
# to exit 1 (being done in parallel on 2026-09-25), and this catches both.
# While an experiment runs, <output>/_job_summaries/<jobid>.running names it.
#
# ── ENVIRONMENT ─────────────────────────────────────────────────────────────
#   NOISE_CONDITION      which injected noise the noise experiments use (below)
#   PRUNING_MODE         default aggressive_pruning; every 2026-09 caller passes
#                        calibrated
#   KIRBY_NO_COLD_SEARCH 1 = an experiment asking for a search the precompute
#                        never stored raises instead of searching
#   KIRBY_FORCE_RERUN    1 = --force-rerun (rebuild result JSONs)
#   KIRBY_OUTPUT_DIR     run_dataset.py --output. Default: the category's own
#                        tests/results/<category>. With several categories,
#                        each gets <KIRBY_OUTPUT_DIR>/<category>.
#   KIRBY_EXP_LOG_DIR    if set, each experiment's output is also written to
#                        <dir>/<category>_<dataset>_<experiment>[__<condition>].log
#   EXP_TIMEOUT          per-experiment wall, coreutils `timeout` syntax
#                        (e.g. 10h). Unset = none. Exit 124 = hit it.
#   KIRBY_NO_PULL        1 = skip the git pull (arrays: pull once by hand first)
#   KIRBY_SKIP_INSTALLS  default 1 (below)
#   KIRBY_LGBM_THREADS   default 1 (below)
#   KIRBY_REPO           default /data/stat-ecr/scat9264/KIRBy

CATEGORY=${1:?Usage: sbatch submit_dataset.sh CATEGORY DATASET [EXPERIMENTS]}
DATASET=${2:?Usage: sbatch submit_dataset.sh CATEGORY DATASET [EXPERIMENTS]}
EXPERIMENTS=${3:-}
# Pruning is off in every per-fold search since 2026-10-05; PRUNING_MODE is no
# longer passed to run_dataset.py. Kept so an old --export line still parses.
PRUNING_MODE="${PRUNING_MODE:-}"

# WHICH NOISE IS INJECTED. An environment variable, not a fourth positional
# argument: every caller of this script passes three, and PRUNING_MODE and
# KIRBY_NO_COLD_SEARCH already arrive this way.
#
# ADDED 2026-09-23. run_dataset.py has taken --noise-condition since 2026-09-12
# -- it forwards the name to the experiment AND writes it into the output
# filename, so two conditions on one dataset land on two files. Nothing ever
# passed it. This script's two run lines carried a pruning mode, a force flag
# and an experiment list and no condition, so every noise job ever submitted ran
# each experiment's own default: gaussian for n09, n11 and n12, the outlier
# condition for n07. The condition axis existed in the code and had never
# reached the queue.
#
# Empty leaves that default in place, so an existing submission is unchanged.
NOISE_CONDITION="${NOISE_CONDITION:-}"

# Expand 'all' to the bundled category set. Comma-separated CATEGORY arg
# (e.g. 'complexity,similarity') is split below.
if [ "$CATEGORY" = "all" ]; then
    CATEGORY="complexity,similarity,noise"
fi
# Split CATEGORY on commas into an array — handles "complexity" (single)
# and "complexity,similarity,noise" (bundled) uniformly.
IFS=',' read -ra _CATEGORIES <<< "$CATEGORY"

# START DELAY (owner, 2026-10-03). Complexity jobs start at once and build the
# per-fold searches; every other category waits, so when it asks for the same
# search it reads what complexity has already stored instead of fitting it
# again alongside. Similarity waits 5 min, noise and federated 10 min. Only
# under SLURM; a local run starts at once. KIRBY_START_DELAY (seconds)
# overrides. KIRBY_SEARCH_REVERSE=1 makes a waiting job's search work through
# its configurations from the end of the list, so two jobs on one search
# meet in the middle rather than fitting the same configuration together
# (find_best_hybrid re-reads the store every 25 configurations).
if [ -n "${SLURM_JOB_ID:-}" ]; then
    case "${_CATEGORIES[0]}" in
        complexity) _DELAY_DEFAULT=0 ;;
        similarity) _DELAY_DEFAULT=300 ;;
        *)          _DELAY_DEFAULT=600 ;;
    esac
    _DELAY="${KIRBY_START_DELAY:-$_DELAY_DEFAULT}"
    if [ "$_DELAY" -gt 0 ] 2>/dev/null; then
        export KIRBY_SEARCH_REVERSE="${KIRBY_SEARCH_REVERSE:-1}"
        echo "Start delay: ${_DELAY}s (category ${_CATEGORIES[0]}), so complexity's searches are stored first"
        sleep "$_DELAY"
    fi
fi

REPO="${KIRBY_REPO:-/data/stat-ecr/scat9264/KIRBy}"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 2; }

# Pull before recording branch/commit. Without this, queued jobs run whatever
# was on disk at job-start time — which silently diverges from what the user
# just pushed. --ff-only so a non-fast-forward state fails loudly instead of
# merging surprise commits.
#
# ⛔ IT PULLS THE BRANCH THIS CHECKOUT IS ON, NOT 'main'. It said `origin main`
# until 2026-09-16, which on a checkout of any other branch either refused or
# fast-forwarded the wrong history; the || branch then printed a warning and the
# job ran on-disk code with nothing in the log naming the branch it wanted. The
# 2026-09 work is all on similarity-metrics-study.
#
# ⛔ AND IT IS ONE CHECKOUT SHARED BY EVERY JOB. Fifteen jobs starting together
# each run this. --ff-only on an already-current checkout is a fetch and no
# write, so it is cheap and safe, but do not add anything here that writes to
# the working tree. Pull once by hand before submitting and this is a no-op.
#
# KIRBY_NO_PULL=1 skips it (added 2026-09-25). The runbook's array jobs set it:
# hundreds of tasks starting in the same minute would each fetch into one .git.
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [ "${KIRBY_NO_PULL:-0}" = "1" ]; then
    echo "KIRBY_NO_PULL=1: not pulling; running the on-disk checkout"
elif [ "${BRANCH}" = "unknown" ] || [ "${BRANCH}" = "HEAD" ]; then
    echo "WARNING: detached HEAD or not a git checkout; not pulling"
else
    git pull --ff-only origin "${BRANCH}" 2>&1 \
        || echo "WARNING: git pull --ff-only origin ${BRANCH} failed; running with on-disk code"
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# KIRBY_SKIP_INSTALLS=1 BY DEFAULT (2026-09-25). Full-mode setup.sh runs
# `pip install -e .` into the SHARED venv on /data/stat-cadd (setup.sh:138),
# which re-points that venv at whichever checkout the job ran from, for every
# job at once; and on htc it builds HMMER into the same venv (setup.sh:212-240),
# on a filesystem that stood at 99.9% of quota. Skip mode activates the venv,
# sets the cache paths, and returns (setup.sh:101-112). Set
# KIRBY_SKIP_INSTALLS=0 to get the install checks back.
export KIRBY_SKIP_INSTALLS="${KIRBY_SKIP_INSTALLS:-1}"
. setup.sh

# LIGHTGBM THREADS: 1 BY DEFAULT (2026-09-25), for every category.
# Before today this script left it unset, so hybrid.py's LightGBM fits took -1
# (every core; hybrid.py:87) and noisy_labels/core.py's took 1 (core.py:95).
# One value for both, for three reasons:
#   1. noise_cache.py puts it in every key (lgbm_n_jobs, oof_lgb_threads;
#      noise_cache.py:22-25). A job that uses another value misses every entry
#      the prelim and full noise runs wrote, and they must match each other.
#   2. More than one thread after torch has started its own OpenMP pool kills
#      the process with SIGSEGV, exit 139 (hybrid.py:57-73). deeplift, the dnn
#      model and the dnn evaluator all import torch.
#   3. Measured 2026-09-19, one surrogate fit on 300 molecules by 20 features:
#      4.3 s at 1 thread, 2010.3 s at every core (core.py:78-82).
# What this changes: hybrid.py's in-experiment LightGBM fits now run at 1
# thread where they ran at -1. Whether that moves any score is not measured;
# no LightGBM call in src/kirby/hybrid.py sets `deterministic`. The precompute
# (run_precompute_dataset.sh) does not set it and is unaffected.
export KIRBY_LGBM_THREADS="${KIRBY_LGBM_THREADS:-1}"

# HF datasets offline mode. Was set to 1 unconditionally to avoid NFS
# FileLock contention on the 2026-05-17 wave; but for NOT-DONE datasets
# the HF cache may not yet contain the dataset, leading to silent
# load_dataset failures. Default to 0 (online) — the FileLock contention
# was a different bug, and we now pre-stagger job starts with sleep
# random. Override to 1 only when explicitly running on the same dataset
# in many parallel jobs (e.g. precompute fan-out).
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}

# run_dataset.py RESUMES by default: it skips any experiment whose result JSON
# already exists. That is right for a crashed job, but wrong when the JSON needs
# REBUILDING from unchanged cached scores (e.g. adding the fold_reselection block
# to files written before it existed) — the run would skip every dataset and
# emit nothing. Set KIRBY_FORCE_RERUN=1 to rebuild the JSONs instead.
# This does NOT re-run the hybrid search: the per-config scores are served from
# result_db, so it is a cache-hit pass plus any baselines that must be recomputed.
FORCE_FLAG=""
if [ "${KIRBY_FORCE_RERUN:-0}" = "1" ]; then
    FORCE_FLAG="--force-rerun"
    echo "KIRBY_FORCE_RERUN=1 -> passing --force-rerun (rebuilding result JSONs)"
fi

# The refusal switch. Inherited from the submit environment
# (--export=ALL,KIRBY_NO_COLD_SEARCH=1), and printed so a log says which arm it
# is. When set, an experiment that asks for a search the precompute never wrote
# raises in minutes instead of rebuilding 8,160 configurations x 5 folds inside
# this job's wall. It only ever fires on the tiers the precompute owns --
# see _refuse_cold_search in src/kirby/find_best_hybrid.py.
if [ -n "${KIRBY_NO_COLD_SEARCH:-}" ]; then
    export KIRBY_NO_COLD_SEARCH
fi

# Job id for the summary file. An array task is <array job>_<task>.
JOB_TAG="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    JOB_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi

echo "=== ${CATEGORY} experiments: ${DATASET} ==="
echo "Branch: ${BRANCH} (${COMMIT})"
echo "Cold-search refusal: ${KIRBY_NO_COLD_SEARCH:-off}"
echo "Job ID: ${SLURM_JOB_ID:-none}   summary tag: ${JOB_TAG}"
echo "Node:   $(hostname)   cluster ${SLURM_CLUSTER_NAME:-?}   partition ${SLURM_JOB_PARTITION:-?}"
echo "Pruning mode: ${PRUNING_MODE}"
echo "Noise condition: ${NOISE_CONDITION:-per-experiment default}"
echo "KIRBY_LGBM_THREADS=${KIRBY_LGBM_THREADS}  OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}  KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-unset}"
echo "Per-experiment wall: ${EXP_TIMEOUT:-none}"
echo "Start:  $(date)"

# Which experiments to run, one per line: "run <id>" or "skip <id> <reason>".
#
# No list given: the category's SINGLE_DATASET_EXPERIMENTS. For noise with a
# condition, narrowed exactly as noise_helpers.run_all narrows its default
# list: only HONOURS_NOISE_CONDITION | SWEEPS_ALL_CONDITIONS; the rest name
# their own condition and run in the job with no condition.
#
# One addition to run_all's narrowing: n21 under a condition outside
# DETECTION_IS_DEFINED is skipped with its reason. run_all would call it and n21
# raises at once (n21_neighbour_distance.py:671-677): average precision is its
# headline and under those conditions every training label moves. In one
# process run_all caught that and carried on; as its own process it would mark
# every such job failed for a refusal that is by design.
#
# c12 ON tcga_* (added 2026-10-03). complexity_helpers.run_all adds c12 after
# c11 on a tcga_ dataset when it builds its own default list. This script names
# every experiment itself, so until today c12 never ran on any tcga_ dataset.
#
# QUICKEST FIRST (added 2026-10-03, owner's request). complexity, similarity and
# federated run in the order of RUN_ORDER below, shortest expected time first,
# so that a job cut off by its wall has finished as many experiments as
# possible. The times behind each order, and which are measured and which are
# guessed, are in runbook/SUBMIT_TONIGHT.md. Two rules hold over that order:
#   - AFTER: an experiment that reads another's result file runs after it.
#     c08 reads c02's file; c09 reads c02's and c06's (complexity_helpers.py,
#     the c08/c09 kwargs in run_all). noise n06 reads n11's and n13's.
#   - An experiment in SINGLE_DATASET_EXPERIMENTS that RUN_ORDER does not name
#     runs after the named ones, in its list order. Nothing is dropped.
# noise keeps its list order (its order is the owner's separate call).
# KIRBY_KEEP_LIST_ORDER=1 restores the plain list order everywhere.
#
# A list given on the command line runs as given, as run_all does with an
# explicit list: an experiment that cannot honour the condition raises there
# and is recorded as failed.
resolve_experiments() {
    python - "$1" "$NOISE_CONDITION" "$DATASET" <<'PY'
import os, sys, importlib
sys.path[:0] = ['src', 'tests', 'tests/experiments']
cat, cond, dataset = sys.argv[1], sys.argv[2], sys.argv[3]
mods = {'complexity': 'experiments.complexity.complexity_helpers',
        'similarity': 'experiments.similarity.similarity_helpers',
        'noise': 'experiments.noise.noise_helpers',
        'federated': 'experiments.federated.federated_helpers'}
if cat not in mods:
    sys.exit(f"unknown category {cat!r}; run_dataset.py knows {sorted(mods)}")
H = importlib.import_module(mods[cat])

# Shortest expected time first. Sources in runbook/SUBMIT_TONIGHT.md.
RUN_ORDER = {
    'complexity': ['c06', 'c19', 'c22', 'c07', 'c18', 'c01', 'c02', 'c08',
                   'c15', 'c04', 'c13', 'c11', 'c12', 'c16', 'c09', 'c24',
                   'c05', 'c23', 'c33', 'c17', 'c21'],
    'similarity': ['s04', 's03', 's01', 's12', 's06', 's13', 's02', 's10',
                   's05', 's08', 's11'],
    'federated':  ['fed02', 'fed03', 'fed04', 'fed05', 'fed01'],
}
AFTER = {'c08': ['c02'], 'c09': ['c02', 'c06'], 'n06': ['n11', 'n13']}

exps = list(H.SINGLE_DATASET_EXPERIMENTS)
if cat == 'complexity' and dataset.startswith('tcga_') and 'c12' not in exps:
    exps.insert(exps.index('c11') + 1 if 'c11' in exps else len(exps), 'c12')

if cat in RUN_ORDER and os.environ.get('KIRBY_KEEP_LIST_ORDER') != '1':
    rank = {e: i for i, e in enumerate(RUN_ORDER[cat])}
    exps = sorted(exps, key=lambda e: (rank.get(e, len(rank)), exps.index(e)))

# Enforce AFTER: move an experiment to just after the last of its inputs.
changed = True
while changed:
    changed = False
    for e, deps in AFTER.items():
        if e not in exps:
            continue
        last = max((exps.index(d) for d in deps if d in exps), default=-1)
        if exps.index(e) < last:
            exps.remove(e)
            exps.insert(last, e)   # index `last` is now the slot after the input
            changed = True

for e in exps:
    if cat == 'noise' and cond:
        if e not in (H.HONOURS_NOISE_CONDITION | H.SWEEPS_ALL_CONDITIONS):
            print(f"skip {e} names its own condition; runs in the job with no condition")
            continue
        if e == 'n21' and cond not in H.DETECTION_IS_DEFINED:
            print(f"skip {e} detection undefined under {cond} (n21_neighbour_distance.py:671)")
            continue
    print(f"run {e}")
PY
}

TSV_HEADER="job	category	dataset	experiment	condition	exit_code	status	seconds	started	ended	log"

# Loop the requested categories sequentially within this single SLURM job, and
# inside each, the experiments one process at a time. Tracks the highest exit
# code so SLURM reports FAILED on the most-diagnostic signal (e.g. SIGBUS=135 >
# generic Python err=1). Keeps running after a failure so one crash doesn't skip
# the rest.
WORST_RC=0
N_FAIL=0
for CAT in "${_CATEGORIES[@]}"; do
    echo ""
    echo "--- Category: ${CAT} ---"

    if [ -n "${KIRBY_OUTPUT_DIR:-}" ]; then
        if [ "${#_CATEGORIES[@]}" -gt 1 ]; then OUT="${KIRBY_OUTPUT_DIR}/${CAT}"; else OUT="${KIRBY_OUTPUT_DIR}"; fi
    else
        OUT="tests/results/${CAT}"
    fi
    mkdir -p "${OUT}/_job_summaries"
    SUMMARY="${OUT}/_job_summaries/${JOB_TAG}.tsv"
    RUNNING="${OUT}/_job_summaries/${JOB_TAG}.running"
    [ -f "$SUMMARY" ] || printf '%s\n' "$TSV_HEADER" > "$SUMMARY"

    # Only the noise category takes a condition. run_dataset.py prints a
    # warning and ignores it for the others, which in a bundled
    # 'complexity,similarity,noise' job would put that warning in the log once
    # per experiment for something that is working as intended.
    COND_ARGS=()
    COND_TAG=""
    if [ "${CAT}" = "noise" ] && [ -n "${NOISE_CONDITION}" ]; then
        COND_ARGS=(--noise-condition "${NOISE_CONDITION}")
        COND_TAG="${NOISE_CONDITION}"
        echo "    noise condition: ${NOISE_CONDITION}"
    fi

    if [ -n "$EXPERIMENTS" ]; then
        PLAN=$(for e in $EXPERIMENTS; do echo "run $e"; done)
    else
        PLAN=$(resolve_experiments "$CAT")
        if [ $? -ne 0 ] || [ -z "$PLAN" ]; then
            echo "ERROR: could not list ${CAT}'s experiments (see the python error above)"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_TAG" "$CAT" "$DATASET" \
                "-" "${COND_TAG:--}" 2 "list_failed" 0 "$(date -Iseconds)" "$(date -Iseconds)" "-" >> "$SUMMARY"
            [ 2 -gt ${WORST_RC} ] && WORST_RC=2
            N_FAIL=$((N_FAIL + 1))
            continue
        fi
    fi
    echo "    plan:"
    printf '%s\n' "$PLAN" | sed 's/^/      /'

    # The plan is read on fd 3 and every experiment gets /dev/null on stdin,
    # so nothing an experiment reads can swallow the rest of the plan.
    while read -r ACTION EXP REASON <&3; do
        [ -n "${ACTION:-}" ] || continue
        if [ "$ACTION" = "skip" ]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_TAG" "$CAT" "$DATASET" \
                "$EXP" "${COND_TAG:--}" "-" "not_applicable" 0 "$(date -Iseconds)" "$(date -Iseconds)" "$REASON" >> "$SUMMARY"
            continue
        fi

        # FINISHED IN AN EARLIER JOB: skip without starting python. Added
        # 2026-10-03. A resubmitted job reads every summary in
        # <output>/_job_summaries/ and skips an experiment whose LATEST row for
        # this category, dataset and condition says ok (or done_earlier).
        # Before, each finished experiment still cost a python start, a data
        # load and a read of the stored searches before run_all's own resume
        # check skipped it. run_all's check stays as the second line: it also
        # compares the input fingerprint.
        # This trusts the output folder. A folder written by older code is not
        # safe to resume into: give each new submission a new KIRBY_OUTPUT_DIR.
        # KIRBY_FORCE_RERUN=1 turns this skip off.
        if [ "${KIRBY_FORCE_RERUN:-0}" != "1" ]; then
            PREV=$(cat "${OUT}"/_job_summaries/*.tsv 2>/dev/null \
                   | awk -F'\t' -v c="$CAT" -v d="$DATASET" -v e="$EXP" -v k="${COND_TAG:--}" \
                       '$1!="job" && $2==c && $3==d && $4==e && $5==k' \
                   | sort -t$'\t' -k10,10 | tail -n 1 | awk -F'\t' '{print $1, $7}')
            case "${PREV#* }" in
                ok|done_earlier)
                    echo ">>> ${CAT} ${DATASET} ${EXP} ${COND_TAG:+(${COND_TAG}) }finished in job ${PREV%% *}; skipped"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_TAG" "$CAT" "$DATASET" \
                        "$EXP" "${COND_TAG:--}" 0 "done_earlier" 0 "$(date -Iseconds)" "$(date -Iseconds)" "job ${PREV%% *}" >> "$SUMMARY"
                    continue ;;
            esac
        fi

        LOGF=""
        if [ -n "${KIRBY_EXP_LOG_DIR:-}" ]; then
            mkdir -p "$KIRBY_EXP_LOG_DIR"
            LOGF="${KIRBY_EXP_LOG_DIR}/${CAT}_${DATASET}_${EXP}${COND_TAG:+__${COND_TAG}}.log"
        fi
        # -u: unbuffered stdout. Without this Python's ~8KB block buffer hides
        # progress from slurm-<jid>.out until the buffer fills (can be hours on
        # cache-hit jobs), making monitor_slurm.sh blind to running jobs.
        CMD=(python -u tests/experiments/run_dataset.py "${CAT}" "${DATASET}"
             --experiments "${EXP}")
        [ -n "$FORCE_FLAG" ] && CMD+=("$FORCE_FLAG")
        [ "${#COND_ARGS[@]}" -gt 0 ] && CMD+=("${COND_ARGS[@]}")
        [ -n "${KIRBY_OUTPUT_DIR:-}" ] && CMD+=(--output "${OUT}")
        if [ -n "${EXP_TIMEOUT:-}" ]; then
            # TERM at the wall, KILL two minutes later if it has not gone.
            # coreutils timeout is on every ARC node; a mac has it as gtimeout
            # only with coreutils installed.
            _TO=$(command -v timeout || command -v gtimeout || true)
            if [ -n "$_TO" ]; then
                CMD=("$_TO" --signal=TERM --kill-after=120 "${EXP_TIMEOUT}" "${CMD[@]}")
            else
                echo "WARNING: EXP_TIMEOUT=${EXP_TIMEOUT} set but no timeout command on this machine; running without a per-experiment wall"
            fi
        fi

        MARK=$(mktemp "${OUT}/_job_summaries/.mark.XXXXXX")
        printf '%s\t%s\t%s\t%s\t%s\n' "$EXP" "${COND_TAG:--}" "$(date -Iseconds)" "${LOGF:--}" "$(hostname)" > "$RUNNING"
        T0=$(date +%s); STARTED=$(date -Iseconds)
        echo ""
        echo ">>> ${CAT} ${DATASET} ${EXP} ${COND_TAG:+(${COND_TAG}) }start ${STARTED}${LOGF:+  log ${LOGF}}"
        if [ -n "$LOGF" ]; then
            "${CMD[@]}" < /dev/null 2>&1 | tee "$LOGF"
            RC=${PIPESTATUS[0]}
        else
            "${CMD[@]}" < /dev/null
            RC=$?
        fi
        SECS=$(( $(date +%s) - T0 ))

        case "$RC" in
            0)   STATUS="ok" ;;
            124) STATUS="timeout" ;;
            137) STATUS="killed_9_oom_or_wall" ;;
            139) STATUS="segfault" ;;
            135) STATUS="bus_error" ;;
            *)   STATUS="failed" ;;
        esac
        # An experiment whose exception was caught by run_all exits 0 on a
        # run_dataset.py that predates the exit-1 change; the new one leaves a
        # file in <output>/_errors/. Since 2026-10-03 run_dataset.py names that
        # file after the one experiment it ran,
        # <dataset>[__<condition>]__<experiment>.json, so only that exact name
        # counts, and only when it is newer than this experiment's start. The
        # older match (dataset and experiment anywhere in the name or text)
        # could blame herg for a herg_ki file in the same folder. "Not older
        # than the start" rather than "newer": -nt compares whole seconds on
        # some systems, and a fast failure lands in the same second.
        ERRF="${OUT}/_errors/${DATASET}${COND_TAG:+__${COND_TAG}}__${EXP}.json"
        if [ "$RC" -eq 0 ] && [ -f "$ERRF" ] && ! [ "$MARK" -nt "$ERRF" ]; then
            STATUS="error_file:$(basename "$ERRF")"; RC=1
        fi
        rm -f "$MARK" "$RUNNING"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$JOB_TAG" "$CAT" "$DATASET" \
            "$EXP" "${COND_TAG:--}" "$RC" "$STATUS" "$SECS" "$STARTED" "$(date -Iseconds)" "${LOGF:--}" >> "$SUMMARY"
        echo "<<< ${CAT} ${DATASET} ${EXP} exit=${RC} ${STATUS} ${SECS}s"
        if [ "$RC" -ne 0 ]; then
            N_FAIL=$((N_FAIL + 1))
            [ "$RC" -gt "$WORST_RC" ] && WORST_RC=$RC
        fi
    done 3<<< "$PLAN"
    echo "--- ${CAT} done; summary ${SUMMARY} ---"
done

echo ""
echo "Done:   $(date)  (${N_FAIL} experiment(s) failed, worst exit=${WORST_RC})"
# Propagate so SLURM reports FAILED, not COMPLETED 0:0 (the masking bug
# from the Wave 1 bus-error batch).
exit ${WORST_RC}
