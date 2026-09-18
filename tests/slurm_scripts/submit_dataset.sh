#!/bin/bash
#SBATCH --job-name=exp_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --account=stat-ecr
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

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
#
# Pruning mode (default: aggressive_pruning):
#   sbatch --export=PRUNING_MODE=default_pruning submit_dataset.sh complexity herg
#
# SLURM defaults: long, 7d, 128G, 32 cpus, account stat-ecr.
# account=stat-ecr: live fair-share ~2x stat-cadd (2026-07-07); data/cache also
# live on /data/stat-ecr. A CLI --account (e.g. from submit_priority.sh) overrides
# this directive. Override any of these at sbatch time if needed:
#   sbatch --time=2-00:00:00 --mem=64G --partition=medium submit_dataset.sh complexity caco2

CATEGORY=${1:?Usage: sbatch submit_dataset.sh CATEGORY DATASET [EXPERIMENTS]}
DATASET=${2:?Usage: sbatch submit_dataset.sh CATEGORY DATASET [EXPERIMENTS]}
EXPERIMENTS=${3:-}
PRUNING_MODE="${PRUNING_MODE:-aggressive_pruning}"

# Expand 'all' to the bundled category set. Comma-separated CATEGORY arg
# (e.g. 'complexity,similarity') is split below.
if [ "$CATEGORY" = "all" ]; then
    CATEGORY="complexity,similarity,noise"
fi
# Split CATEGORY on commas into an array — handles "complexity" (single)
# and "complexity,similarity,noise" (bundled) uniformly.
IFS=',' read -ra _CATEGORIES <<< "$CATEGORY"

cd /data/stat-ecr/scat9264/KIRBy

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
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [ "${BRANCH}" = "unknown" ] || [ "${BRANCH}" = "HEAD" ]; then
    echo "WARNING: detached HEAD or not a git checkout; not pulling"
else
    git pull --ff-only origin "${BRANCH}" 2>&1 \
        || echo "WARNING: git pull --ff-only origin ${BRANCH} failed; running with on-disk code"
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

. setup.sh

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

echo "=== ${CATEGORY} experiments: ${DATASET} ==="
echo "Branch: ${BRANCH} (${COMMIT})"
echo "Cold-search refusal: ${KIRBY_NO_COLD_SEARCH:-off}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Pruning mode: ${PRUNING_MODE}"
echo "Start:  $(date)"

# Loop the requested categories sequentially within this single SLURM job.
# Tracks the highest exit code across categories so SLURM reports FAILED
# on the most-diagnostic signal (e.g. SIGBUS=135 > generic Python err=1).
# Keeps running after a failure so a noise crash doesn't skip similarity.
WORST_RC=0
for CAT in "${_CATEGORIES[@]}"; do
    echo ""
    echo "--- Category: ${CAT} ---"
    # -u: unbuffered stdout. Without this Python's ~8KB block buffer hides
    # progress from slurm-<jid>.out until the buffer fills (can be hours on
    # cache-hit jobs), making monitor_slurm.sh blind to running jobs.
    if [ -n "$EXPERIMENTS" ]; then
        python -u tests/experiments/run_dataset.py ${CAT} ${DATASET} \
            --pruning-mode "${PRUNING_MODE}" \
            ${FORCE_FLAG} \
            --experiments ${EXPERIMENTS}
    else
        python -u tests/experiments/run_dataset.py ${CAT} ${DATASET} \
            --pruning-mode "${PRUNING_MODE}" \
            ${FORCE_FLAG}
    fi
    RC=$?
    # Track the WORST (highest) exit code so signal-killed children (RC 128+sig)
    # surface in sacct instead of being masked by a later generic Python failure.
    if [ ${RC} -gt ${WORST_RC} ]; then
        WORST_RC=${RC}
    fi
    echo "--- ${CAT} done (exit=${RC}) ---"
done

echo ""
echo "Done:   $(date)  (worst exit=${WORST_RC})"
# Propagate so SLURM reports FAILED, not COMPLETED 0:0 (the masking bug
# from the Wave 1 bus-error batch).
exit ${WORST_RC}
