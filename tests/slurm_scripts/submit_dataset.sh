#!/bin/bash
#SBATCH --job-name=exp_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
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
# SLURM defaults: long, 7d, 128G, 8 cpus.
# Override at sbatch time if needed:
#   sbatch --time=2-00:00:00 --mem=64G submit_dataset.sh complexity caco2

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

# Pull the latest main before recording branch/commit. Without this, queued
# jobs run whatever was on disk at job-start time — which silently diverges
# from what the user just pushed. Use --ff-only so a non-fast-forward state
# fails loudly instead of merging surprise commits.
git pull --ff-only origin main 2>&1 || echo "WARNING: git pull failed; running with on-disk code"

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

echo "=== ${CATEGORY} experiments: ${DATASET} ==="
echo "Branch: ${BRANCH} (${COMMIT})"
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
    if [ -n "$EXPERIMENTS" ]; then
        python tests/experiments/run_dataset.py ${CAT} ${DATASET} \
            --pruning-mode "${PRUNING_MODE}" \
            --experiments ${EXPERIMENTS}
    else
        python tests/experiments/run_dataset.py ${CAT} ${DATASET} \
            --pruning-mode "${PRUNING_MODE}"
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
