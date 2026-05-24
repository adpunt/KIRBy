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

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

. setup.sh

# Bypass HF datasets FileLock on shared NFS cache. With cache pre-warmed,
# load_dataset reads directly without acquiring the lock — fixes the
# OSError [Errno 116] thundering-herd seen on the 2026-05-17 wave.
# Override with HF_DATASETS_OFFLINE=0 only when a brand-new dataset
# needs to be downloaded for the first time.
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

echo "=== ${CATEGORY} experiments: ${DATASET} ==="
echo "Branch: ${BRANCH} (${COMMIT})"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Pruning mode: ${PRUNING_MODE}"
echo "Start:  $(date)"

# Loop the requested categories sequentially within this single SLURM job.
# Captures the first non-zero exit code but keeps running the rest so a
# noise failure doesn't skip similarity, etc. SLURM reports the worst RC.
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
    if [ ${RC} -ne 0 ] && [ ${WORST_RC} -eq 0 ]; then
        WORST_RC=${RC}
    fi
    echo "--- ${CAT} done (exit=${RC}) ---"
done

echo ""
echo "Done:   $(date)  (worst exit=${WORST_RC})"
# Propagate the worst python exit code so SLURM reports FAILED, not COMPLETED,
# on bus errors / signals (the previous masking of $? made sacct say
# COMPLETED 0:0 for the entire Wave 1 bus-error batch).
exit ${WORST_RC}
