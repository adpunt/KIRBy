#!/bin/bash
#SBATCH --job-name=similarity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# Similarity Experiments
#
# Quick mode: logd only, fast reps, subset of experiments
# Full mode: split into parts to fit within 48h wall time
#   Default reps: fast+hf (safe). Use 3rd arg to override (e.g. "all" after pre-caching).
#
# Usage:
#   sbatch run_similarity_experiments.sh quick
#
#   # Full mode — submit one job per dataset × part:
#   sbatch run_similarity_experiments.sh full_a logd           # fast+hf reps
#   sbatch run_similarity_experiments.sh full_b logd
#   sbatch run_similarity_experiments.sh full_a logd all       # all reps (needs pre-cache)
#   sbatch run_similarity_experiments.sh full_cross            # Exp 9 (cross-dataset)

MODE=${1:-quick}
DATASET=${2:-logd}
REPS=${3:-}

if [ "$MODE" = "quick" ]; then
    echo "=== QUICK 1: logd, fast reps, Exp 1+2+4+5+8 ==="
    python tests/similarity_experiments.py --mode quick --reps fast \
        --experiments 1 2 4 5 8 --output tests/results/similarity_fast

    echo ""
    echo "=== QUICK 2: logd, all reps, Exp 1+2+5 ==="
    python tests/similarity_experiments.py --mode quick --reps all \
        --experiments 1 2 5 --output tests/results/similarity_all

elif [ "$MODE" = "full_a" ]; then
    REP_FLAG=${REPS:-fast+hf}
    echo "=== FULL PART A: ${DATASET}, ${REP_FLAG} reps, Exp 1+2+4+5+6+8 ==="
    python tests/similarity_experiments.py --mode full --reps ${REP_FLAG} \
        --datasets ${DATASET} \
        --experiments 1 2 4 5 6 8 --output tests/results/similarity_${DATASET}_a

elif [ "$MODE" = "full_b" ]; then
    REP_FLAG=${REPS:-fast+hf}
    echo "=== FULL PART B: ${DATASET}, ${REP_FLAG} reps, Exp 3+7 ==="
    python tests/similarity_experiments.py --mode full --reps ${REP_FLAG} \
        --datasets ${DATASET} \
        --experiments 3 7 --output tests/results/similarity_${DATASET}_b

elif [ "$MODE" = "full_cross" ]; then
    REP_FLAG=${REPS:-fast+hf}
    echo "=== FULL CROSS-DATASET: all 4 datasets, ${REP_FLAG} reps, Exp 9 ==="
    python tests/similarity_experiments.py --mode full --reps ${REP_FLAG} \
        --datasets logd caco2 herg qm9 \
        --experiments 9 --output tests/results/similarity_cross

else
    echo "Usage: sbatch run_similarity_experiments.sh [quick|full_a|full_b|full_cross] [dataset] [reps]"
    echo "Datasets: logd, caco2, herg, qm9"
    echo "Reps: fast+hf (default), all (needs run_precache_reps.sh first)"
    echo ""
    echo "Full mode split:"
    echo "  full_a    = Exp 1,2,4,5,6,8 (core similarity experiments)"
    echo "  full_b    = Exp 3,7          (supplementary: neighbor preservation, noise FP)"
    echo "  full_cross = Exp 9           (cross-dataset summary, loads all datasets)"
    exit 1
fi
