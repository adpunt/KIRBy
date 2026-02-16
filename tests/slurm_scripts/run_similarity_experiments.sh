#!/bin/bash
#SBATCH --job-name=similarity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Similarity Experiments
#
# Quick mode: logd only, fast reps, subset of experiments
# Full mode: split into parts to fit within 48h wall time
#
# Usage:
#   sbatch run_similarity_experiments.sh quick
#
#   # Full mode — submit one job per dataset × part:
#   sbatch run_similarity_experiments.sh full_a logd    # Exp 1,2,4,5,6,8 (core)
#   sbatch run_similarity_experiments.sh full_b logd    # Exp 3,7 (supplementary)
#   sbatch run_similarity_experiments.sh full_a caco2
#   sbatch run_similarity_experiments.sh full_b caco2
#   sbatch run_similarity_experiments.sh full_a herg
#   sbatch run_similarity_experiments.sh full_b herg
#   sbatch run_similarity_experiments.sh full_a qm9
#   sbatch run_similarity_experiments.sh full_b qm9
#   sbatch run_similarity_experiments.sh full_cross     # Exp 9 (cross-dataset summary)

MODE=${1:-quick}
DATASET=${2:-logd}

if [ "$MODE" = "quick" ]; then
    echo "=== QUICK 1: logd, fast reps, Exp 1+2+4+5+8 ==="
    python tests/similarity_experiments.py --mode quick --reps fast \
        --experiments 1 2 4 5 8 --output tests/results/similarity_fast

    echo ""
    echo "=== QUICK 2: logd, all reps, Exp 1+2+5 ==="
    python tests/similarity_experiments.py --mode quick --reps all \
        --experiments 1 2 5 --output tests/results/similarity_all

elif [ "$MODE" = "full_a" ]; then
    echo "=== FULL PART A: ${DATASET}, fast+hf reps, Exp 1+2+4+5+6+8 ==="
    python tests/similarity_experiments.py --mode full \
        --datasets ${DATASET} \
        --experiments 1 2 4 5 6 8 --output tests/results/similarity_${DATASET}_a

elif [ "$MODE" = "full_b" ]; then
    echo "=== FULL PART B: ${DATASET}, fast+hf reps, Exp 3+7 ==="
    python tests/similarity_experiments.py --mode full \
        --datasets ${DATASET} \
        --experiments 3 7 --output tests/results/similarity_${DATASET}_b

elif [ "$MODE" = "full_cross" ]; then
    echo "=== FULL CROSS-DATASET: all 4 datasets, fast+hf reps, Exp 9 ==="
    python tests/similarity_experiments.py --mode full \
        --datasets logd caco2 herg qm9 \
        --experiments 9 --output tests/results/similarity_cross

else
    echo "Usage: sbatch run_similarity_experiments.sh [quick|full_a|full_b|full_cross] [dataset]"
    echo "Datasets: logd, caco2, herg, qm9"
    echo ""
    echo "Full mode split:"
    echo "  full_a    = Exp 1,2,4,5,6,8 (core similarity experiments)"
    echo "  full_b    = Exp 3,7          (supplementary: neighbor preservation, noise FP)"
    echo "  full_cross = Exp 9           (cross-dataset summary, loads all datasets)"
    exit 1
fi
