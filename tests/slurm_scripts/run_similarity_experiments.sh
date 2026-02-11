#!/bin/bash
#SBATCH --job-name=similarity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Similarity Experiments
# Quick: logd with fast + all reps (validates code with pretrained embeddings)
# Full: every dataset with all reps, all experiments

MODE=${1:-quick}

if [ "$MODE" = "quick" ]; then
    echo "=== QUICK 1: logd, fast reps, Exp 1+2+4+5+8 ==="
    python tests/similarity_experiments.py --mode quick --reps fast \
        --experiments 1 2 4 5 8 --output tests/results/similarity_fast

    echo ""
    echo "=== QUICK 2: logd, ALL reps, Exp 1+2+5 ==="
    python tests/similarity_experiments.py --mode quick --reps all \
        --experiments 1 2 5 --output tests/results/similarity_all

elif [ "$MODE" = "full" ]; then
    echo "=== FULL RUN (all datasets, ALL reps, all experiments) ==="
    python tests/similarity_experiments.py --mode full --reps all \
        --experiments 1 2 3 4 5 6 7 8 9 --output tests/results/similarity
else
    echo "Usage: sbatch run_similarity_experiments.sh [quick|full]"
    exit 1
fi
