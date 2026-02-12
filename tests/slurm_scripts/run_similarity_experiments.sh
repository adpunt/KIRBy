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
# Quick: logd with fast + all reps (validates code with pretrained embeddings)
# Full: single dataset with all reps, all experiments (submit 4x for all datasets)
# Usage:
#   sbatch run_similarity_experiments.sh quick
#   sbatch run_similarity_experiments.sh full logd
#   sbatch run_similarity_experiments.sh full caco2
#   sbatch run_similarity_experiments.sh full herg
#   sbatch run_similarity_experiments.sh full qm9

MODE=${1:-quick}
DATASET=${2:-logd}

if [ "$MODE" = "quick" ]; then
    echo "=== QUICK 1: logd, fast reps, Exp 1+2+4+5+8 ==="
    python tests/similarity_experiments.py --mode quick --reps fast \
        --experiments 1 2 4 5 8 --output tests/results/similarity_fast

    echo ""
    echo "=== QUICK 2: logd, all_no_gk reps, Exp 1+2+5 ==="
    python tests/similarity_experiments.py --mode quick --reps all_no_gk \
        --experiments 1 2 5 --output tests/results/similarity_all

elif [ "$MODE" = "full" ]; then
    echo "=== FULL RUN: ${DATASET}, all_no_gk reps, all experiments ==="
    python tests/similarity_experiments.py --mode full --reps all_no_gk \
        --datasets ${DATASET} \
        --experiments 1 2 3 4 5 6 7 8 9 --output tests/results/similarity_${DATASET}
else
    echo "Usage: sbatch run_similarity_experiments.sh [quick|full] [dataset]"
    exit 1
fi
