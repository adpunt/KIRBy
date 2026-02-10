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

MODE=${1:-quick}

if [ "$MODE" = "quick" ]; then
    echo "=== QUICK VALIDATION (logd, Exp 1+2+4+5+8) ==="
    python tests/similarity_experiments.py --mode quick --experiments 1 2 4 5 8 \
        --output tests/results/similarity
elif [ "$MODE" = "full" ]; then
    echo "=== FULL RUN (all datasets, all experiments) ==="
    python tests/similarity_experiments.py --mode full --experiments 1 2 3 4 5 6 7 8 9 \
        --output tests/results/similarity
else
    echo "Usage: sbatch run_similarity_experiments.sh [quick|full]"
    exit 1
fi
