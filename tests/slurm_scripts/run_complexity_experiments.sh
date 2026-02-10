#!/bin/bash
#SBATCH --job-name=complexity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Complexity Theory Experiments
# Quick: ~45 min (logd only, 3-fold, RF/kNN/ridge, Exp 1+2+6+9)
# Full:  ~8-10 hours (logd/caco2/herg/qm9, 5-fold, all models, all 9 experiments)

MODE=${1:-quick}

if [ "$MODE" = "quick" ]; then
    echo "=== QUICK VALIDATION (logd, Exp 1+2+6+9) ==="
    python tests/complexity_theory_experiments.py --mode quick --experiments 1 2 6 9 \
        --output tests/results/complexity_theory
elif [ "$MODE" = "full" ]; then
    echo "=== FULL RUN (all datasets, all experiments) ==="
    python tests/complexity_theory_experiments.py --mode full --experiments 1 2 3 4 5 6 7 8 9 \
        --output tests/results/complexity_theory
else
    echo "Usage: sbatch run_complexity_experiments.sh [quick|full]"
    exit 1
fi
