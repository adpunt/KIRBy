#!/bin/bash
#SBATCH --job-name=complexity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Complexity Theory Experiments
#
# Quick mode: logd only, fast reps, subset of experiments
# Full mode: split into parts to fit within 48h wall time
#
# Usage:
#   sbatch run_complexity_experiments.sh quick
#   sbatch run_complexity_experiments.sh gk
#
#   # Full mode — submit one job per dataset × part:
#   sbatch run_complexity_experiments.sh full_a logd    # Exp 1,2,4,8,9 (core + ANOVA)
#   sbatch run_complexity_experiments.sh full_b logd    # Exp 5,6,7,10 (expensive)
#   sbatch run_complexity_experiments.sh full_a caco2
#   sbatch run_complexity_experiments.sh full_b caco2
#   sbatch run_complexity_experiments.sh full_a herg
#   sbatch run_complexity_experiments.sh full_b herg
#   sbatch run_complexity_experiments.sh full_a qm9
#   sbatch run_complexity_experiments.sh full_b qm9
#   sbatch run_complexity_experiments.sh full_cross     # Exp 3 (cross-dataset, loads all 4)

MODE=${1:-quick}
DATASET=${2:-logd}

if [ "$MODE" = "quick" ]; then
    echo "=== QUICK 1: logd, fast reps, Exp 1+2+6+9+10 ==="
    python tests/complexity_theory_experiments.py --mode quick --reps fast \
        --experiments 1 2 6 9 10 --output tests/results/complexity_theory_fast

    echo ""
    echo "=== QUICK 2: logd, all reps, Exp 1+2+6 ==="
    python tests/complexity_theory_experiments.py --mode quick --reps all \
        --experiments 1 2 6 --output tests/results/complexity_theory_all

elif [ "$MODE" = "full_a" ]; then
    echo "=== FULL PART A: ${DATASET}, fast+hf reps, Exp 1+2+4+8+9 ==="
    python tests/complexity_theory_experiments.py --mode full \
        --datasets ${DATASET} \
        --experiments 1 2 4 8 9 --output tests/results/complexity_${DATASET}_a

elif [ "$MODE" = "full_b" ]; then
    echo "=== FULL PART B: ${DATASET}, fast+hf reps, Exp 5+6+7+10 ==="
    python tests/complexity_theory_experiments.py --mode full \
        --datasets ${DATASET} \
        --experiments 5 6 7 10 --output tests/results/complexity_${DATASET}_b

elif [ "$MODE" = "full_cross" ]; then
    echo "=== FULL CROSS-DATASET: all 4 datasets, fast+hf reps, Exp 3 ==="
    python tests/complexity_theory_experiments.py --mode full \
        --datasets logd caco2 herg qm9 \
        --experiments 3 --output tests/results/complexity_cross

elif [ "$MODE" = "gk" ]; then
    echo "=== GK MODE: logd only, all reps (incl graph_kernel), fast search, Exp 1+2+6 ==="
    python tests/complexity_theory_experiments.py --mode quick --reps all \
        --experiments 1 2 6 --output tests/results/complexity_theory_gk

else
    echo "Usage: sbatch run_complexity_experiments.sh [quick|full_a|full_b|full_cross|gk] [dataset]"
    echo "Datasets: logd, caco2, herg, qm9"
    echo ""
    echo "Full mode split:"
    echo "  full_a  = Exp 1,2,4,8,9 (core grid, ANOVA, calibration)"
    echo "  full_b  = Exp 5,6,7,10  (learning curves, capacity sweeps, subpops, noise)"
    echo "  full_cross = Exp 3       (cross-dataset transfer, loads all datasets)"
    exit 1
fi
