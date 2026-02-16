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
#   Default reps: fast+hf (safe). Use 3rd arg to override (e.g. "all" after pre-caching).
#
# Usage:
#   sbatch run_complexity_experiments.sh quick
#   sbatch run_complexity_experiments.sh gk
#
#   # Full mode — submit one job per dataset × part:
#   sbatch run_complexity_experiments.sh full_a logd           # fast+hf reps
#   sbatch run_complexity_experiments.sh full_b logd
#   sbatch run_complexity_experiments.sh full_a logd all       # all reps (needs pre-cache)
#   sbatch run_complexity_experiments.sh full_b logd all
#   sbatch run_complexity_experiments.sh full_cross            # Exp 3 (cross-dataset)
#   sbatch run_complexity_experiments.sh full_cross "" all     # all reps

MODE=${1:-quick}
DATASET=${2:-logd}
REPS=${3:-}

if [ "$MODE" = "quick" ]; then
    echo "=== QUICK 1: logd, fast reps, Exp 1+2+6+9+10 ==="
    python tests/complexity_theory_experiments.py --mode quick --reps fast \
        --experiments 1 2 6 9 10 --output tests/results/complexity_theory_fast

    echo ""
    echo "=== QUICK 2: logd, all reps, Exp 1+2+6 ==="
    python tests/complexity_theory_experiments.py --mode quick --reps all \
        --experiments 1 2 6 --output tests/results/complexity_theory_all

elif [ "$MODE" = "full_a" ]; then
    REP_FLAG=${REPS:-fast+hf}
    echo "=== FULL PART A: ${DATASET}, ${REP_FLAG} reps, Exp 1+2+4+8+9 ==="
    python tests/complexity_theory_experiments.py --mode full --reps ${REP_FLAG} \
        --datasets ${DATASET} \
        --experiments 1 2 4 8 9 --output tests/results/complexity_${DATASET}_a

elif [ "$MODE" = "full_b" ]; then
    REP_FLAG=${REPS:-fast+hf}
    echo "=== FULL PART B: ${DATASET}, ${REP_FLAG} reps, Exp 5+6+7+10 ==="
    python tests/complexity_theory_experiments.py --mode full --reps ${REP_FLAG} \
        --datasets ${DATASET} \
        --experiments 5 6 7 10 --output tests/results/complexity_${DATASET}_b

elif [ "$MODE" = "full_cross" ]; then
    REP_FLAG=${REPS:-fast+hf}
    echo "=== FULL CROSS-DATASET: all 4 datasets, ${REP_FLAG} reps, Exp 3 ==="
    python tests/complexity_theory_experiments.py --mode full --reps ${REP_FLAG} \
        --datasets logd caco2 herg qm9 \
        --experiments 3 --output tests/results/complexity_cross

elif [ "$MODE" = "gk" ]; then
    echo "=== GK MODE: logd only, all reps (incl graph_kernel), fast search, Exp 1+2+6 ==="
    python tests/complexity_theory_experiments.py --mode quick --reps all \
        --experiments 1 2 6 --output tests/results/complexity_theory_gk

else
    echo "Usage: sbatch run_complexity_experiments.sh [quick|full_a|full_b|full_cross|gk] [dataset] [reps]"
    echo "Datasets: logd, caco2, herg, qm9"
    echo "Reps: fast+hf (default), all (needs run_precache_reps.sh first)"
    echo ""
    echo "Full mode split:"
    echo "  full_a  = Exp 1,2,4,8,9 (core grid, ANOVA, calibration)"
    echo "  full_b  = Exp 5,6,7,10  (learning curves, capacity sweeps, subpops, noise)"
    echo "  full_cross = Exp 3       (cross-dataset transfer, loads all datasets)"
    exit 1
fi
