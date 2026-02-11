#!/bin/bash
#SBATCH --job-name=hybrid_val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Hybrid Validation
# Quick: fast + fast+hf on logd (validates code runs with pretrained embeddings)
# Full: every dataset tested with all reps

MODE=${1:-quick}

if [ "$MODE" = "quick" ]; then
    echo "=== Quick 1: LogD, fast reps, fast search ==="
    python tests/find_best_hybrid_molecular.py logd --reps fast --search fast

    echo ""
    echo "=== Quick 2: LogD, fast+hf reps (pretrained), fast search ==="
    python tests/find_best_hybrid_molecular.py logd --reps fast+hf --search fast

    echo ""
    echo "=== Quick 3: LogD, ALL reps, fast search ==="
    python tests/find_best_hybrid_molecular.py logd --reps all_no_gk --search fast

elif [ "$MODE" = "full" ]; then
    echo "=== Test 1: LogD, all reps, medium search ==="
    python tests/find_best_hybrid_molecular.py logd --reps all_no_gk --search medium

    echo ""
    echo "=== Test 2: Caco2, all reps, medium search ==="
    python tests/find_best_hybrid_molecular.py caco2 --reps all_no_gk --search medium

    echo ""
    echo "=== Test 3: hERG, all reps, long search ==="
    python tests/find_best_hybrid_molecular.py herg --reps all_no_gk --search long

    echo ""
    echo "=== Test 4: User scenarios (logd, fast reps) ==="
    python tests/test_hybrid_user_scenarios.py --dataset logd --reps fast --scenarios all

    echo ""
    echo "=== Test 5: User scenarios (caco2, fast reps) ==="
    python tests/test_hybrid_user_scenarios.py --dataset caco2 --reps fast --scenarios 1 4 5
else
    echo "Usage: sbatch run_hybrid_validation.sh [quick|full]"
    exit 1
fi

echo ""
echo "=== All tests complete ==="
