#!/bin/bash
#SBATCH --job-name=hybrid_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

echo "=== Experiment 1: Aggregated importance (LogD, fast4 methods) ==="
echo "Tests borda/avg/RRF aggregation vs single-method selection"
python tests/test_hybrid_aggregation.py logd --reps fast --budgets 50 100 200 --methods fast4

echo ""
echo "=== Experiment 2: Aggregated importance (LogD, all 6 cluster reps) ==="
echo "Adds permutation + kernelshap to the aggregation"
python tests/test_hybrid_aggregation.py logd --reps fast --budgets 50 100 200 --methods all_cluster

echo ""
echo "=== Experiment 3: Low-dim optimization (LogD) ==="
echo "Tests budgets 15/25/50 with RF, kNN, ridge"
python tests/test_hybrid_low_dim.py logd --reps fast --budgets 15 25 50

echo ""
echo "=== Experiment 4: Low-dim optimization (Caco2) ==="
python tests/test_hybrid_low_dim.py caco2 --reps fast --budgets 15 25 50

echo ""
echo "=== All experiments complete ==="
