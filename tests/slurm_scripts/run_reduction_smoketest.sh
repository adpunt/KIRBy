#!/bin/bash
#SBATCH --job-name=reduction_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

set -e  # Exit on first failure

echo "=== Step 1: Verify GK import fix ==="
python -c "
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'tests')
from graph_kernel_analysis import compare_reductions, test_allocation_methods
print('GK imports OK')
"

echo ""
echo "=== Step 2: Verify reduction script parses ==="
python tests/dimensionality_reduction_experiments.py --help

echo ""
echo "=== Step 3: Smoke-test Exp A (fast reps, logd) ==="
python tests/dimensionality_reduction_experiments.py \
    --dataset logd --reps fast --experiments A \
    --output tests/results/reduction_logd_fast_smoke.json

echo ""
echo "=== ALL CHECKS PASSED ==="
