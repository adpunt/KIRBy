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

echo "=== Test 1: Fast search (LogD) ==="
python tests/find_best_hybrid_molecular.py logd --reps fast --search fast

echo ""
echo "=== Test 2: Medium search (Caco2) ==="
echo "4 models x 2 alloc x 3 budgets x 6 importance = 144 configs"
python tests/find_best_hybrid_molecular.py caco2 --reps fast --search medium

echo ""
echo "=== Test 3: Long search (hERG) ==="
echo "7 models x 4 alloc x 4 budgets x 8 importance = 896 configs"
python tests/find_best_hybrid_molecular.py herg --reps fast --search long

echo ""
echo "=== Test 4: Direct library call with RMSE metric ==="
python -c "
import sys
sys.path.insert(0, 'src')
from kirby.find_best_hybrid import find_best_hybrid
from kirby.representations.molecular import create_ecfp4, create_maccs, create_pdv

# Use the molecular wrapper to load LogD
sys.path.insert(0, 'tests')
from find_best_hybrid_molecular import load_dataset

smiles, labels, _ = load_dataset('logd')

reps = {
    'ecfp4': create_ecfp4(smiles),
    'maccs': create_maccs(smiles),
    'pdv': create_pdv(smiles),
}

import numpy as np
result = find_best_hybrid(reps, labels, metric='rmse', speed='fast')
print(f'Best overall (RMSE): {result.best_overall[\"score_mean\"]:.4f}')
print(f'Models tested: {list(result.best_by_model.keys())}')
print(f'lower_is_better={result.lower_is_better}')
"

echo ""
echo "=== Test 5: Custom model ==="
python -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')
from kirby.find_best_hybrid import find_best_hybrid
from kirby.representations.molecular import create_ecfp4, create_maccs, create_pdv
from find_best_hybrid_molecular import load_dataset
from sklearn.svm import SVR
import numpy as np

smiles, labels, _ = load_dataset('logd')

reps = {
    'ecfp4': create_ecfp4(smiles),
    'maccs': create_maccs(smiles),
    'pdv': create_pdv(smiles),
}

result = find_best_hybrid(reps, labels, models={'svm': SVR(kernel='rbf')}, speed='fast')
print(f'SVM best score: {result.best_overall[\"score_mean\"]:.4f}')
"

echo ""
echo "=== All tests complete ==="
