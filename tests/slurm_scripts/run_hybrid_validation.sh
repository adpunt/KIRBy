#!/bin/bash
#SBATCH --job-name=hybrid_val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

echo "=== Test 1: Quick molecular search (existing workflow) ==="
python tests/find_best_hybrid_molecular.py esol --reps fast --search fast

echo ""
echo "=== Test 2: Direct library call with RMSE metric ==="
python -c "
import sys
sys.path.insert(0, 'src')
from kirby.find_best_hybrid import find_best_hybrid
from kirby.representations.molecular import create_ecfp4, create_maccs, create_pdv
from kirby.datasets.esol import load_esol
import numpy as np

splits = load_esol(splitter='scaffold')
smiles = splits['train']['smiles'] + splits['val']['smiles'] + splits['test']['smiles']
labels = np.concatenate([splits['train']['labels'], splits['val']['labels'], splits['test']['labels']])

reps = {
    'ecfp4': create_ecfp4(smiles),
    'maccs': create_maccs(smiles),
    'pdv': create_pdv(smiles),
}

result = find_best_hybrid(reps, labels, metric='rmse', speed='fast')
print(f'Best overall (RMSE): {result.best_overall[\"score_mean\"]:.4f}')
print(f'Models tested: {list(result.best_by_model.keys())}')
"

echo ""
echo "=== Test 3: Custom model ==="
python -c "
import sys
sys.path.insert(0, 'src')
from kirby.find_best_hybrid import find_best_hybrid
from kirby.representations.molecular import create_ecfp4, create_maccs, create_pdv
from kirby.datasets.esol import load_esol
from sklearn.svm import SVR
import numpy as np

splits = load_esol(splitter='scaffold')
smiles = splits['train']['smiles'] + splits['val']['smiles'] + splits['test']['smiles']
labels = np.concatenate([splits['train']['labels'], splits['val']['labels'], splits['test']['labels']])

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
