#!/bin/bash
# DIAGNOSTIC: Tests each stage of SLURM job execution independently.
# Submit with: sbatch tests/slurm_scripts/diagnostic.sh
# Then: cat slurm-<jobid>.out
#
# This tells you EXACTLY where jobs hang or fail.

#SBATCH --job-name=diagnostic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out

echo "=========================================="
echo "KIRBY SLURM DIAGNOSTIC"
echo "=========================================="
echo "[$(date +%H:%M:%S)] Job started on node: $(hostname)"
echo "[$(date +%H:%M:%S)] Shell: $0 ($(readlink -f /bin/sh 2>/dev/null || echo unknown))"
echo "[$(date +%H:%M:%S)] Working dir: $(pwd)"
echo ""

# --- Stage 1: cd to KIRBy ---
echo "=== STAGE 1: cd to KIRBy ==="
cd /data/stat-ecr/scat9264/KIRBy || { echo "FAILED: cd to KIRBy"; exit 1; }
echo "[$(date +%H:%M:%S)] OK: in $(pwd)"
echo ""

# --- Stage 2: module load CUDA (the suspected hang point) ---
echo "=== STAGE 2: module load CUDA ==="
echo "[$(date +%H:%M:%S)] Starting module load CUDA..."
module load CUDA 2>&1 || echo "[$(date +%H:%M:%S)] module load CUDA failed (exit $?)"
echo "[$(date +%H:%M:%S)] module load CUDA finished"
echo ""

# --- Stage 3: setup.sh with SKIP_INSTALLS (batch mode) ---
echo "=== STAGE 3: setup.sh (SKIP_INSTALLS=1) ==="
echo "[$(date +%H:%M:%S)] Sourcing setup.sh with KIRBY_SKIP_INSTALLS=1..."
export KIRBY_SKIP_INSTALLS=1
. ./setup.sh
echo "[$(date +%H:%M:%S)] setup.sh returned (exit $?)"
echo ""

# --- Stage 4: Python startup ---
echo "=== STAGE 4: Python startup ==="
echo "[$(date +%H:%M:%S)] Testing python..."
python --version 2>&1
echo "[$(date +%H:%M:%S)] Python path: $(which python)"
echo ""

# --- Stage 5: Python imports (the kirby chain) ---
echo "=== STAGE 5: Python imports ==="
echo "[$(date +%H:%M:%S)] Starting Python import test..."
export PYTHONUNBUFFERED=1
python -c "
import sys
sys.stdout.reconfigure(line_buffering=True)
import time

print(f'[{time.strftime(\"%H:%M:%S\")}] Python started, testing imports...')

print(f'[{time.strftime(\"%H:%M:%S\")}] import numpy...', end=' ')
import numpy as np
print(f'OK ({np.__version__})')

print(f'[{time.strftime(\"%H:%M:%S\")}] import scipy...', end=' ')
import scipy
print(f'OK ({scipy.__version__})')

print(f'[{time.strftime(\"%H:%M:%S\")}] import sklearn...', end=' ')
import sklearn
print(f'OK ({sklearn.__version__})')

print(f'[{time.strftime(\"%H:%M:%S\")}] import torch...', end=' ')
try:
    import torch
    print(f'OK ({torch.__version__}, cuda={torch.cuda.is_available()})')
except Exception as e:
    print(f'FAILED: {e}')

print(f'[{time.strftime(\"%H:%M:%S\")}] sys.path insert...')
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')

print(f'[{time.strftime(\"%H:%M:%S\")}] import kirby.hybrid...', end=' ')
from kirby.hybrid import compute_feature_importance, create_hybrid
print('OK')

print(f'[{time.strftime(\"%H:%M:%S\")}] import kirby.find_best_hybrid...', end=' ')
from kirby.find_best_hybrid import SPEED_TIERS, _resolve_models, _resolve_metric
print('OK')

print(f'[{time.strftime(\"%H:%M:%S\")}] import experiments._shared...', end=' ')
from experiments._shared import load_data, FULL_BUDGETS
print('OK')

print(f'[{time.strftime(\"%H:%M:%S\")}] import complexity_helpers...', end=' ')
from experiments.complexity.complexity_helpers import DATASET_REPS
print('OK')

print(f'[{time.strftime(\"%H:%M:%S\")}] All imports successful!')
print(f'[{time.strftime(\"%H:%M:%S\")}] DATASET_REPS keys: {sorted(DATASET_REPS.keys())}')
print(f'[{time.strftime(\"%H:%M:%S\")}] SPEED_TIERS keys: {sorted(SPEED_TIERS.keys())}')

# Quick data load test
print(f'[{time.strftime(\"%H:%M:%S\")}] Loading herg_ki data...')
try:
    data = load_data('herg_ki', rep_names=['ecfp4'])
    print(f'[{time.strftime(\"%H:%M:%S\")}] OK: {len(data[\"labels\"])} samples, reps={list(data[\"reps\"].keys())}')
except Exception as e:
    print(f'[{time.strftime(\"%H:%M:%S\")}] FAILED: {e}')

print(f'[{time.strftime(\"%H:%M:%S\")}] All tests passed!')
"
PYTHON_EXIT=$?
echo "[$(date +%H:%M:%S)] Python exited with code: $PYTHON_EXIT"
echo ""

echo "=========================================="
echo "[$(date +%H:%M:%S)] DIAGNOSTIC COMPLETE"
echo "=========================================="
