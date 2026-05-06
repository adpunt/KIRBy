#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:00:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

DATASET=${1:?Usage: sbatch submit_f12.sh DATASET}

cd /data/stat-ecr/scat9264/KIRBy
git fetch origin 2>/dev/null
git checkout f12-speedup 2>/dev/null
git reset --hard origin/f12-speedup 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

echo "=== F12: ${DATASET} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log --oneline -1)"
echo "Start:  $(date)"

python tests/experiments/framework/f12_allocation_v2.py \
    --dataset "${DATASET}" --reps all --mode full --max-folds 1 \
    --output tests/results/framework

echo "Done: $(date)"
