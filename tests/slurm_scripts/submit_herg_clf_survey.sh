#!/bin/bash
#SBATCH --job-name=herg_clf_survey
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# hERG classification survey (TDC binary blocker, ~10K molecules)
# No GPU needed — molecular reps are CPU-only.

set -eo pipefail

echo "=== hERG classification baseline survey ==="
echo "Branch: $(git branch --show-current) ($(git rev-parse --short HEAD))"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo ""

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null || true
. ./setup.sh

export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

OUT="tests/results/phase5_baselines/molecular"
mkdir -p "$OUT"

python tests/run_baseline_survey.py herg \
    --reps all \
    --n-folds 5 \
    --augment \
    --output "$OUT/"

echo ""
echo "=== Done: $(date) ==="
