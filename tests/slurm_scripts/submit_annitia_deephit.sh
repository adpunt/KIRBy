#!/bin/bash
#SBATCH --job-name=annitia_deephit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --partition=medium
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ANNITIA branch: feat/deephit-mt — competing-risks discrete-time NN with
# shared trunk over (hepatic, death). Joint training borrows strength across
# the death endpoint to compensate for the 47 hepatic events.
#
# Backend: tries pycox first; if pycox import fails (server install issues)
# the wrapper falls back to a pure-torch implementation that produces the
# same discrete-time competing-risks loss. CPU only — 1253 patients.

REPO=/data/stat-ecr/scat9264/KIRBy
BRANCH=feat/deephit-mt
LABEL=deephit
WORK=/tmp/kirby_${LABEL}_${SLURM_JOB_ID:-$$}
OUT=$REPO/tests/results/annitia_${LABEL}

cd $REPO
git fetch origin $BRANCH
git worktree add $WORK origin/$BRANCH
cd $WORK
. ./setup.sh

# Optional: install pycox into the active env. If this fails, the wrapper
# silently falls back to the torch backend (so do NOT abort on failure).
pip install pycox==0.3.0 torchtuples==0.2.2 || true

mkdir -p $OUT

echo "$(date): launching $LABEL on $BRANCH (worktree $WORK)"

python tests/experiments/annitia/run_deephit.py \
    --n-folds 5 \
    --num-durations 15,20,30 \
    --hidden 64x64,128x64 \
    --dropout 0.2,0.3,0.5 \
    --lr 1e-3,5e-4 \
    --epochs 200 \
    --patience 20 \
    --output-dir $OUT

python tests/experiments/annitia/submit_deephit.py \
    --validation-csv $REPO/data/annitia/val_test.csv \
    --models-dir $OUT \
    --output $OUT/annitia_submission_${LABEL}.csv

cd $REPO
git worktree remove --force $WORK
echo "$(date): done"
ls -la $OUT/
