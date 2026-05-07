#!/bin/bash
#SBATCH --job-name=annitia_dsm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --partition=medium
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ANNITIA branch: feat/dsm-cr — Deep Survival Machines (Nagpal et al. 2021).
# Runs the joint competing-risks DSM (DeepSurvivalMachinesCR) — two cause-
# specific Weibull-mixture heads sharing a trunk. The hepatic head borrows
# representational strength from the larger death event count.
#
# Backend: tries auton-survival first; falls back to a pure-torch implementation
# if auton-survival isn't installed. Both code paths are CPU-friendly at n=1253.

REPO=/data/stat-ecr/scat9264/KIRBy
BRANCH=feat/dsm-cr
LABEL=dsm
WORK=/tmp/kirby_${LABEL}_${SLURM_JOB_ID:-$$}
OUT=$REPO/tests/results/annitia_${LABEL}

cd $REPO
git fetch origin $BRANCH
git worktree add $WORK origin/$BRANCH
cd $WORK
. ./setup.sh

# auton-survival is the preferred backend; if install fails the wrapper
# silently uses the torch-only implementation.
pip install auton-survival || true

mkdir -p $OUT

echo "$(date): launching $LABEL on $BRANCH (worktree $WORK)"

python tests/experiments/annitia/run_dsm.py \
    --n-folds 5 \
    --k 3,4,6 \
    --hidden 64x64,128x64 \
    --dropout 0.2,0.3,0.5 \
    --lr 1e-3,5e-4 \
    --epochs 200 \
    --patience 20 \
    --output-dir $OUT

cd $REPO
git worktree remove --force $WORK
echo "$(date): done"
ls -la $OUT/
