#!/bin/bash
#SBATCH --job-name=annitia_views
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=medium
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ANNITIA branch: feat/extra-views — adds recent3, interactions, gap_flags views.
# 11 views total. Subset enumeration capped at size 4 to keep runtime tractable
# (C(11,1..4) = 561 vs 2047 full).

REPO=/data/stat-cadd/scat9264/KIRBy
BRANCH=feat/extra-views
LABEL=views
WORK=/tmp/kirby_${LABEL}_${SLURM_JOB_ID:-$$}
OUT=$REPO/tests/results/annitia_${LABEL}

cd $REPO
git fetch origin $BRANCH
git worktree add $WORK origin/$BRANCH
cd $WORK
. ./setup.sh
mkdir -p $OUT

echo "$(date): launching $LABEL on $BRANCH (worktree $WORK)"

python tests/experiments/annitia/run.py \
    --models cox,coxnet,rsf,gbsa \
    --prune-levels none,50 \
    --top-k-keep 3 \
    --endpoints hepatic,death \
    --n-folds 5 \
    --max-subset-size 4 \
    --output-dir $OUT

python tests/experiments/annitia/submit.py \
    --validation-csv $REPO/data/annitia/val_test.csv \
    --models-dir $OUT \
    --output $OUT/annitia_submission_${LABEL}.csv

cd $REPO
git worktree remove --force $WORK
echo "$(date): done"
ls -la $OUT/
