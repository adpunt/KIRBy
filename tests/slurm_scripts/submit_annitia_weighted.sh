#!/bin/bash
#SBATCH --job-name=annitia_weighted
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=medium
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ANNITIA branch: feat/weighted — sample-weighted RSF/GBSA variants.
# Adds rsf_w/gbsa_w (k-NN reliability weights, importance-weighted distance,
# adapted from tests/experiments/noise/n02) and rsf_if/gbsa_if (inverse-
# frequency event weighting). 8 model registry entries total.

REPO=/data/stat-cadd/scat9264/KIRBy
BRANCH=feat/weighted
LABEL=weighted
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
    --models cox,coxnet,rsf,gbsa,rsf_w,gbsa_w,rsf_if,gbsa_if \
    --prune-levels none,50 \
    --top-k-keep 3 \
    --endpoints hepatic,death \
    --n-folds 5 \
    --output-dir $OUT

python tests/experiments/annitia/submit.py \
    --validation-csv $REPO/data/annitia/val_test.csv \
    --models-dir $OUT \
    --output $OUT/annitia_submission_${LABEL}.csv

cd $REPO
git worktree remove --force $WORK
echo "$(date): done"
ls -la $OUT/
