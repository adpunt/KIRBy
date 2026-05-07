#!/bin/bash
#SBATCH --job-name=annitia_nhanes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=medium
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# ANNITIA branch: feat/nhanes-pretrain — pretrain an MLP trunk on NHANES
# linked-mortality (FIB-4 > 1.3 stratum), then finetune a Cox head on
# ANNITIA folds. Targets the death endpoint primarily; on hepatic the
# external-data lift is expected to be small (NHANES barely has hepatic
# decompensation events).
#
# The encoder is saved at $OUT/encoder.pt and the search.py runner is
# pointed at it via the ANNITIA_NHANES_ENCODER env var. The standard
# search loop then runs the pretrained_mlp model on the nhanes_proj view
# (and only that view, because it's the only view in the NHANES feature
# schema).

REPO=/data/stat-ecr/scat9264/KIRBy
BRANCH=feat/nhanes-pretrain
LABEL=nhanes
WORK=/tmp/kirby_${LABEL}_${SLURM_JOB_ID:-$$}
OUT=$REPO/tests/results/annitia_${LABEL}

cd $REPO
git fetch origin $BRANCH
git worktree add $WORK origin/$BRANCH
cd $WORK
. ./setup.sh

mkdir -p $OUT
echo "$(date): launching $LABEL on $BRANCH"

# Phase 1 — build cohort + pretrain encoder. Cohort assembly downloads
# NHANES Continuous + linked mortality on first run; results cached at
# $NHANES_DATA_DIR (defaults to /data/stat-ecr/scat9264/KIRBy/data/nhanes).
export NHANES_DATA_DIR=$REPO/data/nhanes

python tests/experiments/annitia/pretrain_nhanes.py \
    --restrict fib4_high \
    --target all_cause \
    --hidden 128x64 \
    --epochs 100 \
    --output $OUT/encoder.pt

# Phase 2 — finetune via the standard search runner. We pin the view to
# nhanes_proj (the only view that matches the encoder's schema) and the
# model to pretrained_mlp. The Cox/RSF/GBSA baselines on the SAME view
# are also included so we can read off the lift from pretraining
# directly.
export ANNITIA_NHANES_ENCODER=$OUT/encoder.pt

python tests/experiments/annitia/run.py \
    --views nhanes_proj \
    --models pretrained_mlp,cox,coxnet,rsf,gbsa \
    --prune-levels none \
    --max-subset-size 1 \
    --top-k-keep 5 \
    --n-folds 5 \
    --endpoints hepatic,death \
    --output-dir $OUT/finetune

cd $REPO
git worktree remove --force $WORK
echo "$(date): done"
ls -la $OUT/ $OUT/finetune/ 2>/dev/null
