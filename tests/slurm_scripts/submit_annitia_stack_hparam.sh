#!/bin/bash
#SBATCH --job-name=annitia_stack_hparam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Targeted hparam sweep on the stacker winner.
# Restricts the search to a single view subset (the death winner from the
# stack branch — snapshot+slope+missingness) and tries 12 stacker variants:
#   outer in {coxnet, rsf, gbsa}
#   l1_ratio in {0.1, 0.3, 0.5, 0.7, 0.9}  (coxnet only)
#   n_inner_folds in {5, 10}
# Total: ~12 candidates per endpoint x 5 folds x 2 endpoints = ~120 fits,
# but stacker fits are expensive (~25s each) so plan ~1-2h wall.

REPO=/data/stat-ecr/scat9264/KIRBy
BRANCH=feat/stack-hparam
LABEL=stack_hparam
WORK=/tmp/kirby_${LABEL}_${SLURM_JOB_ID:-$$}
OUT=$REPO/tests/results/annitia_${LABEL}

cd $REPO
git fetch origin $BRANCH
git worktree add $WORK origin/$BRANCH
cd $WORK
. ./setup.sh
mkdir -p $OUT

echo "$(date): launching $LABEL on $BRANCH (worktree $WORK)"

# Restrict views to JUST the death-winner subset (3 views) so the search loop
# reduces to single-subset hparam comparison. Hepatic uses delta+nit which
# is also reachable from this list.
python tests/experiments/annitia/run.py \
    --views snapshot,slope,missingness,delta,nit,leakage,clinical \
    --models cox,coxnet,rsf,gbsa,stack,stack_l1_01,stack_l1_03,stack_l1_07,stack_l1_09,stack_k10,stack_k10_l1_03,stack_k10_l1_07,stack_rsf,stack_gbsa,stack_rsf_k10,stack_gbsa_k10 \
    --prune-levels none \
    --top-k-keep 5 \
    --endpoints hepatic,death \
    --n-folds 5 \
    --max-subset-size 3 \
    --resume \
    --output-dir $OUT

python tests/experiments/annitia/submit.py \
    --validation-csv $REPO/data/annitia/val_test.csv \
    --models-dir $OUT \
    --output $OUT/annitia_submission_${LABEL}.csv

cd $REPO
git worktree remove --force $WORK
echo "$(date): done"
ls -la $OUT/
