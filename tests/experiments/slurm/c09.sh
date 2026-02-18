#!/bin/bash
#SBATCH --job-name=c09
#SBATCH --partition=long
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Depends on c02 + c06 results
cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

MODE=${1:-intermediary}
DATASET=${2:-logd}
REPS=${3:-fast}
EXTRA="${@:4}"

python tests/experiments/complexity/c09_formal_analysis.py \
    --mode $MODE --dataset $DATASET --reps $REPS --output tests/results \
    --c02-results tests/results/c02_complexity_grid \
    --c06-results tests/results/c06_within_family_gradient $EXTRA
