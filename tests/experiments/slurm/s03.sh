#!/bin/bash
#SBATCH --job-name=s03
#SBATCH --partition=long
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Depends on s02 results
cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

MODE=${1:-intermediary}
DATASET=${2:-logd}
REPS=${3:-fast}
EXTRA="${@:4}"

python tests/experiments/similarity/s03_feature_concentration.py \
    --mode $MODE --dataset $DATASET --reps $REPS --output tests/results $EXTRA
