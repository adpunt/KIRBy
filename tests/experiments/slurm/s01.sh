#!/bin/bash
#SBATCH --job-name=s01
#SBATCH --partition=long
#SBATCH --time=0:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

MODE=${1:-intermediary}
DATASET=${2:-logd}
REPS=${3:-fast}
EXTRA="${@:4}"

python tests/experiments/similarity/s01_similarity_baseline.py \
    --mode $MODE --dataset $DATASET --reps $REPS --output tests/results $EXTRA
