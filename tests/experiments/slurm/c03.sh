#!/bin/bash
#SBATCH --job-name=c03
#SBATCH --partition=long
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Cross-dataset: run after c02 finishes for all datasets
cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

MODE=${1:-intermediary}
REPS=${2:-fast}
EXTRA="${@:3}"

python tests/experiments/complexity/c03_cross_dataset_transfer.py \
    --mode $MODE --datasets logd caco2 herg qm9 --reps $REPS --output tests/results $EXTRA
