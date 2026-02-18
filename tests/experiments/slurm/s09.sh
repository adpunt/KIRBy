#!/bin/bash
#SBATCH --job-name=s09
#SBATCH --partition=long
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Cross-dataset: run after s02-s08 finish for all datasets
cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

REPS=${1:-fast}
EXTRA="${@:2}"

python tests/experiments/similarity/s09_cross_dataset_summary.py \
    --datasets logd caco2 herg qm9 --results-dir tests/results --output tests/results $EXTRA
