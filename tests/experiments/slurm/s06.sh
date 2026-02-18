#!/bin/bash
#SBATCH --job-name=s06
#SBATCH --partition=long
#SBATCH --time=4:00:00
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

python tests/experiments/similarity/s06_applicability_domain.py \
    --mode $MODE --dataset $DATASET --reps $REPS --output tests/results \
    --s02-results tests/results/s02_hybrid_similarity $EXTRA
