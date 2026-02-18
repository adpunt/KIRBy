#!/bin/bash
#SBATCH --job-name=c01
#SBATCH --partition=long
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

MODE=${1:-intermediary}
DATASET=${2:-logd}
REPS=${3:-fast}
EXTRA="${@:4}"

python tests/experiments/complexity/c01_rep_characterization.py \
    --mode $MODE --dataset $DATASET --reps $REPS --output tests/results $EXTRA
