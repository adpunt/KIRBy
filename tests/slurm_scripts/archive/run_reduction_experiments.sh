#!/bin/bash
#SBATCH --job-name=reduction_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

DATASET=${1:-logd}
REPS=${2:-fast}
EXPERIMENTS=${3:-"A B C"}

echo "=== Dimensionality Reduction Experiments ==="
echo "Dataset: ${DATASET}, Reps: ${REPS}, Experiments: ${EXPERIMENTS}"

python tests/dimensionality_reduction_experiments.py \
    --dataset ${DATASET} \
    --reps ${REPS} \
    --experiments ${EXPERIMENTS} \
    --output tests/results/reduction_${DATASET}_${REPS}.json \
    --folds 3
