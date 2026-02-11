#!/bin/bash
#SBATCH --job-name=gk_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

DATASET=${1:-logd}

echo "=== Graph Kernel Analysis: ${DATASET} ==="
python tests/graph_kernel_analysis.py --dataset ${DATASET} \
    --output tests/results/gk_analysis --folds 3
