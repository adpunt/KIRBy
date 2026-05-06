#!/bin/bash
#SBATCH --job-name=mol_scenarios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# Molecular hybrid user scenario tests
# Tests the full pipeline: data loading, rep generation, hybrid search,
# blueprint reproduction, feature weights extraction.
#
# Usage:
#   sbatch run_molecular_scenarios.sh                    # fast reps, logd, scenarios 1-8
#   sbatch run_molecular_scenarios.sh fast+hf            # with HF models
#   sbatch run_molecular_scenarios.sh all logd           # all reps
#   sbatch run_molecular_scenarios.sh fast caco2         # different dataset

REPS=${1:-fast}
DATASET=${2:-logd}

echo "=== Molecular user scenarios (reps=$REPS, dataset=$DATASET) ==="
echo "Started: $(date)"
echo ""

python tests/test_hybrid_user_scenarios.py \
    --scenarios all --reps "$REPS" --dataset "$DATASET" 2>&1

echo ""
echo "Finished: $(date)"
