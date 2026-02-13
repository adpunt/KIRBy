#!/bin/bash
#SBATCH --job-name=ab_scenarios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Antibody hybrid user scenario tests
# Tests the full pipeline: data loading, rep generation, hybrid search,
# blueprint reproduction, feature weights extraction.
#
# Usage:
#   sbatch run_antibody_scenarios.sh                                          # fast reps, sabdab
#   sbatch run_antibody_scenarios.sh fast+lm                                  # with language models
#   sbatch run_antibody_scenarios.sh mutation_fast flab_expression_koenig --delta --augment

REPS=${1:-fast}
DATASET=${2:-sabdab_affinity}
EXTRA_FLAGS="${@:3}"

echo "=== Antibody user scenarios (reps=$REPS, dataset=$DATASET) ==="
echo "Extra flags: $EXTRA_FLAGS"
echo "Started: $(date)"
echo ""

python tests/test_hybrid_user_scenarios_antibody.py \
    --scenarios all --reps "$REPS" --dataset "$DATASET" $EXTRA_FLAGS 2>&1

echo ""
echo "Finished: $(date)"
