#!/bin/bash
#SBATCH --job-name=ab_scenarios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Antibody hybrid user scenario tests
# Tests the full pipeline: data loading, rep generation, hybrid search,
# blueprint reproduction, feature weights extraction.
#
# Usage:
#   sbatch run_antibody_scenarios.sh                                          # mutation_fast, flab_expression_koenig
#   sbatch run_antibody_scenarios.sh fast sabdab_affinity                     # fast reps, sabdab (small)
#   sbatch run_antibody_scenarios.sh mutation_full flab_expression_koenig --delta --augment

REPS=${1:-mutation_fast}
DATASET=${2:-flab_expression_koenig}
EXTRA_FLAGS="${@:3}"

echo "=== Antibody user scenarios (reps=$REPS, dataset=$DATASET) ==="
echo "Extra flags: $EXTRA_FLAGS"
echo "Started: $(date)"
echo ""

python tests/test_hybrid_user_scenarios_antibody.py \
    --scenarios all --reps "$REPS" --dataset "$DATASET" $EXTRA_FLAGS 2>&1

echo ""
echo "Finished: $(date)"
