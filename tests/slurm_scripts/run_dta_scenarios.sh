#!/bin/bash
#SBATCH --job-name=dta_scenarios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# DTA (Drug-Target Affinity) hybrid scenario tests
# Tests the full pipeline: data download, rep generation, hybrid search.
#
# Usage:
#   sbatch run_dta_scenarios.sh                           # fast, davis
#   sbatch run_dta_scenarios.sh fast davis                # explicit fast, davis
#   sbatch run_dta_scenarios.sh fast+lm davis             # more reps, davis
#   sbatch run_dta_scenarios.sh fast kiba --max-pairs 30000  # kiba subsampled
#   sbatch run_dta_scenarios.sh full davis                # all reps, davis

REPS=${1:-fast}
DATASET=${2:-davis}
EXTRA_FLAGS="${@:3}"

echo "=== DTA scenarios (reps=$REPS, dataset=$DATASET) ==="
echo "Extra flags: $EXTRA_FLAGS"
echo "Started: $(date)"
echo ""

python tests/find_best_hybrid_dta.py "$DATASET" --reps "$REPS" --search fast $EXTRA_FLAGS 2>&1

echo ""
echo "Finished: $(date)"
