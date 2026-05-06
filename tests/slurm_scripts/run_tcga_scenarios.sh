#!/bin/bash
#SBATCH --job-name=tcga_scenarios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# TCGA hybrid user scenario tests
# Tests the full pipeline: data download, rep generation, hybrid search.
#
# Usage:
#   sbatch run_tcga_scenarios.sh                                    # fast, luad_vs_lusc
#   sbatch run_tcga_scenarios.sh fast tcga_brca_pam50               # fast reps, BRCA PAM50
#   sbatch run_tcga_scenarios.sh fast+learned tcga_luad_vs_lusc     # more reps
#   sbatch run_tcga_scenarios.sh all tcga_brca_survival             # all reps, survival
#   sbatch run_tcga_scenarios.sh genomic_fast tcga_luad_vs_lusc     # genomic only

REPS=${1:-fast}
DATASET=${2:-tcga_luad_vs_lusc}
EXTRA_FLAGS="${@:3}"

echo "=== TCGA scenarios (reps=$REPS, dataset=$DATASET) ==="
echo "Extra flags: $EXTRA_FLAGS"
echo "Started: $(date)"
echo ""

python tests/find_best_hybrid_tcga.py "$DATASET" --reps "$REPS" --search fast $EXTRA_FLAGS 2>&1

echo ""
echo "Finished: $(date)"
