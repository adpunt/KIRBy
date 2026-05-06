#!/bin/bash
#SBATCH --job-name=tcga_expanded
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# TCGA expanded reps — test new expression-derived + multi-omic reps
#
# Usage:
#   sbatch run_tcga_expanded.sh                                         # default: fast+diverse, luad_vs_lusc
#   sbatch run_tcga_expanded.sh fast+diverse tcga_brca_pam50            # diverse reps, BRCA
#   sbatch run_tcga_expanded.sh multimodal tcga_luad_vs_lusc            # cross-modality showcase
#   sbatch run_tcga_expanded.sh genomic_diverse tcga_luad_vs_lusc       # multi-omic, no WSI
#   sbatch run_tcga_expanded.sh all tcga_brca_pam50                     # all 27 reps
#
# New presets:
#   fast+diverse  (8 reps)  — maximally diverse paradigms, small
#   multimodal    (12 reps) — cross-modality showcase (genomic + multi-omic + WSI)
#   genomic_diverse (10 reps) — multi-omic without WSI
#   all           (27 reps) — everything

REPS=${1:-fast+diverse}
DATASET=${2:-tcga_luad_vs_lusc}
EXTRA_FLAGS="${@:3}"

echo "=== TCGA expanded reps (reps=$REPS, dataset=$DATASET) ==="
echo "Extra flags: $EXTRA_FLAGS"
echo "Started: $(date)"
echo ""

python tests/find_best_hybrid_tcga.py "$DATASET" --reps "$REPS" --search fast $EXTRA_FLAGS 2>&1

echo ""
echo "Finished: $(date)"
