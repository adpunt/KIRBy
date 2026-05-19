#!/bin/bash
#SBATCH --job-name=x_ab_ness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# T-Ab.4 — Cross-species antibody-ness classifier (canineness OR bovineness).
#
# PREREQUISITES on ARC:
#   - DoggifAI manually unpacked at $KIRBY_CROSS_SPECIES_DIR/canine_antibodies/
#     (target=dog only) — see docs/cross_species_data_inventory.md §2.1
#   - tests/slurm_scripts/submit_bovine_assembly.sh has FINISHED and produced
#     bovine_vh.fasta (target=cow only) — see §2.2
#   - ANARCI + HMMER installed (see SETUP.md X-Ab section)
#
# Usage:
#   sbatch tests/slurm_scripts/submit_x_ab_ness.sh dog
#   sbatch tests/slurm_scripts/submit_x_ab_ness.sh cow

set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
    echo "Usage: sbatch submit_x_ab_ness.sh {dog|cow}"
    exit 2
fi

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null || true
. ./setup.sh
export PYTHONWARNINGS=ignore

echo "x_ab_ness (T-Ab.4)  target=$TARGET"
echo "Start: $(date)"

python tests/experiments/cross_species/x_ab_ness.py \
    --target-species "$TARGET" \
    --n-folds 5

echo "Done: $(date)"
