#!/bin/bash
#SBATCH --job-name=x_mol_species
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# x_mol_species_admet (T-Mol.4) — species-stratified ADMET via species-as-token
# augmentation. Concatenates per-organism ChEMBL slices for COX-2, DRD2, the
# five canine CYP isoforms, and ABCB1 (P-gp), attaches a 5-D one-hot species
# token, and runs find_best_hybrid on the union.
#
# PREREQUISITES on the server:
#   1. ChEMBL caches for the new organisms ('dog', 'cow') under
#      ~/.cache/kirby/COX2/ and ~/.cache/kirby/DRD2/. First run will pull them
#      automatically; allow ~10 min the first time.
#   2. For canine CYP / P-gp coverage beyond the documented-skip path, populate
#      data/raw/canine_cyp_<iso>.csv / canine_pgp.csv from literature curation,
#      or run `python scripts/resolve_chembl_targets.py --genes CYP1A2 CYP2B11
#      CYP2C21 CYP2D15 CYP3A12 ABCB1 --species human dog` to refresh
#      docs/cross_species_targets_resolved.json.
#
# Usage:
#   sbatch tests/slurm_scripts/submit_x_mol_species_admet.sh
#   sbatch tests/slurm_scripts/submit_x_mol_species_admet.sh cox2,drd2  # subset

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

TASKS_ARG=""
if [ -n "$1" ]; then
    TASKS_ARG="--tasks $1"
fi

echo "x_mol_species_admet (T-Mol.4)"
echo "Start: $(date)"

python tests/experiments/cross_species/x_mol_species_admet.py \
    $TASKS_ARG \
    --preset full \
    --n-folds 5

echo "Done: $(date)"
