#!/bin/bash
#SBATCH --job-name=x_mol_rd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# x_mol_rd_analysis (T-Mol.5) — Downstream R&D analysis script.
# Loads the cached T-Mol.4 hybrid blueprints (x_mol_species_admet_<task>),
# refits the best (model, budget) per task on the full task data, attaches
# a ChemBinnedConformalRegressor (T-UQ.2) calibrated on a scaffold-disjoint
# fold, and predicts per-(task, species) interval for each candidate.
#
# PREREQUISITE on the server:
#   tests/slurm_scripts/submit_x_mol_species_admet.sh has FINISHED (or at
#   least one task's hybrid is in ~/.cache/kirby/searches/). Missing-cache
#   tasks are skipped — the script does NOT recompute the hybrid search.
#
# Usage:
#   # Single candidate (positional SMILES + optional name)
#   sbatch tests/slurm_scripts/submit_x_mol_rd_analysis.sh \
#       'CC(=O)OC1=CC=CC=C1C(=O)O' aspirin
#
#   # Candidate list JSON: [{"name": ..., "smiles": ...}, ...]
#   sbatch tests/slurm_scripts/submit_x_mol_rd_analysis.sh \
#       --candidates-json data/raw/rd_demo_candidates.json

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

echo "x_mol_rd_analysis (T-Mol.5)"
echo "Start: $(date)"

# Forward all args verbatim — supports either the
# `'SMILES' name` positional form or full `--flag` flags.
if [ "${1:0:2}" = "--" ]; then
    python tests/experiments/cross_species/x_mol_rd_analysis.py "$@"
elif [ -n "$1" ]; then
    NAME="${2:-candidate}"
    python tests/experiments/cross_species/x_mol_rd_analysis.py \
        --candidate-smiles "$1" --candidate-name "$NAME"
else
    echo "Usage: sbatch submit_x_mol_rd_analysis.sh 'SMILES' [name]"
    echo "   or: sbatch submit_x_mol_rd_analysis.sh --candidates-json <file>"
    exit 2
fi

echo "Done: $(date)"
