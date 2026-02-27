#!/bin/bash
#SBATCH --job-name=kirby_mol_baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Individual rep baselines for the BOM — no hybrid search, just R² per rep.
# All reps load from cache (.npy), so this is pure model evaluation.
# Should finish in 1-3 hours for all 4 datasets × 16 reps.

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

echo "Started: $(date)"
python -u scripts/run_molecular_baselines.py \
    --output tests/results/baseline_survey/molecular_baselines.json
echo "Finished: $(date)"
