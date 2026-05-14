#!/bin/bash
#SBATCH --job-name=species_token_multispecies
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# species_token_multispecies (T-PT.5) — fine-tunes the species-token DNN
# checkpoints on the 4 vet eval targets (cow_CA4, cow_DRD2, dog_PTGS2,
# cow_PTGS2), sweeping the target-token vocab size N in {10, 50, 200}, and runs
# the real-species vs human-token toggle.
#
# PREREQUISITE: pretrain_species_token.sh must run and finish first — it
# produces the N{10,50,200} checkpoint sets this job fine-tunes. Chain them so
# SLURM enforces the order:
#   cd /data/stat-ecr/scat9264/KIRBy && git pull
#   PT=$(sbatch --parsable tests/slurm_scripts/pretrain_species_token.sh)
#   sbatch --dependency=afterok:$PT tests/slurm_scripts/submit_species_token_multispecies.sh
#
# Usage:
#   sbatch tests/slurm_scripts/submit_species_token_multispecies.sh
#   sbatch tests/slurm_scripts/submit_species_token_multispecies.sh cow_DRD2  # one target

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

KEYS_ARG=""
if [ -n "$1" ]; then
    KEYS_ARG="--keys $1"
fi

echo "species_token_multispecies (T-PT.5)"
echo "Start: $(date)"

python tests/experiments/cross_species/species_token_multispecies.py \
    $KEYS_ARG \
    --n-folds 5 \
    --pretrain-dir /data/stat-ecr/scat9264/kirby_data/chembl_pretrain/species_token \
    --vet-cache-dir /data/stat-ecr/scat9264/kirby_data/cross_species/vet_eval

echo "Done: $(date)"
