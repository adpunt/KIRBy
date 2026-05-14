#!/bin/bash
#SBATCH --job-name=pretrain_species_token
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Species-token DNN pretraining for T-PT.5 (species_token_multispecies).
# Trains SpeciesTokenDNNRegressor on the pretrain_corpora 'multispecies_chembl'
# corpus (human + rat + mouse + dog + cow + horse), conditioned on a learned
# 16-D species embedding + an N-D target one-hot. The target-token vocab size N
# is swept N in {10, 50, 200} (design doc §6 Q4); each N gets its own
# checkpoint set and target_vocab.json under .../species_token/N{N}/.
#
# The corpus is leakage-filtered against the human downstream benchmarks AND
# the vet eval targets.
#
# Run BEFORE submit_species_token_multispecies.sh — chain them:
#   cd /data/stat-ecr/scat9264/KIRBy && git pull
#   PT=$(sbatch --parsable tests/slurm_scripts/pretrain_species_token.sh)
#   sbatch --dependency=afterok:$PT tests/slurm_scripts/submit_species_token_multispecies.sh
#
# Usage:
#   sbatch tests/slurm_scripts/pretrain_species_token.sh
#   sbatch tests/slurm_scripts/pretrain_species_token.sh ecfp4   # single rep

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

REP=${1:-ecfp4,maccs,pdv,mol2vec,mordred,avalon}
OUTPUT_DIR=/data/stat-ecr/scat9264/kirby_data/chembl_pretrain/species_token

echo "Pretraining species-token DNN for reps: ${REP}"
echo "N sweep: 10 50 200"
echo "Start: $(date)"

for N in 10 50 200; do
    echo ""
    echo "=== N=${N} ==="
    python -m kirby.models.species_token_dnn \
        --rep "${REP}" \
        --n-targets ${N} \
        --corpus multispecies_chembl \
        --n-compounds 500000 \
        --max-targets 200 \
        --hidden-sizes 256,128,64 \
        --species-emb-dim 16 \
        --output-dir ${OUTPUT_DIR} \
        --data-dir /data/stat-cadd/scat9264/kirby_data/chembl/ \
        --vet-cache-dir /data/stat-ecr/scat9264/kirby_data/cross_species/vet_eval \
        --epochs 200 \
        --patience 30
done

echo "Done: $(date)"
