#!/bin/bash
#SBATCH --job-name=pretrain_rodent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Rodent (rat + mouse) ChEMBL DNN pretraining for T-PT.4 (rodent_to_vet).
# Uses the pretrain_corpora 'rodent_chembl' corpus; checkpoints land in
# .../chembl_pretrain/rodent_chembl/{rep}_{hs}.pt. The corpus is leakage-
# filtered against the human downstream benchmarks AND the vet eval targets.
#
# Run BEFORE submit_rodent_to_vet.sh.
#
# Usage:
#   sbatch tests/slurm_scripts/pretrain_rodent.sh
#   sbatch tests/slurm_scripts/pretrain_rodent.sh ecfp4   # single rep

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

REP=${1:-ecfp4,maccs,pdv,mol2vec,mordred,avalon}

echo "Pretraining rodent_chembl DNN for reps: ${REP}"
echo "Start: $(date)"

python -m kirby.models.pretrain_chembl_dnn \
    --rep "${REP}" \
    --corpus rodent_chembl \
    --n-compounds 500000 \
    --max-targets 200 \
    --hidden-sizes 256,128,64 \
    --output-dir /data/stat-ecr/scat9264/kirby_data/chembl_pretrain/ \
    --data-dir /data/stat-cadd/scat9264/kirby_data/chembl/ \
    --vet-cache-dir /data/stat-ecr/scat9264/kirby_data/cross_species/vet_eval \
    --epochs 200 \
    --patience 30

echo "Done: $(date)"
