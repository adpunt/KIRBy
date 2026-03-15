#!/bin/bash
#SBATCH --job-name=imrie_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# One-time ChEMBL DNN pre-training for Imrie transfer learning (OPIG C4).
# Run this BEFORE imrie_eval.sh.
#
# Usage:
#   sbatch tests/slurm_scripts/imrie_pretrain.sh
#   sbatch tests/slurm_scripts/imrie_pretrain.sh ecfp4   # single rep

cd /data/stat-cadd/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

REP=${1:-ecfp4,maccs,pdv,mol2vec,mordred,avalon}

echo "Pre-training DNN on ChEMBL for reps: ${REP}"
echo "Start: $(date)"

python -m kirby.models.pretrain_chembl_dnn \
    --rep "${REP}" \
    --n-compounds 500000 \
    --max-targets 200 \
    --hidden-sizes 256,128,64 \
    --output-dir /data/stat-cadd/scat9264/kirby_data/chembl_pretrain/ \
    --data-dir /data/stat-cadd/scat9264/kirby_data/chembl/ \
    --epochs 200 \
    --patience 30

echo "Done: $(date)"
