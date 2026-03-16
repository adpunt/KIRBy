#!/bin/bash
#SBATCH --job-name=imrie_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Imrie transfer learning evaluation (OPIG C4).
# Run AFTER imrie_pretrain.sh has completed.
#
# Usage:
#   sbatch tests/slurm_scripts/imrie_eval.sh

cd /data/stat-cadd/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

echo "Imrie transfer learning evaluation"
echo "Start: $(date)"

python tests/experiments/imrie_transfer_learning.py \
    --datasets logd,caco2,herg \
    --reps ecfp4,maccs,pdv,mol2vec,mordred,avalon \
    --n-folds 5 \
    --pretrain-dir /data/stat-cadd/scat9264/kirby_data/chembl_pretrain/

echo "Done: $(date)"
