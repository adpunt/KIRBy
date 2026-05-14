#!/bin/bash
#SBATCH --job-name=same_tgt_transfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# same_target_transfer (T-PT.3) — pretrained-DNN vs from-scratch on the four
# small vet ChEMBL eval targets (cow_CA4, cow_DRD2, dog_PTGS2, cow_PTGS2).
#
# PREREQUISITE: imrie_pretrain.sh must have been (re-)run AFTER 2026-05-14 so
# the ChEMBL DNN checkpoints carry 'feat_mean'/'feat_scale'. The zero_shot
# condition raises a clear error on older checkpoints. dnn_tl_full /
# dnn_tl_frozen / dnn_scratch / rf do NOT need the re-run; only zero_shot does.
#
# Usage:
#   sbatch tests/slurm_scripts/submit_same_target_transfer.sh
#   sbatch tests/slurm_scripts/submit_same_target_transfer.sh cow_DRD2  # one target

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

KEYS_ARG=""
if [ -n "$1" ]; then
    KEYS_ARG="--keys $1"
fi

echo "same_target_transfer (T-PT.3)"
echo "Start: $(date)"

python tests/experiments/cross_species/same_target_transfer.py \
    $KEYS_ARG \
    --n-folds 5 \
    --pretrain-dir /data/stat-ecr/scat9264/kirby_data/chembl_pretrain/ \
    --vet-cache-dir /data/stat-ecr/scat9264/kirby_data/cross_species/vet_eval

echo "Done: $(date)"
