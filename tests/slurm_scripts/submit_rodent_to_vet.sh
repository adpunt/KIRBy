#!/bin/bash
#SBATCH --job-name=rodent_to_vet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# rodent_to_vet (T-PT.4) — fine-tunes the rodent_chembl pretrained checkpoints
# on the 4 vet eval targets (cow_CA4, cow_DRD2, dog_PTGS2, cow_PTGS2).
#
# PREREQUISITE: pretrain_rodent.sh must run and finish first — it produces the
# rodent_chembl checkpoints this job fine-tunes. Chain them so SLURM enforces
# the order:
#   cd /data/stat-ecr/scat9264/KIRBy && git pull
#   PT=$(sbatch --parsable tests/slurm_scripts/pretrain_rodent.sh)
#   sbatch --dependency=afterok:$PT tests/slurm_scripts/submit_rodent_to_vet.sh
#
# Usage:
#   sbatch tests/slurm_scripts/submit_rodent_to_vet.sh
#   sbatch tests/slurm_scripts/submit_rodent_to_vet.sh cow_DRD2   # one target

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

KEYS_ARG=""
if [ -n "$1" ]; then
    KEYS_ARG="--keys $1"
fi

echo "rodent_to_vet (T-PT.4)"
echo "Start: $(date)"

python tests/experiments/cross_species/rodent_to_vet.py \
    $KEYS_ARG \
    --n-folds 5 \
    --pretrain-dir /data/stat-ecr/scat9264/kirby_data/chembl_pretrain/rodent_chembl \
    --vet-cache-dir /data/stat-ecr/scat9264/kirby_data/cross_species/vet_eval

echo "Done: $(date)"
