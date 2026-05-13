#!/bin/bash
#SBATCH --job-name=fnd_hyb_vet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# foundation_hybrid_vet (T-PT.2) — KIRBy hybrid search on small vet ChEMBL
# targets (cow_CA4, cow_DRD2, dog_PTGS2, cow_PTGS2).
#
# Usage:
#   sbatch tests/slurm_scripts/submit_foundation_hybrid_vet.sh           # general preset (18 reps)
#   sbatch tests/slurm_scripts/submit_foundation_hybrid_vet.sh minimal   # minimal preset (4 reps)

PRESET="${1:-general}"

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

echo "foundation_hybrid_vet (T-PT.2)"
echo "Preset: $PRESET"
echo "Start:  $(date)"

python tests/experiments/cross_species/foundation_hybrid_vet.py \
    --preset "$PRESET" \
    --n-folds 5 \
    --vet-cache-dir /data/stat-ecr/scat9264/kirby_data/cross_species/vet_eval

echo "Done: $(date)"
