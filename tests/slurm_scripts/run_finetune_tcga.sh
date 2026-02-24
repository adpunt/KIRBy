#!/bin/bash
#SBATCH --job-name=ft_tcga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Fine-tuning TCGA experiments (GPU-enabled)
# Tests Geneformer fine-tuning on TCGA classification/regression tasks.
#
# Usage:
#   sbatch run_finetune_tcga.sh                                         # defaults
#   sbatch run_finetune_tcga.sh fast tcga_luad_vs_lusc geneformer       # Geneformer
#   sbatch run_finetune_tcga.sh fast tcga_brca_pam50 geneformer         # BRCA PAM50
#   sbatch run_finetune_tcga.sh fast tcga_luad_vs_lusc mock_lora        # test plumbing

REPS=${1:-fast}
DATASET=${2:-tcga_luad_vs_lusc}
FINETUNE=${3:-geneformer}
EXTRA_FLAGS="${@:4}"

echo "=== TCGA fine-tuning (reps=$REPS, dataset=$DATASET, finetune=$FINETUNE) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Extra flags: $EXTRA_FLAGS"
echo "Started: $(date)"
echo ""

# Install fine-tuning deps
pip install --quiet peft accelerate bitsandbytes 2>/dev/null

python tests/find_best_hybrid_tcga.py "$DATASET" \
    --reps "$REPS" \
    --search fast \
    --finetune "$FINETUNE" \
    $EXTRA_FLAGS 2>&1

echo ""
echo "Finished: $(date)"
