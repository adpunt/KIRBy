#!/bin/bash
#SBATCH --job-name=ft_antibody
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=11:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Fine-tuning antibody experiments (GPU-enabled)
# Tests LoRA and MLM fine-tuning for DMS intra-family datasets.
#
# Usage:
#   sbatch run_finetune_antibody.sh                                              # defaults
#   sbatch run_finetune_antibody.sh dms flab_expression_koenig lora_esm2         # LoRA ESM-2
#   sbatch run_finetune_antibody.sh dms flab_expression_koenig lora_esm2,mlm_esm2  # both
#   sbatch run_finetune_antibody.sh dms flab_expression_koenig mock_lora         # test plumbing

REPS=${1:-dms}
DATASET=${2:-flab_expression_koenig}
FINETUNE=${3:-lora_esm2}
EXTRA_FLAGS="${@:4}"

echo "=== Antibody fine-tuning (reps=$REPS, dataset=$DATASET, finetune=$FINETUNE) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Extra flags: $EXTRA_FLAGS"
echo "Started: $(date)"
echo ""

# Install fine-tuning deps
pip install --quiet 'urllib3<2' 'peft>=0.7' 'accelerate>=0.25' bitsandbytes 2>/dev/null

python tests/find_best_hybrid_antibody.py "$DATASET" \
    --reps "$REPS" \
    --search fast \
    --finetune "$FINETUNE" \
    --delta \
    $EXTRA_FLAGS 2>&1

echo ""
echo "Finished: $(date)"
