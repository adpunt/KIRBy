#!/bin/bash
#SBATCH --job-name=ft_molecular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=11:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# Fine-tuning molecular experiments (GPU-enabled)
# Tests LoRA fine-tuning of ChemBERTa / MolFormer on QSAR datasets.
#
# Usage:
#   sbatch run_finetune_molecular.sh                                        # defaults
#   sbatch run_finetune_molecular.sh fast logd lora_chemberta               # ChemBERTa LoRA
#   sbatch run_finetune_molecular.sh fast logd lora_chemberta,lora_molformer  # both
#   sbatch run_finetune_molecular.sh fast logd mock_lora                    # test plumbing

REPS=${1:-fast}
DATASET=${2:-logd}
FINETUNE=${3:-lora_chemberta}
EXTRA_FLAGS="${@:4}"

echo "=== Molecular fine-tuning (reps=$REPS, dataset=$DATASET, finetune=$FINETUNE) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Extra flags: $EXTRA_FLAGS"
echo "Started: $(date)"
echo ""

# Install fine-tuning deps
pip install --quiet 'urllib3<2' 'peft>=0.7' 'accelerate>=0.25' bitsandbytes 2>/dev/null

python tests/find_best_hybrid_molecular.py "$DATASET" \
    --reps "$REPS" \
    --search fast \
    --finetune "$FINETUNE" \
    $EXTRA_FLAGS 2>&1

echo ""
echo "Finished: $(date)"
