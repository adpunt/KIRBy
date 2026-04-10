#!/bin/bash
# CGR-SMILES corpus generation + MLM pretraining (Phase 4 of CGR_EMBEDDING_RESEARCH.md).
#
# Two stages:
#   1. Corpus: Download USPTO-STEREO, convert ~1M reactions to CGR-SMILES (~30 min, CPU)
#   2. Pretrain: Continue MLM pretraining of ChemBERTa on CGR-SMILES (~2-4 hours, GPU)
#
# Output: /data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/
#   corpus.txt        — CGR-SMILES lines (~600K-800K)
#   corpus_stats.json — conversion statistics
#   checkpoint/       — pretrained model + extended tokenizer
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_cgr_smiles_pretrain.sh           # full pipeline
#   bash tests/slurm_scripts/submit_cgr_smiles_pretrain.sh corpus    # corpus only
#   bash tests/slurm_scripts/submit_cgr_smiles_pretrain.sh pretrain  # pretrain only

STAGE=${1:-all}
OUTPUT_DIR="/data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' >&2)"

case "$STAGE" in
    corpus)
        # CPU-only job for corpus generation
        J=$(sbatch \
            --job-name="cgr_corpus" \
            --nodes=1 --ntasks=1 --cpus-per-task=16 \
            --time=02:00:00 --partition=short \
            --mem=64G \
            --mail-user="${EMAIL}" --mail-type=END,FAIL \
            --wrap="${WRAP_PREFIX} && python -m kirby.finetuning.cgr_smiles_pretrain --stage corpus --output-dir ${OUTPUT_DIR} --n-workers 16 ${STDERR_FILTER}" \
            | awk '{print $4}')
        echo "  $J  cgr_corpus  (short, 64G, 2h, 16 CPU, no GPU)"
        ;;
    pretrain)
        # GPU job for MLM pretraining (assumes corpus already exists)
        J=$(sbatch \
            --job-name="cgr_pretrain" \
            --nodes=1 --ntasks=1 --cpus-per-task=8 \
            --time=06:00:00 --partition=medium \
            --mem=128G \
            --gres=gpu:1 \
            --mail-user="${EMAIL}" --mail-type=END,FAIL \
            --wrap="${WRAP_PREFIX} && python -m kirby.finetuning.cgr_smiles_pretrain --stage pretrain --output-dir ${OUTPUT_DIR} --max-steps 50000 --batch-size 32 ${STDERR_FILTER}" \
            | awk '{print $4}')
        echo "  $J  cgr_pretrain  (medium, 128G, 6h, GPU)"
        ;;
    all)
        # Single job: corpus + pretrain
        J=$(sbatch \
            --job-name="cgr_smiles" \
            --nodes=1 --ntasks=1 --cpus-per-task=16 \
            --time=08:00:00 --partition=long \
            --mem=128G \
            --gres=gpu:1 \
            --mail-user="${EMAIL}" --mail-type=END,FAIL \
            --wrap="${WRAP_PREFIX} && python -m kirby.finetuning.cgr_smiles_pretrain --stage all --output-dir ${OUTPUT_DIR} --n-workers 16 --max-steps 50000 --batch-size 32 ${STDERR_FILTER}" \
            | awk '{print $4}')
        echo "  $J  cgr_smiles  (long, 128G, 8h, GPU + 16 CPU)"
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_cgr_smiles_pretrain.sh [STAGE]"
        echo ""
        echo "Stages:"
        echo "  all       Full pipeline: corpus + pretrain (default)"
        echo "  corpus    Corpus generation only (CPU, ~30 min)"
        echo "  pretrain  MLM pretraining only (GPU, ~2-4 hours)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Output: ${OUTPUT_DIR}/"
