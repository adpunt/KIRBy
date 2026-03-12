#!/bin/bash
# Submit comparison experiments as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_comparisons.sh
#
# Experiments:
#   qm9        - Frozen ECFP vs Frozen/E2E SchNet/DimeNet++ (GPU)
#   dta_e2e    - KIRBy Frozen vs DeepDTA vs GraphDTA (GPU)
#   ab_ft      - Frozen vs Fine-tuned antibody LMs (12 models) (GPU)
#   ab_region  - Region ablation: per-CDR, FW-only, entropy pooling (GPU)
#   mol_ft     - Molecular fine-tuning: frozen vs LoRA vs freeze+head (GPU)
#   dta_ft     - DTA LoRA fine-tuning: protein/drug LMs (GPU)
#   tcga_fm    - TCGA foundation models: frozen vs fine-tuned (GPU)
#   reaction   - Reaction rep comparison: 8 reps × 3 models (CPU-only)
#   wsi        - WSI aggregation: mean pool vs gated attention vs ABMIL (GPU)
#
# Usage:
#   bash tests/slurm_scripts/submit_comparisons.sh           # all
#   bash tests/slurm_scripts/submit_comparisons.sh qm9
#   bash tests/slurm_scripts/submit_comparisons.sh mol_ft
#   bash tests/slurm_scripts/submit_comparisons.sh dta_ft
#   bash tests/slurm_scripts/submit_comparisons.sh tcga_fm
#   bash tests/slurm_scripts/submit_comparisons.sh reaction
#   bash tests/slurm_scripts/submit_comparisons.sh wsi

MODE=${1:-all}
OUTPUT="tests/results/comparisons"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit_gpu() {
    local name=$1
    local cmd=$2
    local mem=${3:-64G}
    local wall=${4:-23:59:00}

    J=$(sbatch \
        --job-name="cmp_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}, GPU)"
}

submit_cpu() {
    # HTC requires --gres=gpu:1 for scheduling even for CPU-only jobs
    local name=$1
    local cmd=$2
    local mem=${3:-32G}
    local wall=${4:-12:00:00}

    J=$(sbatch \
        --job-name="cmp_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall})"
}

echo "Submitting comparison experiments..."
echo "  Output: ${OUTPUT}/"
echo ""

run_qm9() {
    echo "--- QM9 3D (5 methods, 3 properties) ---"
    submit_gpu qm9_3d \
        "python tests/experiments/qm9_3d_comparison.py --property gap,homo,mu --n-samples 10000 --output ${OUTPUT}" \
        64G 12:00:00
}

run_dta_e2e() {
    echo "--- DTA E2E (DeepDTA, GraphDTA) ---"
    submit_gpu dta_e2e_davis \
        "python tests/experiments/dta_e2e_comparison.py davis --output ${OUTPUT}" \
        96G 23:59:00
    submit_gpu dta_e2e_kiba \
        "python tests/experiments/dta_e2e_comparison.py kiba --max-pairs 30000 --output ${OUTPUT}" \
        96G 23:59:00
}

run_ab_ft() {
    echo "--- Antibody Fine-Tuning (12 models, frozen vs FT) ---"
    submit_gpu ab_ft \
        "python tests/experiments/antibody_finetune_comparison.py --n-folds 5 --output ${OUTPUT}" \
        96G 47:59:00
}

run_ab_region() {
    echo "--- Antibody Region Ablation (3 LMs) ---"
    submit_gpu ab_region \
        "python tests/experiments/antibody_region_ablation.py --n-folds 5 --lms esm2,esmc,ablang2 --output ${OUTPUT}" \
        96G 23:59:00
}

run_mol_ft() {
    echo "--- Molecular Fine-Tuning (5 LMs × 4 methods + GNN + GROVER) ---"
    submit_gpu mol_ft \
        "python tests/experiments/molecular_finetune_comparison.py --n-folds 5 --output ${OUTPUT}/molecular_finetune_comparison.json" \
        96G 47:59:00
}

run_dta_ft() {
    echo "--- DTA LoRA Fine-Tuning (8 methods) ---"
    submit_gpu dta_ft_davis \
        "python tests/experiments/dta_finetune_comparison.py davis --n-folds 5 --output ${OUTPUT}/dta_finetune_davis.json" \
        96G 47:59:00
    submit_gpu dta_ft_kiba \
        "python tests/experiments/dta_finetune_comparison.py kiba --n-folds 5 --max-pairs 30000 --output ${OUTPUT}/dta_finetune_kiba.json" \
        96G 47:59:00
}

run_tcga_fm() {
    echo "--- TCGA Foundation Models (frozen vs FT) ---"
    submit_gpu tcga_fm \
        "python tests/experiments/tcga_foundation_comparison.py --n-folds 5 --output ${OUTPUT}" \
        256G 47:59:00
}

run_reaction() {
    echo "--- Reaction Rep Comparison (8 reps × 3 models, CPU) ---"
    submit_cpu reaction_reps \
        "python tests/experiments/reaction_rep_comparison.py --n-folds 5 --output ${OUTPUT}/reaction_rep_comparison.json" \
        32G 06:00:00
}

run_wsi() {
    echo "--- WSI Aggregation (7 encoders × 3 methods) ---"
    submit_gpu wsi_luad \
        "python tests/experiments/wsi_abmil_comparison.py --dataset tcga_luad_vs_lusc --n-folds 5 --output ${OUTPUT}" \
        128G 23:59:00
    submit_gpu wsi_brca \
        "python tests/experiments/wsi_abmil_comparison.py --dataset tcga_brca_pam50 --n-folds 5 --output ${OUTPUT}" \
        128G 23:59:00
}

case "$MODE" in
    qm9)        run_qm9 ;;
    dta|dta_e2e) run_dta_e2e ;;
    ab_ft)      run_ab_ft ;;
    ab_region)  run_ab_region ;;
    mol_ft)     run_mol_ft ;;
    dta_ft)     run_dta_ft ;;
    tcga_fm)    run_tcga_fm ;;
    reaction)   run_reaction ;;
    wsi)        run_wsi ;;
    all)
        run_qm9
        run_dta_e2e
        run_ab_ft
        run_ab_region
        run_mol_ft
        run_dta_ft
        run_tcga_fm
        run_reaction
        run_wsi
        echo ""
        echo "  Total: 13 jobs"
        ;;
    *)
        echo "Usage: bash submit_comparisons.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all        All comparison experiments (13 jobs)"
        echo "  qm9        QM9 3D: frozen ECFP vs SchNet/DimeNet++ (1 GPU)"
        echo "  dta_e2e    DTA E2E: KIRBy vs DeepDTA vs GraphDTA (2 GPU)"
        echo "  ab_ft      Antibody fine-tuning: 12 models frozen vs FT (1 GPU)"
        echo "  ab_region  Antibody region ablation: 3 LMs (1 GPU)"
        echo "  mol_ft     Molecular fine-tuning: 5 LMs × 4 methods (1 GPU)"
        echo "  dta_ft     DTA LoRA: 8 protein/drug FT combos (2 GPU)"
        echo "  tcga_fm    TCGA foundation models: frozen vs FT (1 GPU)"
        echo "  reaction   Reaction reps: 8 reps × 3 models (1 CPU)"
        echo "  wsi        WSI aggregation: 7 encoders × 3 methods (2 GPU)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
