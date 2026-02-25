#!/bin/bash
#SBATCH --job-name=ft_survey
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

# =============================================================================
# FINE-TUNING SURVEY: Cheap triage of all fine-tuning methods
# =============================================================================
#
# Phase 1 of the two-phase fine-tuning workflow:
#   Phase 1 (this): Fast survey — 2 folds, 2 epochs, fast search.
#     Determines which FT methods improve over frozen baselines.
#   Phase 2: Full run — 5 folds, 5 epochs, medium search.
#     Only for methods that passed the survey.
#     Use existing run_finetune_{molecular,antibody,tcga}.sh scripts.
#
# Each method is tested alongside frozen reps so the output's BASELINE
# COMPARISON table shows frozen vs fine-tuned head-to-head.
#
# Usage:
#   sbatch run_finetune_survey.sh              # everything
#   sbatch run_finetune_survey.sh molecular    # molecular only
#   sbatch run_finetune_survey.sh antibody     # antibody only
#   sbatch run_finetune_survey.sh tcga         # TCGA only

DOMAIN=${1:-all}
OUTPUT="tests/results/finetune_survey"

mkdir -p "$OUTPUT"

# Install fine-tuning deps (pin urllib3 to avoid DEFAULT_CIPHERS removal in v2)
pip install --quiet 'urllib3<2' 'peft>=0.7' 'accelerate>=0.25' bitsandbytes 2>/dev/null

echo "============================================================"
echo "KIRBy Fine-Tuning Survey (PHASE 1)"
echo "  Domain:    $DOMAIN"
echo "  Strategy:  2 folds, 2 epochs, fast search, performance_weighted allocation"
echo "  Output:    $OUTPUT"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Started:   $(date)"
echo "============================================================"

PASS=0
FAIL=0

run() {
    local script=$1
    local dataset=$2
    local preset=$3
    local finetune=$4
    local tag=$5
    shift 5
    local extra="$@"

    echo ""
    echo "--- $tag ---"
    echo "  Script:    $script"
    echo "  Dataset:   $dataset"
    echo "  Preset:    $preset"
    echo "  Finetune:  $finetune"
    echo "  Extra:     ${extra:-none}"
    echo "  Started:   $(date)"

    if python "$script" "$dataset" \
        --reps "$preset" --search fast \
        --n-folds 2 \
        --allocation performance_weighted \
        --finetune "$finetune" \
        --finetune-epochs 2 \
        $extra \
        --output "${OUTPUT}/${tag}.json" 2>&1; then
        echo "  PASSED: $(date)"
        PASS=$((PASS + 1))
    else
        echo "  FAILED (exit $?): $(date)"
        FAIL=$((FAIL + 1))
    fi
}

# =============================================================================
# MOLECULAR: LoRA ChemBERTa + LoRA MolFormer on representative datasets
# 'fast' preset (3 frozen reps) + 2 fine-tuned = 5 total reps
# =============================================================================
run_molecular() {
    echo ""
    echo "========== MOLECULAR FINE-TUNING SURVEY =========="

    for dataset in logd caco2 herg; do
        run tests/find_best_hybrid_molecular.py \
            "$dataset" fast "lora_chemberta,lora_molformer" \
            "ft_mol_${dataset}"
    done
}

# =============================================================================
# ANTIBODY: LoRA ESM-2 + LoRA ESM-2-small + MLM ESM-2 on representative DMS
# 'dms' preset (9 frozen reps) + 3 fine-tuned = 12 total reps
# =============================================================================
run_antibody() {
    echo ""
    echo "========== ANTIBODY FINE-TUNING SURVEY =========="

    # Intra-family (DMS): test on the largest/most representative dataset
    run tests/find_best_hybrid_antibody.py \
        flab_expression_koenig dms "lora_esm2,lora_esm2_small,mlm_esm2" \
        "ft_ab_flab_expression_koenig" --delta

    # Also test on a binding dataset to see if FT helps there
    run tests/find_best_hybrid_antibody.py \
        flab_binding_koenig dms "lora_esm2,lora_esm2_small,mlm_esm2" \
        "ft_ab_flab_binding_koenig" --delta

    # One AbAgym dataset as a third reference point
    run tests/find_best_hybrid_antibody.py \
        abagym_Ang2_2017_G6 dms "lora_esm2,lora_esm2_small,mlm_esm2" \
        "ft_ab_abagym_Ang2" --delta
}

# =============================================================================
# TCGA: Geneformer on classification tasks
# 'fast' preset (5-8 frozen reps) + 1 fine-tuned = 6-9 total reps
# =============================================================================
run_tcga() {
    echo ""
    echo "========== TCGA FINE-TUNING SURVEY =========="

    for dataset in tcga_luad_vs_lusc tcga_brca_pam50; do
        run tests/find_best_hybrid_tcga.py \
            "$dataset" fast "geneformer" \
            "ft_tcga_${dataset}"
    done
}

# =============================================================================
# DISPATCH
# =============================================================================
case "$DOMAIN" in
    molecular)  run_molecular ;;
    antibody)   run_antibody ;;
    tcga)       run_tcga ;;
    all)
        run_molecular
        run_antibody
        run_tcga
        ;;
    *)
        echo "Usage: sbatch run_finetune_survey.sh [all|molecular|antibody|tcga]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "FINE-TUNING SURVEY COMPLETE"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Results: $OUTPUT"
echo "  Finished: $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Check BASELINE COMPARISON tables in each .json output"
echo "  2. For methods that beat frozen baselines, run full fine-tuning:"
echo "     sbatch run_finetune_molecular.sh fast logd lora_chemberta"
echo "     sbatch run_finetune_antibody.sh dms flab_expression_koenig lora_esm2"
echo "     sbatch run_finetune_tcga.sh fast tcga_luad_vs_lusc geneformer"
