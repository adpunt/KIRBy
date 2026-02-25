#!/bin/bash
# Submit fine-tuning survey as parallel SLURM jobs (one per domain).
# Run this on the server login node (NOT via sbatch).
#
# Two-phase fine-tuning workflow:
#   Phase 1: survey (this) — cheap triage of all FT methods
#   Phase 2: full run — use run_finetune_{molecular,antibody,tcga}.sh
#
# Usage:
#   bash tests/slurm_scripts/submit_finetune_survey.sh        # all domains
#   bash tests/slurm_scripts/submit_finetune_survey.sh quick   # molecular only (fastest)

MODE=${1:-all}
SCRIPT=tests/slurm_scripts/run_finetune_survey.sh

echo "Submitting fine-tuning survey jobs (PHASE 1)..."
echo "  Settings: 2 folds, 2 epochs, fast search, GPU required"
echo ""

if [ "$MODE" = "quick" ]; then
    J1=$(sbatch --job-name=ft_survey_mol "$SCRIPT" molecular | awk '{print $4}')
    echo "Submitted:"
    echo "  $J1  molecular  (3 datasets × 2 FT methods: lora_chemberta, lora_molformer)"
    echo ""
    echo "Total: 1 job, 3 runs"

elif [ "$MODE" = "all" ]; then
    J1=$(sbatch --job-name=ft_survey_mol  "$SCRIPT" molecular | awk '{print $4}')
    J2=$(sbatch --job-name=ft_survey_ab   "$SCRIPT" antibody  | awk '{print $4}')
    J3=$(sbatch --job-name=ft_survey_tcga --mem=96G "$SCRIPT" tcga | awk '{print $4}')
    echo "Submitted:"
    echo "  $J1  molecular  (3 datasets × 2 FT methods: lora_chemberta, lora_molformer)"
    echo "  $J2  antibody   (3 datasets × 3 FT methods: lora_esm2, lora_esm2_small, mlm_esm2)"
    echo "  $J3  tcga       (2 datasets × 1 FT method:  geneformer, 96G mem)"
    echo ""
    echo "Total: 3 GPU jobs, 8 runs"

else
    echo "Usage: bash submit_finetune_survey.sh [all|quick]"
    exit 1
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results in:   tests/results/finetune_survey/"
echo ""
echo "After survey completes, check which methods beat frozen baselines,"
echo "then run Phase 2 (full) with the winning methods:"
echo "  sbatch run_finetune_molecular.sh fast logd lora_chemberta"
echo "  sbatch run_finetune_antibody.sh dms flab_expression_koenig lora_esm2"
echo "  sbatch run_finetune_tcga.sh fast tcga_luad_vs_lusc geneformer"
