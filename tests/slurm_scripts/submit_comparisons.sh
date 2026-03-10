#!/bin/bash
# Submit comparison experiments as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_comparisons.sh
#
# Experiments:
#   qm9_3d    - Frozen ECFP vs Frozen SchNet vs End-to-End SchNet (GPU)
#   dta_e2e   - KIRBy Frozen vs DeepDTA vs GraphDTA (GPU)
#   ab_ft     - Frozen vs Fine-tuned antibody LMs (ESM-2 LoRA, AbLang2 FT, AntiBERTy FT)
#   ab_region - Region ablation: per-CDR, FW-only, entropy-guided pooling
#
# Usage:
#   bash tests/slurm_scripts/submit_comparisons.sh           # all
#   bash tests/slurm_scripts/submit_comparisons.sh qm9       # QM9 only
#   bash tests/slurm_scripts/submit_comparisons.sh dta       # DTA only
#   bash tests/slurm_scripts/submit_comparisons.sh ab_ft     # antibody FT only
#   bash tests/slurm_scripts/submit_comparisons.sh ab_region # antibody region ablation

MODE=${1:-all}
OUTPUT="tests/results/comparisons"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

submit() {
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
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error && mkdir -p ${OUTPUT} && ${cmd} 2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}, GPU)"
}

echo "Submitting comparison experiments..."
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    qm9)
        submit qm9_gap "python tests/experiments/qm9_3d_comparison.py --property gap --n-samples 10000 --output ${OUTPUT}" 64G 06:00:00
        ;;
    dta)
        submit dta_davis "python tests/experiments/dta_e2e_comparison.py davis --output ${OUTPUT}" 96G 23:59:00
        submit dta_kiba "python tests/experiments/dta_e2e_comparison.py kiba --max-pairs 30000 --output ${OUTPUT}" 96G 23:59:00
        echo ""
        echo "  Total: 2 jobs (Davis + KIBA)"
        ;;
    ab_ft)
        submit ab_ft "python tests/experiments/antibody_finetune_comparison.py --n-folds 5 --output ${OUTPUT}" 96G 47:59:00
        ;;
    ab_region)
        submit ab_region "python tests/experiments/antibody_region_ablation.py --n-folds 5 --output ${OUTPUT}" 96G 23:59:00
        ;;
    all)
        submit qm9_gap "python tests/experiments/qm9_3d_comparison.py --property gap --n-samples 10000 --output ${OUTPUT}" 64G 06:00:00
        submit dta_davis "python tests/experiments/dta_e2e_comparison.py davis --output ${OUTPUT}" 96G 23:59:00
        submit dta_kiba "python tests/experiments/dta_e2e_comparison.py kiba --max-pairs 30000 --output ${OUTPUT}" 96G 23:59:00
        submit ab_ft "python tests/experiments/antibody_finetune_comparison.py --n-folds 5 --output ${OUTPUT}" 96G 47:59:00
        submit ab_region "python tests/experiments/antibody_region_ablation.py --n-folds 5 --output ${OUTPUT}" 96G 23:59:00
        echo ""
        echo "  Total: 5 jobs (QM9 + Davis + KIBA + Antibody FT + Region Ablation)"
        ;;
    *)
        echo "Usage: bash submit_comparisons.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all       All comparison experiments (5 GPU jobs)"
        echo "  qm9       QM9 3D comparison only (frozen vs end-to-end SchNet)"
        echo "  dta       DTA e2e comparison only (KIRBy vs DeepDTA vs GraphDTA)"
        echo "  ab_ft     Antibody fine-tuning (ESM-2 LoRA, AbLang2 FT, AntiBERTy FT)"
        echo "  ab_region Antibody region ablation (per-CDR, FW, entropy pooling)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
