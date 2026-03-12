#!/bin/bash
# Phase 5 baseline jobs for HTC (GPU jobs).
# Run from HTC login node.
#
# Usage:
#   bash tests/slurm_scripts/submit_phase5_htc.sh           # all jobs
#   bash tests/slurm_scripts/submit_phase5_htc.sh antibody
#   bash tests/slurm_scripts/submit_phase5_htc.sh dta
#   bash tests/slurm_scripts/submit_phase5_htc.sh comparisons
#   bash tests/slurm_scripts/submit_phase5_htc.sh molecular
#   bash tests/slurm_scripts/submit_phase5_htc.sh reaction
#   bash tests/slurm_scripts/submit_phase5_htc.sh tcga

set -euo pipefail

MODE=${1:-all}
BASE_OUTPUT="tests/results/phase5_baselines"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-64G}
    local wall=${4:-47:59:00}

    J=$(sbatch \
        --job-name="p5_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}, GPU)"
}

echo "============================================"
echo "Phase 5: HTC Jobs (all require --gres=gpu:1 for scheduling)"
echo "============================================"

run_molecular() {
    local OUT="${BASE_OUTPUT}/molecular"
    echo "--- Molecular (8 datasets, ~35 reps) ---"
    submit mol_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py molecular --n-folds 5 --augment --output ${OUT}/" \
        64G 47:59:00
    echo ""
}

run_reaction() {
    local OUT="${BASE_OUTPUT}/reaction"
    echo "--- Reaction (3 datasets, 8 reps) ---"
    submit rxn_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py reaction --n-folds 5 --augment --output ${OUT}/" \
        32G 23:59:00
    echo ""
}

run_antibody_fast() {
    local OUT="${BASE_OUTPUT}/antibody"
    echo "--- Antibody fast reps (onehot, blosum, aac, paratope) ---"
    submit ab_fast_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps onehot,blosum,aac,paratope" \
        32G 23:59:00
    submit ab_fast_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps onehot,blosum,aac,paratope" \
        32G 23:59:00
    echo ""
}

run_tcga() {
    local OUT="${BASE_OUTPUT}/tcga"
    echo "--- TCGA (3 datasets, 57 reps) ---"
    submit tcga_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py tcga --n-folds 5 --augment --output ${OUT}/" \
        256G 47:59:00
    echo ""
}

run_antibody_gpu() {
    local OUT="${BASE_OUTPUT}/antibody"
    echo "--- Antibody GPU reps ---"

    # LM reps
    submit ab_lm_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr" \
        96G 47:59:00
    submit ab_lm_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr" \
        96G 47:59:00

    # Structural reps
    submit ab_struct_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps structural" \
        64G 47:59:00
    submit ab_struct_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps structural" \
        64G 47:59:00

    # Mutation reps
    submit ab_mut_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr" \
        96G 47:59:00
    submit ab_mut_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr" \
        96G 47:59:00

    # Variation reps
    submit ab_var_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps esm2_entropy_top20,esm2_per_cdr,esm2_per_region" \
        96G 47:59:00
    submit ab_var_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps esm2_entropy_top20,esm2_per_cdr,esm2_per_region" \
        96G 47:59:00

    # MINT
    submit ab_mint_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps mint" \
        96G 47:59:00
    submit ab_mint_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps mint" \
        96G 47:59:00

    echo "  Total: 10 antibody GPU jobs"
    echo ""
}

run_dta() {
    local OUT="${BASE_OUTPUT}/dta"
    echo "--- DTA Baselines ---"
    submit dta_davis \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis.json" \
        64G 47:59:00
    submit dta_kiba \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --output ${OUT}/kiba.json" \
        64G 47:59:00
    submit dta_bindingdb \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py bindingdb_kd --n-folds 5 --max-pairs 30000 --output ${OUT}/bindingdb_kd.json" \
        64G 47:59:00

    local OUT2="${BASE_OUTPUT}/dta_universal"
    submit dta_universal \
        "mkdir -p ${OUT2} && python tests/run_baseline_survey.py dta --n-folds 5 --augment --output ${OUT2}/" \
        96G 47:59:00
    echo ""
}

run_comparisons() {
    local OUT="tests/results/comparisons"
    echo "--- Comparison Experiments (13 jobs: 12 GPU + 1 CPU) ---"

    # QM9 3D: 5 methods × 3 properties
    submit cmp_qm9 \
        "mkdir -p ${OUT} && python tests/experiments/qm9_3d_comparison.py --property gap,homo,mu --n-samples 10000 --output ${OUT}" \
        64G 12:00:00

    # DTA E2E: DeepDTA, GraphDTA
    submit cmp_dta_e2e_davis \
        "mkdir -p ${OUT} && python tests/experiments/dta_e2e_comparison.py davis --output ${OUT}" \
        96G 23:59:00
    submit cmp_dta_e2e_kiba \
        "mkdir -p ${OUT} && python tests/experiments/dta_e2e_comparison.py kiba --max-pairs 30000 --output ${OUT}" \
        96G 23:59:00

    # Antibody fine-tuning: 12 models
    submit cmp_ab_ft \
        "mkdir -p ${OUT} && python tests/experiments/antibody_finetune_comparison.py --n-folds 5 --output ${OUT}" \
        96G 47:59:00

    # Antibody region ablation: 3 LMs
    submit cmp_ab_region \
        "mkdir -p ${OUT} && python tests/experiments/antibody_region_ablation.py --n-folds 5 --lms esm2,esmc,ablang2 --output ${OUT}" \
        96G 23:59:00

    # Molecular fine-tuning: 5 LMs × 4 methods + GNN + GROVER
    submit cmp_mol_ft \
        "mkdir -p ${OUT} && python tests/experiments/molecular_finetune_comparison.py --n-folds 5 --output ${OUT}/molecular_finetune_comparison.json" \
        96G 47:59:00

    # DTA LoRA fine-tuning: 8 methods
    submit cmp_dta_ft_davis \
        "mkdir -p ${OUT} && python tests/experiments/dta_finetune_comparison.py davis --n-folds 5 --output ${OUT}/dta_finetune_davis.json" \
        96G 47:59:00
    submit cmp_dta_ft_kiba \
        "mkdir -p ${OUT} && python tests/experiments/dta_finetune_comparison.py kiba --n-folds 5 --max-pairs 30000 --output ${OUT}/dta_finetune_kiba.json" \
        96G 47:59:00

    # TCGA foundation models: frozen vs fine-tuned
    submit cmp_tcga_fm \
        "mkdir -p ${OUT} && python tests/experiments/tcga_foundation_comparison.py --n-folds 5 --output ${OUT}" \
        256G 47:59:00

    # Reaction rep comparison: 8 reps × 3 models (CPU-only, no GPU needed)
    submit cmp_reaction \
        "mkdir -p ${OUT} && python tests/experiments/reaction_rep_comparison.py --n-folds 5 --output ${OUT}/reaction_rep_comparison.json" \
        32G 06:00:00

    # WSI aggregation: 7 encoders × 3 methods
    submit cmp_wsi_luad \
        "mkdir -p ${OUT} && python tests/experiments/wsi_abmil_comparison.py --dataset tcga_luad_vs_lusc --n-folds 5 --output ${OUT}" \
        128G 23:59:00
    submit cmp_wsi_brca \
        "mkdir -p ${OUT} && python tests/experiments/wsi_abmil_comparison.py --dataset tcga_brca_pam50 --n-folds 5 --output ${OUT}" \
        128G 23:59:00

    echo "  Total: 13 comparison jobs"
    echo ""
}

case "$MODE" in
    molecular)   run_molecular ;;
    reaction)    run_reaction ;;
    antibody_fast) run_antibody_fast ;;
    tcga)        run_tcga ;;
    antibody)    run_antibody_gpu ;;
    dta)         run_dta ;;
    comparisons) run_comparisons ;;
    all)
        run_molecular
        run_reaction
        run_antibody_fast
        run_tcga
        run_antibody_gpu
        run_dta
        run_comparisons
        echo "============================================"
        echo "Total: 32 jobs (5 CPU + 10 ab_gpu + 4 dta + 13 cmp)"
        echo "============================================"
        ;;
    *) echo "Usage: bash submit_phase5_htc.sh [all|molecular|reaction|antibody_fast|tcga|antibody|dta|comparisons]"; exit 1 ;;
esac
