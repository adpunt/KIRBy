#!/bin/bash
# Phase 5A: Re-run ALL baseline surveys with corrected pipeline.
#
# This is the master script for Phase 5 of ISSUES.md. It runs every
# baseline survey across all domains with the full Phase 0-4 changes:
#   - Corrected preprocessing (SMILES standardization, InChIKey dedup, BOS/EOS)
#   - Per-fold PCA/scaling (no data leakage)
#   - Domain-appropriate primary metrics (RMSE molecular, CI DTA, AUROC/F1w TCGA)
#   - 11 downstream models (ridge, rf, xgboost, lightgbm, knn, gp, mlp, dnn, cnn, catboost)
#   - Position-modulo DMS splits, GroupKFold reaction splits
#   - All Phase 3+4 representations included
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_phase5_baselines.sh              # everything
#   bash tests/slurm_scripts/submit_phase5_baselines.sh molecular    # one domain
#   bash tests/slurm_scripts/submit_phase5_baselines.sh antibody     # antibody only
#   bash tests/slurm_scripts/submit_phase5_baselines.sh dta          # DTA only
#   bash tests/slurm_scripts/submit_phase5_baselines.sh tcga         # TCGA only
#   bash tests/slurm_scripts/submit_phase5_baselines.sh reaction     # reaction only
#
# Output: tests/results/phase5_baselines/<domain>/<dataset>.json

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
    local wall=${4:-23:59:00}
    local gpu=${5:-}
    local partition=${6:-long}

    J=$(sbatch \
        --job-name="p5_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition="${partition}" \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        ${gpu:+--gres=gpu:1} \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}${gpu:+, GPU})"
}

echo "============================================"
echo "Phase 5A: Baseline Surveys (corrected pipeline)"
echo "============================================"
echo "  Output: ${BASE_OUTPUT}/"
echo ""

# ---------------------------------------------------------------
# MOLECULAR (8 datasets, ~35 reps, 10 models)
# ---------------------------------------------------------------
run_molecular() {
    local OUT="${BASE_OUTPUT}/molecular"
    echo "--- Molecular (8 datasets) ---"
    submit mol_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py molecular --n-folds 5 --augment --output ${OUT}/" \
        64G 23:59:00
    echo ""
}

# ---------------------------------------------------------------
# MOLECULAR with fine-tuning (ChemBERTa/MolFormer LoRA + SchNet E2E)
# ---------------------------------------------------------------
run_molecular_ft() {
    local OUT="${BASE_OUTPUT}/molecular_ft"
    echo "--- Molecular Fine-tuned (8 datasets, GPU) ---"
    submit mol_ft_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py molecular --n-folds 5 --augment --finetune --output ${OUT}/" \
        64G 23:59:00 gpu
    echo ""
}

# ---------------------------------------------------------------
# TCGA (3 datasets, 57 reps, 10 models)
# ---------------------------------------------------------------
run_tcga() {
    local OUT="${BASE_OUTPUT}/tcga"
    echo "--- TCGA (3 datasets, 57 reps) ---"
    submit tcga_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py tcga --n-folds 5 --augment --output ${OUT}/" \
        256G 23:59:00
    echo ""
}

# ---------------------------------------------------------------
# TCGA with fine-tuning (Geneformer/scFoundation)
# ---------------------------------------------------------------
run_tcga_ft() {
    local OUT="${BASE_OUTPUT}/tcga_ft"
    echo "--- TCGA Fine-tuned (3 datasets, GPU) ---"
    submit tcga_ft_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py tcga --n-folds 5 --augment --finetune --output ${OUT}/" \
        256G 23:59:00 gpu
    echo ""
}

# ---------------------------------------------------------------
# DTA baseline survey (individual rep scores with CI metric)
# Uses dta_baseline_survey.py for per-rep CI/R²/Spearman
# ---------------------------------------------------------------
run_dta() {
    local OUT="${BASE_OUTPUT}/dta"
    echo "--- DTA Baselines (davis, kiba, bindingdb_kd) ---"
    # Davis — all reps
    submit dta_davis \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis.json" \
        64G 47:59:00 gpu
    # KIBA — subsampled to 30K
    submit dta_kiba \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --output ${OUT}/kiba.json" \
        64G 47:59:00 gpu
    # BindingDB_Kd
    submit dta_bindingdb \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py bindingdb_kd --n-folds 5 --max-pairs 30000 --output ${OUT}/bindingdb_kd.json" \
        64G 47:59:00 gpu
    echo ""
}

# ---------------------------------------------------------------
# DTA also via run_baseline_survey.py (uses 10 downstream models)
# ---------------------------------------------------------------
run_dta_universal() {
    local OUT="${BASE_OUTPUT}/dta_universal"
    echo "--- DTA Universal Survey (davis, kiba, bindingdb_kd via run_baseline_survey.py) ---"
    submit dta_universal \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py dta --n-folds 5 --augment --output ${OUT}/" \
        96G 47:59:00 gpu
    echo ""
}

# ---------------------------------------------------------------
# DTA cold-drug evaluation
# ---------------------------------------------------------------
run_dta_cold() {
    local OUT="${BASE_OUTPUT}/dta_cold"
    echo "--- DTA Cold-Drug (davis, kiba) ---"
    submit dta_davis_cold \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --cold-setting drug --output ${OUT}/davis_cold_drug.json" \
        64G 47:59:00 gpu
    submit dta_kiba_cold \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --cold-setting drug --output ${OUT}/kiba_cold_drug.json" \
        64G 47:59:00 gpu
    echo ""
}

# ---------------------------------------------------------------
# REACTION (3 datasets, 8 reps, 10 models)
# ---------------------------------------------------------------
run_reaction() {
    local OUT="${BASE_OUTPUT}/reaction"
    echo "--- Reaction Yield (3 datasets, GroupKFold) ---"
    submit rxn_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py reaction --n-folds 5 --augment --output ${OUT}/" \
        32G 06:00:00
    echo ""
}

# ---------------------------------------------------------------
# ANTIBODY (35 reps × 25 datasets, split into manageable jobs)
# ---------------------------------------------------------------
run_antibody() {
    local OUT="${BASE_OUTPUT}/antibody"
    echo "--- Antibody (35 reps × 25 datasets) ---"

    # Fast reps (CPU): onehot, blosum, aac, paratope
    submit ab_fast_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps onehot,blosum,aac,paratope" \
        32G 02:00:00
    submit ab_fast_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps onehot,blosum,aac,paratope" \
        32G 02:00:00

    # LM reps (GPU): 25 reps including Phase 4 additions
    submit ab_lm_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr" \
        96G 23:59:00 gpu
    submit ab_lm_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr" \
        96G 23:59:00 gpu

    # Structural reps (GPU)
    submit ab_struct_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps structural" \
        64G 23:59:00 gpu
    submit ab_struct_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps structural" \
        64G 23:59:00 gpu

    # Mutation reps (GPU)
    submit ab_mut_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr" \
        96G 23:59:00 gpu
    submit ab_mut_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr" \
        96G 23:59:00 gpu

    # Variation reps (GPU)
    submit ab_var_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps esm2_entropy_top20,esm2_per_cdr,esm2_per_region" \
        96G 23:59:00 gpu
    submit ab_var_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps esm2_entropy_top20,esm2_per_cdr,esm2_per_region" \
        96G 23:59:00 gpu

    # MINT (antibody-specific Phase 4 rep, needs own job due to GPU+special weights)
    submit ab_mint_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps mint" \
        96G 23:59:00 gpu
    submit ab_mint_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps mint" \
        96G 23:59:00 gpu

    echo ""
    echo "  Total: 14 antibody jobs (4 fast CPU + 10 GPU)"
    echo ""
}

# ---------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------

case "$MODE" in
    molecular)
        run_molecular
        ;;
    molecular_ft)
        run_molecular_ft
        ;;
    tcga)
        run_tcga
        ;;
    tcga_ft)
        run_tcga_ft
        ;;
    dta)
        run_dta
        run_dta_universal
        ;;
    dta_cold)
        run_dta_cold
        ;;
    reaction)
        run_reaction
        ;;
    antibody)
        run_antibody
        ;;
    all)
        run_molecular
        run_tcga
        run_dta
        run_dta_universal
        run_reaction
        run_antibody
        echo "============================================"
        echo "Total: ~25 jobs across all domains"
        echo "  Molecular:  1 frozen (+ optional 1 FT GPU)"
        echo "  TCGA:       1 frozen (+ optional 1 FT GPU)"
        echo "  DTA:        3 baseline + 1 universal"
        echo "  Reaction:   1"
        echo "  Antibody:   14 (4 CPU + 10 GPU)"
        echo "============================================"
        ;;
    all_with_ft)
        run_molecular
        run_molecular_ft
        run_tcga
        run_tcga_ft
        run_dta
        run_dta_universal
        run_dta_cold
        run_reaction
        run_antibody
        echo "============================================"
        echo "Total: ~30 jobs (all domains + fine-tuning + cold-drug)"
        echo "============================================"
        ;;
    *)
        echo "Phase 5A: Re-run ALL baseline surveys"
        echo ""
        echo "Usage: bash submit_phase5_baselines.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all           All domains, frozen reps (~25 jobs)"
        echo "  all_with_ft   All domains + fine-tuning + cold-drug (~30 jobs)"
        echo "  molecular     8 datasets, ~35 reps, 10 models"
        echo "  molecular_ft  + ChemBERTa/MolFormer LoRA + SchNet E2E (GPU)"
        echo "  tcga          3 datasets, 57 reps, 10 models"
        echo "  tcga_ft       + Geneformer/scFoundation FT (GPU)"
        echo "  dta           davis/kiba/bindingdb baselines + universal"
        echo "  dta_cold      Cold-drug splits (davis, kiba)"
        echo "  reaction      BH/Suzuki/BH-Doyle, GroupKFold, 8 reps"
        echo "  antibody      35 reps × 25 datasets (14 jobs)"
        echo ""
        echo "Output: ${BASE_OUTPUT}/<domain>/<dataset>.json"
        echo ""
        echo "After baselines complete, run:"
        echo "  bash tests/slurm_scripts/submit_phase5_hybrids.sh"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${BASE_OUTPUT}/"
