#!/bin/bash
# Submit all OPIG experiment surveys as SLURM jobs.
# Run from login node: bash tests/slurm_scripts/submit_opig_surveys.sh
#
# Usage:
#   bash tests/slurm_scripts/submit_opig_surveys.sh           # everything
#   bash tests/slurm_scripts/submit_opig_surveys.sh clf        # B1+B3: classification
#   bash tests/slurm_scripts/submit_opig_surveys.sh ginkgo     # B5: developability
#   bash tests/slurm_scripts/submit_opig_surveys.sh ddg        # B2: synthetic DDG
#   bash tests/slurm_scripts/submit_opig_surveys.sh covariate  # C2: covariate shift
#   bash tests/slurm_scripts/submit_opig_surveys.sh imrie      # C4: pretrain + eval

set -euo pipefail

MODE=${1:-all}
OUTPUT="tests/results/opig"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# Rep groups (same as submit_antibody_survey.sh)
ALL_LM="esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr"
ALL_FAST="onehot,blosum,aac,paratope"
ALL_STRUCT="structural"
ALL_MUTATION="mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr"
ALL_VARIATION="esm2_entropy_top20,esm2_per_cdr,esm2_per_region"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-128G}
    local wall=${4:-47:59:00}
    local gpu=${5:-gpu}

    local gpu_flag=""
    if [ "$gpu" = "gpu" ]; then
        gpu_flag="--gres=gpu:1"
    fi

    J=$(sbatch \
        --job-name="opig_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        ${gpu_flag} \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, time=${wall}${gpu_flag:+, GPU})"
}

echo "============================================"
echo "OPIG Experiments"
echo "============================================"

# --- B1+B3: Antibody Classification (4 datasets × 40 reps) ---
run_clf() {
    local OUT="${OUTPUT}/clf"
    echo "--- B1+B3: Antibody Classification (tz_her2, humanness) ---"

    # Fast reps (CPU)
    submit clf_fast \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_clf --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_FAST}" \
        32G 04:00:00 nogpu

    # LM reps (GPU)
    submit clf_lm \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_clf --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_LM}" \
        128G 47:59:00

    # Structural
    submit clf_struct \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_clf --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_STRUCT}" \
        64G 12:00:00

    # Mutation
    submit clf_mut \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_clf --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_MUTATION}" \
        128G 23:59:00

    # Variation
    submit clf_var \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_clf --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_VARIATION}" \
        128G 23:59:00

    echo "  Total: 5 clf jobs"
    echo ""
}

# --- B5: Ginkgo Developability (9 datasets × 40 reps) ---
run_ginkgo() {
    local OUT="${OUTPUT}/ginkgo"
    echo "--- B5: Ginkgo Developability (9 properties, 94-242 samples) ---"

    # Fast reps (CPU) — small datasets, should be quick
    submit ginkgo_fast \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ginkgo --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_FAST}" \
        32G 02:00:00 nogpu

    # LM reps (GPU)
    submit ginkgo_lm \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ginkgo --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_LM}" \
        128G 23:59:00

    # Structural
    submit ginkgo_struct \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ginkgo --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_STRUCT}" \
        64G 06:00:00

    # Mutation (unlikely to help — no mutation metadata — but run for completeness)
    submit ginkgo_mut \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ginkgo --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_MUTATION}" \
        128G 12:00:00

    # Variation
    submit ginkgo_var \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ginkgo --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_VARIATION}" \
        128G 12:00:00

    echo "  Total: 5 ginkgo jobs"
    echo ""
}

# --- B2: Synthetic DDG (3+2 datasets × 40 reps) ---
run_ddg() {
    local OUT="${OUTPUT}/ddg"
    echo "--- B2: Graphinity DDG + existing DDG (5 datasets) ---"

    # Fast reps (CPU) — graphinity_foldx is 50K so not instant
    submit ddg_fast \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ddg --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_FAST}" \
        64G 06:00:00 nogpu

    # LM reps (GPU)
    submit ddg_lm \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ddg --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_LM}" \
        128G 47:59:00

    # Structural
    submit ddg_struct \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ddg --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_STRUCT}" \
        128G 23:59:00

    # Mutation (most relevant for DDG!)
    submit ddg_mut \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ddg --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_MUTATION}" \
        128G 47:59:00

    # Variation
    submit ddg_var \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ddg --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_VARIATION}" \
        128G 23:59:00

    echo "  Total: 5 ddg jobs"
    echo ""
}

# --- C2: Covariate Shift (3 splits × 36 reps × 3 models × 3 datasets) ---
run_covariate() {
    local OUT="${OUTPUT}/covariate_shift"
    echo "--- C2: Covariate Shift Robustness ---"

    submit covariate \
        "mkdir -p ${OUT} && python tests/experiments/covariate_shift.py --datasets logd,caco2,herg --reps all --n-folds 5 --output ${OUT}" \
        128G 47:59:00

    echo "  Total: 1 covariate shift job"
    echo ""
}

# --- C4: Imrie Transfer Learning (pretrain then eval) ---
run_imrie() {
    echo "--- C4: Imrie Transfer Learning ---"

    # Step 1: Pre-train (CPU-only, no GPU needed)
    echo "  Step 1: Pre-training DNN on ChEMBL (run first, ~2-4h)"
    echo "    sbatch tests/slurm_scripts/imrie_pretrain.sh"
    echo ""
    echo "  Step 2: After pretrain completes, run eval:"
    echo "    sbatch tests/slurm_scripts/imrie_eval.sh"
    echo ""

    # Submit pretrain only — user must submit eval after pretrain finishes
    submit imrie_pretrain \
        "python -m kirby.models.pretrain_chembl_dnn --rep ecfp4,maccs,pdv,mol2vec,mordred,avalon --n-compounds 500000 --max-targets 200 --hidden-sizes 256,128,64 --output-dir /data/stat-cadd/scat9264/kirby_data/chembl_pretrain/ --data-dir /data/stat-cadd/scat9264/kirby_data/chembl/ --epochs 200 --patience 30" \
        128G 04:00:00 nogpu

    echo "  ⚠  imrie_eval must be submitted MANUALLY after pretrain completes!"
    echo "  Total: 1 pretrain job (eval is manual)"
    echo ""
}

case "$MODE" in
    clf)        run_clf ;;
    ginkgo)     run_ginkgo ;;
    ddg)        run_ddg ;;
    covariate)  run_covariate ;;
    imrie)      run_imrie ;;
    all)
        run_clf
        run_ginkgo
        run_ddg
        run_covariate
        run_imrie
        echo "============================================"
        echo "Total: 16 survey jobs + 1 pretrain"
        echo ""
        echo "After pretrain completes, manually submit:"
        echo "  sbatch tests/slurm_scripts/imrie_eval.sh"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_opig_surveys.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all        Everything below (17 jobs)"
        echo "  clf        B1+B3: tz_her2 + humanness classification (5 jobs)"
        echo "  ginkgo     B5: 9 Ginkgo developability properties (5 jobs)"
        echo "  ddg        B2: Graphinity + existing DDG (5 jobs)"
        echo "  covariate  C2: Covariate shift robustness (1 job)"
        echo "  imrie      C4: Imrie TL pretrain (1 job; eval manual after)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER | grep opig"
echo "Results: ${OUTPUT}/"
