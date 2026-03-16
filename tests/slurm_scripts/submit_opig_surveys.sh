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

# All antibody reps — one list, no splitting
ALL_REPS="onehot,blosum,aac,paratope,esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr,structural,mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr,esm2_entropy_top20,esm2_per_cdr,esm2_per_region,mint,aido_protein,xtrimopglm"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-128G}

    J=$(sbatch \
        --job-name="opig_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=47:59:00 --partition=long \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, 48h, GPU)"
}

echo "============================================"
echo "OPIG Experiments"
echo "============================================"

# --- B1+B3: Antibody Classification (4 datasets × 42 reps) ---
run_clf() {
    local OUT="${OUTPUT}/clf"
    echo "--- B1+B3: Antibody Classification (tz_her2, humanness) ---"

    submit clf \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_clf --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_REPS}" \
        256G

    echo "  Total: 1 clf job"
    echo ""
}

# --- B5: Ginkgo Developability (9 datasets × 42 reps) ---
run_ginkgo() {
    local OUT="${OUTPUT}/ginkgo"
    echo "--- B5: Ginkgo Developability (9 properties, 94-242 samples) ---"

    submit ginkgo \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ginkgo --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_REPS}" \
        256G

    echo "  Total: 1 ginkgo job"
    echo ""
}

# --- B2: Synthetic DDG (5 datasets × 42 reps) ---
run_ddg() {
    local OUT="${OUTPUT}/ddg"
    echo "--- B2: Graphinity DDG + existing DDG (5 datasets) ---"

    submit ddg \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_ddg --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_REPS}" \
        256G

    echo "  Total: 1 ddg job"
    echo ""
}

# --- C2: Covariate Shift (3 splits × 36 reps × 3 models × 3 datasets) ---
run_covariate() {
    local OUT="${OUTPUT}/covariate_shift"
    echo "--- C2: Covariate Shift Robustness ---"

    submit covariate \
        "mkdir -p ${OUT} && python tests/experiments/covariate_shift.py --datasets logd,caco2,herg --reps all --n-folds 5 --output ${OUT}" \
        256G

    echo "  Total: 1 covariate shift job"
    echo ""
}

# --- C4: Imrie Transfer Learning (pretrain then eval) ---
run_imrie() {
    echo "--- C4: Imrie Transfer Learning ---"

    submit imrie_pretrain \
        "python -m kirby.models.pretrain_chembl_dnn --rep ecfp4,maccs,pdv,mol2vec,mordred,avalon --n-compounds 500000 --max-targets 200 --hidden-sizes 256,128,64 --output-dir /data/stat-cadd/scat9264/kirby_data/chembl_pretrain/ --data-dir /data/stat-cadd/scat9264/kirby_data/chembl/ --epochs 200 --patience 30" \
        256G

    echo ""
    echo "  After pretrain completes, manually submit:"
    echo "    sbatch tests/slurm_scripts/imrie_eval.sh"
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
        echo "Total: 4 survey jobs + 1 pretrain"
        echo ""
        echo "After pretrain completes, manually submit:"
        echo "  sbatch tests/slurm_scripts/imrie_eval.sh"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_opig_surveys.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all        Everything below (5 jobs)"
        echo "  clf        B1+B3: tz_her2 + humanness classification (1 job)"
        echo "  ginkgo     B5: 9 Ginkgo developability properties (1 job)"
        echo "  ddg        B2: Graphinity + existing DDG (1 job)"
        echo "  covariate  C2: Covariate shift robustness (1 job)"
        echo "  imrie      C4: Imrie TL pretrain (1 job; eval manual after)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER | grep opig"
echo "Results: ${OUTPUT}/"
