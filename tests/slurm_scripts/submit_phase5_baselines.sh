#!/bin/bash
# Phase 5A: Re-run ALL baseline surveys with corrected pipeline.
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
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# All antibody reps — one list
ALL_AB_REPS="onehot,blosum,aac,paratope,esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr,structural,mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr,esm2_entropy_top20,esm2_per_cdr,esm2_per_region,mint,aido_protein,xtrimopglm"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}
    local gpu=${4:-gpu}

    local gpu_flag=""
    if [ "$gpu" = "gpu" ]; then
        gpu_flag="--gres=gpu:1"
    fi

    J=$(sbatch \
        --job-name="p5_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=47:59:00 --partition=long \
        --mem="${mem}" \
        ${gpu_flag} \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, 48h${gpu_flag:+, GPU})"
}

echo "============================================"
echo "Phase 5A: Baseline Surveys (corrected pipeline)"
echo "============================================"
echo "  Output: ${BASE_OUTPUT}/"
echo ""

run_molecular() {
    local OUT="${BASE_OUTPUT}/molecular"
    echo "--- Molecular (8 datasets, ~35 reps) ---"
    submit mol_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py molecular --n-folds 5 --augment --output ${OUT}/"
    echo ""
}

run_molecular_ft() {
    local OUT="${BASE_OUTPUT}/molecular_ft"
    echo "--- Molecular Fine-tuned (8 datasets, GPU) ---"
    submit mol_ft_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py molecular --n-folds 5 --augment --finetune --output ${OUT}/"
    echo ""
}

run_tcga() {
    local OUT="${BASE_OUTPUT}/tcga"
    echo "--- TCGA (3 datasets, 57 reps) ---"
    submit tcga_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py tcga --n-folds 5 --augment --output ${OUT}/"
    echo ""
}

run_tcga_ft() {
    local OUT="${BASE_OUTPUT}/tcga_ft"
    echo "--- TCGA Fine-tuned (3 datasets, GPU) ---"
    submit tcga_ft_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py tcga --n-folds 5 --augment --finetune --output ${OUT}/"
    echo ""
}

run_dta() {
    local OUT="${BASE_OUTPUT}/dta"
    echo "--- DTA Baselines (davis, kiba, bindingdb_kd) ---"
    submit dta_davis \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis.json"
    submit dta_kiba \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --output ${OUT}/kiba.json"
    submit dta_bindingdb \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py bindingdb_kd --n-folds 5 --max-pairs 30000 --output ${OUT}/bindingdb_kd.json"
    echo ""
}

run_dta_universal() {
    local OUT="${BASE_OUTPUT}/dta_universal"
    echo "--- DTA Universal Survey ---"
    submit dta_universal \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py dta --n-folds 5 --augment --output ${OUT}/"
    echo ""
}

run_dta_cold() {
    local OUT="${BASE_OUTPUT}/dta_cold"
    echo "--- DTA Cold-Drug (davis, kiba) ---"
    submit dta_davis_cold \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --cold-setting drug --output ${OUT}/davis_cold_drug.json"
    submit dta_kiba_cold \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --cold-setting drug --output ${OUT}/kiba_cold_drug.json"
    echo ""
}

run_reaction() {
    local OUT="${BASE_OUTPUT}/reaction"
    echo "--- Reaction Yield (3 datasets) ---"
    submit rxn_baselines \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py reaction --n-folds 5 --augment --output ${OUT}/"
    echo ""
}

run_antibody() {
    local OUT="${BASE_OUTPUT}/antibody"
    echo "--- Antibody (42 reps × dms + inter) ---"

    submit ab_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"
    submit ab_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"

    echo "  Total: 2 antibody jobs"
    echo ""
}

case "$MODE" in
    molecular)    run_molecular ;;
    molecular_ft) run_molecular_ft ;;
    tcga)         run_tcga ;;
    tcga_ft)      run_tcga_ft ;;
    dta)          run_dta; run_dta_universal ;;
    dta_cold)     run_dta_cold ;;
    reaction)     run_reaction ;;
    antibody)     run_antibody ;;
    all)
        run_molecular
        run_tcga
        run_dta
        run_dta_universal
        run_reaction
        run_antibody
        echo "============================================"
        echo "Total: 10 jobs"
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
        echo "Total: 14 jobs (all domains + fine-tuning + cold-drug)"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash submit_phase5_baselines.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all           All domains, frozen reps (10 jobs)"
        echo "  all_with_ft   All domains + fine-tuning + cold-drug (14 jobs)"
        echo "  molecular     8 datasets, ~35 reps"
        echo "  molecular_ft  + ChemBERTa/MolFormer LoRA + SchNet E2E"
        echo "  tcga          3 datasets, 57 reps"
        echo "  tcga_ft       + Geneformer/scFoundation FT"
        echo "  dta           davis/kiba/bindingdb baselines + universal"
        echo "  dta_cold      Cold-drug splits (davis, kiba)"
        echo "  reaction      BH/Suzuki/BH-Doyle, 8 reps"
        echo "  antibody      42 reps × dms + inter (2 jobs)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${BASE_OUTPUT}/"
