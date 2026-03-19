#!/bin/bash
# HTC resubmission — GPU jobs only.
# Run from HTC login node: bash tests/slurm_scripts/submit_missing_htc.sh [MODE]
#
# All jobs: long partition, GPU, 256G or 128G, 48h.

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

ALL_AB_REPS="onehot,blosum,aac,paratope,esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr,structural,mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr,esm2_entropy_top20,esm2_per_cdr,esm2_per_region,mint,aido_protein,xtrimopglm"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}

    J=$(sbatch \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=47:59:00 --partition=long \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (long, ${mem}, 48h, GPU)"
}

echo "============================================"
echo "HTC Missing Surveys (GPU jobs)"
echo "============================================"
echo ""

# ── Molecular: 8 datasets × full reps ──
run_molecular() {
    local OUT="tests/results/phase5_baselines/molecular"
    echo "--- Molecular (8 datasets, 256G each) ---"
    for ds in logd caco2 herg qm9 adme_hlm adme_rlm adme_mdr1 adme_solubility; do
        submit "mol_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --skip-existing --output ${OUT}/" \
            256G
    done
    echo ""
}

# ── TCGA: 3 datasets × full reps ──
run_tcga() {
    local OUT="tests/results/phase5_baselines/tcga"
    echo "--- TCGA (3 datasets, 256G each) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        submit "tcga_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --skip-existing --output ${OUT}/" \
            256G
    done
    echo ""
}

# ── TCGA FT: 3 datasets ──
run_tcga_ft() {
    local OUT="tests/results/phase5_baselines/tcga_ft"
    echo "--- TCGA FT (3 datasets, 256G each) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        submit "tcgaft_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --skip-existing --finetune --output ${OUT}/" \
            256G
    done
    echo ""
}

# ── Antibody DMS: 11 datasets × 42 reps ──
run_antibody_dms() {
    local OUT="tests/results/phase5_baselines/antibody"
    echo "--- Antibody DMS (11 datasets) ---"
    # Large DMS (n>4000)
    for ds in flab_binding_koenig flab_expression_koenig \
              flab_binding_phillips_h3 flab_binding_adams flab_expression_adams; do
        submit "abdms_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            256G
    done
    # Smaller DMS (n<700)
    for ds in abagym_HER2 abagym_VEGF abagym_EGFR abagym_lysozyme abagym_Ang2 abagym_all; do
        submit "abdms_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            128G
    done
    echo ""
}

# ── Antibody Inter-Family: 14 datasets × 42 reps ──
run_antibody_inter() {
    local OUT="tests/results/phase5_baselines/antibody"
    echo "--- Antibody Inter-Family (14 datasets, 128G each) ---"
    for ds in flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt \
              flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain \
              abcov_kd flab_aggregation_jain flab_immunogenicity_marks \
              flab_polyreact_jain_bvp flab_thermo_hutchinson flab_aggregation_jain_sas \
              ddg_combined ddg_combined_full; do
        submit "abint_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            128G
    done
    echo ""
}

# ── Antibody CLF: 4 datasets × 42 reps ──
run_antibody_clf() {
    local OUT="tests/results/opig/clf"
    echo "--- Antibody CLF (4 datasets) ---"
    for ds in tz_her2_binary tz_her2_multiclass; do
        submit "abclf_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            256G
    done
    for ds in humanness_species humanness_therapeutic; do
        submit "abclf_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            128G
    done
    echo ""
}

# ── Ginkgo: 9 datasets × 42 reps ──
run_ginkgo() {
    local OUT="tests/results/opig/ginkgo"
    echo "--- Ginkgo (9 datasets, 128G each) ---"
    for ds in ginkgo_hic ginkgo_tm2 ginkgo_titer ginkgo_pr_cho ginkgo_acsins \
              ginkgo_purity ginkgo_aggregation ginkgo_colloidal ginkgo_heparin; do
        submit "ginkgo_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            128G
    done
    echo ""
}

# ── Graphinity DDG: 3 datasets ──
run_ddg_synth() {
    local OUT="tests/results/opig/ddg"
    echo "--- Graphinity DDG (3 datasets, 256G each) ---"
    for ds in graphinity_foldx graphinity_rosetta graphinity_augmented; do
        submit "ddg_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            256G
    done
    echo ""
}

# ── DTA: 3 dedicated + 2 cold-drug ──
run_dta() {
    local OUT="tests/results/phase5_baselines/dta"
    local OUT2="tests/results/phase5_baselines/dta_cold"
    echo "--- DTA (5 jobs, 256G each) ---"
    for ds in davis kiba bindingdb_kd; do
        submit "dta_${ds}" \
            "mkdir -p ${OUT} && python tests/dta_baseline_survey.py ${ds} --n-folds 5 --skip-existing --output ${OUT}/${ds}.json" \
            256G
    done
    submit "dta_davis_cold" \
        "mkdir -p ${OUT2} && python tests/dta_baseline_survey.py davis --n-folds 5 --cold-setting drug --output ${OUT2}/davis_cold_drug.json" \
        256G
    submit "dta_kiba_cold" \
        "mkdir -p ${OUT2} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --cold-setting drug --output ${OUT2}/kiba_cold_drug.json" \
        256G
    echo ""
}

# ── Forward Compat: GPU jobs only (fc_rxnt5 goes on ARC) ──
run_forward_compat() {
    local OUT="tests/results/forward_compat"
    echo "--- Forward Compat (GPU jobs, 128G each) ---"
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "fc_mint_${ds}" \
            "mkdir -p ${OUT}/exp1_mint && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp1_mint/ --n-folds 5 --augment --skip-existing --reps mint" \
            128G
    done
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "fc_sap_${ds}" \
            "mkdir -p ${OUT}/exp2_saprot && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp2_saprot/ --n-folds 5 --augment --skip-existing --reps saprot,saprot_cdr" \
            128G
    done
    for ds in flab_expression_koenig flab_binding_koenig flab_binding_phillips_h3 \
              ddg_combined flab_thermo_tresanco abcov_kd; do
        submit "fc_ablm_${ds}" \
            "mkdir -p ${OUT}/exp4_ablm && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp4_ablm/ --n-folds 5 --augment --skip-existing --reps igt5,abroberta,abroberta_cdr" \
            128G
    done
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "fc_xtri_${ds}" \
            "mkdir -p ${OUT}/exp6_xtrimopglm && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp6_xtrimopglm/ --n-folds 5 --augment --skip-existing --reps xtrimopglm" \
            128G
    done
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "fc_aido_${ds}" \
            "mkdir -p ${OUT}/exp9_aido_protein && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp9_aido_protein/ --n-folds 5 --augment --skip-existing --reps aido_protein" \
            256G
    done
    echo ""
}

case "$MODE" in
    molecular)      run_molecular ;;
    tcga)           run_tcga ;;
    tcga_ft)        run_tcga_ft ;;
    antibody_dms)   run_antibody_dms ;;
    antibody_inter) run_antibody_inter ;;
    antibody_clf)   run_antibody_clf ;;
    antibody)       run_antibody_dms; run_antibody_inter; run_antibody_clf ;;
    ginkgo)         run_ginkgo ;;
    ddg_synth)      run_ddg_synth ;;
    opig)           run_antibody_clf; run_ginkgo; run_ddg_synth ;;
    dta)            run_dta ;;
    forward_compat) run_forward_compat ;;
    all)
        run_molecular
        run_tcga
        run_tcga_ft
        run_antibody_dms
        run_antibody_inter
        run_antibody_clf
        run_ginkgo
        run_ddg_synth
        run_dta
        run_forward_compat
        ;;
    *)
        echo "Usage: bash submit_missing_htc.sh [MODE]"
        echo "  Run on HTC login node. All jobs: long partition, GPU, 48h."
        echo ""
        echo "  molecular       8 GPU jobs (256G)"
        echo "  tcga            3 GPU jobs (256G)"
        echo "  tcga_ft         3 GPU jobs (256G)"
        echo "  antibody_dms    11 GPU jobs (128-256G)"
        echo "  antibody_inter  14 GPU jobs (128G)"
        echo "  antibody_clf    4 GPU jobs (128-256G)"
        echo "  antibody        29 GPU jobs"
        echo "  ginkgo          9 GPU jobs (128G)"
        echo "  ddg_synth       3 GPU jobs (256G)"
        echo "  opig            16 GPU jobs"
        echo "  dta             5 GPU jobs (256G)"
        echo "  forward_compat  ~21 GPU jobs (128-256G)"
        echo "  all             ~100 GPU jobs"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
