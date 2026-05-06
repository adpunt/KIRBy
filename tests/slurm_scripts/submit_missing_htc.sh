#!/bin/bash
# HTC resubmission — GPU jobs only. Per-dataset. --skip-existing.
# Run from HTC login node: bash tests/slurm_scripts/submit_missing_htc.sh [MODE]
#
# Partition strategy (based on actual elapsed times from completed jobs):
#   short (12h):  small antibody, inter-family, TCGA, ginkgo, ddg_combined
#   medium (48h): large DMS, molecular, DTA, abagym_all
#
# All jobs use --skip-existing: only computes missing reps, merges into existing JSON.

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

ALL_AB_REPS="onehot,blosum,aac,paratope,esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr,structural,mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr,esm2_entropy_top20,esm2_per_cdr,esm2_per_region,mint,aido_protein,xtrimopglm"

submit() {
    local name=$1
    local cmd=$2
    local mem=$3
    local wall=$4
    local partition=$5

    J=$(sbatch \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition="${partition}" \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (${partition}, ${mem}, ${wall}, GPU)"
}

echo "============================================"
echo "HTC Missing Surveys (GPU, per-dataset)"
echo "============================================"
echo ""

# ── Molecular ──
# logd took 11h/17reps → est ~24h for 43 reps. Others 1-2h → ~6h.
# With --skip-existing (17 reps done), remaining ~26 reps → est ~16h logd, ~4h others.
run_molecular() {
    local OUT="tests/results/phase5_baselines/molecular"
    echo "--- Molecular (8 datasets) ---"
    # logd: largest, medium partition
    submit "mol_logd" \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py logd --n-folds 5 --augment --skip-existing --output ${OUT}/" \
        256G 1-23:59:00 medium
    # caco2/herg: have 17 reps, need ~26 more, ~6-8h
    for ds in caco2 herg; do
        submit "mol_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --skip-existing --output ${OUT}/" \
            256G 11:59:00 short
    done
    # qm9 + ADME: no existing data, full run, ~6-12h each
    for ds in qm9 adme_hlm adme_rlm adme_mdr1 adme_solubility; do
        submit "mol_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --skip-existing --output ${OUT}/" \
            256G 1-23:59:00 medium
    done
    echo ""
}

# ── TCGA ──
# luad 1.6h, brca_pam50 1.8h, survival 13min — all with 36 reps.
# 25 missing factory reps, est ~3-4h additional. Short fits.
run_tcga() {
    local OUT="tests/results/phase5_baselines/tcga"
    echo "--- TCGA (3 datasets, short) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        submit "tcga_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --skip-existing --output ${OUT}/" \
            256G 11:59:00 short
    done
    echo ""
}

# ── TCGA FT ──
# Fine-tuning = GPU training per fold. Est ~6-12h per dataset.
run_tcga_ft() {
    local OUT="tests/results/phase5_baselines/tcga_ft"
    echo "--- TCGA FT (3 datasets, medium) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        submit "tcgaft_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --skip-existing --finetune --output ${OUT}/" \
            256G 1-23:59:00 medium
    done
    echo ""
}

# ── Antibody DMS ──
# Large DMS (koenig 14-25h at 23 reps → ~40h at 42 reps, medium)
# Small DMS (HER2 4min at 4 reps → ~1-3h at 42, short)
run_antibody_dms() {
    local OUT="tests/results/phase5_baselines/antibody"
    echo "--- Antibody DMS (11 datasets) ---"
    # Large: need medium
    for ds in flab_binding_koenig flab_expression_koenig \
              flab_binding_phillips_h3 flab_binding_adams flab_expression_adams; do
        submit "abdms_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            256G 1-23:59:00 medium
    done
    # Small AbAgym (n<700): short
    for ds in abagym_HER2 abagym_VEGF abagym_EGFR abagym_lysozyme abagym_Ang2; do
        submit "abdms_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            128G 11:59:00 short
    done
    # abagym_all (n=2247, 35 reps done, ~7 missing → ~3h, but be safe)
    submit "abdms_abagym_all" \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py abagym_all --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
        256G 11:59:00 short
    echo ""
}

# ── Antibody Inter-Family ──
# All small (n=130-672). Most have 37/42 reps → ~5 missing → <1h.
# tresanco (n=672, 34 reps) and ddg_combined_full (1536, 33 reps) slightly longer.
run_antibody_inter() {
    local OUT="tests/results/phase5_baselines/antibody"
    echo "--- Antibody Inter-Family (14 datasets, short) ---"
    for ds in flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt \
              flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain \
              abcov_kd flab_aggregation_jain flab_immunogenicity_marks \
              flab_polyreact_jain_bvp flab_thermo_hutchinson flab_aggregation_jain_sas \
              ddg_combined ddg_combined_full; do
        submit "abint_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            128G 11:59:00 short
    done
    echo ""
}

# ── Antibody CLF ──
# tz_her2 (n=5000, 37 reps done → ~5 missing → ~2h)
# humanness (small, 1 rep done → ~41 missing → ~6-8h)
run_antibody_clf() {
    local OUT="tests/results/opig/clf"
    echo "--- Antibody CLF (4 datasets, short) ---"
    for ds in tz_her2_binary tz_her2_multiclass humanness_species humanness_therapeutic; do
        submit "abclf_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            256G 11:59:00 short
    done
    echo ""
}

# ── Ginkgo ──
# Small datasets (n<250). 34/42 reps done → ~8 missing → ~1-2h.
# ginkgo_purity 4/42 → ~38 missing → ~4-6h.
run_ginkgo() {
    local OUT="tests/results/opig/ginkgo"
    echo "--- Ginkgo (9 datasets, short) ---"
    for ds in ginkgo_hic ginkgo_tm2 ginkgo_titer ginkgo_pr_cho ginkgo_acsins \
              ginkgo_purity ginkgo_aggregation ginkgo_colloidal ginkgo_heparin; do
        submit "ginkgo_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            128G 11:59:00 short
    done
    echo ""
}

# ── Graphinity DDG ──
# No existing data. foldx=50K samples → could be long. rosetta=21K. augmented=mixed.
run_ddg_synth() {
    local OUT="tests/results/opig/ddg"
    echo "--- Graphinity DDG (3 datasets, medium) ---"
    for ds in graphinity_foldx graphinity_rosetta graphinity_augmented; do
        submit "ddg_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            256G 1-23:59:00 medium
    done
    echo ""
}

# ── DTA ──
# No existing Phase 5 data. Boltz-2 removed. Est 12-24h per dataset.
run_dta() {
    local OUT="tests/results/phase5_baselines/dta"
    local OUT2="tests/results/phase5_baselines/dta_cold"
    echo "--- DTA (5 datasets, medium) ---"
    for ds in davis kiba bindingdb_kd; do
        submit "dta_${ds}" \
            "mkdir -p ${OUT} && python tests/dta_baseline_survey.py ${ds} --n-folds 5 --skip-existing --output ${OUT}/${ds}.json" \
            256G 1-23:59:00 medium
    done
    submit "dta_davis_cold" \
        "mkdir -p ${OUT2} && python tests/dta_baseline_survey.py davis --n-folds 5 --cold-setting drug --output ${OUT2}/davis_cold_drug.json" \
        256G 1-23:59:00 medium
    submit "dta_kiba_cold" \
        "mkdir -p ${OUT2} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --cold-setting drug --output ${OUT2}/kiba_cold_drug.json" \
        256G 1-23:59:00 medium
    echo ""
}

# ── Forward Compat (GPU jobs — fc_rxnt5 is on ARC) ──
# Single-rep or 2-3 rep experiments. Small work per job.
run_forward_compat() {
    local OUT="tests/results/forward_compat"
    echo "--- Forward Compat (short, 128G each) ---"
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "fc_mint_${ds}" \
            "mkdir -p ${OUT}/exp1_mint && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp1_mint/ --n-folds 5 --augment --skip-existing --reps mint" \
            128G 11:59:00 short
    done
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "fc_sap_${ds}" \
            "mkdir -p ${OUT}/exp2_saprot && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp2_saprot/ --n-folds 5 --augment --skip-existing --reps saprot,saprot_cdr" \
            128G 11:59:00 short
    done
    for ds in flab_expression_koenig flab_binding_koenig flab_binding_phillips_h3 \
              ddg_combined flab_thermo_tresanco abcov_kd; do
        submit "fc_ablm_${ds}" \
            "mkdir -p ${OUT}/exp4_ablm && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp4_ablm/ --n-folds 5 --augment --skip-existing --reps igt5,abroberta,abroberta_cdr" \
            128G 11:59:00 short
    done
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "fc_xtri_${ds}" \
            "mkdir -p ${OUT}/exp6_xtrimopglm && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp6_xtrimopglm/ --n-folds 5 --augment --skip-existing --reps xtrimopglm" \
            128G 11:59:00 short
    done
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "fc_aido_${ds}" \
            "mkdir -p ${OUT}/exp9_aido_protein && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp9_aido_protein/ --n-folds 5 --augment --skip-existing --reps aido_protein" \
            256G 11:59:00 short
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
        echo "  Run on HTC login node."
        echo ""
        echo "  short (12h): tcga, antibody_inter, antibody_clf, ginkgo, mol caco2/herg, forward_compat"
        echo "  medium (48h): molecular (logd/qm9/adme), antibody_dms (large), tcga_ft, dta, ddg_synth"
        echo ""
        echo "  molecular       8 jobs (2 short + 6 medium)"
        echo "  tcga            3 jobs (short)"
        echo "  tcga_ft         3 jobs (medium)"
        echo "  antibody_dms    11 jobs (5 medium + 5 short + 1 short)"
        echo "  antibody_inter  14 jobs (short)"
        echo "  antibody_clf    4 jobs (short)"
        echo "  antibody        29 jobs total"
        echo "  ginkgo          9 jobs (short)"
        echo "  ddg_synth       3 jobs (medium)"
        echo "  dta             5 jobs (medium)"
        echo "  forward_compat  ~20 jobs (short)"
        echo "  all             ~100 jobs"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
