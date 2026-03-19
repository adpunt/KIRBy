#!/bin/bash
# Consolidated resubmission for ALL missing survey data.
# One job per missing experiment — no redundant reruns.
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_missing_surveys.sh           # everything
#   bash tests/slurm_scripts/submit_missing_surveys.sh molecular # one section
#   bash tests/slurm_scripts/submit_missing_surveys.sh antibody
#   bash tests/slurm_scripts/submit_missing_surveys.sh dta
#   bash tests/slurm_scripts/submit_missing_surveys.sh opig
#   bash tests/slurm_scripts/submit_missing_surveys.sh tcga
#   bash tests/slurm_scripts/submit_missing_surveys.sh nucleotide
#   bash tests/slurm_scripts/submit_missing_surveys.sh geometry
#   bash tests/slurm_scripts/submit_missing_surveys.sh forward_compat

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# All antibody reps
ALL_AB_REPS="onehot,blosum,aac,paratope,esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr,structural,mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr,esm2_entropy_top20,esm2_per_cdr,esm2_per_region,mint,aido_protein,xtrimopglm"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}
    local wall=${4:-47:59:00}
    local partition=${5:-long}
    local gpu=${6:-gpu}

    local gpu_flag=""
    if [ "$gpu" = "gpu" ]; then
        gpu_flag="--gres=gpu:1"
    fi

    J=$(sbatch \
        --job-name="miss_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition="${partition}" \
        --mem="${mem}" \
        ${gpu_flag} \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (mem=${mem}, ${wall}, ${partition}${gpu_flag:+, GPU})"
}

echo "============================================"
echo "Missing Survey Resubmission"
echo "============================================"
echo ""

# ---- MOLECULAR: QM9 + 4x ADME (5 datasets, frozen reps) ----
run_molecular() {
    local OUT="tests/results/phase5_baselines/molecular"
    echo "--- Molecular: QM9 + 4x ADME (5 missing datasets) ---"
    # Run each dataset as a separate job to avoid timeout
    for ds in qm9 adme_hlm adme_rlm adme_mdr1 adme_solubility; do
        submit "mol_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/"
    done
    echo ""
}

# ---- MOLECULAR FT: All 8 datasets with fine-tuned reps ----
run_molecular_ft() {
    local OUT="tests/results/phase5_baselines/molecular_ft"
    echo "--- Molecular FT: 8 datasets with fine-tuned reps ---"
    submit mol_ft \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py molecular --n-folds 5 --augment --finetune --output ${OUT}/"
    echo ""
}

# ---- DTA: davis + bindingdb_kd (kiba already running) ----
run_dta() {
    local OUT="tests/results/phase5_baselines/dta"
    local OUT2="tests/results/phase5_baselines/dta_universal"
    local OUT3="tests/results/phase5_baselines/dta_cold"
    echo "--- DTA: davis + bindingdb (dedicated + universal + cold) ---"
    submit dta_davis \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis.json"
    submit dta_bindingdb \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py bindingdb_kd --n-folds 5 --max-pairs 30000 --output ${OUT}/bindingdb_kd.json"
    submit dta_universal \
        "mkdir -p ${OUT2} && python tests/run_baseline_survey.py dta --n-folds 5 --augment --output ${OUT2}/"
    submit dta_cold_davis \
        "mkdir -p ${OUT3} && python tests/dta_baseline_survey.py davis --n-folds 5 --cold-setting drug --output ${OUT3}/davis_cold_drug.json"
    submit dta_cold_kiba \
        "mkdir -p ${OUT3} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --cold-setting drug --output ${OUT3}/kiba_cold_drug.json"
    echo ""
}

# ---- ANTIBODY DMS: 8 datasets that only have 4 reps, need full 40-rep ----
run_antibody_dms() {
    local OUT="tests/results/phase5_baselines/antibody"
    echo "--- Antibody DMS: 8 datasets needing full 40-rep survey ---"
    # These 8 DMS datasets only have 4 fast reps — run full suite
    submit ab_dms_full \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py flab_binding_phillips_h3,flab_binding_adams,flab_expression_adams,abagym_HER2,abagym_VEGF,abagym_EGFR,abagym_lysozyme,abagym_Ang2 --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"
    echo ""
}

# ---- OPIG CLF: full reps for multiclass + humanness ----
run_opig_clf() {
    local OUT="tests/results/opig/clf"
    echo "--- OPIG CLF: full 40-rep survey for clf datasets ---"
    submit opig_clf_full \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_clf --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"
    echo ""
}

# ---- OPIG DDG: all 3 Graphinity datasets ----
run_opig_ddg() {
    local OUT="tests/results/opig/ddg"
    echo "--- OPIG DDG: 3 Graphinity synthetic DDG datasets ---"
    submit opig_ddg \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py graphinity_foldx,graphinity_rosetta,graphinity_augmented --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"
    echo ""
}

# ---- OPIG GINKGO: purity only (rest complete at 33 reps) ----
run_opig_ginkgo_purity() {
    local OUT="tests/results/opig/ginkgo"
    echo "--- OPIG Ginkgo: purity needs full reps ---"
    submit opig_ginkgo_purity \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ginkgo_purity --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"
    echo ""
}

# ---- TCGA FT: 3 datasets with fine-tuned reps ----
run_tcga_ft() {
    local OUT="tests/results/phase5_baselines/tcga_ft"
    echo "--- TCGA FT: 3 datasets with fine-tuned reps ---"
    submit tcga_ft \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py tcga --n-folds 5 --augment --finetune --output ${OUT}/"
    echo ""
}

# ---- NUCLEOTIDE: all 22 datasets ----
run_nucleotide() {
    local OUT="tests/results/nucleotide_survey"
    echo "--- Nucleotide: 22 datasets (classical reps, no GPU) ---"
    # Split into chunks to avoid timeout
    # NT Revised (18 tasks)
    for ds in nt_promoter_all nt_promoter_tata nt_promoter_no_tata nt_enhancers nt_enhancers_types nt_splice_sites_all nt_splice_sites_acceptors nt_splice_sites_donors nt_H2AFZ; do
        submit "nuc_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --output ${OUT}/" \
            128G 11:59:00 medium nogpu
    done
    for ds in nt_H3K27ac nt_H3K27me3 nt_H3K36me3 nt_H3K4me1 nt_H3K4me2 nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_H4K20me1; do
        submit "nuc_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --output ${OUT}/" \
            128G 11:59:00 medium nogpu
    done
    # Genomic Benchmarks (4 tasks)
    for ds in gb_human_nontata_promoters gb_human_enhancers_cohn gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm; do
        submit "nuc_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --output ${OUT}/" \
            128G 11:59:00 medium nogpu
    done
    echo ""
}

# ---- GEOMETRY SURVEY ----
run_geometry() {
    local OUT="tests/results/geometry_survey"
    local BASE_REPS="esm2,ablang2,prot_t5,esm3,mint,saprot,antiberty,blosum,aac"
    echo "--- Geometry: inter-family + ginkgo datasets ---"
    # Inter-family datasets
    for ds in flab_thermo_tresanco flab_pk_jain_fcrnrt flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain abcov_kd; do
        submit "geo_${ds}" \
            "mkdir -p ${OUT} && python tests/experiments/structural_geometry_survey.py ${ds} --base-reps ${BASE_REPS} --models ridge,rf --output ${OUT}/"
    done
    # Ginkgo datasets
    for ds in ginkgo_hic ginkgo_tm2 ginkgo_titer ginkgo_pr_cho ginkgo_acsins ginkgo_purity ginkgo_aggregation ginkgo_colloidal ginkgo_heparin; do
        submit "geo_gk_${ds}" \
            "mkdir -p ${OUT} && python tests/experiments/structural_geometry_survey.py ${ds} --base-reps ${BASE_REPS} --models ridge,rf --output ${OUT}/"
    done
    echo ""
}

# ---- FORWARD COMPAT ----
# Skip exp3 (MolE/MiniMol/GP-MoLFormer — not installed, repeated failures)
# Skip exp5 (Boltz-2 — removed from DTA, too slow)
# Exp8 (Geneformer V2) already validated in Phase 5 baselines
run_forward_compat() {
    local OUT="tests/results/forward_compat"
    echo "--- Forward Compat: exp1,2,4,6,7,9 (skip exp3/5/8) ---"

    # Exp 1: MINT — antibody baselines (DTA covered by p5 baselines)
    local OUT1="${OUT}/exp1_mint"
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "fc_mint_${ds}" \
            "mkdir -p ${OUT1} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT1}/ --n-folds 5 --augment --reps mint" \
            128G 23:59:00
    done

    # Exp 2: SaProt — antibody baselines
    local OUT2="${OUT}/exp2_saprot"
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "fc_sap_${ds}" \
            "mkdir -p ${OUT2} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT2}/ --n-folds 5 --augment --reps saprot,saprot_cdr" \
            128G 23:59:00
    done

    # Exp 4: IgT5 / AbRoBERTa — antibody baselines
    local OUT4="${OUT}/exp4_ablm"
    for ds in flab_expression_koenig flab_binding_koenig flab_binding_phillips_h3 ddg_combined flab_thermo_tresanco abcov_kd; do
        submit "fc_ablm_${ds}" \
            "mkdir -p ${OUT4} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT4}/ --n-folds 5 --augment --reps igt5,abroberta,abroberta_cdr" \
            128G 23:59:00
    done

    # Exp 6: xTrimoPGLM — antibody baselines (DTA covered by p5)
    local OUT6="${OUT}/exp6_xtrimopglm"
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "fc_xtri_${ds}" \
            "mkdir -p ${OUT6} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT6}/ --n-folds 5 --augment --reps xtrimopglm" \
            128G 23:59:00
    done

    # Exp 7: ReactionT5
    local OUT7="${OUT}/exp7_reactiont5"
    for ds in buchwald_hartwig suzuki bh_doyle_plates; do
        submit "fc_rxnt5_${ds}" \
            "mkdir -p ${OUT7} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT7}/" \
            128G 23:59:00
    done

    # Exp 9: AIDO.Protein — antibody only (DTA has enable_gqa error)
    local OUT9="${OUT}/exp9_aido_protein"
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "fc_aido_${ds}" \
            "mkdir -p ${OUT9} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT9}/ --n-folds 5 --augment --reps aido_protein" \
            256G 47:59:00
    done
    echo ""
}

case "$MODE" in
    molecular)      run_molecular ;;
    molecular_ft)   run_molecular_ft ;;
    dta)            run_dta ;;
    antibody)       run_antibody_dms ;;
    opig)           run_opig_clf; run_opig_ddg; run_opig_ginkgo_purity ;;
    tcga)           run_tcga_ft ;;
    nucleotide)     run_nucleotide ;;
    geometry)       run_geometry ;;
    forward_compat) run_forward_compat ;;
    all)
        run_molecular
        run_molecular_ft
        run_dta
        run_antibody_dms
        run_opig_clf
        run_opig_ddg
        run_opig_ginkgo_purity
        run_tcga_ft
        run_nucleotide
        run_geometry
        run_forward_compat
        echo "============================================"
        echo "Total: ~50 jobs submitted"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash submit_missing_surveys.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all            Everything missing"
        echo "  molecular      QM9 + 4x ADME (5 jobs)"
        echo "  molecular_ft   All 8 molecular FT (1 job)"
        echo "  dta            davis + bindingdb + universal + cold (5 jobs)"
        echo "  antibody       8 DMS datasets full reps (1 job)"
        echo "  opig           clf + ddg + ginkgo_purity (3 jobs)"
        echo "  tcga           TCGA FT (1 job)"
        echo "  nucleotide     22 datasets (22 jobs)"
        echo "  geometry       inter-family + ginkgo (15 jobs)"
        echo "  forward_compat Forward compat (deferred)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
