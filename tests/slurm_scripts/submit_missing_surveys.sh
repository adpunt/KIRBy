#!/bin/bash
# Consolidated resubmission for ALL missing survey data.
# Per-dataset jobs with full rep lists. Results merge into existing JSONs.
#
# Based on audit_results.py gap report (2026-03-19): 1,299 rep×dataset gaps.
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_missing_surveys.sh           # everything
#   bash tests/slurm_scripts/submit_missing_surveys.sh molecular
#   bash tests/slurm_scripts/submit_missing_surveys.sh reaction
#   bash tests/slurm_scripts/submit_missing_surveys.sh tcga
#   bash tests/slurm_scripts/submit_missing_surveys.sh antibody_dms
#   bash tests/slurm_scripts/submit_missing_surveys.sh antibody_inter
#   bash tests/slurm_scripts/submit_missing_surveys.sh antibody_clf
#   bash tests/slurm_scripts/submit_missing_surveys.sh ginkgo
#   bash tests/slurm_scripts/submit_missing_surveys.sh ddg_synth
#   bash tests/slurm_scripts/submit_missing_surveys.sh dta
#   bash tests/slurm_scripts/submit_missing_surveys.sh molecular_ft
#   bash tests/slurm_scripts/submit_missing_surveys.sh tcga_ft
#   bash tests/slurm_scripts/submit_missing_surveys.sh nucleotide
#   bash tests/slurm_scripts/submit_missing_surveys.sh geometry
#   bash tests/slurm_scripts/submit_missing_surveys.sh forward_compat

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# Full antibody rep list (42 reps)
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
        --job-name="${name}" \
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
echo "Missing Survey Resubmission (per-dataset)"
echo "============================================"
echo ""

# ========================================================================
# MOLECULAR: 5 missing datasets + all 8 need full rep list rerun
# Audit: 17/43 reps on 3 datasets, 5 datasets have no file
# Strategy: rerun ALL 8 datasets with full preset (includes factory reps)
# ========================================================================
run_molecular() {
    local OUT="tests/results/phase5_baselines/molecular"
    echo "--- Molecular: 8 datasets × full rep list (per-dataset) ---"
    for ds in logd caco2 herg qm9 adme_hlm adme_rlm adme_mdr1 adme_solubility; do
        submit "mol_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/"
    done
    echo ""
}

# ========================================================================
# MOLECULAR FT: fine-tuned reps on all 8 datasets
# ========================================================================
run_molecular_ft() {
    local OUT="tests/results/phase5_baselines/molecular_ft"
    echo "--- Molecular FT: 8 datasets (per-dataset) ---"
    for ds in logd caco2 herg qm9 adme_hlm adme_rlm adme_mdr1 adme_solubility; do
        submit "molft_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --finetune --output ${OUT}/"
    done
    echo ""
}

# ========================================================================
# REACTION: 4 reps missing on all 3 datasets
# Audit: 10/14 reps, missing diff_ecfp4, diff_maccs, component_descriptors, ligand_descriptors
# Strategy: rerun with full preset (will add missing reps to existing JSONs)
# ========================================================================
run_reaction() {
    local OUT="tests/results/phase5_baselines/reaction"
    echo "--- Reaction: 3 datasets × full rep list ---"
    for ds in buchwald_hartwig suzuki bh_doyle_plates; do
        submit "rxn_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/"
    done
    echo ""
}

# ========================================================================
# TCGA: 25 factory/signature/Tier2 reps missing on all 3 datasets
# Strategy: rerun with full preset + --augment (factory reps need per-fold)
# ========================================================================
run_tcga() {
    local OUT="tests/results/phase5_baselines/tcga"
    echo "--- TCGA: 3 datasets × full rep list (incl factory) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        submit "tcga_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/"
    done
    echo ""
}

# ========================================================================
# TCGA FT: fine-tuned reps
# ========================================================================
run_tcga_ft() {
    local OUT="tests/results/phase5_baselines/tcga_ft"
    echo "--- TCGA FT: 3 datasets ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        submit "tcgaft_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --finetune --output ${OUT}/"
    done
    echo ""
}

# ========================================================================
# ANTIBODY DMS: 8 datasets with only 4 reps, 2 koenig with 23 reps, 1 complete
# Strategy: per-dataset with full 42-rep list
# ========================================================================
run_antibody_dms() {
    local OUT="tests/results/phase5_baselines/antibody"
    echo "--- Antibody DMS: 11 datasets × 42 reps (per-dataset) ---"
    for ds in flab_binding_koenig flab_expression_koenig \
              flab_binding_phillips_h3 flab_binding_adams flab_expression_adams \
              abagym_HER2 abagym_VEGF abagym_EGFR abagym_lysozyme abagym_Ang2 \
              abagym_all; do
        submit "abdms_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"
    done
    echo ""
}

# ========================================================================
# ANTIBODY INTER-FAMILY: mostly 37/42 but some gaps (abcov_kd 22/42, immunogenicity 23/42)
# Strategy: per-dataset with full 42-rep list
# ========================================================================
run_antibody_inter() {
    local OUT="tests/results/phase5_baselines/antibody"
    echo "--- Antibody Inter-Family: 14 datasets × 42 reps (per-dataset) ---"
    for ds in flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt \
              flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain \
              abcov_kd flab_aggregation_jain flab_immunogenicity_marks \
              flab_polyreact_jain_bvp flab_thermo_hutchinson flab_aggregation_jain_sas \
              ddg_combined ddg_combined_full; do
        submit "abint_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"
    done
    echo ""
}

# ========================================================================
# ANTIBODY CLF: tz_binary 37/42, rest 1/42 or missing
# ========================================================================
run_antibody_clf() {
    local OUT="tests/results/opig/clf"
    echo "--- Antibody CLF: 4 datasets × 42 reps (per-dataset) ---"
    for ds in tz_her2_binary tz_her2_multiclass humanness_species humanness_therapeutic; do
        submit "abclf_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"
    done
    echo ""
}

# ========================================================================
# GINKGO: 8 datasets at 34/42, purity at 4/42
# ========================================================================
run_ginkgo() {
    local OUT="tests/results/opig/ginkgo"
    echo "--- Ginkgo: 9 datasets × 42 reps (per-dataset) ---"
    for ds in ginkgo_hic ginkgo_tm2 ginkgo_titer ginkgo_pr_cho ginkgo_acsins \
              ginkgo_purity ginkgo_aggregation ginkgo_colloidal ginkgo_heparin; do
        submit "ginkgo_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"
    done
    echo ""
}

# ========================================================================
# GRAPHINITY DDG: all 3 datasets completely missing (bug fixed)
# ========================================================================
run_ddg_synth() {
    local OUT="tests/results/opig/ddg"
    echo "--- Graphinity DDG: 3 datasets × 42 reps (per-dataset) ---"
    for ds in graphinity_foldx graphinity_rosetta graphinity_augmented; do
        submit "ddg_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}"
    done
    echo ""
}

# ========================================================================
# DTA: all 3 datasets completely missing in phase5
# ========================================================================
run_dta() {
    local OUT="tests/results/phase5_baselines/dta"
    local OUT2="tests/results/phase5_baselines/dta_cold"
    echo "--- DTA: 3 datasets dedicated + 2 cold-drug (per-dataset) ---"
    for ds in davis kiba bindingdb_kd; do
        submit "dta_${ds}" \
            "mkdir -p ${OUT} && python tests/dta_baseline_survey.py ${ds} --n-folds 5 --output ${OUT}/${ds}.json" \
            256G 47:59:00
    done
    # Cold-drug splits
    submit "dta_davis_cold" \
        "mkdir -p ${OUT2} && python tests/dta_baseline_survey.py davis --n-folds 5 --cold-setting drug --output ${OUT2}/davis_cold_drug.json" \
        256G 47:59:00
    submit "dta_kiba_cold" \
        "mkdir -p ${OUT2} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --cold-setting drug --output ${OUT2}/kiba_cold_drug.json" \
        256G 47:59:00
    echo ""
}

# ========================================================================
# NUCLEOTIDE: all 22 datasets missing (pyarrow fix needed first)
# ========================================================================
run_nucleotide() {
    local OUT="tests/results/nucleotide_survey"
    echo "--- Nucleotide: 22 datasets (no GPU, medium partition) ---"
    for ds in nt_promoter_all nt_promoter_tata nt_promoter_no_tata \
              nt_enhancers nt_enhancers_types \
              nt_splice_sites_all nt_splice_sites_acceptors nt_splice_sites_donors \
              nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 \
              nt_H3K4me1 nt_H3K4me2 nt_H3K4me3 \
              nt_H3K9ac nt_H3K9me3 nt_H4K20me1 \
              gb_human_nontata_promoters gb_human_enhancers_cohn \
              gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm; do
        submit "nuc_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --output ${OUT}/" \
            128G 11:59:00 medium nogpu
    done
    echo ""
}

# ========================================================================
# GEOMETRY SURVEY
# ========================================================================
run_geometry() {
    local OUT="tests/results/geometry_survey"
    local BASE_REPS="esm2,ablang2,prot_t5,esm3,mint,saprot,antiberty,blosum,aac"
    echo "--- Geometry: inter-family + ginkgo (per-dataset) ---"
    for ds in flab_thermo_tresanco flab_pk_jain_fcrnrt flab_polyreact_jain_cic \
              flab_aggregation_jain_hicrt flab_thermo_jain abcov_kd \
              ginkgo_hic ginkgo_tm2 ginkgo_titer ginkgo_pr_cho ginkgo_acsins \
              ginkgo_purity ginkgo_aggregation ginkgo_colloidal ginkgo_heparin; do
        submit "geo_${ds}" \
            "mkdir -p ${OUT} && python tests/experiments/structural_geometry_survey.py ${ds} --base-reps ${BASE_REPS} --models ridge,rf --output ${OUT}/"
    done
    echo ""
}

# ========================================================================
# FORWARD COMPAT: targeted per-rep experiments
# Skip exp3 (MolE/MiniMol/GP-MoLFormer — install broken)
# Skip exp5 (Boltz-2 — removed)
# Skip exp8 (Geneformer V2 — already validated in p5 baselines)
# ========================================================================
run_forward_compat() {
    local OUT="tests/results/forward_compat"
    echo "--- Forward Compat: exp1,2,4,6,7,9 ---"

    # Exp 1: MINT
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "fc_mint_${ds}" \
            "mkdir -p ${OUT}/exp1_mint && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp1_mint/ --n-folds 5 --augment --reps mint" \
            128G 23:59:00
    done

    # Exp 2: SaProt
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "fc_sap_${ds}" \
            "mkdir -p ${OUT}/exp2_saprot && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp2_saprot/ --n-folds 5 --augment --reps saprot,saprot_cdr" \
            128G 23:59:00
    done

    # Exp 4: IgT5 / AbRoBERTa
    for ds in flab_expression_koenig flab_binding_koenig flab_binding_phillips_h3 \
              ddg_combined flab_thermo_tresanco abcov_kd; do
        submit "fc_ablm_${ds}" \
            "mkdir -p ${OUT}/exp4_ablm && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp4_ablm/ --n-folds 5 --augment --reps igt5,abroberta,abroberta_cdr" \
            128G 23:59:00
    done

    # Exp 6: xTrimoPGLM
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "fc_xtri_${ds}" \
            "mkdir -p ${OUT}/exp6_xtrimopglm && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp6_xtrimopglm/ --n-folds 5 --augment --reps xtrimopglm" \
            128G 23:59:00
    done

    # Exp 7: ReactionT5
    for ds in buchwald_hartwig suzuki bh_doyle_plates; do
        submit "fc_rxnt5_${ds}" \
            "mkdir -p ${OUT}/exp7_reactiont5 && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/exp7_reactiont5/" \
            128G 23:59:00
    done

    # Exp 9: AIDO.Protein
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "fc_aido_${ds}" \
            "mkdir -p ${OUT}/exp9_aido_protein && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp9_aido_protein/ --n-folds 5 --augment --reps aido_protein" \
            256G 47:59:00
    done
    echo ""
}

# ========================================================================
# DISPATCH
# ========================================================================

case "$MODE" in
    molecular)      run_molecular ;;
    molecular_ft)   run_molecular_ft ;;
    reaction)       run_reaction ;;
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
    nucleotide)     run_nucleotide ;;
    geometry)       run_geometry ;;
    forward_compat) run_forward_compat ;;
    all)
        run_molecular
        run_molecular_ft
        run_reaction
        run_tcga
        run_tcga_ft
        run_antibody_dms
        run_antibody_inter
        run_antibody_clf
        run_ginkgo
        run_ddg_synth
        run_dta
        run_nucleotide
        run_geometry
        run_forward_compat
        ;;
    *)
        echo "Usage: bash submit_missing_surveys.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all             Everything (~120 jobs)"
        echo "  molecular       8 datasets × full reps (8 jobs)"
        echo "  molecular_ft    8 datasets × FT reps (8 jobs)"
        echo "  reaction        3 datasets × full reps (3 jobs)"
        echo "  tcga            3 datasets × full reps (3 jobs)"
        echo "  tcga_ft         3 datasets × FT reps (3 jobs)"
        echo "  antibody_dms    11 datasets × 42 reps (11 jobs)"
        echo "  antibody_inter  14 datasets × 42 reps (14 jobs)"
        echo "  antibody_clf    4 datasets × 42 reps (4 jobs)"
        echo "  antibody        All antibody (29 jobs)"
        echo "  ginkgo          9 datasets × 42 reps (9 jobs)"
        echo "  ddg_synth       3 Graphinity datasets (3 jobs)"
        echo "  opig            clf + ginkgo + ddg (16 jobs)"
        echo "  dta             3 datasets + 2 cold-drug (5 jobs)"
        echo "  nucleotide      22 datasets (22 jobs)"
        echo "  geometry        15 datasets (15 jobs)"
        echo "  forward_compat  6 experiments (~24 jobs)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
