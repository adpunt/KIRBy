#!/bin/bash
# Consolidated resubmission for ALL missing survey data.
# Per-dataset jobs with full rep lists.
#
# Based on audit_results.py gap report (2026-03-19): 1,299 rep×dataset gaps.
#
# Resource strategy (based on sinfo 2026-03-19):
#   GPU jobs → long partition (no time limit), 256G or 128G
#   CPU-only → short partition (12h limit, CPU nodes have 270-370G free, less contention)
#   Nucleotide → medium partition (48h, no GPU, some datasets are large)
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_missing_surveys.sh all
#   bash tests/slurm_scripts/submit_missing_surveys.sh molecular
#   bash tests/slurm_scripts/submit_missing_surveys.sh reaction
#   (etc — see bottom for full list)

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
    local mem=$3
    local wall=$4
    local partition=$5
    local gpu=$6

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
    echo "  $J  ${name}  (${partition}, ${mem}, ${wall}${gpu_flag:+, GPU})"
}

echo "============================================"
echo "Missing Survey Resubmission (per-dataset)"
echo "============================================"
echo ""

# ========================================================================
# MOLECULAR: all 8 datasets, full rep list (includes LMs → needs GPU)
# 17/43 reps on 3 datasets, 5 datasets no file. Per-dataset ~20-48h.
# ========================================================================
run_molecular() {
    local OUT="tests/results/phase5_baselines/molecular"
    echo "--- Molecular: 8 datasets × full reps (GPU, long, 48h) ---"
    for ds in logd caco2 herg qm9 adme_hlm adme_rlm adme_mdr1 adme_solubility; do
        submit "mol_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/" \
            256G 47:59:00 long gpu
    done
    echo ""
}

# ========================================================================
# MOLECULAR FT: SKIPPED — comparison data shows FT doesn't help molecular.
# See comparisons/molecular_finetune_comparison.json.
# ========================================================================
run_molecular_ft() {
    echo "--- Molecular FT: SKIPPED (comparison data shows FT does not help) ---"
    echo "  See: tests/results/comparisons/molecular_finetune_comparison.json"
    echo ""
}

# ========================================================================
# REACTION: 10/14 reps, 4 missing. No GPU needed (fingerprints + T5 cached).
# Short partition, 12h is plenty for 3 small datasets.
# ========================================================================
run_reaction() {
    local OUT="tests/results/phase5_baselines/reaction"
    echo "--- Reaction: 3 datasets × full reps (CPU, short, 12h) ---"
    for ds in buchwald_hartwig suzuki bh_doyle_plates; do
        submit "rxn_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/" \
            64G 11:59:00 short nogpu
    done
    echo ""
}

# ========================================================================
# TCGA: 36/61 reps — 25 factory/signature reps missing. Needs GPU for FMs.
# ========================================================================
run_tcga() {
    local OUT="tests/results/phase5_baselines/tcga"
    echo "--- TCGA: 3 datasets × full reps (GPU, long, 48h) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        submit "tcga_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/" \
            256G 47:59:00 long gpu
    done
    echo ""
}

# ========================================================================
# TCGA FT: Geneformer + scFoundation FT show real gains. GPU needed.
# ========================================================================
run_tcga_ft() {
    local OUT="tests/results/phase5_baselines/tcga_ft"
    echo "--- TCGA FT: 3 datasets (GPU, long, 48h) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
        submit "tcgaft_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --finetune --output ${OUT}/" \
            256G 47:59:00 long gpu
    done
    echo ""
}

# ========================================================================
# ANTIBODY DMS: 8 datasets at 4/42, 2 koenig at 23/42, 1 at 35/42.
# Large DMS datasets (koenig 4275, phillips 5000, adams 5000) need more time.
# Small AbAgym (146-659 samples) faster but still need GPU for LMs.
# ========================================================================
run_antibody_dms() {
    local OUT="tests/results/phase5_baselines/antibody"
    echo "--- Antibody DMS: 11 datasets × 42 reps (GPU, long) ---"
    # Large DMS (n>4000, 42 reps × 5 models × 5-fold)
    for ds in flab_binding_koenig flab_expression_koenig \
              flab_binding_phillips_h3 flab_binding_adams flab_expression_adams; do
        submit "abdms_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}" \
            256G 47:59:00 long gpu
    done
    # Smaller DMS (n<700)
    for ds in abagym_HER2 abagym_VEGF abagym_EGFR abagym_lysozyme abagym_Ang2 abagym_all; do
        submit "abdms_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}" \
            128G 47:59:00 long gpu
    done
    echo ""
}

# ========================================================================
# ANTIBODY INTER-FAMILY: mostly 37/42 but some at 22-23/42.
# All small (n=130-672). GPU needed for LM reps. 5 models per rep.
# ========================================================================
run_antibody_inter() {
    local OUT="tests/results/phase5_baselines/antibody"
    echo "--- Antibody Inter-Family: 14 datasets × 42 reps (GPU, long, 48h) ---"
    for ds in flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt \
              flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain \
              abcov_kd flab_aggregation_jain flab_immunogenicity_marks \
              flab_polyreact_jain_bvp flab_thermo_hutchinson flab_aggregation_jain_sas \
              ddg_combined ddg_combined_full; do
        submit "abint_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}" \
            128G 47:59:00 long gpu
    done
    echo ""
}

# ========================================================================
# ANTIBODY CLF: tz_binary 37/42, rest 1/42 or missing. GPU for LMs.
# tz_her2 datasets are 5000 samples = slower.
# ========================================================================
run_antibody_clf() {
    local OUT="tests/results/opig/clf"
    echo "--- Antibody CLF: 4 datasets × 42 reps (GPU, long, 48h) ---"
    for ds in tz_her2_binary tz_her2_multiclass; do
        submit "abclf_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}" \
            256G 47:59:00 long gpu
    done
    for ds in humanness_species humanness_therapeutic; do
        submit "abclf_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}" \
            128G 47:59:00 long gpu
    done
    echo ""
}

# ========================================================================
# GINKGO: 8 at 34/42, purity at 4/42. Small datasets (n<250). GPU for LMs.
# ========================================================================
run_ginkgo() {
    local OUT="tests/results/opig/ginkgo"
    echo "--- Ginkgo: 9 datasets × 42 reps (GPU, long, 48h) ---"
    for ds in ginkgo_hic ginkgo_tm2 ginkgo_titer ginkgo_pr_cho ginkgo_acsins \
              ginkgo_purity ginkgo_aggregation ginkgo_colloidal ginkgo_heparin; do
        submit "ginkgo_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}" \
            128G 47:59:00 long gpu
    done
    echo ""
}

# ========================================================================
# GRAPHINITY DDG: all 3 missing (column bug fixed). foldx is 50K samples.
# ========================================================================
run_ddg_synth() {
    local OUT="tests/results/opig/ddg"
    echo "--- Graphinity DDG: 3 datasets × 42 reps (GPU, long, 48h) ---"
    for ds in graphinity_foldx graphinity_rosetta graphinity_augmented; do
        submit "ddg_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --reps ${ALL_AB_REPS}" \
            256G 47:59:00 long gpu
    done
    echo ""
}

# ========================================================================
# DTA: all missing. GPU needed for protein LMs.
# ========================================================================
run_dta() {
    local OUT="tests/results/phase5_baselines/dta"
    local OUT2="tests/results/phase5_baselines/dta_cold"
    echo "--- DTA: 3 dedicated + 2 cold-drug (GPU, long, 48h) ---"
    for ds in davis kiba bindingdb_kd; do
        submit "dta_${ds}" \
            "mkdir -p ${OUT} && python tests/dta_baseline_survey.py ${ds} --n-folds 5 --output ${OUT}/${ds}.json" \
            256G 47:59:00 long gpu
    done
    submit "dta_davis_cold" \
        "mkdir -p ${OUT2} && python tests/dta_baseline_survey.py davis --n-folds 5 --cold-setting drug --output ${OUT2}/davis_cold_drug.json" \
        256G 47:59:00 long gpu
    submit "dta_kiba_cold" \
        "mkdir -p ${OUT2} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --cold-setting drug --output ${OUT2}/kiba_cold_drug.json" \
        256G 47:59:00 long gpu
    echo ""
}

# ========================================================================
# NUCLEOTIDE: all 22 missing. No GPU (classical reps only). Medium for 48h.
# Some datasets are 100K samples — 12h short may not be enough.
# ========================================================================
run_nucleotide() {
    local OUT="tests/results/nucleotide_survey"
    echo "--- Nucleotide: 22 datasets (CPU, medium, 48h) ---"
    for ds in nt_promoter_all nt_promoter_tata nt_promoter_no_tata \
              nt_enhancers nt_enhancers_types \
              nt_splice_sites_all nt_splice_sites_acceptors nt_splice_sites_donors \
              nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 \
              nt_H3K4me1 nt_H3K4me2 nt_H3K4me3 \
              nt_H3K9ac nt_H3K9me3 nt_H4K20me1; do
        submit "nuc_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --output ${OUT}/" \
            128G 1-23:59:00 medium nogpu
    done
    # Genomic Benchmarks (larger datasets, up to 100K)
    for ds in gb_human_nontata_promoters gb_human_enhancers_cohn \
              gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm; do
        submit "nuc_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --output ${OUT}/" \
            128G 1-23:59:00 medium nogpu
    done
    echo ""
}

# ========================================================================
# GEOMETRY SURVEY: uses cached antibody reps. No GPU needed. CPU short.
# ========================================================================
run_geometry() {
    local OUT="tests/results/geometry_survey"
    local BASE_REPS="esm2,ablang2,prot_t5,esm3,mint,saprot,antiberty,blosum,aac"
    echo "--- Geometry: 15 datasets (CPU, short, 12h) ---"
    for ds in flab_thermo_tresanco flab_pk_jain_fcrnrt flab_polyreact_jain_cic \
              flab_aggregation_jain_hicrt flab_thermo_jain abcov_kd \
              ginkgo_hic ginkgo_tm2 ginkgo_titer ginkgo_pr_cho ginkgo_acsins \
              ginkgo_purity ginkgo_aggregation ginkgo_colloidal ginkgo_heparin; do
        submit "geo_${ds}" \
            "mkdir -p ${OUT} && python tests/experiments/structural_geometry_survey.py ${ds} --base-reps ${BASE_REPS} --models ridge,rf --output ${OUT}/" \
            64G 11:59:00 short nogpu
    done
    echo ""
}

# ========================================================================
# FORWARD COMPAT: single-rep or few-rep targeted experiments. GPU for LMs.
# ========================================================================
run_forward_compat() {
    local OUT="tests/results/forward_compat"
    echo "--- Forward Compat: exp1,2,4,6,7,9 (GPU, long, 48h) ---"

    # Exp 1: MINT — antibody
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "fc_mint_${ds}" \
            "mkdir -p ${OUT}/exp1_mint && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp1_mint/ --n-folds 5 --augment --reps mint" \
            128G 47:59:00 long gpu
    done

    # Exp 2: SaProt — antibody
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined flab_thermo_tresanco; do
        submit "fc_sap_${ds}" \
            "mkdir -p ${OUT}/exp2_saprot && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp2_saprot/ --n-folds 5 --augment --reps saprot,saprot_cdr" \
            128G 47:59:00 long gpu
    done

    # Exp 4: IgT5 / AbRoBERTa — antibody
    for ds in flab_expression_koenig flab_binding_koenig flab_binding_phillips_h3 \
              ddg_combined flab_thermo_tresanco abcov_kd; do
        submit "fc_ablm_${ds}" \
            "mkdir -p ${OUT}/exp4_ablm && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp4_ablm/ --n-folds 5 --augment --reps igt5,abroberta,abroberta_cdr" \
            128G 47:59:00 long gpu
    done

    # Exp 6: xTrimoPGLM — antibody
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "fc_xtri_${ds}" \
            "mkdir -p ${OUT}/exp6_xtrimopglm && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp6_xtrimopglm/ --n-folds 5 --augment --reps xtrimopglm" \
            128G 47:59:00 long gpu
    done

    # Exp 7: ReactionT5 — no GPU needed (T5 encoding is fast)
    for ds in buchwald_hartwig suzuki bh_doyle_plates; do
        submit "fc_rxnt5_${ds}" \
            "mkdir -p ${OUT}/exp7_reactiont5 && python tests/run_baseline_survey.py ${ds} --n-folds 5 --augment --output ${OUT}/exp7_reactiont5/" \
            64G 11:59:00 short nogpu
    done

    # Exp 9: AIDO.Protein — large model, needs GPU + high memory
    for ds in flab_expression_koenig flab_binding_koenig ddg_combined; do
        submit "fc_aido_${ds}" \
            "mkdir -p ${OUT}/exp9_aido_protein && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/exp9_aido_protein/ --n-folds 5 --augment --reps aido_protein" \
            256G 47:59:00 long gpu
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
        echo "  Partition/resource allocation:"
        echo "    GPU (long, 48h):  molecular, tcga, tcga_ft, antibody_*, ginkgo, ddg_synth, dta, fc (most)"
        echo "    CPU (short, 12h): reaction, geometry, fc_rxnt5"
        echo "    CPU (medium, 48h): nucleotide"
        echo ""
        echo "Modes:"
        echo "  all             Everything (~120 jobs)"
        echo "  molecular       8 datasets × full reps (8 GPU jobs)"
        echo "  molecular_ft    SKIPPED (FT doesn't help)"
        echo "  reaction        3 datasets × full reps (3 CPU jobs, short)"
        echo "  tcga            3 datasets × full reps (3 GPU jobs)"
        echo "  tcga_ft         3 datasets × FT reps (3 GPU jobs)"
        echo "  antibody_dms    11 datasets × 42 reps (11 GPU jobs)"
        echo "  antibody_inter  14 datasets × 42 reps (14 GPU jobs)"
        echo "  antibody_clf    4 datasets × 42 reps (4 GPU jobs)"
        echo "  antibody        All antibody (29 GPU jobs)"
        echo "  ginkgo          9 datasets × 42 reps (9 GPU jobs)"
        echo "  ddg_synth       3 Graphinity datasets (3 GPU jobs)"
        echo "  opig            clf + ginkgo + ddg (16 GPU jobs)"
        echo "  dta             3 datasets + 2 cold-drug (5 GPU jobs)"
        echo "  nucleotide      22 datasets (22 CPU jobs, medium)"
        echo "  geometry        15 datasets (15 CPU jobs, short)"
        echo "  forward_compat  6 experiments (~24 mixed jobs)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
