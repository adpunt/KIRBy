#!/bin/bash
# Resubmit inference-only jobs on ARC (no GPU).
# These jobs all use torch.cuda.is_available() fallback — safe on CPU.
#
# DO NOT add --gres=gpu to any job in this script.
#
# Usage:
#   bash tests/slurm_scripts/submit_arc_nogpu.sh              # all 29 jobs
#   bash tests/slurm_scripts/submit_arc_nogpu.sh antibody      # ab surveys + clf
#   bash tests/slurm_scripts/submit_arc_nogpu.sh dta
#   bash tests/slurm_scripts/submit_arc_nogpu.sh nucleotide
#   bash tests/slurm_scripts/submit_arc_nogpu.sh reaction
#   bash tests/slurm_scripts/submit_arc_nogpu.sh augmentation
#   bash tests/slurm_scripts/submit_arc_nogpu.sh graphinity

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

ALL_AB_REPS="onehot,blosum,aac,paratope,esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr,structural,mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr,esm2_entropy_top20,esm2_per_cdr,esm2_per_region,mint,aido_protein,xtrimopglm"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-128G}
    local wall=${4:-47:59:00}
    local partition=${5:-long}

    J=$(sbatch --parsable \
        --job-name="${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition="${partition}" \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}")
    echo "  $J  ${name}  (${partition}, ${mem}, ${wall}, NO GPU)"
}

echo "============================================"
echo "ARC Resubmission (NO GPU — CPU inference)"
echo "Filter: ${MODE}"
echo "============================================"
echo ""

# ── Antibody Surveys (2 jobs) ──
run_antibody() {
    local OUT="tests/results/phase5_baselines/antibody"
    echo "--- Antibody Surveys (all_dms + all_inter, 42 reps, CPU) ---"
    submit ab_all_dms \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_dms --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
        256G 47:59:00 long
    submit ab_all_inter \
        "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py all_inter --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
        256G 47:59:00 long
    echo ""
}

# ── Antibody Classification (1 job — humanness) ──
run_antibody_clf() {
    local OUT="tests/results/opig/clf"
    echo "--- Antibody CLF (humanness, 42 reps, CPU) ---"
    for ds in humanness_therapeutic; do
        submit "abclf_${ds}" \
            "mkdir -p ${OUT} && python tests/antibody_baseline_survey.py ${ds} --output ${OUT}/ --n-folds 5 --augment --skip-existing --reps ${ALL_AB_REPS}" \
            128G 47:59:00 long
    done
    echo ""
}

# ── Graphinity DDG (3 jobs) ──
run_graphinity() {
    local OUT="tests/results/opig/ddg"
    # CUT 2026-07-09: graphinity_foldx/rosetta/augmented predict FoldX/Rosetta
    # OUTPUT, not experimental ddG. Removed from DATASET_REGISTRY.
    echo "--- Graphinity DDG: CUT, nothing to submit ---"
    echo ""
}

# ── DTA Surveys (3 jobs) ──
run_dta() {
    local OUT="tests/results/phase5_baselines/dta"
    echo "--- DTA (davis, kiba, bindingdb_kd, CPU) ---"
    submit dta_davis \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py davis --n-folds 5 --output ${OUT}/davis.json" \
        256G 47:59:00 long
    submit dta_kiba \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py kiba --n-folds 5 --max-pairs 30000 --output ${OUT}/kiba.json" \
        256G 47:59:00 long
    submit dta_bindingdb \
        "mkdir -p ${OUT} && python tests/dta_baseline_survey.py bindingdb_kd --n-folds 5 --max-pairs 30000 --output ${OUT}/bindingdb_kd.json" \
        256G 47:59:00 long
    echo ""
}

# ── Reaction (1 job) ──
run_reaction() {
    local OUT="tests/results/phase5_baselines/reaction"
    echo "--- Reaction (bh_doyle_plates, CPU) ---"
    submit rxn_bh_doyle_plates \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py bh_doyle_plates --n-folds 5 --augment --output ${OUT}/" \
        64G 11:59:00 short
    echo ""
}

# ── Molecular Augmentation (2 jobs) ──
run_augmentation() {
    local OUT="tests/results/molecular_aug_fill"
    echo "--- Molecular Augmentation (adme_mdr1 + qm9, screening, CPU) ---"
    submit aug_adme_mdr1 \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py adme_mdr1 --augment --models ridge,rf,xgboost --n-folds 3 --output ${OUT}/" \
        128G 23:59:00 long
    submit aug_qm9 \
        "mkdir -p ${OUT} && python tests/run_baseline_survey.py qm9 --augment --models ridge,rf,xgboost --n-folds 3 --output ${OUT}/" \
        128G 23:59:00 long
    echo ""
}

# ── Nucleotide (22 jobs) ──
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
            128G 1-23:59:00 medium
    done
    for ds in gb_human_nontata_promoters gb_human_enhancers_cohn \
              gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm; do
        submit "nuc_${ds}" \
            "mkdir -p ${OUT} && python tests/run_baseline_survey.py ${ds} --n-folds 5 --output ${OUT}/" \
            128G 1-23:59:00 medium
    done
    echo ""
}

# ── Dispatch ──
case "$MODE" in
    all)
        run_antibody
        run_antibody_clf
        run_graphinity
        run_dta
        run_reaction
        run_augmentation
        run_nucleotide
        ;;
    antibody)    run_antibody; run_antibody_clf ;;
    graphinity)  run_graphinity ;;
    dta)         run_dta ;;
    reaction)    run_reaction ;;
    augmentation) run_augmentation ;;
    nucleotide)  run_nucleotide ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Options: all, antibody, graphinity, dta, reaction, augmentation, nucleotide"
        exit 1
        ;;
esac

echo "============================================"
echo "All jobs submitted WITHOUT GPU."
echo "Remember to scancel the old HTC versions!"
echo "============================================"
