#!/bin/bash
# Precompute hybrid searches — one SLURM job PER DATASET for max parallelism.
# No GPU needed (hybrid search is sklearn models only).
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute.sh              # all domains
#   bash tests/slurm_scripts/submit_precompute.sh molecular     # one domain
#   bash tests/slurm_scripts/submit_precompute.sh antibody

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
COUNT=0

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}
    local time_limit=${4:-24:00:00}

    J=$(sbatch \
        --job-name="precomp_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${time_limit}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}" \
        | awk '{print $4}')
    echo "  $J  precomp_${name}  (mem=${mem}, ${time_limit})"
    COUNT=$((COUNT + 1))
}

echo "============================================"
echo "Hybrid Precompute (1 job per dataset)"
echo "============================================"
echo ""

run_molecular() {
    echo "--- Molecular (7 datasets) ---"
    for ds in herg_ki caco2 logd qm9 adme_hlm adme_rlm adme_mdr1; do
        submit "mol_${ds}" \
            "python tests/precompute_hybrids.py ${ds} --speed fast"
    done
    echo ""
}

run_antibody() {
    echo "--- Antibody Inter-Family (11 datasets) ---"
    for ds in flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt \
              flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain \
              abcov_kd flab_aggregation_jain flab_immunogenicity_marks \
              flab_polyreact_jain_bvp ddg_combined; do
        submit "ab_${ds}" \
            "python tests/precompute_hybrids.py ${ds} --speed fast" \
            256G 24:00:00
    done

    echo "--- Antibody DMS (5 datasets) ---"
    for ds in flab_binding_koenig flab_expression_koenig flab_binding_adams \
              flab_expression_adams flab_binding_phillips_h3; do
        submit "ab_${ds}" \
            "python tests/precompute_hybrids.py ${ds} --speed fast" \
            256G 24:00:00
    done

    echo "--- AbAgym (5 datasets) ---"
    for ds in abagym_Ang2 abagym_EGFR abagym_HER2 abagym_lysozyme abagym_VEGF; do
        submit "ab_${ds}" \
            "python tests/precompute_hybrids.py ${ds} --speed fast" \
            256G 12:00:00
    done

    echo "--- Antibody CLF + Ginkgo (8 datasets) ---"
    for ds in tz_her2_binary humanness_therapeutic \
              ginkgo_acsins ginkgo_colloidal ginkgo_heparin \
              ginkgo_hic ginkgo_pr_cho ginkgo_titer; do
        submit "ab_${ds}" \
            "python tests/precompute_hybrids.py ${ds} --speed fast" \
            128G 12:00:00
    done
    echo ""
}

run_tcga() {
    echo "--- TCGA (2 datasets) ---"
    for ds in tcga_luad_vs_lusc tcga_brca_pam50; do
        submit "tcga_${ds}" \
            "python tests/precompute_hybrids.py ${ds} --speed fast" \
            256G 24:00:00
    done
    echo ""
}

run_reaction() {
    echo "--- Reaction (2 datasets) ---"
    for ds in buchwald_hartwig bh_doyle_plates; do
        submit "rxn_${ds}" \
            "python tests/precompute_hybrids.py ${ds} --speed fast" \
            128G 12:00:00
    done
    echo ""
}

case "$MODE" in
    molecular) run_molecular ;;
    antibody)  run_antibody ;;
    tcga)      run_tcga ;;
    reaction)  run_reaction ;;
    all)
        run_molecular
        run_antibody
        run_tcga
        run_reaction
        echo "============================================"
        echo "Total: ${COUNT} SLURM jobs (1 per dataset)"
        echo "  Molecular:  7"
        echo "  Antibody:  29"
        echo "  TCGA:       2"
        echo "  Reaction:   2"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash submit_precompute.sh [all|molecular|antibody|tcga|reaction]"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Cache:   ls -lt ~/.cache/kirby/hybrid_search/"
