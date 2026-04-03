#!/bin/bash
# Precompute hybrid searches for all datasets.
# One SLURM job per domain (molecular/antibody/tcga/reaction).
# Each job runs precompute_hybrids.py --domain <domain> with cache=True.
#
# Usage:
#   bash tests/slurm_scripts/submit_precompute.sh              # all domains
#   bash tests/slurm_scripts/submit_precompute.sh molecular     # one domain
#   bash tests/slurm_scripts/submit_precompute.sh antibody
#
# After completion, run experiments — they'll load cached SearchResults instantly.

set -euo pipefail

MODE=${1:-all}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"

submit() {
    local name=$1
    local cmd=$2
    local mem=${3:-256G}
    local time_limit=${4:-47:59:00}
    local gpu=${5:-}

    local gpu_flag=""
    if [ "$gpu" = "gpu" ]; then
        gpu_flag="--gres=gpu:1"
    fi

    J=$(sbatch \
        --job-name="precomp_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${time_limit}" --partition=long \
        --mem="${mem}" \
        ${gpu_flag} \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd}" \
        | awk '{print $4}')
    echo "  $J  precomp_${name}  (mem=${mem}, ${time_limit}${gpu_flag:+, GPU})"
}

echo "============================================"
echo "Hybrid Precompute (all datasets × tiers × aug)"
echo "============================================"
echo ""

run_molecular() {
    echo "--- Molecular (7 datasets) ---"
    submit molecular \
        "python tests/precompute_hybrids.py --domain molecular --speed fast" \
        256G 24:00:00
    echo ""
}

run_antibody() {
    echo "--- Antibody (29 datasets) ---"
    # Split into smaller groups to avoid 48h timeout
    # Inter-family (11 datasets)
    submit ab_inter \
        "python tests/precompute_hybrids.py flab_thermo_tresanco flab_pk_jain_heprt flab_pk_jain_fcrnrt flab_polyreact_jain_cic flab_aggregation_jain_hicrt flab_thermo_jain abcov_kd flab_aggregation_jain flab_immunogenicity_marks flab_polyreact_jain_bvp ddg_combined --speed fast" \
        256G 47:59:00 gpu
    # DMS (5 datasets)
    submit ab_dms \
        "python tests/precompute_hybrids.py flab_binding_koenig flab_expression_koenig flab_binding_adams flab_expression_adams flab_binding_phillips_h3 --speed fast" \
        256G 47:59:00 gpu
    # AbAgym (5 datasets)
    submit ab_abagym \
        "python tests/precompute_hybrids.py abagym_Ang2 abagym_EGFR abagym_HER2 abagym_lysozyme abagym_VEGF --speed fast" \
        256G 24:00:00 gpu
    # CLF + Ginkgo (8 datasets)
    submit ab_clf_ginkgo \
        "python tests/precompute_hybrids.py tz_her2_binary humanness_therapeutic ginkgo_acsins ginkgo_colloidal ginkgo_heparin ginkgo_hic ginkgo_pr_cho ginkgo_titer --speed fast" \
        256G 24:00:00 gpu
    echo ""
}

run_tcga() {
    echo "--- TCGA (2 datasets) ---"
    submit tcga \
        "python tests/precompute_hybrids.py --domain tcga --speed fast" \
        256G 24:00:00
    echo ""
}

run_reaction() {
    echo "--- Reaction (2 datasets) ---"
    submit reaction \
        "python tests/precompute_hybrids.py --domain reaction --speed fast" \
        128G 12:00:00
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
        echo "Total: 7 SLURM jobs"
        echo "  Molecular:    1 job (7 datasets)"
        echo "  Antibody:     4 jobs (29 datasets)"
        echo "  TCGA:         1 job (2 datasets)"
        echo "  Reaction:     1 job (2 datasets)"
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
