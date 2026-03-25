#!/bin/bash
# ESM-2 LoRA fine-tuning on ALL active antibody datasets.
#
# Already tested (from antibody_finetune_comparison.json):
#   flab_binding_phillips_h3, flab_expression_koenig, flab_binding_koenig,
#   flab_binding_adams, flab_thermo_tresanco, abcov_kd
#
# This script runs per-dataset to get individual results.
# Uses antibody_baseline_survey.py --finetune (adds ESM-2 LoRA).
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_antibody_ft_fill.sh          # all datasets
#   bash tests/slurm_scripts/submit_antibody_ft_fill.sh inter    # inter-family only
#   bash tests/slurm_scripts/submit_antibody_ft_fill.sh dms      # DMS only
#   bash tests/slurm_scripts/submit_antibody_ft_fill.sh clf      # CLF only
#   bash tests/slurm_scripts/submit_antibody_ft_fill.sh ginkgo   # Ginkgo only
#   bash tests/slurm_scripts/submit_antibody_ft_fill.sh DATASET  # single dataset

MODE=${1:-all}
OUTPUT="tests/results/antibody_ft_fill"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

MODELS="ridge,rf,xgboost"
FOLDS=3

submit() {
    local name=$1
    local dataset=$2
    local mem=${3:-128G}
    local wall=${4:-23:59:00}

    J=$(sbatch \
        --job-name="abft_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --gres=gpu:1 \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && pip install --quiet 'urllib3<2' 'peft>=0.7' 'accelerate>=0.25' bitsandbytes 2>/dev/null && mkdir -p ${OUTPUT} && python tests/antibody_baseline_survey.py ${dataset} --finetune --models ${MODELS} --n-folds ${FOLDS} --output ${OUTPUT}/ ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  abft_${name}  (long, ${mem}, ${wall}, GPU, ${FOLDS} folds)"
}

# Inter-family datasets (11 active)
INTER="flab_thermo_tresanco flab_thermo_jain flab_aggregation_jain flab_aggregation_jain_hicrt flab_immunogenicity_marks flab_polyreact_jain_bvp flab_polyreact_jain_cic flab_pk_jain_fcrnrt flab_pk_jain_heprt abcov_kd ddg_combined"

# DMS datasets (10)
DMS="flab_expression_koenig flab_binding_koenig flab_binding_phillips_h3 flab_binding_adams flab_expression_adams abagym_Ang2 abagym_EGFR abagym_lysozyme abagym_VEGF abagym_HER2"

# CLF datasets (2 active)
CLF="tz_her2_binary humanness_therapeutic"

# Ginkgo datasets (6 active)
GINKGO="ginkgo_hic ginkgo_acsins ginkgo_colloidal ginkgo_heparin ginkgo_pr_cho ginkgo_titer"

echo "Submitting antibody ESM-2 LoRA FT fill..."
echo "  Output: ${OUTPUT}/"
echo "  Models: ${MODELS} (screening)"
echo "  Folds: ${FOLDS}"
echo ""

submit_group() {
    local group_name=$1
    shift
    local count=0
    for ds in "$@"; do
        submit "${ds}" "${ds}"
        count=$((count + 1))
    done
    echo "  ${group_name}: ${count} jobs"
}

case "$MODE" in
    all)
        submit_group "Inter-family" ${INTER}
        echo ""
        submit_group "DMS" ${DMS}
        echo ""
        submit_group "CLF" ${CLF}
        echo ""
        submit_group "Ginkgo" ${GINKGO}
        echo ""
        echo "  Total: 29 jobs"
        ;;
    inter)
        submit_group "Inter-family" ${INTER}
        ;;
    dms)
        submit_group "DMS" ${DMS}
        ;;
    clf)
        submit_group "CLF" ${CLF}
        ;;
    ginkgo)
        submit_group "Ginkgo" ${GINKGO}
        ;;
    *)
        # Try as single dataset name
        submit "$MODE" "$MODE"
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
