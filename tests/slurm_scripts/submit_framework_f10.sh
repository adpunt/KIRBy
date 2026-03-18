#!/bin/bash
# Submit F10 small dataset allocation fixes as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_framework_f10.sh [mode]
#
# 5 small antibody datasets (~130 samples each) + 1 control (672 samples).
# 12 reps × 4 models × 5 folds baselines + 5 conditions × 4 budgets × 5 folds.
# Small data → fast. Expect ~30min per dataset.
#
# Usage:
#   bash tests/slurm_scripts/submit_framework_f10.sh          # all 6 datasets
#   bash tests/slurm_scripts/submit_framework_f10.sh small    # 5 small only
#   bash tests/slurm_scripts/submit_framework_f10.sh control  # control only
#   bash tests/slurm_scripts/submit_framework_f10.sh flab_pk_jain_fcrnrt  # single

MODE=${1:-all}
OUTPUT="tests/results/framework"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"
REPS="ablang2,prot_t5,esm2,esmc,blosum,onehot,aac,structural,antiberty,saprot,esm2_early,igt5"

submit() {
    local name=$1
    local dataset=$2

    J=$(sbatch \
        --job-name="f10_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=11:59:00 --partition=short \
        --mem=128G \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/experiments/framework/f10_small_dataset_allocation.py --dataset ${dataset} --reps ${REPS} --mode full --output ${OUTPUT} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  f10_${name}  (${dataset}, 128G, 12h)"
}

echo "Submitting F10: Small Dataset Allocation Fixes"
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    small)
        submit fcrnrt flab_pk_jain_fcrnrt
        submit cic flab_polyreact_jain_cic
        submit hicrt flab_aggregation_jain_hicrt
        submit heprt flab_pk_jain_heprt
        submit abcov abcov_kd
        echo "  Total: 5 jobs (small datasets)"
        ;;
    control)
        submit tresanco flab_thermo_tresanco
        echo "  Total: 1 job (control)"
        ;;
    all)
        submit fcrnrt flab_pk_jain_fcrnrt
        submit cic flab_polyreact_jain_cic
        submit hicrt flab_aggregation_jain_hicrt
        submit heprt flab_pk_jain_heprt
        submit abcov abcov_kd
        submit tresanco flab_thermo_tresanco
        echo "  Total: 6 jobs"
        ;;
    *)
        # Try as a single dataset name
        submit "${MODE}" "${MODE}"
        echo "  Total: 1 job"
        ;;
esac
