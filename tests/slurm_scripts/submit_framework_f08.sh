#!/bin/bash
# Submit F08 iterative augmentation design sweep as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_framework_f08.sh
#
# Submits one job per dataset (4 domains). Single fold, XGBoost, all reps.
# ~1152 configs per dataset. Permutation-based configs are slow (~7s each),
# allocation_scores configs are fast (<1s each). Expect 30-90min per dataset.
#
# Usage:
#   bash tests/slurm_scripts/submit_framework_f08.sh          # all 4 datasets
#   bash tests/slurm_scripts/submit_framework_f08.sh molecular # just molecular
#   bash tests/slurm_scripts/submit_framework_f08.sh antibody  # just antibody

MODE=${1:-all}
OUTPUT="tests/results/framework"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-ecr/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local dataset=$2

    J=$(sbatch \
        --job-name="f08_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=47:59:00 --partition=long \
        --mem=256G \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/experiments/framework/f08_iterative_augmentation.py --dataset ${dataset} --reps all --mode full --output ${OUTPUT} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  f08_${name}  (${dataset}, 256G, 48h)"
}

echo "Submitting F08: Iterative Augmentation Design Sweep"
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    molecular)
        submit logd logd
        echo "  Total: 1 job"
        ;;
    antibody)
        submit flab flab_thermo_tresanco
        echo "  Total: 1 job"
        ;;
    tcga)
        submit tcga tcga_luad_vs_lusc
        echo "  Total: 1 job"
        ;;
    dta)
        submit kiba kiba
        echo "  Total: 1 job"
        ;;
    all)
        submit logd logd
        submit flab flab_thermo_tresanco
        submit tcga tcga_luad_vs_lusc
        submit kiba kiba
        echo "  Total: 4 jobs"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [all|molecular|antibody|tcga|dta]"
        exit 1
        ;;
esac
