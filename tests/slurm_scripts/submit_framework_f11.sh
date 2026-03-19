#!/bin/bash
# Submit F11 allocation × evaluator interaction as SLURM jobs.
# Run on the login node: bash tests/slurm_scripts/submit_framework_f11.sh
#
# 5 datasets: 4 small antibody (n~130) + tresanco control (n=672).
# Per dataset: (5 allocs × 4 evaluators + 2 importance) × 2 budgets × 3 models × 5-fold
# = 44 configs × 3 models × 5 folds = 660 evaluations. Fast — each ~1-5s.
# Expect 30-60min per small dataset, 1-2h for tresanco.
#
# Usage:
#   bash tests/slurm_scripts/submit_framework_f11.sh          # all 5 datasets
#   bash tests/slurm_scripts/submit_framework_f11.sh small     # just 4 small ones
#   bash tests/slurm_scripts/submit_framework_f11.sh control   # just tresanco

MODE=${1:-all}
OUTPUT="tests/results/framework"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

submit() {
    local name=$1
    local dataset=$2
    local partition=$3
    local timelimit=$4

    J=$(sbatch \
        --job-name="f11_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${timelimit}" --partition="${partition}" \
        --mem=128G \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/experiments/framework/f11_allocation_evaluator_interaction.py --dataset ${dataset} --reps all --mode full --output ${OUTPUT} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  f11_${name}  (${dataset}, ${partition}, ${timelimit})"
}

echo "Submitting F11: Allocation × Evaluator Interaction"
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    small)
        submit fcrnrt flab_pk_jain_fcrnrt short 11:59:00
        submit heprt flab_pk_jain_heprt short 11:59:00
        submit hicrt flab_aggregation_jain_hicrt short 11:59:00
        submit cic flab_polyreact_jain_cic short 11:59:00
        echo "  Total: 4 jobs (small datasets)"
        ;;
    control)
        submit tresanco flab_thermo_tresanco medium 47:59:00
        echo "  Total: 1 job (control)"
        ;;
    all)
        submit fcrnrt flab_pk_jain_fcrnrt short 11:59:00
        submit heprt flab_pk_jain_heprt short 11:59:00
        submit hicrt flab_aggregation_jain_hicrt short 11:59:00
        submit cic flab_polyreact_jain_cic short 11:59:00
        submit tresanco flab_thermo_tresanco medium 47:59:00
        echo "  Total: 5 jobs"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [all|small|control]"
        exit 1
        ;;
esac
