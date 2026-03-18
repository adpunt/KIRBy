#!/bin/bash
# Submit molecular graph/structural representation augmentation survey.
# Run on the login node: bash tests/slurm_scripts/submit_molecular_graph_survey.sh
#
# Prereq: base reps (ecfp4, mhg_gnn, grover, mordred, chemberta) must be
# cached from a prior run_baseline_survey.py molecular run.
#
# Usage:
#   bash tests/slurm_scripts/submit_molecular_graph_survey.sh           # all 3 datasets
#   bash tests/slurm_scripts/submit_molecular_graph_survey.sh herg      # hERG only
#   bash tests/slurm_scripts/submit_molecular_graph_survey.sh logd      # LogD only
#   bash tests/slurm_scripts/submit_molecular_graph_survey.sh qm9       # QM9 only

MODE=${1:-all}
OUTPUT="tests/results/molecular_graph_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

# Base reps to augment with (must already be cached from molecular survey)
BASE_REPS="ecfp4,mhg_gnn,grover,mordred,chemberta"

submit() {
    local name=$1
    local dataset=$2
    local mem=$3
    local timelimit=$4
    local partition=$5

    J=$(sbatch \
        --job-name="molgraph_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=${timelimit} --partition=${partition} \
        --mem=${mem} \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/experiments/molecular_graph_survey.py ${dataset} --base-reps ${BASE_REPS} --models ridge,rf --output ${OUTPUT}/ ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${name}  (${mem}, ${timelimit}, ${partition})"
}

echo "Submitting molecular graph/structural augmentation survey..."
echo "  Output: ${OUTPUT}/"
echo "  Base reps: ${BASE_REPS}"
echo ""

case "$MODE" in
    herg)
        # 1.4K mols — fast conformer gen, ~2h total
        submit herg herg 64G 05:59:00 short
        echo "  Total: 1 job (hERG)"
        ;;
    logd)
        # 5K mols — moderate conformer gen, ~8h total
        submit logd logd 64G 11:59:00 short
        echo "  Total: 1 job (LogD)"
        ;;
    qm9)
        # 10K mols — heavy conformer gen, ~16h total
        submit qm9 qm9 128G 23:59:00 long
        echo "  Total: 1 job (QM9)"
        ;;
    all)
        submit herg herg 64G 05:59:00 short
        submit logd logd 64G 11:59:00 short
        submit qm9 qm9 128G 23:59:00 long
        echo "  Total: 3 jobs (hERG + LogD + QM9)"
        ;;
    *)
        echo "Usage: bash submit_molecular_graph_survey.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all       All 3 datasets (3 jobs, default)"
        echo "  herg      hERG only (1 job, ~1.4K mols)"
        echo "  logd      LogD only (1 job, ~5K mols)"
        echo "  qm9       QM9 only (1 job, ~10K mols)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
