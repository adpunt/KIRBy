#!/bin/bash
# Submit nucleotide baseline survey as SLURM job.
# Run on the login node: bash tests/slurm_scripts/submit_nucleotide_survey.sh
#
# Covers: 18 NT Revised tasks + 4 Genomic Benchmarks = 22 datasets.
# Phase G0: classical reps only (no GPU needed).
#
# Usage:
#   bash tests/slurm_scripts/submit_nucleotide_survey.sh           # all 22 tasks
#   bash tests/slurm_scripts/submit_nucleotide_survey.sh quick     # single representative task

MODE=${1:-all}
OUTPUT="tests/results/nucleotide_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

submit() {
    local name=$1
    local dataset=$2
    local mem=${3:-128G}
    local wall=${4:-11:59:00}

    J=$(sbatch \
        --job-name="nuc_${name}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=medium \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && mkdir -p ${OUTPUT} && python tests/run_baseline_survey.py ${dataset} --n-folds 5 --output ${OUTPUT}/ 2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' >&2)" \
        | awk '{print $4}')
    echo "  $J  ${name}  (dataset=${dataset}, mem=${mem}, time=${wall})"
}

echo "Submitting nucleotide baseline survey..."
echo "  Output: ${OUTPUT}/"
echo ""

case "$MODE" in
    quick)
        submit quick "nt_promoter_tata" 64G 06:00:00
        ;;
    all)
        # Submit all 22 nucleotide datasets as one job via domain shortcut
        submit all "nucleotide" 128G 11:59:00
        ;;
    *)
        # Treat argument as a dataset name
        submit "${MODE}" "${MODE}" 128G 11:59:00
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
