#!/bin/bash
# Nucleotide baseline survey: 22 datasets × 26 reps (14 FMs + 11 classical + 1 factory).
# Screening: 3 models × 3 folds. GPU needed for FM reps.
#
# Usage (on login node):
#   bash tests/slurm_scripts/submit_nucleotide_survey.sh

OUTPUT="tests/results/nucleotide_survey"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
WRAP_PREFIX="cd /data/stat-cadd/scat9264/KIRBy && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' -e 'NoneType.*split' >&2)"

MODELS="ridge,rf,xgboost"
FOLDS=3

echo "Submitting nucleotide survey (26 reps, ${MODELS}, ${FOLDS} folds)..."
echo "  Output: ${OUTPUT}/"
echo ""

for ds in \
    nt_promoter_all nt_promoter_tata nt_promoter_no_tata \
    nt_enhancers nt_enhancers_types \
    nt_splice_sites_all nt_splice_sites_acceptors nt_splice_sites_donors \
    nt_H2AFZ nt_H3K27ac nt_H3K27me3 nt_H3K36me3 \
    nt_H3K4me1 nt_H3K4me2 nt_H3K4me3 nt_H3K9ac nt_H3K9me3 nt_H4K20me1 \
    gb_human_nontata_promoters gb_human_enhancers_cohn \
    gb_demo_coding_vs_intergenomic_seqs gb_demo_human_or_worm \
; do
    J=$(sbatch \
        --job-name="nuc_${ds}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=11:59:00 --partition=short \
        --mem=256G \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && mkdir -p ${OUTPUT} && python tests/run_baseline_survey.py ${ds} --reps all --models ${MODELS} --n-folds ${FOLDS} --output ${OUTPUT}/ ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  $J  ${ds}"
done

echo ""
echo "Total: 22 jobs"
echo "Monitor: squeue -u \$USER"
