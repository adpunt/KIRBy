#!/bin/bash
# Re-run alternative_data_noise_robustness.py for PDV only (with StandardScaler fix).
# Merges into existing all_results.csv per dataset, replacing old PDV rows.
#
# Usage: bash tests/slurm_scripts/submit_validation_pdv_rerun.sh

OUTPUT="tests/results/validation"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

J=$(sbatch \
    --job-name="val_pdv_rerun" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --time=23:59:00 --partition=long \
    --mem=64G \
    --mail-user="${EMAIL}" --mail-type=END,FAIL \
    --wrap="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python tests/alternative_data_noise_robustness.py --reps PDV --results-root ${OUTPUT}" \
    | awk '{print $4}')
echo "Submitted job ${J}: PDV rerun (3 datasets × all models × 6 strategies × 5 folds)"
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
