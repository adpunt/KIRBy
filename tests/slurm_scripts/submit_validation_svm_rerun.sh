#!/bin/bash
# Re-run alternative_data_noise_robustness.py for SVM only, on all three
# validation datasets (LogD, Caco-2, hERG-Ki).
#
# Reason for this re-run: previously the validation SVM used Optuna-tuned
# hyperparameters per representation (RBF for ECFP4, poly/deg-2 for SNS and
# PDV), while the QM9 SVM used a fixed RBF kernel (C=1.0, gamma='scale').
# That made the validation SVM a different model from the QM9 SVM, breaking
# the "does QM9 robustness generalise to experimental data" comparison and
# introducing a kernel--representation confound in the validation ANOVA.
# alternative_data_noise_robustness.py has been patched to use the QM9
# defaults (fixed RBF) for every representation; this script regenerates
# the SVM rows under the matched configuration.
#
# Merge behaviour: --models SVM removes existing SVM rows from
# all_results.csv, summary.csv, and combined_summary.csv before writing
# new ones (see alternative_data_noise_robustness.py:764-778, 826-839,
# 976-983). Other models' rows are untouched. SVM is deterministic and
# does not produce uncertainty CSVs.
#
# Usage: bash tests/slurm_scripts/submit_validation_svm_rerun.sh

OUTPUT="tests/results/validation"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

J=$(sbatch \
    --job-name="val_svm_rerun" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --time=23:59:00 --partition=long \
    --mem=64G \
    --mail-user="${EMAIL}" --mail-type=END,FAIL \
    --wrap="cd /data/stat-ecr/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python tests/alternative_data_noise_robustness.py --models SVM --datasets all --results-root ${OUTPUT}" \
    | awk '{print $4}')
echo "Submitted job ${J}: SVM rerun (3 datasets x 4 reps x 6 strategies x 11 sigmas x 5 folds)"
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
