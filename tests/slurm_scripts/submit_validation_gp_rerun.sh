#!/bin/bash
# Re-run alternative_data_noise_robustness.py for GP only, on all three
# validation datasets (LogD, Caco-2, hERG-Ki).
#
# Reason: previously GP ran on PDV only (RBF kernel), so GP had no cross-rep
# coverage on the validation sets and was excluded from the cross-rep ANOVA.
# alternative_data_noise_robustness.py now takes --gp-reps / --gp-kernel:
#   - RBF-GP on ALL reps -> one consistent GP model spanning every rep, so GP
#     can enter the validation ANOVA exactly like SVM (fixed-RBF across reps).
#   - Tanimoto-GP on ECFP4 -> RBF-vs-Tanimoto head-to-head on the same rep.
#
# Merge behaviour: --models GP removes existing GP rows from all_results.csv /
# summary.csv / combined_summary.csv before writing new ones; other models'
# rows are untouched. GP is deterministic and does not produce uncertainty CSVs.
#
# Usage: bash tests/slurm_scripts/submit_validation_gp_rerun.sh

OUTPUT="tests/results/validation"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

# 1. RBF-GP on all four reps (the ANOVA-enabler; replaces the old PDV-only GP).
J1=$(sbatch \
    --job-name="val_gp_rbf_rerun" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --time=23:59:00 --partition=long \
    --mem=64G \
    --mail-user="${EMAIL}" --mail-type=END,FAIL \
    --wrap="cd /data/stat-ecr/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python tests/alternative_data_noise_robustness.py --models GP --gp-reps ECFP4 PDV SNS MHG-GNN-pretrained --gp-kernel rbf --datasets all --results-root ${OUTPUT}" \
    | awk '{print $4}')
echo "Submitted job ${J1}: RBF-GP on all 4 reps x 3 datasets x 6 strategies x 5 folds"

# 2. Tanimoto-GP on ECFP4 only (RBF-vs-Tanimoto head-to-head; Tanimoto is only
#    valid on binary fingerprints). Saved under model name 'GP-Tanimoto'.
J2=$(sbatch \
    --job-name="val_gp_tanimoto_ecfp4" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --time=23:59:00 --partition=long \
    --mem=64G \
    --mail-user="${EMAIL}" --mail-type=END,FAIL \
    --wrap="cd /data/stat-ecr/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python tests/alternative_data_noise_robustness.py --models GP-Tanimoto --gp-reps ECFP4 --gp-kernel tanimoto --datasets all --results-root ${OUTPUT}" \
    | awk '{print $4}')
echo "Submitted job ${J2}: Tanimoto-GP on ECFP4 x 3 datasets"

echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT}/"
