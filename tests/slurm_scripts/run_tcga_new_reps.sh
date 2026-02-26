#!/bin/bash
#SBATCH --job-name=tcga_new_reps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# =============================================================================
# TCGA New/Fixed Reps Baseline Survey
# =============================================================================
# Tests the 7 fixed pathway reps + 5 new expression-derived reps on LUAD vs
# LUSC and BRCA PAM50. Also includes the known-working reps as reference
# baselines for hybrid comparison.
#
# Usage:
#   sbatch run_tcga_new_reps.sh
#
# Output: tests/results/tcga_new_reps/
#   - tcga_luad_new_reps.json   (baselines + hybrid for LUAD vs LUSC)
#   - tcga_brca_new_reps.json   (baselines + hybrid for BRCA PAM50)
#
# Each JSON contains a "baselines" section with per-rep individual scores
# (best model, accuracy) that maps directly to the BILL_OF_MATERIALS.md table.

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

OUTPUT="tests/results/tcga_new_reps"
mkdir -p "$OUTPUT"

# Delete stale pathway caches (scoring method changed: ssGSEA/decoupler -> mean z-score/weighted)
echo "Deleting stale pathway caches..."
rm -f ~/.cache/kirby/reps/tcga_*/gene_hallmark.npy
rm -f ~/.cache/kirby/reps/tcga_*/gene_kegg.npy
rm -f ~/.cache/kirby/reps/tcga_*/gene_reactome.npy
rm -f ~/.cache/kirby/reps/tcga_*/gene_oncogenic.npy
rm -f ~/.cache/kirby/reps/tcga_*/gene_progeny.npy
rm -f ~/.cache/kirby/reps/tcga_*/gene_dorothea.npy
rm -f ~/.cache/kirby/reps/tcga_*/gene_combined_pathway.npy
echo "Done."

# 7 fixed pathway reps + 5 new reps + 6 known-working reference reps = 18
REPS="gene_pca100,gene_nmf50,gene_ica50,gene_stats,gene_immune,gene_hvg2000,gene_hallmark,gene_kegg,gene_reactome,gene_oncogenic,gene_progeny,gene_dorothea,gene_combined_pathway,gene_quantile_rank,gene_corr_modules,gene_chromosome_arm,gene_autoencoder50,gene_sparse_pca50"

echo "============================================================"
echo "TCGA New/Fixed Reps Baseline Survey"
echo "  Reps:    $REPS"
echo "  Search:  medium"
echo "  Started: $(date)"
echo "============================================================"

PASS=0
FAIL=0

# LUAD vs LUSC (binary classification, ~1017 samples)
echo ""
echo "--- LUAD vs LUSC ---"
if python tests/find_best_hybrid_tcga.py tcga_luad_vs_lusc \
    --reps "$REPS" --search medium --genomic-only \
    --adaptive-pruning \
    --output "${OUTPUT}/tcga_luad_new_reps.json" 2>&1; then
    echo "  PASSED: $(date)"
    PASS=$((PASS + 1))
else
    echo "  FAILED (exit $?): $(date)"
    FAIL=$((FAIL + 1))
fi

# BRCA PAM50 (4-class classification, ~821 samples)
echo ""
echo "--- BRCA PAM50 ---"
if python tests/find_best_hybrid_tcga.py tcga_brca_pam50 \
    --reps "$REPS" --search medium --genomic-only \
    --adaptive-pruning \
    --output "${OUTPUT}/tcga_brca_new_reps.json" 2>&1; then
    echo "  PASSED: $(date)"
    PASS=$((PASS + 1))
else
    echo "  FAILED (exit $?): $(date)"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "============================================================"
echo "TCGA New Reps Survey COMPLETE"
echo "  Passed: $PASS / 2"
echo "  Failed: $FAIL / 2"
echo "  Results: $OUTPUT"
echo "  Finished: $(date)"
echo "============================================================"
