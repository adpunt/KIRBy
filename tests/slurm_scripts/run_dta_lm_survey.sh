#!/bin/bash
#SBATCH --job-name=dta_lm_survey
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# =============================================================================
# DTA LM + KMER BASELINE SURVEY
# =============================================================================
#
# Fills the remaining PENDING cells in BOM §2D:
#   - drug_chemberta, drug_molformer, drug_selformer (GPU for speed)
#   - protein_esm2 (GPU required — 650M params, 379-442 proteins)
#   - protein_kmer (CPU-only but 8000D, needs time for model training)
#
# Lightweight reps (ecfp4/maccs/pdv/aac/descriptors) already done locally.
#
# Usage:
#   sbatch run_dta_lm_survey.sh           # both datasets
#   sbatch run_dta_lm_survey.sh davis     # davis only
#   sbatch run_dta_lm_survey.sh kiba      # kiba only
# =============================================================================

DATASET=${1:-all}
OUTPUT="tests/results/baseline_survey"
mkdir -p "$OUTPUT"

echo "============================================================"
echo "DTA LM + Kmer Baseline Survey"
echo "  Dataset:  $DATASET"
echo "  Output:   $OUTPUT"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Started:  $(date)"
echo "============================================================"

PASS=0
FAIL=0

run_survey() {
    local ds=$1
    local tag="dta_${ds}_lm_survey"
    local extra=""

    if [ "$ds" = "kiba" ]; then
        extra="--max-pairs 30000"
    fi

    echo ""
    echo "--- $tag ---"
    echo "  Started: $(date)"

    if python tests/dta_baseline_survey.py "$ds" \
        $extra \
        --output "${OUTPUT}/${tag}.json" 2>&1; then
        echo "  PASSED: $(date)"
        PASS=$((PASS + 1))
    else
        echo "  FAILED (exit $?): $(date)"
        FAIL=$((FAIL + 1))
    fi
}

case "$DATASET" in
    davis)  run_survey davis ;;
    kiba)   run_survey kiba ;;
    all)    run_survey davis; run_survey kiba ;;
    *)
        echo "Usage: sbatch run_dta_lm_survey.sh [all|davis|kiba]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "DTA LM SURVEY COMPLETE"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Results: $OUTPUT"
echo "  Finished: $(date)"
echo "============================================================"
