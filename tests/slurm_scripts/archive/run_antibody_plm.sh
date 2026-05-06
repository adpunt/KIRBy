#!/bin/bash
#SBATCH --job-name=ab_plm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# Antibody experiments with PLM-augmented mutation presets (esm2 + prot_t5).
# Runs all C01-C10 + S01-S08 with resume support (skips existing results).
#
# Usage:
#   sbatch tests/slurm_scripts/run_antibody_plm.sh                          # mutation_fast, --delta
#   sbatch tests/slurm_scripts/run_antibody_plm.sh mutation_full "--delta --augment"
#   sbatch tests/slurm_scripts/run_antibody_plm.sh mutation_structural "--delta --augment"

REPS=${1:-mutation_fast}
EXTRA=${2:---delta}
DATASET=flab_expression_koenig
OUTPUT=results_v3_ab
MODE=intermediary

COMMON="--mode $MODE --dataset $DATASET --reps $REPS --output $OUTPUT $EXTRA"

echo "============================================================"
echo "KIRBy antibody PLM experiments (resume mode)"
echo "  Mode:    $MODE"
echo "  Reps:    $REPS"
echo "  Dataset: $DATASET"
echo "  Output:  $OUTPUT"
echo "  Extra:   ${EXTRA}"
echo "============================================================"

# Check sentencepiece is available (needed for ProtT5)
python -c "import sentencepiece" 2>/dev/null || {
    echo "ERROR: sentencepiece not installed. Run: pip install sentencepiece>=0.1.99"
    exit 1
}

set -e

run_if_missing() {
    local exp_id=$1
    local label=$2
    local script=$3
    if [ -f "$OUTPUT/$exp_id/$DATASET.json" ]; then
        echo ""
        echo ">>> $label: SKIP (already exists: $OUTPUT/$exp_id/$DATASET.json)"
    else
        echo ""
        echo ">>> $label"
        python "$script" $COMMON
    fi
}

# --- Complexity experiments ---
run_if_missing c01_rep_characterization "C01: Rep Characterization" tests/experiments/complexity/c01_rep_characterization.py
run_if_missing c02_complexity_grid "C02: Complexity Grid" tests/experiments/complexity/c02_complexity_grid.py
run_if_missing c04_importance_as_metric "C04: Importance as Metric" tests/experiments/complexity/c04_importance_as_metric.py
run_if_missing c05_learning_curves "C05: Learning Curves" tests/experiments/complexity/c05_learning_curves.py
run_if_missing c06_within_family_gradient "C06: Within-Family Gradient" tests/experiments/complexity/c06_within_family_gradient.py
run_if_missing c07_data_complexity_partitions "C07: Data Complexity Partitions" tests/experiments/complexity/c07_data_complexity_partitions.py
run_if_missing c08_calibration "C08: Calibration" tests/experiments/complexity/c08_calibration.py
run_if_missing c09_formal_analysis "C09: Formal Analysis" tests/experiments/complexity/c09_formal_analysis.py
run_if_missing c10_noise_robustness "C10: Noise Robustness" tests/experiments/complexity/c10_noise_robustness.py

# --- Similarity experiments ---
run_if_missing s01_similarity_baseline "S01: Similarity Baseline" tests/experiments/similarity/s01_similarity_baseline.py
run_if_missing s02_hybrid_similarity "S02: Hybrid Similarity" tests/experiments/similarity/s02_hybrid_similarity.py
run_if_missing s03_feature_concentration "S03: Feature Concentration" tests/experiments/similarity/s03_feature_concentration.py
run_if_missing s04_activity_cliffs "S04: Activity Cliffs" tests/experiments/similarity/s04_activity_cliffs.py
run_if_missing s05_knn_prediction "S05: kNN Prediction" tests/experiments/similarity/s05_knn_prediction.py
run_if_missing s06_applicability_domain "S06: Applicability Domain" tests/experiments/similarity/s06_applicability_domain.py
run_if_missing s07_noise_bridge "S07: Noise Bridge" tests/experiments/similarity/s07_noise_bridge.py
run_if_missing s08_optimal_dimension "S08: Optimal Dimension" tests/experiments/similarity/s08_optimal_dimension.py

echo ""
echo "============================================================"
echo "ANTIBODY PLM EXPERIMENTS COMPLETE"
echo "Results in: $OUTPUT"
echo "============================================================"
