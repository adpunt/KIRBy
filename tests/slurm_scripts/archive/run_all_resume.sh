#!/bin/bash
#SBATCH --job-name=kirby_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-ecr/scat9264/KIRBy
. setup.sh

# Resume from where run_all.sh timed out.
# Skips experiments that already have results in OUTPUT/*/DATASET.json.
#
# Usage: same as run_all.sh
#   sbatch tests/slurm_scripts/run_all_resume.sh [mode] [reps] [output_dir] [dataset] [extra_flags]

MODE=${1:-intermediary}
REPS=${2:-fast}
OUTPUT=${3:-tests/results}
DATASET=${4:-logd}
EXTRA=${5:-}

COMMON="--mode $MODE --dataset $DATASET --reps $REPS --output $OUTPUT $EXTRA"

echo "============================================================"
echo "KIRBy experiment run (resume mode)"
echo "  Mode:    $MODE"
echo "  Reps:    $REPS"
echo "  Dataset: $DATASET"
echo "  Output:  $OUTPUT"
echo "  Extra:   ${EXTRA:-none}"
echo "============================================================"

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
run_if_missing c11_allocation_strategies "C11: Allocation Strategies" tests/experiments/complexity/c11_allocation_strategies.py
run_if_missing c12_adaptive_pruning "C12: Adaptive Pruning" tests/experiments/complexity/c12_adaptive_pruning.py
run_if_missing c13_greedy_step_size "C13: Greedy Step Size" tests/experiments/complexity/c13_greedy_step_size.py
run_if_missing c14_two_phase_search "C14: Two-Phase Search" tests/experiments/complexity/c14_two_phase_search.py

# --- Similarity experiments ---
run_if_missing s01_similarity_baseline "S01: Similarity Baseline" tests/experiments/similarity/s01_similarity_baseline.py
run_if_missing s02_hybrid_similarity "S02: Hybrid Similarity" tests/experiments/similarity/s02_hybrid_similarity.py
run_if_missing s03_feature_concentration "S03: Feature Concentration" tests/experiments/similarity/s03_feature_concentration.py
run_if_missing s04_activity_cliffs "S04: Activity Cliffs" tests/experiments/similarity/s04_activity_cliffs.py
run_if_missing s05_knn_prediction "S05: kNN Prediction" tests/experiments/similarity/s05_knn_prediction.py
run_if_missing s06_applicability_domain "S06: Applicability Domain" tests/experiments/similarity/s06_applicability_domain.py
run_if_missing s07_noise_bridge "S07: Noise Bridge" tests/experiments/similarity/s07_noise_bridge.py
run_if_missing s08_optimal_dimension "S08: Optimal Dimension" tests/experiments/similarity/s08_optimal_dimension.py

# --- Cross-dataset experiments (skipped — need all datasets loaded) ---
# Run these separately for full runs:
#   python tests/experiments/complexity/c03_cross_dataset_transfer.py --mode full --reps $REPS --output $OUTPUT --datasets logd caco2 herg qm9
#   python tests/experiments/similarity/s09_cross_dataset_summary.py --mode full --reps $REPS --output $OUTPUT --datasets logd caco2 herg qm9

echo ""
echo "============================================================"
echo "RESUME COMPLETE (except C03, S09 — run cross-dataset separately)"
echo "Results in: $OUTPUT"
echo "============================================================"
