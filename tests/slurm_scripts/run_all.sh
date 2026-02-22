#!/bin/bash
#SBATCH --job-name=kirby_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Run all refactored experiments (individual scripts).
#
# Usage:
#   sbatch tests/slurm_scripts/run_all.sh [mode] [reps] [output_dir] [dataset] [extra_flags]
#
# Examples:
#   sbatch tests/slurm_scripts/run_all.sh intermediary fast results_v2
#   sbatch tests/slurm_scripts/run_all.sh intermediary mutation_fast results_v2_ab flab_expression_koenig "--delta"
#   sbatch tests/slurm_scripts/run_all.sh intermediary mutation_full results_v2_ab flab_expression_koenig "--delta --augment"
#   sbatch tests/slurm_scripts/run_all.sh full fast+hf results_full logd

MODE=${1:-intermediary}
REPS=${2:-fast}
OUTPUT=${3:-tests/results}
DATASET=${4:-logd}
EXTRA=${5:-}

COMMON="--mode $MODE --dataset $DATASET --reps $REPS --output $OUTPUT $EXTRA"

echo "============================================================"
echo "KIRBy full experiment run"
echo "  Mode:    $MODE"
echo "  Reps:    $REPS"
echo "  Dataset: $DATASET"
echo "  Output:  $OUTPUT"
echo "  Extra:   ${EXTRA:-none}"
echo "============================================================"

set -e  # stop on first error

# --- Complexity experiments ---
echo ""
echo ">>> C01: Rep Characterization"
python tests/experiments/complexity/c01_rep_characterization.py $COMMON

echo ""
echo ">>> C02: Complexity Grid"
python tests/experiments/complexity/c02_complexity_grid.py $COMMON

echo ""
echo ">>> C04: Importance as Metric"
python tests/experiments/complexity/c04_importance_as_metric.py $COMMON

echo ""
echo ">>> C05: Learning Curves"
python tests/experiments/complexity/c05_learning_curves.py $COMMON

echo ""
echo ">>> C06: Within-Family Gradient"
python tests/experiments/complexity/c06_within_family_gradient.py $COMMON

echo ""
echo ">>> C07: Data Complexity Partitions"
python tests/experiments/complexity/c07_data_complexity_partitions.py $COMMON

echo ""
echo ">>> C08: Calibration"
python tests/experiments/complexity/c08_calibration.py $COMMON

echo ""
echo ">>> C09: Formal Analysis"
python tests/experiments/complexity/c09_formal_analysis.py $COMMON

echo ""
echo ">>> C10: Noise Robustness"
python tests/experiments/complexity/c10_noise_robustness.py $COMMON

echo ""
echo ">>> C11: Allocation Strategies"
python tests/experiments/complexity/c11_allocation_strategies.py $COMMON

echo ""
echo ">>> C12: Adaptive Pruning"
python tests/experiments/complexity/c12_adaptive_pruning.py $COMMON

# --- Similarity experiments ---
echo ""
echo ">>> S01: Similarity Baseline"
python tests/experiments/similarity/s01_similarity_baseline.py $COMMON

echo ""
echo ">>> S02: Hybrid Similarity"
python tests/experiments/similarity/s02_hybrid_similarity.py $COMMON

echo ""
echo ">>> S03: Feature Concentration"
python tests/experiments/similarity/s03_feature_concentration.py $COMMON

echo ""
echo ">>> S04: Activity Cliffs"
python tests/experiments/similarity/s04_activity_cliffs.py $COMMON

echo ""
echo ">>> S05: kNN Prediction"
python tests/experiments/similarity/s05_knn_prediction.py $COMMON

echo ""
echo ">>> S06: Applicability Domain"
python tests/experiments/similarity/s06_applicability_domain.py $COMMON

echo ""
echo ">>> S07: Noise Bridge"
python tests/experiments/similarity/s07_noise_bridge.py $COMMON

echo ""
echo ">>> S08: Optimal Dimension"
python tests/experiments/similarity/s08_optimal_dimension.py $COMMON

# --- Cross-dataset experiments (skipped — need all datasets loaded) ---
# Run these separately for full runs:
#   python tests/experiments/complexity/c03_cross_dataset_transfer.py --mode full --reps $REPS --output $OUTPUT --datasets logd caco2 herg qm9
#   python tests/experiments/similarity/s09_cross_dataset_summary.py --mode full --reps $REPS --output $OUTPUT --datasets logd caco2 herg qm9

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE (except C03, S09 — run cross-dataset separately)"
echo "Results in: $OUTPUT"
echo "============================================================"
