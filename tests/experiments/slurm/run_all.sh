#!/bin/bash
#SBATCH --job-name=all_exp
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Run all experiments sequentially in one job.
#
# Usage:
#   sbatch run_all.sh                          # intermediary logd fast
#   sbatch run_all.sh intermediary logd fast+hf
#   sbatch run_all.sh full logd fast+hf
#
# Phase 1: independent (c01,c02,c04,c05,c06,c10,s01,s02,s08)
# Phase 2: needs c02/c06/s02 results (c07,c08,c09,s03-s07)

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

MODE=${1:-intermediary}
DATASET=${2:-logd}
REPS=${3:-fast}
EXTRA="${@:4}"

COMMON="--mode $MODE --dataset $DATASET --reps $REPS --output tests/results $EXTRA"

echo "=== KIRBy Experiment Pipeline ==="
echo "Mode: $MODE, Dataset: $DATASET, Reps: $REPS, Extra: $EXTRA"
echo "Started: $(date)"
echo ""

run() {
    echo "--- $1 --- $(date)"
    python "$2" $COMMON ${@:3}
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "  FAILED (exit $RC)"
    else
        echo "  OK"
    fi
    echo ""
    return $RC
}

# Phase 1: Independent experiments
run "C01: Rep characterization"   tests/experiments/complexity/c01_rep_characterization.py
run "S01: Similarity baseline"    tests/experiments/similarity/s01_similarity_baseline.py
run "C04: Importance as metric"   tests/experiments/complexity/c04_importance_as_metric.py
run "C05: Learning curves"        tests/experiments/complexity/c05_learning_curves.py

# Core searches (needed by phase 2)
run "C02: Complexity grid"        tests/experiments/complexity/c02_complexity_grid.py
run "S02: Hybrid similarity"      tests/experiments/similarity/s02_hybrid_similarity.py
run "C06: Within-family gradient" tests/experiments/complexity/c06_within_family_gradient.py

# Independent but slower
run "S08: Optimal dimension"      tests/experiments/similarity/s08_optimal_dimension.py
run "C10: Noise robustness"       tests/experiments/complexity/c10_noise_robustness.py

# Phase 2: Depends on c02
run "C07: Data complexity partitions" tests/experiments/complexity/c07_data_complexity_partitions.py
run "C08: Calibration"            tests/experiments/complexity/c08_calibration.py \
    --c02-results tests/results/c02_complexity_grid

# Depends on c02 + c06
run "C09: Formal analysis"        tests/experiments/complexity/c09_formal_analysis.py \
    --c02-results tests/results/c02_complexity_grid \
    --c06-results tests/results/c06_within_family_gradient

# Phase 2: Depends on s02
run "S03: Feature concentration"  tests/experiments/similarity/s03_feature_concentration.py
run "S04: Activity cliffs"        tests/experiments/similarity/s04_activity_cliffs.py
run "S05: kNN prediction"         tests/experiments/similarity/s05_knn_prediction.py
run "S06: Applicability domain"   tests/experiments/similarity/s06_applicability_domain.py
run "S07: Noise bridge"           tests/experiments/similarity/s07_noise_bridge.py

echo "=== All experiments complete: $(date) ==="
