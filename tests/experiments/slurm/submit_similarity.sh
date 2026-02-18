#!/bin/bash
# Submit similarity experiment pipeline with SLURM dependency chains.
#
# Usage:
#   ./submit_similarity.sh intermediary logd fast
#   ./submit_similarity.sh full logd fast+hf
#
# Phase 1 (independent): s01, s02, s08
# Phase 2 (after s02):   s03, s04, s05, s06, s07
# Phase 3 (cross-dataset): s09 (after all datasets)

MODE=${1:-intermediary}
DATASET=${2:-logd}
REPS=${3:-fast}
EXTRA_ARGS="${@:4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$(dirname "$EXP_DIR")")"
OUTPUT="tests/results"

# SLURM defaults
PARTITION="long"
MEM="64G"
CPUS=8
MAIL="adelaide.punt@stcatz.ox.ac.uk"

# Time estimates
if [ "$MODE" = "intermediary" ]; then
    T_S01="0:15:00"
    T_S02="1:00:00"
    T_S03="0:30:00"
    T_S04="1:00:00"
    T_S05="1:00:00"
    T_S06="2:00:00"
    T_S07="0:30:00"
    T_S08="2:00:00"
else
    T_S01="0:30:00"
    T_S02="4:00:00"
    T_S03="1:00:00"
    T_S04="2:00:00"
    T_S05="2:00:00"
    T_S06="4:00:00"
    T_S07="1:00:00"
    T_S08="6:00:00"
fi

echo "=== Similarity Pipeline ==="
echo "Mode: $MODE, Dataset: $DATASET, Reps: $REPS"
echo "Extra args: $EXTRA_ARGS"
echo ""

cd "$PROJECT_DIR"
. setup.sh

COMMON="--mode $MODE --dataset $DATASET --reps $REPS --output $OUTPUT $EXTRA_ARGS"

# Phase 1: Independent experiments (parallel)
J1=$(sbatch --parsable --job-name="s01_${DATASET}" --partition=$PARTITION \
    --time=$T_S01 --mem=$MEM --cpus-per-task=$CPUS --mail-user=$MAIL \
    --wrap="python tests/experiments/similarity/s01_similarity_baseline.py $COMMON")
echo "s01 submitted: $J1"

J2=$(sbatch --parsable --job-name="s02_${DATASET}" --partition=$PARTITION \
    --time=$T_S02 --mem=$MEM --cpus-per-task=$CPUS --mail-user=$MAIL \
    --wrap="python tests/experiments/similarity/s02_hybrid_similarity.py $COMMON")
echo "s02 submitted: $J2"

J8=$(sbatch --parsable --job-name="s08_${DATASET}" --partition=$PARTITION \
    --time=$T_S08 --mem=$MEM --cpus-per-task=$CPUS --mail-user=$MAIL \
    --wrap="python tests/experiments/similarity/s08_optimal_dimension.py $COMMON")
echo "s08 submitted: $J8"

# Phase 2: Depends on s02
J3=$(sbatch --parsable --dependency=afterok:$J2 --job-name="s03_${DATASET}" \
    --partition=$PARTITION --time=$T_S03 --mem=$MEM --cpus-per-task=$CPUS \
    --mail-user=$MAIL \
    --wrap="python tests/experiments/similarity/s03_feature_concentration.py $COMMON --s02-results $OUTPUT/s02_hybrid_similarity")
echo "s03 submitted: $J3 (after s02:$J2)"

J4=$(sbatch --parsable --dependency=afterok:$J2 --job-name="s04_${DATASET}" \
    --partition=$PARTITION --time=$T_S04 --mem=$MEM --cpus-per-task=$CPUS \
    --mail-user=$MAIL \
    --wrap="python tests/experiments/similarity/s04_activity_cliffs.py $COMMON --s02-results $OUTPUT/s02_hybrid_similarity")
echo "s04 submitted: $J4 (after s02:$J2)"

J5=$(sbatch --parsable --dependency=afterok:$J2 --job-name="s05_${DATASET}" \
    --partition=$PARTITION --time=$T_S05 --mem=$MEM --cpus-per-task=$CPUS \
    --mail-user=$MAIL \
    --wrap="python tests/experiments/similarity/s05_knn_prediction.py $COMMON --s02-results $OUTPUT/s02_hybrid_similarity")
echo "s05 submitted: $J5 (after s02:$J2)"

J6=$(sbatch --parsable --dependency=afterok:$J2 --job-name="s06_${DATASET}" \
    --partition=$PARTITION --time=$T_S06 --mem=$MEM --cpus-per-task=$CPUS \
    --mail-user=$MAIL \
    --wrap="python tests/experiments/similarity/s06_applicability_domain.py $COMMON --s02-results $OUTPUT/s02_hybrid_similarity")
echo "s06 submitted: $J6 (after s02:$J2)"

J7=$(sbatch --parsable --dependency=afterok:$J2 --job-name="s07_${DATASET}" \
    --partition=$PARTITION --time=$T_S07 --mem=$MEM --cpus-per-task=$CPUS \
    --mail-user=$MAIL \
    --wrap="python tests/experiments/similarity/s07_noise_bridge.py $COMMON --s02-results $OUTPUT/s02_hybrid_similarity")
echo "s07 submitted: $J7 (after s02:$J2)"

echo ""
echo "All similarity jobs submitted for $DATASET."
echo "Job IDs: s01=$J1 s02=$J2 s03=$J3 s04=$J4 s05=$J5 s06=$J6 s07=$J7 s08=$J8"
echo ""
echo "For cross-dataset summary (s09), run after all datasets complete:"
echo "  sbatch --dependency=afterok:J2_logd:J2_caco2:... \\"
echo "    --wrap='python tests/experiments/similarity/s09_cross_dataset_summary.py \\"
echo "      --datasets logd caco2 herg qm9 --results-dir $OUTPUT --output $OUTPUT'"
