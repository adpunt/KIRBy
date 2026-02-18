#!/bin/bash
# Submit complexity experiment pipeline with SLURM dependency chains.
#
# Usage:
#   ./submit_complexity.sh intermediary logd fast
#   ./submit_complexity.sh full logd fast+hf
#   ./submit_complexity.sh full logd all --delta --augment
#
# Phase 1 (independent): c01, c02, c04, c05, c06, c10
# Phase 2 (after c02):   c07, c08
# Phase 3 (after c02+c06): c09
# Phase 4 (cross-dataset, separate): c03

MODE=${1:-intermediary}
DATASET=${2:-logd}
REPS=${3:-fast}
EXTRA_ARGS="${@:4}"  # e.g. --delta --augment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$(dirname "$EXP_DIR")")"
OUTPUT="tests/results"

# SLURM defaults
PARTITION="long"
MEM="64G"
CPUS=8
MAIL="adelaide.punt@stcatz.ox.ac.uk"

# Time estimates (generous)
if [ "$MODE" = "intermediary" ]; then
    T_C01="0:30:00"
    T_C02="2:00:00"
    T_C04="1:00:00"
    T_C05="2:00:00"
    T_C06="4:00:00"
    T_C07="1:00:00"
    T_C08="1:00:00"
    T_C09="2:00:00"
    T_C10="6:00:00"
else
    T_C01="1:00:00"
    T_C02="8:00:00"
    T_C04="2:00:00"
    T_C05="8:00:00"
    T_C06="12:00:00"
    T_C07="4:00:00"
    T_C08="4:00:00"
    T_C09="6:00:00"
    T_C10="24:00:00"
fi

echo "=== Complexity Pipeline ==="
echo "Mode: $MODE, Dataset: $DATASET, Reps: $REPS"
echo "Extra args: $EXTRA_ARGS"
echo ""

cd "$PROJECT_DIR"
. setup.sh

COMMON="--mode $MODE --dataset $DATASET --reps $REPS --output $OUTPUT $EXTRA_ARGS"

# Phase 1: Independent experiments (parallel)
J1=$(sbatch --parsable --job-name="c01_${DATASET}" --partition=$PARTITION \
    --time=$T_C01 --mem=$MEM --cpus-per-task=$CPUS --mail-user=$MAIL \
    --wrap="python tests/experiments/complexity/c01_rep_characterization.py $COMMON")
echo "c01 submitted: $J1"

J2=$(sbatch --parsable --job-name="c02_${DATASET}" --partition=$PARTITION \
    --time=$T_C02 --mem=$MEM --cpus-per-task=$CPUS --mail-user=$MAIL \
    --wrap="python tests/experiments/complexity/c02_complexity_grid.py $COMMON")
echo "c02 submitted: $J2"

J4=$(sbatch --parsable --job-name="c04_${DATASET}" --partition=$PARTITION \
    --time=$T_C04 --mem=$MEM --cpus-per-task=$CPUS --mail-user=$MAIL \
    --wrap="python tests/experiments/complexity/c04_importance_as_metric.py $COMMON")
echo "c04 submitted: $J4"

J5=$(sbatch --parsable --job-name="c05_${DATASET}" --partition=$PARTITION \
    --time=$T_C05 --mem=$MEM --cpus-per-task=$CPUS --mail-user=$MAIL \
    --wrap="python tests/experiments/complexity/c05_learning_curves.py $COMMON")
echo "c05 submitted: $J5"

J6=$(sbatch --parsable --job-name="c06_${DATASET}" --partition=$PARTITION \
    --time=$T_C06 --mem=$MEM --cpus-per-task=$CPUS --mail-user=$MAIL \
    --wrap="python tests/experiments/complexity/c06_within_family_gradient.py $COMMON")
echo "c06 submitted: $J6"

J10=$(sbatch --parsable --job-name="c10_${DATASET}" --partition=$PARTITION \
    --time=$T_C10 --mem=$MEM --cpus-per-task=$CPUS --mail-user=$MAIL \
    --wrap="python tests/experiments/complexity/c10_noise_robustness.py $COMMON")
echo "c10 submitted: $J10"

# Phase 2: Depends on c02
J7=$(sbatch --parsable --dependency=afterok:$J2 --job-name="c07_${DATASET}" \
    --partition=$PARTITION --time=$T_C07 --mem=$MEM --cpus-per-task=$CPUS \
    --mail-user=$MAIL \
    --wrap="python tests/experiments/complexity/c07_data_complexity_partitions.py $COMMON --c02-results $OUTPUT/c02_complexity_grid")
echo "c07 submitted: $J7 (after c02:$J2)"

J8=$(sbatch --parsable --dependency=afterok:$J2 --job-name="c08_${DATASET}" \
    --partition=$PARTITION --time=$T_C08 --mem=$MEM --cpus-per-task=$CPUS \
    --mail-user=$MAIL \
    --wrap="python tests/experiments/complexity/c08_calibration.py $COMMON --c02-results $OUTPUT/c02_complexity_grid")
echo "c08 submitted: $J8 (after c02:$J2)"

# Phase 3: Depends on c02 + c06
J9=$(sbatch --parsable --dependency=afterok:$J2:$J6 --job-name="c09_${DATASET}" \
    --partition=$PARTITION --time=$T_C09 --mem=$MEM --cpus-per-task=$CPUS \
    --mail-user=$MAIL \
    --wrap="python tests/experiments/complexity/c09_formal_analysis.py $COMMON --c02-results $OUTPUT/c02_complexity_grid --c06-results $OUTPUT/c06_within_family_gradient")
echo "c09 submitted: $J9 (after c02:$J2 + c06:$J6)"

echo ""
echo "All complexity jobs submitted for $DATASET."
echo "Job IDs: c01=$J1 c02=$J2 c04=$J4 c05=$J5 c06=$J6 c07=$J7 c08=$J8 c09=$J9 c10=$J10"
echo ""
echo "For cross-dataset (c03), run after all datasets complete:"
echo "  sbatch --dependency=afterok:J2_logd:J2_caco2:J2_herg:J2_qm9 \\"
echo "    --wrap='python tests/experiments/complexity/c03_cross_dataset_transfer.py \\"
echo "      --mode $MODE --datasets logd caco2 herg qm9 --reps $REPS --output $OUTPUT'"
