#!/bin/bash
#SBATCH --job-name=ptrep_mtransfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# PT-Rep P5 — model transfer on a pre-trained hybrid representation.
# Builds a KIRBy hybrid on the PT corpus, pre-trains a DNN on it, fine-tunes
# that DNN on the FT target, and compares against a scratch DNN on the same
# representation.
#
# Usage:
#   sbatch tests/slurm_scripts/submit_pretrain_rep_model_transfer.sh <pt_source> <ft_dataset> [pt_task]
#
# Examples:
#   sbatch tests/slurm_scripts/submit_pretrain_rep_model_transfer.sh human_chembl herg_ki
#
# Both PT and FT must be regression tasks (PretrainedDNNRegressor is
# regression-only). Args:
#   1 pt_source    PT corpus name. REQUIRED.
#   2 ft_dataset   Fine-tuning target dataset name. REQUIRED.
#   3 pt_task      PT sub-task. Optional, default none.

PT_SOURCE="$1"
FT_DATASET="$2"
PT_TASK="${3:-}"

if [ -z "$PT_SOURCE" ] || [ -z "$FT_DATASET" ]; then
    echo "ERROR: need <pt_source> and <ft_dataset>."
    echo "Usage: sbatch submit_pretrain_rep_model_transfer.sh <pt_source> <ft_dataset> [pt_task]"
    exit 1
fi

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

echo "PT-Rep P5 — model transfer"
echo "PT source:  $PT_SOURCE"
echo "FT dataset: $FT_DATASET"
echo "PT task:    ${PT_TASK:-<none>}"
echo "Start:      $(date)"

PT_TASK_ARG=""
if [ -n "$PT_TASK" ]; then
    PT_TASK_ARG="--pt-task $PT_TASK"
fi

python tests/experiments/pretrain_rep/pretrain_rep_model_transfer.py \
    --pt-source "$PT_SOURCE" \
    --ft-dataset "$FT_DATASET" \
    $PT_TASK_ARG \
    --mode full

echo "Done: $(date)"
