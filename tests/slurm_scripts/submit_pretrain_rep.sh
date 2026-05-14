#!/bin/bash
#SBATCH --job-name=pretrain_rep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# PT-Rep Phase 1 — pre-trained representation experiment.
# Runs tests/experiments/pretrain_rep/pretrain_rep_experiment.py for one
# (PT source, FT target) pair.
#
# Usage:
#   sbatch tests/slurm_scripts/submit_pretrain_rep.sh <pt_source> <ft_dataset> [pt_task] [conditions]
#
# Examples:
#   sbatch tests/slurm_scripts/submit_pretrain_rep.sh oas_species ginkgo_hic
#   sbatch tests/slurm_scripts/submit_pretrain_rep.sh ginkgo_gdpa1 ginkgo_hic ginkgo_kd
#   sbatch tests/slurm_scripts/submit_pretrain_rep.sh oas_species ginkgo_hic "" A,B,H
#
# Args:
#   1 pt_source    PT corpus name (pt_registry.list_pt_sources()). REQUIRED.
#   2 ft_dataset   Fine-tuning target dataset name. REQUIRED.
#   3 pt_task      PT sub-task (e.g. a GDPa1 property). Optional, default none.
#   4 conditions   Comma-separated subset of A,B,F,G,H. Optional, default all.

PT_SOURCE="$1"
FT_DATASET="$2"
PT_TASK="${3:-}"
CONDITIONS="${4:-A,B,F,G,H}"

if [ -z "$PT_SOURCE" ] || [ -z "$FT_DATASET" ]; then
    echo "ERROR: need <pt_source> and <ft_dataset>."
    echo "Usage: sbatch submit_pretrain_rep.sh <pt_source> <ft_dataset> [pt_task] [conditions]"
    exit 1
fi

cd /data/stat-ecr/scat9264/KIRBy
git pull 2>/dev/null
. ./setup.sh
export PYTHONWARNINGS=ignore

echo "PT-Rep Phase 1 experiment"
echo "PT source:  $PT_SOURCE"
echo "FT dataset: $FT_DATASET"
echo "PT task:    ${PT_TASK:-<none>}"
echo "Conditions: $CONDITIONS"
echo "Start:      $(date)"

PT_TASK_ARG=""
if [ -n "$PT_TASK" ]; then
    PT_TASK_ARG="--pt-task $PT_TASK"
fi

python tests/experiments/pretrain_rep/pretrain_rep_experiment.py \
    --pt-source "$PT_SOURCE" \
    --ft-dataset "$FT_DATASET" \
    $PT_TASK_ARG \
    --conditions "$CONDITIONS" \
    --mode full

echo "Done: $(date)"
