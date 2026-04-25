#!/bin/bash
# Single-dataset SLURM job: run precompute_hybrids.py with multi-stage pruning.
#
# Required env vars:
#   DATASET                  Dataset name (e.g. caco2, flab_thermo_jain)
#   KIND                     "standard" or "distance"
#
# Optional env vars (defaults shown):
#   PRUNING_PRESET=baseline                 (one of SCHEDULE_PRESETS)
#   MODEL_OVERRIDE_PRESET=gp_aggressive     (one of MODEL_OVERRIDE_PRESETS)
#   MEM=64G                                 (use 128G for nt_*/gb_*)
#
# Submit:
#   sbatch --export=DATASET=caco2,KIND=standard \
#       tests/slurm_scripts/run_precompute_pruned.sh
#
#   sbatch --export=DATASET=caco2,KIND=distance,PRUNING_PRESET=light,MODEL_OVERRIDE_PRESET=gp_svm_aggressive \
#       tests/slurm_scripts/run_precompute_pruned.sh

#SBATCH --partition=long
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

set -euo pipefail

: "${DATASET:?DATASET env var required}"
: "${KIND:?KIND env var required (standard|distance)}"
PRUNING_PRESET="${PRUNING_PRESET:-baseline}"
MODEL_OVERRIDE_PRESET="${MODEL_OVERRIDE_PRESET:-gp_aggressive}"

cd /data/stat-cadd/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

DIST_FLAG=""
[[ "$KIND" == "distance" ]] && DIST_FLAG="--distance"

echo "=========================================="
echo "  pc_${DATASET}_all_${KIND}"
echo "  pruning-preset=${PRUNING_PRESET}"
echo "  model-override-preset=${MODEL_OVERRIDE_PRESET}"
echo "  partition=long, time=7d, mem=$(scontrol show job $SLURM_JOB_ID | awk '/MinMemory/ {print $1}' || echo unknown)"
echo "  start=$(date)"
echo "=========================================="

python tests/precompute_hybrids.py \
    "${DATASET}" \
    --speed full \
    --pruning-preset "${PRUNING_PRESET}" \
    --model-override-preset "${MODEL_OVERRIDE_PRESET}" \
    ${DIST_FLAG}

echo "=========================================="
echo "  done=$(date)"
echo "=========================================="
