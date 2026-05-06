#!/bin/bash
# Single-dataset SLURM job: run precompute_hybrids.py with adaptive pruning.
#
# Required env vars:
#   DATASET        Dataset name (e.g. caco2, flab_thermo_jain)
#   KIND           "standard" or "distance"
#
# Optional env vars (defaults shown):
#   PRUNING_MODE=default_pruning   (default_pruning | aggressive_pruning)
#   PRUNING_CHECKPOINTS=           override checkpoint budgets (comma-list)
#   PRUNING_TAIL_DROP=             override tail-drop fraction (e.g. 0.5)
#   PRUNING_MIN_SAMPLES_GATE=      override n_samples threshold
#   MEM=64G                        128G for nt_*/gb_*
#
# Submit:
#   sbatch --export=DATASET=caco2,KIND=standard,PRUNING_MODE=aggressive_pruning \
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
PRUNING_MODE="${PRUNING_MODE:-default_pruning}"

cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

DIST_FLAG=""
[[ "$KIND" == "distance" ]] && DIST_FLAG="--distance"

EXTRA_FLAGS=()
[[ -n "${PRUNING_CHECKPOINTS:-}" ]] && EXTRA_FLAGS+=(--pruning-checkpoints "${PRUNING_CHECKPOINTS}")
[[ -n "${PRUNING_TAIL_DROP:-}"   ]] && EXTRA_FLAGS+=(--pruning-tail-drop "${PRUNING_TAIL_DROP}")
[[ -n "${PRUNING_MIN_SAMPLES_GATE:-}" ]] && EXTRA_FLAGS+=(--pruning-min-samples-gate "${PRUNING_MIN_SAMPLES_GATE}")

echo "=========================================="
echo "  pc_${DATASET}_all_${KIND}"
echo "  pruning-mode=${PRUNING_MODE}"
[[ ${#EXTRA_FLAGS[@]} -gt 0 ]] && echo "  extra: ${EXTRA_FLAGS[*]}"
echo "  start=$(date)"
echo "=========================================="

python tests/precompute_hybrids.py \
    "${DATASET}" \
    --speed full \
    --pruning-mode "${PRUNING_MODE}" \
    "${EXTRA_FLAGS[@]}" \
    ${DIST_FLAG}

echo "=========================================="
echo "  done=$(date)"
echo "=========================================="
