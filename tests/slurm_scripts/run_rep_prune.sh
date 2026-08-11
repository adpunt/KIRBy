#!/bin/bash
# Rep-pool cleanup EVIDENCE — runs the datasets passed as ARGUMENTS successively
# inside ONE job (no array). rep_prune_evidence.py loops over them, so a whole
# domain is a single queue entry instead of many small ones.
#
# Not usually submitted directly — use submit_rep_prune.sh, which fires one of
# these per domain with that domain's dataset list. Direct use:
#   sbatch --account=X --partition=Y --export=ALL,KIRBY_PRUNE_FOLDS=3,KIRBY_PRUNE_CAP=0 \
#          tests/slurm_scripts/run_rep_prune.sh logd herg_ki caco2
#
# Per dataset it writes $KIRBY_CACHE_DIR/rep_prune/<ds>/ : per-rep solo scores, the
# rep x rep PREDICTION-correlation matrix (the complementarity data), and a greedy
# decorrelation sweep. Selects nothing — the user picks the complementary ~10.
#SBATCH --job-name=rep_prune
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

if [ "$#" -eq 0 ]; then
    echo "ERROR: no datasets given. Pass them as arguments (or use submit_rep_prune.sh)." >&2
    exit 1
fi
echo "rep_prune job -> ${#} datasets: $*   (KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR})"
python -u tests/rep_prune_evidence.py "$@"
