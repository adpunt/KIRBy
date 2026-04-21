#!/bin/bash
#SBATCH --job-name=c02_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=medium
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Verify the aug_strategy='none'→'uncapped' cache collision fix.
#
# Runs c02 on ginkgo_heparin via the same pathway that produced the 9255
# stale hybrid_evals last time (run_dataset.py → complexity_helpers.run_all
# → precompute_hybrid_searches → find_best_hybrid).
#
# Expected behaviour after the fix:
#   - [cache] summary at end should show near-zero hybrid_eval stale
#   - Runtime << 8h (last run timed out at 12h on 80% stale)
#
# To change dataset: edit DATASET below.
# To test a different experiment: edit EXPERIMENT below (c01–c24).

set -euo pipefail

DATASET=ginkgo_heparin
EXPERIMENT=c02

cd /data/stat-cadd/scat9264/KIRBy
git pull
. setup.sh

export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

echo "=== $(date -u +%FT%TZ) — c02 verify run ==="
echo "  dataset=${DATASET}  experiment=${EXPERIMENT}"
echo "  commit=$(git rev-parse --short HEAD)"
echo ""

python tests/experiments/run_dataset.py complexity "${DATASET}" \
    --experiments "${EXPERIMENT}" \
    --output tests/results/c02_verify

echo ""
echo "=== $(date -u +%FT%TZ) — done ==="
