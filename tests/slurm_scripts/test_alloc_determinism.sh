#!/bin/bash
#SBATCH --job-name=alloc_det
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

# Run the allocation-scoring determinism test across 6 datasets spanning
# antibody, molecular, and reaction domains.
#
# Each dataset: runs alloc scoring twice on identical inputs, compares
# per-rep scores + top-5 rankings + silent-failure counts. Summary
# verdict printed at the bottom.
#
# To run on a different dataset set:
#   sbatch tests/slurm_scripts/test_alloc_determinism.sh ginkgo_heparin caco2
#   sbatch tests/slurm_scripts/test_alloc_determinism.sh all

set -euo pipefail

cd /data/stat-ecr/scat9264/KIRBy
git pull
. setup.sh

export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

echo "=== $(date -u +%FT%TZ) — alloc determinism test ==="
echo "  commit=$(git rev-parse --short HEAD)"
echo "  args=$*"
echo ""

python tests/scripts/test_alloc_determinism.py "$@"

echo ""
echo "=== $(date -u +%FT%TZ) — done ==="
