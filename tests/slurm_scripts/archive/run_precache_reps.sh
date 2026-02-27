#!/bin/bash
#SBATCH --job-name=precache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# Pre-generate and cache representations for a dataset.
# After this completes, experiment runs with --reps all load instantly from cache.
#
# Usage:
#   sbatch run_precache_reps.sh logd
#   sbatch run_precache_reps.sh caco2
#   sbatch run_precache_reps.sh herg
#   sbatch run_precache_reps.sh qm9

DATASET=${1:-logd}

echo "=== Pre-caching all representations for ${DATASET} ==="
echo "Started: $(date)"
echo ""

python -c '
import sys
sys.path.insert(0, "tests")
from find_best_hybrid_molecular import load_dataset, get_representations

dataset = "'"${DATASET}"'"
print(f"Loading {dataset}...")
smiles, labels, task_type = load_dataset(dataset)
print(f"  {len(smiles)} molecules")

print("Generating all representations (will cache to ~/.cache/kirby/reps/)...")
reps = get_representations(smiles, "all", cache=True)

print()
print("Cached representations:")
for name, arr in sorted(reps.items()):
    print(f"  {name}: {arr.shape}")
print(f"Total: {len(reps)} representations")
'

echo ""
echo "Finished: $(date)"
