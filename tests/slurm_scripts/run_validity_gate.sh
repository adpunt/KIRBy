#!/bin/bash
# Step-1 validity gate — one array task per surviving dataset.
# Scores every base rep under the dataset's group CV (lightgbm; +ridge for
# nucleotide) and writes $KIRBY_CACHE_DIR/validity/<ds>.json with a PASS/FAIL.
#
# --account and --partition are LIVE STATE — pass them at submit time, e.g.:
#   N=$(python -c "import sys;sys.path.insert(0,'tests');sys.path.insert(0,'tests/experiments');\
#from experiments.complexity.complexity_helpers import DATASET_REPS;\
#print(len([d for d in DATASET_REPS if d!='cdr_flexibility']))")
#   sbatch --account=stat-cadd --partition=medium --array=0-$((N-1))%15 \
#          tests/slurm_scripts/run_validity_gate.sh
#SBATCH --job-name=validity_gate
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# Surviving roster minus cdr_flexibility (needs a load_data path — wired separately).
mapfile -t DATASETS < <(python -c "
import sys; sys.path.insert(0,'tests'); sys.path.insert(0,'tests/experiments')
from experiments.complexity.complexity_helpers import DATASET_REPS
for d in DATASET_REPS:
    if d != 'cdr_flexibility':
        print(d)
")

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "validity array task ${SLURM_ARRAY_TASK_ID} -> ${ds}  (KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR})"
python -u tests/validity_gate.py "${ds}"
