#!/bin/bash
# Rep-pool cleanup EVIDENCE — one array task per dataset. Per dataset: per-rep OOF
# prediction vectors, solo scores, the rep x rep PREDICTION-correlation matrix, and
# a greedy decorrelation sweep over several cutoffs. Writes
# $KIRBY_CACHE_DIR/rep_prune/<ds>/. Selects NOTHING — gathers the numbers the user
# reads to drop duplicate-signal reps down to ~10 per dataset.
#
# --account and --partition are LIVE STATE — pass them at submit time. Real-run
# tunables via --export: full-feature reps (CAP=0) for faithful correlations; the
# concat check stays OFF (that is combination work, not cleanup):
#   N=$(python -c "import sys;sys.path.insert(0,'tests');sys.path.insert(0,'tests/experiments');\
#from experiments.complexity.complexity_helpers import DATASET_REPS;print(len(DATASET_REPS))")
#   sbatch --account=stat-cadd --partition=medium --array=0-$((N-1))%15 \
#          --export=ALL,KIRBY_PRUNE_FOLDS=3,KIRBY_PRUNE_CAP=0 \
#          tests/slurm_scripts/run_rep_prune.sh
# Cost is ~reps CVs per dataset (concat off). Full-width LM-embedding antibody/DTA
# sets are the long poles; if a task hits the wall, resubmit that --array index.
#SBATCH --job-name=rep_prune
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. setup.sh

# Same roster ordering as run_validity_gate.sh (DATASET_REPS insertion order):
# herg=0, herg_ki=1, ..., cdr_flexibility=42. Submit --array=0-$((N-1)) to cover
# the full roster (the validity run only went 0-41 and missed cdr_flexibility).
mapfile -t DATASETS < <(python -c "
import sys; sys.path.insert(0,'tests'); sys.path.insert(0,'tests/experiments')
from experiments.complexity.complexity_helpers import DATASET_REPS
for d in DATASET_REPS:
    print(d)
")

ds="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "rep_prune array task ${SLURM_ARRAY_TASK_ID} -> ${ds}  (KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR})"
python -u tests/rep_prune_evidence.py "${ds}"
