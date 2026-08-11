#!/bin/bash
#SBATCH --job-name=scaf_hyb_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --account=stat-ecr
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

# Standalone comparison: do the scaffold-diff mutation reps (dphyschem,
# dphyschem_pos) beat the base reps in the REAL pipeline, and does the hybrid
# pick their features? Runs tests/experiments/opig_scaffold_hybrid.py, which
# injects the two scaffold reps into the real rep pool and runs honest nested CV
# under a distinct cache tier ('scaffold_cmp') — it does NOT touch the datasets'
# existing result DBs and requires NO framework changes.
#
# Usage:
#   sbatch submit_scaffold_hybrid.sh                       # all 4 DMS datasets
#   sbatch submit_scaffold_hybrid.sh flab_expression_koenig
#
# LIVE-STATE CHECK BEFORE LAUNCH (do not trust these directives blindly):
#   run tests/slurm_scripts/where_to_submit.sh and confirm the partition/account
#   that are actually available, then override at sbatch time if needed, e.g.
#   sbatch --partition=medium --time=2-00:00:00 submit_scaffold_hybrid.sh
#
# DEPENDENCY: this reuses the cached language-model rep matrices for these
# datasets. If those embeddings are NOT already cached on the server, load_data
# will try to build them (esm2/prot_t5/ablang2/... need GPU) and this CPU job
# will be slow or fail — precompute the embeddings first in that case.

DATASETS="${*:-flab_expression_koenig flab_binding_koenig flab_expression_adams flab_binding_adams}"

cd /data/stat-ecr/scat9264/KIRBy

# Pull the CURRENTLY CHECKED-OUT branch, not main — this experiment lives on
# the similarity-metrics-study feature branch. Checkout the branch on the server
# once; this keeps it up to date. (Differs from submit_dataset.sh, which pulls main.)
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
git pull --ff-only origin "${BRANCH}" 2>&1 || echo "WARNING: git pull failed; running on-disk code"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

. setup.sh
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}

echo "=== scaffold-diff vs base reps (hybrid pipeline) ==="
echo "Datasets: ${DATASETS}"
echo "Branch:   ${BRANCH} (${COMMIT})"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "Start:    $(date)"

python -u tests/experiments/opig_scaffold_hybrid.py --mode full --datasets ${DATASETS}
RC=$?

echo "End: $(date)  (exit ${RC})"
exit ${RC}
