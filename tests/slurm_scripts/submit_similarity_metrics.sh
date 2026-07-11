#!/bin/bash
#SBATCH --job-name=sim_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Similarity-metric battery on the MoleculeACE 30 cliff-rich ChEMBL targets.
# ONE job (no split — per SLURM-resources guidance): build datasets, precompute
# transformer embeddings (needs the GPU; pin V100 per ARC convention), then run
# the classical grid + learned embeddings + triplet metric-learner through the
# activity-grounded battery, aggregated with paired Wilcoxon + SRD.
#
# The GPU is used only for the ChemBERTa/MolFormer embedding pass; the long tail
# is CPU battery work on the same node. If GPU queue time is the bottleneck, drop
# --gres and set WHICH="" to skip transformer embeddings (classical + triplet
# still run; triplet is pure-torch and works CPU-only).
#
# PREREQUISITES (run once on the LOGIN node, inside the venv):
#   . setup.sh
#   pip install MoleculeACE                       # ships the 30 benchmark CSVs
#   python - <<'PY'                               # pre-download HF models (compute nodes may lack network)
#   from transformers import AutoModel, AutoTokenizer
#   for m in ['seyonec/ChemBERTa-zinc-base-v1','ibm/MoLFormer-XL-both-10pct']:
#       AutoTokenizer.from_pretrained(m, trust_remote_code=True)
#       AutoModel.from_pretrained(m, trust_remote_code=True)
#   PY
#
# Output (under $KIRBY_CACHE_DIR/similarity_metrics/sim_data/):
#   sim_<TARGET>.npz, emb_<TARGET>_<model>.npy,
#   moleculeace_rows.json, moleculeace_summary.json  <- the headline table
#
# rsync the two JSONs back for local analysis (JSON-only, per the server rules).

set -euo pipefail
REPO="/data/stat-ecr/scat9264/KIRBy"
DIR="tests/experiments/similarity_metrics"
WHICH="${WHICH:-chemberta molformer}"

cd "${REPO}"
git pull 2>/dev/null || true
. ./setup.sh
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error

echo "=== [1/3] build MoleculeACE datasets ==="
python "${DIR}/build_moleculeace.py"

if [[ -n "${WHICH}" ]]; then
  echo "=== [2/3] transformer embeddings (${WHICH}) ==="
  python "${DIR}/learned_embeddings.py" --which ${WHICH}
else
  echo "=== [2/3] skipped (WHICH empty) ==="
fi

echo "=== [3/3] run the 30-target battery ==="
python "${DIR}/run_battery_moleculeace.py"

echo "Done. Fetch: \$KIRBY_CACHE_DIR/similarity_metrics/sim_data/moleculeace_summary.json"
