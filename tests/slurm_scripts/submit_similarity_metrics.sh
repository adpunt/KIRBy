#!/bin/bash
# Similarity-metric battery on the MoleculeACE 30 cliff-rich ChEMBL targets.
#
# NOTHING about the queue is hardcoded. Account/partition/QOS are LIVE STATE —
# run where_to_submit.sh and pass what it tells you. Defaults here are only a
# starting point and are overridable at submit time.
#
# GPU IS OPTIONAL AND OFF BY DEFAULT. The GPU is used ONLY to precompute the
# ChemBERTa/MolFormer embeddings. Requesting --gres=gpu on unavailable nodes
# means the job sits in PD (ReqNodeNotAvail) forever. The classical metric grid,
# the triplet metric-learner, and the whole noise sigma x rate sweep are
# CPU-ONLY and give you the core result without a GPU. Ask for the GPU only when
# you know GPU nodes are actually available.
#
# Usage (login node):
#   # CPU only (default) — classical + triplet + noise sweep:
#   bash tests/slurm_scripts/submit_similarity_metrics.sh
#
#   # override the queue from where_to_submit.sh:
#   ACCOUNT=stat-cadd PARTITION=medium bash tests/slurm_scripts/submit_similarity_metrics.sh
#
#   # add the transformer arm (ONLY if GPU nodes are free):
#   GPU=1 PARTITION=<gpu-partition> bash tests/slurm_scripts/submit_similarity_metrics.sh
#
# PREREQUISITES (login node, once, inside the venv):
#   . setup.sh
#   pip install MoleculeACE                    # ships the 30 benchmark CSVs
#   # only if GPU=1, pre-download the models (compute nodes may lack network):
#   python -c "from transformers import AutoModel,AutoTokenizer; \
#     [ (AutoTokenizer.from_pretrained(m,trust_remote_code=True), \
#        AutoModel.from_pretrained(m,trust_remote_code=True)) \
#       for m in ['seyonec/ChemBERTa-zinc-base-v1','ibm/MoLFormer-XL-both-10pct'] ]"
#
# Output: $KIRBY_CACHE_DIR/similarity_metrics/sim_data/
#   moleculeace_summary.json   <- the headline table (rsync this back)
#   moleculeace_rows.json

set -euo pipefail

REPO="${REPO:-/data/stat-ecr/scat9264/KIRBy}"
DIR="tests/experiments/similarity_metrics"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"

# ── QUEUE: live state. Override from where_to_submit.sh output. ──────────────
ACCOUNT="${ACCOUNT:-stat-ecr}"
PARTITION="${PARTITION:-medium}"
WALL="${WALL:-23:59:00}"
QOS="${QOS:-}"
MEM="${MEM:-64G}"
CPUS="${CPUS:-8}"

# ── GPU: OFF unless GPU=1. Only affects the transformer-embedding arm. ───────
GPU="${GPU:-0}"
if [[ "${GPU}" == "1" ]]; then
    GRES_FLAG="--gres=gpu:1"
    WHICH="${WHICH:-chemberta molformer}"
else
    GRES_FLAG=""
    WHICH=""          # skip transformers entirely -> no GPU needed
fi

CMD="cd ${REPO} && git pull 2>/dev/null; . ./setup.sh \
&& export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error \
&& python ${DIR}/build_moleculeace.py \
&& { [ -z \"${WHICH}\" ] || python ${DIR}/learned_embeddings.py --which ${WHICH}; } \
&& python ${DIR}/run_battery_moleculeace.py"

echo "==================================================================="
echo "Similarity-metric battery (MoleculeACE 30 targets)"
echo "  account=${ACCOUNT}  partition=${PARTITION}  qos=${QOS:-standard}"
echo "  wall=${WALL}  mem=${MEM}  cpus=${CPUS}"
if [[ "${GPU}" == "1" ]]; then
    echo "  GPU: REQUESTED (transformer arm: ${WHICH})"
    echo "  !! if GPU nodes are unavailable this will sit in PD (ReqNodeNotAvail)."
else
    echo "  GPU: none (CPU-only: classical grid + triplet learner + noise sweep)"
fi
echo "==================================================================="

J=$(sbatch \
    --job-name=sim_metrics \
    --account="${ACCOUNT}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" \
    --mem="${MEM}" --time="${WALL}" --partition="${PARTITION}" \
    ${QOS:+--qos="${QOS}"} ${GRES_FLAG} \
    --mail-user="${EMAIL}" --mail-type=END,FAIL \
    --wrap="${CMD}" | awk '{print $4}')

echo "  submitted ${J}"
echo "  watch:  squeue -j ${J}"
echo "  if it shows PD with ReqNodeNotAvail, the partition/gres is wrong —"
echo "  re-run where_to_submit.sh and resubmit with ACCOUNT=/PARTITION= set."
