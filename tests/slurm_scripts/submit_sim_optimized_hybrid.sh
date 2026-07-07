#!/bin/bash
# Similarity-Optimized Hybrid — submit one CPU job per molecular dataset.
#
# Runs tests/experiments/similarity_optimized_hybrid.py, which finds a hybrid by
# optimizing SIMILARITY PRESERVATION (objective='similarity_sac') and compares it
# against the PREDICTIVE-optimized hybrid and the best single rep on:
#   Payoff 1 — held-out SAC + violation rate (similarity quality)
#   Payoff 2 — kNN label-denoising recovery under injected label noise
#
# ── CACHE REUSE ────────────────────────────────────────────────────────────
# --cache-dir "$KIRBY_CACHE_DIR" makes the run:
#   * REUSE per-rep baseline + per-fold importance content-caches (keyed by rep
#     hash) across the predictive and similarity searches and across reruns.
#   * CACHE both searches in the per-dataset SQLite store so reruns are cheap:
#       predictive comparator -> tier 'general'
#       similarity search     -> tier 'general_sim' (fresh namespace; new work)
# Both searches are CONTROLLED (rf model only, budgets < 300, aggressive
# pruning) — an apples-to-apples A/B that differs only in the objective. We do
# NOT pull from the full precompute pool: on a cache miss that path runs a heavy
# default-model (svm/mlp) search and can segfault on large molecular sets.
# The sim tier does NOT collide with any existing predictive/distance rows
# (tier is part of the cache identity).
#
# Datasets: the sound, scaffold-split, precompute-built molecular set
# (DATASET_STATUS_2026_07_07 §5/§9). herg's stale graph_kernel rep is dropped in
# the experiment; chembl_qsar is intentionally excluded (never precomputed).
#
# CPU-only (RandomForest + numpy; no GPU). Search space is small by design
# (budgets < 300, aggressive pruning) — see the experiment module.
#
# Usage (login node):
#   bash tests/slurm_scripts/submit_sim_optimized_hybrid.sh            # all datasets
#   bash tests/slurm_scripts/submit_sim_optimized_hybrid.sh logd herg  # subset
#
# Output: tests/results/sim_optimized_hybrid/<dataset>.json (+ summary.json)

set -euo pipefail

REPO="/data/stat-ecr/scat9264/KIRBy"
EXP="tests/experiments/similarity_optimized_hybrid.py"
OUT="tests/results/sim_optimized_hybrid"
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
SEEDS=3

WRAP_PREFIX="cd ${REPO} && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' >&2)"

# Sound molecular datasets (DATASET_STATUS §5/§9). qm9 is the largest → more wall/mem.
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(caco2 logd herg herg_ki qm9 adme_hlm adme_rlm adme_mdr1)
fi

# run_ds <dataset> [mem] [wall]
run_ds() {
    local ds=$1
    local mem=${2:-64G}
    local wall=${3:-23:59:00}
    local cmd="mkdir -p ${OUT} && python ${EXP} --datasets ${ds} --seeds ${SEEDS} --cache-dir \"\${KIRBY_CACHE_DIR}\" --output ${OUT}"
    local J
    J=$(sbatch \
        --job-name="simhyb_${ds}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${wall}" --partition=long \
        --mem="${mem}" \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  ${J}  simhyb_${ds}  (mem=${mem}, time=${wall}, CPU)"
}

echo "==================================================="
echo "Similarity-Optimized Hybrid — per-dataset submission"
echo "  Output: ${OUT}/   Cache: \$KIRBY_CACHE_DIR (reused)"
echo "==================================================="
for ds in "${DATASETS[@]}"; do
    case "${ds}" in
        qm9)  run_ds "${ds}" 96G 35:59:00 ;;   # largest n
        *)    run_ds "${ds}" 64G 23:59:00 ;;
    esac
done
echo "Done. Monitor with: squeue -u \$USER -n simhyb_${DATASETS[0]}"
