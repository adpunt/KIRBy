#!/bin/bash
# Similarity-Optimized Hybrid — submit one CPU job per molecular dataset.
#
# Runs tests/experiments/similarity_optimized_hybrid.py, which finds a hybrid by
# optimizing SIMILARITY PRESERVATION (objective='similarity_sac') and compares it
# against the PREDICTIVE-optimized hybrid and the best single rep on:
#   Payoff 1 — held-out SAC + violation rate (similarity quality)
#   Payoff 2 — noise mitigation: sweeps every noise-mitigation method (grouped by
#              similarity-reliance) and reports detection F1 vs the known noise
#              mask + downstream recovery, under injected label noise
#
# ── CACHE REUSE ────────────────────────────────────────────────────────────
# Caching is ALWAYS on. The experiment resolves the cache root itself
# ($KIRBY_CACHE_DIR — set by setup.sh to /data/stat-ecr/scat9264/.cache/kirby —
# or ~/.cache/kirby), so no --cache-dir flag is needed. It:
#   * REUSES per-rep baseline + per-fold importance content-caches (keyed by rep
#     hash) across the predictive and similarity searches and across reruns.
#   * CACHES both searches in the per-dataset SQLite store so reruns are cheap:
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

# ── QUEUE (overridable at submit time; matches submit_priority/opig_priority) ─
# account : stat-ecr (higher fair-share / all billing goes here). Override with
#           ACCOUNT=stat-cadd (partitions are often more free there → sooner start).
# partition: long by default (long/medium share the same nodes; short is swamped
#           at 2000+ pending). A 24h wall backfills into idle long slots where the
#           7-day jobs cannot. Override with PARTITION=medium.
# wall    : 24h by request (qm9 may not fully finish — it still writes whatever
#           datasets/seeds completed; reruns resume from cache). Override with WALL=.
ACCOUNT="${ACCOUNT:-stat-ecr}"
PARTITION="${PARTITION:-long}"
WALL="${WALL:-23:59:00}"
# QOS: empty = default (standard). Set QOS=priority to use the priority QOS — the
# real queue-jump lever when jobs are stuck on (Priority)/fair-share. Partition
# (long vs medium) is NOT the lever: they share the same nodes.
QOS="${QOS:-}"

WRAP_PREFIX="cd ${REPO} && git pull 2>/dev/null; . ./setup.sh && export PYTHONWARNINGS=ignore && export TRANSFORMERS_VERBOSITY=error"
STDERR_FILTER="2> >(grep -v -e 'DEPRECATION WARNING' -e 'threadpoolctl' -e 'ctypes callback' >&2)"

# Sound molecular datasets (DATASET_STATUS §5/§9). qm9 is the largest → more wall/mem.
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(caco2 logd herg herg_ki qm9 adme_hlm adme_rlm adme_mdr1)
fi

# run_ds <dataset> [mem]
run_ds() {
    local ds=$1
    local mem=${2:-64G}
    # --cache-dir omitted: the experiment resolves $KIRBY_CACHE_DIR (or
    # ~/.cache/kirby) itself and caching is always on, so baseline/importance
    # caches under the standard cache root are reused with zero fragility.
    local cmd="mkdir -p ${OUT} && python ${EXP} --datasets ${ds} --seeds ${SEEDS} --output ${OUT}"
    local J
    J=$(sbatch \
        --job-name="simhyb_${ds}" \
        --account="${ACCOUNT}" \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time="${WALL}" --partition="${PARTITION}" \
        --mem="${mem}" ${QOS:+--qos="${QOS}"} \
        --mail-user="${EMAIL}" --mail-type=END,FAIL \
        --wrap="${WRAP_PREFIX} && ${cmd} ${STDERR_FILTER}" \
        | awk '{print $4}')
    echo "  ${J}  simhyb_${ds}  (acct=${ACCOUNT}, part=${PARTITION}, qos=${QOS:-standard}, mem=${mem}, time=${WALL}, CPU)"
}

echo "==================================================="
echo "Similarity-Optimized Hybrid — per-dataset submission"
echo "  Output: ${OUT}/   Account: ${ACCOUNT}   Partition: ${PARTITION}   QOS: ${QOS:-standard}   Wall: ${WALL}"
echo "  Caching: ON (cache root resolved by the experiment; existing caches reused)"
echo "==================================================="
for ds in "${DATASETS[@]}"; do
    case "${ds}" in
        qm9)  run_ds "${ds}" 96G ;;   # largest n → more mem (wall still 24h)
        *)    run_ds "${ds}" 64G ;;
    esac
done
echo "Done. Monitor with: squeue -u \$USER -n simhyb_${DATASETS[0]}"
