#!/bin/bash
#SBATCH --job-name=alloc_cache_wipe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# DESTRUCTIVE: deletes every cache layer that is downstream of an ALLOCATION
# decision, after a change to hybrid.py / find_best_hybrid.py.
#
# Runs on a compute node because hybrid_evals/ alone is hundreds of thousands of
# small files -- a metadata-heavy walk that should not run on the login node.
#
# Two-stage delete (same pattern as delete_av2_data.sh):
#   1. mv each target into ONE staging dir (atomic, instant). The moment this
#      finishes the caches are unreachable, so nothing can serve a stale hit
#      even though the actual delete is still running.
#   2. rm -rf the staging dir (slow, but on a compute node it does not matter).
# If the rm dies halfway, the sources are already gone from their real paths --
# re-running this script just resumes the cleanup.
#
# WHY THESE, AND ONLY THESE
# -------------------------
# create_hybrid's internal validation split moved from GroupShuffleSplit to
# GroupKFold (src/kirby/hybrid.py). That split feeds rep_dict_train/rep_dict_val,
# which is what every allocator scores reps on -- so allocations move, but the
# split appears in NO cache key. These layers therefore HIT and serve
# pre-change answers rather than missing and rebuilding:
#
#   result_db/            row key = IDENTITY_COLS + CONFIG_COLS
#                         (result_cache.py:44-60); inner split in neither.
#   searches/             _make_search_meta (find_best_hybrid.py:337) records
#                         nothing about the inner split.
#   hybrid_evals/         _make_hybrid_eval_meta (:671) hashes the OUTER
#                         train/val idx only.
#   hybrid_fold_cache/    WORST ONE. Stores the selected column indices
#                         themselves; key = sha256(train_idx + rep_shapes +
#                         alloc + importance + budget + evaluator + aug)
#                         (_shared.py:58-69). A hit replays the OLD features.
#   create_hybrid_cache/  legacy sibling of hybrid_fold_cache, same failure mode.
#   hybrid_search/        legacy pre-searches/ hashed cache.
#   hybrid_registry.json  index over the above.
#
# KEPT ON PURPOSE (never passes through create_hybrid; this is most of the
# compute): baselines/ folds/ rep_cache/ reps/ ft_embeddings/ esm2_per_residue/
# finetuned_reps/ and the raw data files. The script ABORTS if it cannot see
# baselines/ and rep_cache/ where it expects them -- that is the guard against
# a mispointed KIRBY_CACHE_DIR.
#
# NOT touched, but same failure mode: tests/results/<cat>/<exp>/<ds>.json.
# run_dataset.py skips any experiment whose JSON exists, so re-run with
# --force-rerun rather than deleting them.
#
# ALSO NOTE: per-fold feature importance is not a separate on-disk cache -- it
# lives inside searches/*.pkl as SearchResult.importance_cache
# (find_best_hybrid.py:1002). It is still VALID after this change (computed on
# the outer fold's train rows), but it goes with the search pickle and is
# recomputed on the next search (:3773).
#
# USAGE
#   sbatch tests/slurm_scripts/wipe_allocation_caches.sh
#   (log lands in slurm-<jobid>.out as usual)

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
. ./setup.sh

CACHE="${KIRBY_CACHE_DIR:-}"
STAGING_PARENT="${CACHE}"

echo "=========================================="
echo "ALLOCATION CACHE WIPE: $(date)"
echo "  cache root: ${CACHE:-<UNSET>}"
echo "=========================================="

# ---- guards -------------------------------------------------------------
if [ -z "${CACHE}" ]; then
    echo "ABORT: KIRBY_CACHE_DIR is empty after sourcing setup.sh."
    exit 1
fi
if [ ! -d "${CACHE}" ]; then
    echo "ABORT: cache root does not exist: ${CACHE}"
    exit 1
fi
# Sanity: this must LOOK like a kirby cache, or we are pointed somewhere wrong.
# Both of these are KEEP dirs, so their presence also proves we are not about to
# delete the wrong tree.
if [ ! -d "${CACHE}/baselines" ] || [ ! -d "${CACHE}/rep_cache" ]; then
    echo "ABORT: ${CACHE} has no baselines/ and/or rep_cache/ — this does not"
    echo "       look like the kirby cache root. Refusing to delete anything."
    exit 1
fi

STAGING="${STAGING_PARENT}/ALLOC-CACHE-DELETE-STAGING"

TARGETS=(result_db searches hybrid_evals hybrid_fold_cache
         create_hybrid_cache hybrid_search hybrid_registry.json)

# ---- pre-wipe inventory (shallow only: no du, no recursive find) ---------
echo
echo "[before]"
for t in "${TARGETS[@]}"; do
    p="${CACHE}/${t}"
    if [ -d "${p}" ]; then
        printf '  %-22s %s top-level entrie(s)\n' "${t}" "$(ls -1 "${p}" 2>/dev/null | wc -l)"
    elif [ -f "${p}" ]; then
        printf '  %-22s present (file)\n' "${t}"
    else
        printf '  %-22s absent\n' "${t}"
    fi
done
echo
echo "[keeping]"
for k in baselines folds rep_cache reps ft_embeddings; do
    p="${CACHE}/${k}"
    if [ -d "${p}" ]; then
        printf '  %-22s %s top-level entrie(s)\n' "${k}" "$(ls -1 "${p}" 2>/dev/null | wc -l)"
    else
        printf '  %-22s absent\n' "${k}"
    fi
done

# ---- stage 1: atomic moves ----------------------------------------------
echo
mkdir -p "${STAGING}" || { echo "ABORT: cannot create ${STAGING}"; exit 1; }
moved=0
for t in "${TARGETS[@]}"; do
    p="${CACHE}/${t}"
    if [ -e "${p}" ]; then
        # Timestamped suffix so a resumed run never collides with a previous
        # staged copy that stage 2 did not finish removing.
        if mv "${p}" "${STAGING}/${t}.$$" 2>/dev/null; then
            echo "[stage 1] staged ${t}"
            moved=$((moved+1))
        else
            echo "[stage 1] FAILED to stage ${t} — leaving it in place"
        fi
    fi
done
echo "[stage 1] ${moved} target(s) staged at $(date)"
echo "[stage 1] caches are now UNREACHABLE — no stale hit is possible from here on"

if [ "${moved}" -eq 0 ] && [ -z "$(ls -A "${STAGING}" 2>/dev/null)" ]; then
    echo "nothing to delete — already clean"
    rmdir "${STAGING}" 2>/dev/null
    echo "done: $(date)"
    exit 0
fi

# ---- stage 2: the slow part ---------------------------------------------
echo
echo "[stage 2] rm -rf ${STAGING} — this is the slow part"
rm -rf "${STAGING}"
RC=$?
echo "[stage 2] rm exit: ${RC} at $(date)"

# ---- post-wipe verification ---------------------------------------------
echo
echo "[after] surviving entries in ${CACHE}:"
ls -1 "${CACHE}" 2>/dev/null | sed 's/^/  /'
echo
echo "Next: re-run precompute, then experiments with --force-rerun."
echo "done: $(date)"
exit ${RC}
