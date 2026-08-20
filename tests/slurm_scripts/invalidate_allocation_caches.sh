#!/bin/bash
# Invalidate every cached artefact that was produced by an ALLOCATION decision,
# after a change to hybrid.py / find_best_hybrid.py. DRY RUN by default.
#
#   bash tests/slurm_scripts/invalidate_allocation_caches.sh              # show, delete nothing
#   bash tests/slurm_scripts/invalidate_allocation_caches.sh --datasets herg_ki logd
#   bash tests/slurm_scripts/invalidate_allocation_caches.sh --go         # actually delete
#
# WHY THIS IS NEEDED FOR THE CURRENT WORKING-TREE CHANGE
# ------------------------------------------------------
# hybrid.py create_hybrid now draws its internal validation fold from
# GroupKFold instead of a single GroupShuffleSplit draw (src/kirby/hybrid.py,
# the `if groups is not None` branch of create_hybrid). That split feeds
# rep_dict_train / rep_dict_val, which is what every allocator scores reps on.
# So EVERY allocation decision can move -- but NOTHING in any cache key records
# it. The caches below therefore hit and return pre-change answers. This is the
# silent-wrong-answer class, not the wasted-compute class.
#
# WHY EACH ARTEFACT (verified by code read, not assumed)
# -----------------------------------------------------
#   result_db/<ds>.db            -- REQUIRED. Row key = IDENTITY_COLS (tier,
#                                   speed, labels_hash, groups_hash, n_folds,
#                                   seed, cv_splitter_id) + CONFIG_COLS
#                                   (allocation_method, budget, importance_method,
#                                   model, ...) -- src/kirby/result_cache.py:44-60.
#                                   The internal val split is in NEITHER. Same
#                                   config -> same row -> stale score served.
#                                   That file's own header says: table layout /
#                                   behaviour change => WIPE the DBs and re-run.
#
#   searches/*.pkl (+ .ckpt.pkl) -- REQUIRED. Whole SearchResult per
#                                   (dataset, tier, aug_strategy, speed).
#                                   _make_search_meta (find_best_hybrid.py:337)
#                                   records data/models/budgets/allocation_methods
#                                   etc., but nothing about create_hybrid's
#                                   internal split -- so the meta check passes and
#                                   the stale search is served whole.
#
#   hybrid_evals/<ds>/*.pkl      -- REQUIRED. Per-(rep_subset, config, fold,
#                                   model) hybrid scores. _make_hybrid_eval_meta
#                                   (find_best_hybrid.py:671) hashes the OUTER
#                                   train/val idx only; the inner split that
#                                   moved is invisible to it.
#
#   hybrid_fold_cache/*.pkl      -- REQUIRED, AND THE MOST DANGEROUS ONE. Stores
#                                   create_hybrid's feature_info (the selected
#                                   column indices themselves). Its key is
#                                   sha256(train_idx + rep_shapes + alloc +
#                                   importance + budget + evaluator + aug bits)
#                                   -- tests/experiments/_shared.py:58-69. No
#                                   split geometry, no code version. A hit
#                                   replays the OLD selected features verbatim.
#
#   create_hybrid_cache/*.pkl    -- REQUIRED if present. Same 24-hex key format
#                                   as hybrid_fold_cache; legacy location, no
#                                   current writer in the tree. Same failure mode.
#
#   hybrid_search/*.pkl          -- REQUIRED if present. Legacy 16-hex hashed
#                                   search cache that predates searches/. Still
#                                   read by tests/supervisor_presentation.py:33
#                                   and tests/analysis/print_hybrid_vs_baseline.py:206.
#
#   hybrid_registry.json         -- REQUIRED if present. Index over the above.
#
#   tests/analysis/blueprints/*  -- REQUIRED if present. Traced from the old
#                                   best hybrid, i.e. from an old allocation.
#
# WHAT IS DELIBERATELY **KEPT** (allocation-independent; the expensive stuff)
# --------------------------------------------------------------------------
#   baselines/       per-(dataset, rep, model, metric, n_folds, seed) single-rep
#                    CV scores. Keyed on rep_hash + labels/groups hash
#                    (find_best_hybrid.py:129-148, _make_baseline_meta). Never
#                    goes through create_hybrid. This is the big one -- keeping
#                    it is most of the compute saved.
#   folds/           CV split indices per (dataset, n_folds, seed, grouped).
#   rep_cache/       per-dataset representation matrices (.npy).
#   reps/            content-hashed representation matrices from the CLI wrappers.
#   ft_embeddings/   per-(dataset, factory_id, rep, fold) fine-tuned embeddings.
#   esm2_per_residue/, finetuned_reps/, lp_pdbbind/, *.csv  raw data + embeddings.
#
#   NOTE ON FEATURE IMPORTANCE: per-fold importance scores are NOT a separate
#   on-disk cache. They live inside searches/*.pkl as SearchResult.importance_cache
#   (find_best_hybrid.py:1002), keyed (fold_idx, importance_method). They are
#   still VALID after this change -- importance is computed on the outer fold's
#   train rows, not on create_hybrid's inner split -- but they are lost with the
#   search pickle they ride in, and recomputed on the next search
#   (find_best_hybrid.py:3773). If that recompute cost matters, salvage them out
#   of the pickles BEFORE running this with --go; there is no salvage step here.
#
#   NOTE ON EXPERIMENT JSONs: tests/results/<cat>/<exp>/<ds>.json are NOT touched
#   by this script. run_dataset.py resumes by skipping any experiment whose JSON
#   exists (complexity_helpers.py:1133, similarity_helpers.py:959,
#   noise_helpers.py:334), so a re-run would silently keep pre-change numbers.
#   Use `--force-rerun` on run_dataset.py rather than deleting them.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

GO=0
DATASETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --go) GO=1; shift ;;
    --datasets) shift; while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    -h|--help) sed -n '1,95p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

CACHE="${KIRBY_CACHE_DIR:-$HOME/.cache/kirby}"
SCOPED=0; [[ ${#DATASETS[@]} -gt 0 ]] && SCOPED=1

echo "cache root : $CACHE"
echo "mode       : $([[ $GO -eq 1 ]] && echo 'DELETE (--go)' || echo 'DRY RUN')"
if [[ $SCOPED -eq 1 ]]; then
  echo "scope      : ${DATASETS[*]}"
else
  echo "scope      : ALL datasets"
fi
echo

[[ -d "$CACHE" ]] || { echo "cache root does not exist: $CACHE"; exit 1; }

targets=()

# Does a path belong to one of the requested datasets? Unscoped runs take all.
# Boundary-anchored on purpose: a bare prefix match would pull `logd_smoke` and
# `logd_like` in when the user asked for `logd`. Accepted forms are the ones the
# writers actually produce:
#   result_db     <ds>.db                         (result_cache.db_path_for)
#   searches      <ds>__tier__aug__speed[...].pkl (_search_cache_path)
#                 <ds>_fold<N>__...               (per-fold callers)
#   blueprints    <ds>_*                          (looser; blueprint tracer)
in_scope() {
  [[ $SCOPED -eq 0 ]] && return 0
  local name; name="$(basename "$1")"
  local loose="${2:-no}"
  local ds
  for ds in "${DATASETS[@]}"; do
    [[ "$name" == "$ds.db" ]] && return 0
    [[ "$name" == "$ds"__* ]] && return 0
    [[ "$name" == "$ds"_fold* ]] && return 0
    [[ "$loose" == "yes" && "$name" == "$ds"_* ]] && return 0
  done
  return 1
}

add_dir() {  # add_dir <abs dir> <label> <dataset-attributable: yes|no> [loose-match]
  local dir="$1" label="$2" attributable="$3" loose="${4:-no}" n=0
  [[ -d "$dir" ]] || { printf '── %-22s (absent)\n' "$label"; return; }
  if [[ "$attributable" == "no" && $SCOPED -eq 1 ]]; then
    printf '── %-22s SKIPPED: filenames are content hashes, not dataset names,\n' "$label"
    printf '   %-22s so it cannot be filtered by --datasets. Re-run without\n' ""
    printf '   %-22s --datasets to clear it (it is REQUIRED for correctness).\n' ""
    return
  fi
  local f
  while IFS= read -r f; do
    if [[ "$attributable" == "yes" ]]; then
      in_scope "$f" "$loose" || continue
    fi
    targets+=("$f"); n=$((n+1))
  done < <(find "$dir" -maxdepth 1 -type f 2>/dev/null)
  printf '── %-22s %5d file(s)\n' "$label" "$n"
}

# hybrid_evals is one subdir per dataset, so scope on the SUBDIR name.
add_hybrid_evals() {
  local dir="$CACHE/hybrid_evals" n=0
  [[ -d "$dir" ]] || { printf '── %-22s (absent)\n' "hybrid_evals"; return; }
  local d f
  for d in "$dir"/*/; do
    [[ -d "$d" ]] || continue
    if [[ $SCOPED -eq 1 ]]; then
      local base; base="$(basename "$d")"
      local hit=0 ds
      for ds in "${DATASETS[@]}"; do [[ "$base" == "$ds" ]] && hit=1; done
      [[ $hit -eq 1 ]] || continue
    fi
    while IFS= read -r f; do targets+=("$f"); n=$((n+1)); done \
      < <(find "$d" -maxdepth 1 -type f 2>/dev/null)
  done
  printf '── %-22s %5d file(s)\n' "hybrid_evals" "$n"
}

add_dir "$CACHE/result_db"            "result_db"            yes
add_dir "$CACHE/searches"             "searches"             yes
add_hybrid_evals
add_dir "$CACHE/hybrid_fold_cache"    "hybrid_fold_cache"    no
add_dir "$CACHE/create_hybrid_cache"  "create_hybrid_cache"  no
add_dir "$CACHE/hybrid_search"        "hybrid_search"        no
if [[ -f "$CACHE/hybrid_registry.json" && $SCOPED -eq 0 ]]; then
  targets+=("$CACHE/hybrid_registry.json")
  printf '── %-22s %5d file(s)\n' "hybrid_registry.json" 1
fi
add_dir "tests/analysis/blueprints"   "analysis/blueprints"  yes  yes

echo
echo "KEPT (untouched): baselines/ folds/ rep_cache/ reps/ ft_embeddings/"
echo "                  esm2_per_residue/ finetuned_reps/ lp_pdbbind/ *.csv"
echo "                  tests/results/**/*.json  (use run_dataset.py --force-rerun)"
echo
echo "${#targets[@]} artefact(s) selected."
[[ ${#targets[@]} -eq 0 ]] && exit 0

if [[ $GO -eq 0 ]]; then
  echo
  echo "First 20 selected paths (no recursive sizing -- no du on this filesystem):"
  printf '   %s\n' "${targets[@]:0:20}"
  [[ ${#targets[@]} -gt 20 ]] && echo "   ... and $(( ${#targets[@]} - 20 )) more"
  echo
  echo "DRY RUN — nothing deleted. Re-run with --go to delete the list above."
  exit 0
fi

echo
read -r -p "Delete these ${#targets[@]} paths under $CACHE? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted"; exit 1; }
removed=0
for t in "${targets[@]}"; do rm -f -- "$t" && removed=$((removed+1)); done
echo "removed $removed artefact(s)."
echo "Next: re-run precompute, then the experiments with --force-rerun."
