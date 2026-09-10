#!/bin/bash
# Invalidate cached artefacts for the datasets whose FOLDS or LABELS changed on
# 2026-07-09 (split + preprocessing fixes). DRY RUN by default.
#
#   bash tests/slurm_scripts/invalidate_refolded.sh          # show, delete nothing
#   bash tests/slurm_scripts/invalidate_refolded.sh --go     # actually delete
#
# WHY EACH ARTEFACT (they are not equally dangerous):
#
#   result_db/<ds>.db   -- REQUIRED. `labels_hash` and `groups_hash` are both in
#                          the PRIMARY KEY of the `results` table
#                          (src/kirby/result_cache.py:47,165), so rows written
#                          under the OLD folds can never be READ back under the
#                          new ones -- they miss and rebuild. Correct results are
#                          therefore safe either way. But check_precompute_status
#                          counts rows grouped by (tier, aug_strategy) ONLY
#                          (_scan_db), never by groups_hash -- so it would keep
#                          reporting these datasets DONE while every cached row is
#                          unreachable, landing them in the "ready now" bucket
#                          when they need a full rebuild. Delete, or the status
#                          tool lies about exactly the four datasets you refolded.
#
#   tests/results/<cat>/<exp>/<ds>.json  -- REQUIRED. run_dataset.py resumes by
#                          skipping any experiment whose output JSON already
#                          exists (complexity_helpers.py:1256,
#                          similarity_helpers.py:959, noise_helpers.py:250).
#                          Leave these in place and the re-run silently keeps the
#                          OLD leaky-split numbers. This is the one that bites.
#
#   tests/analysis/blueprints/<ds>_*    -- REQUIRED if present; traced from the
#                          old best hybrid.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

# Folds and/or labels changed for exactly these four:
#   flab_binding_phillips_h3  position_modulo -> edit_distance; label binarised
#   flab_binding_adams        position_modulo -> edit_distance; pKd + binarised
#   flab_expression_adams     position_modulo -> edit_distance
#   cdr_flexibility           per-CDR3 (size-1 groups) -> IGHV V-gene family
REFOLDED=(flab_binding_phillips_h3 flab_binding_adams flab_expression_adams cdr_flexibility)

GO=0; [[ "${1:-}" == "--go" ]] && GO=1
# Point at the real cache, and refuse if it is not obviously one. Neither script
# sourced setup.sh, so on the server KIRBY_CACHE_DIR was unset and CACHE fell back
# to $HOME/.cache/kirby -- which exists on the login node, passed the -d check,
# and deleted a near-empty directory while the real store under /data/stat-ecr
# went on serving pre-change allocations. Added 2026-09-10.
if [[ -z "${KIRBY_CACHE_DIR:-}" && -f ./setup.sh ]]; then
  # KIRBY_SKIP_INSTALLS=1 or setup.sh runs its whole pip pass -- and an HMMER
  # build if hmmscan is missing -- before a script that only deletes files.
  # shellcheck disable=SC1091
  KIRBY_SKIP_INSTALLS=1 . ./setup.sh >/dev/null 2>&1 || true
fi
CACHE="${KIRBY_CACHE_DIR:-}"
[[ -n "$CACHE" ]] || {
  echo "ABORT: KIRBY_CACHE_DIR is empty. Run '. ./setup.sh' first and check the"
  echo "       'cache root :' line below before passing --go. Refusing to guess"
  echo "       \$HOME/.cache/kirby."; exit 1; }

echo "cache root : $CACHE"
echo "mode       : $([[ $GO -eq 1 ]] && echo 'DELETE (--go)' || echo 'DRY RUN')"
echo

[[ -d "$CACHE" ]] || { echo "ABORT: cache root does not exist: $CACHE"; exit 1; }
# Both of these are KEEP directories, so their presence also proves this is the
# kirby cache root and not somewhere else that happens to exist.
[[ -d "$CACHE/baselines" && -d "$CACHE/rep_cache" ]] || {
  echo "ABORT: $CACHE has no baselines/ and/or rep_cache/ -- this does not look"
  echo "       like the kirby cache root. Refusing to delete anything."; exit 1; }

targets=()
for ds in "${REFOLDED[@]}"; do
  echo "── $ds"
  db="$CACHE/result_db/${ds}.db"
  if [[ -f "$db" ]]; then
    echo "   [REQUIRED] $(ls -lh "$db" | awk '{print $5}')  $db"
    targets+=("$db")
  else
    echo "   [REQUIRED] no result_db entry"
  fi
  while IFS= read -r j; do
    echo "   [REQUIRED] $j"
    targets+=("$j")
  done < <(find tests/results -name "${ds}.json" 2>/dev/null)
  while IFS= read -r b; do
    echo "   [REQUIRED] $b"
    targets+=("$b")
  done < <(find tests/analysis/blueprints -name "${ds}_*" 2>/dev/null)
done

echo
echo "${#targets[@]} artefact(s) selected."
if [[ $GO -eq 0 ]]; then
  echo "DRY RUN — nothing deleted. Re-run with --go to delete the list above."
  exit 0
fi

read -r -p "Delete these ${#targets[@]} paths? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted"; exit 1; }
for t in "${targets[@]}"; do rm -f -- "$t" && echo "removed $t"; done
echo "done. Re-run the four datasets; precompute will rebuild under the new folds."
