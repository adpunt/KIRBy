#!/bin/bash
# READ-ONLY. Reports the state of the cache on the server. Deletes nothing,
# writes nothing except one output file under /tmp.
#
#   bash tests/server_cache_check.sh
#
# Then paste /tmp/kirby_cache_check.txt back.
#
# What each section answers, and why it is here (CACHE_AUDIT_2026_09_10.md):
#   commit      -- the 2026-09-10/11 changes are working-tree edits until they
#                  are pushed, and eight commits carry that date without
#                  carrying them. A date check passes on the wrong code; the
#                  HASH and the three greps below do not.
#   layers      -- which cache directories exist and how large. searches/ and
#                  folds/ should be absent or stale: nothing writes them.
#                  hybrid_evals/ likewise, switched off 2026-09-11.
#   libraries   -- until 2026-09-11 a missing lightgbm or torch was silently
#                  replaced by another library and the score was stored under
#                  the name that was asked for. If a compute node cannot import
#                  one of these, every file named for it on this machine is
#                  suspect.
#   databases   -- per stored database: whether the current code can open it at
#                  all, and whether any ONE search in it was written by two
#                  different representation pools, which is proof that a search
#                  read another pool's numbers.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

OUT=/tmp/kirby_cache_check.txt
# setup.sh sets KIRBY_CACHE_DIR to the server's real root, which is what you
# want on ARC. Keep a caller-supplied one so this can be pointed elsewhere for
# a dry test: KIRBY_CACHE_DIR=/some/path bash tests/server_cache_check.sh
_CALLER_CACHE="${KIRBY_CACHE_DIR:-}"
KIRBY_SKIP_INSTALLS=1
export KIRBY_SKIP_INSTALLS
# shellcheck disable=SC1091
. ./setup.sh >/dev/null 2>&1 || true
if [ -n "${_CALLER_CACHE}" ]; then
    export KIRBY_CACHE_DIR="${_CALLER_CACHE}"
fi

{
echo "===== KIRBy cache check — $(date) ====="
echo
echo "----- commit -----"
git log -1 --format='%H  %ad  %s' --date=short
git rev-parse --abbrev-ref HEAD
_dirty=$(git status --porcelain src/kirby | wc -l | tr -d ' ')
if [ "${_dirty}" != "0" ]; then
    echo "  !! ${_dirty} file(s) under src/kirby are modified but not committed."
    echo "     The engine running here is NOT what is committed. Commit and push"
    echo "     from the laptop, pull here, and re-run this."
    git status --porcelain src/kirby | sed 's/^/       /'
fi
echo
echo "  the three 2026-09-10/11 changes (all three must print a line):"
grep -n "'objective'," src/kirby/result_cache.py || echo "  !! MISSING: objective is not an identity column"
grep -n "'rep_pool'," src/kirby/result_cache.py || echo "  !! MISSING: rep_pool is not an identity column"
grep -n "fold_scores" tests/slurm_scripts/wipe_allocation_caches.sh | head -1 || echo "  !! MISSING: fold_scores is not in the wipe list"
echo
echo "----- cache root -----"
echo "KIRBY_CACHE_DIR=${KIRBY_CACHE_DIR:-<UNSET>}"
if [ -z "${KIRBY_CACHE_DIR:-}" ] || [ ! -d "${KIRBY_CACHE_DIR}" ]; then
    echo "  !! no cache root — nothing further can be checked"
    exit 1
fi
echo
echo "----- layers (top-level entries) -----"
for d in result_db fold_scores baselines hybrid_fold_cache hybrid_evals \
         searches folds rep_cache reps ft_embeddings; do
    n=$(ls -1 "${KIRBY_CACHE_DIR}/$d" 2>/dev/null | wc -l | tr -d ' ')
    printf '  %-20s %s\n' "$d" "${n:-0}"
done
echo
echo "----- libraries, on a COMPUTE node (this is finding 4.4) -----"
srun -p short -n1 -t2 --mem=2G python -c \
  "import torch, lightgbm, xgboost; print('  all three import')" 2>&1 | tail -3 \
  || echo "  (srun unavailable — run 'python -c \"import torch, lightgbm, xgboost\"' on a compute node by hand)"
echo
echo "----- stored databases -----"
python - <<'PY'
import os, glob, sqlite3, sys
sys.path.insert(0, 'src')
try:
    from kirby.result_cache import ALL_COLS
except Exception as e:
    print(f"  !! cannot import result_cache ({e}) — wrong checkout?"); raise SystemExit
root = os.environ['KIRBY_CACHE_DIR']
files = sorted(glob.glob(os.path.join(root, 'result_db', '*.db')))
if not files:
    print("  no stored databases — nothing to check, and nothing to delete")
for p in files:
    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    try:
        have = {r[1] for r in c.execute('PRAGMA table_info(results)')}
        miss = [x for x in ALL_COLS if x not in have]
        if miss:
            print(f"  STALE {os.path.basename(p):48s} missing {miss}")
            continue
        n = c.execute('select count(*) from results').fetchone()[0]
        mixed = c.execute('''SELECT COUNT(*) FROM (SELECT 1 FROM results
            GROUP BY tier,objective,metric,task_type,labels_hash,groups_hash,
                     n_folds,seed,cv_splitter_id
            HAVING COUNT(DISTINCT rep_pool)>1)''').fetchone()[0]
        topk = c.execute("select count(*) from results "
                         "where allocation_method like 'top!_%' escape '!'"
                         ).fetchone()[0]
        flag = '  <-- MIXED POOLS' if mixed else ''
        print(f"  ok    {os.path.basename(p):48s} rows={n:<6d} "
              f"mixed_pool_searches={mixed} top_k_rows={topk}{flag}")
    finally:
        c.close()
PY
echo
echo "===== end ====="
} 2>&1 | tee "$OUT"

echo
echo "Wrote $OUT — paste it back."
