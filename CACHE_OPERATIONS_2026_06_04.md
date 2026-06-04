# Cache operations — when & how to remove stale cache entries (2026-06-04)

Run these **on the server** (where `$KIRBY_CACHE_DIR` lives). **No `find`, no `du`** (they wreck the ARC filesystem) — these use `ls` on specific subdirs + globs, which are fast and don't recurse. Deletion is always review-first.

## Cache layout
```
$KIRBY_CACHE_DIR/            # default ~/.cache/kirby ; on ARC = /data/stat-ecr/scat9264/.cache/kirby
  searches/      {dataset}__{tier}__{aug}__{speed}.pkl        # full hybrid-search result (+ .ckpt.pkl)
  baselines/     {dataset}__{rep}__{model}__{metric}__{nf}f_{seed}.pkl
  hybrid_evals/  per-(subset,model,budget,...) cells          # filenames include subset content-hashes
  folds/         {dataset}__{nf}f_{seed}...                   # materialized CV splits
  reps/          cached representation embeddings             # may be content-hashed (no dataset in name)
```
`searches`/`baselines`/`folds` embed the **dataset name** in the filename → glob-removable. `reps` and some `hybrid_evals` are **content-hashed** → use the python script (inspects `_cache_meta`).

## ⚠️ WHEN you MUST manually purge (the CACHE-1 caveat)
The CACHE-1 anti-churn fix makes the cache **ignore rep content hashes**. So a cache entry auto-invalidates only when **shape / labels / n_samples / dataset_name / n_folds / metric** change. If a fix changes a representation's **VALUES but not its shape/labels** (e.g. the Graphinity mutation-position parser fix), the cache will **silently reuse the stale/corrupt embeddings** — you must purge by hand. Cases that DO auto-invalidate (no action): InChIKey dedup (row count changes), task_type / metric changes, the StratifiedGroupKFold change (fold structure changes → folds cache misses and rebuilds).

## 1. Graphinity (corrupt mutant sequences → purge before re-running) — REQUIRED
```bash
cd /data/stat-ecr/scat9264/KIRBy        # repo root on ARC
. setup.sh                               # so $KIRBY_CACHE_DIR is set
python tests/purge_graphinity_cache.py                 # DRY-RUN: lists what it would delete
python tests/purge_graphinity_cache.py --delete        # after reviewing the list
```
(The script walks only $KIRBY_CACHE_DIR, matches on filename + `_cache_meta.dataset_name` in {graphinity_foldx, graphinity_rosetta}, dry-run by default. It catches the content-hashed `reps/` entries that a glob would miss.)

## 2. Force-recompute ONE dataset (e.g. after any loader/value fix) — review-first glob
```bash
DS=graphinity_foldx                       # the dataset to invalidate
ls -lh $KIRBY_CACHE_DIR/searches/${DS}__*.pkl* 2>/dev/null      # review first
ls -lh $KIRBY_CACHE_DIR/baselines/${DS}__*.pkl 2>/dev/null
ls -lh $KIRBY_CACHE_DIR/folds/${DS}__*.pkl 2>/dev/null
# then, once you've confirmed the list:
rm $KIRBY_CACHE_DIR/searches/${DS}__*.pkl*  $KIRBY_CACHE_DIR/baselines/${DS}__*.pkl  $KIRBY_CACHE_DIR/folds/${DS}__*.pkl
# reps/ for that dataset (content-hashed) → use the python script pattern:
python tests/purge_graphinity_cache.py --targets ${DS}            # dry-run; --delete to remove
```

## 3. Inspect one cache file's metadata (why did/didn't it hit?)
```bash
python -c "import pickle,sys; m=pickle.load(open(sys.argv[1],'rb')).get('_cache_meta',{}); print({k:m[k] for k in sorted(m) if k!='code_timestamp'})" \
  $KIRBY_CACHE_DIR/searches/graphinity_foldx__general__none__full.pkl
```

## 4. NO purge needed for the SCREENING→FULL budget change
The precompute always used FULL_BUDGETS regardless of the screening setting, so `searches/` files are unaffected by the 3i fix. The 7 experiments that read `config['budgets']` only change their own downstream sweep, not the cached search.

## 5. Compare in-session vs SLURM cache alignment (PCB-50 verification, optional)
```bash
python -c "import pickle; a=pickle.load(open('A.pkl','rb'))['_cache_meta']; b=pickle.load(open('B.pkl','rb'))['_cache_meta']; print({k:(a.get(k),b.get(k)) for k in set(a)|set(b) if k!='code_timestamp' and a.get(k)!=b.get(k)})"
```
Empty `{}` = aligned (ignore `code_timestamp`).

## RULE
Never blanket-delete the cache. Always: (1) `ls`/dry-run to see the list, (2) confirm it's only the intended dataset/rep, (3) then `rm`/`--delete`. The python script enforces this (dry-run default + `_cache_meta` inspection).
