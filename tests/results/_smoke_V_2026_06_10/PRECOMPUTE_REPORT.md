# §V PRECOMPUTE agent report — precompute → SQLite result-cache handoff (2026-06-10)

**Verdict: PASS (clf + reg).** The precompute → experiment handoff through the new
per-dataset SQLite result store (`src/kirby/result_cache.py`) works end-to-end
locally for both a regression and a classification small dataset. Cold compute
writes the DB; an identical re-run reuses every row (0 recompute); a downstream
experiment-style call hits the precomputed rows; and adding a model reuses the
old rows and computes ONLY the new model — the exact core bug the redesign fixes.

## Environment / lane
- Python `/Users/apunt/anaconda3/bin/python` (3.11), single-thread env, run from repo root.
- `KIRBY_CACHE_DIR=/tmp/kirby_V_precompute` (started empty each run).
- No shared code edited. The only file added is a SMOKE driver in the experiments
  tree: `tests/experiments/_run_precompute_handoff.py`. It monkeypatches
  module-level objects in-process only (FULL_BUDGETS=[25], INTERNAL_EVALUATORS=['rf'],
  DEFAULT_MODELS={'rf':…}, SPEED_TIERS trims) — identical pattern to the existing
  `tests/experiments/_run_complexity_locally.py`; server jobs import fresh and are
  unaffected.

## How the handoff actually works (verified by reading the code)
- `precompute_hybrid_searches` (`_shared.py:2755`) calls `find_best_hybrid(...,
  cache=True, dataset_name=, tier=)`.
- `find_best_hybrid` (`find_best_hybrid.py:3648-3656`) opens
  `ResultCache(dataset, cache_dir)`, SELECTs all rows for the search IDENTITY,
  appends them to `results`, and prints `ResultCache: N stored configs reused;
  computing only the remainder of <total>`. The per-config loop (3667-3702) skips
  any config whose canonical key is already in the DB and INSERTs only newly
  computed finite-score rows.
- DB path: `$KIRBY_CACHE_DIR/result_db/<dataset>.db`. IDENTITY = tier, speed,
  labels_hash, groups_hash, n_folds, seed, cv_splitter_id. The MODEL/ranker/budget
  are per-ROW CONFIG columns, so changing them is a different SELECT, never a
  rebuild.
- An experiment hits this same store via `get_precomputed_search(precomputed=None)`
  (`_shared.py:3122-3159`) which falls back to `find_best_hybrid(cache=True)` —
  the standard experiment cache-miss path.
- NOTE: `src/kirby/precompute_hybrids.py` does NOT exist; precompute lives in
  `_shared.precompute_hybrid_searches`.

## Commands run

REG (ginkgo_heparin, n=94, task=regression, ranking metric=spearman):
```
rm -rf /tmp/kirby_V_precompute/result_db
OMP_NUM_THREADS=1 ... KIRBY_CACHE_DIR=/tmp/kirby_V_precompute PYTHONPATH=src:tests \
  perl -e 'alarm 900; exec @ARGV' \
  /Users/apunt/anaconda3/bin/python tests/experiments/_run_precompute_handoff.py reg
```

CLF (humanness_therapeutic, subsampled n=200, task=classification, metric=auc):
```
rm -rf /tmp/kirby_V_precompute/result_db
OMP_NUM_THREADS=1 ... KIRBY_CACHE_DIR=/tmp/kirby_V_precompute KIRBY_SMOKE_N=200 \
  PYTHONPATH=src:tests perl -e 'alarm 900; exec @ARGV' \
  /Users/apunt/anaconda3/bin/python tests/experiments/_run_precompute_handoff.py clf
```
Both exit 0. Reps = aac/blosum/onehot (local). 5 GroupKFold folds (from groups).

## SMOKE RESULTS

| dataset | task | step | result |
|---|---|---|---|
| ginkgo_heparin | regression | 2 cold precompute | PASS — wrote `result_db/ginkgo_heparin.db`, 1 row |
| ginkgo_heparin | regression | 3 warm precompute | PASS — 1 reused, 0 recomputed, no meta drift |
| ginkgo_heparin | regression | 4 downstream get_precomputed_search | PASS — hit precomputed row, rows unchanged |
| ginkgo_heparin | regression | 5 add ridge model | PASS — 1 reused, only ridge computed, +1 row |
| ginkgo_heparin | regression | §D large-n guard reached | PASS — no crash on small n |
| humanness_therapeutic | classification | 2 cold precompute | PASS — wrote `result_db/humanness_therapeutic.db`, 1 row (metric=auc) |
| humanness_therapeutic | classification | 3 warm precompute | PASS — 1 reused, 0 recomputed, no meta drift |
| humanness_therapeutic | classification | 4 downstream get_precomputed_search | PASS — hit precomputed row, rows unchanged |
| humanness_therapeutic | classification | 5 add lightgbm model | PASS — 1 reused, only lightgbm computed, +1 row |
| humanness_therapeutic | classification | §D large-n guard reached | PASS — no crash on small n |

## CACHE EVIDENCE (pasted stdout)

### REG — cold → warm → downstream → changed-model
```
----- STEP 2: COLD precompute -----
  [1/1] rf/top_2/treeshap/b=25/ev=rf: score=0.3192+/-0.326, dim=25
 done 25.4s (1 models, top=rf score=0.3192)
[COLD] DB now exists=True rows=1

----- STEP 3: WARM precompute (identical args) -----
  ResultCache: 1 stored configs reused; computing only the remainder of 1
 done 18.2s (1 models, top=rf score=0.3192)
[WARM] rows=1 (cold was 1)

----- STEP 4: downstream get_precomputed_search (precomputed=None) -----
  ResultCache: 1 stored configs reused; computing only the remainder of 1
[STEP4] downstream search returned best_by_model=1 models
[STEP4] rows after downstream call=1 (was 1)

----- STEP 5: add a model (ridge) — old rows reused, only new computed -----
  model list now: ['rf', 'ridge']
  ResultCache: 1 stored configs reused; computing only the remainder of 2
  [2/2] ridge/top_2/treeshap/b=25/ev=rf: score=0.3791+/-0.286, dim=25
[STEP5] rows after adding model=2 (was 1)
[STEP5] +1 new rows for the added model; old 1 reused. PASS
  cold_rows=1 warm_rows=1 after_downstream=1 after_add_model=2
```
No `meta drift` line appeared in any run. The cold run printed NO ResultCache
reuse line (`search: 0 hit / 0 miss / 0 stale`); the reuse line appears only from
the warm run onward.

### CLF — cold → warm → downstream → changed-model
```
[COLD]  done 70.3s (1 models, top=rf score=0.7796)   # cache stats: search 0/0/0 — cold
[COLD] DB now exists=True rows=1

----- STEP 3: WARM precompute (identical args) -----
  ResultCache: 1 stored configs reused; computing only the remainder of 1
[WARM] rows=1 (cold was 1)

----- STEP 4: downstream get_precomputed_search (precomputed=None) -----
  ResultCache: 1 stored configs reused; computing only the remainder of 1
[STEP4] rows after downstream call=1 (was 1)

----- STEP 5: add a model (lightgbm) — old rows reused, only new computed -----
  model list now: ['rf', 'lightgbm']
  ResultCache: 1 stored configs reused; computing only the remainder of 2
  [2/2] lightgbm/top_2/treeshap/b=25/ev=rf: score=0.7383+/-0.027, dim=25
[STEP5] rows after adding model=2 (was 1)
  cold_rows=1 warm_rows=1 after_downstream=1 after_add_model=2
```

### Persisted DB rows (after CLF run)
```
db: /tmp/kirby_V_precompute/result_db/humanness_therapeutic.db
 row: tier=general speed=full n_folds=5 seed=42 cv_splitter_id=''
      model=rf       internal_evaluator=rf budget=25 ranking_metric=auc score=0.7796 rep_pool=('aac','blosum','onehot')
 row: tier=general speed=full n_folds=5 seed=42 cv_splitter_id=''
      model=lightgbm internal_evaluator=rf budget=25 ranking_metric=auc score=0.7383 rep_pool=('aac','blosum','onehot')
```
The IDENTITY is shared across both rows; model is a per-row CONFIG column; rep_pool
is stored as self-describing DATA. Classification correctly resolved metric=auc;
regression resolved spearman — distinct, correct per task.

## §D large-n guard (reachability)
`_STRATEGY_MAX_SAMPLES[('svm','uncapped')] = 5000` (`find_best_hybrid.py:1375`) and
the capped-skip / `_check_max_samples` path are evaluated on every search. On
n=94 / n=200 the comparisons run as no-ops (n < 5000) and both runs completed
cleanly (exit 0), confirming the guard code path is reached without crashing on
small data. (No real large-n dataset was needed per the brief.)

## DECISIONS / SHARED-CODE FINDINGS
None. No shared-code or cache bug found. The new SQLite result cache exhibits the
exact behavior the redesign promised:
- cold computes + persists,
- identical re-run reuses 100% (0 recompute, no meta drift),
- a different model list reuses old rows and computes only the delta (the months-
  long-stall bug would have been caught here).

One observation (not a bug): `precompute_hybrid_searches` ignores `config['budgets']`
and always uses `_shared.FULL_BUDGETS` (documented at `_shared.py:2802-2810`), so a
local smoke run must shrink `FULL_BUDGETS` to stay tractable — done in the driver,
matching the brief's proven recipe.
