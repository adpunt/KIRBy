# §V CACHE + DETERMINISM REPORT (2026-06-10)

Agent: CACHE+DETERMINISM (headline §V gate). Local, real code, real small data.
Env: `/Users/apunt/anaconda3/bin/python` 3.11, single-thread OMP/BLAS, `KIRBY_CACHE_DIR=/tmp/kirby_V_cache*` (started empty).
Throwaway test scripts: `/tmp/det_test.py`, `/tmp/cache_test.py`, `/tmp/cache_clf_test.py`, `/tmp/aug_and_nocache_test.py` (NOT committed).

---

## PART 1 — DETERMINISM (PASS, all 3 functions)

Method per function: run twice in-process + on a row-shuffled copy (un-permuted back to
original order); assert identical partition (who-groups-with-whom), identical `groups_hash`
(`_hash_bytes(_canonical_bytes(groups))`), and integer groups. Also re-ran the whole script in a
SECOND fresh process with a different `PYTHONHASHSEED` (random vs 12345) and confirmed the printed
hashes are byte-identical across processes.

| Function | file:line | groups_hash (run1=run2=shuffled) | int? | partition stable? | Verdict |
|---|---|---|---|---|---|
| `sequence_identity_clusters` (antibody Levenshtein) | `tests/experiments/_shared.py:764` | `3e93d53f2c002562` (all 3) | yes | yes (12 clusters, 12 near-dup families merged) | **PASS** |
| `assign_scaffold_groups` (molecular scaffold) | `tests/find_best_hybrid_molecular.py:78` | `bc0df00598cd8ac0` (all 3) | yes | yes (5 scaffolds) | **PASS** |
| `_canonical_bytes` object/string group hash (§A1) | `src/kirby/find_best_hybrid.py:264` | `bd71f2d26df8c724` (run1=run2=perm) | yes | yes | **PASS** |

Key sub-assertions:
- `import Levenshtein` works (v0.27.3) — **no difflib fallback** (the `from Levenshtein import distance`
  hard import at `_shared.py:828`). The §B divergence (server used difflib → different folds) cannot
  recur once Levenshtein is installed on ARC.
- Cross-process stability: the 3 hashes above were identical under `PYTHONHASHSEED=random` AND
  `PYTHONHASHSEED=12345` → object-pointer non-determinism is gone (§A1 factorize-to-int-codes).
- `_canonical_bytes` on an OBJECT array equals the hash of the same partition's canonical
  sorted-unique int codes (verified directly), AND a NUMERIC array passes through byte-unchanged vs
  the old `np.ascontiguousarray(x).tobytes()` path → existing integer-group caches are NOT forced to
  rebuild (the documented invariant at `find_best_hybrid.py:274-276`).

---

## PART 2 — CACHE REUSE (PASS — the months-long-stall bug is provably dead)

Real `find_best_hybrid.find_best_hybrid(...)` on `ginkgo_heparin` (reg, n=94), local reps
`aac/blosum/onehot`, tiny config: `budgets=[25]`, `n_splits=2`, `internal_evaluators=['rf']`,
`allocation_methods=['top_2','performance_weighted']`, `importance_methods=['lightgbm_gain']`,
`adaptive_pruning=False`, `tier='general'`, `cache=True`.
(NOTE: the brief's `internal_evaluators=['rf']` is correct, but `allocation_methods=['greedy']` does
NOT work in the standalone `find_best_hybrid` call — `greedy` requires a precomputed-importance cache
that only `precompute_hybrid_searches` builds; it raises `No importance scores for rep 'aac'`. Used
the real `full`-tier allocation methods instead. This is a TEST-HARNESS note, not a code bug — the
engine computes importance for the tier allocations fine.)

### (a) COLD run (empty cache) — computed 4, wrote DB
```
Searching 4 unique configurations (full mode)...
  Evaluating 4 configs (all 2 folds)...
  [1/4] rf/top_2/lightgbm_gain/b=25/ev=rf: score=-0.0837+/-0.113, dim=25
  [2/4] ridge/top_2/lightgbm_gain/b=25/ev=rf: score=-0.2656+/-0.251, dim=25
  [3/4] rf/performance_weighted/lightgbm_gain/b=25/ev=rf: score=-0.0837+/-0.113, dim=25
  [4/4] ridge/performance_weighted/lightgbm_gain/b=25/ev=rf: score=-0.2656+/-0.251, dim=25
```
DB written at `/tmp/kirby_V_cache/result_db/ginkgo_heparin.db` (4 rows).

### (b) WARM run (same args) — reused 4, computed 0, NO meta drift
```
##### PHASE=warm models=['rf', 'ridge'] #####
  ResultCache: 4 stored configs reused; computing only the remainder of 4
  Evaluating 4 configs (all 2 folds)...
##### SCORES #####
  rf: score_mean=-0.08369927541449029
  ridge: score_mean=-0.2655750956532028
```
- `ResultCache: 4 stored configs reused` printed.
- NO `[N/4]` config-eval lines → 0 new computed.
- NO `meta drift` line anywhere.
- Scores byte-identical to cold (rf=-0.08369927541449029, ridge=-0.2655750956532028).

### (c) CHANGED-MODEL-LIST run (add `lightgbm`) — reused 4, computed ONLY the 2 new rows
```
##### PHASE=changed models=['rf', 'ridge', 'lightgbm'] #####
Searching 6 unique configurations (full mode)...
  ResultCache: 4 stored configs reused; computing only the remainder of 6
  Evaluating 6 configs (all 2 folds)...
  [3/6] lightgbm/top_2/lightgbm_gain/b=25/ev=rf: score=-0.0856+/-0.054, dim=25
  [6/6] lightgbm/performance_weighted/lightgbm_gain/b=25/ev=rf: score=-0.0856+/-0.054, dim=25
```
- Only the 2 `lightgbm` rows ([3/6], [6/6]) computed; the 4 rf/ridge rows reused, NOT recomputed.
- rf/ridge scores unchanged; no full rebuild; no `meta drift`.
- **This is the exact behaviour whose ABSENCE caused the months-long stall. PROVEN DEAD.**

### Classification sanity (`humanness_therapeutic`, clf, KIRBY_SMOKE_N=150, f1_weighted)
```
cold:  [1/2] rf/...: score=0.6653 ; [2/2] ridge/...: score=0.7150
warm:  ResultCache: 2 stored configs reused; computing only the remainder of 2  (0 config-eval lines, no meta drift)
```
Same reuse behaviour on a clf task with `groups` from identity clustering.

### (d) DB inspection (`sqlite3`)
- Row count grows 4 → 6 across the changed-model run (incremental fill, not rebuild).
- **Distinct CONFIG_COLS** per row (model × allocation_method), all share `budget=25`,
  `importance_method=lightgbm_gain`, `internal_evaluator=rf`, `aug_strategy=none`.
- **IDENTITY_COLS constant**: `SELECT COUNT(DISTINCT tier||speed||labels_hash||groups_hash||...)` = **1**.
- **`rep_pool` recorded**: `('aac', 'blosum', 'onehot')` on every row (self-describing, not a key).
- **`metrics_json` populated (D6)**: full metric set present, e.g.
  `{"mae": {"mean": 0.939..., "std": 0.069...}, "mse": {...}, ...}`.
- `score_mean`, `ranking_metric='r2'`, `had_imputation=0`, `blueprint` (BLOB) all non-null on every row.
- **aug coexistence**: `ginkgo_heparin` has no augmentations (uncapped→none normalisation, so a live
  none-vs-uncapped pair can't be exercised on it). Verified instead via a direct `ResultCache` unit
  test: two rows differing ONLY in `aug_strategy` ('none' vs 'uncapped') coexist as 2 separate rows
  after fetch — `aug_strategy` is a genuine CONFIG_COLS key column (`result_cache.py:56`, populated
  from `result_db_config_const` at `find_best_hybrid.py:4487`).

### (e) cache=False regression guard
`find_best_hybrid(..., cache=False)` runs to completion, returns a `SearchResult` with
`best_by_model['rf']` (score_mean=-0.08369927541449035), **no NameError**.

---

## DECISIONS / SHARED-CODE FINDINGS

**No shared-code/cache bug found. No STOP-the-commit finding.** The redesigned SQLite cache behaves
exactly as the §X design specifies: data+fold+task is the spine (IDENTITY_COLS), models/rankers/
budgets/allocations/aug are per-row contents (CONFIG_COLS), and changing the model/ranker/budget LIST
is a SELECT + incremental fill, never an all-or-nothing rebuild. Determinism prerequisites (§A1 / §B)
hold across fresh processes and row permutation.

Minor TEST-HARNESS note (not a code issue): `greedy` allocation can't be driven through the bare
`find_best_hybrid()` entrypoint (needs precompute's importance cache); use tier allocations
(`top_2`, `performance_weighted`, ...) for standalone calls. Worth knowing for any future smoke
driver, but it is the intended precompute/find_best_hybrid split.

---

## VERDICT

- **Determinism: PASS** — antibody Levenshtein, scaffold, and `_canonical_bytes` object/string path
  all reproduce identical partition + `groups_hash` across reruns, fresh processes, and row shuffles;
  groups are integers; Levenshtein is mandatory (no difflib).
- **Cache reuse: PASS** — cold writes the DB, warm reuses all rows with zero recompute and no
  `meta drift`, scores identical; a changed model list reuses old rows and computes only the new
  model's rows. DB rows are self-describing with full metrics (D6). `cache=False` is NameError-free.
- **Overall: PASS. The cache works as intended and the months-long-stall (all-or-nothing rebuild on a
  changed sweep list) is provably dead.**
