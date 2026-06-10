# §V LOCAL SMOKE GATE — COMPLEXITY Family Report (2026-06-10)

Complexity-lane agent for the §V local smoke fleet (the last gate before the SQLite cache
redesign is committed + resubmitted to ARC). Lane: `tests/experiments/complexity/`.

Environment for every run: single-thread prefix
(`OMP/OPENBLAS/MKL/VECLIB/NUMEXPR=1`, `KMP_DUPLICATE_LIB_OK=TRUE`), `PYTHONPATH=src`,
`/Users/apunt/anaconda3/bin/python` (3.11), isolated cache `KIRBY_CACHE_DIR=/tmp/kirby_V_complexity`
(cache-evidence run uses a SEPARATE empty `/tmp/kirby_V_cev`), perl-alarm wall wrapper.
Driver: `tests/experiments/_run_complexity_locally.py` (in-process trims FULL_BUDGETS=[25],
1 allocation, treeshap importance, rf-only model — smoke only, server imports fresh).
Small reg dataset: **buchwald_hartwig** (reaction, n=3955→200 via `KIRBY_SMOKE_N`, regression,
reps `diff_maccs`+`onehot_components`). Small clf dataset: **humanness_therapeutic**
(antibody, n=564→200, classification, reps `aac`+`blosum`+`onehot`).

**Verdict: PASS-with-caveats.** All exercised experiments run end-to-end or SKIP cleanly on
both reg and clf; c06 correctly skips clf (regression-only); the NEW per-dataset SQLite
ResultCache is PROVEN reused with NO `meta drift`, including the changed-model-list case (the
core months-long-stall bug). Caveat: in the trimmed smoke driver (no `precompute_hybrid_searches`
pool built), the pool-fed sub-paths of c09/c15/c17/c24 record `nan`+fallback instead of a
hybrid — graceful (no crash), but it means the hybrid arm of those experiments is NOT scored
locally. This is a driver-coverage limitation, not a cache-redesign bug.

---

## 1. SMOKE RESULTS

### Regression — buchwald_hartwig (n→200, 2 folds)

| Experiment | Result | Notes |
|---|---|---|
| `c01_rep_characterization` | **PASS** | end-to-end; per-rep intrinsic-dim/MI/effective-rank + rf baselines (diff_maccs rf=26.19, onehot_components rf=23.99). `_meta` metric block present. |
| `c09_formal_analysis` | **PASS** | returns `information`/`effective_rank`/`bias_variance`. Pool sub-paths (9b/9d) emit `[pool] no usable hybrid` → record nan + fallback (no precompute pool in smoke). No crash. |
| `c15_ml_theory_depth` | **PASS** | returns `distance_concentration`/`kernel_analysis`/`double_descent`/`approximation_theory`. c15c per-budget fast-search ran (`tier='c15c_fast'`); blueprints nan at b160/180/200 → "recording nan (skipped)" (graceful, smoke pool empty). |
| `c17_information_theoretic_decomposition` | **PASS** | mRMR/RFE/PCA/random MI-density all computed (mRMR best, density 0.0786); hybrid arm nan (no pool). Returns `per_budget`/`summary`. |
| `c24_blueprint_transfer` | **SKIP (clean, env-driven)** | "no valid config found" per fold → "need at least 2 successful folds" → `{'skipped': True, 'reason': ..., 'n_folds_succeeded': ..., 'fold_diagnostics': ...}`. Graceful skip because the reg run's smoke pool was empty (NOT a c24 bug — the SAME experiment RAN fully on the clf dataset, see clf table). |
| `c13_forward_compatibility` | **PASS** | full A/B/C/D/E forward-compat matrix ran (detected diff_maccs as "new rep", `[SMOKE] FULL_BUDGETS shrunk to [25]` knob fired). D-C/E-B deltas nan (no pool) but the experiment structure + summary complete. Returns full result dict. |

### Classification — humanness_therapeutic (n→200, 2 folds, reps aac/blosum/onehot)

| Experiment | Result | Notes |
|---|---|---|
| `c01_rep_characterization` | **PASS** | end-to-end; per-rep intrinsic-dim/MI/effective-rank + AUROC baselines for aac/blosum/onehot. Returns `_meta`+per-rep keys. (Slow tail — 5600D blosum/onehot MLE+MI — but finishes.) |
| `c06_within_family_gradient` | **SKIP (clean)** | `Skipping: c06 is regression-only (task_type=classification)` → `{'skipped': True, 'reason': 'regression_only', 'task_type': 'classification'}` (c06:51-55). Verified twice (full batch + an isolated fast `aac`-only run, `/tmp/kirby_V_c06_clf.log`). |
| `c09_formal_analysis` | **PASS** | runs on clf; returns `information`/`effective_rank`/`bias_variance`. |
| `c15_ml_theory_depth` | **SKIP (clean)** | `Skipping: c15 is regression-only` → `{'skipped': True, 'reason': 'regression_only', ...}`. (Double-descent/kernel theory is reg-only.) |
| `c17_information_theoretic_decomposition` | **SKIP (clean)** | `Skipping: c17 is regression-only` → `{'skipped': True, 'reason': 'regression_only', ...}`. |
| `c24_blueprint_transfer` | **PASS (clf, full hybrid path)** | RAN on clf: discovered blueprints both folds (rf/b=25/top_2/treeshap), cross-fold transfer (own=0.8018, delta=+0.0000), Nogueira stability. Returns `transfer_matrix`/`per_target`/`aggregate`/`stability`/`fold_blueprints`. **This run EXERCISED + STORED to the SQLite cache** (`humanness_therapeutic.db`: 2 rows, tier general/top, model rf — clf cache evidence). |
| `c13_forward_compatibility` | **EARLY-RETURN (clean guard, not a failure)** | Returns `{'error': 'no_new_reps', 'loaded_reps': [aac, blosum, onehot]}` (c13:406) because all local reps are LEGACY_REPS so there is no "new rep" to test. Intentional guard (`WARNING: No new reps detected`), NOT a crash — on the reg run diff_maccs WAS new so c13 ran the full A/B/C/D/E matrix. The `'error'` key here is benign; a wave-audit should treat `no_new_reps` as a SKIP, not a fault. |

Clf coverage: 1 native-clf PASS that exercised the full hybrid+SQLite path (c24), 1 reg-only
SKIP that the c06 guard handles, 3 more reg-only SKIPs (c15/c17 + c06), c01/c09 clf-PASS. No
unexpected `{'error'}` (the only one, c13 `no_new_reps`, is an intentional local-reps guard) and
no traceback/segfault in the clf log; exit 0.

---

## 2. CACHE EVIDENCE (the headline §V part-2 test) — PASS

Driver `tests/experiments/complexity/_cache_evidence_local.py` (in-lane, new this session)
calls `find_best_hybrid(cache=True, dataset_name='cev_buchwald', tier='cache_evidence',
speed='full', verbose=True)` THREE times against ONE fresh empty `KIRBY_CACHE_DIR=/tmp/kirby_V_cev`.
DB: `/tmp/kirby_V_cev/result_db/cev_buchwald.db`. 3 reps + budgets `[10,25]` + `test_rep_subsets=True`
so the grid is non-empty (8 rf configs). Actual stdout (`/tmp/kirby_V_cev_run.log`):

**RUN 1 — COLD (models=[rf]):** computes 8, stores 8 rows.
```
Searching 8 unique configurations (full mode)...
  Evaluating 8 configs (all 2 folds)...
  [1/8] rf/top_2/treeshap/b=10/ev=rf: score=26.8857+/-0.566, dim=10
  ...
  rf            26.7218  24.5818
```

**RUN 2 — WARM (models=[rf], expect full reuse):**
```
Searching 8 unique configurations (full mode)...
  ResultCache: 8 stored configs reused; computing only the remainder of 8
  Evaluating 8 configs (all 2 folds)...
  rf            26.7218  24.5818
```
→ all 8 rf rows reused; best `rf b=10=26.7218 / b=25=24.5818` **byte-identical to RUN 1**
(reused, not recomputed). **NO `meta drift` line anywhere** (grep confirms the only "meta drift"
string is my own RUN 2 header label "(no meta drift)").

**RUN 3 — WARM + CHANGED MODEL LIST (models=[rf, rf_deep]) — the core bug the gate exists for:**
```
Searching 16 unique configurations (full mode)...
  ResultCache: 8 stored configs reused; computing only the remainder of 16
  Evaluating 16 configs (all 2 folds)...
  [3/16] rf_deep/top_2/treeshap/b=10/ev=rf: score=26.7203+/-0.400, dim=10
  [4/16] rf_deep/top_2/treeshap/b=25/ev=rf: score=25.0282+/-1.772, dim=25
  ... (only rows 3,4,7,8,11,12,15,16 — the rf_deep configs — computed)
  rf            26.7218  24.5818
  rf_deep       26.7164  24.4772
```
→ The 8 **rf** rows were reused (rows 1,2,5,6,9,10,13,14 SKIPPED — only the 8 NEW **rf_deep**
configs computed), and `rf` scores are byte-identical to RUN 1. This is exactly the designed
behaviour: **changing the model list reuses old rows and computes ONLY the new model.**

**DB inspection** (`sqlite3 cev_buchwald.db`): final 16 rows — `rf|8`, `rf_deep|8`. Identity
columns populated and shared across both models: `tier=cache_evidence, speed=full, n_folds=2,
seed=42, labels_hash=11118922619129298662, groups_hash=11073919630898510758`. The changed
model is a per-row CONFIG column (`model`), not part of the identity — confirming the redesign's
"swap a model = SELECT, not rebuild" contract (`src/kirby/result_cache.py` IDENTITY_COLS vs
CONFIG_COLS).

Full log: `/tmp/kirby_V_cev_run.log`.

**Secondary (clf) cache evidence:** the clf-batch `c24_blueprint_transfer` run on
humanness_therapeutic drove the real experiment hybrid-pool path through the SQLite store and
WROTE rows: `/tmp/kirby_V_complexity/result_db/humanness_therapeutic.db` ends with 2 rows
(`tier` ∈ {general, top}, `model=rf`, integer labels/groups hashes). This confirms the new cache
is exercised + stored on a CLASSIFICATION dataset via an unmodified experiment, not just the
verbose harness.

---

## 3. IN-LANE FIXES / EDITS

1. **`tests/experiments/complexity/c06_within_family_gradient.py` (lines 160, 171, 199)** —
   prior-agent edit, VERIFIED correct. The result cache keys on the `model` column; variant
   names like `C=0.01` repeat across families, so two families' rows would collide on one DB
   key and overwrite each other. Fix prefixes the downstream-model name with the family:
   `model_key = f"{fam_name}:{var_name}"`, used consistently for both `models={model_key: ...}`
   and the `best_by_model_budget.get(model_key, ...)` lookup. The output dict stays keyed by the
   display `var_name`, so no result-shape change. Lane-clean; server runs unaffected.

2. **`tests/experiments/complexity/_cache_evidence_local.py` (NEW, in-lane)** — the §V part-2
   cache-evidence driver (cold/warm/changed-model, verbose=True). Smoke-only; not imported by any
   server path. Adds a deterministic synthetic 3rd rep + 2 budgets + `test_rep_subsets=True`
   purely to keep the search grid non-empty (with only 2 reps `top_2` is skipped and the grid
   collapses to "0 unique configurations", which stores nothing and cannot demonstrate reuse).

No edits to `_shared.py`, `src/**`, `find_best_hybrid.py`, `result_cache.py`, or any other
family's directory. Nothing committed.

---

## 4. SHARED-CODE / CACHE FINDINGS (record-only; outside lane, NOT fixed)

1. **Empty SQLite DBs are created for all-NaN searches (expected, but worth noting).**
   `find_best_hybrid.py:3685-3692` stores a row ONLY when `score_mean` is finite (a NaN config
   re-attempts next run by design). In the trimmed smoke driver several experiment-driven
   searches (c15c `tier=c15c_fast` at b160/180/200, c07 partition DBs from the prior agent)
   produced all-NaN scores, so their `result_db/*.db` files exist with **0 rows**. This is the
   documented "missing = recompute" behaviour, NOT corruption — but a wave-audit that counts
   "DB file exists" as DONE would be fooled (exactly the class of mirage §V exists to catch).
   *Suggestion:* the precompute-status tool should count DB **rows**, not DB **files**
   (it already does per the ledger; flagging so the server audit stays row-based).

2. **Pool-fed experiment arms degrade to nan when no precompute pool exists (reg run only).**
   On the REGRESSION run, c09 (9b/9d), c15 (c15/c15c), c17 (hybrid density), c24 (blueprint
   discovery), c13 (D/E conditions) call `get_best_search_for_downstream` /
   `get_fitted_pool_hybrid` (`tests/experiments/_shared.py:3233/3459`), which want a
   `precompute_hybrid_searches`-built `tier='general'/'top'` pool. The local smoke driver never
   runs that precompute, so those lookups returned `None` → the experiments correctly print
   `[pool] WARNING: no usable hybrid ...` and fell back to nan. **Not a bug** (the WARNING is the
   intended loud-fail added 06-05), no crash. NOTE the asymmetry: c24 on the CLF run DID populate
   the pool on its own (it calls `find_best_hybrid` to discover blueprints, which builds + stores
   the general/top rows) and ran fully (own=0.8018) — so the cache+pool DOES work end-to-end when
   the experiment builds its own search. The pure-LOOKUP arms (c09 9b, c15c) that expect a
   *separately precomputed* pool are the ones that nan locally. To smoke those arms end-to-end the
   driver would need a `precompute_hybrid_searches` pass first (out of lane / scope here).

3. **c09 9b emits a non-fatal `ValueError: No importance scores for rep 'diff_maccs'`** per fold
   (`precomputed_importance and importance_cache are both missing this rep`). Caught internally
   ("fold hybrid build failed"), recorded as nan, experiment returns OK. Same root cause as #2
   (no precompute pool → the importance_cache the pool would have populated is absent). Worth a
   glance to confirm the server path (with precompute) populates it; harmless in smoke.

No cache-redesign bug found. The SQLite store behaved exactly as designed: identity =
data+folds+tier/speed, sweep (model/budget/allocation/importance) = per-row config,
changed-model-list = SELECT-not-rebuild, NaN = not stored (re-attempt), no `meta drift`.

---

## 5. FAMILY VERDICT: **PASS-with-caveats**

- Regression (buchwald_hartwig): c01/c09/c15/c17/c13 PASS, c24 clean SKIP (insufficient folds in
  the empty-pool smoke env). No crash, no unexpected `{'error'}`.
- Classification (humanness_therapeutic): c01/c09 PASS; c24 PASS **and exercised the full hybrid
  + SQLite-cache path** (stored 2 rows); c06/c15/c17 clean regression-only SKIP; c13
  `no_new_reps` intentional guard (local reps all legacy — not a fault). No traceback/segfault.
- c06 regression-only SKIP on clf: correct (`task_type != 'regression'` guard, c06:51-55).
- Cache: REUSED with NO `meta drift`; changed-model-list reuse PROVEN (8 rf rows reused, only 8
  rf_deep computed) on reg; clf store confirmed via c24 → humanness_therapeutic.db. DB row counts
  confirmed via sqlite3.
- Caveat (not blocking): the pure-LOOKUP hybrid arms of c09(9b)/c15(c15c) nan on the reg run
  because the smoke driver builds no separate precompute pool — graceful degradation, recorded
  in §4. (c24 shows the pool+cache itself works when an experiment builds its own search.)
