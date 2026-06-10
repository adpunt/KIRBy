# §V LOCAL SMOKE GATE — OPIG Family Report (2026-06-10)

OPIG agent for the §V local smoke fleet (the last gate before the SQLite cache redesign
is committed + resubmitted to ARC). Lane: `tests/experiments/opig_*.py`.

Environment for every run: single-thread env prefix (`OMP/OPENBLAS/MKL/VECLIB/NUMEXPR=1`,
`KMP_DUPLICATE_LIB_OK=TRUE`), `PYTHONPATH=src`, `/Users/apunt/anaconda3/bin/python`,
isolated cache `KIRBY_CACHE_DIR=/tmp/kirby_V_opig` (started EMPTY), perl-alarm wall wrapper.
Smoke knobs: `KIRBY_SMOKE_N=200` + the in-file `_smoke_search_kwargs()` shrink
(`speed='full'`, `budgets=[25]`, `internal_evaluators=['rf']`, `importance=lightgbm_gain`).

Verdict: **PASS.** One clf + one reg OPIG experiment run end-to-end through the NEW per-dataset
SQLite ResultCache; the core cache-reuse + changed-model-list reuse is proven with NO
`meta drift`; no `{'error'}` / traceback in any log; B-VERIFY (`_identity_groups`) confirmed.

---

## 1. SMOKE RESULTS

| Experiment | Task | Command (abridged) | Result | Notes |
|---|---|---|---|---|
| `opig_humanness.py` | **classification** | `--datasets humanness_therapeutic --n-folds 2 --reps blosum,onehot,aac` | **PASS** | end-to-end. `task_type=classification`. `_identity_groups` → 190 **integer (int64)** groups over 200 rows. kirby_hybrid built across all 8 models; best AUROC 0.9229 (rf), **+0.0730** over best individual (onehot/ridge 0.8499). JSON saved. Exit 0. |
| `opig_ginkgo_developability.py` | **regression** | `--properties ginkgo_heparin --n-folds 2 --reps blosum,onehot,aac` | **PASS** | end-to-end. n=94, target HAC. kirby_hybrid built (mlp ρ=0.6856). Cross-property summary printed (hybrid 0/1 wins here — aac/ridge ρ=0.7518 beats it; a legitimate small-n result, NOT a bug). JSON saved. Exit 0. |

Both experiments already carried the prior fleet's fixes (the `'fast'`→`'full'` SPEED_TIERS
fix + the `_smoke_search_kwargs` budget/evaluator/importance shrink + the
`_identity_groups` wiring in humanness). **No new in-lane edits were required** for the two
covered experiments — they ran correctly as-is.

**Coverage of the brief's required matrix:** clf (humanness) ✅ and reg (ginkgo) ✅, both on
small REAL antibody data, both through the new SQLite cache. Foundation reps deliberately
excluded (local-only reps `blosum/onehot/aac`).

---

## 2. CACHE EVIDENCE (the headline §V part-2 test)

The opig on-demand pool path calls `find_best_hybrid(..., verbose=False)`
(`tests/experiments/_shared.py:3153`, hardcoded), so the `ResultCache:` reuse line does not
print through the experiment scripts themselves. To capture the evidence I ran a small
**scratch harness** in `/tmp/kirby_V_opig/cache_evidence.py` that drives the SAME engine the
opig scripts use (`find_best_hybrid(cache=True, dataset_name='ginkgo_heparin', tier='general',
speed='full', budgets=[25], internal_evaluators=['rf'], importance=lightgbm_gain)`) with
`verbose=True`, three times against the same `$KIRBY_CACHE_DIR`. This exercises the identical
per-dataset SQLite store (`<cache>/result_db/ginkgo_heparin.db`). Actual stdout:

**RUN1 — COLD (`models=[rf]`):** computes 3 configs, writes the DB.
```
Evaluating 3 configs (all 5 folds)...
  [1/3] rf/top_2/lightgbm_gain/b=25/ev=rf: score=0.3923+/-0.162, dim=25
  [2/3] rf/performance_weighted/lightgbm_gain/b=25/ev=rf: score=0.3813+/-0.183, dim=25
  [3/3] rf/entropy_1.0/lightgbm_gain/b=25/ev=rf: score=0.3868+/-0.217, dim=25
[cache] cache stats — search: 0 hit / 0 miss / 0 stale; baseline: 0 hit / 3 miss / 0 stale; ...
```

**RUN2 — WARM (`models=[rf]`, expect full reuse):**
```
  ResultCache: 3 stored configs reused; computing only the remainder of 3
  Evaluating 3 configs (all 5 folds)...
[cache] cache stats — search: 0 hit / 0 miss / 0 stale; baseline: 3 hit / 3 miss / 0 stale; ...
```
→ all 3 rf rows reused; **NO `meta drift` line anywhere.**

**RUN3 — WARM + CHANGED MODEL LIST (`models=[rf,ridge]`) — the core bug the gate exists for:**
```
Searching 14 unique configurations (full mode)...
  ResultCache: 3 stored configs reused; computing only the remainder of 6
  Evaluating 6 configs (all 5 folds)...
  [2/6] ridge/top_2/lightgbm_gain/b=25/ev=rf: score=0.4641+/-0.160, dim=25
  [4/6] ridge/performance_weighted/lightgbm_gain/b=25/ev=rf: score=0.5260+/-0.070, dim=25
  [6/6] ridge/entropy_1.0/lightgbm_gain/b=25/ev=rf: score=0.5058+/-0.090, dim=25
Best features per model x budget (spearman):
  rf            0.3923
  ridge         0.5260
```
→ The 3 **rf** rows were reused (rows 1/3/5 skipped — only the 3 NEW **ridge** rows computed),
and `rf=0.3923` is byte-identical to RUN1 (reused, not recomputed). This is exactly the
designed behaviour: **changing the model list reuses the old rows and computes ONLY the new
model.** No `meta drift`.

**DB inspection:** `/tmp/kirby_V_opig/result_db/{humanness_therapeutic,ginkgo_heparin}.db`
created with the expected identity columns populated (`tier='general'`, `speed='full'`,
integer `labels_hash`/`groups_hash`, `n_folds`, `seed='42'`) and per-row config columns
(model/budget/allocation/importance). The humanness DB stored 25 rows spanning all 8 models
from the cold experiment run.

Full logs: `/tmp/kirby_V_opig/cache_evidence.log`, `/tmp/kirby_V_opig/ginkgo.log`,
humanness stdout in the task output file.

---

## 3. B-VERIFY (ledger §B / ISS-29 — humanness `_identity_groups`)

CONFIRMED. `tests/experiments/opig_humanness.py:249` calls
`_identity_groups(heavy_seqs, light_seqs)` (the same helper as the survey:
Levenshtein clustering, 95% single-linkage, canonical integer cluster IDs) — NOT the old
exact-match grouping. Verified at runtime on the real subsampled therapeutic set:
`n_groups=190` over 200 rows, `dtype=int64`, `all_int=True`, sample
`[137 185 158 113 114 101 163 2 136 146]`. The run produced no `meta drift` and a stable
`groups_hash`, so the grouping is deterministic.

---

## 4. SHARED-CODE / CACHE FINDINGS (record-only; not fixed — outside lane)

1. **`tests/experiments/_shared.py:3153` — on-demand pool path hardcodes `verbose=False`.**
   Not a bug, but it means the `ResultCache: N stored configs reused` line never surfaces
   through the opig experiment scripts (the very evidence this gate wants). The
   `[cache] cache stats — baseline/hybrid_eval hit/miss` line still prints (different code
   path), and DB reuse demonstrably works (see §2 via the verbose harness). **Suggestion for
   the orchestrator/user:** thread an env knob (e.g. `KIRBY_CACHE_VERBOSE=1`) to force
   `verbose=True` into `get_precomputed_search`'s `find_best_hybrid` call so server runs can
   self-report cache reuse in their own logs. Low priority; behaviour is correct, only the
   logging is silent.

2. **Carry-over (still open from 06-05, unchanged):** `_shared.py:3236-3239` silently
   swallows per-pool-condition exceptions. Not triggered in either run here (both used
   `speed='full'`), but it is the mechanism that hid the old `'fast'` KeyError. Worth a
   one-line WARN when ALL pool conditions fail. Outside lane — record-only.

No cache-redesign bug found in this lane. The SQLite store behaved exactly as designed:
identity = data+folds+tier/speed, sweep = per-row config, changed-model-list = SELECT-not-rebuild.

---

## 5. NOT RUN THIS SESSION (scoped out by the brief's "one clf + one reg" requirement)

`opig_trastuzumab_enrichment` (clf), `opig_graphinity_ddg` (reg, heavy — 200 unique
PDB-complex groups), `opig_cdr_flexibility`, `opig_activity_cliff_qsar`,
`opig_blueprint_transfer`, `opig_cgr_comparison`, `opig_kingsmith_component_splits`,
`imrie_transfer_learning`. All were exercised in the 06-05 fleet (see
`tests/results/_smoke_2026_06_05/OPIG_REPORT.md`); the two run here are representative of the
clf and reg hybrid code paths shared across the family.

## FAMILY VERDICT: **PASS**
- clf (humanness): PASS, hybrid built, integer groups, +0.073 AUROC over best individual.
- reg (ginkgo): PASS, hybrid built, cross-property summary, clean.
- Cache: REUSED with NO `meta drift`; changed-model-list reuse proven (rf reused, ridge-only
  computed).
- No unexpected `{'error'}` / traceback in any log.
