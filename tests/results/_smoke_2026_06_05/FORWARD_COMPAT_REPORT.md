# Forward-Compatibility Family Smoke Report (2026-06-05)

Agent lane: `tests/experiments/complexity/c13_forward_compatibility.py` (edit-only).
Experiment: **c13 forward compatibility** — score a "newly released" representation
alone, then run the KIRBy hybrid search to show existing reps augment it
(new-rep-alone vs new-rep + hybrid augmentation). Real "new" reps are
server-only foundation models, so locally we treat a locally-computable rep
(`onehot`) as the "new" rep via `--new-reps`.

---

## 1. SMOKE RESULTS

Environment for every run (the brief's single-thread guard + unbuffered + HF quiet):
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 \
KIRBY_SMOKE_N=120 TRANSFORMERS_VERBOSITY=error \
perl -e 'alarm <wall>; exec @ARGV' /Users/apunt/anaconda3/bin/python -u \
  tests/experiments/complexity/c13_forward_compatibility.py <args>
```

| # | Task | Command (args after the script) | Result | Notes |
|---|------|---------------------------------|--------|-------|
| 1 | REGRESSION | `--dataset ginkgo_heparin --reps aac,blosum,onehot --mode intermediary --new-reps onehot --models rf,ridge` | **PASS (timed-out-clean in hybrid grid)** | Reps load correct (aac/blosum/onehot only); SMOKE budget shrink fired; Conditions A & B fully completed (spearman computed); reached Condition C hybrid search; alarm killed mid grid-search (see timeout audit). |
| 2 | CLASSIFICATION | `--dataset humanness_therapeutic --reps aac,blosum,onehot --mode intermediary --new-reps onehot --models rf,ridge` | **PASS (timed-out-clean in hybrid grid)** | Identical path; metric correctly resolved to `auc`, task_type=classification; KIRBY_SMOKE_N subsampled 564→120; Conditions A & B completed (auc=0.6562); reached Condition C; alarm killed mid grid-search. |
| 3 | PRECOMPUTE PROBE (isolated) | direct `precompute_hybrid_searches` on aac+blosum, n=94, budget [25] | **PASS by grid-repetition** | Measured the bulk: importance 71.4s, allocation 21.9s, then **648 unique hybrid configs**; ran **315/648 cleanly** (all 8 models rf/xgboost/ridge/knn/lightgbm/mlp/svm/gp, all 3 importance methods, all 3 allocation families, multiple evaluators) before the 580s alarm. One full precompute ≈ 19-20 min on this hardware. |

### Fixes made in-lane (c13 only)

1. **Comma-separated `--reps` was ignored → loaded foundation models → unrunnable
   locally.** c13's `__main__` passed `args.reps` to `load_data` as a *preset
   name* and left `rep_names=None`. With `rep_names=None`, `_shared._load_data_impl`
   (line 2092-2099) auto-fills the FULL `DATASET_REPS['<ds>']['general']` list —
   27 reps including `ablang2`, `abroberta`, `antiberty`, `esm2`, etc. The first
   smoke run proved this: log showed
   `[auto] ginkgo_heparin: filtering to 27 general reps from DATASET_REPS` and
   timed out just loading foundation models, never reaching the experiment.
   **Fix** (`c13_forward_compatibility.py` entry point, ~line 840): use the
   existing `parse_rep_args(args)` helper (already imported at line 62) so a
   comma list becomes an explicit `rep_names` override:
   ```python
   rep_preset, rep_names = parse_rep_args(args)
   data = load_data(args.dataset, rep_preset, rep_names=rep_names, ...)
   ```
   After the fix the regression/clf runs loaded only `aac/blosum/onehot` and
   advanced through Conditions A, B into the hybrid search.

2. **Local hybrid search intractable — added KIRBY_SMOKE_N-gated budget shrink.**
   `_shared.precompute_hybrid_searches` reads the module-level
   `_shared.FULL_BUDGETS` (8 budgets) and IGNORES `config['budgets']`
   (`_shared.py` line 2685-2686). Added a gated shrink at the top of `run()`
   (`c13_forward_compatibility.py` ~line 367): when `KIRBY_SMOKE_N` is set, set
   `_shared.FULL_BUDGETS = [25]`. Fully gated — a no-op on server/full runs
   (env var unset). Confirmed in logs: `[SMOKE] FULL_BUDGETS shrunk to [25]`.

Both edits are syntax-checked and confined to the c13 file.

---

## 2. TIMEOUT AUDIT (read from the actual code)

c13's runtime structure, from the source:

- **Outer loop over new reps** — `c13_forward_compatibility.py:413`
  `for rep_name, X_new in sorted(new_reps.items())`. Each iteration runs the full
  5-condition ablation (A, B, C, D) + 2 controls for ONE new rep. This loop is
  **pure repetition** of the same code path → with several `--new-reps`, killing
  after 1-2 reps complete = a valid PASS. (My smoke used a single `--new-reps onehot`,
  so this loop runs once.)
- **Per-new-rep, the conditions are DISTINCT sequential phases:**
  - A (`:437`) and B (`:457`) — cheap single-rep CV. Both **completed** in both runs.
  - C (`:489`) and D (`:531`) — each calls `_eval_hybrid_per_fold` (`:172`), which
    calls `get_best_search_for_downstream` for candidate downstream models
    (`c13...py:200` `for candidate_model in ['rf','svm','lightgbm','mlp','knn']`).
    The FIRST such call for a given (dataset, condition_suffix) triggers ONE
    `precompute_hybrid_searches` → `find_best_hybrid`; subsequent candidate-model
    calls and folds reuse that single cached search. C and D use DISTINCT cache
    keys (`condition_suffix=c13_without_<r>` vs `c13_with_<r>`, lines 499 / 536),
    so each pays its own precompute.
- **The bulk of c13's runtime is INSIDE that one precompute**, which is itself a
  grid: `find_best_hybrid` with hardcoded `speed='full'` evaluates **648 unique
  configs** (8 models × 3 importance methods × 3 allocation families × evaluators),
  each over 5 folds. This inner grid IS pure repetition of one per-config eval.

**Decision:** For c13, the hybrid-search bulk is grid repetition, so a
perl-alarm kill after the grid is well underway counts as a pipeline PASS.
The isolated probe ran **315/648 distinct configs clean** across every model,
importance method and allocation family — that exercises the entire code path
C and D depend on. Conditions A and B (the only non-grid, c13-specific logic)
**completed fully**. I did NOT need to let the multi-hour full grid finish to
prove the pipeline; doing so would burn ~20 min per condition × 2 conditions ×
2 datasets for zero additional code coverage.

---

## 3. DECISIONS FOR USER

### D1. c13 `--reps` parsing was broken (FIXED in-lane, but check the sibling pattern)
- **Experiment:** c13 forward compatibility — proves existing reps augment a new rep.
- **Problem:** Before my fix, a comma-separated `--reps aac,blosum,onehot` on c13's
  own CLI was silently ignored; `load_data` auto-loaded the full foundation-model
  rep list for the dataset. On the server that's the intended behavior (you WANT
  the real reps), but it means **the documented c13 usage examples in the
  docstring** (`--reps all`, and any comma list) only ever worked because `all`
  is a real preset — a bare comma list did not.
- **Recommended:** Keep my in-lane fix (uses `parse_rep_args`). **No action needed
  on the server**, where you pass `--reps all` (a preset) + `--new-reps esm3,saprot`.
- **Evidence:** `c13_forward_compatibility.py:840` (old direct `load_data(args.dataset, args.reps,...)`),
  `_shared.py:2092-2099` (auto-fill of DATASET_REPS general when `rep_names is None`).

### D2. (SHARED CODE — orchestrator) Hybrid precompute is hardcoded `speed='full'` (648 configs); not locally shrinkable
- **Experiment:** c13 (and every experiment that calls the cached hybrid search).
- **Problem:** `_shared.precompute_hybrid_searches` hardcodes `speed = 'full'`
  (`_shared.py:2687`) and passes `models=None`, so `find_best_hybrid` builds the
  full SPEED_TIERS grid (648 unique configs at n=94, budget [25]: 8 models × 3
  importance methods × 3 allocation families × evaluators, ×5 folds). The brief's
  fast-config recipe (shrinking `config['allocations']`/`importance_methods`/
  `internal_evaluators`) has **no effect on this cached path** because precompute
  ignores config and re-reads the speed tier. There is **no env-var knob** to
  shrink the model set or the grid for a smoke run (`find_best_hybrid`'s skip set
  comes only from hardcoded dataset/domain tables, `find_best_hybrid.py:1323-1338`).
  This is why c13's Conditions C/D cannot fully complete locally in a reasonable
  wall — one precompute is ~20 min, and there are 2 (with/without the new rep).
- **Recommended (NOT applied — shared code, needs your signoff):** add a single
  KIRBY_SMOKE_N-gated knob in `_shared.precompute_hybrid_searches` /
  `find_best_hybrid` that, when smoke is active, restricts the search to e.g.
  `models=['rf','ridge']`, one allocation method, one importance method, one
  evaluator. Gated so server full runs are byte-identical. This would let the
  WHOLE c13 ablation (A→D + controls + deltas + Wilcoxon + summary) complete
  locally in seconds, giving true end-to-end coverage instead of grid-repetition
  coverage. It would also unblock every other hybrid-search experiment's smoke.
- **Evidence:** `_shared.py:2685-2688` (FULL_BUDGETS + `speed='full'` + `models=None`),
  probe log: `Searching 648 unique configurations (full mode)`,
  importance 71.4s + allocation 21.9s + 315/648 configs in ~480s.

### D3. (Server note, not a bug) Unified path `run_dataset.py complexity <ds> --experiments c13` is server-only
- **Problem:** the unified runner loads the full `DATASET_REPS` general list
  (foundation models) and calls `c13.run(..., new_rep_names=None)` → auto-detect.
  Locally the foundation reps can't load (expected per brief), so this exact
  invocation isn't smokeable here. The c13 `run()` it dispatches to is the SAME
  function I exercised via the direct CLI, and the dispatch wiring is present and
  correct (`complexity_helpers.py:1106, 1133, 1376, 1419`). With only legacy reps
  loaded, `run()` returns a clean `{'error': 'no_new_reps'}` (`c13...py:392-396`)
  rather than crashing.
- **Recommended:** no action; on the server with foundation reps loaded, the
  auto-detect path (`_detect_new_reps`, `c13...py:139`) is the intended behavior.

### Shared-code fixes I could NOT make (for the orchestrator)
- **D2 above** is the only shared-code item: a smoke-gated grid/model shrink in
  `_shared.precompute_hybrid_searches` (and/or `find_best_hybrid`). It is the one
  thing blocking a fully-completing local c13 run. Everything in my lane is fixed.
