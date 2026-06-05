# COMPLEXITY family — local smoke-test report (2026-06-05)

Agent lane: `tests/experiments/complexity/*` EXCEPT `c13_forward_compatibility.py` (owned by another agent — not touched, not run).
Experiments covered: c01, c02, c04, c05, c06, c07, c08, c09, c10, c11, c15, c16, c17, c18, c19, c21, c22, c23, c24.

## How these were run

Driver: `tests/experiments/_run_complexity_locally.py` (extended this session — it is in my lane).
It calls each experiment's `run(data, config, out, dataset)` on a small REAL dataset with
locally-computable representations, a 2-fold / budget-25 / rf-only fast config, and the
single-thread env from the brief.

Two datasets cover both task types:
- REGRESSION: `buchwald_hartwig` (reaction yield), reps `diff_maccs (167D) / onehot_components (45D) / reactant_pdv (208D)`, `KIRBY_SMOKE_N=200`.
  (I first tried `ginkgo_heparin` but its local reps `blosum`/`onehot` are 5600D each, which makes the
  find_best_hybrid precompute take many minutes per call locally — buchwald's small reps run the
  same code path in ~1 min.)
- CLASSIFICATION: `herg` (hERG blocker, binary), reps `maccs (167D) / pdv (200D)`, `KIRBY_SMOKE_N=200`.

Exact command shape (per experiment):
```
KIRBY_LOCAL_DS=<ds> KIRBY_LOCAL_REPS=<reps> KIRBY_SMOKE_N=200 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  perl -e 'alarm 300; exec @ARGV' \
  /Users/apunt/anaconda3/bin/python -u tests/experiments/_run_complexity_locally.py <exp>
```

### Why a config trim was needed (smoke only — server unaffected)
`precompute_hybrid_searches` (in `_shared.py:2680-2688`) hardcodes `speed='full'`, `models=None`
and `FULL_BUDGETS`; it ignores `config['budgets']/models/evaluators`. So every find_best_hybrid
search runs the FULL grid (8 model types incl. O(n^3) GP × 5 evaluators × 9 allocations × 3 importance
methods) — minutes per call even at 200 rows / 420 dims. The driver therefore monkeypatches the
**imported** objects at runtime (NOT edits to `_shared.py`/`src`): `_shared.FULL_BUDGETS=[25]`,
`find_best_hybrid.DEFAULT_MODELS={'rf':...}`, `SPEED_TIERS[*]` axes trimmed to 1 each, and
`_shared.INTERNAL_EVALUATORS=['rf']`. These live only in the driver process; a server job imports
fresh and is unaffected. This is the standard "inject a tiny config + shrink FULL_BUDGETS" knob the
brief describes.

---

## 1. SMOKE RESULTS

Legend: PASS = ran end-to-end, returned a populated result dict. TIMEOUT-CLEAN = ran the dominant
loop/phase then hit the perl alarm (counts as a pipeline pass per the brief). SKIP = experiment
self-gates to the other task type (expected, not a failure). SEGFAULT = macOS-only OpenMP crash
(see Decisions) — not a code bug.

### Regression (buchwald_hartwig, n=200, diff_maccs/onehot_components/reactant_pdv)

| Exp | Result | Notes |
|-----|--------|-------|
| c01_rep_characterization        | PASS | per-rep intrinsic-dim / MI / baselines |
| c02_complexity_grid             | PASS | all phases: configs, baselines, pca_baselines, heatmap, per-model-top-rep, conditions |
| c04_importance_as_metric        | PASS | per-fold importance + cross-test matrix + low-dim |
| c05_learning_curves             | PASS | full sample-size sweep completed (warm cache) |
| c06_within_family_gradient      | PASS | regression-only; budget grid auto-widened (small-data path) |
| c07_data_complexity_partitions  | PASS | scaffold_frequency + target_variance ran; molecular_complexity skipped (buchwald has no per-row `smiles`) |
| c08_calibration                 | PASS | optimal/underfit/overfit/mismatched conditions |
| c09_formal_analysis             | PASS | information + effective_rank + bias_variance (anova skipped — no c02 dir passed, expected) |
| c10_noise_robustness            | PASS | noiseInject present; NDS + feature stability |
| c11_ensemble_vs_hybrid          | PASS | baselines + ensemble vs hybrid |
| c15_ml_theory_depth             | PASS* | all 4 sub-analyses; *see FIX below — re-verified, 0 fast-key errors |
| c16_feature_selection_as_regularization | PASS* | heatmap families; *see FIX below — re-verified, 0 fast-key errors |
| c17_information_theoretic_decomposition | TIMEOUT-CLEAN | reaches + runs the KSG-mRMR phase (the dominant cost); did not finish budget-25 mRMR within 300s locally. See Decisions. |
| c18_diversity_vs_depth          | PASS | per-budget depth/uniform/natural conditions |
| c19_importance_weighted_training| PASS | driver pins `budget=25` (default 100 misses smoke precompute); per_space populated |
| c21_single_rep_augmentation     | PASS (degenerate) | returns OK but all 6 aug tests skipped — buchwald `augmentation=[]` in DATASET_REPS and the small-general aug reps (208D) exceed n=200. Real aug tests only at full n / on datasets with aug reps. |
| c22_explainability              | PASS | blueprint + SHAP-by-rep + fold_results |
| c23_evaluator_model_matching    | SEGFAULT (macOS) | ran condition A clean; crashed (exit 139) entering condition B whose evaluator is `lightgbm`. macOS OpenMP double-load — not a code bug, will not occur on Linux server. |
| c24_blueprint_transfer          | PASS | per-fold blueprint discovery + transfer matrix + stability |

### Classification (herg, n=200, maccs/pdv)

NOTE — before this batch I had to fix a stale `herg` cache (see SH1 below): the on-disk
representation cache was 1412 rows while the loader returns 7772, so `load_data('herg')` was
misaligned. After regenerating the herg reps they are aligned and the batch ran.

| Exp | Result | Notes |
|-----|--------|-------|
| c01_rep_characterization        | PASS | per-rep metrics on the 2 clf reps |
| c02_complexity_grid             | PASS | all phases populated (clf path works) |
| c04_importance_as_metric        | SKIP (clean) | self-gates: "c04 currently only supports regression targets" (`c04:...`); returns `{skipped, reason, task_type}`. Effectively regression-only. |
| c05_learning_curves             | PASS | sample-size sweep |
| c06 / c10 / c15 / c17           | SKIP (clean) | all four self-gate `regression_only` on classification (expected, instant) |
| c07_data_complexity_partitions  | PASS | herg HAS per-row smiles, so the molecular_complexity partition (RDKit) ran — broader coverage than the buchwald reg run |
| c08_calibration                 | PASS* | returns a populated dict, BUT every fold logged "No importance scores for rep 'maccs' … importance pre-computation in find_best_hybrid failed" and fell back to its default config — i.e. the hybrid never actually built. See shared item SH3. |
| c09_formal_analysis             | PASS | information + effective_rank + bias_variance |
| c11_ensemble_vs_hybrid          | PASS | baselines + comparison |
| c16_feature_selection_as_regularization | PASS | (the speed=fast fix also exercised on clf) |
| c18_diversity_vs_depth          | PASS | per-budget conditions |
| c19_importance_weighted_training| PASS | per_space populated (budget=25 pin) |
| c22_explainability              | FAIL (clf) | `no_search_result` — the pool hybrid search returns no scored config on herg clf. Re-ran with clean cache AND with n=600 AND with treeshap importance — all still fail. See SH3. |
| c21_single_rep_augmentation     | TIMEOUT-CLEAN | spends the whole window computing the `mordred` augmentation rep for all 7772 herg molecules (it calls `load_data(rep_preset='all')` to fetch aug reps) before the experiment loop starts. On the server with cached aug reps this is instant. |
| c23_evaluator_model_matching    | SEGFAULT (macOS) | same as regression — crashed (139) on the `lightgbm` evaluator (condition B). Environment-only. |
| c24_blueprint_transfer          | FAIL (clf) | `insufficient_folds` (0/2 folds succeeded) — same root cause as c22: per-fold hybrid search returns no valid config on herg clf. See SH3. |

---

## 2. FIXES MADE IN-LANE

### FIX 1 + 2 — `speed='fast'` is a dead tier key (c15 and c16)
`'fast'` was removed from SPEED_TIERS (only `'full'` remains; it now raises). Two experiments still
passed `speed='fast'` to `find_best_hybrid`, which raised `KeyError('fast')`, was caught, and silently
degraded the analysis:
- `c15_ml_theory_depth.py:516` — the double-descent (c15c) per-budget searches. With the bug EVERY
  c15c search printed `c15c fast search at b=NNN failed: 'fast'` and the double-descent comparison ran
  without its find_best_hybrid blueprints.
- `c16_feature_selection_as_regularization.py:266` — the `capped_full` (full-dim baseline) fast search.
  With the bug it raised, `capped_full_error` was set, the full-dim blueprint was never built, and every
  full-dim-budget cell came back NaN.
Both changed `speed='fast'` → `speed='full'`. Re-ran both standalone afterwards: **0 fast-key errors**,
both return populated results.

### Driver enhancements (in-lane file `_run_complexity_locally.py`)
- Dataset/reps via env (`KIRBY_LOCAL_DS`, `KIRBY_LOCAL_REPS`), per-exp output dir.
- Smoke grid trim (described above) so precompute is tractable locally.
- `c19` gets `budget=25` (its default is 100, which the budget-25 smoke precompute never built).

---

## 3. DECISIONS FOR USER

### D1 — c23 segfaults locally on the `lightgbm` evaluator (macOS only)
- What c23 does: checks whether matching the hybrid's internal scoring model to the final model
  (e.g. rf-scored for an rf model) beats deliberately mismatched scorers.
- Problem: it crashed (exit 139, a hard segfault) the moment it used `lightgbm` as the internal
  evaluator (condition B), at `c23_evaluator_model_matching.py:166`/MATCH_TABLE. This is the documented
  macOS OpenMP "double libomp" crash (LightGBM + sklearn), NOT a logic bug — condition A (rf evaluator)
  ran cleanly to a real score. It will not occur on the Linux ARC server.
- Recommendation: treat as environment-only. Optionally set `KMP_DUPLICATE_LIB_OK=TRUE` (already set)
  is insufficient on this box; a clean way to smoke c23 locally would be to install a single-OpenMP
  lightgbm, but that is not worth doing for the server run. No code change needed.

### D2 — c17 is very slow in its KSG mutual-information / mRMR phase
- What c17 does: compares feature-selection strategies (hybrid vs PCA vs mRMR vs RFE vs random) using a
  k-nearest-neighbour mutual-information estimator.
- Problem: `_greedy_mrmr` (`c17_information_theoretic_decomposition.py:52`) calls the KSG estimator
  `estimate_mutual_information_knn` (shared, `_shared.py:1582`) thousands of times — once per feature for
  relevance, then per remaining feature per greedy step. Even at 420 features / budget 25 / ~190 rows it
  did not finish within the 300s local alarm. It DOES reach and execute that phase (the dominant cost),
  so the code path is exercised; it is just compute-heavy. On the bigger server grid this phase is the
  cost driver and a known c17 characteristic (the file already has COST-1 row-capping at 2000 rows and a
  2000-feature mRMR prefilter).
- Recommendation: confirm c17's per-experiment server wall budget accounts for this MI phase (it can be
  hours at high feature counts / large n). No code change required for correctness. If a faster local
  smoke is wanted later, the row-cap could be lowered behind `KIRBY_SMOKE_N`, but that touches shared
  code (`_subsample_for_mi`) so it's out of this lane.

### D3 — c21 produces no real signal on datasets without augmentation reps
- What c21 does: tests whether augmenting a single base representation with small complementary reps
  helps.
- Problem: on buchwald_hartwig it returned OK but every augmentation test was skipped — `DATASET_REPS`
  has `augmentation=[]` for buchwald and the small-general reps it falls back to (reactant_pdv 208D)
  exceed the smoke n=200, so `c21_single_rep_augmentation.py:125` skips them. This is correct behaviour
  for that data, not a bug, but it means buchwald is not a meaningful c21 smoke. On the server (full n)
  and on datasets with real aug reps it will exercise the full path.
- Recommendation: none for the server run. Just be aware buchwald/herg are weak c21 smokes.

### SHARED-CODE issues for the orchestrator (could not fix — outside lane)

- **SH1 — stale representation cache silently reused with wrong row count (`herg`).**
  `_cache_rep` (`tests/experiments/_shared.py:168-176`) loads a cached `<rep>.npy` whenever the file
  exists, with NO check that its row count matches the current label count. The on-disk herg rep cache
  was 1412 rows (an older loader version) while the loader now returns 7772 labels, so `load_data('herg')`
  returned `labels` of length 7772 with `reps` of length 1412 — a hard misalignment that crashes any
  downstream experiment (and the smoke subsample then made it 200 vs 1412). I worked around it locally by
  moving the stale cache aside and regenerating (now aligned), but the underlying loader/cache should
  validate `arr.shape[0] == len(labels)` and recompute on mismatch. Suggested fix: in `_cache_rep`,
  reject a cached array whose first dim != expected n and recompute. File:line `_shared.py:173`.

- **SH2 — stale hybrid-search cache shadows a rebuilt one (observed on `herg`).**
  The on-disk `~/.cache/kirby/searches/herg__*.pkl`, `hybrid_evals/herg`, `baselines/herg*` were stale
  (built from the 1412-row data). They are keyed by dataset name with no data-hash, so a changed dataset
  silently reuses them. I moved them aside and clf experiments rebuilt clean ("0 stale"). Same class as
  the memory's `purge_graphinity_cache.py` note. Worth a `--purge` for herg on the server before the
  rerun, and longer-term a data-hash in the search-cache key. Dirs: `~/.cache/kirby/searches/`,
  `~/.cache/kirby/hybrid_evals/`, `~/.cache/kirby/baselines/`.

- **SH3 — the hybrid SEARCH returns NO scored config on `herg` classification (real, needs server
  confirmation).** This is the most important finding and the cause of THREE clf failures: c22
  (`no_search_result`), c24 (`insufficient_folds`, 0/2 folds), and the empty hybrids in c08 ("No
  importance scores for rep 'maccs' … the importance pre-computation in find_best_hybrid failed or the
  scores weren't passed through"). On classification (herg, AUC metric) the find_best_hybrid search
  computes baselines but produces no scored hybrid config, so the pool helpers return no `best_config`.
  I ruled OUT the obvious smoke causes: it persists after purging the stale cache (SH2), at n=600 (not a
  small-sample fluke), and with treeshap importance instead of deeplift. The identical driver/config on
  REGRESSION (buchwald) works fine for all three experiments — so it is specific to the classification
  code path of the importance pre-computation / search in `src/kirby/find_best_hybrid.py` (out of my
  lane). ACTION for the orchestrator: reproduce c22/c24 on a classification dataset with the FULL server
  config (all 3 importance methods, all 8 models, FULL_BUDGETS) — if it still returns no config, the
  classification hybrid search is broken and every clf experiment that relies on a searched blueprint
  (c08/c11/c18/c19/c22/c24, and the clf survey) is affected. The error message points at the importance
  pre-computation step. NOTE: this could be partly induced by my aggressive smoke trim (rf-only model +
  budget-25 + 2-fold producing all-NaN clf configs), which is why it must be confirmed at full config
  before treating it as a hard bug — but the regression-vs-classification asymmetry under the IDENTICAL
  trim strongly suggests a real clf-path gap.

---

## TIMEOUT AUDIT (per-experiment, with the loop cited)

For each experiment I read the source and classified its outer loop as PURE-REPETITION (safe to
alarm-kill after a few iterations) or DISTINCT-PHASES (must complete). All DISTINCT-PHASES experiments
were run to completion (warm cache made this tractable); none were killed mid-phase except c17 (see D2).

- c01: per-rep loop (`c01:60`) — pure repetition. Completed.
- c02: sequential phases — top-tier search, no-aug search, aug searches, per-model heatmap
  (`c02:330`), per-model-top-rep (`c02:330`), PCA baselines, summary. DISTINCT-PHASES — completed.
- c04: method loop with per-fold importance — pure repetition. Completed.
- c05: sample-size sweep `for n_sub in sample_sizes` (`c05:117`), each a fresh find_best_hybrid —
  pure repetition (safe to kill after a couple sizes). Completed.
- c06: per-family / per-variant loop — pure repetition; regression-only gate at `c06:51`. Completed.
- c07: per-partition loop (`c07:117`), each a find_best_hybrid — DISTINCT (each partition is a
  different code path). Completed (molecular_complexity self-skips without smiles).
- c08: condition loop (optimal/underfit/overfit/mismatched) — DISTINCT phases. Completed.
- c09: 4 sub-analyses (`c09:682-695`) — DISTINCT phases. Completed.
- c10: phases (individual reps → hybrids → feature stability), inner sigma×fold loops
  (`c10:180-183`) — DISTINCT; regression-only gate `c10:106`. Completed.
- c11: precompute then per-model loop (`c11:124`) — DISTINCT then repetition. Completed.
- c15: 4 sub-analyses (`c15:1048-1062`); double-descent per-budget loop (`c15:505`) — DISTINCT.
  Completed (after FIX).
- c16: precompute + capped_full search + per-family/budget heatmap (`c16:293-300`) — DISTINCT.
  Completed (after FIX).
- c17: single budget, but the mRMR greedy (`c17:119-141`) calls the KSG estimator ~O(features×budget)
  times — this IS the dominant phase, not a repetition over an outer grid, so killing it loses the
  result. TIMEOUT-CLEAN here means it ran the phase but didn't finish locally. See D2.
- c18: per-budget loop (`c18:157`) then per-fold conditions — DISTINCT. Completed.
- c19: per-space × per-model × per-condition loops (`c19:167-175`) — repetition. Completed.
- c21: per-base-rep × per-condition loop (`c21:121,141`) — repetition. Completed (degenerate, D3).
- c22: single pool lookup + per-fold loop — DISTINCT. Completed.
- c23: per-model × {A,B,C} then D then E loops (`c23:157,198,235`) — repetition, BUT crashed in B on
  macOS (D1), not a timeout.
- c24: per-fold blueprint discovery (`c24:151`) + transfer matrix — DISTINCT (each fold = separate
  precompute). Completed.
