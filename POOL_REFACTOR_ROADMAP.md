# Pool Refactor Roadmap (2026-05-23)

**Goal:** Every experiment that picks a hybrid blueprint should compare across the available pool of (tier × aug_condition) cache files and pick the best per (downstream_model, budget) tuple. Currently every experiment hardcodes ONE (tier, aug_condition) tuple — picking from a pool of size 1, which defeats the purpose.

This document is the single source of truth for the refactor. Verified line-by-line via direct grep + manual file reads on 2026-05-22/23. Past agent reports have hallucinated; trust only what's cited with file:line below.

---

## Current state of capped cleanup (already done)

Commit `8550761` dropped `general/capped` precompute and redirected consumers to `top/capped`:

**Infrastructure (4 files):**
- `tests/experiments/_shared.py:2139-2143` — `precompute_hybrid_searches` only emits `'capped'` when `tier='top'`
- `tests/precompute_hybrids.py:307-322` — dropped `('general', _, _, 'capped')` condition emission
- `tests/slurm_scripts/submit_precompute.sh:88-95` — dropped `${ds}_general_capped` SLURM job
- `tests/check_precompute_status.py:75-89` — dropped `('general', 'capped')` from DONE criterion

**Consumers switched to top/capped (commit 8550761):**
- `tests/experiments/complexity/c02_complexity_grid.py:163` (search_capped pass, also switched base_reps to top_tier_reps)
- `tests/experiments/complexity/c09_formal_analysis.py:310, 454, 596` (3 find_best_hybrid calls in 9b/9c/9d)
- `tests/experiments/complexity/c10_noise_robustness.py:123`
- `tests/experiments/complexity/c11_ensemble_vs_hybrid.py:93`
- `tests/experiments/complexity/c16_feature_selection_as_regularization.py:205`
- `tests/experiments/complexity/c17_information_theoretic_decomposition.py:244`
- `tests/experiments/complexity/c18_diversity_vs_depth.py:149`
- `tests/experiments/complexity/c19_importance_weighted_training.py:120`
- `tests/experiments/similarity/s08_optimal_dimension.py:105`

**Still using `'capped'` literal — but DON'T read precompute (own per-iteration cache):**
- `tests/experiments/complexity/c05_learning_curves.py:102` (per sample-size × seed)
- `tests/experiments/complexity/c06_within_family_gradient.py:74` (per model variant)
- `tests/experiments/complexity/c07_data_complexity_partitions.py:143` (per partition × subpop)
- `tests/experiments/complexity/c09_formal_analysis.py:349, 486` (create_hybrid inside per-fold loops, no cache)

**Verified clean of `'capped'` entirely (direct grep):**
- All 13 noise files (n01–n13)
- All 5 federated files (fed01–fed05) + federated_helpers
- All 10 OPIG files + imrie_transfer_learning
- All 17 framework files (f00–f16). f15 has a local variable named `capped` for *budget* capping — different concept.
- Similarity except s08

---

## The pool helper (to be written)

### Where it goes
`tests/experiments/_shared.py`, alongside existing `get_best_precomputed_search` (line 2316).

### Existing reusable pieces
- `_top_score(search)` at `_shared.py:2364` — extract top-across-models score, respecting `lower_is_better`. Reusable.
- `get_best_for_downstream(search, model, budget)` at `_shared.py:2499` — 5-level lookup chain that picks best config for (model, budget) from one SearchResult.
- `get_precomputed_search` at `_shared.py:2235` — fetches one (tier, aug_cond) SearchResult from in-memory dict or disk cache.

### New helpers

```python
def get_best_blueprint_for_downstream(
    data, downstream_model, budget,
    *, dataset_name=None, task_type=None, domain=None, metric='auto',
    speed='full', extra_aug_pool=None,
):
    """Pick the best blueprint across the full pool of available
    (tier × aug_condition) precomputed SearchResults.

    Pool considered:
      - data['precomputed_searches']      keyed by ['none', 'uncapped']  (general tier)
      - data['precomputed_searches_top']  keyed by ['none', 'uncapped', 'capped']  (top tier)
      - (general/capped intentionally excluded — dropped 2026-05-22)

    For each available SearchResult, look up the best (allocation, importance,
    internal_evaluator) config for (downstream_model, budget) via
    get_best_for_downstream. Compare all candidates by score, respecting
    lower_is_better. Return the winner.

    Returns:
        (blueprint, metadata) where metadata = {
            'tier': 'general' | 'top',
            'aug_condition': 'none' | 'uncapped' | 'capped',
            'score_mean': float,
            'score_std': float,
            'n_features': int,
            'model': str,
        }
        Returns (None, None) if no candidates exist.
    """

def get_best_search_for_downstream(
    data, downstream_model, budget,
    *, dataset_name=None, task_type=None, domain=None, metric='auto',
    speed='full',
):
    """Same selection logic but returns the winning SearchResult itself
    (for callers that need search.baselines, search.fitted_results,
    extract_top3_base_reps, etc).

    Returns:
        (search, metadata)
    """
```

### Edge cases the helper must handle

1. **`data['precomputed_searches_top']` doesn't exist** — fall back to general only
2. **`data['precomputed_searches']` doesn't exist** — fall back to top only
3. **Both missing** — return `(None, None)`; caller decides whether to error or run on-the-fly
4. **All searches have no `best_by_model[downstream_model]`** — return `(None, None)`
5. **Some scores are NaN/inf** — skip those candidates
6. **All candidates have identical scores** — prefer simpler in this order: none > uncapped > capped, general > top
7. **`lower_is_better` differs across SearchResults** — should never happen (same dataset, same metric) but guard with assert
8. **Classification vs regression** — `lower_is_better` is carried by SearchResult; no special handling needed

### Distance tier — separate pool, separate helper

Distance precompute (`tier='general_distance'` / `tier='top_distance'`) is a different cache lineage searched with `DISTANCE_EVALUATORS` and `DISTANCE_BUDGETS`. Per user decision (2026-05-22) keep distance as-is — no new precompute jobs.

Distance pool: `{general_distance, top_distance} × {none, uncapped}` = 4 combinations. No capped in distance precompute.

Same helper structure but iterates the distance-specific cache files. Could be the same function with a `distance=False` flag, or two functions. Recommend two functions (`get_best_distance_blueprint_for_downstream`) to avoid bug surface from a flag.

---

## Per-experiment treatment decisions

Decided by user 2026-05-22/23.

### c05 learning curves
"Use both" — add a second pass that applies the dataset-level best blueprint (from the pool helper, using full-data precompute) at each sample size, *in addition to* the existing per-subsample search. Gives two comparisons:
- Existing: best hybrid AT each n (per-subsample search)
- New: how well does the full-data optimal hybrid hold up at each n

### c06 within-family gradient
"Pool inside the loop" — each per-variant search picks best from the pool. More compute, but methodologically pure.

### c07 data complexity partitions
Same as c05 — add a second pass using the dataset-level best blueprint, alongside the existing per-partition search.

### c14 cross-domain summary
**No capped, no refactor needed.** Uses `'uncapped'` only (verified at c14:197). Convert to pool helper anyway for consistency (best blueprint instead of one hardcoded condition).

### c15 ml theory depth
**No capped, no refactor needed.** Uses `'none'` only. Convert to pool helper for consistency.

### s08 optimal dimension
Already redirected to top/capped. Refactor to use pool helper (currently still hardcodes `'capped' if augmentations else 'none'`).

### All others
Convert to pool helper. Each hardcoded (tier, aug_cond) call becomes a pool-best call.

---

## Full inventory of sites to refactor

Verified by `grep -rn "find_best_hybrid(\|get_precomputed_search(\|get_fitted_hybrid(\|create_hybrid(\|get_best_for_downstream(" tests/experiments/` on 2026-05-23.

### Complexity (refactor all that read precompute)

| File | Line | Function | Current (tier, aug_cond) | Treatment |
|---|---|---|---|---|
| c02_complexity_grid.py | 76 | find_best_hybrid | (top, none) | leave — explicit baseline |
| c02_complexity_grid.py | 107 | find_best_hybrid | (general, none) | leave — explicit baseline |
| c02_complexity_grid.py | 134 | find_best_hybrid | (general, uncapped) | leave — explicit baseline |
| c02_complexity_grid.py | 163 | find_best_hybrid | (top, capped) | leave — explicit baseline |
| c02_complexity_grid.py | 203 | find_best_hybrid | (general_ft, none) | leave — FT-specific |
| c02_complexity_grid.py | 323 | find_best_hybrid | (general, none) | leave — per-model top-rep-dim |
| c03_cross_dataset_transfer.py | 85 | find_best_hybrid | (general_distance, uncapped/none) | distance pool helper |
| c03_cross_dataset_transfer.py | 103 | get_best_for_downstream | (svm) | feeds from c03:85 search |
| c03_cross_dataset_transfer.py | 127 | create_hybrid | (blueprint-driven) | follows c03:85 result |
| c05_learning_curves.py | 102 | find_best_hybrid | (general, capped) | KEEP existing call; ADD second pass that fetches dataset-level pool-best blueprint and applies it to each subsample |
| c06_within_family_gradient.py | 74 | find_best_hybrid | (general, capped) | EACH per-variant call should iterate the pool of (tier, strategy) and pick best — needs internal pool sweep, OR convert to use the precomputed pool-best blueprint per variant (cheaper) |
| c07_data_complexity_partitions.py | 143 | find_best_hybrid | (general, capped) | Same as c05 — keep existing + add pool-best-blueprint pass per partition |
| c08_calibration.py | 95 | find_best_hybrid | (general, uncapped/none) | Convert to pool helper |
| c08_calibration.py | 181 | create_hybrid | (blueprint-driven) | follows c08:95 |
| c09_formal_analysis.py | 310 | find_best_hybrid | (top, capped) | Convert to pool helper |
| c09_formal_analysis.py | 349 | create_hybrid | ('capped' hardcoded) | follows c09:310 blueprint — should use blueprint's strategy, not hardcoded |
| c09_formal_analysis.py | 454 | find_best_hybrid | (top, capped) | Convert to pool helper |
| c09_formal_analysis.py | 486 | create_hybrid | ('capped' hardcoded) | follows c09:454 blueprint |
| c09_formal_analysis.py | 596 | find_best_hybrid | (top, capped) | Convert to pool helper |
| c09_formal_analysis.py | 635 | augmentation_strategy in evaluate_cv_with_train | ('capped' hardcoded) | follows c09:596 blueprint |
| c10_noise_robustness.py | 123 | get_precomputed_search | (top, capped) | Convert to pool helper |
| c10_noise_robustness.py | 130 | get_best_for_downstream | feeds from c10:123 | follows |
| c10_noise_robustness.py | 248 | create_hybrid | (blueprint-driven) | follows c10:123 blueprint |
| c10_noise_robustness.py | 349 | create_hybrid | (blueprint-driven) | follows |
| c11_ensemble_vs_hybrid.py | 93 | get_precomputed_search | (top, capped) | Convert to pool helper |
| c11_ensemble_vs_hybrid.py | 100 | get_best_for_downstream | feeds from c11:93 | follows |
| c13_forward_compatibility.py | 184 | find_best_hybrid | (general, none/uncapped) | Convert to pool helper |
| c14_cross_domain_summary.py | 200 | find_best_hybrid | (no tier=, defaults; uncapped/none) | Convert to pool helper; also pass dataset_name=task_name (currently missing — bug) and FIX missing FULL_BUDGETS import |
| c15_ml_theory_depth.py | 103 | get_fitted_hybrid | (general, none) | Convert to pool helper |
| c15_ml_theory_depth.py | 293 | get_fitted_hybrid | (general, none) | Convert to pool helper |
| c15_ml_theory_depth.py | 326 | get_fitted_hybrid | (general, none) | Convert to pool helper |
| c15_ml_theory_depth.py | 611 | get_precomputed_search | (general, none) | Convert to pool helper |
| c15_ml_theory_depth.py | 615 | get_best_for_downstream | feeds from c15:611 | follows |
| c15_ml_theory_depth.py | 617 | get_best_for_downstream | fallback 'rf' | follows |
| c15_ml_theory_depth.py | 877 | get_precomputed_search | (general, none) | Convert to pool helper |
| c15_ml_theory_depth.py | 881 | get_best_for_downstream | feeds from c15:877 | follows |
| c15_ml_theory_depth.py | 883 | get_best_for_downstream | fallback | follows |
| c16_feature_selection_as_regularization.py | 205 | get_precomputed_search | (top, capped) | Convert to pool helper |
| c16_feature_selection_as_regularization.py | 212 | get_best_for_downstream | follows c16:205 | follows |
| c17_information_theoretic_decomposition.py | 244 | get_fitted_hybrid | (top, capped) | Convert to pool helper |
| c18_diversity_vs_depth.py | 149 | get_precomputed_search | (top, capped) | Convert to pool helper |
| c18_diversity_vs_depth.py | 204 | get_best_for_downstream | follows c18:149 | follows |
| c19_importance_weighted_training.py | 120 | get_fitted_hybrid | (top, capped) | Convert to pool helper |
| c21_single_rep_augmentation.py | — | (no hybrid calls — single-rep aug study) | n/a | leave |
| c22_explainability.py | 71 | get_precomputed_search | (general, none/uncapped) | Convert to pool helper |
| c23_evaluator_model_matching.py | 85 | find_best_hybrid | (general, dynamic) | Convert to pool helper (varies per matched evaluator) |
| c23_evaluator_model_matching.py | 229 | find_best_hybrid | (general, dynamic) | Convert to pool helper |
| c24_blueprint_transfer.py | 155 | find_best_hybrid | (general, dynamic) | Convert to pool helper (blueprint discovery) |
| c24_blueprint_transfer.py | 191 | create_hybrid | (blueprint-driven) | follows c24:155 |
| c24_blueprint_transfer.py | 254 | create_hybrid | (blueprint-driven) | follows |

### Similarity (refactor — distinguish main pool from distance pool)

Main pool (`tier='general'` or `tier='top'`):

| File | Line | Function | Current | Treatment |
|---|---|---|---|---|
| s02_hybrid_similarity.py | 149 | get_fitted_hybrid | (general, uncapped/none) | Convert to **main** pool helper |
| s03_feature_concentration.py | 55 | get_fitted_hybrid | (general, uncapped/none) | main pool helper |
| s08_optimal_dimension.py | 105 | get_fitted_hybrid | (top, capped/none) | main pool helper |
| s11_bayesian_optimization.py | 710 | get_precomputed_search | (general, none) | main pool helper |
| s12_cross_domain_cliffs.py | 158 | get_precomputed_search | (general, uncapped/none) | main pool helper |

Distance pool (`tier='general_distance'`):

| File | Line | Function | Current | Treatment |
|---|---|---|---|---|
| s04_activity_cliffs.py | 120 | get_fitted_hybrid | (general_distance, uncapped/none) | **distance** pool helper |
| s05_knn_prediction.py | 67 | get_fitted_hybrid | (general_distance, uncapped/none) | distance pool helper |
| s06_applicability_domain.py | 66 | get_fitted_hybrid | (general_distance, uncapped/none) | distance pool helper |
| s07_noise_bridge.py | 123 | get_fitted_hybrid | (general_distance, uncapped/none) | distance pool helper |
| s10_importance_downstream.py | 694 | get_precomputed_search | (general_distance, uncapped/none) | distance pool helper |
| s13_clustering_retrieval.py | 196 | get_fitted_hybrid | (general_distance, none) | distance pool helper |

s11 also has create_hybrid at 567, 1110, 1143 — these are BO-internal (rebuild hybrid as new training data arrives). Keep as-is, they use a blueprint that came from s11:710 (which becomes pool-best after refactor).

### Noise (refactor — all main pool)

All currently use `('general', 'uncapped' if augs else 'none')`. Convert all to main pool helper. Verified clean of `'capped'`.

| File | Line | Function | Current | Treatment |
|---|---|---|---|---|
| n01_activity_cliff_framework.py | 596 | get_fitted_hybrid | (general, uncapped/none) | main pool helper |
| n02_sample_weighting.py | 243 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| n03_sas_landscape.py | 414 | get_fitted_hybrid | (general, uncapped/none) | main pool helper |
| n04_gp_loo_detection.py | 188 | get_fitted_hybrid | (general, uncapped/none) | main pool helper |
| n05_noise_injection_benchmark.py | 104 | get_fitted_hybrid | (general, uncapped/none) | main pool helper |
| n06_bindingdb_ground_truth.py | 93 | get_fitted_hybrid | (general, uncapped/none) | main pool helper |
| n07_remediation.py | 181 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| n08_estimation_survey.py | 90 | get_fitted_hybrid | (general, uncapped/none) | main pool helper |
| n09_lnl_survey.py | 105 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| n10_hybrid_neighborhood_lnl.py | 420 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| n11_estimation_remediation_grid.py | 278 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| n12_importance_weighted_distance.py | 238 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| n13_noise_distributions.py | 212 | get_precomputed_search | (general, uncapped/none) | main pool helper |

n10/n11/n12/n13 also have create_hybrid calls inside per-fold helpers — those use `bp.augmentation_strategy` from the blueprint, which becomes pool-best after refactor. Leave as-is.

### Federated (refactor — all main pool)

All currently use `('general', 'uncapped' if augs else 'none')` for blueprint discovery, plus a couple of per-silo `find_best_hybrid` calls hardcoded to `'none'`.

| File | Line | Function | Current | Treatment |
|---|---|---|---|---|
| fed01_noise_heterogeneity.py | 133 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| fed02_rep_diversity_silos.py | 111 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| fed03_synthetic_private.py | 95 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| fed03_synthetic_private.py | 545 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| fed03_synthetic_private.py | 826 | find_best_hybrid | (general, 'none') | per-silo blueprint discovery — leave as-is OR convert to pool. User decision needed. |
| fed04_cluster_aware_importance.py | 150 | get_precomputed_search | (general, uncapped/none) | main pool helper |
| fed05_antibody_federated.py | 153 | find_best_hybrid | (general, 'none') | per-silo blueprint discovery — same question as fed03:826 |

### OPIG (refactor — main pool)

OPIG scripts are standalone — they load data and run `find_best_hybrid` directly. They may not have `data['precomputed_searches']` populated. Pool helper needs to gracefully fall back to on-demand search across the pool.

| File | Line | Function | Current | Treatment |
|---|---|---|---|---|
| imrie_transfer_learning.py | 193 | find_best_hybrid | (general, none) | main pool helper |
| opig_activity_cliff_qsar.py | 311 (approx — verify by re-read) | find_best_hybrid | (general, uncapped/none) | main pool helper |
| opig_blueprint_transfer.py | 112 | get_precomputed_search | (general, none) | main pool helper — source dataset |
| opig_blueprint_transfer.py | 123 | get_precomputed_search | (general, none) | main pool helper — target dataset |
| opig_blueprint_transfer.py | 154–155 | get_best_for_downstream | follows c-112/123 | follows |
| opig_blueprint_transfer.py | 176 | create_hybrid | (blueprint-driven) | follows |
| opig_cdr_flexibility.py | (verify line) | find_best_hybrid | (general, none) | main pool helper |
| opig_cgr_comparison.py | (verify line) | find_best_hybrid | (general, none) | main pool helper |
| opig_ginkgo_developability.py | 277 | find_best_hybrid | (general, uncapped/none) | main pool helper |
| opig_graphinity_ddg.py | 322 | find_best_hybrid | (general, uncapped/none) | main pool helper |
| opig_humanness.py | 240 | find_best_hybrid | (general, none) | main pool helper |
| opig_kingsmith_component_splits.py | (verify line) | find_best_hybrid | (general, uncapped/none) | main pool helper |
| opig_trastuzumab_enrichment.py | (verify line) | find_best_hybrid | (general, none) | main pool helper |

### Framework — DO NOT TOUCH

User directive: framework experiments (F00–F15) are off-limits.

### Other top-level scripts — verify scope

| File | Line | Function | Treatment |
|---|---|---|---|
| reaction_rep_comparison.py | 410 | find_best_hybrid | unknown scope — ask user before touching |
| covariate_shift.py | 239 | find_best_hybrid | unknown scope — ask user before touching |

---

## Implementation order

1. **Write the two pool helpers in `_shared.py`** (main + distance variants). Test in isolation by writing a tiny `test_pool_helpers.py` that loads a known dataset's precomputed dict and verifies the helpers return the expected winner.

2. **Convert ONE consumer end-to-end as a template** — pick `c10_noise_robustness.py` since it's small and we already touched it. Verify the script imports cleanly, the call site is unambiguous, and the metadata returned makes sense.

3. **Batch-convert similar consumers** by call pattern:
   - All `get_precomputed_search(...)` → `get_best_search_for_downstream(...)` callers (one batch)
   - All `get_fitted_hybrid(...)` callers (another batch — these also need a fitted-on-full-data hybrid; might need to keep get_fitted_hybrid as a thin wrapper over pool helper + fit)
   - All `find_best_hybrid(...)` callers that should now read precompute — convert to pool helper, fall back to find_best_hybrid only when no precompute exists
   - Distance-tier callers (separate batch with distance helper)

4. **Special cases done individually:**
   - c02 — leave its 6 calls as explicit baselines (they're the precompute *producers*, not consumers)
   - c05/c07 — add second pass for "full-data blueprint applied to subsample"
   - c06 — pool inside the per-variant loop (each variant search picks best from pool)
   - c09 — find_best_hybrid switches to pool helper; create_hybrid calls follow blueprint
   - fed03:826 + fed05:153 — user decision on whether per-silo blueprint discovery should pool

5. **After every batch:** AST-parse every edited file with python3. Diff every changed file manually. Run `grep -rn "'capped'\|\"capped\"" tests/experiments/ src/kirby/` to catch regressions.

6. **End-to-end smoke test:** import every edited experiment module locally (just `python -c "import experiments.complexity.cXX_..."` — should not error). Cannot run the actual experiments locally per the no-heavy-local-compute rule.

7. **Commit in 4-5 logical chunks** (helpers, simple consumers, complex consumers, distance consumers, special cases). Push after each.

---

## Smoke test protocol

After every chunk of edits:

```bash
# Syntax check every edited Python file
python3 -c "import ast; ast.parse(open('FILE').read())"

# Grep for residual hardcoded patterns
grep -rn "'capped'\|\"capped\"" tests/experiments/ src/kirby/

# Grep for any consumer still bypassing the pool helper
grep -rn "aug_condition=.*precomputed=data.get" tests/experiments/

# Verify imports work
cd tests && python3 -c "from experiments._shared import get_best_blueprint_for_downstream, get_best_search_for_downstream, get_best_distance_blueprint_for_downstream"
```

---

## Anti-hallucination reminders for future-self

- The 2026-05-21/22 audit agents hallucinated about c11 (claimed it used capped — true), c14 (claimed it used capped — FALSE, uses uncapped), c15 (claimed it used capped — FALSE, uses none), c10 line numbers (claimed lines 255/356 — FALSE, only line 125).
- Always verify with `grep -n "'capped'\|\"capped\""` directly before believing any audit summary.
- The user has been explicit: NO summaries from memory, ALWAYS re-grep before claiming anything about file contents.
- Direct file reads beat agent reports.
- Save user decisions to memory immediately (already done for the per-experiment treatments above).

---

## What hasn't been decided yet

1. **fed03:826 and fed05:153** per-silo `find_best_hybrid` calls with `'none'` hardcoded — these *create* per-silo blueprints by searching from scratch on silo data. The pool concept doesn't directly apply (no precompute per silo). Options:
   - (a) leave hardcoded `'none'` since silos don't have augmentations anyway
   - (b) extend to also try `'uncapped'` per silo when augmentations are available
   - (c) consider this a separate "per-silo precompute" question
2. **reaction_rep_comparison.py:410** and **covariate_shift.py:239** — top-level scripts not in any taxonomy. Skip or include?
3. **Pool helper API name** — `get_best_blueprint_for_downstream` vs shorter `pool_best_blueprint`. User preference?
