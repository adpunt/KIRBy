# Multi-stage adaptive pruning — design + implementation plan

## Context

GP at high budgets (b≥100) dominates fold-0 wall time on mid-size datasets (n≈1000–5000). It rarely contributes to the winning hybrid in those settings, but currently runs to completion before any pruning decision is made. The existing `adaptive_pruning=0.15` mechanism prunes only at the end of fold 0 and only across allocation strategies — it cannot reduce search space *within* fold 0.

A model-aware multi-stage pruning scheme would let the search prune expensive configs based on cheaper-config evidence collected mid-fold-0, and let experiment owners dial pruning aggressiveness per model. **The pruning never removes a model entirely** — every model still produces a `best_by_model[m]` entry, possibly from a smaller search space.

This is a change to the search algorithm, not a hack to unblock the current wave. It needs calibration before landing.

## Goals

1. Reduce GP/SVM wall time at high budgets without removing them from `best_by_model`.
2. Make pruning aggressiveness per-model configurable.
3. Replace the binary `speed='fast'/'full'` distinction with a continuum of pruning levels.
4. Keep `n_folds=5` everywhere (no 3-fold mode for small datasets — costs cross-experiment consistency for ~15% wall savings).
5. Default = current behavior, no result changes for existing experiments.

## Non-goals

- Removing models from the search.
- Auto-deciding model inclusion based on baselines.
- Switching to 3-fold for n<1000.

## API shape (final target)

Replace `adaptive_pruning: float = 0.15` with a schedule dict. Backwards-compatible default reproduces current behavior:

```python
adaptive_pruning_schedule = {
    # Budgets at which to checkpoint and drop tail configs.
    # Empty list = single end-of-fold-0 prune only (= current).
    'checkpoints': [],

    # Within each (model, evaluator) group at a checkpoint, drop bottom
    # X% by fold-0 score. Dropped configs do NO further fold-0 work AND
    # are excluded from full eval.
    'tail_drop_per_checkpoint': 0.5,

    # End-of-fold-0 strategy pruning threshold (current top-3 logic).
    'final_strategy_threshold': 0.15,

    # Floor: never prune a model below this many surviving configs.
    # Guarantees best_by_model[m] exists for every m.
    'min_configs_per_model': 1,
}

# Per-model override. Models not listed use the global schedule.
model_pruning_aggressiveness = {
    'gp':  {'checkpoints': [10, 25, 50], 'tail_drop_per_checkpoint': 0.75},
    'svm': {'checkpoints': [25, 50],     'tail_drop_per_checkpoint': 0.6},
}
```

Backwards compatibility: if caller passes `adaptive_pruning=0.15` (old API), translate to the new schedule with `final_strategy_threshold=0.15` and empty `checkpoints` — identical behavior to today.

## Implementation phases

### Phase 1 (THIS SESSION): f16 calibration experiment

**File:** `tests/experiments/framework/f16_multistage_pruning.py`

**What it does:** runs `find_best_hybrid` with various pruning schedules on a small set of representative datasets, measures (a) wall time, (b) result quality vs. an unpruned reference, (c) fraction of dropped configs that would have made it into top-K had they not been dropped.

**Test model:** **SVM**, not GP. SVM is kernel-based and slow at high budgets — same compute shape as GP — but doesn't have GP's LBFGS-NaN pathology to confound rankings. If multi-stage pruning preserves SVM rankings, the methodology generalizes; we then apply it to GP knowing the algorithm is sound (separate from GP's numerical issues).

**Datasets:**
- `flab_thermo_jain` (antibody n≈137, GP/SVM expensive)
- `caco2` (molecular n≈2158, GP/SVM expensive)
- `flab_pk_jain_heprt` (antibody n≈241, has both GP-success and GP-fail signal)

These are the regimes where multi-stage pruning would actually bite. qm9 already sample-count-skips GP/KNN, so it's not a useful test bed.

**Schedules to compare:**

| Name | checkpoints | tail_drop | final_threshold |
|---|---|---|---|
| baseline | [] | — | 0.15 |
| light | [50] | 0.25 | 0.15 |
| medium | [25, 50] | 0.50 | 0.15 |
| aggressive | [25, 50, 100] | 0.50 | 0.15 |
| very_aggressive | [10, 25, 50] | 0.75 | 0.15 |

Each schedule × dataset, 3 seeds for variance estimate. ~15 runs per dataset × 3 datasets = 45 runs. Each run is one `find_best_hybrid` call at full search; total compute ~1-2 days on long partition.

**Outputs (per run):**
- Wall time (search loop only, excludes I/O)
- `best_by_model[m].score_mean` for every m
- Top-3 (model, alloc, importance, budget, evaluator) configs by score
- Number of configs dropped at each checkpoint
- For each dropped config: would it have ranked in top-K of its model group at fold 0 if it had been allowed to complete?

**Success criteria:**
- A schedule is "acceptable" if best_by_model scores are within 2% of baseline for every model on every dataset, AND wall time is reduced ≥30%.
- Pick the most aggressive acceptable schedule as the new default-aggressive policy.
- Configs are tagged for per-model override based on which models drove the wall-time reduction.

**Format:** standard framework experiment. Subclasses or follows the same scaffolding as `f02_adaptive_pruning.py` and `f15_adaptive_pruning_validation.py`. Outputs JSON to `tests/results/framework/f16/`.

**Deliverable from Phase 1:**
- The experiment file itself.
- A calibration JSON (`tests/results/framework/f16/calibration.json`) with the recommended global schedule and per-model overrides — written by a small post-analysis script after results land.

### Phase 2: schedule API in `find_best_hybrid`

**File:** `src/kirby/find_best_hybrid.py`

**Change:** replace the single `adaptive_pruning: Union[bool, float]` parameter with `adaptive_pruning_schedule: Optional[Dict]` AND keep the old parameter as a deprecated alias that translates to a schedule.

**Mechanics inside `_search_loop`:**

```
1. Build configs in original order.
2. For each model that has a checkpoint schedule applied:
   - Reorder its configs ascending by budget.
3. Iterate fold 0:
   for cfg in configs:
       if cfg.model_name in dropped_configs[cfg.cfg_key]:
           continue                        # pruned at a prior checkpoint
       result = _eval_cfg(cfg, fold_indices=[0])
       fold0_results.append((ck, result))

       if cfg.budget in checkpoints[cfg.model_name]:
           # Time to maybe prune this model's tail at this budget tier
           _maybe_prune_model_tail(cfg.model_name, cfg.budget, fold0_results,
                                    schedule, dropped_configs,
                                    floor=schedule.min_configs_per_model)

4. End-of-fold-0 strategy pruning at final_strategy_threshold (existing logic).
5. Full eval on surviving configs (existing logic).
```

Per-model floor (`min_configs_per_model`): when ranking + dropping bottom X%, never drop below floor. So `best_by_model[m]` is guaranteed populated.

**Cache key impact:** the schedule needs to be in `_make_search_meta` so different schedules produce different cache entries. Otherwise a cache written with `aggressive` would silently be returned to a caller asking for `baseline`. Add `'pruning_schedule_hash': hash(canonicalized_schedule)` to meta.

**Checkpoint format:** existing checkpoint already stores `pruned_strategies`. Extend with `dropped_configs` (set of cfg_keys dropped at mid-fold checkpoints) so SLURM resume continues correctly.

**Backwards compatibility:**
- Old API: `adaptive_pruning=0.15` (boolean or float).
- Translation: `{'checkpoints': [], 'tail_drop_per_checkpoint': 0.0, 'final_strategy_threshold': 0.15, 'min_configs_per_model': 1}`.
- Old callers see byte-identical behavior. Cache hits unchanged because translated-empty schedule hashes the same as no-checkpoints schedule.

### Phase 3: per-model overrides plumbed end-to-end

**File:** `src/kirby/find_best_hybrid.py`, `tests/experiments/_shared.py`, `tests/precompute_hybrids.py`.

Add `model_pruning_aggressiveness: Optional[Dict[str, Dict]]` to `find_best_hybrid` signature. Merge per-model override on top of global schedule when iterating that model's configs. Plumb through precompute and `_shared.get_precomputed_search`.

Default in precompute (after f16 calibration lands) — something like:

```python
DEFAULT_MODEL_PRUNING = {
    'gp':  {'checkpoints': [10, 25, 50], 'tail_drop_per_checkpoint': 0.75},
    'svm': {'checkpoints': [25, 50],     'tail_drop_per_checkpoint': 0.6},
}
```

Other models keep the global schedule (light or empty).

### Phase 4: retire `'fast'` tier

**File:** `src/kirby/find_best_hybrid.py`, `tests/precompute_ft_reps.py`, `tests/analysis/general_hybrid_vs_base.py`.

After Phase 2 + 3 land and validate:
1. Move `'full'` tier defaults to be the only tier (rename to `DEFAULT_SEARCH_CONFIG` or just inline).
2. Replace remaining `speed='fast'` callers with `adaptive_pruning_schedule='aggressive'` (a named preset that maps to the f16-calibrated heavy-pruning schedule).
3. Delete `SPEED_TIERS['fast']`.

The `EXPERIMENT_TRACKER.md` note about silent `speed='fast'` regression makes this concrete: removing the tier prevents that class of bug from recurring.

## Validation plan

After Phase 2 lands:
- Run `pytest tests/test_hybrid_search.py` and existing find_best_hybrid integration tests. All pass with default schedule.
- Run f15 (existing pruning validation) with default schedule. Results match prior f15 baseline (no behavior change).

After Phase 3 lands:
- Re-run f16 with `DEFAULT_MODEL_PRUNING` enabled. Confirm wall savings and quality match Phase 1 calibration.
- Spot-check on one new dataset (e.g. `ddg_combined`) that wasn't in f16 calibration. Same wall-time saving pattern should hold.

After Phase 4 lands:
- All experiment scripts that referenced `speed='fast'` updated. Grep confirms zero remaining production references.
- One full precompute wave on a small representative subset (3 antibody + 1 molecular) using the new defaults. Compare timing to old wave.

## What's deferred (explicitly NOT in this plan)

- Mid-flight pruning across **strategies** during fold 0 (the current end-of-fold-0 strategy pruning stays as-is; we're only adding mid-flight pruning across **configs within a model**).
- Successive-halving with re-evaluation at higher folds (current scheme prunes by fold-0 score only).
- Adaptive checkpoint placement based on observed score variance (checkpoints are static budget tiers, not learned).
- Removing or modifying `MODEL_MAX_BUDGET`. Existing per-model budget caps are orthogonal.
- Anything about the in-flight wave (11825801–11825990). The wave is operational, not methodological — handled separately.

## Open questions for discussion before Phase 1 starts

1. **Reference for "result quality":** is the comparison "best_by_model[m] within 2% of unpruned baseline" the right success criterion? Alternative: "top-3 hybrid configs (across all models) overlap by ≥2/3 with unpruned baseline." The second is more sensitive to ranking changes within the top region, which is what experiments actually consume.

2. **Seeds:** 3 seeds per (schedule, dataset) gives a noise floor estimate but multiplies compute. Acceptable, or scale down to 2?

3. **Per-model override format:** dict-per-model as above, or a single integer aggressiveness level (0=none, 1=light, ..., 4=very) that maps to a fixed schedule? Integer is cleaner UX but less flexible. Lean dict.

4. **Should the schedule include a "skip directly to full-eval if a model is dominated by ≥3σ at fold 0" rule?** This would be a per-model adaptive cutoff *during fold 0* if the data is overwhelming. Adds complexity. Probably defer to a v2.

5. **f16 datasets:** the three above span small antibody, mid molecular, and a checkpoint-rich antibody. Add a fourth (e.g. nucleotide n≈30K with GP/KNN sample-skipped) to confirm the schedule doesn't break anything in the regime where pruning is irrelevant?

## Implementation order

1. **f16 experiment file** ← start here (this session)
2. f16 SLURM submit script + run on server
3. f16 results analysis + calibration JSON
4. Phase 2: schedule API in find_best_hybrid (no behavior change at default)
5. Phase 3: per-model override plumbing through precompute + _shared
6. Phase 4: retire `'fast'` tier
7. New precompute wave with calibrated defaults applied

Everything between step 4 and step 7 is mechanical once step 3 (calibration) settles. Steps 1–3 are the real research; the rest is plumbing.
