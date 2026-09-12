# c18-natural-arm-runs-at-a-different-budget

**title:** c18's natural arm is built at the blueprint's budget, not the sweep's, and the pool helper silently returns another budget's row

**area:** tests/experiments/complexity/c01–c19 + complexity_helpers.py (cache behaviour audit)

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/complexity/c18_diversity_vs_depth.py:171 and :243; tests/experiments/_shared.py:5356-5359; tests/experiments/complexity/c15_ml_theory_depth.py:105,344,417

## scenario as reported
Run 1 is the suite precompute on logd. logd is in _BUDGET_CAPPED_DATASETS (tests/experiments/dataset_policy.py:180-190) with _BUDGET_CAP_MAX=1000, so find_best_hybrid.py:7017 drops 2000 and no row at budget 2000 is ever searched. Run 2 is c18 on logd: c18:134 takes budgets from config['budgets'] = FULL_BUDGETS = [...,1000,2000] and filters only on total_dim, so it asks the pool for budget=2000. get_best_for_downstream finds 'rf' in best_by_model_budget but no key 2000, falls through to the 'best across all budgets' branch, and returns (say) the budget=250 row. c18:243 then calls build_fold_hybrid with no budget_override, so the natural arm is a 250-feature hybrid, while depth_1/depth_2/depth_3 and uniform_all in the same loop really do use 2000 features. The row written to results['per_budget'][2000] compares a 250-wide hybrid against 2000-wide baselines and calls both 'budget 2000'. c15's per_budget/gamma sweep has the same shape: get_fitted_pool_hybrid builds with bp.budget (_shared.py:5359-5361 in that function) and c15 files it under the requested budget.

## evidence as reported
_shared.py:5350-5359 — `if budget is not None and budget in model_budgets: ... ` then unconditionally `valid = _valid(model_budgets.values()); if valid: return (min if lower else max)(valid, key=...)`. c18:243-249 — `build_fold_hybrid(train_reps, val_reps_fold, y_train, bp, augmentations_train=None, augmentations_val=None, groups=..., model=_RF(...), metric_fn=metric_fn)` with no budget_override. c18:216 and :224 pass `budget` to _select_top_features/_uniform_allocation. c10 does this correctly (c10:387 `budget=budget,  # pin to sweep axis (not bp.budget)`) and c17 does too (c17:401 `budget_override=budget`).

## fix as proposed
Pass `budget_override=budget` at c18:243 the way c17:401 does, and have get_fitted_pool_hybrid build at the requested budget rather than bp.budget for c15. Separately, make get_best_for_downstream's cross-budget fallback record which budget it actually returned in the metadata (e.g. meta['budget_served']) so a caller can refuse or label it — today the substitution is invisible.
