# c23-c24-mangle-dataset-name

**title:** c23 and c24 namespace the cache by rewriting dataset_name, which also turns off the budget cap and the model skips

**area:** tests/experiments/complexity c20+ (c20, c21, c22, c23, c24, c30, c31, c32, c33, exp_b, c43_signal, c43_step3) plus tests/experiments/_alloc_harness.py and _run_complexity_locally.py

**class:** B-never-hits

**severity as reported:** large

**location:** tests/experiments/complexity/c23_evaluator_model_matching.py:104, c23_evaluator_model_matching.py:296, tests/experiments/complexity/c24_blueprint_transfer.py:192, c24_blueprint_transfer.py:341; contrast src/kirby/find_best_hybrid.py:6796-6801 and tests/experiments/dataset_policy.py:189-201

## scenario as reported
On herg_ki, c23 calls nested_cv_evaluate with dataset_name='herg_ki_rf_lightgbm'. find_best_hybrid runs _apply_budget_policy on that string. resolve_budget_cap tests `dataset_name in _BUDGET_CAPPED_DATASETS`; 'herg_ki_rf_lightgbm' is not in the set, so the 2000-feature budget survives. Every other experiment on herg_ki searches budgets capped at 1000. c23's evaluator comparison is therefore decided over a budget grid one step wider than the rest of the suite, and no reader of the file can see that. Separately, all 11 downstream models x 3 evaluator conditions produce 33 different dataset_name strings on one dataset, so baselines/, folds/, fold_scores/ and result_db/ are 33 separate stores. A_matched for 'rf' (evaluator rf), B_slight_off for 'rf' (evaluator lightgbm) and C_strong_off for 'rf' (evaluator ridge) each need the same per-rep baseline for model 'rf' on outer fold 0 and the same per-fold importance scores; each computes them from cold because the dataset key differs. c24 does the same thing per fold at line 192.

## evidence as reported
c23:104 `dataset_name=f"{dataset_name}_{model_name}_{evaluator}"`. find_best_hybrid.py:6796-6801 (the cache_namespace docstring): "Do NOT namespace by mangling dataset_name instead: that silently bypasses _apply_budget_policy / _applied_skips, changing the budget list and model set for the run." find_best_hybrid.py:7150-7153 `_ds_cache = dataset_name; if dataset_name and cache_namespace: _ds_cache = f'{dataset_name}__{...}'` with the comment "Every policy lookup above this line uses the TRUE dataset_name; every cache path below uses _ds_cache." dataset_policy.py:189-201 `capped = (dataset_name in _BUDGET_CAPPED_DATASETS or dataset_name.startswith(('nt_','gb_')))` — exact membership, so any suffix defeats it. tests/experiments/_shared.py:329 `FULL_BUDGETS = [10, 25, 50, 100, 250, 500, 1000, 2000]`.

## fix as proposed
Pass the true dataset_name and move the discriminator into the parameters that exist for it: `cache_namespace=` (namespaces every cache path, leaves the policy lookups alone) or, where only the search pickle needs separating, `condition_suffix=`. c23:104 becomes dataset_name=dataset_name, cache_namespace=f'{model_name}_{evaluator}'; c23:296 cache_namespace=f'quick_{model_name}'; c24:192 cache_namespace=f'fold{fold_i}'. Note this changes the budget grid c23/c24 search on the 22 capped datasets, so the existing c23/c24 result JSONs are not comparable to reruns.
