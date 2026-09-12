# mangled-dataset-name-bypasses-the-budget-policy

**title:** c05, c07 and c16 namespace the cache by mangling dataset_name, which the parameter's own contract forbids

**area:** tests/experiments/complexity/c01–c19 + complexity_helpers.py (cache behaviour audit)

**class:** A-wrong-answer

**severity as reported:** small

**location:** tests/experiments/complexity/c05_learning_curves.py:241; tests/experiments/complexity/c07_data_complexity_partitions.py:285; tests/experiments/complexity/c16_feature_selection_as_regularization.py:291; src/kirby/find_best_hybrid.py:6800-6803, :7017

## scenario as reported
c07 on logd: the whole-dataset precompute goes through find_best_hybrid.py:7017 `budgets = _apply_budget_policy(dataset_name, budgets)`, logd is in _BUDGET_CAPPED_DATASETS, and 2000 is dropped — the suite never searches it. c07:285 then passes dataset_name='logd_label_quartile_easy'; resolve_budget_cap (dataset_policy.py:193-200) does an exact-name membership test, misses, and returns the budgets untouched, so the easy/hard subpopulations ARE searched at budget 2000. The 'easy' vs 'hard' vs whole-dataset comparison c07 draws is then over two different budget axes, and the extra cell is compute the cap exists to refuse. c05:241 does the same for every subsample point and c16:291 for the capped_full slot. Both arms of c05 are affected in one file: all_rows_blueprints at c05:131 is fetched under the REAL name (capped at 1000) while the per-subsample searches at c05:229 are not.

## evidence as reported
find_best_hybrid.py:6800-6803, the cache_namespace docstring — "Do NOT namespace by mangling dataset_name instead: that silently bypasses _apply_budget_policy / _applied_skips, changing the budget list and model set for the run." dataset_policy.py:196-200 — `capped = (dataset_name in _BUDGET_CAPPED_DATASETS or dataset_name.startswith(('nt_','gb_')))` — an exact-membership test that a suffixed name cannot satisfy (the nt_/gb_ prefix check survives, the fifteen listed names do not). c06:194-198 is the file in this area that gets it right: "Use real dataset_name + condition_suffix so the _DATASET_STRATEGY_SKIPS lookup still matches (the mangled dataset_name would never hit those entries)."

## fix as proposed
Replace the mangled names with the sanctioned namespace: c05 `dataset_name=dataset_name, condition_suffix=f'c05_n{n_sub}_s{seed}'`; c07 `dataset_name=dataset_name, condition_suffix=f'c07_{partition_name}_{subpop_name}'`; c16 `dataset_name=dataset_name` (tier='c16_fast' already separates it) with `condition_suffix=f'c16_capped{capped_full}'`. condition_suffix rides on tier into both the search pickle filename and IDENTITY_COLS[0] (find_best_hybrid.py:7345-7347), so the namespacing is unchanged and the policies resolve. Existing files under the mangled names become unreachable — delete them.
