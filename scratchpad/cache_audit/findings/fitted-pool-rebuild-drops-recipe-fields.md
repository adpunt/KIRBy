# fitted-pool-rebuild-drops-recipe-fields

**title:** get_fitted_pool_hybrid and get_fitted_hybrid rebuild a cached blueprint without seven of its own fields

**area:** Experiment-side caches in tests/experiments/_shared.py (create_hybrid_cached / _hybrid_fold_cache_key, rep_cache, the search-reuse helpers, and the model/evaluator lists that ride in the search cache key)

**class:** A-wrong-answer

**severity as reported:** small

**location:** tests/experiments/_shared.py:5163-5185 (get_fitted_pool_hybrid), 5266-5285 (get_fitted_hybrid); compare tests/experiments/_shared.py:4225-4258 (build_fold_hybrid, which does forward them); src/kirby/find_best_hybrid.py:1162-1202 (the Blueprint fields)

## scenario as reported
Both functions pick a blueprint out of a cached SearchResult and then call create_hybrid with allocation_method, budget, step_size, importance_method, internal_evaluator, augmentation_strategy and groups — and stop. Seven recipe fields the Blueprint carries specifically so a rebuild reproduces the search are dropped: aug_per_source_budget, augmentation_margin, augmentation_fraction, candidate_ranking, augmentation_threshold, augmentation_budget, aug_max_candidates. Concrete: a caller does get_fitted_pool_hybrid(..., candidate_ranking='rf_importance'); the kwarg rides through get_best_search_for_downstream into find_best_hybrid, so the search that comes back ranked augmentation candidates base-blind and the blueprint records candidate_ranking='rf_importance'. Line 5163 then rebuilds the returned X_hybrid with create_hybrid's 'residual_correlation' default, which ranks each candidate by what the base representation could not explain. The matrix handed to the caller was not built by the recipe the search chose, and nothing raises. build_fold_hybrid was fixed for exactly this on 2026-08-26 (its comment at 4239-4247 names it as 'the same shape of defect'); these two copies were missed.

## evidence as reported
tests/experiments/_shared.py:5163-5185 — the create_hybrid call lists allocation_method, budget, step_size, importance_method, internal_evaluator, augmentations, augmentation_strategy, groups and objective, and nothing else.
tests/experiments/_shared.py:4235-4257 — build_fold_hybrid passes augmentation_margin, augmentation_fraction, candidate_ranking, augmentation_threshold, augmentation_budget and aug_max_candidates, each with a getattr fallback to the blueprint.
src/kirby/find_best_hybrid.py:1180-1186 — 'candidate_ranking ... is a column of the stored result's KEY (result_cache.CONFIG_COLS), so the search files two rankers as two rows'.

## fix as proposed
Copy build_fold_hybrid's seven getattr(blueprint, ...) forwards into both create_hybrid calls. No stored result changes meaning today: every one of the seven Blueprint defaults equals create_hybrid's own default, so with the current in-scope callers the rebuilt matrix is already the right one — the defect is that it stops being right the moment a search runs under a non-default value.
