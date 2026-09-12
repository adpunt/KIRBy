# aug-logic-version-cannot-invalidate-a-result-db-row

**title:** AUG_LOGIC_VERSION says bumping it invalidates old entries with no deletion needed; the result database ignores it and the config is never re-evaluated

**area:** The search cache key (_make_search_meta / _search_cache_path / _search_checkpoint_path) and the result_db row identity (IDENTITY_COLS + CONFIG_COLS)

**class:** C-stale-undetected

**severity as reported:** large

**location:** src/kirby/find_best_hybrid.py:158-166 (the claim), 949 ('aug_logic_version' only in _make_hybrid_eval_meta), 6157-6160 (a stored row short-circuits the whole config)

## scenario as reported
Someone changes create_hybrid's augmentation admission logic and bumps AUG_LOGIC_VERSION 2 -> 3, following the instruction in the comment. Run 2 on flab_thermo_tresanco with aug_strategy='uncapped': every hybrid_evals/*.pkl for the dataset now mismatches on aug_logic_version and would recompute -- but the result_db row for (all, performance_weighted, 500, lightgbm_gain, rf, match, uncapped, ...) still matches its identity and config key, so line 6157 hits `continue` and _eval_cfg is never called. The per-fold recompute the bump was meant to trigger never happens, and the run reports scores produced by the old augmentation logic. Nothing prints or warns.

## evidence as reported
find_best_hybrid.py:158-166: "Bump when augmentation default params or logic changes so old hybrid_eval cache entries ... get invalidated automatically. Old entries lack this key -> meta-mismatch -> silently recomputed on demand. No deletion needed." AUG_LOGIC_VERSION appears exactly once more, at line 949 inside _make_hybrid_eval_meta. It is absent from _make_search_meta (480-655), from IDENTITY_COLS and from CONFIG_COLS (result_cache.py:46-91). The result_db is the only durable search-result store (result_cache.py:15-20; find_best_hybrid.py:271-273 'search-level hit/miss/stale counters were removed').

## fix as proposed
This one is a documentation-vs-behaviour defect, not a code fix -- no cache versioning is a locked decision. Either correct the comment at 158-166 to say a bump now also requires deleting result_db/*.db, or state the wipe in the same place the bump is made. Worth deciding with the owner because the comment currently reads as a safe, no-cost invalidation lever and is not one.
