# rep-factories-id-not-in-result-db-identity

**title:** rep_factories_id is demanded when caching is on and then never reaches the result database

**area:** The search cache key (_make_search_meta / _search_cache_path / _search_checkpoint_path) and the result_db row identity (IDENTITY_COLS + CONFIG_COLS)

**class:** A-wrong-answer

**severity as reported:** large

**location:** src/kirby/find_best_hybrid.py:7377-7383 (the requirement), 7485-7505 (identity has no slot for it), 7139-7150 (_ds_cache builds the DB filename from cache_namespace only), 9053-9061 (compare: outer_rep_builders_id IS folded into the namespace)

## scenario as reported
Run 1: a nested-CV search on flab_thermo_tresanco with rep_factories={'esm_ft': ...}, rep_factories_id='esm_ft_v1', tier='general'. Each inner fold builds a label-fitted embedding, so the pool is 11 representations and the winning budget split includes esm_ft. Rows land under cache_namespace 'ocv5f0s42'. Run 2: the fine-tuning code changes and the caller bumps rep_factories_id='esm_ft_v2' -- which is the documented way to invalidate it. _ds_cache is unchanged ('flab_thermo_tresanco__ocv5f0s42'), so the DB file is the same, and rep_factories_id is in no identity column, so run 2 reads run 1's rows and reports v1's numbers as v2's. The same collision happens between a run with rep_factories and a run with none at all: the second pool is one representation smaller and every allocation splits the budget differently, but both key as rep_subset='all'. The ft_embeddings layer is keyed correctly (7625-7630 composes rep_factories_id into each factory id) -- it is only the search rows that are not.

## evidence as reported
find_best_hybrid.py:7377-7383 raises "find_best_hybrid(cache=True) with rep_factories requires rep_factories_id=... A stable per-version identifier is the caller's responsibility". The value goes into _make_search_meta at 7476 ('rep_factories_id', line 647) -- which is only checkpoint_meta (7737) -- and into _factory_ids at 7625. It appears nowhere in result_db_identity (7485-7505) and nowhere in _ds_cache (7147-7150). nested_cv_evaluate at 9057-9058 does fold outer_rep_builders_id into the cache namespace (`_ns = f"{_ns}__orb_{outer_rep_builders_id}"`) for exactly this reason, and 9086-9091 composes rep_factories_id into rep_factories_id itself rather than into _ns.

## fix as proposed
Fold rep_factories_id into the cache namespace at 9057 alongside outer_rep_builders_id, and in _find_best_hybrid_inner append it to _ds_cache at 7150 when rep_factories is set, so the DB file separates. That costs a re-run for any factory-based search already on disk but changes nothing for the 12 datasets that pass no factories.
