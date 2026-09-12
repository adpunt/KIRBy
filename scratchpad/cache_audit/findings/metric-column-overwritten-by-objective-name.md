# metric-column-overwritten-by-objective-name

**title:** Under a blend or predictive_corr the identity's metric column holds the objective's name, so the dataset metric drops out of the key entirely

**area:** The search cache key (_make_search_meta / _search_cache_path / _search_checkpoint_path) and the result_db row identity (IDENTITY_COLS + CONFIG_COLS)

**class:** A-wrong-answer

**severity as reported:** large

**location:** src/kirby/find_best_hybrid.py:7277-7286 (_metric_name_for_meta overwritten), 7503 ('metric': _esm.get('metric_name')), 5761 (_rc_ranking_metric holds the real metric as unchecked data), 7492 (objective already has its own identity column)

## scenario as reported
Run 1: buchwald_hartwig, objective={'predictive': 0.5, 'similarity': 0.5}, metric='rmse', tier='general'. _metric_name_for_meta becomes 'blend_predictive0.5_similarity0.5' at line 7286, so the row's metric column says the blend name and its ranking_metric column says 'rmse'. Run 2: the same dataset and blend with metric='spearman'. objective, tier, labels_hash, groups_hash, n_folds, seed, cv_splitter_id, metric and task_type all match, so fetch_rows returns run 1's rows. Each stored row's metrics['predictive'] is an RMSE and is served to a run that asked for spearman; ranking_metric records the difference and is read by nothing. I am not certain the collision reaches score_mean as well: _is_blend_objective is true for both a dict and 'predictive_corr' (line 1600-1606), which makes _skip_alloc_precompute true (5463-5464) so the metric-selected allocation scores at 4105 are bypassed -- but the metric still fits and scores the model, and the baselines it produces still feed _predict_top_k_subset's fallback (line 862-871). That uncertainty is why this is large.

## evidence as reported
find_best_hybrid.py:7284-7286: "What the run SELECTS on is the blend, so the meta has to say so -- otherwise a blend and a plain predictive run on the same dataset record the same metric_name and a cached search for one satisfies the meta check for the other", then `_metric_name_for_meta = _objective_metric_name(objective)`. That reasoning predates 2026-09-10, when 'objective' became IDENTITY_COLS[8] (result_cache.py:66) and 7492 started writing it into the identity on its own. The overwrite is now redundant for separating a blend from a plain run, and it costs the real metric its column. _run_search still receives the real metric (7746 `metric_name=_metric_name_str`, from 7686) and stores it as ranking_metric (5761), which fetch_rows returns but no code compares.

## fix as proposed
Drop the overwrite at 7286 so 'metric' keeps the dataset metric (or 'custom'), and let the 'objective' identity column carry the blend name -- it already does. Existing blend rows have to be deleted; no plain-predictive row moves, since 7286 only fires when _blend_needs_model is true.
