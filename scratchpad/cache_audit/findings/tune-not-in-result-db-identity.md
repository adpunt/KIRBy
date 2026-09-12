# tune-not-in-result-db-identity

**title:** tune / tune_n_iter / tune_knn_k are keyed everywhere except the one layer that now stores the search results

**area:** The search cache key (_make_search_meta / _search_cache_path / _search_checkpoint_path) and the result_db row identity (IDENTITY_COLS + CONFIG_COLS)

**class:** A-wrong-answer

**severity as reported:** large

**location:** src/kirby/find_best_hybrid.py:5080 (row['tune'] written), src/kirby/result_cache.py:255-275 (upsert_rows never reads it), src/kirby/find_best_hybrid.py:7485-7505 (identity), src/kirby/find_best_hybrid.py:6157-6160 (config skipped on a stored row)

## scenario as reported
Run 1: precompute on herg_ki, tier='general', tune=False. It stores a row for (all, performance_weighted, 250, lightgbm_gain, knn, match). Run 2: the same dataset, tier, folds, seed, metric, task_type and objective with tune=True, tune_knn_k=True passed through run_nested_cv's **search_kwargs. The baselines and the hybrid_evals both miss correctly (their filenames carry a __tuned20 suffix -- _baseline_cache_path line 208, _hybrid_eval_cache_path line 798), but the result_db identity has no tune column and tier is not marked either, so fetch_rows returns run 1's row, line 6159 does `continue`, and no tuned model is ever fitted. The tuned arm reports the untuned score. The 2026-09-03 measurement that tuning kNN moves 6 of 8 datasets and picks a different budget on 4 of them is exactly the size of the error this hides.

## evidence as reported
find_best_hybrid.py:5080 `row['tune'] = result.get('tune', False)` -- but result_cache.upsert_rows (lines 255-275) builds its value list from KEY_COLS, DATA_COLS and nine named result fields, and 'tune' is in none of ALL_COLS, so the value is silently discarded. find_best_hybrid.py:419-421 states the intent: "`tune` and `tune_knn_k` are not in the FILENAME either, so both arms of a tuned-vs-untuned comparison land on one path -- the untuned arm silently read the tuned arm's answer." That was fixed for the .pkl layers via _OPT_IN_META_KEYS (line 312-328) and the two filename suffixes, and never for the result database.

## fix as proposed
Either add 'tune' (canonically '', 'tuned20', or 'tuned20_k') to IDENTITY_COLS, or ride it on tier the way label_noise and condition_suffix already do (find_best_hybrid.py:7315-7347) -- `if tier and tune: tier = f'{tier}__tuned{int(tune_n_iter)}' + ('_k' if tune_knn_k else '')`. The tier route needs no schema change and leaves every untuned row byte-identical. Delete the dead row['tune'] line at 5080 either way.
