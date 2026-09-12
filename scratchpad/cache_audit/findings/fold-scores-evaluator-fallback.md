# fold-scores-evaluator-fallback

**title:** The internal evaluator silently falls back to another library on ImportError, and the stored fold_scores file is named for the evaluator that was requested

**area:** baselines/, folds/, fold_scores/ and ft_embeddings/ cache layers (src/kirby/find_best_hybrid.py)

**class:** A-wrong-answer

**severity as reported:** small

**location:** src/kirby/hybrid.py:2416-2422 (lightgbm -> xgboost), :2395-2401 (xgboost -> rf); src/kirby/find_best_hybrid.py:5475-5479 (path + meta named by the requested evaluator), :5540-5542 (write)

## scenario as reported
Run 1 is a precompute on a node where lightgbm cannot be imported. _evaluate_allocation is asked for internal_evaluator='lightgbm', catches the ImportError and recurses with internal_evaluator='xgboost' (and if that is also absent, 'rf'). The per-representation scores it returns are an XGBoost fit, but they are written to fold_scores/<ds>/alloc__f0__lightgbm__<input_hash>.pkl with meta {'kind': 'fold_alloc', 'evaluator': 'lightgbm', ...}. Run 2 on a machine where lightgbm imports fine asks for the same evaluator on the same fold and the same rows, matches the filename and every meta key, and splits its budget by XGBoost scores while the run is recorded as using the lightgbm evaluator. The evaluator axis is one of the axes the framework deliberately keeps, so the two are meant to be distinguishable.

## evidence as reported
hybrid.py:2416-2422: `elif internal_evaluator == 'lightgbm': try: from lightgbm import LGBMRegressor, LGBMClassifier / except ImportError: return _evaluate_allocation(..., internal_evaluator='xgboost', return_predictions=return_predictions)`. The fallback returns before anything records what it actually ran. find_best_hybrid.py:5475: `_ap = _fold_score_cache_path(dataset_name, 'alloc', fold_idx, ev, _alloc_in_hash, noise_id)` and 5477: `_am = {'kind': 'fold_alloc', 'evaluator': ev, 'fold_idx': fold_idx, 'input_hash': _alloc_in_hash, 'metrics': tuple(sorted(_alloc_metrics))}` - `ev` is the requested name throughout.

## fix as proposed
Have _evaluate_allocation return the evaluator it actually fitted (or raise instead of substituting when the caller named one explicitly), and put that returned name in the fold_scores meta so a substituted file is a miss for a run that can import the real library.
