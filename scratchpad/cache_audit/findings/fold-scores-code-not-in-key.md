# fold-scores-code-not-in-key

**title:** Nothing about the code that computes importance or allocation scores is in the fold_scores key, so a change to either function serves the old numbers forever

**area:** baselines/, folds/, fold_scores/ and ft_embeddings/ cache layers (src/kirby/find_best_hybrid.py)

**class:** C-stale-undetected

**severity as reported:** large

**location:** src/kirby/find_best_hybrid.py:216-238 (_fold_score_cache_path), :5392-5408 (importance write/read), :5475-5542 (alloc write/read); src/kirby/hybrid.py:3846 (compute_feature_importance), :2241 (_evaluate_allocation)

## scenario as reported
The key is dataset, kind, fold index, method name, an input hash over the fold's representation matrices and labels, and the noise id. Change compute_feature_importance - the treeshap pre-screen default `max_features_shap = kwargs.get('max_features_shap', 2000)` at hybrid.py:3951, or the internal forest's n_estimators=50/max_depth=10 at :3906-3912 - and every fold_scores/<ds>/importance__f*__treeshap__<hash>.pkl on disk still matches, because the reps and the labels did not move. The next run reads the pre-change scores, selects features with them, and reports the result as the new code's. Same for _evaluate_allocation: the fitted estimators are hardcoded (n_estimators=50, max_depth=4, random_state=42 at hybrid.py:2424-2428) and _SCORER_FIT_MAX_SAMPLES caps the svm/mlp/gp fits at hybrid.py:2232; changing any of them leaves every alloc__f*__<ev>__<hash>.pkl matching. There is no version field, no code hash, and the repo's rule is 'code change -> wipe + re-run' - but the wipe tool does not reach this directory (see purge-missing-fold-score-dirs).

## evidence as reported
_fold_score_cache_path's docstring says input_hash 'covers every array the score is computed from', which is true and is the whole guard - the arrays are the only thing checked. The importance meta is `{'kind': 'fold_importance', 'method': imp_method, 'fold_idx': fold_idx, 'input_hash': _imp_in_hash}` (line 5396-5397); the alloc meta adds only 'evaluator' and 'metrics' (line 5477-5479). The alloc meta's 'metrics' key does catch one code change - adding a metric to BUILTIN_METRICS - which shows the layer has no other version signal.

## fix as proposed
Owner decision, since in-cache versioning is a locked no. Either write the fold_scores directory into the documented wipe procedure and into purge_rep_cache.py so a change to compute_feature_importance or _evaluate_allocation is followed by deleting it, or state in the module docstring of both functions that any edit requires deleting $KIRBY_CACHE_DIR/fold_scores. Nothing on disk today can be attributed to a code state.
