# distance-precompute-drops-models

**title:** precompute_hybrid_searches overwrites its own `models` argument with None, so the distance searches never score knn

**area:** Experiment-side caches in tests/experiments/_shared.py (create_hybrid_cached / _hybrid_fold_cache_key, rep_cache, the search-reuse helpers, and the model/evaluator lists that ride in the search cache key) + tests/experiments/similarity/ (s01–s13, 

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/_shared.py:4437 (models = None), 4608 (models=DISTANCE_MODELS passed in), 4963 (read side asks for DISTANCE_MODELS), 5048 (write-back into the aug_condition-keyed dict); tests/experiments/similarity/similarity_helpers.py:1009; tests/precompute_hybrids.py:557

## scenario as reported
populate_precompute_dicts calls precompute_hybrid_searches(..., tier='general_distance', models=DISTANCE_MODELS). Line 4437 discards it: `models = None`. find_best_hybrid then resolves models=None to the tier fast+full set, which does not contain knn (DEFAULT_MODELS['knn']['tier'] == 'extra', AUTO_SELECTED_TIERS == {'fast','full'}), so the stored SearchResult has no knn entry at all. That object is put into data['precomputed_searches_distance'] under the key 'none'. Then s05 (or s04/s07/s13) calls get_best_search_for_downstream(..., distance=True, data=data, downstream_model='knn'). Line 4963 sets extra['models']=DISTANCE_MODELS, but the in-memory store is keyed by aug_condition alone and get_precomputed_search returns precomputed.get('none') before it looks at any kwarg — so the caller is handed a search run under a different model set. downstream_model='knn' is then absent from best_by_model, the candidate is skipped and the helper returns (None, None): the kNN arm of the distance study silently produces nothing. On the disk path the same mismatch is a pure miss — the SLURM precompute (precompute_hybrids.py:557) DOES pass DISTANCE_MODELS, so its file has knn in meta['model_names'], the in-process precompute rejects it, recomputes without knn and overwrites the same searches/<ds>__general_distance__none.pkl.

ALSO REPORTED INDEPENDENTLY: similarity_helpers.run_all precomputes the distance lineage with `models=DISTANCE_MODELS` (similarity_helpers.py:1009). precompute_hybrid_searches accepts that argument and then overwrites it unconditionally: `models = None` (_shared.py:4437). find_best_hybrid therefore resolves the model set from DEFAULT_MODELS' AUTO_SELECTED_TIERS = {'fast','full'} (find_best_hybrid.py:2647), which excludes knn (tier 'extra', find_best_hybrid.py:2403), and stamps model_names=('dnn','lightgbm','rf','svm','xgboost') into the search meta. Two consequences, both live. (1) s06 calls get_fitted_pool_hybrid(downstream_model='knn', distance=True) (s06_applicability_domain.py:144-149); get_best_search_for_downstream reads the in-memory precompute, finds 'knn' absent from search.best_by_model_budget and search.best_by_model, and `continue`s (_shared.py:5057-5060); no pool candidate survives, so it prints "[pool] WARNING: no usable hybrid for model=knn" and returns (None, None) (5069-5077), and get_fitted_pool_hybrid raises RuntimeError (5148-5151), which s06/s08/s02 catch and record as a failed search. The distance experiments lose their hybrid arm. (2) Whenever that in-memory dict is empty — the distance precompute raised (similarity_helpers.py:1014-1023) or an s-script is run outside run_all — get_best_search_for_downstream falls to disk with `extra.setdefault('models', DISTANCE_MODELS)` (_shared.py:4963), which normalises to ('dnn','knn','lightgbm','rf','svm','xgboost') at find_best_hybrid.py:7176-7181. That is meta drift on 'model_names' against the file the precompute wrote, so searches/<ds>__general_distance__none.pkl is a stale miss, the whole search is recomputed, and the file is overwritten with the knn meta. The next run_all precompute (models=None) then misses it the other way and overwrites it back. Neither side ever hits.

## evidence as reported
tests/experiments/_shared.py:4437  models = None      # unconditional, after the parameter is accepted at 4378
tests/experiments/_shared.py:4608  models=DISTANCE_MODELS,
tests/experiments/_shared.py:403   DISTANCE_MODELS = DOWNSTREAM_MODELS + ['knn']
tests/experiments/_shared.py:4963  extra.setdefault('models', DISTANCE_MODELS)
tests/experiments/_shared.py:5048  precomputed[aug_cond] = search    # writes back under aug_cond alone
src/kirby/find_best_hybrid.py:7167-7174  models is None -> sorted(names with spec['tier'] in AUTO_SELECTED_TIERS); resolved set is ['dnn','lightgbm','rf','svm','xgboost'].
_make_search_meta stores 'model_names', and _search_cache_path does not, so both versions share one filename.

## fix as proposed
Delete line 4437 and honour the `models` argument (keep `if models is None: models = None` semantics only as the documented default). Separately, the in-memory precomputed store needs to stop being keyed by aug_condition alone: either key it by (tier, aug_condition, models, budgets, internal_evaluators) or refuse the fast path whenever the caller passed any of those, the way it already refuses a geometry objective. Distance searches on disk must be deleted after the fix.
