# internal-evaluators-list-disagrees

**title:** The evaluator list experiments pass and the one precompute writes are different lists, on the same cache file

**area:** Experiment-side caches in tests/experiments/_shared.py (create_hybrid_cached / _hybrid_fold_cache_key, rep_cache, the search-reuse helpers, and the model/evaluator lists that ride in the search cache key)

**class:** B-never-hits

**severity as reported:** large

**location:** tests/experiments/_shared.py:580; src/kirby/find_best_hybrid.py:3079, 7136-7138, 7443 (internal_evaluators in the meta), 153-161 (filename); tests/precompute_hybrids.py:44, 564-565, 648, 746; tests/experiments/complexity/c02_complexity_grid.py:96

## scenario as reported
Run 1 is the precompute: tests/precompute_hybrids.py is launched without --evaluators, so internal_evaluators stays None and find_best_hybrid falls back to DEFAULT_INTERNAL_EVALUATORS. It writes searches/<ds>__general__none.pkl with meta['internal_evaluators'] = ('lightgbm','match','mlp','rf','svm'). Run 2 is c02 on the same dataset: it passes internal_evaluators=INTERNAL_EVALUATORS, so the expected meta is ('dnn','lightgbm','match','rf'). _load_cache_with_meta rejects on that key, c02 recomputes the whole search from cold and overwrites the same filename with its own four-evaluator version. The next precompute pass rejects that one and overwrites it back. Nothing is ever served; the file ping-pongs. The same call shape is in c05:236, c06:202, c07:280, c11:468, c14:267, c24:187+336, x_ab_ness:233 and reaction_rep_comparison:413.

## evidence as reported
tests/experiments/_shared.py:580  INTERNAL_EVALUATORS = ['match', 'rf', 'lightgbm', 'dnn']
src/kirby/find_best_hybrid.py:3079  DEFAULT_INTERNAL_EVALUATORS = ['match', 'rf', 'svm', 'lightgbm', 'mlp']
src/kirby/find_best_hybrid.py:7136-7138  _norm_eval = (internal_evaluators if internal_evaluators is not None else (FAST_INTERNAL_EVALUATORS if speed == 'fast' else DEFAULT_INTERNAL_EVALUATORS))
_make_search_meta writes 'internal_evaluators': tuple(sorted(internal_evaluators)) and _search_cache_path builds the filename from dataset/tier/aug_strategy only, so the two lists share one file.
tests/precompute_hybrids.py:44 imports INTERNAL_EVALUATORS and it appears nowhere else in that file (grep -c = 1); --evaluators defaults to None at line 648.

## fix as proposed
One list must be canonical, and that is the owner's call because the two lists score different things (svm+mlp vs dnn). Whichever wins, set the other to it and pass it explicitly on both sides: precompute_hybrids.py should pass internal_evaluators=INTERNAL_EVALUATORS (the import at line 44 is already there and unused), and precompute_hybrid_searches/precompute_nested_cv_searches should forward the same list instead of leaving it None. Every existing searches/*.pkl was written under one of the two and must be deleted.
