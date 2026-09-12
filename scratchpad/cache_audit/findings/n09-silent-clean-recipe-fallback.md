# n09-silent-clean-recipe-fallback

**title:** n09 falls back to the clean-label blueprint when its per-fold noisy search finds nothing, and the output cannot tell the two apart

**area:** tests/experiments/noise/ (n01–n15, noise_helpers.py, _similarity_hybrid_space.py, the _smoke_* files) plus src/kirby/noise_spec.py and src/kirby/utils/noise_estimation.py

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/noise/n09_lnl_survey.py:337, 305-340, 174-190, 228

## scenario as reported
n09 fetches `bp` from the CLEAN pool at line 174 (get_best_search_for_downstream with no label_noise, served straight out of data['precomputed_searches']). Its stated purpose is then to discard that recipe: at line 305 it runs a fresh find_best_hybrid on the fold's noisy labels with cache=False, picks the best entry across NOISY_SEARCH_BUDGETS, and uses it. Run 1: the inner search returns an 'rf' entry, the fold's hybrid is built on the noisy recipe. Run 2 (or another fold, or a sigma where every inner score comes back NaN, or a dataset where best_by_model_budget has no 'rf' key): `_best_bp` stays None and line 337 substitutes `bp` — the clean-searched, oracle-biased recipe the experiment exists to avoid. Nothing is recorded: the results dict stores only 'budget': bp.budget (line 228), which is the clean recipe's budget in both cases. Two runs whose hybrid arm was built from two different recipes produce output files that are indistinguishable.

## evidence as reported
n09_lnl_survey.py:337 'inner_bp = _best_bp if _best_bp is not None else bp'; the comment at :299-304 'The pool blueprint `bp` above was fit on clean labels, which would give the LNL methods a feature mix chosen with oracle access to ground truth. Refitting per (noise level, fold) on noisy labels removes that leak.'; :228 "'budget': int(bp.budget) if bp is not None else None" — the only recipe field written out, taken from the clean blueprint; grep for inner_bp shows it is used at :340 and never recorded.

## fix as proposed
Record per (sigma, fold) which recipe was used — the inner blueprint's signature and a flag saying whether it came from the noisy inner search or from the clean pool fallback — and put a count of fallbacks in the summary. Then decide whether a fold whose noisy search produced nothing should use the clean recipe at all or be dropped from both arms; today it silently rejoins the comparison carrying the leak.
