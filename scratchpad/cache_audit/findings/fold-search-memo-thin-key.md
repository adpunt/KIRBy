# fold-search-memo-thin-key

**title:** The in-process fold-search memo does not key on metric, task_type or the training rows themselves

**area:** Experiment-side caches in tests/experiments/_shared.py (create_hybrid_cached / _hybrid_fold_cache_key, rep_cache, the search-reuse helpers, and the model/evaluator lists that ride in the search cache key)

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/_shared.py:1020-1037 (memo_key), 1038-1050 (the search_on_fold call it stands in for), 884-886 (_FOLD_SEARCH_MEMO)

## scenario as reported
memo_key holds dataset_name, tier, cache_namespace, fold_i, n_outer_folds, seed, budgets, internal_evaluators, models, objective, label_noise, rep names, aug names and int(len(train_idx)). It does not hold metric, task_type, domain, adaptive_pruning_schedule, model_pruning_aggressiveness, or the CONTENT of train_idx and labels. Concrete: a caller loops a dataset under two metrics — call 1 get_fold_best_search(..., fold_i=0, n_outer_folds=5, metric='rmse'), call 2 the same with metric='spearman'. Every field in memo_key is identical, so call 2 never reaches search_on_fold and is handed the search whose winner was picked by RMSE; the on-disk layer would have caught this (metric_name is in _make_search_meta) but the memo sits in front of it. The row-content gap is the sharper one: with an explicit cache_namespace the docstring at :917-920 invites a caller to hand in a training subset that is NOT a nested-CV fold, and two different subsets of the same length under one namespace label collide on the memo even though their labels_hash would have separated them on disk. s11_bayesian_optimization.py:906 is the caller on that path; it happens to be safe today because its namespace names the seed (f'bo_seed{seed_i}_s{seed}') and the seed rows are a deterministic function of that seed.

## evidence as reported
tests/experiments/_shared.py:1020-1034
    memo_key = (
        dataset_name, tier, cache_namespace,
        int(fold_i), int(n_outer_folds), int(seed),
        tuple(budgets) ..., tuple(internal_evaluators) ..., tuple(models) ...,
        objective, str(label_noise) ...,
        tuple(sorted(reps.keys())),
        tuple(sorted(augmentations.keys())) if augmentations else None,
        int(len(train_idx)),
    )
search_on_fold at :936-938 resolves metric via _resolve_search_metric and passes it into find_best_hybrid, and _make_search_meta records it as 'metric_name' — so the disk layer separates what the memo does not.

## fix as proposed
Put the same identity in the memo that the disk layer uses: add the resolved metric and task_type, and replace int(len(train_idx)) with a hash of the sorted train_idx (and of labels[train_idx]). Cheap, and it makes the memo strictly narrower than the disk key rather than wider. In-process only, so nothing on disk changes.
