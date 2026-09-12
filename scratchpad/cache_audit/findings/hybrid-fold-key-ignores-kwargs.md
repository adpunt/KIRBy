# hybrid-fold-key-ignores-kwargs

**title:** create_hybrid_cached forwards **kwargs to create_hybrid and hashes none of them

**area:** Experiment-side caches in tests/experiments/_shared.py (create_hybrid_cached / _hybrid_fold_cache_key, rep_cache, the search-reuse helpers, and the model/evaluator lists that ride in the search cache key)

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/_shared.py:71-101 (_hybrid_fold_cache_key), 104-149 (create_hybrid_cached and its key call), 165-174 (the **kwargs forward); src/kirby/hybrid.py:4787-4837 (create_hybrid signature)

## scenario as reported
create_hybrid_cached passes exactly nine things into the key: train_idx, dataset_name, labels, groups, rep shapes, allocation_method, importance_method, budget, internal_evaluator, plus step_size/patience/greedy_tolerance/augmentation_strategy/aug names pulled from kwargs. Everything else in create_hybrid's signature arrives through **kwargs at line 173 and never touches the hash: objective, is_clf, row_index, structural_pairs, candidate_ranking, augmentation_threshold, augmentation_budget, aug_per_source_budget, aug_max_candidates, n_per_rep, validation_split, apply_filters, model, metric_fn, lower_is_better, precomputed_importance, precomputed_alloc_scores. Concrete collision: run 1 calls create_hybrid_cached(train_reps, y_train, train_idx, dataset_name='logd', allocation_method='performance_weighted', budget=100) with no objective, so the budget is split on downstream model performance and the selected indices are written to hybrid_fold_cache/<key>.pkl. Run 2 calls it with the identical positional arguments plus objective='similarity', which makes the allocators maximise held-out neighbour geometry instead. Both build the same 24-hex key, run 2 hits the file and is handed the predictive selection, and its result is reported as a geometry-selected hybrid. The same story holds for candidate_ranking='rf_importance' vs 'residual_correlation', for n_per_rep set vs None (total width n_per_rep*n_reps vs budget), and for a model+metric_fn pair vs none (accept/reject scored by the real metric vs the internal evaluator's r2). Today the only in-scope caller is c09_formal_analysis.py:366 and :543, which passes none of these, so nothing on disk is currently wrong — the defect is that the door accepts them silently.

## evidence as reported
tests/experiments/_shared.py:143-150
    key = _hybrid_fold_cache_key(
        train_idx, rep_shapes, allocation_method, importance_method,
        budget, internal_evaluator, aug_names, augmentation_strategy,
        step_size=kwargs.get('step_size', 5),
        patience=kwargs.get('patience', 20),
        greedy_tolerance=kwargs.get('greedy_tolerance', 0.0),
        dataset_name=dataset_name, labels=y_train, groups=groups,
    )
tests/experiments/_shared.py:165-174  create_hybrid(..., **kwargs)
src/kirby/hybrid.py:4787-4837 lists allocation_method, n_per_rep, budget, step_size, patience, validation_split, apply_filters, importance_method, augmentations, augmentation_strategy, augmentation_threshold, augmentation_budget, candidate_ranking, model, metric_fn, lower_is_better, internal_evaluator, precomputed_importance, precomputed_alloc_scores, groups, objective, is_clf, row_index, structural_pairs, **kwargs.

## fix as proposed
Make the key cover the whole surface rather than a hand-picked subset: hash a canonicalised form of every remaining kwarg (sorted key/value pairs, with the two precomputed_* arrays reduced to their content hash or excluded by an explicit documented rule), or refuse a kwarg the key does not know about. A whitelist that silently ignores the rest is the shape that produced this. Existing hybrid_fold_cache/*.pkl can stay: no current caller passes any of the unhashed parameters.
