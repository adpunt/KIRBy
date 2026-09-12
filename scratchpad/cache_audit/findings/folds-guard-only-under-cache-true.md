# folds-guard-only-under-cache-true

**title:** The cv_splitter_id requirement is inside `if cache:`, but the folds, baselines and fold_scores layers cache on dataset_name alone

**area:** baselines/, folds/, fold_scores/ and ft_embeddings/ cache layers (src/kirby/find_best_hybrid.py)

**class:** A-wrong-answer

**severity as reported:** small

**location:** src/kirby/find_best_hybrid.py:7366-7387 (the guards, inside `if cache:`), :7147-7150 (_ds_cache set unconditionally), :1034-1047 (folds cache), :4730-4736 (_baseline_cache_enabled), :5352 (_fold_score_cache_on)

## scenario as reported
Run 1: find_best_hybrid(dataset_name='flab_thermo_tresanco', cache=False, rep_factories={...}, rep_factories_id='v1', cv_splitter=<a custom 5-fold splitter>) and no cv_splitter_id. cache=False means the block at line 7366 never runs, so nothing raises. _precompute_fold_reps still receives dataset_name=_ds_cache, so _load_or_compute_folds writes folds/flab_thermo_tresanco__5f_42_grouped.pkl with cv_splitter_id=None in its meta. Run 2: the same call with no cv_splitter, i.e. the default GroupKFold(5). The filename matches and every meta key matches (cv_splitter_id is None on both sides, task_type and labels_hash are the same), so run 2 is handed run 1's splits and every fine-tuned representation is fitted on run 1's train rows. The baselines layer has the same exposure in the same run, and there it is not even recorded in the meta.

## evidence as reported
Line 7366 `if cache:` opens the block that contains, at 7374, `if cv_splitter is not None and not cv_splitter_id: raise ValueError(...)` and at 7381 the matching rep_factories_id check. Line 7148 `_ds_cache = dataset_name` sits above that block and is unconditional. Line 1044 `cached = _load_cache_with_meta(path, meta, kind='search', verbose=False)` runs whenever dataset_name is truthy. Line 4732-4736: `_baseline_cache_enabled = (use_baseline_cache and dataset_name is not None and metric_name is not None)` - no reference to `cache`. Line 5352: `_fold_score_cache_on = bool(dataset_name)`.

## fix as proposed
Move the cv_splitter/cv_splitter_id and rep_factories/rep_factories_id checks above the `if cache:` block, so they fire whenever dataset_name is set - which is what actually turns the four filename caches on.
