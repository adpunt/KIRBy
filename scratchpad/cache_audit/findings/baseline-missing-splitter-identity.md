# baseline-missing-splitter-identity

**title:** The baseline value depends on the fold splitter, but neither cv_splitter_id nor task_type is in the baseline filename or meta

**area:** baselines/, folds/, fold_scores/ and ft_embeddings/ cache layers (src/kirby/find_best_hybrid.py)

**class:** A-wrong-answer

**severity as reported:** small

**location:** src/kirby/find_best_hybrid.py:176-212 (_baseline_cache_path), :645-688 (_make_baseline_meta), :3778-3780 (_evaluate_single builds its own splitter), :4826-4831 (the geometry branch builds its own splitter)

## scenario as reported
Run 1: find_best_hybrid(dataset_name='herg_ki', cache=True, cv_splitter=GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42), cv_splitter_id='gss20', tier='general'). _evaluate_single passes that splitter through _make_cv_splitter, which returns it unchanged, and every per-representation baseline is a mean over those five draws. It is written to baselines/herg_ki__ecfp4__rf__rmse__5f_42.pkl. Run 2: the same call with no cv_splitter, so the folds are GroupKFold(5) - genuinely different train/val rows. The search pickle and the result_db rows separate correctly (cv_splitter_id is in _make_search_meta line 638 and in result_db_identity line 7486), but the baselines filename and every meta key are identical, so run 2's 'best single representation' comparator and its top_k allocation ranking come from run 1's splits. task_type has the same hole in the same key: it selects StratifiedGroupKFold over GroupKFold (line 3313-3316) and, under a geometry objective, it also switches the neighbour score between the classification and regression forms (line 4827 `_is_clf_b = (task_type == 'classification')`).

## evidence as reported
_evaluate_single line 3778: `splitter, use_groups = _make_cv_splitter(groups, n_folds, seed, cv_splitter=cv_splitter, task_type=task_type)` - the baseline never uses precomputed_folds for frozen representations, it re-splits. _make_folds_meta (line 690-728) records BOTH cv_splitter_id and task_type, with the comment 'task_type picks the splitter ... Without both in the key, the same dataset/n_folds/seed/groups would serve cached splits computed for a different label vector or a different splitter'. _make_baseline_meta records neither.

## fix as proposed
Add cv_splitter_id and task_type to _make_baseline_meta, both emitted only when non-None so files written without them still match, and add 'cv_splitter_id' to _OPT_IN_META_KEYS so a stored non-default splitter id is rejected for a caller that asked for none.
