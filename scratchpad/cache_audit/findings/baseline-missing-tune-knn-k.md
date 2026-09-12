# baseline-missing-tune-knn-k

**title:** tune_knn_k changes the kNN baseline but appears in neither the baseline filename nor its meta

**area:** baselines/, folds/, fold_scores/ and ft_embeddings/ cache layers (src/kirby/find_best_hybrid.py)

**class:** A-wrong-answer

**severity as reported:** large

**location:** src/kirby/find_best_hybrid.py:7612-7618 (marks the estimator), :3719-3733 (_set_knn_k), :3830-3840 (_evaluate_single tune + _set_knn_k), :176-212 and :645-688 (baseline path + meta)

## scenario as reported
Run 1: find_best_hybrid(dataset_name='logd', tune=True, tune_knn_k=True). Line 7615 sets KNN_TUNED_K_ATTR on the resolved knn estimator, so _set_knn_k keeps the k the per-fold search chose. The knn baseline lands at baselines/logd__ecfp4__knn__rmse__5f_42__tuned20.pkl. Run 2: the same call with tune_knn_k=False. The estimator is not marked, so _set_knn_k overwrites n_neighbors with min(5, sqrt(n_train)) - a different estimator, a different number. The filename is identical (the `tn` suffix is built from tune and tune_n_iter only) and the expected meta is identical, so run 2 is served run 1's tuned-k baseline. These two runs are exactly the two arms of the kNN-tuning comparison already on record (tuning kNN moved the winning budget on 4 of 8 datasets and the allocation method on 5), so the arm that is supposed to show k=5 reports the tuned-k number instead.

## evidence as reported
Line 7613-7616: `if tune_knn_k: for _mdl in resolved_models.values(): if hasattr(_mdl, 'n_neighbors'): setattr(_mdl, KNN_TUNED_K_ATTR, True)` - and `resolved_models` is the same dict handed to _evaluate_baselines at line 7653. Line 3722-3729: `if getattr(m, KNN_TUNED_K_ATTR, False): k = int(getattr(m, 'n_neighbors', 5) or 5) else: k = min(5, max(1, int(np.sqrt(n_train))))`. _baseline_cache_path line 211: `tn = f'__tuned{int(tune_n_iter)}' if tune else ''`. _make_baseline_meta has no tune key at all. The search layer treats this as identity - 'tune_knn_k' is in _OPT_IN_META_KEYS at line 330 with the comment at line 422 saying tune and tune_knn_k are 'not in the FILENAME either, so both arms of a tuned-vs-untuned comparison land on one path'. The baseline layer got the filename half of that fix (the `tn` suffix) but not the tune_knn_k half.

## fix as proposed
Pass tune_knn_k into _evaluate_baselines and either extend the filename suffix (e.g. `__tuned{n}k` when tune_knn_k) or write 'tune_knn_k': True into the baseline meta, which _load_cache_with_meta's second pass already rejects when stored-but-unrequested. Delete existing baselines/*__tuned*.pkl for knn - they cannot be attributed to a k policy.
