# baseline-model-name-only

**title:** A caller-supplied estimator object is keyed by its name alone, and a single instance is always named 'custom'

**area:** baselines/, folds/, fold_scores/ and ft_embeddings/ cache layers (src/kirby/find_best_hybrid.py)

**class:** A-wrong-answer

**severity as reported:** small

**location:** src/kirby/find_best_hybrid.py:2664-2670 (_resolve_models passthrough), :4750-4761 (baseline path built from model_name), :645-688 (meta records model_name only), :7366-7387 (guards for cv_splitter and rep_factories, none for models)

## scenario as reported
Run 1: find_best_hybrid(dataset_name='logd', models=SVR(C=1.0), metric='rmse'). _resolve_models wraps it as {'custom': SVR(C=1.0)} and the baseline for each representation is written to baselines/logd__ecfp4__custom__rmse__5f_42.pkl. Run 2: find_best_hybrid(dataset_name='logd', models=SVR(C=100.0), metric='rmse'). Same name, same filename, same meta - run 2 is served the C=1.0 scores. The dict form has the same hole: models={'rf': RandomForestRegressor(n_estimators=1000)} collides with models=['rf'], whose registry default is a different forest.

## evidence as reported
Line 2664-2666: `if isinstance(models, dict): return {name: deepcopy(m) for name, m in models.items()}`; line 2668-2670: `if hasattr(models, 'fit') ...: return {'custom': deepcopy(models)}`. _baseline_cache_path takes model_name: str and interpolates it directly. _make_baseline_meta stores 'model_name': model_name and nothing about the estimator. The same file already refuses two other un-hashable caller objects for exactly this reason: line 7374 for cv_splitter, line 7381 for rep_factories.

## fix as proposed
Refuse a dict or bare-instance `models` when dataset_name is set, the way cv_splitter and rep_factories are refused, or require a caller-supplied models_id and put it in the baseline filename.
