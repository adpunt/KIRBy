# fed05-federated-fs-passes-a-model-with-no-metric

**title:** fed05's noise condition hands federated_feature_selection a downstream model and no metric, and the raise FRAMEWORK_FINALIZATION fix 11 credits cannot fire on that path

**area:** tests/experiments/federated/ (fed01-fed05 + federated_helpers.py + lhasa_herg_loader.py), and the cache layers they reach in src/kirby/find_best_hybrid.py, src/kirby/hybrid.py, src/kirby/result_cache.py, tests/experiments/_shared.py

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/federated/fed05_antibody_federated.py:531-539; tests/experiments/federated/federated_helpers.py:1231-1239; src/kirby/hybrid.py:2347-2361 (the raise), src/kirby/hybrid.py:1380-1388 and 2629-2675 (the path actually taken), src/kirby/hybrid.py:5130 (augmentation gate)

## scenario as reported
Answer to question 1. Eight federated sites hand a model to create_hybrid: fed02:418, fed02:483, fed03:469, fed03:758, fed03:1104, fed03:1254, federated_helpers:1231 (inside federated_feature_selection) and federated_helpers:2019 (inside fluid_within_kirby_across). Seven now pass a metric: fed02 and fed03 pass metric_fn at the call, and every caller of the two helper sites passes metric_fn (fed01:479, fed01:616, fed01:831, fed03:397, fed02:536) except one. fed05:531-539 calls federated_feature_selection with downstream_model=fed_to_kirby_model(fed_model) and NO metric_fn, so metric_fn stays None at federated_helpers.py:1102, an rf/xgboost/mlp instance is built at :1160-1168, and create_hybrid at :1231-1239 gets model=<that instance>, metric_fn=None. It does not raise. The raise at hybrid.py:2347-2361 lives in _evaluate_allocation's model branch, and the budget split no longer goes through that branch: allocate_performance_weighted scores each representation through _score_rep_for_budget (hybrid.py:1380-1388), which fits the internal evaluator and then at :2665 does `if metric_fn is None or y_pred is None: return score` -- the evaluator's own r2. The only call sites that still pass `model` into _evaluate_allocation are augment_capped:3559 and augment_uncapped:3781, and federated_feature_selection passes no augmentations, so hybrid.py:5130 skips both. Net: fed05's C4 splits the feature budget across its three antibody silos by r2 on the z-scored pooled labels, not by the silo metric, and the guard the finalization doc says protects it is unreachable from every federated call.

## evidence as reported
fed05:538-539 `groups=pooled_groups, task_type='regression', downstream_model=fed_to_kirby_model(fed_model))` -- the argument list ends there, no metric_fn. federated_helpers.py:1102 `metric_fn=None, lower_is_better=False` in the signature; :1231-1239 passes `model=downstream_model_instance, metric_fn=metric_fn`. hybrid.py:2347 `if metric_fn is None:` then :2357 "_evaluate_allocation was given a model but no metric_fn." hybrid.py:2665 `if metric_fn is None or y_pred is None: return score`, reached from :1380 where _score_rep_for_budget is called with no `model` argument at all. hybrid.py:5130 `if augmentations is not None and augmentation_strategy != 'none':`.

## fix as proposed
Needs the owner's call on what metric a pooled, per-silo z-scored label column should be scored by -- the three silos are kcal/mol, a normalised DMS score and degrees C, and fed05 z-scores each before pooling (fed05:480-483). Once that is settled, pass metric_fn and lower_is_better into the fed05:532 call the way fed01:479 and fed03:397 already do. Separately, the raise at hybrid.py:2347 no longer covers the budget split: if it is meant to, the guard belongs in create_hybrid at entry (model is not None and metric_fn is None), which is where _shared.build_fold_hybrid:4207 already puts it.
