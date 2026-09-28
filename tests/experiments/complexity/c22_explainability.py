#!/usr/bin/env python3
"""C22: Hybrid Explainability.

Paper: Complexity
Conclusion: The best hybrid is interpretable — we can trace every
    feature back to its source representation, provide per-rep
    importance breakdowns, and identify the most influential features.

Dependencies: Precomputed search (from precompute step).

Design:
    1. Retrieve the best hybrid PER DOWNSTREAM MODEL from the precomputed
       search (best_by_model), one recipe per model, never one model's recipe
       read under another model's name. A model the pool holds no recipe for is
       recorded in 'models_unscored' and is not trained at all.
    2. Build each model's own features on 80% of data (train), evaluate on 20%
       (val). The keep/drop test inside a model's build is run with that same
       model and the dataset's own metric.
    3. Train each downstream model on its own hybrid features.
    4. Run SHAP analysis on the best model → per-feature SHAP values, read
       against that model's own feature map.
    5. Use explain_hybrid to map hybrid features back to source reps
       with human-readable names where possible.
    6. Aggregate importance by rep → shows which reps contribute most.
    7. Report: blueprint, allocation, per-rep breakdown, top features,
       per-fold model scores, SHAP summary.

Usage:
    python c22_explainability.py --dataset herg --reps all --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split, get_domain_metric,
    get_precomputed_search, build_fold_hybrid,
    get_best_search_for_downstream,
)
from kirby.hybrid import create_hybrid, compute_feature_importance
from kirby.find_best_hybrid import _predict_for_metric
from kirby.importance.hybrid import (
    explain_hybrid, aggregate_importance_by_rep, format_explanation_summary,
)


def _reduce_shap_to_per_feature(shap_vals, n_features):
    """Collapse a SHAP output to a 1-D mean-abs-importance per feature.

    OTHER-29: downstream `explain_hybrid` / `aggregate_importance_by_rep`
    index importances as a 1-D array of length n_features. But SHAP's shape
    is task-dependent:
      - regression / single-output: (n_samples, n_features)
      - multiclass (shap>=0.40): a 3-D ndarray (n_samples, n_features, n_classes)
      - older shap multiclass: a Python list of (n_samples, n_features) arrays
    Without normalisation, binary/multiclass classification produces a
    2-D (n_features, n_classes) importance array → explain_hybrid crashes on
    float(importances[i]) (length-2 array) → SHAP silently marked 'fail' and
    the per-rep SHAP breakdown is empty. Reduce all cases to 1-D here.
    Returns None if the result can't be aligned to n_features.
    """
    arr = shap_vals
    if isinstance(arr, list):
        # list of per-class (n_samples, n_features) -> mean abs across classes
        arr = np.mean([np.abs(sv) for sv in arr], axis=0)
    arr = np.abs(np.asarray(arr))
    # Mean over the sample axis (axis 0) first.
    if arr.ndim >= 1:
        arr = arr.mean(axis=0)
    # arr is now either (n_features,) or (n_features, n_classes); collapse any
    # trailing class axis by averaging.
    while arr.ndim > 1:
        arr = arr.mean(axis=-1)
    if arr.shape[0] != n_features:
        return None
    return arr


def _fold_alloc_scores(train_reps, y_train, groups_train, blueprint,
                       metric_fn, is_clf):
    """{rep: score on this fold's training rows} when rep_count cuts, else None.

    Added 2026-09-28, same fix as c15's _fold_alloc_scores and c18's
    _fold_heldout_rep_scores. When the recipe's rep_count is smaller than the
    pool, build_fold_hybrid keeps the best rep_count representations. Without
    scores passed in it ranks them by blueprint.rep_scores, which the search
    measured over every row, this fold's held-out rows included. Under the
    local run logd and buchwald_hartwig hold 6 representations and rep_count
    is 5, so the dropped representation was chosen with the held-out labels.

    This is the call the search makes to rank them: the recipe's internal
    evaluator, fitted on the internal carve of this fold's training rows and
    scored with the dataset's metric, higher is better. Returns None when
    nothing is cut, so those folds build exactly as before.
    """
    rep_count = getattr(blueprint, 'rep_count', None)
    if rep_count is None or int(rep_count) >= len(train_reps):
        return None
    evaluator = getattr(blueprint, 'internal_evaluator', None)
    objective = getattr(blueprint, 'objective', None)
    from kirby.find_best_hybrid import (OBJECTIVE_PREDICTIVE,
                                        _is_geometry_objective)
    if (not isinstance(evaluator, str) or evaluator == 'match'
            or (objective is not None and (isinstance(objective, dict)
                                           or _is_geometry_objective(objective)))):
        print(f"    [c22] fold-local representation ranking not computed "
              f"(evaluator={evaluator!r}); build_fold_hybrid ranks on the "
              f"whole-dataset rep_scores")
        return None
    from kirby.hybrid import _score_rep_for_budget, internal_holdout_split
    from experiments._shared import object_direction
    lib = object_direction(blueprint, 'the c22 recipe')
    inner_tr, inner_va = internal_holdout_split(
        len(y_train), y_train, groups_train)
    out = {}
    for name in sorted(train_reps):
        X = train_reps[name]
        Xtr = np.nan_to_num(np.clip(X[inner_tr], -1e10, 1e10),
                            nan=0.0, posinf=1e10, neginf=-1e10)
        Xva = np.nan_to_num(np.clip(X[inner_va], -1e10, 1e10),
                            nan=0.0, posinf=1e10, neginf=-1e10)
        out[name] = float(_score_rep_for_budget(
            Xtr, Xva, y_train[inner_tr], y_train[inner_va],
            internal_evaluator=evaluator, metric_fn=metric_fn,
            lower_is_better=lib,
            objective=objective or OBJECTIVE_PREDICTIVE, is_clf=is_clf))
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """C22: Hybrid Explainability."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C22: HYBRID EXPLAINABILITY")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    n_samples = len(labels)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    augmentations = data.get('augmentations')

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")

    # Dirty local run 2026-09-28: this was the literal ['rf', 'xgboost'], so
    # --models was ignored, and under KIRBY_LOCAL_MODELS xgboost has no recipe
    # in the pool and was dropped, leaving rf alone. The run's own model list is
    # honoured now, kept to the tree models TreeExplainer handles; a non-tree
    # model (svm, dnn) would fall to KernelExplainer over every validation row,
    # which is not affordable. Falls back to the old pair when none is named.
    _TREE = ('rf', 'xgboost', 'lightgbm', 'catboost')
    _asked = list(config.get('models') or [])
    model_names = [m for m in _asked if m in _TREE] or ['rf', 'xgboost']
    _not_tree = [m for m in _asked if m not in _TREE]
    if _not_tree:
        print(f"  Not explained (no tree SHAP): {_not_tree}")
    models = get_model_factories(model_names, task_type=task_type,
                                 n_samples=n_samples)
    print(f"  Models: {list(models.keys())}")

    # ── One recipe per downstream model, and one feature build per recipe ──
    #
    # ⛔ WHAT THIS REPLACED, 2026-09-28. Until today one recipe was fetched,
    # with downstream_model='rf', and one feature set was built from it per
    # fold. Both rf and xgboost were then trained on that one feature set and
    # both had their SHAP values read off it. The two things that disagree: the
    # search chose those features by fitting a random forest and scoring it,
    # and one of the two models whose attributions were reported is not a
    # random forest. So xgboost's score and its per-representation SHAP share
    # were produced on features selected for a different model, under a name
    # that said they were xgboost's. Each model now gets the recipe the pool
    # stored for itself and its own per-fold feature build, and the keep/drop
    # test inside its build is run with that same model.
    #
    # A model the pool holds no recipe for is recorded in 'models_unscored'
    # and is not trained at all. No other model's recipe is put in its place.
    bp_by_model = {}
    meta_by_model = {}
    models_unscored = {}
    for _mname in list(models):
        try:
            _search_m, _meta_m = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=_mname, budget=None,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
            )
        except Exception as _exc:
            models_unscored[_mname] = (
                f'the pool lookup for this downstream model raised '
                f'{type(_exc).__name__}: {_exc}')
            continue
        if not _meta_m:
            models_unscored[_mname] = (
                'the precomputed pool holds no search for this downstream '
                'model, so it has no recipe of its own')
            continue
        _bp_m = _meta_m['best_config']['blueprint']
        # The pool sets the augmentation strategy per slot; realign it so the
        # rebuild happens under the condition that won (2026-04-21
        # cache-collision pattern).
        _aug_m = _meta_m.get('aug_condition', 'uncapped')
        if getattr(_bp_m, 'augmentation_strategy', None) != _aug_m:
            try:
                _bp_m.augmentation_strategy = _aug_m
            except Exception:
                pass
        bp_by_model[_mname] = _bp_m
        meta_by_model[_mname] = _meta_m
        print(f"  {_mname}: pool winner tier={_meta_m.get('tier')} "
              f"aug={_aug_m}; alloc={_bp_m.allocation_method}, "
              f"imp={_bp_m.importance_method}, "
              f"eval={_bp_m.internal_evaluator}, budget={_bp_m.budget}")
        # The score the allocation method, importance method, internal
        # evaluator and budget were CHOSEN by, computed over every row of the
        # dataset — including the rows each fold below scores as validation. It
        # is not performance on held-out rows; fold_results carries those, per
        # fold and per model.
        print(f"    recipe selection score (all rows, NOT held out): "
              f"{_meta_m['best_config'].get('score_mean', 0):.4f}")
    for _mname, _why in models_unscored.items():
        print(f"  {_mname}: unscored -- {_why}")
        models.pop(_mname, None)
    if not bp_by_model:
        print("  No hybrid search result available for any downstream model "
              "— skipping C22")
        return {'error': 'no_search_result',
                'models_unscored': models_unscored}

    # ── Per-fold hybrid construction + model evaluation ──
    fold_results = []
    all_shap_values = []
    shap_failures = []
    all_feature_infos = []
    # Per-fold representation shares, and per-fold budget splits, so the
    # movement between folds is on the record instead of being averaged away.
    shap_by_rep_folds = []
    allocation_by_fold = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        cv_split(groups, n_folds, next(iter(reps.values())), labels, mode=mode,
                 task_type=task_type)
    ):
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
        val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

        # ── One build per model, from that model's own recipe ──
        # The keep/drop test inside each build is given the same model the
        # search that produced this recipe fitted, and the dataset's own
        # metric. Without a model create_hybrid scores accept/reject with
        # r2_score or accuracy_score instead of the metric this dataset is
        # reported in (_shared.py:4442-4447).
        fold_model_scores = {}
        fold_widths = {}
        best_model_name = None
        best_score = -np.inf if not lower_is_better else np.inf
        best_model_obj = None
        # The feature map, the scaled validation features and the recipe of the
        # model SHAP explains on this fold. They must all three come from the
        # same build, because explain_hybrid zips a SHAP vector to a feature map
        # by position.
        best_feature_info = None
        best_X_val_s = None
        best_X_train_s = None
        fold_build_failures = {}
        fold_reps_kept = {}
        _g_tr = groups[train_idx] if groups is not None else None
        for model_name, model_factory in models.items():
            _bp_m = bp_by_model.get(model_name)
            if _bp_m is None:
                continue
            try:
                # Fold-local ranking when rep_count cuts the pool; None
                # otherwise, which leaves the build exactly as it was.
                _alloc = _fold_alloc_scores(
                    train_reps, y_train, _g_tr, _bp_m, metric_fn,
                    is_clf=(task_type == 'classification'))
                X_train, X_val, feature_info = build_fold_hybrid(
                    train_reps, val_reps, y_train, _bp_m,
                    augmentations_train=train_augs,
                    augmentations_val=val_augs,
                    groups=_g_tr,
                    model=model_factory(),
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better,
                    precomputed_alloc_scores=_alloc)
                fold_reps_kept[model_name] = {
                    'kept': feature_info.get('rep_count_kept') or sorted(train_reps),
                    'ranked_by': feature_info.get('rep_count_ranked_by'),
                }
            except Exception as e:
                fold_build_failures[model_name] = f'{type(e).__name__}: {e}'
                print(f"    Fold {fold_idx}: {model_name} hybrid build "
                      f"failed — {e}")
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            m = model_factory()
            m.fit(X_train_s, y_train)
            # OTHER-29: use _predict_for_metric so a binary-AUROC metric is
            # scored on class-1 probabilities, not hard labels (raw .predict
            # gave a degraded AUROC on classification datasets like humanness).
            pred = _predict_for_metric(m, X_val_s, metric_fn)
            score = float(metric_fn(y_val, pred))
            fold_model_scores[model_name] = score
            fold_widths[model_name] = int(X_train.shape[1])

            is_better = (score > best_score) if not lower_is_better else (score < best_score)
            if is_better:
                best_score = score
                best_model_name = model_name
                best_model_obj = m  # retain the FITTED model for SHAP
                best_feature_info = feature_info
                best_X_val_s = X_val_s
                best_X_train_s = X_train_s

        if best_model_name is None:
            print(f"    Fold {fold_idx}: no model both built and scored")
            continue

        # The feature map that pairs with this fold's SHAP vector is the
        # explained model's own, built from the explained model's own recipe.
        feature_info = best_feature_info
        X_train_s, X_val_s = best_X_train_s, best_X_val_s
        all_feature_infos.append((fold_idx, feature_info, best_model_name))

        # SHAP on the best model for this fold
        shap_info = None
        _lifted = {}
        try:
            import shap
            # ⛔ LIFT transformers OUT OF sys.modules FOR THE SHAP CALL.
            #
            # caco2's shap_share_by_rep came back EMPTY on all five folds while
            # flab_thermo_tresanco and herg_ki filled it, and it is not a caco2
            # defect. caco2's pool holds molformer, grover and smited, whose
            # builders import transformers, so sys.modules['transformers'] is a
            # lazy module by the time this loop runs. shap's safe_isinstance
            # SKIPS a class whose module is absent from sys.modules but calls
            # getattr(module, class_name) when it is present, and shap asks for
            # transformers.TFPreTrainedModel. The lazy module answers that by
            # importing transformers.modeling_tf_utils, which imports
            # TensorFlow, which dies here: this machine's protobuf has dropped
            # FieldDescriptor.is_repeated and the installed TensorFlow still
            # calls it. `import tensorflow` alone raises the same error in a
            # bare process, so TensorFlow is unusable here whatever shap does.
            #
            # Removing the entries makes safe_isinstance take its skip path.
            # They go back in the finally below, so nothing downstream sees a
            # different sys.modules.
            _lifted = {k: v for k, v in list(sys.modules.items())
                       if k == 'transformers' or k.startswith('transformers.')}
            for k in _lifted:
                del sys.modules[k]
            # Use the same fitted instance the score was selected from
            # (re-training a fresh factory would discard the scored model and
            # introduce stochastic drift between scored model and explanation).
            best_model = best_model_obj

            # Use TreeExplainer for tree models, KernelExplainer otherwise
            if best_model_name in ('rf', 'xgboost', 'lightgbm', 'catboost'):
                explainer = shap.TreeExplainer(best_model)
                shap_vals = explainer.shap_values(X_val_s)
            else:
                bg = shap.sample(X_train_s, min(100, len(X_train_s)))
                explainer = shap.KernelExplainer(best_model.predict, bg)
                shap_vals = explainer.shap_values(X_val_s)

            # Reduce to a 1-D mean-abs importance per hybrid feature, handling
            # regression (2-D), multiclass-list, and multiclass-3-D-ndarray
            # (shap>=0.40) shapes uniformly (OTHER-29).
            n_feat = X_val_s.shape[1]
            mean_abs_shap = _reduce_shap_to_per_feature(shap_vals, n_feat)
            if mean_abs_shap is None:
                print(f"    Fold {fold_idx}: SHAP shape could not be aligned "
                      f"to {n_feat} features — skipping this fold's SHAP.")
            else:
                shap_info = {
                    'mean_abs_shap': mean_abs_shap,
                    'model': best_model_name,
                    'n_val_samples': len(X_val_s),
                }
                all_shap_values.append((fold_idx, mean_abs_shap))
                # Aggregate to representations INSIDE the fold, while this
                # fold's own feature map is in hand. Position 37 names a
                # different feature in each fold, because create_hybrid splits
                # the budget from each fold's own training rows, so averaging
                # the raw per-position scores first and then reading them
                # against one fold's map adds up scores from different
                # features. Per-fold shares are what gets averaged below.
                shap_by_rep_folds.append(
                    aggregate_importance_by_rep(feature_info, mean_abs_shap))
                allocation_by_fold.append(
                    {k: int(v) for k, v in
                     (feature_info.get('allocation') or {}).items()})
        except Exception as e:
            # Recorded, not just printed: an empty shap_share_by_rep used to be
            # indistinguishable from a run where SHAP was never asked for.
            shap_failures.append(f'fold {fold_idx}: {type(e).__name__}: {e}')
            print(f"    Fold {fold_idx}: SHAP failed — {e}")
        finally:
            sys.modules.update(_lifted)

        fold_results.append({
            'fold_idx': fold_idx,
            'model_scores': fold_model_scores,
            'best_model': best_model_name,
            'n_train': len(y_train),
            'n_val': len(y_val),
            # The width of the space the explained model was scored on. Each
            # model has its own build now, so the widths are per model in
            # 'n_hybrid_features_by_model' beside it.
            'n_hybrid_features': int(X_train_s.shape[1]),
            'n_hybrid_features_by_model': dict(fold_widths),
            'build_failed_by_model': dict(fold_build_failures),
            # Which representations each model's build was restricted to, and
            # what ranked them ('fold_alloc_scores' = this fold's training
            # rows; None = rep_count did not cut).
            'reps_kept_by_model': dict(fold_reps_kept),
            'shap_available': shap_info is not None,
        })

        print(f"    Fold {fold_idx}: "
              f"{', '.join(f'{k}={v:.4f} ({fold_widths.get(k)}D)' for k, v in fold_model_scores.items())} "
              f"(explained {best_model_name}, shap={'ok' if shap_info else 'fail'})")

    if not fold_results:
        return {'error': 'all_folds_failed'}

    # ── Aggregate across folds ──
    # feature_info is NOT the same between folds: create_hybrid splits the
    # budget from each fold's own training rows, so which representation gets
    # how many features moves, and position 37 names a different feature in
    # each one. The last fold's map is kept only for the top-feature listing,
    # which is therefore that fold's listing and is labelled as such.
    # ⛔ THE MAP AND THE SHAP VECTOR MUST COME FROM THE SAME FOLD. These used
    # to be two bare lists taken at [-1] each: a feature map was appended for
    # every fold that BUILT, a SHAP vector only for every fold that also
    # EXPLAINED. So whenever SHAP failed on the last built fold and succeeded
    # on an earlier one, the two came from different folds -- and explain_hybrid
    # zips them by position. Position 37 names a different feature in each fold,
    # which is the reason given a few lines above for not averaging across
    # folds in the first place. Both lists now carry their fold index and the
    # pair is chosen from the folds that have both.
    #
    # Since 2026-09-28 each entry also carries the model the map was built
    # FOR, because there is now one build per model and the map that pairs
    # with a SHAP vector is the explained model's own.
    _shap_folds = {f for f, _ in all_shap_values}
    _paired = [f for f, _, _ in all_feature_infos if f in _shap_folds]
    _pair_fold = _paired[-1] if _paired else all_feature_infos[-1][0]
    feature_info, explained_model = next(
        (fi, mn) for f, fi, mn in all_feature_infos if f == _pair_fold)
    top_feature_fold = int(_pair_fold)
    # The recipe behind the feature map every per-representation number below
    # is read against, and the model it was searched for.
    bp = bp_by_model[explained_model]
    recipe_selection_score = meta_by_model[explained_model][
        'best_config'].get('score_mean', 0)

    # Which representations actually moved between folds. Two folds with
    # different allocations cannot share a feature-position map at all.
    allocation_moved = (len({tuple(sorted(a.items()))
                             for a in allocation_by_fold}) > 1
                        if allocation_by_fold else False)
    if allocation_moved:
        print("  Budget split differs between folds; the per-representation "
              "SHAP shares below are averaged per fold, not by feature "
              "position.")

    # The top-feature listing reads one fold's SHAP vector against that same
    # fold's feature map, always the last one. There is no cross-fold average
    # any more: averaging by feature position needs position 37 to name the
    # same feature in every fold, and equal per-representation counts do not
    # give that. create_hybrid ranks each representation's own features from
    # each fold's own training rows, so two folds can both take 12 features
    # from morgan and not take the same 12. Every cross-fold representation
    # number comes from shap_by_rep_folds, which was matched to each fold's own
    # map before anything was averaged.
    last_fold_shap = None
    if all_shap_values:
        last_fold_shap = next((sv for f, sv in all_shap_values
                               if f == _pair_fold), None)
        if last_fold_shap is None:
            print(f"  No fold both built and explained; the top-feature "
                  f"listing is dropped rather than paired across folds.")
        feature_counts = {len(sv) for _f, sv in all_shap_values}
        if len(feature_counts) > 1:
            print(f"  Hybrid width differs between folds "
                  f"{sorted(feature_counts)}; the top-feature listing is the "
                  f"last fold's, as it is on every run.")
        # If feature_info has an augmentation_info block, note that the by-rep
        # SHAP breakdown does not currently account for augmentation features.
        if feature_info.get('augmentation_info'):
            print("  NOTE: feature_info contains augmentation_info — "
                  "aggregate_importance_by_rep / explain_hybrid ignore these "
                  "columns, so the per-representation SHAP breakdown is "
                  "incomplete.")

    # ── Run explain_hybrid ──
    smiles = data.get('smiles')
    explanation = explain_hybrid(
        feature_info,
        smiles_list=smiles,
        importances=last_fold_shap,
        top_k=30,
    )

    # Per-rep importance aggregation
    rep_breakdown = explanation.get('by_representation', {})

    # Per-representation SHAP: each fold's own shares, averaged. Every fold's
    # scores were matched to that fold's own feature map before averaging, so
    # nothing sums scores from different features.
    # ⛔ THE MODEL EXPLAINED IS CHOSEN PER FOLD, SO THIS AVERAGE CAN MIX TWO
    # KINDS OF ATTRIBUTION. rf and xgboost are both fitted on every fold and
    # whichever scores better on that fold's validation rows is the one SHAP
    # explains. A per-representation share averaged over folds is then a mean
    # of a forest's attributions and a boosted model's wherever the winner
    # changed. Since 2026-09-28 each model also builds its features from its
    # own recipe, so where the winner changed the recipe behind the feature map
    # changed with it. fold_results carries best_model per fold, so both are
    # recoverable — they just were not stated beside the number they affect.
    shap_by_rep = {}
    shap_share_by_rep = {}
    shap_by_rep_std = {}
    shap_models_by_fold = [fr.get('best_model') for fr in fold_results]
    shap_models_used = sorted({m for m in shap_models_by_fold if m})
    if len(shap_models_used) > 1:
        print(f"\n  ⛔ the SHAP averages below mix {len(shap_models_used)} "
              f"models: per fold {shap_models_by_fold}. A share is a mean over "
              f"folds explained by different estimators.")
    if shap_by_rep_folds:
        _reps_seen = sorted({r for d in shap_by_rep_folds for r in d})
        for r in _reps_seen:
            vals = [d.get(r, 0.0) for d in shap_by_rep_folds]
            shap_by_rep[r] = float(np.mean(vals))
            shap_by_rep_std[r] = float(np.std(vals))
        # Shares first, then averaged: a fold with a larger total SHAP does
        # not get a larger vote on which representation carries the weight.
        for r in _reps_seen:
            shares = []
            for d in shap_by_rep_folds:
                tot = sum(abs(v) for v in d.values())
                if tot > 0:
                    shares.append(abs(d.get(r, 0.0)) / tot)
            shap_share_by_rep[r] = float(np.mean(shares)) if shares else 0.0

    # Print summary
    print(f"\n{'=' * 70}")
    print("EXPLAINABILITY SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Blueprint of the explained model ({explained_model}, fold "
          f"{top_feature_fold}): {bp.allocation_method} / "
          f"{bp.importance_method} / {bp.internal_evaluator} / "
          f"budget={bp.budget}")
    for _mname, _bp_m in bp_by_model.items():
        if _mname == explained_model:
            continue
        print(f"  Blueprint of {_mname}: {_bp_m.allocation_method} / "
              f"{_bp_m.importance_method} / {_bp_m.internal_evaluator} / "
              f"budget={_bp_m.budget}")

    # Per-representation budget split. The count is the paired fold's; the
    # spread across folds is printed beside it when it moves.
    #
    # ⛔ 'allocation' MEANS TWO DIFFERENT THINGS DEPENDING ON THE RECIPE.
    # performance_weighted, power_2, softmax_T1 and rank_exp_d0.3 all resolve
    # to the per-representation quota path, where hybrid.py:6150 stores what
    # the allocator ASKED FOR, while concatenate_features independently cuts
    # each representation at its last feature scoring above zero and records
    # the shortfall as n_unfilled. wglobal_* resolves to
    # allocate_weighted_global, which builds the same dict from the counts
    # actually SELECTED (hybrid.py:2457). So under four of the five the
    # printed split can sum to more than the hybrid is wide, and under the
    # fifth it cannot -- and the SHAP share printed on the same line is always
    # built-based. Both numbers are now shown when they differ.
    allocation = feature_info.get('allocation', {})
    built = {r: int(v.get('n_features', 0))
             for r, v in feature_info.items()
             if isinstance(v, dict) and 'n_features' in v}
    _asked_total = sum(allocation.values()) if allocation else 0
    _built_total = sum(built.values()) if built else 0
    print(f"\n  Per-representation split (budget={bp.budget}, "
          f"counts from fold {top_feature_fold}):")
    if built and _built_total != _asked_total:
        print(f"    the allocator asked for {_asked_total} features and "
              f"{_built_total} were built; 'asked' below is the quota, "
              f"'built' is what the importance ranking could fill")
    for rep_name, n_feat in sorted(allocation.items(),
                                    key=lambda x: x[1], reverse=True):
        shap_pct = ''
        if rep_name in shap_share_by_rep:
            shap_pct = f"  SHAP={100*shap_share_by_rep[rep_name]:.1f}%"
        spread = ''
        if allocation_by_fold:
            counts = [a.get(rep_name, 0) for a in allocation_by_fold]
            if len(set(counts)) > 1:
                spread = f"  (per fold {counts})"
        _b = built.get(rep_name)
        _bs = (f" asked, {_b:4d} built" if _b is not None and _b != n_feat
               else " features")
        print(f"    {rep_name:30s} {n_feat:4d}{_bs}{shap_pct}{spread}")

    # Augmentation info
    aug_info = feature_info.get('augmentation_info', {})
    if aug_info:
        print(f"\n  Augmentation features kept:")
        for aug_name, aug_data in aug_info.items():
            n = aug_data.get('n_features', len(aug_data.get('selected_indices', [])))
            print(f"    {aug_name}: {n} features")

    # Top features — from the LAST fold's feature map and that fold's own
    # SHAP vector. A cross-fold top-feature list would need the same feature
    # in the same position in every fold, which the per-fold budget split does
    # not give.
    summary = format_explanation_summary(explanation, max_features=15)
    print(f"\n  Top features (last fold only):\n{summary}")

    # Per-model mean scores
    print(f"\n  Per-model mean scores ({metric_name}):")
    for model_name in models:
        scores = [fr['model_scores'].get(model_name, float('nan'))
                  for fr in fold_results]
        print(f"    {model_name}: {np.nanmean(scores):.4f} +/- {np.nanstd(scores):.4f}")

    # ── Build result dict ──
    return convert_numpy({
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_folds': n_folds,
        'n_samples': n_samples,
        # The recipe behind the feature map every per-representation number in
        # this file is read against: the explained model's own, on the fold
        # named by 'top_feature_fold'. Every model's own recipe is in
        # 'blueprint_by_model' beside it.
        'blueprint': bp.to_dict(),
        'blueprint_by_model': {_m: _b.to_dict()
                               for _m, _b in bp_by_model.items()},
        'explained_model': explained_model,
        # Downstream models the pool holds no recipe for. They are not trained
        # and have no rows, because the only recipe on offer belonged to
        # another model. Added 2026-09-28 with the per-model builds.
        'models_unscored': models_unscored,
        'models_scored': sorted(bp_by_model),
        # Renamed from 'search_score', which read as performance. It is the
        # selection score of the recipe, measured over every row. It is the
        # explained model's recipe's score; every model's is beside it.
        'recipe_selection_score': recipe_selection_score,
        'recipe_selection_score_by_model': {
            _m: meta_by_model[_m]['best_config'].get('score_mean', 0)
            for _m in bp_by_model},
        'recipe_selected_on': 'all rows',
        # Both meanings, side by side, so a reader never has to know which
        # allocation method wrote which. 'allocation' is the quota under the
        # per-representation methods and the built count under wglobal_*;
        # 'allocation_built' is always the built count.
        'allocation': allocation,
        'allocation_built': built,
        # Which estimator's SHAP each fold contributed, and whether the
        # averages below mix more than one.
        'shap_model_by_fold': shap_models_by_fold,
        'shap_models_mixed': bool(len(shap_models_used) > 1),
        'allocation_asked_total': _asked_total,
        'allocation_built_total': _built_total,
        'augmentation_info': {k: {'n_features': v.get('n_features', 0)}
                              for k, v in aug_info.items()} if aug_info else {},
        'rep_breakdown': {k: {'n_features': v.get('n_features', 0)}
                          for k, v in rep_breakdown.items()},
        # Mean over folds of each fold's own representation total, matched to
        # that fold's own feature map before averaging.
        'shap_by_rep': shap_by_rep,
        'shap_by_rep_std': shap_by_rep_std,
        # Mean over folds of each representation's SHARE within its fold. Read
        # this one for "which representation carries the weight" — it does not
        # let a fold with a larger total SHAP outvote the others.
        'shap_share_by_rep': shap_share_by_rep,
        # How many folds SHAP actually scored. An empty shap_share_by_rep used
        # to be indistinguishable from a run where SHAP was never asked for; on
        # caco2 it was empty on all five folds because shap's safe_isinstance
        # reached into a lazily-imported transformers and pulled in TensorFlow.
        'shap_folds_scored': len(all_shap_values),
        # Which fold the top-feature listing and its SHAP vector both come
        # from, and how many folds built at all. A build that raised used to
        # be invisible here: n_folds is the count ASKED for.
        'top_feature_fold': top_feature_fold,
        'build_folds_scored': len(all_feature_infos),
        'build_folds_requested': int(n_folds),
        'shap_folds_requested': n_folds,
        'shap_failures': shap_failures,
        'shap_by_rep_folds': shap_by_rep_folds,
        'shap_by_rep_basis': 'per fold, then averaged',
        # The budget split per fold. When these differ, no feature position is
        # comparable between folds.
        'allocation_by_fold': allocation_by_fold,
        'allocation_moved_between_folds': allocation_moved,
        'allocation_is_from_fold': 'last',
        'top_features': explanation.get('top_features', [])[:30],
        'top_features_from_fold': 'last',
        # One fold's SHAP read against that same fold's feature map. No
        # cross-fold average is taken at any hybrid width.
        'top_features_basis': "the last fold's own SHAP vector",
        # The model create_hybrid's keep/drop test was run with inside the
        # build behind the reported feature map, scored with the dataset's own
        # metric. Since 2026-09-28 every model builds its own features from its
        # own recipe and is the model its own keep/drop test is run with, so
        # this is the explained model's name.
        'fold_build_model': explained_model,
        'fold_build_model_is_per_model': True,
        'fold_build_metric': metric_name,
        'fold_results': fold_results,
    })


if __name__ == '__main__':
    parser = create_experiment_parser('C22: Hybrid Explainability',
                                     has_delta=False, has_augment=False)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c22_explainability',
                            args.dataset)
