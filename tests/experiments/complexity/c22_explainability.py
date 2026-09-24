#!/usr/bin/env python3
"""C22: Hybrid Explainability.

Paper: Complexity
Conclusion: The best hybrid is interpretable — we can trace every
    feature back to its source representation, provide per-rep
    importance breakdowns, and identify the most influential features.

Dependencies: Precomputed search (from precompute step).

Design:
    1. Retrieve the best hybrid from precomputed search (best_by_model).
    2. Build it on 80% of data (train), evaluate on 20% (val).
    3. Train downstream models on the hybrid.
    4. Run SHAP analysis on the best model → per-feature SHAP values.
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

    model_names = ['rf', 'xgboost']
    models = get_model_factories(model_names, task_type=task_type,
                                 n_samples=n_samples)
    print(f"  Models: {list(models.keys())}")

    # The model handed to each fold's build. The recipe below is fetched for a
    # random forest downstream, so the keep/drop test inside the build is run
    # with a random forest too; both downstream models are then scored on the
    # features that chose.
    build_model_name = 'rf' if 'rf' in models else next(iter(models))
    build_model_factory = models[build_model_name]

    # ── Get best hybrid via pool helper (rf at budget=None = best across budgets) ──
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type,
        data=data,
    )
    if not meta:
        print("  No hybrid search result available — skipping C22")
        return {'error': 'no_search_result'}

    best = meta['best_config']
    bp = best['blueprint']
    print(f"  Pool winner: tier={meta['tier']} aug={meta['aug_condition']}")
    # The score the allocation method, importance method, internal evaluator
    # and budget were CHOSEN by, computed over every row of the dataset —
    # including the rows each fold below scores as validation. It is not
    # performance on held-out rows; fold_results carries those, per fold and
    # per model.
    recipe_selection_score = best.get('score_mean', 0)
    print(f"  Blueprint: alloc={bp.allocation_method}, imp={bp.importance_method}, "
          f"eval={bp.internal_evaluator}, budget={bp.budget}")
    print(f"  Recipe selection score (all rows, NOT held out): "
          f"{recipe_selection_score:.4f}")

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
        cv_split(groups, n_folds, next(iter(reps.values())), labels, mode=mode)
    ):
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
        val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

        try:
            # The recipe was chosen for a random forest downstream, so the
            # keep/drop test inside the build is given a random forest and the
            # dataset's own metric. Without a model create_hybrid scores
            # accept/reject with r2_score or accuracy_score instead of the
            # metric this dataset is reported in (_shared.py:4442-4447), so the
            # features being explained here were chosen under the wrong
            # quantity.
            X_train, X_val, feature_info = build_fold_hybrid(
                train_reps, val_reps, y_train, bp,
                augmentations_train=train_augs,
                augmentations_val=val_augs,
                groups=(groups[train_idx] if groups is not None else None),
                model=build_model_factory(),
                metric_fn=metric_fn,
                lower_is_better=lower_is_better)
        except Exception as e:
            print(f"    Fold {fold_idx}: hybrid build failed — {e}")
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        all_feature_infos.append((fold_idx, feature_info))

        # Train each model, collect scores
        fold_model_scores = {}
        best_model_name = None
        best_score = -np.inf if not lower_is_better else np.inf

        best_model_obj = None
        for model_name, model_factory in models.items():
            m = model_factory()
            m.fit(X_train_s, y_train)
            # OTHER-29: use _predict_for_metric so a binary-AUROC metric is
            # scored on class-1 probabilities, not hard labels (raw .predict
            # gave a degraded AUROC on classification datasets like humanness).
            pred = _predict_for_metric(m, X_val_s, metric_fn)
            score = float(metric_fn(y_val, pred))
            fold_model_scores[model_name] = score

            is_better = (score > best_score) if not lower_is_better else (score < best_score)
            if is_better:
                best_score = score
                best_model_name = model_name
                best_model_obj = m  # retain the FITTED model for SHAP

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
            'n_hybrid_features': X_train.shape[1],
            'shap_available': shap_info is not None,
        })

        print(f"    Fold {fold_idx}: {', '.join(f'{k}={v:.4f}' for k, v in fold_model_scores.items())} "
              f"(hybrid={X_train.shape[1]}D, shap={'ok' if shap_info else 'fail'})")

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
    _shap_folds = {f for f, _ in all_shap_values}
    _paired = [f for f, _ in all_feature_infos if f in _shap_folds]
    _pair_fold = _paired[-1] if _paired else all_feature_infos[-1][0]
    feature_info = next(fi for f, fi in all_feature_infos if f == _pair_fold)
    top_feature_fold = int(_pair_fold)

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
    shap_by_rep = {}
    shap_share_by_rep = {}
    shap_by_rep_std = {}
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
    print(f"  Blueprint: {bp.allocation_method} / {bp.importance_method} / "
          f"{bp.internal_evaluator} / budget={bp.budget}")

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
        'blueprint': bp.to_dict(),
        # Renamed from 'search_score', which read as performance. It is the
        # selection score of the recipe, measured over every row.
        'recipe_selection_score': recipe_selection_score,
        'recipe_selected_on': 'all rows',
        # Both meanings, side by side, so a reader never has to know which
        # allocation method wrote which. 'allocation' is the quota under the
        # per-representation methods and the built count under wglobal_*;
        # 'allocation_built' is always the built count.
        'allocation': allocation,
        'allocation_built': built,
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
        # The model create_hybrid's keep/drop test was run with inside each
        # fold's build, scored with the dataset's own metric.
        'fold_build_model': build_model_name,
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
