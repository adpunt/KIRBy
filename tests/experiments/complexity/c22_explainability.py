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
    search_score = best.get('score_mean', 0)
    print(f"  Blueprint: alloc={bp.allocation_method}, imp={bp.importance_method}, "
          f"eval={bp.internal_evaluator}, budget={bp.budget}")
    print(f"  Search score: {search_score:.4f}")

    # ── Per-fold hybrid construction + model evaluation ──
    fold_results = []
    all_shap_values = []
    all_feature_infos = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        cv_split(groups, n_folds, next(iter(reps.values())), labels, mode=mode)
    ):
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
        val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

        try:
            X_train, X_val, feature_info = build_fold_hybrid(
                train_reps, val_reps, y_train, bp,
                augmentations_train=train_augs,
                augmentations_val=val_augs)
        except Exception as e:
            print(f"    Fold {fold_idx}: hybrid build failed — {e}")
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        all_feature_infos.append(feature_info)

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
        try:
            import shap
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
                all_shap_values.append(mean_abs_shap)
        except Exception as e:
            print(f"    Fold {fold_idx}: SHAP failed — {e}")

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
    # Feature_info (selected indices, allocation) is deterministic given the same
    # blueprint, so all folds produce identical structure. Use the last fold.
    # SHAP values are properly averaged across folds below.
    feature_info = all_feature_infos[-1]

    # Average SHAP values across folds (if available)
    avg_shap = None
    if all_shap_values:
        # All folds should have same n_features. If they diverge, warn loudly
        # and skip rather than silently truncating (which would misalign SHAP
        # values with the last fold's feature indices).
        feature_counts = {len(sv) for sv in all_shap_values}
        if len(feature_counts) == 1:
            avg_shap = np.mean(all_shap_values, axis=0)
        else:
            print(f"  WARNING: SHAP arrays have differing lengths "
                  f"{sorted(feature_counts)} across folds — skipping aggregation "
                  f"to avoid misaligned feature indices.")
            avg_shap = None
        # If feature_info has an augmentation_info block, warn that the by-rep
        # SHAP breakdown does not currently account for augmentation features.
        if feature_info.get('augmentation_info'):
            print("  WARNING: feature_info contains augmentation_info — "
                  "aggregate_importance_by_rep / explain_hybrid ignore these "
                  "columns, so the per-rep SHAP breakdown is incomplete.")

    # ── Run explain_hybrid ──
    smiles = data.get('smiles')
    explanation = explain_hybrid(
        feature_info,
        smiles_list=smiles,
        importances=avg_shap,
        top_k=30,
    )

    # Per-rep importance aggregation
    rep_breakdown = explanation.get('by_representation', {})

    # If we have SHAP, also aggregate SHAP by rep
    shap_by_rep = {}
    if avg_shap is not None:
        shap_by_rep = aggregate_importance_by_rep(feature_info, avg_shap)

    # Print summary
    print(f"\n{'=' * 70}")
    print("EXPLAINABILITY SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Blueprint: {bp.allocation_method} / {bp.importance_method} / "
          f"{bp.internal_evaluator} / budget={bp.budget}")

    # Per-rep allocation
    allocation = feature_info.get('allocation', {})
    print(f"\n  Per-rep allocation (budget={bp.budget}):")
    for rep_name, n_feat in sorted(allocation.items(),
                                    key=lambda x: x[1], reverse=True):
        shap_pct = ''
        if rep_name in shap_by_rep:
            total_shap = sum(shap_by_rep.values())
            if total_shap > 0:
                shap_pct = f"  SHAP={100*shap_by_rep[rep_name]/total_shap:.1f}%"
        print(f"    {rep_name:30s} {n_feat:4d} features{shap_pct}")

    # Augmentation info
    aug_info = feature_info.get('augmentation_info', {})
    if aug_info:
        print(f"\n  Augmentation features kept:")
        for aug_name, aug_data in aug_info.items():
            n = aug_data.get('n_features', len(aug_data.get('selected_indices', [])))
            print(f"    {aug_name}: {n} features")

    # Top features
    summary = format_explanation_summary(explanation, max_features=15)
    print(f"\n  Top features:\n{summary}")

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
        'search_score': search_score,
        'allocation': allocation,
        'augmentation_info': {k: {'n_features': v.get('n_features', 0)}
                              for k, v in aug_info.items()} if aug_info else {},
        'rep_breakdown': {k: {'n_features': v.get('n_features', 0)}
                          for k, v in rep_breakdown.items()},
        'shap_by_rep': shap_by_rep,
        'top_features': explanation.get('top_features', [])[:30],
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
