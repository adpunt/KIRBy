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
from kirby.importance.hybrid import (
    explain_hybrid, aggregate_importance_by_rep, format_explanation_summary,
)


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
        speed=config.get('search_speed', 'full'),
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

        for model_name, model_factory in models.items():
            m = model_factory()
            m.fit(X_train_s, y_train)
            pred = m.predict(X_val_s)
            score = float(metric_fn(y_val, pred))
            fold_model_scores[model_name] = score

            is_better = (score > best_score) if not lower_is_better else (score < best_score)
            if is_better:
                best_score = score
                best_model_name = model_name

        # SHAP on the best model for this fold
        shap_info = None
        try:
            import shap
            best_model = models[best_model_name]()  # Re-train best model type
            best_model.fit(X_train_s, y_train)

            # Use TreeExplainer for tree models, KernelExplainer otherwise
            if best_model_name in ('rf', 'xgboost', 'lightgbm', 'catboost'):
                explainer = shap.TreeExplainer(best_model)
                shap_vals = explainer.shap_values(X_val_s)
            else:
                bg = shap.sample(X_train_s, min(100, len(X_train_s)))
                explainer = shap.KernelExplainer(best_model.predict, bg)
                shap_vals = explainer.shap_values(X_val_s)

            # For multi-output (classification), take mean abs across classes
            if isinstance(shap_vals, list):
                shap_vals = np.mean([np.abs(sv) for sv in shap_vals], axis=0)

            # Mean absolute SHAP per feature
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
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
        # All folds should have same n_features; take element-wise mean
        min_len = min(len(sv) for sv in all_shap_values)
        avg_shap = np.mean([sv[:min_len] for sv in all_shap_values], axis=0)

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
