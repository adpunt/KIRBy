#!/usr/bin/env python3
"""C18: Degrees of Freedom — Diversity vs Depth.

Paper: Complexity
Core question: Given a fixed total budget, is it better to draw features from many
    reps (diversity) or concentrate on few reps (depth)? Tests whether the hybrid's
    allocation across rep families adds value beyond selecting the best single rep.

Conditions at fixed budget:
  - depth_1: all features from top-1 rep
  - depth_2: all features split between top-2 reps
  - depth_3: all features from top-3 reps
  - uniform_all: equal allocation across all reps
  - natural: let the hybrid allocate freely (greedy)

Usage:
    python c18_diversity_vs_depth.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric,
    estimate_mutual_information_knn, compute_effective_rank,
    compute_feature_importance,
)
from kirby.hybrid import create_hybrid
from kirby.hybrid import apply_feature_selection


# =========================================================================
# FORCED ALLOCATION
# =========================================================================

def _rank_reps_by_importance(reps, labels):
    """Rank reps by aggregate importance. Returns sorted list of (name, score)."""
    # Concat all, get importance, aggregate per rep
    rep_names = sorted(reps.keys())
    X_concat = np.hstack([reps[n] for n in rep_names])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_concat)
    imp = compute_feature_importance(X_s, labels, method='lightgbm_gain')

    # Map feature indices back to reps
    rep_scores = {}
    offset = 0
    for name in rep_names:
        dim = reps[name].shape[1]
        rep_scores[name] = float(np.sum(imp[offset:offset + dim]))
        offset += dim

    ranked = sorted(rep_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def _select_top_features_from_reps(reps, labels, rep_subset, budget):
    """Select top-budget features from a subset of reps using importance."""
    sub_reps = {k: reps[k] for k in rep_subset if k in reps}
    if not sub_reps:
        return None

    X_concat = np.hstack([sub_reps[n] for n in sorted(sub_reps.keys())])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_concat)
    imp = compute_feature_importance(X_s, labels, method='lightgbm_gain')

    # Select top-budget features
    n_select = min(budget, X_s.shape[1])
    top_idx = np.argsort(imp)[::-1][:n_select]
    return X_s[:, top_idx]


def _forced_uniform_allocation(reps, labels, rep_subset, budget):
    """Equal allocation: budget // n_reps features per rep, top by importance."""
    sub_names = sorted([k for k in rep_subset if k in reps])
    if not sub_names:
        return None

    per_rep = max(1, budget // len(sub_names))
    selected_parts = []

    for name in sub_names:
        X = StandardScaler().fit_transform(reps[name])
        imp = compute_feature_importance(X, labels, method='lightgbm_gain')
        n_sel = min(per_rep, X.shape[1])
        top_idx = np.argsort(imp)[::-1][:n_sel]
        selected_parts.append(X[:, top_idx])

    return np.hstack(selected_parts)


# =========================================================================
# EVALUATE
# =========================================================================

def _evaluate(X, labels, groups, n_folds, mode, metric_fn):
    """Evaluate a feature matrix. Returns dict with R², MI, erank."""
    scores = []
    for train_idx, val_idx in cv_split(groups, n_folds, X, labels, mode=mode):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, labels[train_idx])
        scores.append(metric_fn(labels[val_idx], rf.predict(X_va)))

    r2 = float(np.mean(scores)) if scores else float('nan')

    # MI
    n_pca = min(50, X.shape[1], len(labels) - 1)
    from sklearn.decomposition import PCA
    X_pca = PCA(n_components=max(1, n_pca)).fit_transform(
        StandardScaler().fit_transform(X))
    mi = estimate_mutual_information_knn(X_pca, labels, k=5)

    erank = compute_effective_rank(X)

    return {
        'r2': r2,
        'mi': float(mi),
        'mi_density': float(mi / X.shape[1]) if X.shape[1] > 0 else 0.0,
        'effective_rank': float(erank),
        'n_features': X.shape[1],
    }


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """C18: Diversity vs Depth."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C18: DIVERSITY VS DEPTH")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, _ = get_domain_metric(data)

    budgets = kwargs.get('budgets', [50, 100])
    all_rep_names = sorted(reps.keys())
    n_reps = len(all_rep_names)

    print(f"  {n_reps} reps, budgets: {budgets}")

    # Rank reps by importance
    ranked = _rank_reps_by_importance(reps, labels)
    print(f"  Rep ranking: {', '.join(f'{n}({s:.2f})' for n, s in ranked[:5])}...")

    results = {
        'metric': metric_name,
        'budgets': budgets,
        'n_reps': n_reps,
        'rep_ranking': [(n, float(s)) for n, s in ranked],
        'per_budget': {},
    }

    for budget in budgets:
        print(f"\n  Budget={budget}:")
        conditions = {}

        # Depth conditions
        for n_top in [1, 2, 3]:
            if n_top > n_reps:
                continue
            cond_name = f'depth_{n_top}'
            top_reps = [name for name, _ in ranked[:n_top]]
            X = _select_top_features_from_reps(reps, labels, top_reps, budget)
            if X is not None:
                metrics = _evaluate(X, labels, groups, n_folds, mode, metric_fn)
                metrics['reps_used'] = top_reps
                conditions[cond_name] = metrics
                print(f"    {cond_name} ({', '.join(top_reps)}): "
                      f"R²={metrics['r2']:.4f}, MI_d={metrics['mi_density']:.4f}")

        # Uniform allocation across all reps
        X_uniform = _forced_uniform_allocation(reps, labels, all_rep_names, budget)
        if X_uniform is not None:
            metrics = _evaluate(X_uniform, labels, groups, n_folds, mode, metric_fn)
            metrics['reps_used'] = all_rep_names
            conditions['uniform_all'] = metrics
            print(f"    uniform_all ({n_reps} reps): "
                  f"R²={metrics['r2']:.4f}, MI_d={metrics['mi_density']:.4f}")

        # Natural hybrid allocation (greedy)
        augmentations = data.get('augmentations')
        try:
            X_h, fi = create_hybrid(
                reps, labels,
                allocation_method='entropy_1.0',
                importance_method='lightgbm_gain',
                budget=budget,
                augmentations=augmentations,
                augmentation_strategy='capped',
            )
            metrics = _evaluate(X_h, labels, groups, n_folds, mode, metric_fn)
            # Count how many reps contributed
            reps_used = [name for name, info in fi.items()
                         if info.get('n_selected', 0) > 0]
            metrics['reps_used'] = reps_used
            metrics['n_reps_contributing'] = len(reps_used)
            conditions['natural'] = metrics
            print(f"    natural ({len(reps_used)} reps contributing): "
                  f"R²={metrics['r2']:.4f}, MI_d={metrics['mi_density']:.4f}")
        except Exception as e:
            print(f"    natural FAILED: {e}")

        results['per_budget'][budget] = convert_numpy(conditions)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C18: Diversity vs Depth')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c18_diversity_vs_depth',
                            args.dataset)
