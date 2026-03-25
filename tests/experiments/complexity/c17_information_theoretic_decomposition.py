#!/usr/bin/env python3
"""C17: Information-Theoretic Feature Selection Comparison.

Paper: Complexity
Core thesis: The hybrid selects features that maximize MI with the target per unit of
    dimension — it finds the minimum description length representation. Compare against
    PCA (variance-based), mRMR (MI-greedy), RFE (model-based), and random.

5 strategies × multiple budgets → MI density curves.

Usage:
    python c17_information_theoretic_decomposition.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    estimate_mutual_information_knn, compute_effective_rank,
    cv_split, get_domain_metric,
)
from experiments.complexity.complexity_helpers import cached_create_hybrid as create_hybrid


# =========================================================================
# mRMR (greedy max-relevance minus redundancy)
# =========================================================================

def _greedy_mrmr(X, y, budget, k_mi=5):
    """Greedy mRMR feature selection using KSG MI estimator.

    At each step, select the feature that maximizes:
        MI(x_j, y) - (1/|S|) * sum_{x_i in S} MI(x_j, x_i)

    Returns selected feature indices.
    """
    n_features = X.shape[1]
    budget = min(budget, n_features)

    # Pre-compute MI(x_j, y) for all features
    relevance = np.zeros(n_features)
    for j in range(n_features):
        try:
            relevance[j] = estimate_mutual_information_knn(
                X[:, j:j+1], y, k=k_mi)
        except Exception:
            relevance[j] = 0.0

    selected = []
    remaining = list(range(n_features))

    # First feature: max relevance
    best_first = max(remaining, key=lambda j: relevance[j])
    selected.append(best_first)
    remaining.remove(best_first)

    # Greedy selection
    for _ in range(budget - 1):
        if not remaining:
            break
        best_score = -np.inf
        best_j = remaining[0]

        for j in remaining:
            # Redundancy with already-selected features
            redundancy = 0.0
            for s in selected:
                try:
                    redundancy += estimate_mutual_information_knn(
                        X[:, j:j+1], X[:, s], k=k_mi)
                except Exception:
                    pass
            redundancy /= len(selected)

            score = relevance[j] - redundancy
            if score > best_score:
                best_score = score
                best_j = j

        selected.append(best_j)
        remaining.remove(best_j)

    return selected


# =========================================================================
# RFE (recursive feature elimination with RF)
# =========================================================================

def _rfe_select(X, y, budget):
    """Select top features via sklearn RFE with RF."""
    from sklearn.feature_selection import RFE
    budget = min(budget, X.shape[1])
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rfe = RFE(rf, n_features_to_select=budget, step=max(1, X.shape[1] // 10))
    rfe.fit(X, y)
    return list(np.where(rfe.support_)[0])


# =========================================================================
# STRATEGY EVALUATION
# =========================================================================

def _evaluate_strategy(X_selected, y, groups, n_folds, mode, metric_fn):
    """Evaluate a feature-selected matrix: MI, MI density, erank, downstream R²."""
    n_feat = X_selected.shape[1]

    # MI with labels (PCA-reduce for stability if high-dim)
    n_pca = min(50, n_feat, len(y) - 1)
    if n_pca >= 2:
        X_pca = PCA(n_components=n_pca).fit_transform(
            StandardScaler().fit_transform(X_selected))
    else:
        X_pca = StandardScaler().fit_transform(X_selected)
    mi = estimate_mutual_information_knn(X_pca, y, k=5)

    # MI density
    mi_density = mi / n_feat if n_feat > 0 else 0.0

    # Effective rank
    erank = compute_effective_rank(X_selected)

    # Downstream R² with RF
    scores = []
    for train_idx, val_idx in cv_split(groups, n_folds, X_selected, y, mode=mode):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_selected[train_idx])
        X_va = scaler.transform(X_selected[val_idx])
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y[train_idx])
        scores.append(metric_fn(y[val_idx], rf.predict(X_va)))
    r2 = float(np.mean(scores)) if scores else float('nan')

    return {
        'mi': float(mi),
        'mi_density': float(mi_density),
        'effective_rank': float(erank),
        'downstream_r2': r2,
        'n_features': n_feat,
    }


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """C17: Information-theoretic feature selection comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C17: INFORMATION-THEORETIC DECOMPOSITION")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, _ = get_domain_metric(data)

    # Concatenate all reps
    X_concat = np.hstack([v for v in reps.values()])
    total_dim = X_concat.shape[1]
    X_concat_scaled = StandardScaler().fit_transform(X_concat)

    budgets = [b for b in [25, 50, 100, 200] if b < total_dim]
    # mRMR is O(budget * n_features * n_samples) — limit for large concat
    mrmr_max_features = 2000
    run_mrmr = total_dim <= mrmr_max_features

    print(f"  Total dim: {total_dim}, budgets: {budgets}")
    print(f"  mRMR: {'enabled' if run_mrmr else f'disabled (total_dim={total_dim} > {mrmr_max_features})'}")

    strategies = ['hybrid', 'pca', 'rfe', 'random']
    if run_mrmr:
        strategies.insert(2, 'mrmr')

    results = {
        'metric': metric_name,
        'budgets': budgets,
        'total_dim': total_dim,
        'strategies': strategies,
        'per_budget': {},
    }

    for budget in budgets:
        print(f"\n  Budget={budget}:")
        budget_results = {}

        for strategy in strategies:
            print(f"    {strategy}...", end='', flush=True)

            try:
                if strategy == 'hybrid':
                    X_h, fi = create_hybrid(
                        reps, labels,
                        allocation_method='greedy',
                        importance_method='lightgbm_gain',
                        budget=budget,
                    )
                    X_selected = X_h

                elif strategy == 'pca':
                    n_comp = min(budget, total_dim, len(labels) - 1)
                    pca = PCA(n_components=n_comp, random_state=42)
                    X_selected = pca.fit_transform(X_concat_scaled)

                elif strategy == 'mrmr':
                    indices = _greedy_mrmr(X_concat_scaled, labels, budget, k_mi=3)
                    X_selected = X_concat_scaled[:, indices]

                elif strategy == 'rfe':
                    indices = _rfe_select(X_concat_scaled, labels, budget)
                    X_selected = X_concat_scaled[:, indices]

                elif strategy == 'random':
                    rng = np.random.RandomState(42)
                    indices = rng.choice(total_dim, min(budget, total_dim), replace=False)
                    X_selected = X_concat_scaled[:, indices]

                metrics = _evaluate_strategy(X_selected, labels, groups, n_folds,
                                             mode, metric_fn)
                budget_results[strategy] = metrics
                print(f" MI={metrics['mi']:.3f}, density={metrics['mi_density']:.4f}, "
                      f"R²={metrics['downstream_r2']:.4f}")

            except Exception as e:
                print(f" FAILED: {e}")
                budget_results[strategy] = {'error': str(e)}

        results['per_budget'][budget] = convert_numpy(budget_results)

    # Summary: which strategy has highest MI density on average?
    avg_density = {}
    for strategy in strategies:
        densities = []
        for budget in budgets:
            d = results['per_budget'].get(budget, {}).get(strategy, {}).get('mi_density')
            if d is not None and not np.isnan(d):
                densities.append(d)
        avg_density[strategy] = float(np.mean(densities)) if densities else float('nan')

    results['summary'] = {
        'avg_mi_density': avg_density,
        'best_strategy': max(avg_density, key=lambda k: avg_density[k]
                             if not np.isnan(avg_density[k]) else -1e10),
    }
    print(f"\n  Avg MI density: {avg_density}")
    print(f"  Best strategy: {results['summary']['best_strategy']}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser(
        'C17: Information-Theoretic Feature Selection Comparison')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output,
                            'c17_information_theoretic_decomposition', args.dataset)
