#!/usr/bin/env python3
"""S03: Feature Concentration Analysis -- WHY hybrid similarity works.

Paper: Similarity
Conclusion: Feature importance concentrates MI in selected dimensions.
Dependencies: S02 (reads JSON, optional).

For each rep: computes MI of full, selected, random, PCA-matched subsets.
Measures MI density ratio (selected/full) and Gini coefficient of importance scores.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results,
    get_fitted_hybrid, get_fitted_pool_hybrid,
    estimate_mutual_information_knn, compute_effective_rank,
    FULL_BUDGETS,
)

from kirby.hybrid import compute_feature_importance


# COST / LARGE-N GATE (2026-06-08, RESULTS-AFFECTING, reversible).
#   The kNN mutual-information estimator (estimate_mutual_information_knn) does a
#   per-feature Kraskov neighbour search whose cost grows with the number of
#   rows. On the n=30k–100k DTA / nucleotide datasets this is the wall-buster
#   for s03 (every full/selected/random/PCA arm calls it). The KSG family
#   converges to a stable estimate well before ~2000 samples (Kraskov et al.
#   2004; the estimator is consistent and low-variance once n exceeds a few
#   hundred), so a fixed seeded row subsample gives a statistically equivalent
#   MI value while making s03 cost independent of n.
#   Notes:
#     - This is a MI-ESTIMATION subsample only. It does NOT touch any train/test
#       split (s03 has no CV — it reports MI/density/effective-rank summaries),
#       so it cannot introduce cross-group leakage. effective_rank is still
#       computed on the FULL matrix.
#     - Deterministic (RandomState(42)), so reruns are identical and the
#       subsample does not depend on group structure.
#     - REVERT: set MI_MAX_ROWS = None (or delete _mi_knn_capped and call
#       estimate_mutual_information_knn directly).
MI_MAX_ROWS = 2000


def _mi_knn_capped(X, y, k=5, max_rows=MI_MAX_ROWS, seed=42):
    """estimate_mutual_information_knn on a seeded row subsample (large-n gate).

    Returns the MI on a fixed random subset of <= max_rows rows. Below the cap
    every row is used (identical to calling the estimator directly).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if max_rows is not None and X.shape[0] > max_rows:
        idx = np.random.RandomState(seed).choice(X.shape[0], max_rows, replace=False)
        X, y = X[idx], y[idx]
    return estimate_mutual_information_knn(X, y, k=k)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Explain WHY hybrid similarity works: higher MI density, better effective
    rank distribution in selected features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S03: FEATURE CONCENTRATION ANALYSIS")
    print("=" * 70)

    weight_methods = kwargs.get('weight_methods', ['lightgbm_gain'])
    search_speed = config.get('search_speed', 'full')
    task_type = data['task_type']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'per_rep': {}, 'gini': {}}

    # Sweep budgets × downstream models. Scoped to ['rf', 'lightgbm', 'mlp'] —
    # the full DOWNSTREAM_MODELS set would explode result size with little gain.
    budgets = kwargs.get('budgets', config.get('budgets', FULL_BUDGETS))
    downstream_models = kwargs.get('downstream_models', ['rf', 'lightgbm', 'mlp'])
    augmentations = data.get('augmentations')

    rng = np.random.RandomState(42)

    for downstream_model in downstream_models:
        for budget in budgets:
            print(f"\n  ── model={downstream_model}, budget={budget} ──")
            try:
                X_hybrid, fi, search, _meta = get_fitted_pool_hybrid(
                    reps, labels, groups,
                    downstream_model=downstream_model, budget=budget,
                    augmentations=augmentations,
                    speed=search_speed, data=data,
                    dataset_name=dataset_name, task_type=task_type)
                fitted = search.fitted_results.get(downstream_model)
            except Exception as e:
                print(f"  find_best_hybrid failed: {e}")
                results['per_rep'][(downstream_model, budget)] = {'error': str(e)}
                continue

            per_rep_results = {}
            for rep_name, X in reps.items():
                print(f"\n  {rep_name} (dim={X.shape[1]}):")
                rep_result = {}

                n_features = X.shape[1]
                n_samples = X.shape[0]

                # Full rep MI — let estimate_mutual_information_knn handle PCA
                # internally (max_dim=50, capped at n_samples//5). This ensures
                # consistent dimensionality treatment across full/selected/random.
                X_scaled_full = StandardScaler().fit_transform(X)
                try:
                    mi_full = _mi_knn_capped(X_scaled_full, labels, k=5)  # large-n row cap
                except Exception:
                    mi_full = float('nan')
                rep_result['mi_full'] = mi_full
                rep_result['effective_rank_full'] = compute_effective_rank(X)

                # Selected features for this rep
                if fitted and rep_name in fitted.selected_indices:
                    sel_idx = fitted.selected_indices[rep_name]
                    n_selected = len(sel_idx)

                    if n_selected > 0:
                        X_sel = X[:, sel_idx]

                        # MI of selected
                        try:
                            mi_sel = _mi_knn_capped(
                                StandardScaler().fit_transform(X_sel), labels, k=5)
                        except Exception:
                            mi_sel = float('nan')
                        rep_result['mi_selected'] = mi_sel
                        rep_result['n_selected'] = n_selected
                        rep_result['effective_rank_selected'] = compute_effective_rank(X_sel)

                        # Random subset of same size
                        rand_idx = rng.choice(n_features, n_selected, replace=False)
                        X_rand = X[:, rand_idx]
                        try:
                            mi_rand = _mi_knn_capped(
                                StandardScaler().fit_transform(X_rand), labels, k=5)
                        except Exception:
                            mi_rand = float('nan')
                        rep_result['mi_random'] = mi_rand
                        rep_result['effective_rank_random'] = compute_effective_rank(X_rand)

                        # PCA to same dim — cap at min(50, n//5) so the KSG
                        # estimator's internal PCA does not further reduce
                        # this arm. Matches the implicit cap applied to the
                        # full/selected/random arms inside the helper.
                        ksg_cap = min(50, max(1, n_samples // 5))
                        n_pca_comp = min(n_selected, n_samples - 1, ksg_cap)
                        X_scaled_rep = StandardScaler().fit_transform(X)
                        if n_features > 2000:
                            from sklearn.decomposition import TruncatedSVD
                            X_pca_match = TruncatedSVD(n_components=n_pca_comp, random_state=42).fit_transform(X_scaled_rep)
                        else:
                            X_pca_match = PCA(n_components=n_pca_comp, random_state=42).fit_transform(X_scaled_rep)
                        try:
                            mi_pca = _mi_knn_capped(X_pca_match, labels, k=5)
                        except Exception:
                            mi_pca = float('nan')
                        rep_result['mi_pca'] = mi_pca
                        rep_result['effective_rank_pca'] = compute_effective_rank(X_pca_match)

                        # MI density = MI / n_features
                        rep_result['mi_density_full'] = mi_full / n_features if n_features > 0 else 0
                        rep_result['mi_density_selected'] = mi_sel / n_selected if n_selected > 0 else 0
                        rep_result['mi_density_ratio'] = (
                            rep_result['mi_density_selected'] / rep_result['mi_density_full']
                            if rep_result['mi_density_full'] > 0 else float('nan')
                        )

                        print(f"    MI full={mi_full:.4f} ({n_features}d), "
                              f"selected={mi_sel:.4f} ({n_selected}d), "
                              f"random={mi_rand:.4f}, pca={mi_pca:.4f}")
                        print(f"    MI density ratio (selected/full): "
                              f"{rep_result['mi_density_ratio']:.2f}x")
                    else:
                        print(f"    No features selected from this rep")
                else:
                    print(f"    Not in hybrid selection")

                per_rep_results[rep_name] = rep_result

            results['per_rep'][(downstream_model, budget)] = per_rep_results

    # Gini coefficient of importance scores per method
    for wm in weight_methods:
        print(f"\n  Gini coefficients ({wm}):")
        try:
            importance = compute_feature_importance(reps, labels, method=wm, task_type=task_type)
            for rep_name, scores in importance.items():
                scores_sorted = np.sort(np.abs(scores))
                n = len(scores_sorted)
                if n == 0 or scores_sorted.sum() == 0:
                    gini = 0.0
                else:
                    index = np.arange(1, n + 1)
                    gini = float((2 * np.sum(index * scores_sorted) /
                                 (n * np.sum(scores_sorted))) - (n + 1) / n)
                results['gini'].setdefault(wm, {})[rep_name] = gini
                print(f"    {rep_name}: Gini={gini:.3f}")
        except Exception as e:
            print(f"    {wm} failed: {e}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S03: Feature Concentration Analysis',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's03_feature_concentration', args.dataset)
