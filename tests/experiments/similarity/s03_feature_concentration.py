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
    get_fitted_hybrid, flatten_weights,
    estimate_mutual_information_knn, compute_effective_rank,
)

from kirby.hybrid import compute_feature_importance


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

    # Get hybrid selection
    budget = kwargs.get('budget', 100)
    augmentations = data.get('augmentations')
    try:
        X_hybrid, fi, search = get_fitted_hybrid(
            reps, labels, groups,
            aug_condition='uncapped' if augmentations else 'none', augmentations=augmentations,
            speed=search_speed,
            precomputed=data.get('precomputed_searches'),
            downstream_model='rf',
            dataset_name=dataset_name, tier='general', task_type=task_type)
        fitted = search.fitted_results.get('rf')
        weights_flat = flatten_weights(fitted) if fitted else np.ones(X_hybrid.shape[1])
    except Exception as e:
        print(f"  find_best_hybrid failed: {e}")
        return {'error': str(e)}

    rng = np.random.RandomState(42)

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
            mi_full = estimate_mutual_information_knn(X_scaled_full, labels, k=5)
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
                    mi_sel = estimate_mutual_information_knn(
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
                    mi_rand = estimate_mutual_information_knn(
                        StandardScaler().fit_transform(X_rand), labels, k=5)
                except Exception:
                    mi_rand = float('nan')
                rep_result['mi_random'] = mi_rand
                rep_result['effective_rank_random'] = compute_effective_rank(X_rand)

                # PCA to same dim
                n_pca_comp = min(n_selected, n_samples - 1)
                X_scaled_rep = StandardScaler().fit_transform(X)
                if n_features > 2000:
                    from sklearn.decomposition import TruncatedSVD
                    X_pca_match = TruncatedSVD(n_components=n_pca_comp, random_state=42).fit_transform(X_scaled_rep)
                else:
                    X_pca_match = PCA(n_components=n_pca_comp).fit_transform(X_scaled_rep)
                try:
                    mi_pca = estimate_mutual_information_knn(X_pca_match, labels, k=5)
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

        results['per_rep'][rep_name] = rep_result

    # Gini coefficient of importance scores per method
    for wm in weight_methods:
        print(f"\n  Gini coefficients ({wm}):")
        try:
            importance = compute_feature_importance(reps, labels, method=wm)
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
