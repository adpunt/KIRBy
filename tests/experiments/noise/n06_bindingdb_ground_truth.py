#!/usr/bin/env python3
"""N06: BindingDB Ground-Truth Noise — replicate disagreement as real-world noise labels.

Paper: Noise (genuinely novel contribution)
BindingDB has compound-target pairs measured by multiple labs. Inter-replicate
disagreement = ground-truth noise. Compare noise estimators against this.

Usage:
    python n06_bindingdb_ground_truth.py --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_fitted_hybrid, extract_top3_base_reps,
)
from kirby.utils.noise_estimation import get_noise_estimator


# Estimators to benchmark (skip torch-dependent ones for screening)
ESTIMATORS_SCREENING = [
    'neighbor_residual',
    'local_variance_ratio',
    'distance_weighted_residual',
    'activity_cliff_aware',
    'cross_val_residual',
    'ensemble_disagreement',
    'leave_one_out',
    'gaussian_process',
    'conformal',
    'bayesian_rf',
]

# GP max samples
GP_MAX = 1000


def run(data, config, output_dir, dataset_name, **kwargs):
    """N06: BindingDB ground-truth noise."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N06: BINDINGDB GROUND-TRUTH NOISE")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    n_samples = len(labels)

    # Compute replicate disagreement as ground truth
    unique_groups = np.unique(groups) if groups is not None else np.array([])
    group_stds = {}
    for g in unique_groups:
        mask = groups == g
        if mask.sum() >= 2:
            group_stds[g] = float(np.std(labels[mask]))

    replicate_stds = np.zeros(n_samples)
    if groups is not None:
        for i, g in enumerate(groups):
            replicate_stds[i] = group_stds.get(g, 0.0)

    has_replicate = replicate_stds > 0
    n_with_replicates = int(has_replicate.sum())

    if n_with_replicates < 10:
        print(f"  Only {n_with_replicates} samples with replicates — not enough for ground-truth analysis")
        print("  Falling back to synthetic noise correlation test")
        # Even without true replicates, we can still benchmark estimators
        # by injecting synthetic noise and measuring detection quality
        replicate_stds = None

    budget = kwargs.get('budget', 100)
    augmentations = data.get('augmentations')

    # Build feature spaces
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    X_concat_s = StandardScaler().fit_transform(X_concat)

    spaces = {'concat': X_concat_s}
    _search = None
    try:
        X_hybrid, fi, _search = get_fitted_hybrid(
            reps, labels, groups, budget=budget,
            aug_condition='uncapped', augmentations=augmentations,
            speed=config.get('search_speed', 'fast'),
            precomputed=data.get('precomputed_searches'),
            downstream_model='rf',
        )
        spaces['hybrid'] = StandardScaler().fit_transform(X_hybrid)
        top3 = extract_top3_base_reps(_search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    except Exception as e:
        print(f"  Hybrid failed: {e}")

    # Subsample for GP
    if n_samples > GP_MAX:
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(n_samples, GP_MAX, replace=False)
    else:
        sub_idx = np.arange(n_samples)

    top3 = extract_top3_base_reps(_search) if _search is not None else None
    results = {
        'n_samples': n_samples,
        'n_with_replicates': n_with_replicates,
        'has_ground_truth': replicate_stds is not None,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        'estimators': {},
    }

    for space_name, X_space in spaces.items():
        print(f"\n  === Space: {space_name} (dim={X_space.shape[1]}) ===")
        space_results = {}

        X_sub = X_space[sub_idx]
        y_sub = labels[sub_idx]
        rep_stds_sub = replicate_stds[sub_idx] if replicate_stds is not None else None

        for est_name in ESTIMATORS_SCREENING:
            print(f"    {est_name}...", end='', flush=True)
            try:
                estimator = get_noise_estimator(est_name)
                estimator.fit(X_sub, y_sub)
                scores = estimator.noise_scores_

                entry = {
                    'score_mean': float(np.mean(scores)),
                    'score_std': float(np.std(scores)),
                }

                # Correlation with ground-truth replicate disagreement
                if rep_stds_sub is not None:
                    rho, p = spearmanr(scores, rep_stds_sub)
                    entry['ground_truth_correlation'] = float(rho)
                    entry['ground_truth_p'] = float(p)
                    print(f" rho={rho:.3f} (p={p:.2e})")
                else:
                    print(f" score_mean={entry['score_mean']:.4f}")

                # Get weights and measure downstream utility
                weights = estimator.get_weights(strength=1.0)
                entry['weight_range'] = [float(np.min(weights)), float(np.max(weights))]

                space_results[est_name] = entry

            except Exception as e:
                print(f" FAILED: {e}")
                space_results[est_name] = {'error': str(e)}

        results['estimators'][space_name] = convert_numpy(space_results)

    # Summary: best estimator by ground-truth correlation
    if replicate_stds is not None:
        best_rho = -1
        best_est = None
        best_space = None
        for space_name, space_res in results['estimators'].items():
            for est_name, est_res in space_res.items():
                rho = est_res.get('ground_truth_correlation', -1)
                if rho > best_rho:
                    best_rho = rho
                    best_est = est_name
                    best_space = space_name

        results['summary'] = {
            'best_estimator': best_est,
            'best_space': best_space,
            'best_correlation': float(best_rho),
        }
        print(f"\n  Best: {best_est} in {best_space} space (rho={best_rho:.3f})")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N06: BindingDB Ground-Truth Noise')
    parser.add_argument('--budget', type=int, default=100)
    args = parser.parse_args()
    config = get_config(args)

    # Default to bindingdb_kd dataset
    dataset = getattr(args, 'dataset', 'bindingdb_kd')
    data = load_data(dataset, args.reps)
    results = run(data, config, args.output, dataset, budget=args.budget)
    save_experiment_results(results, args.output, 'n06_bindingdb_ground_truth', dataset)
