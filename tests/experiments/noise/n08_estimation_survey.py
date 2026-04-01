#!/usr/bin/env python3
"""N08: Noise Estimation Survey — benchmark all 15 estimators from noise_estimation.py.

Paper: Noise
Exercises the entire dormant noise_estimation library. For each estimator:
score distributions, cross-estimator correlations, downstream impact via
sample weighting, and cliff discrimination.

Usage:
    python n08_estimation_survey.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from scipy.stats import spearmanr, skew
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_fitted_hybrid,
)
from kirby.utils.noise_estimation import get_noise_estimator
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


# All estimators to test (skip torch-dependent for screening)
ESTIMATORS = [
    'neighbor_residual',
    'local_variance_ratio',
    'distance_weighted_residual',
    'activity_cliff_aware',
    'cross_val_residual',
    'ensemble_disagreement',
    'leave_one_out',
    'hybrid_noise',
    'gaussian_process',
    'conformal',
    'bayesian_rf',
]

# Torch-dependent (try-import)
TORCH_ESTIMATORS = [
    'heteroscedastic_nn',
    'deep_ensemble',
    'quantile_regression',
]

# Optional
OPTIONAL_ESTIMATORS = [
    'ngboost',
]

GP_MAX = 1000


def run(data, config, output_dir, dataset_name, **kwargs):
    """N08: Noise estimation survey."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N08: NOISE ESTIMATION SURVEY")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    task_type = data.get('task_type', 'regression')
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    budget = kwargs.get('budget', 100)
    n_samples = len(labels)

    # Build feature spaces
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    X_concat_s = StandardScaler().fit_transform(X_concat)

    spaces = {'concat': X_concat_s}
    try:
        X_hybrid, fi, _search = get_fitted_hybrid(
            reps, labels, groups, budget=budget,
            aug_condition='uncapped', augmentations=augmentations,
            speed=config.get('search_speed', 'fast'),
            precomputed=data.get('precomputed_searches'),
        )
        spaces['hybrid'] = StandardScaler().fit_transform(X_hybrid)
    except Exception as e:
        print(f"  Hybrid failed: {e}")

    # Per-rep spaces
    for rep_name, X_rep in reps.items():
        spaces[rep_name] = StandardScaler().fit_transform(X_rep)

    # Subsample for GP-based estimators
    if n_samples > GP_MAX:
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(n_samples, GP_MAX, replace=False)
    else:
        sub_idx = np.arange(n_samples)

    # Identify cliff compounds for cliff discrimination test
    D = compute_distance_matrix(X_concat, standardize=True)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)
    if is_clf:
        delta_y = (labels[:, None] != labels[None, :]).astype(float)
        cliff_mask = ((sim >= 0.55) & (delta_y > 0.5)).any(axis=1)
    else:
        mad_y = float(np.median(np.abs(labels - np.median(labels))))
        delta_y = np.abs(labels[:, None] - labels[None, :])
        cliff_mask = ((sim >= 0.55) & (delta_y > 2.0 * mad_y)).any(axis=1)
    # Subsample cliff mask
    cliff_mask_sub = cliff_mask[sub_idx]
    n_cliffs = int(cliff_mask_sub.sum())
    print(f"  Cliffs: {n_cliffs}/{len(sub_idx)}")

    # Determine which estimators to test
    all_estimators = list(ESTIMATORS)
    try:
        import torch
        all_estimators.extend(TORCH_ESTIMATORS)
    except ImportError:
        print("  PyTorch not available — skipping torch-dependent estimators")
    for opt in OPTIONAL_ESTIMATORS:
        try:
            get_noise_estimator(opt)
            all_estimators.append(opt)
        except Exception:
            pass

    print(f"  Testing {len(all_estimators)} estimators × {len(spaces)} spaces")

    results = {
        'metric': metric_name,
        'n_samples': n_samples,
        'n_subsampled': len(sub_idx),
        'n_cliffs': n_cliffs,
        'estimators': all_estimators,
        'per_space': {},
    }

    all_scores_by_space = {}  # for cross-correlation

    for space_name, X_full in spaces.items():
        print(f"\n  === Space: {space_name} (dim={X_full.shape[1]}) ===")
        X_sub = X_full[sub_idx]
        y_sub = labels[sub_idx]
        space_results = {}
        space_scores = {}

        for est_name in all_estimators:
            print(f"    {est_name}...", end='', flush=True)
            t0 = time.time()

            try:
                estimator = get_noise_estimator(est_name)
                estimator.fit(X_sub, y_sub)
                scores = estimator.noise_scores_
                elapsed = time.time() - t0

                entry = {
                    'time_s': float(elapsed),
                    'score_mean': float(np.mean(scores)),
                    'score_std': float(np.std(scores)),
                    'score_skew': float(skew(scores)),
                    'score_min': float(np.min(scores)),
                    'score_max': float(np.max(scores)),
                }

                # Cliff discrimination: Spearman(score, is_cliff)
                # Good estimators should NOT correlate positively (cliffs aren't noise)
                if n_cliffs >= 5:
                    rho_cliff, p_cliff = spearmanr(scores, cliff_mask_sub.astype(float))
                    entry['cliff_correlation'] = float(rho_cliff)
                    entry['cliff_p'] = float(p_cliff)
                    entry['cliff_discriminates'] = rho_cliff < 0.1  # not flagging cliffs

                # Sample weights
                try:
                    weights = estimator.get_weights(strength=1.0)
                    entry['weight_mean'] = float(np.mean(weights))
                    entry['weight_std'] = float(np.std(weights))
                except Exception:
                    pass

                space_scores[est_name] = scores
                space_results[est_name] = entry
                cliff_str = f", cliff_rho={entry.get('cliff_correlation', '?'):.3f}" if 'cliff_correlation' in entry else ""
                print(f" {elapsed:.1f}s, mean={entry['score_mean']:.4f}{cliff_str}")

            except Exception as e:
                print(f" FAILED: {e}")
                space_results[est_name] = {'error': str(e)}

        # Cross-estimator correlations
        est_names_ok = [e for e in all_estimators if e in space_scores]
        if len(est_names_ok) >= 2:
            corr_matrix = {}
            for i, e1 in enumerate(est_names_ok):
                for e2 in est_names_ok[i+1:]:
                    rho, _ = spearmanr(space_scores[e1], space_scores[e2])
                    corr_matrix[f'{e1}_vs_{e2}'] = float(rho)

            space_results['_cross_correlations'] = corr_matrix
            # Find estimator clusters (high mutual correlation)
            avg_corr = {}
            for e in est_names_ok:
                rhos = [v for k, v in corr_matrix.items() if e in k]
                avg_corr[e] = float(np.mean(rhos)) if rhos else 0.0
            space_results['_avg_cross_correlation'] = avg_corr

        results['per_space'][space_name] = convert_numpy(space_results)
        all_scores_by_space[space_name] = space_scores

    # Summary
    summary = {}
    for space_name in spaces:
        sr = results['per_space'].get(space_name, {})
        # Best cliff-discriminating estimator (lowest cliff correlation)
        best_cliff_est = None
        best_cliff_rho = 1.0
        for est_name in all_estimators:
            rho = sr.get(est_name, {}).get('cliff_correlation', 1.0)
            if isinstance(rho, (int, float)) and rho < best_cliff_rho:
                best_cliff_rho = rho
                best_cliff_est = est_name
        summary[space_name] = {
            'best_cliff_discriminator': best_cliff_est,
            'cliff_correlation': float(best_cliff_rho),
            'n_estimators_ok': sum(1 for e in all_estimators if 'error' not in sr.get(e, {})),
        }

    results['summary'] = summary
    print(f"\n  Summary: {summary}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N08: Noise Estimation Survey')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n08_estimation_survey', args.dataset)
