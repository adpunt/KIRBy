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

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_fitted_hybrid, get_fitted_pool_hybrid,
    cap_dimensions,
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
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    # metric_name surfaces in results; the fn / direction aren't used here —
    # this experiment correlates noise scores with cliff masks, not against
    # the domain metric. n_folds / mode were set but never read either; the
    # estimator survey runs a single in-sample fit on a fixed subsample by
    # design, so removing them removes a misleading config knob.
    metric_name, _metric_fn, _lower_is_better = get_domain_metric(data)
    budget = kwargs.get('budget', 100)
    n_samples = len(labels)

    # Build feature spaces
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    X_concat_s = StandardScaler().fit_transform(X_concat)

    spaces = {'concat': X_concat_s}
    try:
        X_hybrid, fi, _search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=augmentations,
            speed=config.get('search_speed', 'full'),
            data=data,
            dataset_name=dataset_name,
            task_type=task_type)
        spaces['hybrid'] = StandardScaler().fit_transform(X_hybrid)
    except Exception as e:
        print(f"  Hybrid failed: {e}")

    # Per-rep spaces
    for rep_name, X_rep in reps.items():
        spaces[rep_name] = cap_dimensions(StandardScaler().fit_transform(X_rep))[0]

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

                # Cliff discrimination: |Spearman(score, is_cliff)| should be
                # small. A good estimator's noise score is INDEPENDENT of
                # cliff status (cliffs are real chemistry signal, not noise).
                # Both strong positive correlation (rates cliffs as noisy) AND
                # strong negative correlation (rates cliffs as the cleanest
                # samples) are failure modes; we want |rho| ≈ 0.
                if n_cliffs >= 5:
                    rho_cliff, p_cliff = spearmanr(scores, cliff_mask_sub.astype(float))
                    entry['cliff_correlation'] = float(rho_cliff)
                    entry['cliff_abs_correlation'] = float(abs(rho_cliff))
                    entry['cliff_p'] = float(p_cliff)
                    entry['cliff_discriminates'] = abs(rho_cliff) < 0.1

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

        # Cross-estimator correlations. Keep the pair-level results in two
        # parallel structures: a string-keyed `_cross_correlations` (kept for
        # backwards compat with anything that reads it) and a tuple-keyed
        # internal map used for the aggregation. The aggregation uses
        # tuple-membership lookup rather than substring matching on the
        # string keys — substring matching would silently double-count if a
        # future estimator's name contains another's as a substring (e.g.
        # adding "ensemble" with existing "ensemble_disagreement" /
        # "deep_ensemble").
        est_names_ok = [e for e in all_estimators if e in space_scores]
        if len(est_names_ok) >= 2:
            corr_matrix = {}
            corr_pairs = {}
            for i, e1 in enumerate(est_names_ok):
                for e2 in est_names_ok[i+1:]:
                    rho, _ = spearmanr(space_scores[e1], space_scores[e2])
                    corr_matrix[f'{e1}_vs_{e2}'] = float(rho)
                    corr_pairs[(e1, e2)] = float(rho)

            space_results['_cross_correlations'] = corr_matrix
            avg_corr = {}
            for e in est_names_ok:
                rhos = [v for (a, b), v in corr_pairs.items() if e in (a, b)]
                avg_corr[e] = float(np.mean(rhos)) if rhos else 0.0
            space_results['_avg_cross_correlation'] = avg_corr

        results['per_space'][space_name] = convert_numpy(space_results)
        all_scores_by_space[space_name] = space_scores

    # Summary — rank cliff discrimination by |rho|, smaller is better.
    # (Signed rho would happily pick an estimator with rho=-0.9 as the
    # 'best discriminator', which is just the inverted failure mode.)
    summary = {}
    for space_name in spaces:
        sr = results['per_space'].get(space_name, {})
        best_cliff_est = None
        best_cliff_abs = float('inf')
        best_cliff_signed = None
        for est_name in all_estimators:
            rho = sr.get(est_name, {}).get('cliff_correlation')
            if isinstance(rho, (int, float)) and abs(rho) < best_cliff_abs:
                best_cliff_abs = abs(rho)
                best_cliff_signed = float(rho)
                best_cliff_est = est_name
        summary[space_name] = {
            'best_cliff_discriminator': best_cliff_est,
            'cliff_abs_correlation': (float(best_cliff_abs)
                                      if best_cliff_est else None),
            'cliff_correlation_signed': best_cliff_signed,
            'n_estimators_ok': sum(1 for e in all_estimators
                                   if 'error' not in sr.get(e, {})),
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
