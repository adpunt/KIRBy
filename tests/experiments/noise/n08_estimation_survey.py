#!/usr/bin/env python3
"""N08: Noise Estimation Survey — benchmark estimators from noise_estimation.py.

Paper: Noise
Exercises the noise_estimation library. For each estimator: score
distributions, cross-estimator correlations, and cliff discrimination
(noise scores should be roughly independent of cliff status — both strong
positive and strong negative correlation are failure modes).

Note: the file's earlier "downstream impact via sample weighting" claim
was never implemented and has been removed from the docstring. Only
cliff discrimination + cross-estimator correlations + score statistics
are reported. The number of estimators varies based on which optional
dependencies (ngboost, torch) are present.

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

    # GP-only subsample. Most estimators run cheaply on the full dataset;
    # GP-based ones (`gaussian_process`, `bayesian_rf`, `conformal`) are
    # O(n²)-O(n³) in memory/time and need a smaller subset. Previously a
    # single `sub_idx` was applied to every estimator, throwing away 97% of
    # data even for cheap methods on a 30K-sample dataset. Now each
    # estimator pulls its own X/y subset based on whether it's GP-class.
    GP_ESTIMATORS = {'gaussian_process', 'bayesian_rf', 'conformal'}
    if n_samples > GP_MAX:
        rng = np.random.RandomState(42)
        gp_sub_idx = rng.choice(n_samples, GP_MAX, replace=False)
    else:
        gp_sub_idx = np.arange(n_samples)
    full_idx = np.arange(n_samples)
    # The downstream loop uses `sub_idx` for the cliff-mask / printing; that
    # subset corresponds to whatever the estimator actually ran on, so we
    # set it per-estimator below. For now define it as the GP subset for
    # compatibility with the rest of the function — cliff_mask_sub is
    # computed from sub_idx and is shared across estimators.
    sub_idx = gp_sub_idx

    # Identify cliff compounds for cliff discrimination test. The pairwise
    # distance matrix is O(n²) memory — on 30K samples that's ~7GB. Cap the
    # cliff computation to the GP subsample (≤GP_MAX) and use that mask for
    # all estimators. Estimators that ran on the full data will have their
    # cliff_correlation computed against the subset rather than the full N.
    X_for_cliffs = X_concat[gp_sub_idx]
    labels_for_cliffs = labels[gp_sub_idx]
    D = compute_distance_matrix(X_for_cliffs, standardize=True)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)
    if is_clf:
        delta_y = (labels_for_cliffs[:, None] != labels_for_cliffs[None, :]).astype(float)
        cliff_mask_subset = ((sim >= 0.55) & (delta_y > 0.5)).any(axis=1)
    else:
        mad_y = float(np.median(np.abs(labels_for_cliffs - np.median(labels_for_cliffs))))
        delta_y = np.abs(labels_for_cliffs[:, None] - labels_for_cliffs[None, :])
        cliff_mask_subset = ((sim >= 0.55) & (delta_y > 2.0 * mad_y)).any(axis=1)
    # Project the subset-cliff-mask back onto the full-index space: False
    # everywhere except at gp_sub_idx positions. Per-estimator code below
    # subsets this with cliff_mask[est_idx]; for full-index estimators we
    # only have cliff information for the gp_sub_idx ⊂ full_idx samples.
    cliff_mask = np.zeros(n_samples, dtype=bool)
    cliff_mask[gp_sub_idx] = cliff_mask_subset
    # Backwards-compat alias for code that still references cliff_mask_sub.
    cliff_mask_sub = cliff_mask[sub_idx]
    n_cliffs = int(cliff_mask_sub.sum())
    print(f"  Cliffs: {n_cliffs}/{len(sub_idx)} (computed on GP subsample of size {len(gp_sub_idx)})")

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
        space_results = {}
        space_scores = {}

        for est_name in all_estimators:
            print(f"    {est_name}...", end='', flush=True)
            t0 = time.time()

            # Per-estimator subsample: GP-class estimators use the small
            # subset; everything else sees the full data. Cross-estimator
            # correlations below will only compare estimators that ran on
            # the same indices (i.e. same length scores arrays).
            est_idx = gp_sub_idx if est_name in GP_ESTIMATORS else full_idx
            X_sub = X_full[est_idx]
            y_sub = labels[est_idx]

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
                # Per-estimator subset of the cliff mask (estimators may now
                # have ran on different index subsets).
                cliff_mask_est = cliff_mask[est_idx]
                if int(cliff_mask_est.sum()) >= 5:
                    rho_cliff, p_cliff = spearmanr(scores, cliff_mask_est.astype(float))
                    entry['cliff_correlation'] = float(rho_cliff)
                    entry['cliff_abs_correlation'] = float(abs(rho_cliff))
                    entry['cliff_p'] = float(p_cliff)
                    entry['cliff_discriminates'] = abs(rho_cliff) < 0.1
                    entry['cliff_n_used'] = int(cliff_mask_est.sum())

                # Sample weights
                try:
                    weights = estimator.get_weights(strength=1.0)
                    entry['weight_mean'] = float(np.mean(weights))
                    entry['weight_std'] = float(np.std(weights))
                except Exception:
                    pass

                space_scores[est_name] = scores
                space_results[est_name] = entry
                _cliff_rho = entry.get('cliff_correlation')
                cliff_str = (f", cliff_rho={_cliff_rho:.3f}"
                             if _cliff_rho is not None else "")
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
                    # Only correlate estimators that ran on the same number
                    # of samples (i.e. either both GP-class on the small
                    # subset or both non-GP on the full data). Mismatched
                    # lengths would raise; previously this was impossible
                    # because every estimator used the same subset.
                    if len(space_scores[e1]) != len(space_scores[e2]):
                        continue
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
