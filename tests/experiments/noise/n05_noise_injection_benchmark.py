#!/usr/bin/env python3
"""N05: Noise Injection Benchmark — all 15 binary detectors × noise levels × distance spaces.

Paper: Noise
Systematically evaluates every noise detection method in noise_mitigation_methods.py
on synthetically noisy data. Tests in concat, hybrid, and per-rep distance spaces for
the 9 distance-aware methods; non-distance methods are run on each feature space.

Usage:
    python n05_noise_injection_benchmark.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_fitted_hybrid, extract_top3_base_reps,
    cap_dimensions,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)
from kirby.utils.noise_mitigation_methods import (
    get_mitigation_method, compute_distance_matrix,
    DISTANCE_AWARE_METHODS,
)


# All 15 registered method names
ALL_METHODS = [
    'knn_k5', 'knn_k10', 'knn_k20',
    'lof', 'lof_k10', 'lof_k20',
    'distance_weighted', 'neighborhood_consensus', 'activity_cliffs',
    'mahalanobis', 'cv_disagreement', 'bootstrap_ensemble',
    'co_teaching', 'dividemix',
    'zscore', 'prediction_error',
]


def _inject_noise(y, sigma_frac, seed=42, is_clf=False):
    """Inject noise. Regression: Gaussian additive. Classification: label flipping."""
    rng = np.random.RandomState(seed)
    n = len(y)
    y_noisy = y.copy()

    if is_clf:
        # Classification: flip labels for sigma_frac fraction of samples
        n_flip = max(1, int(n * sigma_frac))
        flip_idx = rng.choice(n, n_flip, replace=False)
        classes = np.unique(y)
        for i in flip_idx:
            other = classes[classes != y[i]]
            if len(other) > 0:
                y_noisy[i] = rng.choice(other)
        noise_mask = np.zeros(n, dtype=bool)
        noise_mask[flip_idx] = True
    else:
        y_range = np.max(y) - np.min(y)
        noise = rng.normal(0, sigma_frac * y_range, n)
        y_noisy = y + noise
        # Ground truth: samples where |noise| > MAD(y) are "truly noisy"
        mad = float(np.median(np.abs(y - np.median(y))))
        noise_mask = np.abs(noise) > mad

    return y_noisy, noise_mask


def run(data, config, output_dir, dataset_name, **kwargs):
    """N05: Noise injection benchmark."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N05: NOISE INJECTION BENCHMARK")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    augmentations = data.get('augmentations')
    task_type = data.get('task_type', 'regression')
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    sigmas = kwargs.get('sigmas', [0.05, 0.1, 0.2, 0.3])
    budget = kwargs.get('budget', 100)

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    X_concat_s = StandardScaler().fit_transform(X_concat)

    # Precompute distance matrices
    D_concat = compute_distance_matrix(X_concat, standardize=True)

    D_hybrid = None
    X_hybrid = None
    _search = None
    groups = data.get('groups')
    try:
        X_hybrid, fi, _search = get_fitted_hybrid(
            reps, labels, groups, budget=budget,
            aug_condition='uncapped', augmentations=augmentations,
            speed=config.get('search_speed', 'full'),
            precomputed=data.get('precomputed_searches'),
            internal_evaluators=DISTANCE_EVALUATORS,
            budgets=DISTANCE_BUDGETS,
            downstream_model='knn',
            dataset_name=dataset_name, tier='general',
        )
        D_hybrid = compute_distance_matrix(X_hybrid, standardize=True)
        print(f"  Hybrid distance matrix: {X_hybrid.shape[1]}D")
        top3 = extract_top3_base_reps(_search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    except Exception as e:
        print(f"  Hybrid failed: {e} — distance-aware methods will use concat only")

    # Per-rep distance matrices and standardised features
    per_rep_D = {}
    per_rep_Xs = {}
    for rep_name, X_rep in reps.items():
        per_rep_Xs[rep_name] = cap_dimensions(StandardScaler().fit_transform(X_rep))[0]
        per_rep_D[rep_name] = compute_distance_matrix(X_rep, standardize=True)
    print(f"  Per-rep distance matrices: {list(per_rep_D.keys())}")

    print(f"  Sigmas: {sigmas}")
    print(f"  Methods: {len(ALL_METHODS)}")

    top3 = extract_top3_base_reps(_search) if _search is not None else None
    results = {
        'metric': metric_name,
        'sigmas': sigmas,
        'methods': ALL_METHODS,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        'per_sigma': {},
    }

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        y_noisy, noise_mask = _inject_noise(labels, sigma, is_clf=is_clf)
        n_noisy = int(noise_mask.sum())
        print(f"    {n_noisy}/{len(labels)} truly noisy samples")

        sigma_results = {'n_noisy': n_noisy}

        for method_name in ALL_METHODS:
            is_distance_aware = method_name in DISTANCE_AWARE_METHODS

            # Run with concat distance
            try:
                kwargs_m = {}
                if is_distance_aware:
                    kwargs_m['precomputed_distances'] = D_concat

                detector = get_mitigation_method(method_name, **kwargs_m)
                flagged_idx = detector.identify_noise(X_concat_s, y_noisy)
                flagged_mask = np.zeros(len(labels), dtype=bool)
                flagged_mask[flagged_idx] = True

                tp = int((flagged_mask & noise_mask).sum())
                fp = int((flagged_mask & ~noise_mask).sum())
                fn = int((~flagged_mask & noise_mask).sum())
                n_flagged = int(flagged_mask.sum())

                precision = tp / max(n_flagged, 1)
                recall = tp / max(n_noisy, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-10)

                entry = {
                    'concat': {
                        'n_flagged': n_flagged,
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                    },
                }
            except Exception as e:
                entry = {'concat': {'error': str(e)}}

            # Run with hybrid distance (distance-aware methods only)
            if is_distance_aware and D_hybrid is not None:
                try:
                    X_h_s = StandardScaler().fit_transform(X_hybrid)
                    detector_h = get_mitigation_method(
                        method_name, precomputed_distances=D_hybrid)
                    flagged_h = detector_h.identify_noise(X_h_s, y_noisy)
                    flagged_h_mask = np.zeros(len(labels), dtype=bool)
                    flagged_h_mask[flagged_h] = True

                    tp_h = int((flagged_h_mask & noise_mask).sum())
                    n_flagged_h = int(flagged_h_mask.sum())
                    precision_h = tp_h / max(n_flagged_h, 1)
                    recall_h = tp_h / max(n_noisy, 1)
                    f1_h = 2 * precision_h * recall_h / max(precision_h + recall_h, 1e-10)

                    entry['hybrid'] = {
                        'n_flagged': n_flagged_h,
                        'precision': float(precision_h),
                        'recall': float(recall_h),
                        'f1': float(f1_h),
                    }
                except Exception as e:
                    entry['hybrid'] = {'error': str(e)}

            # Run with per-rep distance matrices
            for rep_name in per_rep_D:
                try:
                    kwargs_r = {}
                    if is_distance_aware:
                        kwargs_r['precomputed_distances'] = per_rep_D[rep_name]

                    detector_r = get_mitigation_method(method_name, **kwargs_r)
                    flagged_r = detector_r.identify_noise(per_rep_Xs[rep_name], y_noisy)
                    flagged_r_mask = np.zeros(len(labels), dtype=bool)
                    flagged_r_mask[flagged_r] = True

                    tp_r = int((flagged_r_mask & noise_mask).sum())
                    n_flagged_r = int(flagged_r_mask.sum())
                    precision_r = tp_r / max(n_flagged_r, 1)
                    recall_r = tp_r / max(n_noisy, 1)
                    f1_r = 2 * precision_r * recall_r / max(precision_r + recall_r, 1e-10)

                    entry[rep_name] = {
                        'n_flagged': n_flagged_r,
                        'precision': float(precision_r),
                        'recall': float(recall_r),
                        'f1': float(f1_r),
                    }
                except Exception as e:
                    entry[rep_name] = {'error': str(e)}

            sigma_results[method_name] = entry

            # Print compact summary
            c = entry.get('concat', {})
            h = entry.get('hybrid', {})
            c_str = f"F1={c.get('f1', 0):.3f}" if 'f1' in c else c.get('error', '?')[:20]
            h_str = f"F1={h.get('f1', 0):.3f}" if 'f1' in h else ''
            rep_strs = [f"{rn}:F1={entry[rn].get('f1', 0):.3f}"
                        for rn in per_rep_D if rn in entry and 'f1' in entry[rn]]
            rep_summary = '  '.join(rep_strs[:3])  # show first 3 to avoid line overflow
            if len(rep_strs) > 3:
                rep_summary += f"  (+{len(rep_strs)-3} more)"
            print(f"    {method_name:<25} concat: {c_str}" +
                  (f"  hybrid: {h_str}" if h_str else "") +
                  (f"  {rep_summary}" if rep_summary else ""))

        results['per_sigma'][str(sigma)] = convert_numpy(sigma_results)

    # Summary: best method per distance space (including per-rep spaces)
    all_spaces = ['concat', 'hybrid'] + list(per_rep_D.keys())
    summary = {}
    for space in all_spaces:
        best_f1 = -1
        best_method = None
        for sigma_key, sigma_res in results['per_sigma'].items():
            for method_name in ALL_METHODS:
                f1 = sigma_res.get(method_name, {}).get(space, {}).get('f1', -1)
                if f1 > best_f1:
                    best_f1 = f1
                    best_method = method_name
        summary[space] = {'best_method': best_method, 'best_f1': float(best_f1)}

    results['summary'] = summary
    print(f"\n  Best: concat={summary['concat']}, hybrid={summary.get('hybrid', 'N/A')}")
    for rep_name in per_rep_D:
        print(f"        {rep_name}={summary[rep_name]}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N05: Noise Injection Benchmark')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n05_noise_injection_benchmark',
                            args.dataset)
