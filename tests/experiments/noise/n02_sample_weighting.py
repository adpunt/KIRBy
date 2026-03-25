#!/usr/bin/env python3
"""N02: Importance-weighted sample weighting for noise robustness.

Paper: Noise
Hypothesis: Feature importance scores define an importance-weighted distance
    space where per-sample reliability can be measured. Samples whose labels
    are consistent with their importance-weighted neighbors are "reliable";
    inconsistent samples are "unreliable" (noisy). Training with reliability-
    based sample weights should improve predictions.

Key question: Do importance-weighted sample weights help MORE with base reps
    or hybrids?
    - If base reps benefit more  -> hybrids already handle noise implicitly
    - If both benefit equally    -> effects are complementary

4 conditions:
    1. base_unweighted   — individual rep, no sample weights (baseline)
    2. base_weighted     — individual rep, importance-weighted sample weights
    3. hybrid_unweighted — hybrid rep, no sample weights
    4. hybrid_weighted   — hybrid rep, importance-weighted sample weights

Usage:
    python n02_sample_weighting.py --dataset logd --reps fast --mode intermediary
    python n02_sample_weighting.py --dataset caco2 --reps all --mode full
    python n02_sample_weighting.py --dataset flab_thermo_tresanco --reps fast --mode full
    python n02_sample_weighting.py --dataset logd --k-neighbors 15
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split,
    apply_feature_selection, compute_feature_importance,
)
from experiments.complexity.complexity_helpers import cached_create_hybrid as create_hybrid
# TODO: create_hybrid API not finalised — revisit after framework experiments.


# Models that support sample_weight in .fit()
WEIGHTED_MODELS = ['rf', 'xgboost', 'lightgbm', 'ridge']


def compute_reliability_scores(X, y, importance_scores, k=10):
    """Compute per-sample reliability as neighbor-agreement in importance-weighted space.

    For each training sample, find its k nearest neighbors in the
    importance-weighted feature space, then measure how well the sample's
    label agrees with its neighbors' mean label.

    reliability_i = 1 - |y_i - mean(y_neighbors)| / range(y)

    Clamped to [0.1, 1.0] so no sample is fully zero-weighted.

    Args:
        X: Feature matrix (n_samples, n_features), already scaled.
        y: Labels (n_samples,).
        importance_scores: Per-feature importance (n_features,).
        k: Number of neighbors.

    Returns:
        reliability: Array (n_samples,) in [0.1, 1.0].
    """
    # Weight features by sqrt(importance)
    weights = np.sqrt(np.maximum(importance_scores, 0))
    weights = weights / (weights.sum() + 1e-10)
    X_weighted = X * weights

    # Find k nearest neighbors in importance-weighted space
    k_actual = min(k, len(y) - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1)  # +1 because query includes self
    nn.fit(X_weighted)
    _, indices = nn.kneighbors(X_weighted)
    # Exclude self (first neighbor)
    neighbor_idx = indices[:, 1:]

    # Neighbor mean labels
    neighbor_mean = np.mean(y[neighbor_idx], axis=1)

    # Label range (avoid division by zero)
    y_range = y.max() - y.min()
    if y_range < 1e-10:
        return np.ones(len(y))

    # Agreement: 1 - normalized deviation
    deviation = np.abs(y - neighbor_mean) / y_range
    reliability = 1.0 - deviation

    # Clamp to [0.1, 1.0]
    reliability = np.clip(reliability, 0.1, 1.0)

    return reliability


def run(data, config, output_dir, dataset_name, **kwargs):
    """Evaluate importance-weighted sample weighting across 4 conditions."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N02: IMPORTANCE-WEIGHTED SAMPLE WEIGHTING")
    print("=" * 70)

    k_neighbors = kwargs.get('k_neighbors', 10)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(data['labels'])
    task_type = data.get('task_type', 'regression')
    domain = data.get('domain', 'molecular')

    # Only use models that support sample_weight
    model_factories = get_model_factories(
        WEIGHTED_MODELS, task_type=task_type, n_samples=n_samples)

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    sample_X = next(iter(reps.values()))

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Models: {list(model_factories.keys())}")
    print(f"  Reps: {len(reps)} ({', '.join(sorted(reps.keys()))})")
    print(f"  k-neighbors: {k_neighbors}, folds: {n_folds}\n", flush=True)

    results = {
        'per_rep': {},      # conditions 1-2: per individual rep
        'hybrid': {},       # conditions 3-4: hybrid
        'summary': {},
    }

    # =========================================================================
    # CONDITIONS 1-2: Individual reps (base_unweighted, base_weighted)
    # =========================================================================
    print("  --- Conditions 1-2: Individual reps ---")

    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        print(f"\n  Rep: {rep_name} (dim={X_rep.shape[1]})")

        rep_results = {}
        for model_name, model_factory in model_factories.items():
            unweighted_scores = []
            weighted_scores = []
            reliability_stats = []

            for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                                labels, mode=mode):
                y_train, y_val = labels[train_idx], labels[val_idx]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_rep[train_idx])
                X_val = scaler.transform(X_rep[val_idx])

                # Condition 1: unweighted
                try:
                    model_uw = model_factory()
                    model_uw.fit(X_train, y_train)
                    unweighted_scores.append(
                        metric_fn(y_val, model_uw.predict(X_val)))
                except Exception:
                    continue

                # Condition 2: importance-weighted sample weights
                try:
                    # Compute feature importance on training data
                    imp = compute_feature_importance(
                        {rep_name: X_rep[train_idx]}, y_train,
                        method='lightgbm_gain')
                    imp_scores = imp[rep_name]

                    # Compute per-sample reliability in importance-weighted space
                    sample_weights = compute_reliability_scores(
                        X_train, y_train, imp_scores, k=k_neighbors)

                    reliability_stats.append({
                        'mean': float(np.mean(sample_weights)),
                        'std': float(np.std(sample_weights)),
                        'min': float(np.min(sample_weights)),
                        'pct_below_05': float(
                            np.mean(sample_weights < 0.5) * 100),
                    })

                    model_w = model_factory()
                    model_w.fit(X_train, y_train, sample_weight=sample_weights)
                    weighted_scores.append(
                        metric_fn(y_val, model_w.predict(X_val)))
                except Exception as e:
                    weighted_scores.append(float('nan'))

            if not unweighted_scores:
                continue

            uw_mean = float(np.mean(unweighted_scores))
            uw_std = float(np.std(unweighted_scores))
            w_mean = float(np.nanmean(weighted_scores)) if weighted_scores else float('nan')
            w_std = float(np.nanstd(weighted_scores)) if weighted_scores else float('nan')
            delta = w_mean - uw_mean if not np.isnan(w_mean) else float('nan')

            rep_results[model_name] = {
                'unweighted_mean': uw_mean,
                'unweighted_std': uw_std,
                'weighted_mean': w_mean,
                'weighted_std': w_std,
                'delta': delta,
                'reliability_stats': reliability_stats,
            }

            sign = '+' if delta > 0 else ''
            print(f"    {model_name}: unweighted={uw_mean:.4f}, "
                  f"weighted={w_mean:.4f} (delta={sign}{delta:.4f})")

        results['per_rep'][rep_name] = rep_results

    # =========================================================================
    # CONDITIONS 3-4: Hybrid (hybrid_unweighted, hybrid_weighted)
    # =========================================================================
    print(f"\n  --- Conditions 3-4: Hybrid ---")

    hybrid_results = {}
    for model_name, model_factory in model_factories.items():
        unweighted_scores = []
        weighted_scores = []
        reliability_stats = []
        n_features_list = []

        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                            labels, mode=mode):
            train_reps = {k: v[train_idx] for k, v in reps.items()}
            val_reps = {k: v[val_idx] for k, v in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Create hybrid per fold
            try:
                X_train_h, feature_info = create_hybrid(
                    train_reps, y_train,
                    allocation_method='greedy',
                    budget=100,
                    importance_method='lightgbm_gain',
                )
                X_val_h = apply_feature_selection(val_reps, feature_info)
            except Exception as e:
                print(f"    {model_name}: hybrid creation failed - {e}")
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_h)
            X_val_s = scaler.transform(X_val_h)
            n_features_list.append(X_train_h.shape[1])

            # Condition 3: hybrid unweighted
            try:
                model_uw = model_factory()
                model_uw.fit(X_train_s, y_train)
                unweighted_scores.append(
                    metric_fn(y_val, model_uw.predict(X_val_s)))
            except Exception:
                continue

            # Condition 4: hybrid + importance-weighted sample weights
            try:
                # Recompute importance on hybrid features for reliability
                # Use the hybrid feature matrix directly as a single "rep"
                imp = compute_feature_importance(
                    {'hybrid': X_train_h}, y_train, method='lightgbm_gain')
                imp_scores = imp['hybrid']

                sample_weights = compute_reliability_scores(
                    X_train_s, y_train, imp_scores, k=k_neighbors)

                reliability_stats.append({
                    'mean': float(np.mean(sample_weights)),
                    'std': float(np.std(sample_weights)),
                    'min': float(np.min(sample_weights)),
                    'pct_below_05': float(
                        np.mean(sample_weights < 0.5) * 100),
                })

                model_w = model_factory()
                model_w.fit(X_train_s, y_train, sample_weight=sample_weights)
                weighted_scores.append(
                    metric_fn(y_val, model_w.predict(X_val_s)))
            except Exception as e:
                weighted_scores.append(float('nan'))

        if not unweighted_scores:
            continue

        uw_mean = float(np.mean(unweighted_scores))
        uw_std = float(np.std(unweighted_scores))
        w_mean = float(np.nanmean(weighted_scores)) if weighted_scores else float('nan')
        w_std = float(np.nanstd(weighted_scores)) if weighted_scores else float('nan')
        delta = w_mean - uw_mean if not np.isnan(w_mean) else float('nan')
        n_features = int(np.mean(n_features_list)) if n_features_list else 0

        hybrid_results[model_name] = {
            'unweighted_mean': uw_mean,
            'unweighted_std': uw_std,
            'weighted_mean': w_mean,
            'weighted_std': w_std,
            'delta': delta,
            'n_features': n_features,
            'reliability_stats': reliability_stats,
        }

        sign = '+' if delta > 0 else ''
        print(f"    {model_name}: unweighted={uw_mean:.4f}, "
              f"weighted={w_mean:.4f} (delta={sign}{delta:.4f}, "
              f"dim={n_features})")

    results['hybrid'] = hybrid_results

    # =========================================================================
    # SUMMARY: Compare weighting benefit for base reps vs hybrid
    # =========================================================================
    print(f"\n  --- Summary ---")

    # Average delta across models for base reps vs hybrid
    base_deltas = []
    for rep_name, rep_results in results['per_rep'].items():
        for model_name, model_res in rep_results.items():
            d = model_res.get('delta', float('nan'))
            if not np.isnan(d):
                base_deltas.append(d)

    hybrid_deltas = []
    for model_name, model_res in results['hybrid'].items():
        d = model_res.get('delta', float('nan'))
        if not np.isnan(d):
            hybrid_deltas.append(d)

    avg_base_delta = float(np.mean(base_deltas)) if base_deltas else float('nan')
    avg_hybrid_delta = float(np.mean(hybrid_deltas)) if hybrid_deltas else float('nan')

    results['summary'] = {
        'avg_base_delta': avg_base_delta,
        'avg_hybrid_delta': avg_hybrid_delta,
        'base_benefits_more': (avg_base_delta < avg_hybrid_delta) if lower_is_better else (avg_base_delta > avg_hybrid_delta),
        'n_base_improved': int(sum(1 for d in base_deltas if (d < 0 if lower_is_better else d > 0))),
        'n_base_total': len(base_deltas),
        'n_hybrid_improved': int(sum(1 for d in hybrid_deltas if (d < 0 if lower_is_better else d > 0))),
        'n_hybrid_total': len(hybrid_deltas),
    }

    sign_label = "negative" if lower_is_better else "positive"
    print(f"  Avg delta (base reps):  {avg_base_delta:+.4f} "
          f"({results['summary']['n_base_improved']}/{len(base_deltas)} improved)")
    print(f"  Avg delta (hybrid):     {avg_hybrid_delta:+.4f} "
          f"({results['summary']['n_hybrid_improved']}/{len(hybrid_deltas)} improved)")

    if not np.isnan(avg_base_delta) and not np.isnan(avg_hybrid_delta):
        # For lower_is_better: more negative delta = bigger improvement
        base_benefit = -avg_base_delta if lower_is_better else avg_base_delta
        hybrid_benefit = -avg_hybrid_delta if lower_is_better else avg_hybrid_delta
        if base_benefit > hybrid_benefit:
            print("  >> Base reps benefit MORE from sample weighting")
            print("     => Hybrids may already handle noise implicitly")
        elif hybrid_benefit > base_benefit:
            print("  >> Hybrid benefits MORE from sample weighting")
            print("     => Effects are complementary")
        else:
            print("  >> Both benefit equally")

    # Per-model comparison table
    print(f"\n  PER-MODEL COMPARISON (avg delta across reps):")
    print(f"  {'Model':<12} {'Base delta':>12} {'Hybrid delta':>14} {'Winner':>10}")
    print(f"  {'-'*48}")
    for model_name in model_factories:
        # Average base delta for this model across all reps
        model_base_deltas = []
        for rep_name, rep_results in results['per_rep'].items():
            if model_name in rep_results:
                d = rep_results[model_name].get('delta', float('nan'))
                if not np.isnan(d):
                    model_base_deltas.append(d)
        model_base_avg = float(np.mean(model_base_deltas)) if model_base_deltas else float('nan')

        model_hybrid_delta = float('nan')
        if model_name in results['hybrid']:
            model_hybrid_delta = results['hybrid'][model_name].get('delta', float('nan'))

        winner = '--'
        if not np.isnan(model_base_avg) and not np.isnan(model_hybrid_delta):
            base_b = -model_base_avg if lower_is_better else model_base_avg
            hyb_b = -model_hybrid_delta if lower_is_better else model_hybrid_delta
            winner = 'base' if base_b > hyb_b else 'hybrid'

        print(f"  {model_name:<12} {model_base_avg:>+12.4f} {model_hybrid_delta:>+14.4f} {winner:>10}")

    return make_structured_output(
        experiment_id='n02',
        dataset=dataset_name,
        data=data,
        config={
            'n_folds': n_folds,
            'models': list(model_factories.keys()),
            'k_neighbors': k_neighbors,
            'conditions': [
                'base_unweighted', 'base_weighted',
                'hybrid_unweighted', 'hybrid_weighted',
            ],
        },
        results=results,
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser(
        'N02: Importance-Weighted Sample Weighting',
        has_delta=True, has_augment=True)
    parser.add_argument('--k-neighbors', type=int, default=10,
                        help='Number of neighbors for reliability scoring (default: 10)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {WEIGHTED_MODELS}")
    print(f"k-neighbors: {args.k_neighbors}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  k_neighbors=args.k_neighbors)
    save_experiment_results(results, args.output, 'n02_sample_weighting',
                            args.dataset)
