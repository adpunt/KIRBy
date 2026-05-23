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
    compute_feature_importance,
    get_precomputed_search, get_best_search_for_downstream, build_fold_hybrid,
    extract_top3_base_reps,
)

from kirby.utils.noise_estimation import ActivityCliffAwareEstimator


# Models that support sample_weight in .fit()
WEIGHTED_MODELS = ['rf', 'xgboost', 'lightgbm', 'ridge']


def compute_cliff_aware_weights(X, y, k=10, strength=1.0):
    """Use ActivityCliffAwareEstimator to compute sample weights.

    Unlike compute_reliability_scores, this distinguishes cliffs from noise:
    cliff compounds get higher weights (they're not noise), while genuinely
    noisy samples get lower weights.
    """
    estimator = ActivityCliffAwareEstimator(k=k, metric='cosine')
    estimator.fit(X, y)
    return estimator.get_weights(strength=strength)


def compute_reliability_scores(X, y, importance_scores, k=10, is_clf=False):
    """Compute per-sample reliability as neighbor-agreement in importance-weighted space.

    For regression: reliability_i = 1 - |y_i - mean(y_neighbors)| / range(y)
    For classification: reliability_i = fraction of k-NN with same class label.

    Clamped to [0.1, 1.0] so no sample is fully zero-weighted.

    Args:
        X: Feature matrix (n_samples, n_features), already scaled.
        y: Labels (n_samples,).
        importance_scores: Per-feature importance (n_features,).
        k: Number of neighbors.
        is_clf: If True, use class-agreement instead of label proximity.

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

    if is_clf:
        # Classification: fraction of neighbors with same class
        same_class = (y[neighbor_idx] == y[:, None]).astype(float)
        reliability = np.mean(same_class, axis=1)
    else:
        # Regression: neighbor mean agreement
        neighbor_mean = np.mean(y[neighbor_idx], axis=1)
        y_range = y.max() - y.min()
        if y_range < 1e-10:
            return np.ones(len(y))
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
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    domain = data.get('domain', 'molecular')

    # Only use models that support sample_weight
    model_factories = get_model_factories(
        WEIGHTED_MODELS, task_type=task_type, n_samples=n_samples)

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    augmentation_strategy = 'uncapped'
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
                        X_train, y_train, imp_scores, k=k_neighbors,
                        is_clf=is_clf)

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
    # CONDITIONS 3-5: Get hybrid blueprint ONCE (shared across all fold loops)
    # =========================================================================
    search, meta = get_best_search_for_downstream(
        reps, labels, groups,
        downstream_model='rf', budget=None,
        augmentations=augmentations,
        speed=config.get('search_speed', 'full'),
        dataset_name=dataset_name, task_type=task_type,
        data=data,
    )
    if search is not None:
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    bp = meta['best_config']['blueprint'] if meta else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping hybrid conditions")
        return {'error': 'no_blueprint', 'metric': metric_name}

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

            # Create hybrid per fold from blueprint
            try:
                train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                X_train_h, X_val_h, feature_info = build_fold_hybrid(
                    train_reps, val_reps, y_train, bp,
                    augmentations_train=train_augs, augmentations_val=val_augs)
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
    # CONDITION 5: Hybrid + cliff-aware weighting (ActivityCliffAwareEstimator)
    # =========================================================================
    print(f"\n  --- Condition 5: Hybrid + cliff-aware weights ---")
    cliff_aware_results = {}
    for model_name, model_factory in model_factories.items():
        ca_scores = []
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                            labels, mode=mode):
            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
            val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]
            try:
                train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                X_train_h, X_val_h, fi = build_fold_hybrid(
                    train_reps_fold, val_reps_fold, y_train, bp,
                    augmentations_train=train_augs, augmentations_val=val_augs)
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train_h)
                X_va_s = scaler.transform(X_val_h)
                ca_weights = compute_cliff_aware_weights(X_tr_s, y_train, k=k_neighbors)
                model = model_factory()
                model.fit(X_tr_s, y_train, sample_weight=ca_weights)
                ca_scores.append(metric_fn(y_val, model.predict(X_va_s)))
            except Exception:
                continue
        if ca_scores:
            ca_mean = float(np.mean(ca_scores))
            uw_ref = hybrid_results.get(model_name, {}).get('unweighted_mean', float('nan'))
            delta_ca = ca_mean - uw_ref if not np.isnan(uw_ref) else float('nan')
            cliff_aware_results[model_name] = {
                'score_mean': ca_mean,
                'delta_vs_unweighted': delta_ca,
            }
            print(f"    {model_name}: cliff_aware={ca_mean:.4f} (delta={delta_ca:+.4f})")

    results['hybrid_cliff_aware'] = cliff_aware_results

    # =========================================================================
    # SUMMARY: Compare weighting benefit for base reps vs hybrid
    # =========================================================================
    print(f"\n  --- Summary ---")

    hybrid_deltas = []
    for model_name, model_res in results['hybrid'].items():
        d = model_res.get('delta', float('nan'))
        if not np.isnan(d):
            hybrid_deltas.append(d)
    n_hybrid_improved = int(sum(1 for d in hybrid_deltas if (d < 0 if lower_is_better else d > 0)))
    print(f"  Hybrid weighting: {n_hybrid_improved}/{len(hybrid_deltas)} models improved")

    # Per-model comparison: top-3 base reps vs hybrid
    print(f"\n  PER-MODEL COMPARISON (top-3 base reps vs hybrid):")
    results['per_model_top3'] = {}
    for model_name in model_factories:
        # Collect (rep_name, unweighted_score) for this model
        rep_scores = []
        for rep_name, rep_results in results['per_rep'].items():
            if model_name in rep_results:
                uw = rep_results[model_name].get('unweighted_mean', float('nan'))
                w = rep_results[model_name].get('weighted_mean', float('nan'))
                d = rep_results[model_name].get('delta', float('nan'))
                if not np.isnan(uw):
                    rep_scores.append((rep_name, uw, w, d))

        # Sort by unweighted score (best first)
        rep_scores.sort(key=lambda x: x[1], reverse=not lower_is_better)
        top3 = rep_scores[:3]

        model_hybrid_uw = float('nan')
        model_hybrid_w = float('nan')
        model_hybrid_delta = float('nan')
        if model_name in results['hybrid']:
            model_hybrid_uw = results['hybrid'][model_name].get('unweighted_mean', float('nan'))
            model_hybrid_w = results['hybrid'][model_name].get('weighted_mean', float('nan'))
            model_hybrid_delta = results['hybrid'][model_name].get('delta', float('nan'))

        results['per_model_top3'][model_name] = {
            'top3': [(n, uw, w, d) for n, uw, w, d in top3],
            'hybrid_unweighted': model_hybrid_uw,
            'hybrid_weighted': model_hybrid_w,
            'hybrid_delta': model_hybrid_delta,
        }

        print(f"\n  {model_name}:")
        for rn, uw, w, d in top3:
            print(f"    {rn:<20} uw={uw:.4f}, w={w:.4f} (delta={d:+.4f})")
        print(f"    {'HYBRID':<20} uw={model_hybrid_uw:.4f}, w={model_hybrid_w:.4f} "
              f"(delta={model_hybrid_delta:+.4f})")

    results['top3_base_reps'] = [(n, uw) for n, uw, _w, _d in top3] if top3 else []

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
