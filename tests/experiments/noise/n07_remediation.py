#!/usr/bin/env python3
"""N07: Noise Remediation Strategies — what to do after detecting noise.

Paper: Noise
Compares 6 remediation strategies after noise detection: remove, relabel (GP mean),
downweight, label smooth, robust loss, kNN-smooth in hybrid space.

Usage:
    python n07_remediation.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import NearestNeighbors
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
)
from kirby.hybrid import create_hybrid
# TODO: create_hybrid API not finalised — revisit after framework experiments.
from kirby.utils.noise_estimation import (
    ActivityCliffAwareEstimator, smooth_by_scores, weights_from_scores,
)
from kirby.hybrid import apply_feature_selection


def _inject_noise(y, fraction=0.1, seed=42):
    """Inject noise into fraction of labels."""
    rng = np.random.RandomState(seed)
    n = len(y)
    n_noisy = max(1, int(n * fraction))
    noise_idx = rng.choice(n, n_noisy, replace=False)
    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[noise_idx] = True
    y_noisy = y.copy()
    y_range = np.max(y) - np.min(y)
    y_noisy[noise_idx] += rng.normal(0, y_range * 0.3, n_noisy)
    return y_noisy, noise_mask


def run(data, config, output_dir, dataset_name, **kwargs):
    """N07: Noise remediation strategies."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N07: NOISE REMEDIATION STRATEGIES")
    print("=" * 70)

    model_factory = get_model_factories(
        ['rf'], task_type=data.get('task_type', 'regression'),
        n_samples=len(data['labels']))['rf']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    augmentation_strategy = 'uncapped'
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    budget = kwargs.get('budget', 100)
    noise_fraction = kwargs.get('noise_fraction', 0.1)

    sample_X = next(iter(reps.values()))

    strategies = [
        'no_noise',         # clean baseline (no injection)
        'noisy_baseline',   # inject noise, do nothing
        'remove',           # detect + remove flagged samples
        'downweight',       # detect + weight by 1/(1+score)
        'label_smooth',     # detect + smooth labels toward neighbor means
        'robust_loss',      # Huber regression (no detection needed)
        'knn_smooth',       # replace flagged labels with kNN-mean in hybrid space
    ]

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_fraction': noise_fraction,
        'strategies': strategies,
        'per_strategy': {},
    }

    for strategy in strategies:
        print(f"\n  {strategy}...", end='', flush=True)
        fold_scores = []

        for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            # Inject noise (except for clean baseline)
            if strategy == 'no_noise':
                y_train = y_train_clean.copy()
            else:
                y_train, noise_mask = _inject_noise(y_train_clean, noise_fraction)

            # Build hybrid features
            train_reps = {k: v[train_idx] for k, v in reps.items()}
            val_reps = {k: v[val_idx] for k, v in reps.items()}
            try:
                X_train_h, fi = create_hybrid(
                    train_reps, y_train,
                    allocation_method='entropy_1.0', budget=budget,
                    importance_method='lightgbm_gain',
                    augmentations=augmentations,
                    augmentation_strategy=augmentation_strategy,
                )
                X_val_h = apply_feature_selection(val_reps, fi)
            except Exception:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train_h)
            X_va_s = scaler.transform(X_val_h)

            # Detect noise (for strategies that need it)
            noise_scores = None
            if strategy in ('remove', 'downweight', 'label_smooth', 'knn_smooth'):
                try:
                    estimator = ActivityCliffAwareEstimator(k=10, metric='cosine')
                    estimator.fit(X_tr_s, y_train)
                    noise_scores = estimator.noise_scores_
                except Exception:
                    noise_scores = np.zeros(len(y_train))

            # Apply strategy
            try:
                if strategy in ('no_noise', 'noisy_baseline'):
                    rf = model_factory()
                    rf.fit(X_tr_s, y_train)
                    fold_scores.append(metric_fn(y_val, rf.predict(X_va_s)))

                elif strategy == 'remove':
                    threshold = np.percentile(noise_scores, 80)
                    keep = noise_scores < threshold
                    if keep.sum() >= 10:
                        rf = model_factory()
                        rf.fit(X_tr_s[keep], y_train[keep])
                        fold_scores.append(metric_fn(y_val, rf.predict(X_va_s)))

                elif strategy == 'downweight':
                    weights = weights_from_scores(noise_scores, strength=1.0)
                    rf = model_factory()
                    rf.fit(X_tr_s, y_train, sample_weight=weights)
                    fold_scores.append(metric_fn(y_val, rf.predict(X_va_s)))

                elif strategy == 'label_smooth':
                    # Smooth noisy labels toward kNN mean
                    nn = NearestNeighbors(n_neighbors=min(10, len(y_train) - 1))
                    nn.fit(X_tr_s)
                    _, nbr_idx = nn.kneighbors(X_tr_s)
                    nbr_means = np.mean(y_train[nbr_idx[:, 1:]], axis=1)
                    y_smoothed = smooth_by_scores(y_train, noise_scores, nbr_means,
                                                  alpha=0.5, method='proportional')
                    rf = model_factory()
                    rf.fit(X_tr_s, y_smoothed)
                    fold_scores.append(metric_fn(y_val, rf.predict(X_va_s)))

                elif strategy == 'robust_loss':
                    huber = HuberRegressor(epsilon=1.35, max_iter=500)
                    huber.fit(X_tr_s, y_train)
                    fold_scores.append(metric_fn(y_val, huber.predict(X_va_s)))

                elif strategy == 'knn_smooth':
                    # Replace high-noise labels with kNN mean in hybrid space
                    threshold = np.percentile(noise_scores, 80)
                    flagged = noise_scores >= threshold
                    nn = NearestNeighbors(n_neighbors=min(10, len(y_train) - 1))
                    nn.fit(X_tr_s)
                    _, nbr_idx = nn.kneighbors(X_tr_s)
                    nbr_means = np.mean(y_train[nbr_idx[:, 1:]], axis=1)
                    y_corrected = y_train.copy()
                    y_corrected[flagged] = nbr_means[flagged]
                    rf = model_factory()
                    rf.fit(X_tr_s, y_corrected)
                    fold_scores.append(metric_fn(y_val, rf.predict(X_va_s)))

            except Exception as e:
                print(f" (fold failed: {e})", end='')

        score = float(np.mean(fold_scores)) if fold_scores else float('nan')
        results['per_strategy'][strategy] = {'score': score, 'n_folds': len(fold_scores)}
        print(f" {metric_name}={score:.4f}")

    # Summary: recovery = (strategy - noisy_baseline) / (no_noise - noisy_baseline)
    clean = results['per_strategy'].get('no_noise', {}).get('score', float('nan'))
    noisy = results['per_strategy'].get('noisy_baseline', {}).get('score', float('nan'))
    gap = clean - noisy if not any(np.isnan(x) for x in [clean, noisy]) else float('nan')

    recovery = {}
    for strategy in strategies:
        if strategy in ('no_noise', 'noisy_baseline'):
            continue
        s = results['per_strategy'].get(strategy, {}).get('score', float('nan'))
        if not np.isnan(s) and not np.isnan(gap) and abs(gap) > 1e-10:
            recovery[strategy] = float((s - noisy) / gap * 100)
        else:
            recovery[strategy] = float('nan')

    results['summary'] = {
        'clean_score': clean,
        'noisy_score': noisy,
        'noise_gap': float(gap) if not np.isnan(gap) else None,
        'recovery_pct': recovery,
        'best_strategy': max(recovery, key=lambda k: recovery[k]
                             if not np.isnan(recovery[k]) else -1e10) if recovery else None,
    }
    print(f"\n  Clean: {clean:.4f}, Noisy: {noisy:.4f}, Gap: {gap:.4f}")
    print(f"  Recovery: {recovery}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N07: Noise Remediation')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n07_remediation', args.dataset)
