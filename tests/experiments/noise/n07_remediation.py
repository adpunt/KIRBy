#!/usr/bin/env python3
"""N07: Noise Remediation Strategies — what to do after detecting noise.

Paper: Noise
Compares 6 remediation strategies after noise detection: remove, relabel (GP mean),
downweight, label smooth, robust loss, kNN-smooth. Evaluated in both hybrid space
and each individual base rep space for proper baseline comparison.

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
    get_precomputed_search, build_fold_hybrid,
    extract_top3_base_reps,
)
from kirby.utils.noise_estimation import (
    ActivityCliffAwareEstimator, smooth_by_scores, weights_from_scores,
)


def _inject_noise(y, fraction=0.1, seed=42, is_clf=False):
    """Inject noise into fraction of labels."""
    rng = np.random.RandomState(seed)
    n = len(y)
    n_noisy = max(1, int(n * fraction))
    noise_idx = rng.choice(n, n_noisy, replace=False)
    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[noise_idx] = True
    y_noisy = y.copy()
    if is_clf:
        classes = np.unique(y)
        for i in noise_idx:
            other = classes[classes != y[i]]
            if len(other) > 0:
                y_noisy[i] = rng.choice(other)
    else:
        y_range = np.max(y) - np.min(y)
        y_noisy[noise_idx] += rng.normal(0, y_range * 0.3, n_noisy)
    return y_noisy, noise_mask


def _evaluate_fold(X_tr_s, X_va_s, y_train, y_val, strategy, noise_fraction,
                   model_factory, metric_fn, is_clf=False):
    """Evaluate a single remediation strategy on one fold.

    Returns the fold score (float) or None if the fold failed.
    """
    # Inject noise (except for clean baseline)
    if strategy == 'no_noise':
        y_tr = y_train.copy()
    else:
        y_tr, _noise_mask = _inject_noise(y_train, noise_fraction, is_clf=is_clf)

    # Detect noise (for strategies that need it)
    noise_scores = None
    if strategy in ('remove', 'downweight', 'label_smooth', 'knn_smooth'):
        try:
            estimator = ActivityCliffAwareEstimator(k=10, metric='cosine')
            estimator.fit(X_tr_s, y_tr)
            noise_scores = estimator.noise_scores_
        except Exception:
            noise_scores = np.zeros(len(y_tr))

    # Apply strategy
    try:
        if strategy in ('no_noise', 'noisy_baseline'):
            m = model_factory()
            m.fit(X_tr_s, y_tr)
            return metric_fn(y_val, m.predict(X_va_s))

        elif strategy == 'remove':
            threshold = np.percentile(noise_scores, 80)
            keep = noise_scores < threshold
            if keep.sum() >= 10:
                m = model_factory()
                m.fit(X_tr_s[keep], y_tr[keep])
                return metric_fn(y_val, m.predict(X_va_s))

        elif strategy == 'downweight':
            weights = weights_from_scores(noise_scores, strength=1.0)
            m = model_factory()
            m.fit(X_tr_s, y_tr, sample_weight=weights)
            return metric_fn(y_val, m.predict(X_va_s))

        elif strategy == 'label_smooth':
            if is_clf:
                # Label smooth uses neighbor means — use neighbor mode for clf
                nn = NearestNeighbors(n_neighbors=min(10, len(y_tr) - 1))
                nn.fit(X_tr_s)
                _, nbr_idx = nn.kneighbors(X_tr_s)
                from scipy.stats import mode as scipy_mode
                nbr_modes = scipy_mode(y_tr[nbr_idx[:, 1:]], axis=1).mode.ravel()
                # Only correct high-noise-score samples
                threshold = np.percentile(noise_scores, 80)
                y_smoothed = y_tr.copy()
                y_smoothed[noise_scores >= threshold] = nbr_modes[noise_scores >= threshold]
                m = model_factory()
                m.fit(X_tr_s, y_smoothed)
                return metric_fn(y_val, m.predict(X_va_s))
            else:
                nn = NearestNeighbors(n_neighbors=min(10, len(y_tr) - 1))
                nn.fit(X_tr_s)
                _, nbr_idx = nn.kneighbors(X_tr_s)
                nbr_means = np.mean(y_tr[nbr_idx[:, 1:]], axis=1)
                y_smoothed = smooth_by_scores(y_tr, noise_scores, nbr_means,
                                              alpha=0.5, method='proportional')
                m = model_factory()
                m.fit(X_tr_s, y_smoothed)
                return metric_fn(y_val, m.predict(X_va_s))

        elif strategy == 'robust_loss':
            if is_clf:
                return None  # HuberRegressor is regression-only
            huber = HuberRegressor(epsilon=1.35, max_iter=500)
            huber.fit(X_tr_s, y_tr)
            return metric_fn(y_val, huber.predict(X_va_s))

        elif strategy == 'knn_smooth':
            threshold = np.percentile(noise_scores, 80)
            flagged = noise_scores >= threshold
            nn = NearestNeighbors(n_neighbors=min(10, len(y_tr) - 1))
            nn.fit(X_tr_s)
            _, nbr_idx = nn.kneighbors(X_tr_s)
            if is_clf:
                from scipy.stats import mode as scipy_mode
                nbr_modes = scipy_mode(y_tr[nbr_idx[:, 1:]], axis=1).mode.ravel()
                y_corrected = y_tr.copy()
                y_corrected[flagged] = nbr_modes[flagged]
            else:
                nbr_means = np.mean(y_tr[nbr_idx[:, 1:]], axis=1)
                y_corrected = y_tr.copy()
                y_corrected[flagged] = nbr_means[flagged]
            m = model_factory()
            m.fit(X_tr_s, y_corrected)
            return metric_fn(y_val, m.predict(X_va_s))

    except Exception:
        return None

    return None


def run(data, config, output_dir, dataset_name, **kwargs):
    """N07: Noise remediation strategies."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N07: NOISE REMEDIATION STRATEGIES")
    print("=" * 70)

    task_type = data.get('task_type', 'regression')
    is_clf = task_type == 'classification'
    model_factory = get_model_factories(
        ['rf'], task_type=task_type,
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

    # Get hybrid blueprint ONCE (shared across all strategies/folds)
    search = get_precomputed_search(
        reps, labels, groups, budget=budget, aug_condition='uncapped',
        augmentations=augmentations, speed=config.get('search_speed', 'full'),
        precomputed=data.get('precomputed_searches'))
    top3 = extract_top3_base_reps(search)
    if top3:
        print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    best_rf = search.best_by_model.get('rf')
    bp = best_rf['blueprint'] if best_rf else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}

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
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        'per_strategy': {},
    }

    # --- Hybrid conditions ---
    for strategy in strategies:
        print(f"\n  hybrid/{strategy}...", end='', flush=True)
        fold_scores = []

        for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            # Build hybrid features from blueprint
            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
            val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
            try:
                train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                X_train_h, X_val_h, fi = build_fold_hybrid(
                    train_reps_fold, val_reps_fold, y_train_clean, bp,
                    augmentations_train=train_augs, augmentations_val=val_augs)
            except Exception:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train_h)
            X_va_s = scaler.transform(X_val_h)

            fold_score = _evaluate_fold(
                X_tr_s, X_va_s, y_train_clean, y_val, strategy,
                noise_fraction, model_factory, metric_fn, is_clf=is_clf)
            if fold_score is not None:
                fold_scores.append(fold_score)

        score = float(np.mean(fold_scores)) if fold_scores else float('nan')
        results['per_strategy'][strategy] = {'score': score, 'n_folds': len(fold_scores)}
        print(f" {metric_name}={score:.4f}")

    # --- Per-rep conditions ---
    for rep_name, X_rep in reps.items():
        for strategy in strategies:
            key = f"{rep_name}/{strategy}"
            print(f"\n  {key}...", end='', flush=True)
            fold_scores = []

            for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
                y_train_clean = labels[train_idx]
                y_val = labels[val_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_rep[train_idx])
                X_va_s = scaler.transform(X_rep[val_idx])

                fold_score = _evaluate_fold(
                    X_tr_s, X_va_s, y_train_clean, y_val, strategy,
                    noise_fraction, model_factory, metric_fn)
                if fold_score is not None:
                    fold_scores.append(fold_score)

            score = float(np.mean(fold_scores)) if fold_scores else float('nan')
            results['per_strategy'][key] = {'score': score, 'n_folds': len(fold_scores)}
            print(f" {metric_name}={score:.4f}")

    # Summary: recovery = (strategy - noisy_baseline) / (no_noise - noisy_baseline)
    clean = results['per_strategy'].get('no_noise', {}).get('score', float('nan'))
    noisy = results['per_strategy'].get('noisy_baseline', {}).get('score', float('nan'))
    gap = clean - noisy if not any(np.isnan(x) for x in [clean, noisy]) else float('nan')

    # Compute recovery for all conditions (hybrid and per-rep)
    # Recovery uses hybrid baselines (no_noise, noisy_baseline) as reference
    recovery = {}
    remediation_strategies = [s for s in strategies if s not in ('no_noise', 'noisy_baseline')]
    all_condition_keys = [k for k in results['per_strategy']
                          if not k.endswith('/no_noise') and not k.endswith('/noisy_baseline')
                          and k not in ('no_noise', 'noisy_baseline')]
    for key in all_condition_keys:
        s = results['per_strategy'].get(key, {}).get('score', float('nan'))
        # For per-rep conditions, use their own baselines if available
        if '/' in key:
            prefix = key.rsplit('/', 1)[0]
            ref_clean = results['per_strategy'].get(f'{prefix}/no_noise', {}).get('score', clean)
            ref_noisy = results['per_strategy'].get(f'{prefix}/noisy_baseline', {}).get('score', noisy)
        else:
            ref_clean = clean
            ref_noisy = noisy
        ref_gap = ref_clean - ref_noisy if not any(np.isnan(x) for x in [ref_clean, ref_noisy]) else float('nan')
        if not np.isnan(s) and not np.isnan(ref_gap) and abs(ref_gap) > 1e-10:
            recovery[key] = float((s - ref_noisy) / ref_gap * 100)
        else:
            recovery[key] = float('nan')

    results['summary'] = {
        'clean_score': clean,
        'noisy_score': noisy,
        'noise_gap': float(gap) if not np.isnan(gap) else None,
        'recovery_pct': recovery,
        'best_strategy': max(recovery, key=lambda k: recovery[k]
                             if not np.isnan(recovery[k]) else -1e10) if recovery else None,
    }
    print(f"\n  Clean: {clean:.4f}, Noisy: {noisy:.4f}, Gap: {gap:.4f}")
    hybrid_recovery = {k: round(v, 1) for k, v in recovery.items() if '/' not in k}
    print(f"  Recovery (hybrid): {hybrid_recovery}")
    for rep_name in reps:
        rep_recovery = {k.split('/')[-1]: v for k, v in recovery.items()
                        if k.startswith(f'{rep_name}/')}
        if rep_recovery:
            print(f"  Recovery ({rep_name}): {rep_recovery}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N07: Noise Remediation')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n07_remediation', args.dataset)
