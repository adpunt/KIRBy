#!/usr/bin/env python3
"""N13: Alternative Noise Distributions.

Paper: Noise
Tests whether methods that work for Gaussian noise generalize to other
distributions: uniform, multiplicative, heavy-tailed, and partial corruption.

Two evaluation tracks:
  Track A: Selected LNL methods covering each family tested in N09
           (data-level, training-loop, GP) — retention metric
  Track B: Selected estimators covering each family tested in N11
           (kNN-based, GP-based, gradient-boosted) — recovery metric

Usage:
    python n13_noise_distributions.py --dataset logd --mode intermediary
    python n13_noise_distributions.py --dataset qm9 --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    get_precomputed_search,
)
from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection
from kirby.utils.lnl_methods import get_lnl_method, _standard_dnn_train_predict
from kirby.utils.noise_estimation import get_noise_estimator, weights_from_scores


NOISE_TYPES = ['gaussian', 'uniform', 'multiplicative', 'heavy_tailed', 'partial']
STRENGTHS = [0.1, 0.3, 0.6]
PARTIAL_FRACTIONS = [0.1, 0.3, 0.5]  # for partial noise type

LNL_METHODS = ['dividemix', 'small_loss', 'uncertainty_curriculum', 'heteroscedastic_gp']
ESTIMATORS = ['local_variance_ratio', 'gaussian_process']
# ngboost is optional
try:
    get_noise_estimator('ngboost')
    ESTIMATORS.append('ngboost')
except Exception:
    pass

LNL_EPOCHS = 50  # reduced for speed


def _inject_noise(y, noise_type, strength, seed=42, partial_fraction=0.3):
    """Inject noise of the specified distribution.

    Args:
        y: Clean labels.
        noise_type: One of NOISE_TYPES.
        strength: Noise magnitude (sigma for Gaussian, width for uniform, etc.).
        seed: Random seed.
        partial_fraction: Fraction of samples to corrupt (for 'partial' type).

    Returns:
        y_noisy: Noisy labels (same shape as y).
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=np.float64)
    y_range = float(np.max(y) - np.min(y))
    n = len(y)

    if noise_type == 'gaussian':
        noise = rng.normal(0, strength * y_range, n)
        return (y + noise).astype(np.float32)

    elif noise_type == 'uniform':
        half_width = strength * y_range
        noise = rng.uniform(-half_width, half_width, n)
        return (y + noise).astype(np.float32)

    elif noise_type == 'multiplicative':
        # Relative noise: y_noisy = y * (1 + N(0, sigma))
        multiplier = 1 + rng.normal(0, strength, n)
        return (y * multiplier).astype(np.float32)

    elif noise_type == 'heavy_tailed':
        # Student-t with df=3, scaled to match Gaussian sigma
        from scipy.stats import t as t_dist
        t_std = np.sqrt(3 / (3 - 2))  # theoretical std of t(df=3)
        raw = rng.standard_t(df=3, size=n)
        noise = raw * (strength * y_range / t_std)
        return (y + noise).astype(np.float32)

    elif noise_type == 'partial':
        # Gaussian noise on only a fraction of samples
        n_corrupt = max(1, int(n * partial_fraction))
        corrupt_idx = rng.choice(n, n_corrupt, replace=False)
        noise = np.zeros(n)
        noise[corrupt_idx] = rng.normal(0, strength * y_range, n_corrupt)
        return (y + noise).astype(np.float32)

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def _build_spaces(train_reps, y_train, augmentations, bp, base_rep_name):
    """Build 2 spaces: base rep + hybrid.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations: {aug_name: X_train} already sliced, or None.
        bp: Blueprint from get_precomputed_search.
        base_rep_name: Name of base representation.

    Returns (spaces, feature_info_or_None, scaler_hybrid_or_None).
    feature_info can be used to apply hybrid selection to val reps without
    a second create_hybrid call.
    """
    spaces = {}
    feature_info = None
    scaler_h = None

    X_base = train_reps[base_rep_name]
    sc = StandardScaler()
    spaces[base_rep_name] = sc.fit_transform(X_base)

    try:
        X_h, feature_info = create_hybrid(
            train_reps, y_train,
            allocation_method=bp.allocation_method,
            importance_method=bp.importance_method,
            internal_evaluator=bp.internal_evaluator,
            budget=bp.budget,
            augmentations=augmentations,
            augmentation_strategy=bp.augmentation_strategy,
        )
        scaler_h = StandardScaler()
        spaces['hybrid'] = scaler_h.fit_transform(X_h)
    except Exception as e:
        print(f"      hybrid failed: {e}")

    return spaces, feature_info, scaler_h


def run(data, config, output_dir, dataset_name, **kwargs):
    """N13: Alternative noise distributions."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N13: ALTERNATIVE NOISE DISTRIBUTIONS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    model_factory = get_model_factories(
        ['rf'], task_type=data.get('task_type', 'regression'),
        n_samples=len(labels))['rf']

    sample_X = next(iter(reps.values()))

    # Find base rep
    base_rep_name = None
    for candidate in ['ecfp4', 'pdv', 'blosum']:
        if candidate in reps:
            base_rep_name = candidate
            break
    if base_rep_name is None:
        base_rep_name = sorted(reps.keys())[0]

    print(f"  Dataset: {dataset_name} ({len(labels)} samples)")
    print(f"  Metric: {metric_name}")
    print(f"  Base rep: {base_rep_name}")
    print(f"  Noise types: {NOISE_TYPES}")
    print(f"  LNL methods: {LNL_METHODS}")
    print(f"  Estimators: {ESTIMATORS}")
    print(f"  Folds: {n_folds}")

    # Get blueprint ONCE from cached search (uses clean labels)
    search = get_precomputed_search(
        reps, labels, groups, budget=100, aug_condition='uncapped',
        augmentations=augmentations, speed=config.get('search_speed', 'fast'))
    bp = search.best_overall['blueprint'] if search.best_overall else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_types': NOISE_TYPES,
        'strengths': STRENGTHS,
        'lnl_methods': LNL_METHODS,
        'estimators': ESTIMATORS,
        'per_noise_type': {},
    }

    # ── Cache fold splits and label-independent eval features ──
    clean_scores = []
    fold_data = []

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                        labels, mode=mode):
        y_train_clean = labels[train_idx]
        y_val = labels[val_idx]

        train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
        val_reps_fold = {k: v[val_idx] for k, v in reps.items()}

        # Base rep eval features (independent of labels)
        X_eval_train = reps[base_rep_name][train_idx]
        X_eval_val = reps[base_rep_name][val_idx]
        sc_eval = StandardScaler()
        X_eval_train_s = sc_eval.fit_transform(X_eval_train)
        X_eval_val_s = sc_eval.transform(X_eval_val)

        # Clean baseline
        rf_clean = model_factory()
        rf_clean.fit(X_eval_train_s, y_train_clean)
        clean_scores.append(float(metric_fn(y_val, rf_clean.predict(X_eval_val_s))))

        train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None

        fold_data.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'y_train_clean': y_train_clean,
            'y_val': y_val,
            'train_reps_fold': train_reps_fold,
            'val_reps_fold': val_reps_fold,
            'train_augs': train_augs,
            'X_eval_train_s': X_eval_train_s,
            'X_eval_val_s': X_eval_val_s,
        })

    clean_mean = float(np.mean(clean_scores))

    # ── Main loop: noise type × strength ──
    for noise_type in NOISE_TYPES:
        print(f"\n  === {noise_type} ===")
        t_nt = time.time()
        nt_result = {'per_strength': {}}

        # Determine strength variants
        if noise_type == 'partial':
            strength_variants = [(0.3, f) for f in PARTIAL_FRACTIONS]
        else:
            strength_variants = [(s, None) for s in STRENGTHS]

        for strength, partial_frac in strength_variants:
            strength_key = str(strength) if partial_frac is None else f'{strength}_frac{partial_frac}'

            noisy_baseline_scores = []
            # {space: {method: [scores]}} for track A
            track_a_acc = {}
            # {space: {estimator: [scores]}} for track B
            track_b_acc = {}

            for fi, fd in enumerate(fold_data):
                # Inject noise
                kw = {}
                if noise_type == 'partial' and partial_frac is not None:
                    kw['partial_fraction'] = partial_frac
                y_noisy = _inject_noise(fd['y_train_clean'], noise_type,
                                         strength, seed=42 + fi, **kw)

                # Noisy baseline
                rf_noisy = model_factory()
                rf_noisy.fit(fd['X_eval_train_s'], y_noisy)
                noisy_baseline_scores.append(float(
                    metric_fn(fd['y_val'], rf_noisy.predict(fd['X_eval_val_s']))))

                # Build spaces from NOISY labels (realistic — no oracle leak)
                spaces, feature_info, scaler_h = _build_spaces(
                    fd['train_reps_fold'], y_noisy, fd['train_augs'],
                    bp, base_rep_name)

                # Build hybrid val features using feature_info (no second create_hybrid)
                hybrid_val_s = None
                if 'hybrid' in spaces and feature_info is not None and scaler_h is not None:
                    try:
                        X_h_val = apply_feature_selection(fd['val_reps_fold'], feature_info)
                        aug_info = feature_info.get('augmentation_info')
                        if aug_info and fd.get('val_augs'):
                            X_aug_val = apply_augmentation_selection(fd['val_augs'], aug_info)
                            if X_aug_val is not None:
                                X_h_val = np.hstack([X_h_val, X_aug_val])
                        hybrid_val_s = scaler_h.transform(X_h_val)
                    except Exception:
                        pass

                # ── Track A: LNL methods ──
                for space_name in [base_rep_name, 'hybrid']:
                    if space_name not in spaces:
                        continue
                    X_train_s = spaces[space_name]

                    if space_name == 'hybrid' and hybrid_val_s is not None:
                        X_val_s = hybrid_val_s
                    elif space_name == base_rep_name:
                        X_val_s = fd['X_eval_val_s']
                    else:
                        continue

                    if space_name not in track_a_acc:
                        track_a_acc[space_name] = {}

                    for method_name in LNL_METHODS:
                        if method_name not in track_a_acc[space_name]:
                            track_a_acc[space_name][method_name] = []
                        try:
                            m = get_lnl_method(method_name, epochs=LNL_EPOCHS,
                                               patience=10, random_state=42 + fi)
                            res = m.train_and_predict(X_train_s, y_noisy,
                                                      X_val_s, fd['y_val'])
                            score = float(metric_fn(fd['y_val'], res['y_pred']))
                            track_a_acc[space_name][method_name].append(score)
                        except Exception:
                            pass

                # ── Track B: Estimation + remediation ──
                for space_name in [base_rep_name, 'hybrid']:
                    if space_name not in spaces:
                        continue
                    X_space = spaces[space_name]

                    if space_name not in track_b_acc:
                        track_b_acc[space_name] = {}

                    for est_name in ESTIMATORS:
                        if est_name not in track_b_acc[space_name]:
                            track_b_acc[space_name][est_name] = []
                        try:
                            est = get_noise_estimator(est_name)
                            scores = est.estimate(X_space, y_noisy)
                            weights = weights_from_scores(scores, strength=0.1,
                                                          method='exponential')
                            rf = model_factory()
                            rf.fit(fd['X_eval_train_s'], y_noisy,
                                   sample_weight=weights)
                            score = float(metric_fn(fd['y_val'],
                                                     rf.predict(fd['X_eval_val_s'])))
                            track_b_acc[space_name][est_name].append(score)
                        except Exception:
                            pass

            # Aggregate
            noisy_mean = float(np.mean(noisy_baseline_scores)) if noisy_baseline_scores else float('nan')
            gap = clean_mean - noisy_mean

            strength_result = {
                'noisy_score': noisy_mean,
                'clean_score': clean_mean,
                'track_a': {},
                'track_b': {},
            }

            # Track A aggregation
            for space_name, methods in track_a_acc.items():
                space_a = {}
                for method_name, score_list in methods.items():
                    if score_list:
                        mean_s = float(np.mean(score_list))
                        # Retention: score / clean_score
                        retention = float(mean_s / clean_mean * 100) if abs(clean_mean) > 1e-10 else float('nan')
                        space_a[method_name] = {
                            'score': mean_s,
                            'retention': retention,
                        }
                strength_result['track_a'][space_name] = space_a

            # Track B aggregation
            for space_name, estimators in track_b_acc.items():
                space_b = {}
                for est_name, score_list in estimators.items():
                    if score_list:
                        denoised = float(np.mean(score_list))
                        recovery = float((denoised - noisy_mean) / gap * 100) if abs(gap) > 1e-10 else float('nan')
                        space_b[est_name] = {
                            'denoised': denoised,
                            'recovery': recovery,
                        }
                strength_result['track_b'][space_name] = space_b

            nt_result['per_strength'][strength_key] = convert_numpy(strength_result)

            # Print compact
            print(f"    σ={strength_key}: noisy={noisy_mean:.4f}", end='')
            # Best track A
            best_a = None
            best_a_score = float('-inf') if not lower_is_better else float('inf')
            for sp_d in strength_result['track_a'].values():
                for mn, mv in sp_d.items():
                    s = mv.get('score', float('nan'))
                    if not np.isnan(s):
                        if (not lower_is_better and s > best_a_score) or \
                           (lower_is_better and s < best_a_score):
                            best_a_score = s
                            best_a = mn
            if best_a:
                print(f", best_LNL={best_a}({best_a_score:.4f})", end='')
            print()

        results['per_noise_type'][noise_type] = convert_numpy(nt_result)
        print(f"    [{time.time() - t_nt:.0f}s]")

    # Summary
    results['summary'] = _compute_summary(results)
    s = results['summary']
    print(f"\n  === SUMMARY ===")
    print(f"  Gaussian rank transfers: {s.get('gaussian_rank_transfers')}")
    print(f"  Hybrid advantage persists: {s.get('hybrid_advantage_persists')}")
    if s.get('best_method_per_distribution'):
        for nt, m in s['best_method_per_distribution'].items():
            print(f"    {nt}: best={m}")

    return results


def _compute_summary(results):
    """Compute summary: does Gaussian ranking transfer? Does hybrid help across distributions?"""
    # Collect best Track A method per (noise_type, space)
    best_per_dist = {}  # noise_type -> best method name
    hybrid_wins = 0
    hybrid_total = 0

    for nt, nt_data in results.get('per_noise_type', {}).items():
        best_method = None
        best_score = float('-inf')
        lower = results.get('lower_is_better', False)

        for sk, sd in nt_data.get('per_strength', {}).items():
            for space_name, space_data in sd.get('track_a', {}).items():
                for mn, mv in space_data.items():
                    s = mv.get('score', float('nan'))
                    if np.isnan(s):
                        continue
                    if lower:
                        if best_method is None or s < best_score:
                            best_score = s
                            best_method = mn
                    else:
                        if s > best_score:
                            best_score = s
                            best_method = mn

            # Check hybrid advantage (Track A)
            base_scores = []
            hybrid_scores = []
            base_key = results.get('base_rep', 'ecfp4')
            for space_name, space_data in sd.get('track_a', {}).items():
                for mv in space_data.values():
                    s = mv.get('score', float('nan'))
                    if not np.isnan(s):
                        if space_name == 'hybrid':
                            hybrid_scores.append(s)
                        else:
                            base_scores.append(s)
            if base_scores and hybrid_scores:
                hybrid_total += 1
                b_avg = np.mean(base_scores)
                h_avg = np.mean(hybrid_scores)
                if (not lower and h_avg > b_avg) or (lower and h_avg < b_avg):
                    hybrid_wins += 1

        if best_method:
            best_per_dist[nt] = best_method

    # Does Gaussian winner transfer?
    gaussian_best = best_per_dist.get('gaussian')
    if gaussian_best:
        transfers = all(best_per_dist.get(nt) == gaussian_best
                        for nt in NOISE_TYPES if nt in best_per_dist)
    else:
        transfers = None

    return convert_numpy({
        'best_method_per_distribution': best_per_dist,
        'gaussian_rank_transfers': transfers,
        'hybrid_advantage_persists': hybrid_wins > hybrid_total / 2 if hybrid_total > 0 else None,
        'hybrid_wins': hybrid_wins,
        'hybrid_total': hybrid_total,
    })


if __name__ == '__main__':
    parser = create_experiment_parser('N13: Noise Distributions')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n13_noise_distributions',
                            args.dataset)
