#!/usr/bin/env python3
"""N11: Noise Estimation × Remediation Grid.

Paper: Noise
Full grid from the confirmation report: estimator × smoothing/weighting × alpha ×
distance metric × representation. Extends with hybrid representations.

~1,120 configs per sigma level. Each config: estimate noise scores, apply
remediation (smoothing or weighting), train RF on ECFP4 with remediated data,
compute recovery rate.

Usage:
    python n11_estimation_remediation_grid.py --dataset logd --mode intermediary
    python n11_estimation_remediation_grid.py --dataset caco2 --mode full
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
    get_precomputed_search, cap_dimensions,
)
from kirby.hybrid import create_hybrid
from kirby.utils.noise_estimation import (
    get_noise_estimator, smooth_by_scores, weights_from_scores,
)


SIGMAS = [0.3, 0.6]
ALPHAS = [0.1, 0.3, 0.5, 0.7]
WEIGHT_METHODS = ['inverse', 'exponential', 'soft']
SMOOTH_METHODS = ['proportional', 'threshold', 'soft_threshold', 'confidence']
NEIGHBOR_METRICS = ['euclidean', 'cosine']

# Estimators that use kNN internally — test with multiple distance metrics
NEIGHBOR_ESTIMATORS = {
    'neighbor_residual', 'local_variance_ratio',
    'activity_cliff_aware', 'distance_weighted', 'hybrid',
}

# All estimators to try (torch-dependent ones are try-imported)
BASE_ESTIMATORS = [
    'neighbor_residual', 'local_variance_ratio', 'activity_cliff_aware',
    'distance_weighted', 'hybrid',
    'cross_val', 'ensemble', 'leave_one_out',
    'gaussian_process', 'conformal', 'bayesian_rf',
]
TORCH_ESTIMATORS = ['heteroscedastic_nn', 'deep_ensemble', 'quantile_regression']
OPTIONAL_ESTIMATORS = ['ngboost']

# GP max samples
GP_MAX = 1000


def _inject_noise(y, sigma_frac, seed=42, is_clf=False):
    """Inject noise. Regression: Gaussian additive. Classification: label flipping."""
    rng = np.random.RandomState(seed)
    n = len(y)
    if is_clf:
        n_flip = max(1, int(n * sigma_frac))
        y_noisy = y.copy()
        flip_idx = rng.choice(n, n_flip, replace=False)
        classes = np.unique(y)
        for i in flip_idx:
            other = classes[classes != y[i]]
            if len(other) > 0:
                y_noisy[i] = rng.choice(other)
        return y_noisy
    else:
        y_range = np.max(y) - np.min(y)
        noise = rng.normal(0, sigma_frac * y_range, n)
        return y + noise


def _get_correction_target(estimator):
    """Extract correction target from estimator for smoothing methods.

    Returns (target_array, source_name) or (None, None) if no target available.
    """
    # Neighbor-based: neighbor_means_ or weighted_means_
    for attr in ['neighbor_means_', 'weighted_means_']:
        target = getattr(estimator, attr, None)
        if target is not None:
            return target, attr

    # Model-based: various prediction attributes
    for attr in ['mu_', 'cv_predictions_', 'prediction_means_',
                 'oob_predictions_', 'predictions_']:
        target = getattr(estimator, attr, None)
        if target is not None:
            return target, attr

    return None, None


def _build_estimation_spaces(train_reps, y_train, augmentations, bp):
    """Build estimation spaces: base rep + hybrid.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations: {aug_name: X_train} already sliced, or None.
        bp: Blueprint from get_precomputed_search.

    Returns (spaces_dict, base_rep_name).
    """
    spaces = {}

    # Find base rep
    base_rep_name = None
    for candidate in ['ecfp4', 'pdv', 'blosum']:
        if candidate in train_reps:
            base_rep_name = candidate
            break
    if base_rep_name is None:
        base_rep_name = sorted(train_reps.keys())[0]

    X_base = train_reps[base_rep_name]
    scaler = StandardScaler()
    spaces[f'{base_rep_name}_raw'] = scaler.fit_transform(X_base)

    # Hybrid via blueprint
    try:
        X_h, fi = create_hybrid(
            train_reps, y_train,
            allocation_method=bp.allocation_method,
            importance_method=bp.importance_method,
            internal_evaluator=bp.internal_evaluator,
            budget=bp.budget,
            augmentations=augmentations,
            augmentation_strategy=bp.augmentation_strategy,
        )
        scaler_h = StandardScaler()
        spaces['hybrid_100'] = scaler_h.fit_transform(X_h)
    except Exception as e:
        print(f"      hybrid_100 failed: {e}")

    # All base rep spaces
    for rep_name, X_rep in train_reps.items():
        scaler = StandardScaler()
        spaces[rep_name] = cap_dimensions(scaler.fit_transform(X_rep))[0]

    return spaces, base_rep_name


def _resolve_estimators():
    """Determine which estimators are available (handle torch/ngboost imports)."""
    available = list(BASE_ESTIMATORS)

    try:
        import torch  # noqa: F401
        available.extend(TORCH_ESTIMATORS)
    except ImportError:
        pass

    for opt in OPTIONAL_ESTIMATORS:
        try:
            get_noise_estimator(opt)
            available.append(opt)
        except Exception:
            pass

    return available


def _run_estimator_grid(est_name, metric_name, X_space, y_noisy,
                         X_eval_train, X_eval_val, y_val,
                         model_factory, metric_fn, n_samples):
    """Run full alpha × remediation grid for one estimator config.

    Returns: {remed_key: denoised_score} where remed_key = f'{method}_{alpha}'
    """
    # Build estimator with appropriate kwargs
    kwargs = {}
    if est_name in NEIGHBOR_ESTIMATORS:
        kwargs['metric'] = metric_name
        kwargs['k'] = 10
    if est_name == 'gaussian_process' and n_samples > GP_MAX:
        # GP will subsample internally, but we should warn
        pass

    try:
        estimator = get_noise_estimator(est_name, **kwargs)
        scores = estimator.estimate(X_space, y_noisy)
    except Exception as e:
        return {'_error': str(e)}

    target, target_source = _get_correction_target(estimator)
    results = {}

    for alpha in ALPHAS:
        # Weighting methods (always available)
        for wm in WEIGHT_METHODS:
            key = f'{wm}_{alpha}'
            try:
                weights = weights_from_scores(scores, strength=alpha, method=wm)
                model = model_factory()
                model.fit(X_eval_train, y_noisy, sample_weight=weights)
                y_pred = model.predict(X_eval_val)
                results[key] = float(metric_fn(y_val, y_pred))
            except Exception:
                pass

        # Smoothing methods (need a correction target)
        if target is not None:
            for sm in SMOOTH_METHODS:
                key = f'{sm}_{alpha}'
                try:
                    y_smoothed = smooth_by_scores(
                        y_noisy, scores, target,
                        alpha=alpha, method=sm,
                    )
                    model = model_factory()
                    model.fit(X_eval_train, y_smoothed)
                    y_pred = model.predict(X_eval_val)
                    results[key] = float(metric_fn(y_val, y_pred))
                except Exception:
                    pass

    return results


def run(data, config, output_dir, dataset_name, **kwargs):
    """N11: Noise estimation × remediation grid."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N11: NOISE ESTIMATION × REMEDIATION GRID")
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
    n_samples = len(labels)

    model_factory = get_model_factories(
        ['rf'], task_type=task_type,
        n_samples=n_samples)['rf']

    sigmas = kwargs.get('sigmas', SIGMAS)
    sample_X = next(iter(reps.values()))

    available_estimators = _resolve_estimators()
    # GP is regression-only
    if is_clf:
        available_estimators = [e for e in available_estimators
                                if e != 'gaussian_process']

    # Resolve base rep name once (same logic as _build_estimation_spaces)
    base_rep_name = None
    for candidate in ['ecfp4', 'pdv', 'blosum']:
        if candidate in reps:
            base_rep_name = candidate
            break
    if base_rep_name is None:
        base_rep_name = sorted(reps.keys())[0]

    print(f"  Dataset: {dataset_name} ({n_samples} samples)")
    print(f"  Metric: {metric_name}")
    print(f"  Estimators: {len(available_estimators)}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Sigmas: {sigmas}")
    print(f"  Folds: {n_folds}")

    # Get blueprint ONCE from cached search (uses clean labels)
    search = get_precomputed_search(
        reps, labels, groups, budget=100, aug_condition='uncapped',
        augmentations=augmentations, speed=config.get('search_speed', 'full'),
        precomputed=data.get('precomputed_searches'))
    best_rf = search.best_by_model.get('rf')
    bp = best_rf['blueprint'] if best_rf else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'estimators': available_estimators,
        'alphas': ALPHAS,
        'sigmas': sigmas,
        'per_sigma': {},
    }

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        # Accumulators across folds
        clean_scores = []
        noisy_scores = []
        # {space: {est_key: {remed_key: [scores]}}}
        grid_acc = {}

        fold_idx = 0
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                            labels, mode=mode):
            fold_idx += 1
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}

            # Fixed evaluation features (base rep — independent of labels)
            X_eval_train = reps[base_rep_name][train_idx]
            X_eval_val = reps[base_rep_name][val_idx]
            scaler_eval = StandardScaler()
            X_eval_train_s = scaler_eval.fit_transform(X_eval_train)
            X_eval_val_s = scaler_eval.transform(X_eval_val)

            # Clean baseline
            rf_clean = model_factory()
            rf_clean.fit(X_eval_train_s, y_train_clean)
            clean_scores.append(float(metric_fn(y_val, rf_clean.predict(X_eval_val_s))))

            # Inject noise BEFORE building estimation spaces (no oracle leak)
            y_noisy = _inject_noise(y_train_clean, sigma, seed=42 + fold_idx,
                                    is_clf=is_clf)

            # Slice augmentations to train fold
            train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None

            # Build estimation spaces from noisy labels (realistic scenario)
            spaces, _brn = _build_estimation_spaces(
                train_reps_fold, y_noisy, train_augs, bp,
            )

            # Noisy baseline
            rf_noisy = model_factory()
            rf_noisy.fit(X_eval_train_s, y_noisy)
            noisy_scores.append(float(metric_fn(y_val, rf_noisy.predict(X_eval_val_s))))

            # Run grid per space
            for space_name, X_space in spaces.items():
                if space_name not in grid_acc:
                    grid_acc[space_name] = {}

                for est_name in available_estimators:
                    # Determine which metrics to test
                    if est_name in NEIGHBOR_ESTIMATORS:
                        metrics_to_test = NEIGHBOR_METRICS
                    else:
                        metrics_to_test = [None]  # no metric variation

                    for metric_var in metrics_to_test:
                        est_key = est_name
                        if metric_var:
                            est_key = f'{est_name}_{metric_var}'

                        grid_result = _run_estimator_grid(
                            est_name, metric_var or 'euclidean',
                            X_space, y_noisy,
                            X_eval_train_s, X_eval_val_s, y_val,
                            model_factory, metric_fn, n_samples,
                        )

                        if est_key not in grid_acc[space_name]:
                            grid_acc[space_name][est_key] = {}

                        if '_error' in grid_result:
                            continue

                        for remed_key, score in grid_result.items():
                            if remed_key not in grid_acc[space_name][est_key]:
                                grid_acc[space_name][est_key][remed_key] = []
                            grid_acc[space_name][est_key][remed_key].append(score)

        # Aggregate across folds
        clean_mean = float(np.mean(clean_scores))
        noisy_mean = float(np.mean(noisy_scores))
        gap = clean_mean - noisy_mean

        sigma_result = {
            'clean_score': clean_mean,
            'noisy_score': noisy_mean,
            'gap': float(gap),
            'per_space': {},
        }

        for space_name in sorted(grid_acc.keys()):
            space_agg = {}
            for est_key in sorted(grid_acc[space_name].keys()):
                est_agg = {}
                for remed_key, score_list in grid_acc[space_name][est_key].items():
                    if score_list:
                        denoised = float(np.mean(score_list))
                        if abs(gap) > 1e-10:
                            recovery = float((denoised - noisy_mean) / gap * 100)
                        else:
                            recovery = float('nan')
                        est_agg[remed_key] = {
                            'denoised_score': denoised,
                            'recovery': recovery,
                        }
                if est_agg:
                    space_agg[est_key] = est_agg
            sigma_result['per_space'][space_name] = space_agg

        results['per_sigma'][str(sigma)] = convert_numpy(sigma_result)

        # Print compact summary
        n_total = 0
        n_positive = 0
        best_recovery = float('-inf')
        best_config = None
        for space_name, space_data in sigma_result['per_space'].items():
            for est_key, est_data in space_data.items():
                for rk, rv in est_data.items():
                    n_total += 1
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r) and r > 0:
                        n_positive += 1
                    if not np.isnan(r) and r > best_recovery:
                        best_recovery = r
                        best_config = f'{space_name}/{est_key}/{rk}'

        print(f"    Clean={clean_mean:.4f}, Noisy={noisy_mean:.4f}")
        print(f"    Configs: {n_total} total, {n_positive} positive "
              f"({100*n_positive/max(n_total,1):.1f}%)")
        if best_config:
            print(f"    Best: {best_config} ({best_recovery:+.1f}%)")
        print(f"    [{time.time() - t_sigma:.0f}s]")

    # Summary
    results['summary'] = _compute_summary(results)

    return results


def _compute_summary(results):
    """Compute summary: best configs, alpha effect, space comparison."""
    summary = {}

    for sigma_key, sigma_data in results['per_sigma'].items():
        gap = sigma_data.get('gap', 0)
        all_configs = []

        for space_name, space_data in sigma_data.get('per_space', {}).items():
            for est_key, est_data in space_data.items():
                for remed_key, rv in est_data.items():
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r):
                        # Parse alpha from remed_key (e.g. 'inverse_0.1')
                        parts = remed_key.rsplit('_', 1)
                        alpha = float(parts[1]) if len(parts) == 2 else 0.0
                        method = parts[0] if len(parts) == 2 else remed_key
                        all_configs.append({
                            'space': space_name,
                            'estimator': est_key,
                            'method': method,
                            'alpha': alpha,
                            'recovery': r,
                        })

        if not all_configs:
            summary[f'sigma_{sigma_key}'] = {}
            continue

        # Sort by recovery (descending)
        all_configs.sort(key=lambda x: x['recovery'], reverse=True)

        # Alpha effect
        alpha_recoveries = {}
        for c in all_configs:
            a = str(c['alpha'])
            if a not in alpha_recoveries:
                alpha_recoveries[a] = []
            alpha_recoveries[a].append(c['recovery'])
        alpha_effect = {
            a: {
                'mean_recovery': float(np.mean(rs)),
                'pct_positive': float(sum(1 for r in rs if r > 0) / len(rs) * 100),
            }
            for a, rs in alpha_recoveries.items()
        }

        # Space comparison
        space_recoveries = {}
        for c in all_configs:
            s = c['space']
            if s not in space_recoveries:
                space_recoveries[s] = []
            space_recoveries[s].append(c['recovery'])
        space_effect = {
            s: {
                'mean_recovery': float(np.mean(rs)),
                'pct_positive': float(sum(1 for r in rs if r > 0) / len(rs) * 100),
            }
            for s, rs in space_recoveries.items()
        }

        n_positive = sum(1 for c in all_configs if c['recovery'] > 0)

        s = {
            'n_total': len(all_configs),
            'n_positive': n_positive,
            'pct_positive': float(n_positive / len(all_configs) * 100),
            'best_config': all_configs[0],
            'top_10': all_configs[:10],
            'alpha_effect': alpha_effect,
            'space_effect': space_effect,
        }

        # Robust configs (positive at BOTH sigma levels if we have multiple)
        # This is computed after all sigmas in the final summary merge
        summary[f'sigma_{sigma_key}'] = convert_numpy(s)

    # Cross-sigma robust configs
    sigma_keys = [k for k in results['per_sigma'] if k != '0.0']
    if len(sigma_keys) >= 2:
        # Collect all config keys that are positive at every sigma
        config_recovery = {}  # config_key -> {sigma: recovery}
        for sigma_key in sigma_keys:
            sigma_data = results['per_sigma'][sigma_key]
            for space_name, space_data in sigma_data.get('per_space', {}).items():
                for est_key, est_data in space_data.items():
                    for remed_key, rv in est_data.items():
                        r = rv.get('recovery', float('nan'))
                        if not np.isnan(r):
                            ck = f'{space_name}/{est_key}/{remed_key}'
                            if ck not in config_recovery:
                                config_recovery[ck] = {}
                            config_recovery[ck][sigma_key] = r

        robust = []
        for ck, sr in config_recovery.items():
            if len(sr) >= 2 and all(r > 0 for r in sr.values()):
                avg = float(np.mean(list(sr.values())))
                robust.append({'config': ck, 'avg_recovery': avg,
                               'per_sigma': sr})

        robust.sort(key=lambda x: x['avg_recovery'], reverse=True)
        summary['robust_configs'] = convert_numpy(robust[:20])
        summary['n_robust'] = len(robust)

    return summary


if __name__ == '__main__':
    parser = create_experiment_parser('N11: Estimation × Remediation Grid')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n11_estimation_remediation_grid',
                            args.dataset)
