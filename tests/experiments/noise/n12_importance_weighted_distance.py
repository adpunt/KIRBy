#!/usr/bin/env python3
"""N12: Feature Importance-Weighted Distance for Noise Estimation.

Paper: Noise
Focused ablation: does scaling features by sqrt(importance) before computing
kNN distances improve neighbor-based noise estimation? Is it redundant with
hybridization?

5 conditions × 3 estimators × 3 k values × 2 importance methods × 2 sigmas.

Usage:
    python n12_importance_weighted_distance.py --dataset logd --mode intermediary
    python n12_importance_weighted_distance.py --dataset caco2 --mode full
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
from kirby.hybrid import create_hybrid, compute_feature_importance
from kirby.utils.noise_estimation import (
    NeighborResidualEstimator,
    LocalVarianceRatioEstimator,
    ActivityCliffAwareEstimator,
    weights_from_scores,
)


SIGMAS = [0.3, 0.6]
K_VALUES = [5, 10, 20]
IMPORTANCE_METHODS = ['lightgbm_gain', 'treeshap']

ESTIMATOR_FACTORIES = {
    'neighbor_residual': NeighborResidualEstimator,
    'local_variance_ratio': LocalVarianceRatioEstimator,
    'activity_cliff_aware': ActivityCliffAwareEstimator,
}

CONDITION_NAMES = [
    'base_unweighted',
    'base_imp_weighted',
    'hybrid_unweighted',
    'hybrid_imp_base',
    'hybrid_imp_own',
]


def _inject_noise(y, sigma_frac, seed=42, is_clf=False):
    """Inject noise. Regression: Gaussian additive. Classification: label flipping."""
    rng = np.random.RandomState(seed)
    if is_clf:
        n_flip = max(1, int(len(y) * sigma_frac))
        flip_idx = rng.choice(len(y), n_flip, replace=False)
        y_noisy = y.copy()
        classes = np.unique(y)
        for i in flip_idx:
            other = classes[classes != y[i]]
            if len(other) > 0:
                y_noisy[i] = rng.choice(other)
        return y_noisy
    y_range = np.max(y) - np.min(y)
    return y + rng.normal(0, sigma_frac * y_range, len(y))


def _weight_features(X, importance):
    """Apply sqrt(importance) weighting to features."""
    w = np.sqrt(np.maximum(importance, 0))
    w = w / (w.sum() + 1e-10)
    return X * w


def _build_conditions(train_reps, y_train, augmentations, bp,
                       imp_method, base_rep_name):
    """Build all 5 conditions for one importance method.

    Args:
        train_reps: {rep_name: X_train} already sliced to fold.
        y_train: Training labels for this fold.
        augmentations: {aug_name: X_train} already sliced, or None.
        bp: Blueprint from get_precomputed_search.
        imp_method: Importance method name for weighting ablation.
        base_rep_name: Name of base representation.

    Returns {condition_name: X_standardized}.
    """
    conditions = {}
    X_base = train_reps[base_rep_name]

    # Condition 1: base unweighted
    sc1 = StandardScaler()
    conditions['base_unweighted'] = sc1.fit_transform(X_base)

    # Compute base rep importance
    try:
        imp_base = compute_feature_importance(
            {base_rep_name: X_base}, y_train, method=imp_method
        )[base_rep_name]
    except Exception as e:
        print(f"      {imp_method} importance failed on base: {e}")
        imp_base = None

    # Condition 2: base importance-weighted
    if imp_base is not None:
        X_weighted = _weight_features(X_base, imp_base)
        sc2 = StandardScaler()
        conditions['base_imp_weighted'] = sc2.fit_transform(X_weighted)

    # Build hybrid via blueprint
    X_hybrid = None
    try:
        X_hybrid, fi = create_hybrid(
            train_reps, y_train,
            allocation_method=bp.allocation_method,
            importance_method=bp.importance_method,
            internal_evaluator=bp.internal_evaluator,
            budget=bp.budget,
            augmentations=augmentations,
            augmentation_strategy=bp.augmentation_strategy,
        )
    except Exception as e:
        print(f"      hybrid creation failed: {e}")

    if X_hybrid is not None:
        # Condition 3: hybrid unweighted
        sc3 = StandardScaler()
        conditions['hybrid_unweighted'] = sc3.fit_transform(X_hybrid)

        # Condition 4: hybrid weighted by blueprint's importance method
        # The blueprint discovered its own importance method during search;
        # use that to weight hybrid features (may differ from imp_method).
        if imp_base is not None:
            try:
                imp_hybrid_via_blueprint = compute_feature_importance(
                    {'hybrid': X_hybrid}, y_train, method=bp.importance_method
                )['hybrid']
                X_h_weighted = _weight_features(X_hybrid, imp_hybrid_via_blueprint)
                sc4 = StandardScaler()
                conditions['hybrid_imp_base'] = sc4.fit_transform(X_h_weighted)
            except Exception as e:
                print(f"      hybrid_imp_base failed: {e}")

        # Condition 5: hybrid weighted by base rep's importance method
        # Uses the same imp_method as conditions 1-2 (e.g. lightgbm_gain)
        # applied to hybrid features — tests whether re-weighting an
        # already-curated hybrid with an external method adds anything.
        try:
            imp_hybrid_own = compute_feature_importance(
                {'hybrid': X_hybrid}, y_train, method=imp_method
            )['hybrid']
            X_h_own_weighted = _weight_features(X_hybrid, imp_hybrid_own)
            sc5 = StandardScaler()
            conditions['hybrid_imp_own'] = sc5.fit_transform(X_h_own_weighted)
        except Exception as e:
            print(f"      hybrid_imp_own failed: {e}")

    return conditions


def _run_condition(X_space, y_noisy, X_eval_train, X_eval_val, y_val,
                    model_factory, metric_fn):
    """Run 3 estimators × 3 k values on one condition.

    Fixed remediation: exponential weighting, alpha=0.1.

    Returns {f'{est_name}_k{k}': denoised_score}.
    """
    results = {}

    for est_name, est_cls in ESTIMATOR_FACTORIES.items():
        for k in K_VALUES:
            key = f'{est_name}_k{k}'
            try:
                est = est_cls(k=k, metric='euclidean')
                scores = est.estimate(X_space, y_noisy)
                weights = weights_from_scores(scores, strength=0.1,
                                              method='exponential')
                model = model_factory()
                model.fit(X_eval_train, y_noisy, sample_weight=weights)
                y_pred = model.predict(X_eval_val)
                results[key] = float(metric_fn(y_val, y_pred))
            except Exception as e:
                results[key] = None

    return results


def run(data, config, output_dir, dataset_name, **kwargs):
    """N12: Feature importance-weighted distance for noise estimation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N12: IMPORTANCE-WEIGHTED DISTANCE FOR NOISE ESTIMATION")
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

    model_factory = get_model_factories(
        ['rf'], task_type=task_type,
        n_samples=len(labels))['rf']

    sigmas = kwargs.get('sigmas', SIGMAS)
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
    print(f"  K values: {K_VALUES}")
    print(f"  Importance methods: {IMPORTANCE_METHODS}")
    print(f"  Sigmas: {sigmas}, Folds: {n_folds}")

    # Get blueprint ONCE from cached search (uses clean labels)
    search = get_precomputed_search(
        reps, labels, groups, budget=100, aug_condition='uncapped',
        augmentations=augmentations, speed=config.get('search_speed', 'full'),
        precomputed=data.get('precomputed_searches'),
        dataset_name=dataset_name, tier='general')
    best_rf = search.best_by_model.get('rf')
    bp = best_rf['blueprint'] if best_rf else None
    if bp is None:
        print("  WARNING: No blueprint found, skipping")
        return {'error': 'no_blueprint', 'metric': metric_name}

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'base_rep': base_rep_name,
        'k_values': K_VALUES,
        'importance_methods': IMPORTANCE_METHODS,
        'sigmas': sigmas,
        'per_sigma': {},
    }

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        clean_scores = []
        noisy_scores = []
        # {imp_method: {condition: {est_k: [scores]}}}
        grid_acc = {}

        fold_idx = 0
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                            labels, mode=mode):
            fold_idx += 1
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            # Fixed eval features
            X_eval_train = reps[base_rep_name][train_idx]
            X_eval_val = reps[base_rep_name][val_idx]
            scaler_eval = StandardScaler()
            X_eval_train_s = scaler_eval.fit_transform(X_eval_train)
            X_eval_val_s = scaler_eval.transform(X_eval_val)

            # Clean baseline
            rf_clean = model_factory()
            rf_clean.fit(X_eval_train_s, y_train_clean)
            clean_scores.append(float(metric_fn(y_val, rf_clean.predict(X_eval_val_s))))

            # Inject noise
            y_noisy = _inject_noise(y_train_clean, sigma, seed=42 + fold_idx, is_clf=is_clf)

            # Noisy baseline
            rf_noisy = model_factory()
            rf_noisy.fit(X_eval_train_s, y_noisy)
            noisy_scores.append(float(metric_fn(y_val, rf_noisy.predict(X_eval_val_s))))

            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
            train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None

            for imp_method in IMPORTANCE_METHODS:
                if imp_method not in grid_acc:
                    grid_acc[imp_method] = {}

                conditions = _build_conditions(
                    train_reps_fold, y_noisy, train_augs, bp,
                    imp_method, base_rep_name,
                )

                for cond_name, X_space in conditions.items():
                    if cond_name not in grid_acc[imp_method]:
                        grid_acc[imp_method][cond_name] = {}

                    cond_results = _run_condition(
                        X_space, y_noisy, X_eval_train_s, X_eval_val_s,
                        y_val, model_factory, metric_fn,
                    )

                    for ek, score in cond_results.items():
                        if ek not in grid_acc[imp_method][cond_name]:
                            grid_acc[imp_method][cond_name][ek] = []
                        if score is not None:
                            grid_acc[imp_method][cond_name][ek].append(score)

        # Aggregate
        clean_mean = float(np.mean(clean_scores))
        noisy_mean = float(np.mean(noisy_scores))
        gap = clean_mean - noisy_mean

        sigma_result = {
            'clean_score': clean_mean,
            'noisy_score': noisy_mean,
            'gap': float(gap),
            'per_importance_method': {},
        }

        for imp_method in IMPORTANCE_METHODS:
            imp_result = {}
            for cond_name in CONDITION_NAMES:
                cond_agg = {}
                for ek, score_list in grid_acc.get(imp_method, {}).get(cond_name, {}).items():
                    if score_list:
                        denoised = float(np.mean(score_list))
                        recovery = float((denoised - noisy_mean) / gap * 100) if abs(gap) > 1e-10 else float('nan')
                        cond_agg[ek] = {
                            'denoised': denoised,
                            'recovery': recovery,
                        }
                if cond_agg:
                    imp_result[cond_name] = cond_agg
            sigma_result['per_importance_method'][imp_method] = imp_result

        results['per_sigma'][str(sigma)] = convert_numpy(sigma_result)

        # Print compact summary
        for imp_method in IMPORTANCE_METHODS:
            print(f"    {imp_method}:")
            imp_data = sigma_result['per_importance_method'].get(imp_method, {})
            for cond_name in CONDITION_NAMES:
                cond_data = imp_data.get(cond_name, {})
                if cond_data:
                    recoveries = [v['recovery'] for v in cond_data.values()
                                  if not np.isnan(v.get('recovery', float('nan')))]
                    avg_r = float(np.mean(recoveries)) if recoveries else float('nan')
                    n_pos = sum(1 for r in recoveries if r > 0)
                    print(f"      {cond_name:25s}: avg_recovery={avg_r:+.1f}%, "
                          f"positive={n_pos}/{len(recoveries)}")
        print(f"    [{time.time() - t_sigma:.0f}s]")

    # Summary
    results['summary'] = _compute_summary(results)

    s = results['summary']
    print(f"\n  === SUMMARY ===")
    for sk, sv in s.items():
        if isinstance(sv, dict) and 'weighting_helps_base' in sv:
            print(f"  {sk}:")
            print(f"    Weighting helps base: {sv['weighting_helps_base']}")
            print(f"    Weighting helps hybrid: {sv['weighting_helps_hybrid']}")
            print(f"    Hybrid already weighted: {sv['hybrid_already_weighted']}")
            print(f"    Best k: {sv.get('best_k')}")
            print(f"    Best importance method: {sv.get('best_importance_method')}")

    return results


def _compute_summary(results):
    """Compute ablation answers."""
    summary = {}

    for sigma_key, sigma_data in results['per_sigma'].items():
        # Collect per-condition average recovery across importance methods
        cond_recoveries = {}  # {condition: [all recoveries]}

        for imp_method, imp_data in sigma_data.get('per_importance_method', {}).items():
            for cond_name, cond_data in imp_data.items():
                if cond_name not in cond_recoveries:
                    cond_recoveries[cond_name] = []
                for ek, rv in cond_data.items():
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r):
                        cond_recoveries[cond_name].append(r)

        cond_avg = {c: float(np.mean(rs)) if rs else float('nan')
                    for c, rs in cond_recoveries.items()}

        # Ablation answers
        base_uw = cond_avg.get('base_unweighted', float('nan'))
        base_w = cond_avg.get('base_imp_weighted', float('nan'))
        hyb_uw = cond_avg.get('hybrid_unweighted', float('nan'))
        hyb_w_base = cond_avg.get('hybrid_imp_base', float('nan'))
        hyb_w_own = cond_avg.get('hybrid_imp_own', float('nan'))

        weighting_helps_base = (not np.isnan(base_w) and not np.isnan(base_uw)
                                and base_w > base_uw)
        weighting_helps_hybrid = (not np.isnan(hyb_w_own) and not np.isnan(hyb_uw)
                                  and (hyb_w_own > hyb_uw or hyb_w_base > hyb_uw))
        # Redundant if hybrid_unweighted ≈ hybrid_imp_own (within 2%)
        hybrid_already_weighted = (not np.isnan(hyb_uw) and not np.isnan(hyb_w_own)
                                   and abs(hyb_uw - hyb_w_own) < 2.0)

        # Best k
        k_recoveries = {}
        for imp_data in sigma_data.get('per_importance_method', {}).values():
            for cond_data in imp_data.values():
                for ek, rv in cond_data.items():
                    # Parse k from key like 'neighbor_residual_k10'
                    if '_k' in ek:
                        k_val = int(ek.split('_k')[-1])
                        if k_val not in k_recoveries:
                            k_recoveries[k_val] = []
                        r = rv.get('recovery', float('nan'))
                        if not np.isnan(r):
                            k_recoveries[k_val].append(r)

        best_k = None
        best_k_avg = float('-inf')
        for k_val, rs in k_recoveries.items():
            avg = float(np.mean(rs))
            if avg > best_k_avg:
                best_k_avg = avg
                best_k = k_val

        # Best importance method
        method_recoveries = {}
        for imp_method, imp_data in sigma_data.get('per_importance_method', {}).items():
            all_r = []
            for cond_data in imp_data.values():
                for rv in cond_data.values():
                    r = rv.get('recovery', float('nan'))
                    if not np.isnan(r):
                        all_r.append(r)
            if all_r:
                method_recoveries[imp_method] = float(np.mean(all_r))

        best_imp = max(method_recoveries, key=method_recoveries.get) if method_recoveries else None

        summary[f'sigma_{sigma_key}'] = convert_numpy({
            'weighting_helps_base': weighting_helps_base,
            'weighting_helps_hybrid': weighting_helps_hybrid,
            'hybrid_already_weighted': hybrid_already_weighted,
            'best_k': best_k,
            'best_importance_method': best_imp,
            'per_condition_avg_recovery': cond_avg,
            'per_k_avg_recovery': {str(k): float(np.mean(rs))
                                    for k, rs in k_recoveries.items() if rs},
            'per_method_avg_recovery': method_recoveries,
        })

    return summary


if __name__ == '__main__':
    parser = create_experiment_parser('N12: Importance-Weighted Distance')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n12_importance_weighted_distance',
                            args.dataset)
