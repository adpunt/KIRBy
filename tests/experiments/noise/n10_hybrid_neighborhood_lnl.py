#!/usr/bin/env python3
"""N10: Hybrid Neighborhood for Similarity-Dependent LNL.

Paper: Noise
Core finding from confirmation report: ALL 20 worst LNL configs used distance
metrics. Hypothesis: standard representations have poor similarity-activity
correlation, breaking neighbor-based reasoning. Hybrids should fix this.

Key distinction from N09: N09 changes INPUT features; N10 changes the DISTANCE
SPACE for methods that compute neighbors. The evaluation model uses ECFP4 + RF
by default (only the neighbor space varies), with an additional weighted hybrid
eval condition that uses importance-weighted hybrid features as eval input.

6 neighbor space conditions × 3 estimators × 4 remediations × 2 sigmas
+ 9 binary detectors × 6 spaces × 2 sigmas
+ weighted hybrid eval (search-derived importance weights as eval features)

Usage:
    python n10_hybrid_neighborhood_lnl.py --dataset logd --mode intermediary
    python n10_hybrid_neighborhood_lnl.py --dataset caco2 --mode full
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
    get_precomputed_search, build_fold_hybrid,
    extract_top3_base_reps, cap_dimensions,
)
from kirby.hybrid import create_hybrid, compute_feature_importance
from kirby.utils.noise_estimation import (
    NeighborResidualEstimator,
    LocalVarianceRatioEstimator,
    ActivityCliffAwareEstimator,
    smooth_by_scores,
    weights_from_scores,
)
from kirby.utils.noise_mitigation_methods import (
    get_mitigation_method, compute_distance_matrix,
    DISTANCE_AWARE_METHODS,
)


SIGMAS = [0.3, 0.6]

ESTIMATOR_CONFIGS = {
    'neighbor_residual': lambda: NeighborResidualEstimator(k=10, metric='euclidean'),
    'local_variance_ratio': lambda: LocalVarianceRatioEstimator(k=10, metric='euclidean'),
    'activity_cliff_aware': lambda: ActivityCliffAwareEstimator(k=10, metric='euclidean'),
}

# Weighting: (method, strength) — confirmation report showed alpha=0.1 dominates
WEIGHT_CONFIGS = [
    ('inverse', 0.1),
    ('exponential', 0.1),
    ('soft', 0.1),
]

# Smoothing: (method, alpha)
SMOOTH_CONFIGS = [
    ('proportional', 0.1),
]

# All 9 distance-aware binary detectors
DETECTION_METHODS = [
    'knn_k5', 'knn_k10', 'knn_k20',
    'lof', 'lof_k10', 'lof_k20',
    'distance_weighted', 'neighborhood_consensus', 'activity_cliffs',
]


def _inject_noise(y, sigma_frac, seed=42, is_clf=False):
    """Inject noise. Returns (y_noisy, noise_vector).
    Classification: label flipping (noise_vector is zero where unchanged)."""
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
        noise = np.zeros(n)  # placeholder — use y_noisy != y for mask
        return y_noisy, noise
    else:
        y_range = np.max(y) - np.min(y)
        noise = rng.normal(0, sigma_frac * y_range, n)
        return y + noise, noise


def _build_neighbor_spaces(train_reps, y_train, blueprints, train_augs):
    """Build all neighbor space conditions from training data.

    Each space is standardized (zero mean, unit variance). Downstream
    compute_distance_matrix calls should use standardize=False to avoid
    double-standardization.

    Args:
        train_reps: {rep_name: X_train} for this fold.
        y_train: Training labels for this fold.
        blueprints: {budget: blueprint} from get_precomputed_search.
        train_augs: {aug_name: X_train} sliced augmentations, or None.

    Returns:
        dict: {space_name: X_standardized}
    """
    spaces = {}

    # 1. ecfp4_raw
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

    # 2-4. Hybrids at different budgets (using precomputed blueprints)
    hybrids = {}  # budget -> X_hybrid_raw
    hybrid_fis = {}  # budget -> feat_info (for per-fold importance weights)
    for budget, bp in blueprints.items():
        try:
            X_h, fi = create_hybrid(
                train_reps, y_train,
                allocation_method=bp.allocation_method,
                importance_method=bp.importance_method,
                internal_evaluator=bp.internal_evaluator,
                budget=bp.budget,
                augmentations=train_augs,
                augmentation_strategy=bp.augmentation_strategy,
            )
            scaler_h = StandardScaler()
            spaces[f'hybrid_{budget}'] = scaler_h.fit_transform(X_h)
            hybrids[budget] = X_h
            hybrid_fis[budget] = fi
        except Exception as e:
            print(f"      hybrid_{budget} failed: {e}")

    # 5. Importance-weighted base rep
    try:
        imp = compute_feature_importance(
            {base_rep_name: X_base}, y_train, method='lightgbm_gain'
        )[base_rep_name]
        weights = np.sqrt(np.maximum(imp, 0))
        weights = weights / (weights.sum() + 1e-10)
        X_weighted = X_base * weights
        scaler_w = StandardScaler()
        spaces[f'imp_weighted_{base_rep_name}'] = scaler_w.fit_transform(X_weighted)
    except Exception as e:
        print(f"      imp_weighted_{base_rep_name} failed: {e}")

    # 6. Per-fold importance-weighted hybrid (budget=100)
    if 100 in hybrids and 100 in hybrid_fis:
        try:
            from experiments._shared import flatten_fold_weights
            X_h100 = hybrids[100]
            fold_w = flatten_fold_weights(hybrid_fis[100])
            if len(fold_w) == X_h100.shape[1]:
                w = np.sqrt(np.maximum(fold_w, 0))
                w = w / (w.sum() + 1e-10)
                X_h_weighted = X_h100 * w
                scaler_hw = StandardScaler()
                spaces['imp_weighted_hybrid'] = scaler_hw.fit_transform(X_h_weighted)
            else:
                print(f"      imp_weighted_hybrid: weight dim mismatch "
                      f"({len(fold_w)} vs {X_h100.shape[1]})")
        except Exception as e:
            print(f"      imp_weighted_hybrid failed: {e}")
    elif 100 in hybrids:
        # Fallback to post-hoc importance if no fitted result available
        try:
            X_h100 = hybrids[100]
            imp_h = compute_feature_importance(
                {'hybrid': X_h100}, y_train, method='lightgbm_gain'
            )['hybrid']
            weights_h = np.sqrt(np.maximum(imp_h, 0))
            weights_h = weights_h / (weights_h.sum() + 1e-10)
            X_h_weighted = X_h100 * weights_h
            scaler_hw = StandardScaler()
            spaces['imp_weighted_hybrid'] = scaler_hw.fit_transform(X_h_weighted)
        except Exception as e:
            print(f"      imp_weighted_hybrid (fallback) failed: {e}")

    # Per-rep neighbor spaces
    for rep_name, X_rep in train_reps.items():
        scaler = StandardScaler()
        spaces[rep_name] = cap_dimensions(scaler.fit_transform(X_rep))[0]

    return spaces, base_rep_name


def _run_estimation_remediation(X_space, X_eval_train, X_eval_val,
                                 y_noisy, y_val, model_factory, metric_fn):
    """Run estimators + remediation for one neighbor space.

    Estimators compute neighbors in X_space.
    Evaluation model trains on X_eval_train (fixed ECFP4 features).

    Returns:
        {est_name: {remed_key: {'denoised_score': float}}}
    """
    results = {}

    for est_name, est_factory in ESTIMATOR_CONFIGS.items():
        est = est_factory()
        try:
            scores = est.estimate(X_space, y_noisy)
            neighbor_means = est.neighbor_means_
        except Exception as e:
            results[est_name] = {'error': str(e)}
            continue

        est_results = {}

        # Weighting remediations
        for weight_method, strength in WEIGHT_CONFIGS:
            key = f'{weight_method}_{strength}'
            try:
                weights = weights_from_scores(scores, strength=strength,
                                              method=weight_method)
                model = model_factory()
                model.fit(X_eval_train, y_noisy, sample_weight=weights)
                y_pred = model.predict(X_eval_val)
                est_results[key] = {
                    'denoised_score': float(metric_fn(y_val, y_pred)),
                }
            except Exception as e:
                est_results[key] = {'error': str(e)}

        # Smoothing remediation
        for smooth_method, alpha in SMOOTH_CONFIGS:
            key = f'{smooth_method}_{alpha}'
            try:
                y_smoothed = smooth_by_scores(
                    y_noisy, scores, neighbor_means,
                    alpha=alpha, method=smooth_method,
                )
                model = model_factory()
                model.fit(X_eval_train, y_smoothed)
                y_pred = model.predict(X_eval_val)
                est_results[key] = {
                    'denoised_score': float(metric_fn(y_val, y_pred)),
                }
            except Exception as e:
                est_results[key] = {'error': str(e)}

        results[est_name] = est_results

    return results


def _run_detection(X_space, y_noisy, noise_mask):
    """Run binary detectors for one neighbor space.

    Distance matrix computed in X_space (already standardized).

    Returns:
        {method_name: {precision, recall, f1, n_flagged}}
    """
    # standardize=False because X_space is already standardized
    D = compute_distance_matrix(X_space, standardize=False)
    n_noisy = int(noise_mask.sum())
    results = {}

    for method_name in DETECTION_METHODS:
        try:
            detector = get_mitigation_method(
                method_name, precomputed_distances=D)
            flagged_idx = detector.identify_noise(X_space, y_noisy)
            flagged_mask = np.zeros(len(y_noisy), dtype=bool)
            flagged_mask[flagged_idx] = True

            tp = int((flagged_mask & noise_mask).sum())
            n_flagged = int(flagged_mask.sum())
            precision = tp / max(n_flagged, 1)
            recall = tp / max(n_noisy, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            results[method_name] = {
                'n_flagged': n_flagged,
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            }
        except Exception as e:
            results[method_name] = {'error': str(e)}

    return results


def _compute_summary(results):
    """Compute summary answering the 5 critical questions."""
    summary = {}

    for sigma_key, sigma_data in results['per_sigma'].items():
        s = {}

        # Collect per-space average recovery
        space_avg_recovery = {}
        space_avg_f1 = {}
        space_positive_count = {}

        for space_name, space_data in sigma_data.get('per_space', {}).items():
            # Average recovery across estimators × remediations
            recoveries = []
            est_data = space_data.get('estimation', {})
            for en, remed_dict in est_data.items():
                if isinstance(remed_dict, dict) and 'error' not in remed_dict:
                    for rk, rv in remed_dict.items():
                        if isinstance(rv, dict) and 'recovery' in rv:
                            r = rv['recovery']
                            if not np.isnan(r):
                                recoveries.append(r)

            if recoveries:
                space_avg_recovery[space_name] = float(np.mean(recoveries))
                space_positive_count[space_name] = sum(1 for r in recoveries if r > 0)

            # Average F1 across detectors
            f1s = []
            det_data = space_data.get('detection', {})
            for mn, mv in det_data.items():
                if isinstance(mv, dict) and 'f1' in mv:
                    f = mv['f1']
                    if not np.isnan(f):
                        f1s.append(f)
            if f1s:
                space_avg_f1[space_name] = float(np.mean(f1s))

        # Q1: Best space for estimation
        s['best_space_estimation'] = (
            max(space_avg_recovery, key=space_avg_recovery.get)
            if space_avg_recovery else None
        )
        s['space_avg_recovery'] = space_avg_recovery

        # Q2: Best space for detection
        s['best_space_detection'] = (
            max(space_avg_f1, key=space_avg_f1.get)
            if space_avg_f1 else None
        )
        s['space_avg_f1'] = space_avg_f1

        # Q3: Does hybrid rescue distance methods?
        # Compare positive recovery count: any hybrid > ecfp4
        ecfp4_key = [k for k in space_positive_count if 'raw' in k]
        ecfp4_pos = space_positive_count.get(ecfp4_key[0], 0) if ecfp4_key else 0
        hybrid_pos = {k: v for k, v in space_positive_count.items()
                      if 'hybrid' in k}
        best_hybrid_pos = max(hybrid_pos.values()) if hybrid_pos else 0
        s['hybrid_rescues'] = best_hybrid_pos > ecfp4_pos
        s['n_positive_ecfp4'] = ecfp4_pos
        s['n_positive_per_space'] = space_positive_count

        # Q4: Budget effect
        budget_effect = {}
        for b in [25, 100, 200]:
            key = f'hybrid_{b}'
            if key in space_avg_recovery:
                budget_effect[str(b)] = space_avg_recovery[key]
        s['budget_effect'] = budget_effect

        # Q5: Importance weighting effect
        s['importance_weighting'] = {
            k: space_avg_recovery.get(k)
            for k in space_avg_recovery if 'imp_weighted' in k or 'raw' in k
        }

        summary[f'sigma_{sigma_key}'] = s

    return convert_numpy(summary)


def run(data, config, output_dir, dataset_name, **kwargs):
    """N10: Hybrid neighborhood for similarity-dependent LNL."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N10: HYBRID NEIGHBORHOOD FOR SIMILARITY-DEPENDENT LNL")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    task_type = data.get('task_type', 'regression')
    is_clf = task_type == 'classification'
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    model_factory = get_model_factories(
        ['rf'], task_type=task_type,
        n_samples=len(labels))['rf']

    sigmas = kwargs.get('sigmas', SIGMAS)
    sample_X = next(iter(reps.values()))

    # Precompute hybrid blueprints ONCE (cached) for all 3 budgets
    hybrid_budgets = [25, 100, 200]
    blueprints = {}
    search_for_top3 = None
    for b in hybrid_budgets:
        try:
            search = get_precomputed_search(
                reps, labels, groups, budget=b,
                aug_condition='uncapped', augmentations=augmentations,
                speed=config.get('search_speed', 'fast'),
                precomputed=data.get('precomputed_searches'))
            best_rf = search.best_by_model.get('rf')
            if best_rf:
                blueprints[b] = best_rf['blueprint']
            if b == 100:
                search_for_top3 = search
        except Exception as e:
            print(f"  Blueprint for budget={b} failed: {e}")

    top3 = extract_top3_base_reps(search_for_top3) if search_for_top3 is not None else None
    if top3:
        print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

    print(f"  Dataset: {dataset_name} ({len(labels)} samples)")
    print(f"  Metric: {metric_name}")
    print(f"  Reps: {sorted(reps.keys())}")
    print(f"  Sigmas: {sigmas}")
    print(f"  Folds: {n_folds}")
    print(f"  Blueprints: {sorted(blueprints.keys())}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'neighbor_spaces': [],
        'estimators': list(ESTIMATOR_CONFIGS.keys()),
        'detection_methods': DETECTION_METHODS,
        'sigmas': sigmas,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        'per_sigma': {},
    }

    for sigma in sigmas:
        print(f"\n  === Sigma={sigma} ===")
        t_sigma = time.time()

        # Accumulators across folds
        clean_scores = []
        noisy_scores = []
        # {space: {est: {remed: [scores]}}}
        space_est_acc = {}
        # {space: {method: {'f1': [], 'precision': [], 'recall': []}}}
        space_det_acc = {}
        # Weighted hybrid eval accumulators
        whybrid_clean_scores = []
        whybrid_noisy_scores = []
        # {space: {est: {remed: [scores]}}} — same neighbor spaces, weighted hybrid eval
        whybrid_est_acc = {}

        fold_idx = 0
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                            labels, mode=mode):
            fold_idx += 1
            y_train_clean = labels[train_idx]
            y_val = labels[val_idx]

            # Build neighbor spaces
            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
            train_augs = ({k: v[train_idx] for k, v in augmentations.items()}
                          if augmentations else None)

            # Inject noise
            y_noisy, noise_vec = _inject_noise(y_train_clean, sigma,
                                                seed=42 + fold_idx,
                                                is_clf=is_clf)

            spaces, base_rep_name = _build_neighbor_spaces(
                train_reps_fold, y_noisy,
                blueprints, train_augs,
            )

            if not results['neighbor_spaces']:
                results['neighbor_spaces'] = sorted(spaces.keys())

            # Fixed evaluation features (base rep, e.g. ECFP4)
            X_eval_train = reps[base_rep_name][train_idx]
            X_eval_val = reps[base_rep_name][val_idx]
            scaler_eval = StandardScaler()
            X_eval_train_s = scaler_eval.fit_transform(X_eval_train)
            X_eval_val_s = scaler_eval.transform(X_eval_val)

            # Clean baseline
            rf_clean = model_factory()
            rf_clean.fit(X_eval_train_s, y_train_clean)
            clean_scores.append(float(metric_fn(y_val, rf_clean.predict(X_eval_val_s))))

            # Noisy baseline
            rf_noisy = model_factory()
            rf_noisy.fit(X_eval_train_s, y_noisy)
            noisy_scores.append(float(metric_fn(y_val, rf_noisy.predict(X_eval_val_s))))

            # Ground-truth noise mask
            if is_clf:
                noise_mask = (y_noisy != y_train_clean)
            else:
                mad = float(np.median(np.abs(y_train_clean - np.median(y_train_clean))))
                noise_mask = np.abs(noise_vec) > max(mad, 1e-10)

            # Per-space evaluation
            for space_name, X_space in spaces.items():
                # Part A: estimation + remediation
                est_res = _run_estimation_remediation(
                    X_space, X_eval_train_s, X_eval_val_s,
                    y_noisy, y_val, model_factory, metric_fn,
                )
                if space_name not in space_est_acc:
                    space_est_acc[space_name] = {}
                for en, remed_dict in est_res.items():
                    if en not in space_est_acc[space_name]:
                        space_est_acc[space_name][en] = {}
                    if isinstance(remed_dict, dict) and 'error' not in remed_dict:
                        for rk, rv in remed_dict.items():
                            if rk not in space_est_acc[space_name][en]:
                                space_est_acc[space_name][en][rk] = []
                            if isinstance(rv, dict) and 'denoised_score' in rv:
                                space_est_acc[space_name][en][rk].append(
                                    rv['denoised_score'])

                # Part B: binary detection
                det_res = _run_detection(X_space, y_noisy, noise_mask)
                if space_name not in space_det_acc:
                    space_det_acc[space_name] = {}
                for mn, mv in det_res.items():
                    if mn not in space_det_acc[space_name]:
                        space_det_acc[space_name][mn] = {
                            'f1': [], 'precision': [], 'recall': []}
                    if isinstance(mv, dict) and 'error' not in mv:
                        for k in ['f1', 'precision', 'recall']:
                            if k in mv:
                                space_det_acc[space_name][mn][k].append(mv[k])

            # --- Weighted hybrid eval: use imp_weighted_hybrid as EVAL features ---
            bp100 = blueprints.get(100)
            if bp100 is not None:
                val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
                val_augs = ({k: v[val_idx] for k, v in augmentations.items()}
                            if augmentations else None)
                try:
                    # Build train+val hybrid via build_fold_hybrid (proper selection)
                    X_h_train, X_h_val, fi_wh = build_fold_hybrid(
                        train_reps_fold, val_reps_fold, y_noisy, bp100,
                        augmentations_train=train_augs,
                        augmentations_val=val_augs)
                    # Per-fold importance weights (aligned with per-fold columns)
                    from experiments._shared import flatten_fold_weights
                    wh = flatten_fold_weights(fi_wh)
                    if len(wh) == X_h_train.shape[1]:
                        w = np.sqrt(np.maximum(wh, 0))
                        w = w / (w.sum() + 1e-10)
                        scaler_wh = StandardScaler()
                        X_wh_eval_train = scaler_wh.fit_transform(X_h_train * w)
                        X_wh_eval_val = scaler_wh.transform(X_h_val * w)

                        # Clean baseline on weighted hybrid
                        rf_wh_clean = model_factory()
                        rf_wh_clean.fit(X_wh_eval_train, y_train_clean)
                        whybrid_clean_scores.append(
                            float(metric_fn(y_val, rf_wh_clean.predict(X_wh_eval_val))))

                        # Noisy baseline on weighted hybrid
                        rf_wh_noisy = model_factory()
                        rf_wh_noisy.fit(X_wh_eval_train, y_noisy)
                        whybrid_noisy_scores.append(
                            float(metric_fn(y_val, rf_wh_noisy.predict(X_wh_eval_val))))

                        # Per-space estimation with weighted hybrid eval
                        for space_name, X_space in spaces.items():
                            est_res_wh = _run_estimation_remediation(
                                X_space, X_wh_eval_train, X_wh_eval_val,
                                y_noisy, y_val, model_factory, metric_fn,
                            )
                            if space_name not in whybrid_est_acc:
                                whybrid_est_acc[space_name] = {}
                            for en, remed_dict in est_res_wh.items():
                                if en not in whybrid_est_acc[space_name]:
                                    whybrid_est_acc[space_name][en] = {}
                                if isinstance(remed_dict, dict) and 'error' not in remed_dict:
                                    for rk, rv in remed_dict.items():
                                        if rk not in whybrid_est_acc[space_name][en]:
                                            whybrid_est_acc[space_name][en][rk] = []
                                        if isinstance(rv, dict) and 'denoised_score' in rv:
                                            whybrid_est_acc[space_name][en][rk].append(
                                                rv['denoised_score'])
                except Exception as e:
                    print(f"      Weighted hybrid eval failed (fold {fold_idx}): {e}")

        # Aggregate across folds
        clean_mean = float(np.mean(clean_scores))
        noisy_mean = float(np.mean(noisy_scores))
        gap = clean_mean - noisy_mean

        sigma_result = {
            'clean_score': clean_mean,
            'noisy_score': noisy_mean,
            'per_space': {},
        }

        for space_name in sorted(set(list(space_est_acc.keys()) +
                                      list(space_det_acc.keys()))):
            space_agg = {'estimation': {}, 'detection': {}}

            # Estimation aggregation
            for en, remed_dict in space_est_acc.get(space_name, {}).items():
                space_agg['estimation'][en] = {}
                for rk, score_list in remed_dict.items():
                    if score_list:
                        denoised = float(np.mean(score_list))
                        if abs(gap) > 1e-10:
                            recovery = float((denoised - noisy_mean) / gap * 100)
                        else:
                            recovery = float('nan')
                        space_agg['estimation'][en][rk] = {
                            'denoised_score': denoised,
                            'recovery': recovery,
                        }

            # Detection aggregation
            for mn, metric_lists in space_det_acc.get(space_name, {}).items():
                space_agg['detection'][mn] = {
                    k: float(np.mean(v)) if v else float('nan')
                    for k, v in metric_lists.items()
                }

            sigma_result['per_space'][space_name] = space_agg

        # Aggregate weighted hybrid eval results
        if whybrid_clean_scores and whybrid_noisy_scores:
            wh_clean = float(np.mean(whybrid_clean_scores))
            wh_noisy = float(np.mean(whybrid_noisy_scores))
            wh_gap = wh_clean - wh_noisy
            sigma_result['weighted_hybrid_eval'] = {
                'clean_score': wh_clean,
                'noisy_score': wh_noisy,
                'per_space': {},
            }
            for space_name in sorted(whybrid_est_acc.keys()):
                wh_space_agg = {}
                for en, remed_dict in whybrid_est_acc.get(space_name, {}).items():
                    wh_space_agg[en] = {}
                    for rk, score_list in remed_dict.items():
                        if score_list:
                            denoised = float(np.mean(score_list))
                            if abs(wh_gap) > 1e-10:
                                recovery = float((denoised - wh_noisy) / wh_gap * 100)
                            else:
                                recovery = float('nan')
                            wh_space_agg[en][rk] = {
                                'denoised_score': denoised,
                                'recovery': recovery,
                            }
                sigma_result['weighted_hybrid_eval']['per_space'][space_name] = wh_space_agg

        results['per_sigma'][str(sigma)] = convert_numpy(sigma_result)

        # Print summary
        print(f"    Clean={clean_mean:.4f}, Noisy={noisy_mean:.4f}, Gap={gap:.4f}")
        for space_name in sorted(sigma_result['per_space'].keys()):
            est_d = sigma_result['per_space'][space_name]['estimation']
            recoveries = []
            for en_d in est_d.values():
                for rv in en_d.values():
                    if isinstance(rv, dict) and 'recovery' in rv:
                        r = rv['recovery']
                        if not np.isnan(r):
                            recoveries.append(r)
            avg_r = float(np.mean(recoveries)) if recoveries else float('nan')
            n_pos = sum(1 for r in recoveries if r > 0)
            print(f"    {space_name:30s}: avg_recovery={avg_r:+.1f}%, "
                  f"positive={n_pos}/{len(recoveries)}")
        # Print weighted hybrid eval summary
        wh_eval = sigma_result.get('weighted_hybrid_eval')
        if wh_eval:
            wh_c = wh_eval['clean_score']
            wh_n = wh_eval['noisy_score']
            print(f"    -- Weighted hybrid eval: clean={wh_c:.4f}, noisy={wh_n:.4f}")
            for sp_name in sorted(wh_eval.get('per_space', {}).keys()):
                sp_d = wh_eval['per_space'][sp_name]
                recs = []
                for en_d in sp_d.values():
                    for rv in en_d.values():
                        if isinstance(rv, dict) and 'recovery' in rv:
                            r = rv['recovery']
                            if not np.isnan(r):
                                recs.append(r)
                avg = float(np.mean(recs)) if recs else float('nan')
                print(f"    wh_eval/{sp_name:25s}: avg_recovery={avg:+.1f}%")
        print(f"    [{time.time() - t_sigma:.0f}s]")

    # Summary
    results['summary'] = _compute_summary(results)

    print(f"\n  === SUMMARY ===")
    for sigma_key, s in results['summary'].items():
        print(f"  {sigma_key}:")
        print(f"    Best estimation space: {s.get('best_space_estimation')}")
        print(f"    Best detection space: {s.get('best_space_detection')}")
        print(f"    Hybrid rescues: {s.get('hybrid_rescues')}")
        print(f"    Budget effect: {s.get('budget_effect')}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N10: Hybrid Neighborhood LNL')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n10_hybrid_neighborhood_lnl',
                            args.dataset)
