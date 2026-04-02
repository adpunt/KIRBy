#!/usr/bin/env python3
"""S10: Importance Scores as Feature Weights for Distance-Dependent Downstream Tasks.

Paper: Similarity
Conclusion: Importance-weighted distance improves GP, SVR, and q-RASAR-style
    similarity-augmented features beyond kNN (already shown in S05).
Dependencies: S05 (extends kNN), C04 (importance weighting).

Goes beyond kNN (S05) to test importance weighting on GP, SVR, and
similarity-augmented features (q-RASAR style). Four downstream tasks:
  1. Weighted kNN regression (extends S05 with per-rep + hybrid)
  2. GP with importance-weighted RBF kernel
  3. SVR with importance-weighted RBF kernel
  4. Similarity-augmented features (q-RASAR style)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVR
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    cv_split, get_domain_metric, get_model_factories, convert_numpy,
    compute_feature_importance, build_fold_hybrid,
    get_precomputed_search, get_best_for_downstream, extract_top3_base_reps,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)

from kirby.hybrid import apply_feature_selection
from kirby.hybrid import create_hybrid
# TODO: create_hybrid API not finalised — revisit after framework experiments.


# =============================================================================
# CONSTANTS
# =============================================================================

_GP_MAX_SAMPLES = 5000
_SVR_MAX_SAMPLES = 10000


# =============================================================================
# LOCAL HELPERS
# =============================================================================

def _importance_weights(imp_scores):
    """Convert raw importance scores to sqrt-normalised feature weights."""
    w = np.sqrt(np.maximum(imp_scores, 0))
    w = w / (w.sum() + 1e-10)
    return w


def _compute_importance_for_reps(train_reps, y_train, method='lightgbm_gain'):
    """Compute importance and return flat weight vector matching sorted rep concat."""
    imp_dict = compute_feature_importance(train_reps, y_train, method=method)
    imp_concat = np.concatenate([imp_dict[k] for k in sorted(imp_dict.keys())])
    return _importance_weights(imp_concat)


def _compute_importance_for_hybrid(train_reps, y_train, feat_info,
                                   method='lightgbm_gain'):
    """Compute importance weights for hybrid-selected features."""
    imp_dict = compute_feature_importance(train_reps, y_train, method=method)
    w_parts = []
    for rname in feat_info.keys():  # preserve insertion order from create_hybrid
        info = feat_info[rname]
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue
        idx = info['selected_indices']
        if rname in imp_dict:
            w_parts.append(imp_dict[rname][idx])
        else:
            w_parts.append(np.ones(len(idx)))
    return _importance_weights(np.concatenate(w_parts))


def _compute_sim_descriptors(X_train, y_train, X_query, k=10):
    """Compute q-RASAR-style similarity descriptors for each query sample.

    For each query sample, finds k nearest neighbors in X_train and computes:
      - mean(y_neighbors): average target value of neighbors
      - std(y_neighbors): std of target values of neighbors
      - min_distance: distance to nearest neighbor
      - mean_distance: mean distance to k neighbors

    Returns (n_query, 4) array.
    """
    # When X_query IS X_train, add +1 to exclude self-match
    is_self_query = X_query is X_train or (
        X_query.shape == X_train.shape and np.allclose(X_query, X_train))
    k_nn = min(k + (1 if is_self_query else 0), len(y_train) - 1)
    if k_nn < 1:
        return np.zeros((X_query.shape[0], 4))

    nn = NearestNeighbors(n_neighbors=k_nn)
    nn.fit(X_train)
    dists, indices = nn.kneighbors(X_query)

    if is_self_query:
        # Drop the first column (self-match with distance=0)
        dists = dists[:, 1:]
        indices = indices[:, 1:]

    y_neighbors = y_train[indices]  # (n_query, k)
    mean_y = np.mean(y_neighbors, axis=1)
    std_y = np.std(y_neighbors, axis=1)
    min_dist = dists[:, 0]
    mean_dist = np.mean(dists, axis=1)

    return np.column_stack([mean_y, std_y, min_dist, mean_dist])


# =============================================================================
# TASK 1: WEIGHTED kNN REGRESSION
# =============================================================================

def _run_weighted_knn(reps, labels, groups, config, hybrid_blueprint,
                      metric_fn, metric_name, lower_is_better=False,
                      augmentations=None, task_type='regression'):
    """Weighted kNN: per-rep and hybrid, weighted vs unweighted."""
    is_clf = task_type == 'classification'
    print(f"\n  --- Task 1: Weighted kNN {'Classification' if is_clf else 'Regression'} ---")

    k_values = [5, 10, 20]
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    results = {'per_rep': {}, 'hybrid': {}}

    # Per-rep evaluation
    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        rep_results = {}

        for k in k_values:
            uw_scores, w_scores = [], []

            for train_idx, val_idx in cv_split(groups, n_folds, X_rep, labels,
                                               mode=mode):
                y_train, y_val = labels[train_idx], labels[val_idx]
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_rep[train_idx])
                X_va = scaler.transform(X_rep[val_idx])
                k_eff = min(k, len(y_train) - 1)

                # Unweighted
                KNN = KNeighborsClassifier if is_clf else KNeighborsRegressor
                model = KNN(n_neighbors=k_eff)
                model.fit(X_tr, y_train)
                uw_scores.append(metric_fn(y_val, model.predict(X_va)))

                # Weighted
                try:
                    imp = compute_feature_importance(
                        {rep_name: X_rep[train_idx]}, y_train,
                        method='lightgbm_gain')
                    w = _importance_weights(imp[rep_name])
                    model_w = KNN(n_neighbors=k_eff)
                    model_w.fit(X_tr * w, y_train)
                    w_scores.append(metric_fn(y_val, model_w.predict(X_va * w)))
                except Exception:
                    w_scores.append(float('nan'))

            uw_mean = float(np.nanmean(uw_scores))
            w_mean = float(np.nanmean(w_scores))
            rep_results[f'k{k}'] = {
                'unweighted': uw_mean,
                'weighted': w_mean,
                'delta': w_mean - uw_mean,
            }

        results['per_rep'][rep_name] = rep_results
        sign = -1 if lower_is_better else 1
        best_k_key = max(rep_results, key=lambda kk: sign * rep_results[kk]['weighted'])
        best = rep_results[best_k_key]
        print(f"    {rep_name}: best {best_k_key} weighted={best['weighted']:.4f} "
              f"(delta={best['delta']:+.4f})")

    # Hybrid evaluation
    if hybrid_blueprint is not None:
        for k in k_values:
            uw_scores, w_scores = [], []

            for train_idx, val_idx in cv_split(groups, n_folds,
                                               next(iter(reps.values())),
                                               labels, mode=mode):
                train_reps = {n: r[train_idx] for n, r in reps.items()}
                val_reps = {n: r[val_idx] for n, r in reps.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]
                aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
                aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

                try:
                    X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                        train_reps, val_reps, y_train, hybrid_blueprint,
                        augmentations_train=aug_tr, augmentations_val=aug_va)
                except Exception:
                    continue

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr_h)
                X_va = scaler.transform(X_va_h)
                k_eff = min(k, len(y_train) - 1)

                # Unweighted
                KNN = KNeighborsClassifier if is_clf else KNeighborsRegressor
                model = KNN(n_neighbors=k_eff)
                model.fit(X_tr, y_train)
                uw_scores.append(metric_fn(y_val, model.predict(X_va)))

                # Weighted
                try:
                    w = _compute_importance_for_hybrid(
                        train_reps, y_train, feat_info)
                    model_w = KNN(n_neighbors=k_eff)
                    model_w.fit(X_tr * w, y_train)
                    w_scores.append(metric_fn(y_val, model_w.predict(X_va * w)))
                except Exception:
                    w_scores.append(float('nan'))

            if uw_scores:
                uw_mean = float(np.nanmean(uw_scores))
                w_mean = float(np.nanmean(w_scores))
                results['hybrid'][f'k{k}'] = {
                    'unweighted': uw_mean,
                    'weighted': w_mean,
                    'delta': w_mean - uw_mean,
                }
                print(f"    hybrid k={k}: uw={uw_mean:.4f}, w={w_mean:.4f} "
                      f"(delta={w_mean - uw_mean:+.4f})")

    return results


# =============================================================================
# TASK 2: GP WITH IMPORTANCE-WEIGHTED RBF
# =============================================================================

def _run_weighted_gp(reps, labels, groups, config, hybrid_blueprint,
                     metric_fn, metric_name, n_samples,
                     augmentations=None):
    """GP with importance-weighted RBF kernel."""
    print("\n  --- Task 2: GP with Importance-Weighted RBF ---")

    if n_samples > _GP_MAX_SAMPLES:
        print(f"    SKIPPED: n_samples={n_samples} > {_GP_MAX_SAMPLES}")
        return {'skipped': True, 'reason': f'n_samples={n_samples} > {_GP_MAX_SAMPLES}'}

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    results = {'per_rep': {}, 'hybrid': {}}

    def _make_gp():
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)
        return GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2,
            alpha=1e-6, random_state=42,
        )

    # Per-rep evaluation
    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        std_scores, w_scores = [], []

        for train_idx, val_idx in cv_split(groups, n_folds, X_rep, labels,
                                           mode=mode):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # Standard GP
            try:
                gp = _make_gp()
                gp.fit(X_tr, y_train)
                std_scores.append(metric_fn(y_val, gp.predict(X_va)))
            except Exception:
                std_scores.append(float('nan'))

            # Importance-weighted GP (pre-transform: X * sqrt(w))
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[train_idx]}, y_train,
                    method='lightgbm_gain')
                w = _importance_weights(imp[rep_name])
                gp_w = _make_gp()
                gp_w.fit(X_tr * w, y_train)
                w_scores.append(metric_fn(y_val, gp_w.predict(X_va * w)))
            except Exception:
                w_scores.append(float('nan'))

        std_mean = float(np.nanmean(std_scores))
        w_mean = float(np.nanmean(w_scores))
        results['per_rep'][rep_name] = {
            'standard': std_mean,
            'weighted': w_mean,
            'delta': w_mean - std_mean,
        }
        print(f"    {rep_name}: std={std_mean:.4f}, weighted={w_mean:.4f} "
              f"(delta={w_mean - std_mean:+.4f})")

    # Hybrid evaluation
    if hybrid_blueprint is not None:
        std_scores, w_scores = [], []

        for train_idx, val_idx in cv_split(groups, n_folds,
                                           next(iter(reps.values())),
                                           labels, mode=mode):
            train_reps = {n: r[train_idx] for n, r in reps.items()}
            val_reps = {n: r[val_idx] for n, r in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]
            aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
            aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps, val_reps, y_train, hybrid_blueprint,
                    augmentations_train=aug_tr, augmentations_val=aug_va)
            except Exception:
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_h)
            X_va = scaler.transform(X_va_h)

            # Standard GP
            try:
                gp = _make_gp()
                gp.fit(X_tr, y_train)
                std_scores.append(metric_fn(y_val, gp.predict(X_va)))
            except Exception:
                std_scores.append(float('nan'))

            # Weighted GP
            try:
                w = _compute_importance_for_hybrid(
                    train_reps, y_train, feat_info)
                gp_w = _make_gp()
                gp_w.fit(X_tr * w, y_train)
                w_scores.append(metric_fn(y_val, gp_w.predict(X_va * w)))
            except Exception:
                w_scores.append(float('nan'))

        if std_scores:
            std_mean = float(np.nanmean(std_scores))
            w_mean = float(np.nanmean(w_scores))
            results['hybrid'] = {
                'standard': std_mean,
                'weighted': w_mean,
                'delta': w_mean - std_mean,
            }
            print(f"    hybrid: std={std_mean:.4f}, weighted={w_mean:.4f} "
                  f"(delta={w_mean - std_mean:+.4f})")

    return results


# =============================================================================
# TASK 3: SVR WITH IMPORTANCE-WEIGHTED RBF
# =============================================================================

def _run_weighted_svr(reps, labels, groups, config, hybrid_blueprint,
                      metric_fn, metric_name, n_samples,
                      augmentations=None):
    """SVR with importance-weighted RBF kernel."""
    print("\n  --- Task 3: SVR with Importance-Weighted RBF ---")

    if n_samples > _SVR_MAX_SAMPLES:
        print(f"    SKIPPED: n_samples={n_samples} > {_SVR_MAX_SAMPLES}")
        return {'skipped': True, 'reason': f'n_samples={n_samples} > {_SVR_MAX_SAMPLES}'}

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    results = {'per_rep': {}, 'hybrid': {}}

    def _make_svr():
        return SVR(kernel='rbf', C=1.0, gamma='scale')

    # Per-rep evaluation
    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        std_scores, w_scores = [], []

        for train_idx, val_idx in cv_split(groups, n_folds, X_rep, labels,
                                           mode=mode):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # Standard SVR
            try:
                svr = _make_svr()
                svr.fit(X_tr, y_train)
                std_scores.append(metric_fn(y_val, svr.predict(X_va)))
            except Exception:
                std_scores.append(float('nan'))

            # Importance-weighted SVR
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[train_idx]}, y_train,
                    method='lightgbm_gain')
                w = _importance_weights(imp[rep_name])
                svr_w = _make_svr()
                svr_w.fit(X_tr * w, y_train)
                w_scores.append(metric_fn(y_val, svr_w.predict(X_va * w)))
            except Exception:
                w_scores.append(float('nan'))

        std_mean = float(np.nanmean(std_scores))
        w_mean = float(np.nanmean(w_scores))
        results['per_rep'][rep_name] = {
            'standard': std_mean,
            'weighted': w_mean,
            'delta': w_mean - std_mean,
        }
        print(f"    {rep_name}: std={std_mean:.4f}, weighted={w_mean:.4f} "
              f"(delta={w_mean - std_mean:+.4f})")

    # Hybrid evaluation
    if hybrid_blueprint is not None:
        std_scores, w_scores = [], []

        for train_idx, val_idx in cv_split(groups, n_folds,
                                           next(iter(reps.values())),
                                           labels, mode=mode):
            train_reps = {n: r[train_idx] for n, r in reps.items()}
            val_reps = {n: r[val_idx] for n, r in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]
            aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
            aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps, val_reps, y_train, hybrid_blueprint,
                    augmentations_train=aug_tr, augmentations_val=aug_va)
            except Exception:
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_h)
            X_va = scaler.transform(X_va_h)

            # Standard SVR
            try:
                svr = _make_svr()
                svr.fit(X_tr, y_train)
                std_scores.append(metric_fn(y_val, svr.predict(X_va)))
            except Exception:
                std_scores.append(float('nan'))

            # Weighted SVR
            try:
                w = _compute_importance_for_hybrid(
                    train_reps, y_train, feat_info)
                svr_w = _make_svr()
                svr_w.fit(X_tr * w, y_train)
                w_scores.append(metric_fn(y_val, svr_w.predict(X_va * w)))
            except Exception:
                w_scores.append(float('nan'))

        if std_scores:
            std_mean = float(np.nanmean(std_scores))
            w_mean = float(np.nanmean(w_scores))
            results['hybrid'] = {
                'standard': std_mean,
                'weighted': w_mean,
                'delta': w_mean - std_mean,
            }
            print(f"    hybrid: std={std_mean:.4f}, weighted={w_mean:.4f} "
                  f"(delta={w_mean - std_mean:+.4f})")

    return results


# =============================================================================
# TASK 4: SIMILARITY-AUGMENTED FEATURES (q-RASAR STYLE)
# =============================================================================

def _run_qrasar(reps, labels, groups, config, hybrid_blueprint,
                metric_fn, metric_name, task_type='regression',
                augmentations=None):
    """q-RASAR-style similarity-augmented features.

    For each sample, find k nearest neighbors and compute similarity
    descriptors: mean(y_neighbors), std(y_neighbors), min_distance,
    mean_distance. Append as 4 extra features.

    Compare:
      - base features only (RF baseline)
      - base + sim descriptors from unweighted distance
      - base + sim descriptors from importance-weighted distance
    For per-rep and hybrid.
    """
    print("\n  --- Task 4: Similarity-Augmented Features (q-RASAR style) ---")

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    k_neighbors = 10
    results = {'per_rep': {}, 'hybrid': {}}

    _rf_fac = get_model_factories(['rf'], task_type=task_type,
                                    n_samples=len(labels))['rf']
    def _make_rf():
        return _rf_fac()

    # Per-rep evaluation
    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        base_scores, uw_sim_scores, w_sim_scores = [], [], []

        for train_idx, val_idx in cv_split(groups, n_folds, X_rep, labels,
                                           mode=mode):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # Base features only
            try:
                rf = _make_rf()
                rf.fit(X_tr, y_train)
                base_scores.append(metric_fn(y_val, rf.predict(X_va)))
            except Exception:
                base_scores.append(float('nan'))

            # Base + unweighted sim descriptors
            try:
                sim_tr = _compute_sim_descriptors(X_tr, y_train, X_tr,
                                                  k=k_neighbors)
                sim_va = _compute_sim_descriptors(X_tr, y_train, X_va,
                                                  k=k_neighbors)
                X_tr_aug = np.hstack([X_tr, sim_tr])
                X_va_aug = np.hstack([X_va, sim_va])
                rf_aug = _make_rf()
                rf_aug.fit(X_tr_aug, y_train)
                uw_sim_scores.append(metric_fn(y_val, rf_aug.predict(X_va_aug)))
            except Exception:
                uw_sim_scores.append(float('nan'))

            # Base + importance-weighted sim descriptors
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[train_idx]}, y_train,
                    method='lightgbm_gain')
                w = _importance_weights(imp[rep_name])
                X_tr_w = X_tr * w
                X_va_w = X_va * w
                sim_tr_w = _compute_sim_descriptors(X_tr_w, y_train, X_tr_w,
                                                    k=k_neighbors)
                sim_va_w = _compute_sim_descriptors(X_tr_w, y_train, X_va_w,
                                                    k=k_neighbors)
                X_tr_aug_w = np.hstack([X_tr, sim_tr_w])
                X_va_aug_w = np.hstack([X_va, sim_va_w])
                rf_aug_w = _make_rf()
                rf_aug_w.fit(X_tr_aug_w, y_train)
                w_sim_scores.append(metric_fn(y_val, rf_aug_w.predict(X_va_aug_w)))
            except Exception:
                w_sim_scores.append(float('nan'))

        base_mean = float(np.nanmean(base_scores))
        uw_mean = float(np.nanmean(uw_sim_scores))
        w_mean = float(np.nanmean(w_sim_scores))
        results['per_rep'][rep_name] = {
            'base_only': base_mean,
            'base_plus_sim_unweighted': uw_mean,
            'base_plus_sim_weighted': w_mean,
            'delta_uw': uw_mean - base_mean,
            'delta_w': w_mean - base_mean,
            'delta_w_vs_uw': w_mean - uw_mean,
        }
        print(f"    {rep_name}: base={base_mean:.4f}, "
              f"+sim_uw={uw_mean:.4f} (d={uw_mean - base_mean:+.4f}), "
              f"+sim_w={w_mean:.4f} (d={w_mean - base_mean:+.4f})")

    # Hybrid evaluation
    if hybrid_blueprint is not None:
        base_scores, uw_sim_scores, w_sim_scores = [], [], []

        for train_idx, val_idx in cv_split(groups, n_folds,
                                           next(iter(reps.values())),
                                           labels, mode=mode):
            train_reps = {n: r[train_idx] for n, r in reps.items()}
            val_reps = {n: r[val_idx] for n, r in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]
            aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
            aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps, val_reps, y_train, hybrid_blueprint,
                    augmentations_train=aug_tr, augmentations_val=aug_va)
            except Exception:
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_h)
            X_va = scaler.transform(X_va_h)

            # Base only
            try:
                rf = _make_rf()
                rf.fit(X_tr, y_train)
                base_scores.append(metric_fn(y_val, rf.predict(X_va)))
            except Exception:
                base_scores.append(float('nan'))

            # Unweighted sim descriptors
            try:
                sim_tr = _compute_sim_descriptors(X_tr, y_train, X_tr,
                                                  k=k_neighbors)
                sim_va = _compute_sim_descriptors(X_tr, y_train, X_va,
                                                  k=k_neighbors)
                X_tr_aug = np.hstack([X_tr, sim_tr])
                X_va_aug = np.hstack([X_va, sim_va])
                rf_aug = _make_rf()
                rf_aug.fit(X_tr_aug, y_train)
                uw_sim_scores.append(metric_fn(y_val, rf_aug.predict(X_va_aug)))
            except Exception:
                uw_sim_scores.append(float('nan'))

            # Weighted sim descriptors
            try:
                w = _compute_importance_for_hybrid(
                    train_reps, y_train, feat_info)
                X_tr_w = X_tr * w
                X_va_w = X_va * w
                sim_tr_w = _compute_sim_descriptors(X_tr_w, y_train, X_tr_w,
                                                    k=k_neighbors)
                sim_va_w = _compute_sim_descriptors(X_tr_w, y_train, X_va_w,
                                                    k=k_neighbors)
                X_tr_aug_w = np.hstack([X_tr, sim_tr_w])
                X_va_aug_w = np.hstack([X_va, sim_va_w])
                rf_aug_w = _make_rf()
                rf_aug_w.fit(X_tr_aug_w, y_train)
                w_sim_scores.append(metric_fn(y_val, rf_aug_w.predict(X_va_aug_w)))
            except Exception:
                w_sim_scores.append(float('nan'))

        if base_scores:
            base_mean = float(np.nanmean(base_scores))
            uw_mean = float(np.nanmean(uw_sim_scores))
            w_mean = float(np.nanmean(w_sim_scores))
            results['hybrid'] = {
                'base_only': base_mean,
                'base_plus_sim_unweighted': uw_mean,
                'base_plus_sim_weighted': w_mean,
                'delta_uw': uw_mean - base_mean,
                'delta_w': w_mean - base_mean,
                'delta_w_vs_uw': w_mean - uw_mean,
            }
            print(f"    hybrid: base={base_mean:.4f}, "
                  f"+sim_uw={uw_mean:.4f} (d={uw_mean - base_mean:+.4f}), "
                  f"+sim_w={w_mean:.4f} (d={w_mean - base_mean:+.4f})")

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Evaluate importance scores as feature weights for distance-dependent
    downstream tasks: kNN, GP, SVR, and q-RASAR similarity-augmented features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S10: IMPORTANCE WEIGHTS FOR DOWNSTREAM TASKS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_samples = len(labels)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")
    print(f"  Samples: {n_samples}, Reps: {len(reps)}")

    # Get hybrid blueprint via find_best_hybrid
    hybrid_blueprint = None
    budget = kwargs.get('budget', config.get('budgets', [100])[0])
    search_speed = config.get('search_speed', 'fast')

    try:
        search = get_precomputed_search(
            reps, labels, groups, budget=budget,
            aug_condition='uncapped', augmentations=data.get('augmentations'),
            speed=search_speed,
            precomputed=None,  # bypass tree-only precomputed; use disk cache
            internal_evaluators=DISTANCE_EVALUATORS,
            budgets=DISTANCE_BUDGETS)
        # Select kNN-optimized blueprint (primary downstream model in S10)
        best_config = get_best_for_downstream(search, 'knn')
        if best_config and 'blueprint' in best_config:
            hybrid_blueprint = best_config['blueprint']
        if hybrid_blueprint:
            print(f"  Hybrid blueprint: alloc={hybrid_blueprint.allocation_method}, "
                  f"budget={hybrid_blueprint.budget}, "
                  f"imp={hybrid_blueprint.importance_method}, "
                  f"ev={hybrid_blueprint.internal_evaluator}")
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    except Exception as e:
        print(f"  Hybrid search failed: {e}")
        top3 = None

    start_total = time.time()
    results = {}
    if top3:
        results['top3_base_reps'] = [(n, s) for n, s in top3]

    augmentations = data.get('augmentations')
    task_type = data.get('task_type', 'regression')
    is_clf = task_type == 'classification'

    # Task 1: Weighted kNN
    t0 = time.time()
    results['weighted_knn'] = _run_weighted_knn(
        reps, labels, groups, config, hybrid_blueprint,
        metric_fn, metric_name, lower_is_better,
        augmentations=augmentations, task_type=task_type)
    print(f"  [Task 1 elapsed: {time.time() - t0:.1f}s]")

    # Task 2: GP with importance-weighted RBF (regression only)
    if is_clf:
        print("\n  --- Task 2: GP --- SKIP (regression-only)")
        results['weighted_gp'] = {'skipped': True, 'reason': 'classification'}
    else:
        t0 = time.time()
        results['weighted_gp'] = _run_weighted_gp(
            reps, labels, groups, config, hybrid_blueprint,
            metric_fn, metric_name, n_samples,
            augmentations=augmentations)
        print(f"  [Task 2 elapsed: {time.time() - t0:.1f}s]")

    # Task 3: SVR with importance-weighted RBF (regression only)
    if is_clf:
        print("  --- Task 3: SVR --- SKIP (regression-only)")
        results['weighted_svr'] = {'skipped': True, 'reason': 'classification'}
    else:
        t0 = time.time()
        results['weighted_svr'] = _run_weighted_svr(
            reps, labels, groups, config, hybrid_blueprint,
            metric_fn, metric_name, n_samples,
            augmentations=augmentations)
        print(f"  [Task 3 elapsed: {time.time() - t0:.1f}s]")

    # Task 4: q-RASAR similarity-augmented features
    t0 = time.time()
    results['qrasar'] = _run_qrasar(
        reps, labels, groups, config, hybrid_blueprint,
        metric_fn, metric_name, task_type=task_type,
        augmentations=augmentations)
    print(f"  [Task 4 elapsed: {time.time() - t0:.1f}s]")

    # Summary
    print(f"\n  Total elapsed: {time.time() - start_total:.1f}s")
    _print_summary(results, metric_name, lower_is_better)

    output = make_structured_output(
        's10_importance_downstream', dataset_name, data, config, results,
        primary_metric=metric_name, lower_is_better=lower_is_better,
    )
    return output


def _print_summary(results, metric_name, lower_is_better):
    """Print a concise cross-task summary."""
    print(f"\n  {'='*60}")
    print(f"  SUMMARY (metric={metric_name}, lower_is_better={lower_is_better})")
    print(f"  {'='*60}")

    # For each task, find the best per-rep delta and hybrid delta
    sign = -1 if lower_is_better else 1

    for task_name, task_key, delta_key in [
        ('Weighted kNN', 'weighted_knn', 'delta'),
        ('Weighted GP', 'weighted_gp', 'delta'),
        ('Weighted SVR', 'weighted_svr', 'delta'),
        ('q-RASAR (w vs base)', 'qrasar', 'delta_w'),
    ]:
        task_res = results.get(task_key, {})
        if task_res.get('skipped'):
            print(f"    {task_name}: SKIPPED ({task_res.get('reason', '')})")
            continue

        per_rep = task_res.get('per_rep', {})
        hybrid = task_res.get('hybrid', {})

        if per_rep:
            # For kNN, per_rep values are dicts of k-values; for others, flat
            if task_key == 'weighted_knn':
                deltas = []
                for rep_name, k_results in per_rep.items():
                    for kk, vals in k_results.items():
                        deltas.append((rep_name, kk, vals.get(delta_key, 0)))
                best = max(deltas, key=lambda x: sign * x[2])
                mean_delta = float(np.mean([d[2] for d in deltas]))
                print(f"    {task_name}: mean_delta={mean_delta:+.4f}, "
                      f"best={best[0]}({best[1]}) delta={best[2]:+.4f}", end='')
            else:
                deltas = [(rn, v.get(delta_key, 0)) for rn, v in per_rep.items()]
                best = max(deltas, key=lambda x: sign * x[1])
                mean_delta = float(np.mean([d[1] for d in deltas]))
                print(f"    {task_name}: mean_delta={mean_delta:+.4f}, "
                      f"best={best[0]} delta={best[1]:+.4f}", end='')

        # Hybrid
        if hybrid and not isinstance(hybrid, dict):
            pass
        elif hybrid and delta_key in hybrid:
            print(f", hybrid_delta={hybrid[delta_key]:+.4f}", end='')
        elif isinstance(hybrid, dict):
            # kNN hybrid: iterate k-values
            h_deltas = [v.get(delta_key, 0) for v in hybrid.values()
                        if isinstance(v, dict) and delta_key in v]
            if h_deltas:
                print(f", hybrid_delta={np.mean(h_deltas):+.4f}", end='')

        print()


if __name__ == '__main__':
    parser = create_experiment_parser(
        'S10: Importance Weights for Distance-Dependent Downstream Tasks',
        has_delta=True, has_augment=True)
    parser.add_argument('--budget', type=int, default=100,
                        help='Budget for hybrid feature selection')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  budget=args.budget)
    save_experiment_results(results, args.output,
                            's10_importance_downstream', args.dataset)
