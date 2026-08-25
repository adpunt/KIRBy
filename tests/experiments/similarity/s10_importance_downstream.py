#!/usr/bin/env python3
"""S10: Importance Scores as Feature Weights for Distance-Dependent Downstream Tasks.

Paper: Similarity
Question: does importance-weighted distance help GP, SVR, and q-RASAR-style
    similarity-augmented features (beyond the kNN case in S05)?
Dependencies: S05 (extends kNN), C04 (importance weighting).

Goes beyond kNN (S05) to test importance weighting on GP, SVR, and
similarity-augmented features (q-RASAR style). Four downstream tasks:
  1. Weighted kNN regression (extends S05 with per-rep + hybrid)
  2. GP with importance-weighted RBF kernel
  3. SVR with importance-weighted RBF kernel
  4. Similarity-augmented features (q-RASAR style)

Delta convention (2026-06-03): every stored `delta*` field is direction-
oriented by `_orient_deltas` so that POSITIVE always means improvement,
regardless of whether the dataset metric is higher-better (Spearman/AUC/F1) or
lower-better (RMSE/MAE). Before this fix, buchwald's RMSE delta "+13.077" (a
near-doubling of error) was misread as a gain. `results['deltas_oriented']`
flags that the convention is applied. Empirically weighting is neutral-to-
negative downstream -- a real result, not a wiring bug (the weights ARE applied).
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
    get_best_search_for_downstream,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
    flatten_fold_weights,
    honest_rescore, build_honest_rescore_input,
)

from kirby.hybrid import apply_feature_selection


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


def _orient_deltas(obj, lower_is_better):
    """Recursively flip the sign of every 'delta*' field so positive always
    means improvement, regardless of metric direction.

    Each stored delta is (treatment - reference). For higher-is-better metrics
    (Spearman/AUC/F1) positive already means improvement -> no-op. For
    lower-is-better metrics (RMSE/MAE, e.g. buchwald) a positive raw delta means
    the treatment got WORSE, so it is negated. This fixes the trap where
    buchwald GP "delta +13.077" (an RMSE increase) read as a gain.
    """
    if not lower_is_better:
        return obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                _orient_deltas(v, lower_is_better)
            elif k.startswith('delta') and isinstance(v, (int, float)) and not isinstance(v, bool):
                obj[k] = -v
    elif isinstance(obj, list):
        for item in obj:
            _orient_deltas(item, lower_is_better)
    return obj


def _compute_importance_for_reps(train_reps, y_train, method='lightgbm_gain'):
    """Compute importance and return flat weight vector matching sorted rep concat."""
    imp_dict = compute_feature_importance(train_reps, y_train, method=method)
    imp_concat = np.concatenate([imp_dict[k] for k in sorted(imp_dict.keys())])
    return _importance_weights(imp_concat)


def _compute_importance_for_hybrid(feat_info):
    """Compute importance weights for hybrid-selected features.

    Uses flatten_fold_weights so the returned vector matches the hybrid
    matrix shape exactly (including augmentation columns appended by
    build_fold_hybrid).
    """
    raw = flatten_fold_weights(feat_info)
    return _importance_weights(raw)


def _compute_sim_descriptors(X_train, y_train, X_query, k=10, is_self=False):
    """Compute q-RASAR-style similarity descriptors for each query sample.

    For each query sample, finds k nearest neighbors in X_train and computes:
      - mean(y_neighbors): average target value of neighbors
      - std(y_neighbors): std of target values of neighbors
      - min_distance: distance to nearest neighbor
      - mean_distance: mean distance to k neighbors

    Args:
        is_self: Pass True when X_query is X_train (training-fold descriptors)
            so the self-match (distance=0) is excluded from neighbors.

    Returns (n_query, 4) array.
    """
    k_nn = min(k + (1 if is_self else 0), len(y_train) - 1)
    if k_nn < 1:
        return np.zeros((X_query.shape[0], 4))

    nn = NearestNeighbors(n_neighbors=k_nn)
    nn.fit(X_train)
    dists, indices = nn.kneighbors(X_query)

    if is_self:
        # Drop the first column (self-match with distance=0)
        dists = dists[:, 1:]
        indices = indices[:, 1:]

    y_neighbors = y_train[indices]  # (n_query, k)
    mean_y = np.mean(y_neighbors, axis=1)
    std_y = np.std(y_neighbors, axis=1)
    min_dist = dists[:, 0]
    mean_dist = np.mean(dists, axis=1)

    return np.column_stack([mean_y, std_y, min_dist, mean_dist])


def _honest_best_k_within_arm(knn_task, lower_is_better):
    """Held-out best-k re-score of the weighted-vs-unweighted kNN comparison.

    The stored kNN result selects its best k on the SAME folds it scores on
    (s10:210 and _print_summary) -> selection-optimistic. Using the per-fold
    scores now persisted by _run_weighted_knn, this re-picks the winning k on
    LEFT-OUT folds via the shared honest_rescore arithmetic: the WEIGHTED arm's
    k-variants are the 'candidates' and the UNWEIGHTED arm's are the 'baselines',
    so honest_rescore's `delta` is the honest weighted-minus-unweighted gap
    (direction-corrected, + == weighted better). Keyed per rep, plus 'hybrid'.

    This is a LOCAL adapter, not a redesign: the shared helper's hybrid-vs-
    baseline shape does not map to a within-arm best-k selection, so we reshape
    the persisted fold scores into it rather than re-implement the LOFO math.
    """
    def _rescore(section):
        cands, base = {}, {}
        for name, kdict in section.items():
            if not isinstance(kdict, dict):
                continue
            cands[name] = [
                {'name': kk, 'fold_scores': v.get('weighted_folds', [])}
                for kk, v in kdict.items() if isinstance(v, dict)
            ]
            base[name] = {
                kk: v.get('unweighted_folds', [])
                for kk, v in kdict.items() if isinstance(v, dict)
            }
        return honest_rescore(cands, base, lower_is_better)

    out = {}
    per_rep = knn_task.get('per_rep') or {}
    if per_rep:
        out['per_rep'] = _rescore(per_rep)
    hybrid = knn_task.get('hybrid') or {}
    if isinstance(hybrid, dict) and hybrid:
        out['hybrid'] = _rescore({'hybrid': hybrid}).get('hybrid', {})
    return out


# =============================================================================
# TASK 1: WEIGHTED kNN REGRESSION
# =============================================================================

def _run_weighted_knn(reps, labels, groups, config, hybrid_blueprint,
                      metric_fn, metric_name, lower_is_better=False,
                      augmentations=None, task_type='regression',
                      fold_splits=None, fold_importance=None):
    """Weighted kNN: per-rep and hybrid, weighted vs unweighted.

    fold_splits: list of (train_idx, val_idx) tuples, shared across all tasks.
    fold_importance: dict {fold_i: {rep_name: importance_array}}, precomputed.
    """
    is_clf = task_type == 'classification'
    print(f"\n  --- Task 1: Weighted kNN {'Classification' if is_clf else 'Regression'} ---")

    k_values = [5, 10, 20]
    results = {'per_rep': {}, 'hybrid': {}}

    # Per-rep evaluation
    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        rep_results = {}

        for k in k_values:
            uw_scores, w_scores = [], []

            for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
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

                # Weighted (use precomputed importance)
                try:
                    imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                    if imp_arr is None:
                        raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                    w = _importance_weights(imp_arr)
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
                # Per-fold scores persisted so a reader can re-pick the best k on
                # held-out folds (honest) instead of on the same folds (s10:210).
                'unweighted_folds': [float(s) for s in uw_scores],
                'weighted_folds': [float(s) for s in w_scores],
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

            for train_idx, val_idx in fold_splits:
                train_reps = {n: r[train_idx] for n, r in reps.items()}
                val_reps = {n: r[val_idx] for n, r in reps.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]
                aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
                aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

                try:
                    X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                        train_reps, val_reps, y_train, hybrid_blueprint,
                        augmentations_train=aug_tr, augmentations_val=aug_va,
                        groups=(groups[train_idx] if groups is not None else None))
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
                    w = _compute_importance_for_hybrid(feat_info)
                    assert w.shape[0] == X_tr.shape[1], (
                        f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
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
                    'unweighted_folds': [float(s) for s in uw_scores],
                    'weighted_folds': [float(s) for s in w_scores],
                }
                print(f"    hybrid k={k}: uw={uw_mean:.4f}, w={w_mean:.4f} "
                      f"(delta={w_mean - uw_mean:+.4f})")

    return results


# =============================================================================
# TASK 2: GP WITH IMPORTANCE-WEIGHTED RBF
# =============================================================================

def _run_weighted_gp(reps, labels, groups, config, hybrid_blueprint,
                     metric_fn, metric_name, n_samples,
                     augmentations=None,
                     fold_splits=None, fold_importance=None):
    """GP with importance-weighted RBF kernel."""
    print("\n  --- Task 2: GP with Importance-Weighted RBF ---")

    if n_samples > _GP_MAX_SAMPLES:
        print(f"    SKIPPED: n_samples={n_samples} > {_GP_MAX_SAMPLES}")
        return {'skipped': True, 'reason': f'n_samples={n_samples} > {_GP_MAX_SAMPLES}'}

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

        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
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
                imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                if imp_arr is None:
                    raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                w = _importance_weights(imp_arr)
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
            'standard_folds': [float(s) for s in std_scores],
            'weighted_folds': [float(s) for s in w_scores],
        }
        print(f"    {rep_name}: std={std_mean:.4f}, weighted={w_mean:.4f} "
              f"(delta={w_mean - std_mean:+.4f})")

    # Hybrid evaluation
    if hybrid_blueprint is not None:
        std_scores, w_scores = [], []

        for train_idx, val_idx in fold_splits:
            train_reps = {n: r[train_idx] for n, r in reps.items()}
            val_reps = {n: r[val_idx] for n, r in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]
            aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
            aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps, val_reps, y_train, hybrid_blueprint,
                    augmentations_train=aug_tr, augmentations_val=aug_va,
                    groups=(groups[train_idx] if groups is not None else None),
                    model=_make_gp(),
                    metric_fn=metric_fn)
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
                w = _compute_importance_for_hybrid(feat_info)
                assert w.shape[0] == X_tr.shape[1], (
                    f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
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
                'standard_folds': [float(s) for s in std_scores],
                'weighted_folds': [float(s) for s in w_scores],
            }
            print(f"    hybrid: std={std_mean:.4f}, weighted={w_mean:.4f} "
                  f"(delta={w_mean - std_mean:+.4f})")

    return results


# =============================================================================
# TASK 3: SVR WITH IMPORTANCE-WEIGHTED RBF
# =============================================================================

def _run_weighted_svr(reps, labels, groups, config, hybrid_blueprint,
                      metric_fn, metric_name, n_samples,
                      augmentations=None,
                      fold_splits=None, fold_importance=None):
    """SVR with importance-weighted RBF kernel."""
    print("\n  --- Task 3: SVR with Importance-Weighted RBF ---")

    if n_samples > _SVR_MAX_SAMPLES:
        print(f"    SKIPPED: n_samples={n_samples} > {_SVR_MAX_SAMPLES}")
        return {'skipped': True, 'reason': f'n_samples={n_samples} > {_SVR_MAX_SAMPLES}'}

    results = {'per_rep': {}, 'hybrid': {}}

    def _make_svr():
        return SVR(kernel='rbf', C=1.0, gamma='scale')

    # Per-rep evaluation
    for rep_name in sorted(reps.keys()):
        X_rep = reps[rep_name]
        std_scores, w_scores = [], []

        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
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
                imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                if imp_arr is None:
                    raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                w = _importance_weights(imp_arr)
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
            'standard_folds': [float(s) for s in std_scores],
            'weighted_folds': [float(s) for s in w_scores],
        }
        print(f"    {rep_name}: std={std_mean:.4f}, weighted={w_mean:.4f} "
              f"(delta={w_mean - std_mean:+.4f})")

    # Hybrid evaluation
    if hybrid_blueprint is not None:
        std_scores, w_scores = [], []

        for train_idx, val_idx in fold_splits:
            train_reps = {n: r[train_idx] for n, r in reps.items()}
            val_reps = {n: r[val_idx] for n, r in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]
            aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
            aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps, val_reps, y_train, hybrid_blueprint,
                    augmentations_train=aug_tr, augmentations_val=aug_va,
                    groups=(groups[train_idx] if groups is not None else None),
                    model=_make_svr(),
                    metric_fn=metric_fn)
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
                w = _compute_importance_for_hybrid(feat_info)
                assert w.shape[0] == X_tr.shape[1], (
                    f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
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
                'standard_folds': [float(s) for s in std_scores],
                'weighted_folds': [float(s) for s in w_scores],
            }
            print(f"    hybrid: std={std_mean:.4f}, weighted={w_mean:.4f} "
                  f"(delta={w_mean - std_mean:+.4f})")

    return results


# =============================================================================
# TASK 4: SIMILARITY-AUGMENTED FEATURES (q-RASAR STYLE)
# =============================================================================

def _run_qrasar(reps, labels, groups, config, hybrid_blueprint,
                metric_fn, metric_name, task_type='regression',
                augmentations=None,
                fold_splits=None, fold_importance=None):
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

        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
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
                                                  k=k_neighbors, is_self=True)
                sim_va = _compute_sim_descriptors(X_tr, y_train, X_va,
                                                  k=k_neighbors, is_self=False)
                X_tr_aug = np.hstack([X_tr, sim_tr])
                X_va_aug = np.hstack([X_va, sim_va])
                rf_aug = _make_rf()
                rf_aug.fit(X_tr_aug, y_train)
                uw_sim_scores.append(metric_fn(y_val, rf_aug.predict(X_va_aug)))
            except Exception:
                uw_sim_scores.append(float('nan'))

            # Base + importance-weighted sim descriptors
            try:
                imp_arr = fold_importance.get(fold_i, {}).get(rep_name)
                if imp_arr is None:
                    raise RuntimeError(f"missing precomputed importance for fold={fold_i} rep={rep_name}")
                w = _importance_weights(imp_arr)
                X_tr_w = X_tr * w
                X_va_w = X_va * w
                sim_tr_w = _compute_sim_descriptors(X_tr_w, y_train, X_tr_w,
                                                    k=k_neighbors, is_self=True)
                sim_va_w = _compute_sim_descriptors(X_tr_w, y_train, X_va_w,
                                                    k=k_neighbors, is_self=False)
                X_tr_aug_w = np.hstack([X_tr_w, sim_tr_w])
                X_va_aug_w = np.hstack([X_va_w, sim_va_w])
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
            'base_only_folds': [float(s) for s in base_scores],
            'base_plus_sim_unweighted_folds': [float(s) for s in uw_sim_scores],
            'base_plus_sim_weighted_folds': [float(s) for s in w_sim_scores],
        }
        print(f"    {rep_name}: base={base_mean:.4f}, "
              f"+sim_uw={uw_mean:.4f} (d={uw_mean - base_mean:+.4f}), "
              f"+sim_w={w_mean:.4f} (d={w_mean - base_mean:+.4f})")

    # Hybrid evaluation
    if hybrid_blueprint is not None:
        base_scores, uw_sim_scores, w_sim_scores = [], [], []

        for train_idx, val_idx in fold_splits:
            train_reps = {n: r[train_idx] for n, r in reps.items()}
            val_reps = {n: r[val_idx] for n, r in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]
            aug_tr = {a: v[train_idx] for a, v in augmentations.items()} if augmentations else None
            aug_va = {a: v[val_idx] for a, v in augmentations.items()} if augmentations else None

            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps, val_reps, y_train, hybrid_blueprint,
                    augmentations_train=aug_tr, augmentations_val=aug_va,
                    groups=(groups[train_idx] if groups is not None else None),
                    model=_make_rf(),
                    metric_fn=metric_fn)
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
                                                  k=k_neighbors, is_self=True)
                sim_va = _compute_sim_descriptors(X_tr, y_train, X_va,
                                                  k=k_neighbors, is_self=False)
                X_tr_aug = np.hstack([X_tr, sim_tr])
                X_va_aug = np.hstack([X_va, sim_va])
                rf_aug = _make_rf()
                rf_aug.fit(X_tr_aug, y_train)
                uw_sim_scores.append(metric_fn(y_val, rf_aug.predict(X_va_aug)))
            except Exception:
                uw_sim_scores.append(float('nan'))

            # Weighted sim descriptors
            try:
                w = _compute_importance_for_hybrid(feat_info)
                assert w.shape[0] == X_tr.shape[1], (
                    f"weight shape {w.shape} != X_tr cols {X_tr.shape[1]}")
                X_tr_w = X_tr * w
                X_va_w = X_va * w
                sim_tr_w = _compute_sim_descriptors(X_tr_w, y_train, X_tr_w,
                                                    k=k_neighbors, is_self=True)
                sim_va_w = _compute_sim_descriptors(X_tr_w, y_train, X_va_w,
                                                    k=k_neighbors, is_self=False)
                X_tr_aug_w = np.hstack([X_tr_w, sim_tr_w])
                X_va_aug_w = np.hstack([X_va_w, sim_va_w])
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
                'base_only_folds': [float(s) for s in base_scores],
                'base_plus_sim_unweighted_folds': [float(s) for s in uw_sim_scores],
                'base_plus_sim_weighted_folds': [float(s) for s in w_sim_scores],
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
    task_type = data['task_type']

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")
    print(f"  Samples: {n_samples}, Reps: {len(reps)}")

    # Get hybrid blueprint via find_best_hybrid
    hybrid_blueprint = None
    domain = data.get('domain', 'molecular')
    budget = kwargs.get('budget', config.get('budgets', [100])[0])

    # SIMA-2: surface (do NOT swallow) the hybrid-search outcome. Previously a
    # failed/empty search silently left hybrid_blueprint=None, so EVERY hybrid
    # branch in Tasks 1-4 was skipped while the experiment still reported as if
    # it produced hybrid results. We now record an explicit flag in the output.
    hybrid_status = 'ok'
    search = None  # set on a successful pooled search; carries per-fold scores
    try:
        # Pass metric/task_type/domain + data so the pooled search resolves the
        # SAME cache lineage used at precompute time and can hit the in-memory
        # precomputed_searches_distance dict (instead of a cold recompute /
        # mismatched metric).
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='knn', budget=budget,
            augmentations=data.get('augmentations'),
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset_name, data=data,
            distance=True,
        )
        if meta:
            hybrid_blueprint = meta['best_config']['blueprint']
        if hybrid_blueprint:
            print(f"  Hybrid blueprint: alloc={hybrid_blueprint.allocation_method}, "
                  f"budget={hybrid_blueprint.budget}, "
                  f"imp={hybrid_blueprint.importance_method}, "
                  f"ev={hybrid_blueprint.internal_evaluator}")
        else:
            hybrid_status = 'no_blueprint'
            print("  WARNING: pooled search returned no blueprint -- all hybrid "
                  "branches (Tasks 1-4) will be SKIPPED (per-rep baselines only).")
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    except Exception as e:
        hybrid_status = f'error: {type(e).__name__}: {e}'
        print(f"  Hybrid search failed: {e}")
        print("  WARNING: all hybrid branches (Tasks 1-4) will be SKIPPED "
              "(per-rep baselines only).")
        top3 = None

    start_total = time.time()
    results = {}
    # Flag the hybrid outcome so downstream readers/aggregators (s09) can tell a
    # baseline-only run apart from a complete hybrid run.
    results['hybrid_status'] = hybrid_status
    results['hybrid_evaluated'] = hybrid_blueprint is not None
    if top3:
        results['top3_base_reps'] = [(n, s) for n, s in top3]

    augmentations = data.get('augmentations')
    is_clf = task_type == 'classification'

    # -------------------------------------------------------------------------
    # Shared precompute: materialize fold splits ONCE and compute per-(fold, rep)
    # importance ONCE so all four downstream tasks reuse the same numbers.
    # Previously each task recomputed importance per (fold, rep), 4x duplicated
    # work. Splits must come from the same reference X for every task so fold
    # iteration order is identical across tasks.
    # -------------------------------------------------------------------------
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    # Reference X for splitting — use first rep (sorted) for determinism. Folds
    # only depend on `groups`/`labels`/`X.shape[0]`, so any rep works as long as
    # it's used consistently across all tasks.
    _ref_rep_name = sorted(reps.keys())[0]
    _ref_X = reps[_ref_rep_name]
    fold_splits = list(cv_split(groups, n_folds, _ref_X, labels, mode=mode))

    t_imp = time.time()
    fold_importance = {}
    for fold_i, (train_idx, _val_idx) in enumerate(fold_splits):
        train_reps = {n: r[train_idx] for n, r in reps.items()}
        try:
            imp_dict = compute_feature_importance(
                train_reps, labels[train_idx], method='lightgbm_gain')
            fold_importance[fold_i] = imp_dict
        except Exception as e:
            print(f"  WARN: precompute importance failed at fold={fold_i}: {e}")
            fold_importance[fold_i] = {}
    print(f"  [Precomputed fold importance for {len(fold_splits)} folds x "
          f"{len(reps)} reps in {time.time() - t_imp:.1f}s]")

    # Task 1: Weighted kNN
    t0 = time.time()
    results['weighted_knn'] = _run_weighted_knn(
        reps, labels, groups, config, hybrid_blueprint,
        metric_fn, metric_name, lower_is_better,
        augmentations=augmentations, task_type=task_type,
        fold_splits=fold_splits, fold_importance=fold_importance)
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
            augmentations=augmentations,
            fold_splits=fold_splits, fold_importance=fold_importance)
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
            augmentations=augmentations,
            fold_splits=fold_splits, fold_importance=fold_importance)
        print(f"  [Task 3 elapsed: {time.time() - t0:.1f}s]")

    # Task 4: q-RASAR similarity-augmented features
    t0 = time.time()
    results['qrasar'] = _run_qrasar(
        reps, labels, groups, config, hybrid_blueprint,
        metric_fn, metric_name, task_type=task_type,
        augmentations=augmentations,
        fold_splits=fold_splits, fold_importance=fold_importance)
    print(f"  [Task 4 elapsed: {time.time() - t0:.1f}s]")

    # -------------------------------------------------------------------------
    # HONEST (selection-corrected) RE-SCORES. Both blocks are kept OUT of the
    # _orient_deltas loop below: their 'delta' fields are ALREADY direction-
    # corrected by honest_rescore, so re-orienting would double-negate them.
    # -------------------------------------------------------------------------
    # (1) Standard contract block: per-fold scores for BOTH arms of the pooled
    #     hybrid search (all hybrid candidate recipes + baseline single reps),
    #     straight from the SearchResult, so a reader can call honest_rescore()
    #     without re-running the search. Empty / old-cache searches yield empty
    #     dicts (the helper tolerates them). Only present when a search was got.
    if search is not None:
        try:
            results['honest_rescore'] = build_honest_rescore_input(search)
        except Exception as e:
            results['honest_rescore_error'] = f'{type(e).__name__}: {e}'

    # (2) Within-file weighted-vs-unweighted kNN: the stored best k is picked on
    #     the SAME folds it scores on (s10:210 + _print_summary). Re-pick the
    #     winning k per held-out fold via the shared honest arithmetic on the
    #     per-fold scores just persisted (weighted arm = candidates, unweighted
    #     arm = baselines; delta + == weighted better).
    if isinstance(results.get('weighted_knn'), dict):
        results['honest_within_arm'] = {
            'weighted_knn': _honest_best_k_within_arm(
                results['weighted_knn'], lower_is_better)
        }

    # Orient every stored delta so positive == improvement regardless of metric
    # direction (RMSE etc.). MUST run before _print_summary, which now treats
    # the stored deltas as already direction-corrected.
    for _task_key in ('weighted_knn', 'weighted_gp', 'weighted_svr', 'qrasar'):
        if isinstance(results.get(_task_key), dict):
            _orient_deltas(results[_task_key], lower_is_better)
    results['deltas_oriented'] = True  # positive delta == improvement everywhere

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

    # Deltas are already direction-oriented (positive == improvement) by
    # _orient_deltas before this is called, so select/print them as-is.
    sign = 1

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
