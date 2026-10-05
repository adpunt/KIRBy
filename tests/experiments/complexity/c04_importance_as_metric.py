#!/usr/bin/env python3
"""C04: Feature importance as learned metric space.

Paper: Complexity
Conclusion: Feature importance doubles as a learned similarity metric.
Dependencies: None (standalone). Requires benchmark_importance_aggregation.

Computes importance scores with each method, then evaluates them as kNN
weights and similarity retrieval weights. Cross-tests selection by method A
with weighting by method B. Low-dim subset evaluation.

All importance scores are computed per-fold on training data only to avoid
data leakage.

Aligned 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4). c04 is an approved
exception to the standard selection (2.3 "Still open", group A): it fits no
downstream model and runs no hybrid search, so there is no 2 base + 2 hybrid
choice to make. What the alignment changed: outer folds come from
kirby.find_best_hybrid.nested_cv_outer_folds with config['n_splits'] (1 = one
held-out split); only DATASET_REPS[ds]['general'] representations are used;
every scaler is FeatureScaler (0/1 features left unscaled, 2.3 item 17);
weighted matrices stay in the representations' own precision (float32, 2.3
item 18); the result carries one record per outer fold.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
# 2026-10-05: every scaler in this file is FeatureScaler, StandardScaler with
# 0/1 features passed through unscaled (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.3
# item 17). It was StandardScaler, which standardised fingerprint bits too.
from kirby.scaling import FeatureScaler
# 2026-10-05: the outer folds every per-fold search uses (2.4 step 5).
from kirby.find_best_hybrid import nested_cv_outer_folds, BUILTIN_METRICS
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
# 2026-10-05, owner delegated: the kNN regressor takes the same distance
# keywords make_nearest_neighbors uses (manhattan unless --distance-metric /
# KIRBY_DISTANCE_METRIC says otherwise), so both arms measure distance alike.
from experiments._shared import sklearn_distance_kwargs, get_distance_metric
from sklearn.metrics import r2_score, mean_absolute_error

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    compute_feature_importance, get_domain_metric,
)
from experiments.complexity.complexity_helpers import DATASET_REPS


# 2026-10-05: stamped into every result. A result file without this value (or
# with an older one) was computed before the per-fold alignment. complexity
# run_all resumes on input_fingerprint only, which this value is not part of;
# see the shared-code request in the alignment report.
RESULT_IDENTITY = 'c04_fold_aligned_2026_10_05'


def _general_reps(reps, dataset_name):
    """Only the dataset's pruned representations (2.4 step 9), 2026-10-05.

    run_all already loads DATASET_REPS[ds]['general'] and drops
    augmentation-only representations; a direct run of this file loads
    whatever --reps names. c04 compares no augmentation (2.3 item 4), so
    augmentation representations are left out on every dataset.
    """
    entry = DATASET_REPS.get(dataset_name)
    if entry is None:
        raise ValueError(
            f"c04: {dataset_name!r} has no DATASET_REPS entry, so its pruned "
            f"representations are unknown")
    general = set(entry.get('general') or [])
    kept = {k: v for k, v in reps.items() if k in general}
    dropped = sorted(k for k in reps if k not in general)
    if dropped:
        print(f"  c04 uses only {dataset_name}'s pruned representations; "
              f"left out: {dropped}")
    if not kept:
        raise ValueError(
            f"c04: none of the loaded representations {sorted(reps)} is in "
            f"{dataset_name}'s pruned list {sorted(general)}")
    return kept


def _scale_fold(X_train, X_val):
    """FeatureScaler fitted on the training rows, in the matrix's own dtype."""
    scaler = FeatureScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, X_val_s


# =========================================================================
# PER-FOLD EVALUATION HELPERS
# =========================================================================

def _importance_to_weights(imp_scores, dtype=None):
    """Convert raw importance scores to normalised non-negative weights.

    `dtype` (2026-10-05): the feature matrix's precision, so multiplying a
    float32 matrix by the weights does not turn it into float64 (2.3 item 18).
    """
    w = np.sqrt(np.maximum(imp_scores, 0))
    w = w / (w.sum() + 1e-10)
    return w if dtype is None else w.astype(dtype, copy=False)


def _compute_fold_importance(reps, labels, folds, methods):
    """Importance scores once per (fold, method), concatenated across reps.

    The kNN, retrieval, cross-test and low-dim arms all rank or weight the same
    concatenated matrix, and each used to call compute_feature_importance
    itself: with 3 methods and 5 folds that came to 120 computations where 15
    cover every arm. Scores are still computed on each fold's training rows
    only, in the same rep order, so no number changes.

    Returns {(fold index, method): concatenated importance scores}. A method
    that raises on a fold is left out; the arm that needs it then reports that
    method as failed instead of scoring it on partial folds.
    """
    rep_order = sorted(reps.keys())
    cache = {}
    for fold_i, (train_idx, _val_idx) in enumerate(folds):
        train_reps = {k: reps[k][train_idx] for k in rep_order}
        y_train = labels[train_idx]
        for method in methods:
            try:
                imp_dict = compute_feature_importance(
                    train_reps, y_train, method=method)
            except Exception as e:
                print(f"    importance {method} failed on fold {fold_i}: {e}")
                continue
            cache[(fold_i, method)] = np.concatenate(
                [imp_dict[rep] for rep in sorted(imp_dict.keys())])
    return cache


def _fold_importance(imp_by_fold, fold_i, method):
    """One (fold, method) entry out of the memo, or say which one is missing."""
    scores = imp_by_fold.get((fold_i, method))
    if scores is None:
        raise RuntimeError(
            f"no importance scores for method {method} on fold {fold_i}")
    return scores


def _evaluate_knn_and_sim(X_concat, labels, folds, imp_by_fold, method,
                          k_values=None, k_neighbors_sim=10):
    """Per-fold importance + kNN / similarity evaluation.

    Returns (knn_results, sim_results) where knn_results is
    {k: {unweighted_r2, weighted_r2, delta}} and sim_results is
    {unweighted_mae, weighted_mae, delta_mae, unweighted_r2, weighted_r2,
     delta_r2}.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    knn_scores = {k: {'uw': [], 'w': []} for k in k_values}
    sim_uw_preds, sim_w_preds, sim_true = [], [], []
    # 2026-10-05, owner delegated, 2026-10-05: R-squared of the
    # 10-nearest-neighbour prediction on each fold's held-out rows. The
    # reported R-squared is now their mean, the way the kNN regressor's
    # always was; before, the predictions of all folds were pooled into one
    # vector and one R-squared taken over it.
    sim_uw_fold_r2, sim_w_fold_r2 = [], []
    _knn_dist_kw = sklearn_distance_kwargs()

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        # Importance was computed on TRAINING data only, once per (fold, method)
        imp_concat = _fold_importance(imp_by_fold, fold_i, method)
        weights = _importance_to_weights(imp_concat, X_concat.dtype)

        X_train = X_concat[train_idx]
        X_val = X_concat[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        X_train_s, X_val_s = _scale_fold(X_train, X_val)

        # ---- kNN evaluation ----
        for k in k_values:
            nn = min(k, len(y_train) - 1)
            # Unweighted
            # 2026-10-05, owner delegated: same distance as the
            # 10-nearest-neighbour prediction below (was sklearn's default,
            # Euclidean).
            knn = KNeighborsRegressor(n_neighbors=nn, **_knn_dist_kw)
            knn.fit(X_train_s, y_train)
            knn_scores[k]['uw'].append(r2_score(y_val, knn.predict(X_val_s)))
            # Weighted
            X_train_w = X_train_s * weights
            X_val_w = X_val_s * weights
            knn_w = KNeighborsRegressor(n_neighbors=nn, **_knn_dist_kw)
            knn_w.fit(X_train_w, y_train)
            knn_scores[k]['w'].append(r2_score(y_val, knn_w.predict(X_val_w)))

        # ---- Similarity retrieval evaluation ----
        nn_k = min(k_neighbors_sim, len(y_train) - 1)
        # Unweighted
        nn_uw = make_nearest_neighbors(n_neighbors=nn_k)
        nn_uw.fit(X_train_s)
        dists_uw, idx_uw = nn_uw.kneighbors(X_val_s)
        w_uw = 1.0 / (dists_uw + 1e-10)
        preds_uw = np.average(y_train[idx_uw], axis=1, weights=w_uw)
        sim_uw_preds.extend(preds_uw)
        # Weighted
        X_train_w = X_train_s * weights
        X_val_w = X_val_s * weights
        nn_w = make_nearest_neighbors(n_neighbors=nn_k)
        nn_w.fit(X_train_w)
        dists_w, idx_w = nn_w.kneighbors(X_val_w)
        w_w = 1.0 / (dists_w + 1e-10)
        preds_w = np.average(y_train[idx_w], axis=1, weights=w_w)
        sim_w_preds.extend(preds_w)
        sim_true.extend(y_val)
        sim_uw_fold_r2.append(float(r2_score(y_val, preds_uw)))
        sim_w_fold_r2.append(float(r2_score(y_val, preds_w)))

    # Aggregate kNN
    knn_results = {}
    for k in k_values:
        uw = float(np.mean(knn_scores[k]['uw']))
        w = float(np.mean(knn_scores[k]['w']))
        knn_results[k] = {'unweighted_r2': uw, 'weighted_r2': w,
                          'delta': w - uw,
                          # 2026-10-05, owner delegated: the per-fold values
                          # the two means are taken over.
                          'unweighted_r2_per_fold': [float(v) for v in knn_scores[k]['uw']],
                          'weighted_r2_per_fold': [float(v) for v in knn_scores[k]['w']],
                          'r2_aggregation': 'mean of per-fold R-squared',
                          'distance_metric': get_distance_metric()}

    # Aggregate similarity
    sim_true = np.array(sim_true)
    sim_uw_preds = np.array(sim_uw_preds)
    sim_w_preds = np.array(sim_w_preds)
    uw_mae = mean_absolute_error(sim_true, sim_uw_preds)
    w_mae = mean_absolute_error(sim_true, sim_w_preds)
    # 2026-10-05, owner delegated: mean of per-fold R-squared (see above).
    # MAE is still taken over the pooled predictions.
    uw_r2 = float(np.mean(sim_uw_fold_r2))
    w_r2 = float(np.mean(sim_w_fold_r2))
    sim_results = {
        'unweighted_mae': float(uw_mae), 'weighted_mae': float(w_mae),
        'delta_mae': float(uw_mae - w_mae),
        'unweighted_r2': float(uw_r2), 'weighted_r2': float(w_r2),
        'delta_r2': float(w_r2 - uw_r2),
        'unweighted_r2_per_fold': sim_uw_fold_r2,
        'weighted_r2_per_fold': sim_w_fold_r2,
        'r2_aggregation': 'mean of per-fold R-squared',
        'mae_aggregation': 'pooled over all folds',
        'distance_metric': get_distance_metric(),
    }

    return knn_results, sim_results


def _evaluate_cross_test(X_concat, labels, folds, imp_by_fold,
                         select_method, weight_method, budget,
                         k_neighbors=10):
    """Cross-test: select features by method A, weight by method B.

    Both methods' scores come out of the per-(fold, method) memo, so the nine
    method pairs share the same fifteen computations instead of recomputing
    both sides inside the loop.
    Returns weighted_r2 for similarity retrieval.
    """
    sim_w_preds, sim_true = [], []
    _fold_r2 = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        y_train, y_val = labels[train_idx], labels[val_idx]

        imp_sel_concat = _fold_importance(imp_by_fold, fold_i, select_method)
        imp_wt_concat = _fold_importance(imp_by_fold, fold_i, weight_method)

        # Select top features by method A
        top_idx = np.argsort(imp_sel_concat)[-budget:]

        X_train_sub = X_concat[train_idx][:, top_idx]
        X_val_sub = X_concat[val_idx][:, top_idx]

        X_train_s, X_val_s = _scale_fold(X_train_sub, X_val_sub)

        # Weight by method B
        weights = _importance_to_weights(imp_wt_concat[top_idx], X_concat.dtype)

        X_train_w = X_train_s * weights
        X_val_w = X_val_s * weights

        nn_k = min(k_neighbors, len(y_train) - 1)
        nn = make_nearest_neighbors(n_neighbors=nn_k)
        nn.fit(X_train_w)
        dists, idx = nn.kneighbors(X_val_w)
        w = 1.0 / (dists + 1e-10)
        preds = np.average(y_train[idx], axis=1, weights=w)

        sim_w_preds.extend(preds)
        sim_true.extend(y_val)
        _fold_r2.append(float(r2_score(y_val, preds)))

    # 2026-10-05, owner delegated, 2026-10-05: mean of per-fold R-squared,
    # like every other 10-nearest-neighbour R-squared in this file (was one
    # R-squared over the predictions of all folds pooled).
    # 2026-10-05: the per-fold values are returned too, for the per-fold record.
    return float(np.mean(_fold_r2)), _fold_r2


def _evaluate_low_dim(X_concat, labels, folds, imp_by_fold, method,
                      budget, k_neighbors=10):
    """Low-dim: per-fold select top-budget features + weight by importance.

    Returns {weighted_r2, unweighted_r2, delta_r2}.
    """
    sim_uw_preds, sim_w_preds, sim_true = [], [], []
    _uw_fold_r2, _w_fold_r2 = [], []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        y_train, y_val = labels[train_idx], labels[val_idx]

        imp_concat = _fold_importance(imp_by_fold, fold_i, method)

        top_idx = np.argsort(imp_concat)[-budget:]
        weights = _importance_to_weights(imp_concat[top_idx], X_concat.dtype)

        X_train_sub = X_concat[train_idx][:, top_idx]
        X_val_sub = X_concat[val_idx][:, top_idx]

        X_train_s, X_val_s = _scale_fold(X_train_sub, X_val_sub)

        nn_k = min(k_neighbors, len(y_train) - 1)

        # Unweighted
        nn_uw = make_nearest_neighbors(n_neighbors=nn_k)
        nn_uw.fit(X_train_s)
        d_uw, i_uw = nn_uw.kneighbors(X_val_s)
        w_uw = 1.0 / (d_uw + 1e-10)
        _p_uw = np.average(y_train[i_uw], axis=1, weights=w_uw)
        sim_uw_preds.extend(_p_uw)
        _uw_fold_r2.append(float(r2_score(y_val, _p_uw)))

        # Weighted
        X_train_w = X_train_s * weights
        X_val_w = X_val_s * weights
        nn_w = make_nearest_neighbors(n_neighbors=nn_k)
        nn_w.fit(X_train_w)
        d_w, i_w = nn_w.kneighbors(X_val_w)
        w_w = 1.0 / (d_w + 1e-10)
        _p_w = np.average(y_train[i_w], axis=1, weights=w_w)
        sim_w_preds.extend(_p_w)
        _w_fold_r2.append(float(r2_score(y_val, _p_w)))

        sim_true.extend(y_val)

    # 2026-10-05, owner delegated, 2026-10-05: mean of per-fold R-squared
    # (was one R-squared over all folds' predictions pooled).
    uw_r2 = float(np.mean(_uw_fold_r2))
    w_r2 = float(np.mean(_w_fold_r2))
    return {'weighted_r2': float(w_r2), 'unweighted_r2': float(uw_r2),
            'delta_r2': float(w_r2 - uw_r2),
            'weighted_r2_per_fold': _w_fold_r2,
            'unweighted_r2_per_fold': _uw_fold_r2,
            'r2_aggregation': 'mean of per-fold R-squared'}


def _evaluate_single_rep_retrieval(reps, labels, folds, k_neighbors=10):
    """Same retrieval rule, one representation at a time and unweighted.

    Predicts each molecule's label from the inverse-distance-weighted average
    of its 10 nearest training molecules, on one representation's own matrix,
    with no importance weighting. Every other arm in this file works inside the
    concatenation of all ten representations, so its numbers have nothing to be
    read against; this is that comparator. R-squared is computed on each
    fold's held-out rows and averaged over folds, which is how the
    concatenation arms aggregate since 2026-10-05 (owner delegated; before,
    the predictions of all folds were pooled into one R-squared). MAE is
    still over the pooled predictions.

    Returns {rep name: {r2, mae}}.
    """
    out = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        preds, truth = [], []
        _fold_r2 = []
        try:
            for train_idx, val_idx in folds:
                y_train, y_val = labels[train_idx], labels[val_idx]
                X_train_s, X_val_s = _scale_fold(X[train_idx], X[val_idx])

                nn_k = min(k_neighbors, len(y_train) - 1)
                nn = make_nearest_neighbors(n_neighbors=nn_k)
                nn.fit(X_train_s)
                dists, idx = nn.kneighbors(X_val_s)
                w = 1.0 / (dists + 1e-10)
                _p = np.average(y_train[idx], axis=1, weights=w)
                preds.extend(_p)
                truth.extend(y_val)
                _fold_r2.append(float(r2_score(y_val, _p)))
            truth = np.array(truth)
            preds = np.array(preds)
            out[rep_name] = {
                # 2026-10-05, owner delegated: mean of per-fold R-squared.
                'r2': float(np.mean(_fold_r2)),
                'r2_per_fold': _fold_r2,
                'mae': float(mean_absolute_error(truth, preds)),
                'n_features': int(X.shape[1]),
            }
        except Exception as e:
            print(f"    {rep_name}: FAILED - {e}")
    return out


def _per_fold_records(results, folds):
    """One record per outer fold, 2026-10-05 (2.4 step 11).

    c04 fits no downstream model and chooses nothing on training-row scores,
    so there is no training-row score to record; every number is the held-out
    R-squared of a 10- (or k-) nearest-neighbour prediction on that fold.
    """
    out = []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        rec = {'fold': fold_i, 'n_train_rows': int(len(train_idx)),
               'n_held_out_rows': int(len(val_idx)),
               'selection_scope': 'fold_train_rows',
               'train_row_score': None,
               'held_out_r2': {}}
        h = rec['held_out_r2']

        def _at(vals):
            return (float(vals[fold_i])
                    if vals is not None and fold_i < len(vals) else None)

        for method, m in results['methods'].items():
            for k, kr in m['knn'].items():
                h[f'knn_k{k}_unweighted'] = _at(kr.get('unweighted_r2_per_fold'))
                h[f'knn_k{k}_weighted__{method}'] = _at(kr.get('weighted_r2_per_fold'))
            sim = m['similarity']
            h['neighbours10_unweighted'] = _at(sim.get('unweighted_r2_per_fold'))
            h[f'neighbours10_weighted__{method}'] = _at(sim.get('weighted_r2_per_fold'))
        for method, ld in results['low_dim'].items():
            h[f'low_dim_unweighted__{method}'] = _at(ld.get('unweighted_r2_per_fold'))
            h[f'low_dim_weighted__{method}'] = _at(ld.get('weighted_r2_per_fold'))
        for key, vals in results['cross_test_per_fold'].items():
            h[f'cross_test__{key}'] = _at(vals)
        for rep_name, e in results['single_rep_comparator'].items():
            h[f'single_rep__{rep_name}'] = _at(e.get('r2_per_fold'))
        out.append(rec)
    return out


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Evaluate importance scores as learned metric spaces."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C04: IMPORTANCE WEIGHTS AS LEARNED METRICS")
    print("=" * 70)

    # Classification support is TODO — the similarity-retrieval section uses
    # continuous label averaging (np.average on y_train) which doesn't
    # translate to class predictions without a majority/soft-voting rewrite.
    # Skip gracefully rather than report nonsense on classification datasets.
    task_type = data['task_type']
    if task_type == 'classification':
        print(f"  SKIPPING: c04 currently only supports regression targets.")
        print(f"  TODO: add classification branch (KNeighborsClassifier + majority-vote retrieval).")
        return {'skipped': True, 'reason': 'classification_not_supported',
                'task_type': task_type}

    # The dataset's decided importance methods (decided_config, set by run_all
    # and by get_config when the run names its dataset).
    importance_methods = config['importance_methods']
    # 2026-10-05: the outer fold count comes from config['n_splits'] only
    # (2.4 step 14); 1 means one held-out split.
    n_folds = int(config['n_splits'])
    budget = kwargs.get('budget', 25)

    # 2026-10-05: only the dataset's pruned representations (2.4 step 9).
    reps = _general_reps(data['reps'], dataset_name)
    labels = data['labels']
    groups = data['groups']
    if data.get('augmentations'):
        print(f"  c04 compares no augmentation (2.3 item 4); the "
              f"{len(data['augmentations'])} augmentation representations "
              f"are not used.")

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    print(f"  Concatenated features: {X_concat.shape} ({X_concat.dtype})")

    # The dataset's own metric, recorded for reference only (2026-10-05).
    # Every score c04 computes is an R-squared or an MAE.
    try:
        _ds_metric, _, _ds_lower = get_domain_metric(data)
    except Exception as e:
        print(f"  dataset metric not resolved: {e}")
        _ds_metric, _ds_lower = None, None

    # metric_name / lower_is_better set here, in the result itself, 2026-10-05.
    # The R-squared stamp in __main__ below only covers a direct run of this
    # file: complexity_helpers.run_all saves through save_experiment_results
    # with the DATASET's metric (rmse on herg_ki, spearman on tresanco), and
    # save_experiment_results only fills those keys when the result does not
    # already carry them. Every score below is an R-squared (higher is better)
    # or an MAE in label units, whatever the dataset is reported in.
    # 2026-10-05: the direction is read from the engine's metric table, not
    # written in by hand.
    results = {'metric_name': 'r2',
               'lower_is_better': bool(BUILTIN_METRICS['r2'][1]),
               'dataset_metric_name': _ds_metric,
               'dataset_lower_is_better': _ds_lower,
               'result_identity': RESULT_IDENTITY,
               # Importance scores, feature choices, scalers and neighbour
               # lists are all fitted on each outer fold's training rows.
               'selection_scope': 'fold_train_rows',
               'n_outer_folds': n_folds,
               'representations': sorted(reps.keys()),
               'importance_methods': list(importance_methods),
               'budget_features': int(budget),
               'aug_condition': 'none',
               'methods': {}, 'cross_test': {}, 'cross_test_per_fold': {},
               'cross_test_failed': {}, 'low_dim': {},
               'single_rep_comparator': {},
               # 2026-10-05, owner delegated, 2026-10-05: the distance both
               # the kNN regressor and the 10-nearest-neighbour prediction
               # use, and how every R-squared here is aggregated over folds.
               'distance_metric': get_distance_metric(),
               'r2_aggregation': 'mean of per-fold R-squared'}

    # One fold list and one importance computation per (fold, method), shared by
    # every arm below. The folds are cut once here so all four arms score the
    # same molecules.
    # 2026-10-05: nested_cv_outer_folds, the split every per-fold search uses
    # (was cv_split; both give GroupKFold on regression and one 80/20 group
    # split at n_splits=1).
    folds = nested_cv_outer_folds(
        reps, labels, groups, n_folds, int(config.get('seed', 42)), task_type)
    # 2026-10-05: record the folds actually made. The splitter lowers the
    # count to the number of groups when a dataset has fewer groups than
    # n_splits, so config['n_splits'] alone can overstate it.
    results['n_outer_folds'] = len(folds)
    results['n_outer_folds_requested'] = n_folds
    if len(folds) == 1:
        results['fold_spread'] = 'one fold, no spread'
    print(f"\n  Computing importance once per (fold, method): "
          f"{len(folds)} folds x {len(importance_methods)} methods...")
    imp_by_fold = _compute_fold_importance(
        reps, labels, folds, importance_methods)

    # Evaluate each method with per-fold importance computation
    print("\n  Evaluating methods (per-fold importance)...")
    for method in importance_methods:
        print(f"  {method}...", end=' ', flush=True)
        start = time.time()
        try:
            knn_result, sim_result = _evaluate_knn_and_sim(
                X_concat, labels, folds, imp_by_fold, method)

            k10 = knn_result.get(10, knn_result.get(5, {}))
            results['methods'][method] = {
                'knn': knn_result,
                'similarity': sim_result,
                'knn_delta_k10': k10.get('delta', 0),
                'sim_delta_r2': sim_result.get('delta_r2', 0),
            }

            elapsed = time.time() - start
            print(f"kNN delta={k10.get('delta', 0):+.4f}, "
                  f"sim delta={sim_result.get('delta_r2', 0):+.4f} "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    # Cross-test: selection by method A, weighting by method B
    method_names = [m for m in importance_methods if m in results['methods']]
    print("\n  Cross-test matrix (select x weight)...")
    if len(method_names) >= 2:
        for select_method in method_names[:4]:
            for weight_method in method_names[:4]:
                key = f"{select_method}__x__{weight_method}"
                try:
                    w_r2, w_r2_folds = _evaluate_cross_test(
                        X_concat, labels, folds, imp_by_fold,
                        select_method, weight_method, budget)
                    results['cross_test'][key] = w_r2
                    results['cross_test_per_fold'][key] = w_r2_folds
                except Exception as e:
                    # 2026-10-05: a failed pairing is printed and recorded
                    # (it was dropped silently).
                    print(f"    {key}: FAILED - {e}")
                    results['cross_test_failed'][key] = (
                        f"{type(e).__name__}: {e}")
    else:
        # 2026-10-05: say why the matrix is empty (logd and qm9 have one
        # decided importance method).
        results['cross_test_skipped'] = (
            f"needs 2 importance methods, this run has {method_names}")
        print(f"    skipped: {results['cross_test_skipped']}")

    # Low-dim: budget=25 selected features + importance weighting
    print(f"\n  Low-dim (budget={budget}) selected + weighted...")
    for method in method_names:
        try:
            ld = _evaluate_low_dim(
                X_concat, labels, folds, imp_by_fold, method, budget)
            results['low_dim'][method] = ld
            print(f"    {method}: weighted R2={ld['weighted_r2']:.4f}, "
                  f"unweighted R2={ld['unweighted_r2']:.4f}")
        except Exception as e:
            print(f"    {method}: FAILED - {e}")

    # Single-representation comparator: the same retrieval from the 10 nearest
    # training molecules, on each representation's own matrix, unweighted, over
    # the same folds. Everything above lives inside the concatenation of all
    # ten, so without these rows the weighted-concatenation R-squared has
    # nothing to be read against.
    print("\n  Single-representation retrieval (unweighted, own matrix)...")
    single_rep = _evaluate_single_rep_retrieval(reps, labels, folds)
    results['single_rep_comparator'] = single_rep
    if single_rep:
        best_rep = max(single_rep, key=lambda r: single_rep[r]['r2'])
        results['best_single_rep'] = best_rep
        for rep_name in sorted(single_rep, key=lambda r: -single_rep[r]['r2']):
            e = single_rep[rep_name]
            print(f"    {rep_name:<24} R2={e['r2']:+.4f}  MAE={e['mae']:.4f}  "
                  f"{e['n_features']} features")

    results['per_fold'] = _per_fold_records(results, folds)

    # Summary
    if results['methods']:
        best_knn = max(results['methods'].items(),
                       key=lambda x: x[1].get('knn_delta_k10', -999))
        best_sim = max(results['methods'].items(),
                       key=lambda x: x[1].get('sim_delta_r2', -999))
        print(f"\n  Best kNN weight source: {best_knn[0]} "
              f"(delta={best_knn[1]['knn_delta_k10']:+.4f})")
        print(f"  Best similarity weight source: {best_sim[0]} "
              f"(delta={best_sim[1]['sim_delta_r2']:+.4f})")
        results['best_knn_method'] = best_knn[0]
        results['best_sim_method'] = best_sim[0]

    # Cross-test summary table
    if results['cross_test']:
        print("\n  CROSS-TEST MATRIX (select x weight -> weighted R2):")
        select_methods = sorted(set(k.split('__x__')[0]
                                    for k in results['cross_test']))
        weight_methods = sorted(set(k.split('__x__')[1]
                                    for k in results['cross_test']))
        col_label = "select\\weight"
        header = f"  {col_label:<20}" + "".join(
            f"{w:>14}" for w in weight_methods)
        print(header)
        for s in select_methods:
            row = f"  {s:<20}"
            for w in weight_methods:
                key = f"{s}__x__{w}"
                val = results['cross_test'].get(key)
                row += f"{val:>14.4f}" if val is not None else f"{'--':>14}"
            print(row)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C04: Importance Weights as Learned Metrics',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--budget', type=int, default=25,
                        help='Budget for low-dim feature subset evaluation')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Importance methods: {config['importance_methods']}")

    # 2026-10-05: --augment is not passed on; c04 compares no augmentation
    # (2.3 item 4) and run() keeps only the pruned representations.
    if getattr(args, 'augment', False):
        print("  --augment ignored: c04 compares no augmentation")
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False))
    results = run(data, config, args.output, args.dataset, budget=args.budget)
    # ⛔ STAMP THE METRIC c04 ACTUALLY USES. Every number in this file is an
    # R-squared (or an MAE), whatever the dataset is reported in: c04 never
    # calls get_domain_metric. Left unset, save_experiment_results fills
    # metric_name and lower_is_better from the dataset registry, so a herg_ki
    # or caco2 file carried `metric_name: rmse, lower_is_better: true` at the
    # top level above a body in which every stored quantity is an R-squared
    # where higher is better. Anything aggregating these files read the
    # direction backwards.
    save_experiment_results(results, args.output, 'c04_importance_as_metric',
                            args.dataset,
                            metric_name='r2',
                            lower_is_better=bool(BUILTIN_METRICS['r2'][1]))
