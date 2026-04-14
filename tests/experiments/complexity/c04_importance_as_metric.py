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
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import r2_score, mean_absolute_error

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    compute_feature_importance, cv_split,
)


# =========================================================================
# PER-FOLD EVALUATION HELPERS
# =========================================================================

def _importance_to_weights(imp_scores):
    """Convert raw importance scores to normalised non-negative weights."""
    w = np.sqrt(np.maximum(imp_scores, 0))
    return w / (w.sum() + 1e-10)


def _evaluate_knn_and_sim(X_concat, reps, labels, groups, n_folds, method,
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

    for train_idx, val_idx in cv_split(groups, n_folds, X_concat, labels):
        # Compute importance on TRAINING data only
        train_reps = {k: reps[k][train_idx] for k in sorted(reps.keys())}
        imp_dict = compute_feature_importance(
            train_reps, labels[train_idx], method=method)
        imp_concat = np.concatenate(
            [imp_dict[rep] for rep in sorted(imp_dict.keys())])
        weights = _importance_to_weights(imp_concat)

        X_train = X_concat[train_idx]
        X_val = X_concat[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # ---- kNN evaluation ----
        for k in k_values:
            nn = min(k, len(y_train) - 1)
            # Unweighted
            knn = KNeighborsRegressor(n_neighbors=nn)
            knn.fit(X_train_s, y_train)
            knn_scores[k]['uw'].append(r2_score(y_val, knn.predict(X_val_s)))
            # Weighted
            X_train_w = X_train_s * weights
            X_val_w = X_val_s * weights
            knn_w = KNeighborsRegressor(n_neighbors=nn)
            knn_w.fit(X_train_w, y_train)
            knn_scores[k]['w'].append(r2_score(y_val, knn_w.predict(X_val_w)))

        # ---- Similarity retrieval evaluation ----
        nn_k = min(k_neighbors_sim, len(y_train) - 1)
        # Unweighted
        nn_uw = NearestNeighbors(n_neighbors=nn_k)
        nn_uw.fit(X_train_s)
        dists_uw, idx_uw = nn_uw.kneighbors(X_val_s)
        w_uw = 1.0 / (dists_uw + 1e-10)
        preds_uw = np.average(y_train[idx_uw], axis=1, weights=w_uw)
        sim_uw_preds.extend(preds_uw)
        # Weighted
        X_train_w = X_train_s * weights
        X_val_w = X_val_s * weights
        nn_w = NearestNeighbors(n_neighbors=nn_k)
        nn_w.fit(X_train_w)
        dists_w, idx_w = nn_w.kneighbors(X_val_w)
        w_w = 1.0 / (dists_w + 1e-10)
        preds_w = np.average(y_train[idx_w], axis=1, weights=w_w)
        sim_w_preds.extend(preds_w)
        sim_true.extend(y_val)

    # Aggregate kNN
    knn_results = {}
    for k in k_values:
        uw = float(np.mean(knn_scores[k]['uw']))
        w = float(np.mean(knn_scores[k]['w']))
        knn_results[k] = {'unweighted_r2': uw, 'weighted_r2': w,
                          'delta': w - uw}

    # Aggregate similarity
    sim_true = np.array(sim_true)
    sim_uw_preds = np.array(sim_uw_preds)
    sim_w_preds = np.array(sim_w_preds)
    uw_mae = mean_absolute_error(sim_true, sim_uw_preds)
    w_mae = mean_absolute_error(sim_true, sim_w_preds)
    uw_r2 = r2_score(sim_true, sim_uw_preds)
    w_r2 = r2_score(sim_true, sim_w_preds)
    sim_results = {
        'unweighted_mae': float(uw_mae), 'weighted_mae': float(w_mae),
        'delta_mae': float(uw_mae - w_mae),
        'unweighted_r2': float(uw_r2), 'weighted_r2': float(w_r2),
        'delta_r2': float(w_r2 - uw_r2),
    }

    return knn_results, sim_results


def _evaluate_cross_test(X_concat, reps, labels, groups, n_folds,
                         select_method, weight_method, budget,
                         k_neighbors=10):
    """Cross-test: select features by method A, weight by method B.

    Per-fold importance computation on training data only.
    Returns weighted_r2 for similarity retrieval.
    """
    sim_w_preds, sim_true = [], []

    for train_idx, val_idx in cv_split(groups, n_folds, X_concat, labels):
        train_reps = {k: reps[k][train_idx] for k in sorted(reps.keys())}
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Compute importance with both methods on training data
        imp_sel = compute_feature_importance(
            train_reps, y_train, method=select_method)
        imp_sel_concat = np.concatenate(
            [imp_sel[r] for r in sorted(imp_sel.keys())])

        imp_wt = compute_feature_importance(
            train_reps, y_train, method=weight_method)
        imp_wt_concat = np.concatenate(
            [imp_wt[r] for r in sorted(imp_wt.keys())])

        # Select top features by method A
        top_idx = np.argsort(imp_sel_concat)[-budget:]

        X_train_sub = X_concat[train_idx][:, top_idx]
        X_val_sub = X_concat[val_idx][:, top_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sub)
        X_val_s = scaler.transform(X_val_sub)

        # Weight by method B
        weights = _importance_to_weights(imp_wt_concat[top_idx])

        X_train_w = X_train_s * weights
        X_val_w = X_val_s * weights

        nn_k = min(k_neighbors, len(y_train) - 1)
        nn = NearestNeighbors(n_neighbors=nn_k)
        nn.fit(X_train_w)
        dists, idx = nn.kneighbors(X_val_w)
        w = 1.0 / (dists + 1e-10)
        preds = np.average(y_train[idx], axis=1, weights=w)

        sim_w_preds.extend(preds)
        sim_true.extend(y_val)

    return float(r2_score(np.array(sim_true), np.array(sim_w_preds)))


def _evaluate_low_dim(X_concat, reps, labels, groups, n_folds, method,
                      budget, k_neighbors=10):
    """Low-dim: per-fold select top-budget features + weight by importance.

    Returns {weighted_r2, unweighted_r2, delta_r2}.
    """
    sim_uw_preds, sim_w_preds, sim_true = [], [], []

    for train_idx, val_idx in cv_split(groups, n_folds, X_concat, labels):
        train_reps = {k: reps[k][train_idx] for k in sorted(reps.keys())}
        y_train, y_val = labels[train_idx], labels[val_idx]

        imp_dict = compute_feature_importance(
            train_reps, y_train, method=method)
        imp_concat = np.concatenate(
            [imp_dict[r] for r in sorted(imp_dict.keys())])

        top_idx = np.argsort(imp_concat)[-budget:]
        weights = _importance_to_weights(imp_concat[top_idx])

        X_train_sub = X_concat[train_idx][:, top_idx]
        X_val_sub = X_concat[val_idx][:, top_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sub)
        X_val_s = scaler.transform(X_val_sub)

        nn_k = min(k_neighbors, len(y_train) - 1)

        # Unweighted
        nn_uw = NearestNeighbors(n_neighbors=nn_k)
        nn_uw.fit(X_train_s)
        d_uw, i_uw = nn_uw.kneighbors(X_val_s)
        w_uw = 1.0 / (d_uw + 1e-10)
        sim_uw_preds.extend(np.average(y_train[i_uw], axis=1, weights=w_uw))

        # Weighted
        X_train_w = X_train_s * weights
        X_val_w = X_val_s * weights
        nn_w = NearestNeighbors(n_neighbors=nn_k)
        nn_w.fit(X_train_w)
        d_w, i_w = nn_w.kneighbors(X_val_w)
        w_w = 1.0 / (d_w + 1e-10)
        sim_w_preds.extend(np.average(y_train[i_w], axis=1, weights=w_w))

        sim_true.extend(y_val)

    sim_true = np.array(sim_true)
    uw_r2 = r2_score(sim_true, np.array(sim_uw_preds))
    w_r2 = r2_score(sim_true, np.array(sim_w_preds))
    return {'weighted_r2': float(w_r2), 'unweighted_r2': float(uw_r2),
            'delta_r2': float(w_r2 - uw_r2)}


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
    task_type = data.get('task_type', 'regression')
    if task_type == 'classification':
        print(f"  SKIPPING: c04 currently only supports regression targets.")
        print(f"  TODO: add classification branch (KNeighborsClassifier + majority-vote retrieval).")
        return {'skipped': True, 'reason': 'classification_not_supported',
                'task_type': task_type}

    importance_methods = config['importance_methods']
    n_folds = config['n_splits']
    budget = kwargs.get('budget', 25)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    print(f"  Concatenated features: {X_concat.shape}")

    results = {'methods': {}, 'cross_test': {}, 'low_dim': {}}

    # Evaluate each method with per-fold importance computation
    print("\n  Evaluating methods (per-fold importance)...")
    for method in importance_methods:
        print(f"  {method}...", end=' ', flush=True)
        start = time.time()
        try:
            knn_result, sim_result = _evaluate_knn_and_sim(
                X_concat, reps, labels, groups, n_folds, method)

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
                try:
                    w_r2 = _evaluate_cross_test(
                        X_concat, reps, labels, groups, n_folds,
                        select_method, weight_method, budget)
                    key = f"{select_method}__x__{weight_method}"
                    results['cross_test'][key] = w_r2
                except Exception:
                    pass

    # Low-dim: budget=25 selected features + importance weighting
    print(f"\n  Low-dim (budget={budget}) selected + weighted...")
    for method in method_names:
        try:
            ld = _evaluate_low_dim(
                X_concat, reps, labels, groups, n_folds, method, budget)
            results['low_dim'][method] = ld
            print(f"    {method}: weighted R2={ld['weighted_r2']:.4f}, "
                  f"unweighted R2={ld['unweighted_r2']:.4f}")
        except Exception as e:
            print(f"    {method}: FAILED - {e}")

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

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, budget=args.budget)
    save_experiment_results(results, args.output, 'c04_importance_as_metric',
                            args.dataset)
