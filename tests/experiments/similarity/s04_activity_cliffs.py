#!/usr/bin/env python3
"""S04: Activity Cliffs -- RMSE_cliff improvement.

Paper: Similarity
Conclusion: Hybrid reduces RMSE on activity cliffs.
Dependencies: S02 (reads JSON, optional).

Identifies cliff compounds using ECFP4 Tanimoto (representation-independent
definition based on Bajorath standard). CV evaluation of kNN on cliff vs
non-cliff compounds for: per-rep kNN baselines, hybrid kNN unweighted,
hybrid kNN weighted, RF baseline.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results,
    get_hybrid_features_and_weights, make_cv_splitter,
)

from kirby.hybrid import apply_feature_selection, compute_feature_importance
from experiments.complexity.complexity_helpers import cached_create_hybrid as create_hybrid
# TODO: create_hybrid API not finalised — revisit after framework experiments.


# =============================================================================
# LOCAL HELPERS
# =============================================================================

def identify_cliff_compounds(ecfp4, y, tanimoto_threshold=0.55, activity_k=2.0):
    """Identify activity cliff compounds using ECFP4 Tanimoto.

    Definition based on Bajorath standard for ECFP4 (Stumpfe & Bajorath 2019):
      - Structural similarity: Tanimoto(ECFP4) >= tanimoto_threshold (default 0.55)
      - Activity difference: |delta_y| > activity_k * MAD(y)

    Note: MoleculeACE uses Tc>=0.9 for curated binding-affinity datasets.
    For diverse property datasets (logD, logP), Tc>=0.55 is standard for ECFP4.
    The activity threshold adapts to each dataset via MAD(y).

    Returns:
        cliff_mask: boolean array, True for cliff compounds
        info: dict with thresholds, counts, per-compound cliff pair counts
    """
    n = len(y)

    # Adaptive activity threshold: k * MAD(y)
    mad_y = float(np.median(np.abs(y - np.median(y))))
    activity_threshold = activity_k * mad_y

    # Tanimoto similarity from binary ECFP4 (Tanimoto = 1 - Jaccard distance)
    jaccard_dist = scipy_cdist(ecfp4, ecfp4, metric='jaccard')
    tanimoto_sim = 1.0 - jaccard_dist
    np.fill_diagonal(tanimoto_sim, 0)

    # Activity difference matrix
    delta_y = np.abs(y[:, None] - y[None, :])

    # Cliff pair mask: high similarity AND large activity difference
    cliff_pair_mask = (tanimoto_sim >= tanimoto_threshold) & (delta_y > activity_threshold)

    # Cliff compounds: involved in at least one cliff pair
    cliff_mask = cliff_pair_mask.any(axis=1)

    # Per-compound cliff pair count (how many cliff partners each molecule has)
    cliff_pair_counts = cliff_pair_mask.sum(axis=1)

    n_cliff_pairs = int(cliff_pair_mask.sum()) // 2  # symmetric
    n_cliff_compounds = int(cliff_mask.sum())

    info = {
        'tanimoto_threshold': tanimoto_threshold,
        'activity_k': activity_k,
        'activity_threshold': activity_threshold,
        'mad_y': mad_y,
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': float(100.0 * n_cliff_compounds / n),
        'cliff_pair_counts': cliff_pair_counts,  # per-compound array
    }

    return cliff_mask, info


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Prediction performance on activity cliff compounds.

    Uses representation-independent cliff definition (Bajorath ECFP4 standard):
      - ECFP4 Tanimoto >= tanimoto_threshold
      - |delta_y| > activity_k * MAD(y)

    Reports RMSE_cliff, R2_cliff, RMSE_noncliff, R2_noncliff for:
      - Per-rep kNN baselines
      - Hybrid kNN (unweighted and weighted)
      - RF baseline
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT S04: PREDICTION PERFORMANCE ON ACTIVITY CLIFFS")
    print("=" * 70)

    tanimoto_threshold = kwargs.get('tanimoto_threshold', 0.55)
    activity_k = kwargs.get('activity_k', 2.0)
    shared_search = kwargs.get('shared_search', None)
    search_speed = config['search_speed']
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    # --- Identify cliff compounds using ECFP4 Tanimoto ---
    if 'ecfp4' not in reps:
        print("  ERROR: ecfp4 not in reps, cannot identify cliffs")
        return {'error': 'ecfp4 not available'}

    cliff_mask, cliff_info = identify_cliff_compounds(
        reps['ecfp4'], labels,
        tanimoto_threshold=tanimoto_threshold,
        activity_k=activity_k,
    )
    cliff_indices = set(np.where(cliff_mask)[0].tolist())

    print(f"  Cliff definition: Tanimoto>={tanimoto_threshold}, "
          f"|delta_y|>{activity_k}*MAD(y) = {cliff_info['activity_threshold']:.3f}")
    print(f"  Cliff compounds: {cliff_info['n_cliff_compounds']} / {len(labels)} "
          f"({cliff_info['pct_cliff_compounds']:.1f}%)")
    print(f"  Cliff pairs: {cliff_info['n_cliff_pairs']}")

    results = {
        'cliff_definition': {
            'tanimoto_threshold': tanimoto_threshold,
            'activity_k': activity_k,
            'activity_threshold': cliff_info['activity_threshold'],
            'mad_y': cliff_info['mad_y'],
            'n_cliff_compounds': cliff_info['n_cliff_compounds'],
            'n_cliff_pairs': cliff_info['n_cliff_pairs'],
            'pct_cliff_compounds': cliff_info['pct_cliff_compounds'],
        },
        'per_rep': {},
        'hybrid': {},
    }

    # --- Get hybrid blueprint for per-fold hybrid construction ---
    hybrid_blueprint = None
    if shared_search is not None:
        X_hybrid, weights_flat, search = shared_search
        if search.best_overall:
            hybrid_blueprint = search.best_overall['blueprint']
        print(f"  Using shared search (dim={X_hybrid.shape[1]})")
    else:
        try:
            X_hybrid, weights_flat, search = get_hybrid_features_and_weights(
                reps, labels, groups, search_speed, 'lightgbm_gain')
            if search.best_overall:
                hybrid_blueprint = search.best_overall['blueprint']
        except Exception as e:
            print(f"  Hybrid search failed: {e}")

    # --- CV evaluation of prediction on cliff vs non-cliff ---
    splitter, use_groups = make_cv_splitter(n_folds, groups)
    sample_X = next(iter(reps.values()))

    def _evaluate_condition(get_Xy_fn, pre_scaled=False):
        """Run CV for a condition, return cliff/noncliff metrics."""
        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_all, fold_rmse_cliff, fold_rmse_noncliff = [], [], []

        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]
            try:
                X_tr, X_va = get_Xy_fn(train_idx, val_idx, y_train)
            except Exception:
                continue
            if X_tr is None:
                continue

            if pre_scaled:
                X_tr_s, X_va_s = X_tr, X_va
            else:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)

            model = KNeighborsRegressor(n_neighbors=min(5, len(y_train) - 1))
            model.fit(X_tr_s, y_train)
            y_pred = model.predict(X_va_s)

            # Overall
            fold_all.append(r2_score(y_val, y_pred))
            fold_rmse_all.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))

            # Stratify by cliff membership
            val_cliff = np.array([i for i, vi in enumerate(val_idx) if vi in cliff_indices])
            val_noncliff = np.array([i for i, vi in enumerate(val_idx) if vi not in cliff_indices])

            if len(val_cliff) > 1:
                fold_cliff.append(r2_score(y_val[val_cliff], y_pred[val_cliff]))
                fold_rmse_cliff.append(float(np.sqrt(mean_squared_error(
                    y_val[val_cliff], y_pred[val_cliff]))))
            if len(val_noncliff) > 1:
                fold_noncliff.append(r2_score(y_val[val_noncliff], y_pred[val_noncliff]))
                fold_rmse_noncliff.append(float(np.sqrt(mean_squared_error(
                    y_val[val_noncliff], y_pred[val_noncliff]))))

        if not fold_all:
            return None
        return {
            'r2_all': float(np.mean(fold_all)),
            'r2_cliff': float(np.mean(fold_cliff)) if fold_cliff else float('nan'),
            'r2_noncliff': float(np.mean(fold_noncliff)) if fold_noncliff else float('nan'),
            'rmse_all': float(np.mean(fold_rmse_all)),
            'rmse_cliff': float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan'),
            'rmse_noncliff': float(np.mean(fold_rmse_noncliff)) if fold_rmse_noncliff else float('nan'),
        }

    # --- Per-rep kNN baselines (unweighted + weighted) ---
    print("\n  Per-rep kNN baselines (k=5):")
    for rep_name, X_rep in reps.items():
        # Unweighted
        def _get_rep(tr, va, yt, _X=X_rep):
            return _X[tr], _X[va]
        res = _evaluate_condition(_get_rep)
        if res:
            results['per_rep'][rep_name] = res

            # Weighted: importance-weighted distance on this rep
            def _get_rep_w(tr, va, yt, _X=X_rep, _rname=rep_name):
                imp = compute_feature_importance({_rname: _X[tr]}, yt, method='lightgbm_gain')
                w = np.sqrt(np.maximum(imp[_rname], 0))
                scaler = StandardScaler()
                return scaler.fit_transform(_X[tr]) * w, scaler.transform(_X[va]) * w
            res_w = _evaluate_condition(_get_rep_w, pre_scaled=True)
            if res_w:
                results['per_rep'][rep_name]['r2_cliff_weighted'] = res_w['r2_cliff']
                results['per_rep'][rep_name]['rmse_cliff_weighted'] = res_w['rmse_cliff']
                results['per_rep'][rep_name]['r2_all_weighted'] = res_w['r2_all']

            print(f"    {rep_name:<15} R2_all={res['r2_all']:.4f}  "
                  f"R2_cliff={res['r2_cliff']:.4f}  RMSE_cliff={res['rmse_cliff']:.4f}  "
                  f"R2_noncliff={res['r2_noncliff']:.4f}"
                  + (f"  | weighted: RMSE_cliff={res_w['rmse_cliff']:.4f}" if res_w else ""))

    # --- Hybrid kNN (unweighted) ---
    if hybrid_blueprint is not None:
        print("\n  Hybrid kNN (unweighted, k=5):")

        def _get_hybrid_uw(tr, va, yt):
            train_reps = {name: rep[tr] for name, rep in reps.items()}
            val_reps = {name: rep[va] for name, rep in reps.items()}
            X_tr_h, feat_info = create_hybrid(
                train_reps, yt,
                allocation_method=hybrid_blueprint.allocation_method,
                budget=hybrid_blueprint.budget,
                importance_method=hybrid_blueprint.importance_method,
            )
            X_va_h = apply_feature_selection(val_reps, feat_info)
            return X_tr_h, X_va_h

        res = _evaluate_condition(_get_hybrid_uw)
        if res:
            results['hybrid']['unweighted'] = res
            print(f"    hybrid_uw      R2_all={res['r2_all']:.4f}  "
                  f"R2_cliff={res['r2_cliff']:.4f}  RMSE_cliff={res['rmse_cliff']:.4f}  "
                  f"R2_noncliff={res['r2_noncliff']:.4f}")

        # --- Hybrid kNN (importance-weighted) ---
        print("  Hybrid kNN (weighted, k=5):")

        def _get_hybrid_w(tr, va, yt):
            train_reps = {name: rep[tr] for name, rep in reps.items()}
            val_reps = {name: rep[va] for name, rep in reps.items()}
            X_tr_h, feat_info = create_hybrid(
                train_reps, yt,
                allocation_method=hybrid_blueprint.allocation_method,
                budget=hybrid_blueprint.budget,
                importance_method=hybrid_blueprint.importance_method,
            )
            X_va_h = apply_feature_selection(val_reps, feat_info)
            imp = compute_feature_importance(train_reps, yt, method='lightgbm_gain')
            w_parts = []
            for rname in feat_info:
                if not isinstance(feat_info[rname], dict):
                    continue
                if 'selected_indices' not in feat_info[rname]:
                    continue
                idx = feat_info[rname]['selected_indices']
                if rname in imp:
                    w_parts.append(imp[rname][idx])
                else:
                    w_parts.append(np.ones(len(idx)))
            w = np.sqrt(np.maximum(np.concatenate(w_parts), 0))
            # Scale-then-weight (matches Exp 5 and compute_distance_matrix convention)
            scaler_w = StandardScaler()
            return scaler_w.fit_transform(X_tr_h) * w, scaler_w.transform(X_va_h) * w

        res = _evaluate_condition(_get_hybrid_w, pre_scaled=True)
        if res:
            results['hybrid']['weighted'] = res
            print(f"    hybrid_w       R2_all={res['r2_all']:.4f}  "
                  f"R2_cliff={res['r2_cliff']:.4f}  RMSE_cliff={res['rmse_cliff']:.4f}  "
                  f"R2_noncliff={res['r2_noncliff']:.4f}")

    # --- RF baseline (on best individual rep) ---
    print("\n  RF baseline (per-rep, best rep):")
    best_rep_name = max(results['per_rep'],
                        key=lambda r: results['per_rep'][r]['r2_all']) if results['per_rep'] else None
    if best_rep_name:
        X_best = reps[best_rep_name]

        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_cliff = []
        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_best[train_idx])
            X_va = scaler.transform(X_best[val_idx])
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_train)
            y_pred = rf.predict(X_va)

            fold_all.append(r2_score(y_val, y_pred))
            val_cliff = np.array([i for i, vi in enumerate(val_idx) if vi in cliff_indices])
            val_noncliff = np.array([i for i, vi in enumerate(val_idx) if vi not in cliff_indices])
            if len(val_cliff) > 1:
                fold_cliff.append(r2_score(y_val[val_cliff], y_pred[val_cliff]))
                fold_rmse_cliff.append(float(np.sqrt(mean_squared_error(
                    y_val[val_cliff], y_pred[val_cliff]))))
            if len(val_noncliff) > 1:
                fold_noncliff.append(r2_score(y_val[val_noncliff], y_pred[val_noncliff]))

        results['rf_baseline'] = {
            'rep': best_rep_name,
            'r2_all': float(np.mean(fold_all)),
            'r2_cliff': float(np.mean(fold_cliff)) if fold_cliff else float('nan'),
            'r2_noncliff': float(np.mean(fold_noncliff)) if fold_noncliff else float('nan'),
            'rmse_cliff': float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan'),
        }
        print(f"    RF({best_rep_name}) R2_all={results['rf_baseline']['r2_all']:.4f}  "
              f"R2_cliff={results['rf_baseline']['r2_cliff']:.4f}  "
              f"RMSE_cliff={results['rf_baseline']['rmse_cliff']:.4f}")

    # --- Summary ---
    print(f"\n  --- Summary (RMSE_cliff comparison) ---")
    all_methods = []
    for rname, rv in results['per_rep'].items():
        all_methods.append((f"kNN({rname})", rv.get('rmse_cliff', float('nan')),
                           rv.get('r2_cliff', float('nan'))))
        if 'rmse_cliff_weighted' in rv:
            all_methods.append((f"kNN({rname})_w", rv.get('rmse_cliff_weighted', float('nan')),
                               rv.get('r2_cliff_weighted', float('nan'))))
    if 'unweighted' in results.get('hybrid', {}):
        rv = results['hybrid']['unweighted']
        all_methods.append(("hybrid_kNN_uw", rv.get('rmse_cliff', float('nan')),
                           rv.get('r2_cliff', float('nan'))))
    if 'weighted' in results.get('hybrid', {}):
        rv = results['hybrid']['weighted']
        all_methods.append(("hybrid_kNN_w", rv.get('rmse_cliff', float('nan')),
                           rv.get('r2_cliff', float('nan'))))
    if 'rf_baseline' in results:
        rv = results['rf_baseline']
        all_methods.append((f"RF({rv['rep']})", rv.get('rmse_cliff', float('nan')),
                           rv.get('r2_cliff', float('nan'))))

    all_methods.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 999)
    print(f"  {'Method':<25} {'RMSE_cliff':<12} {'R2_cliff':<12}")
    print("  " + "-" * 49)
    for name, rmse, r2 in all_methods:
        print(f"  {name:<25} {rmse:<12.4f} {r2:<12.4f}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S04: Activity Cliffs',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's04_activity_cliffs', args.dataset)
