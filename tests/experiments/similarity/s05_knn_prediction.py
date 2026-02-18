#!/usr/bin/env python3
"""S05: kNN Prediction with Hybrid Distance.

Paper: Similarity
Conclusion: Hybrid-weighted kNN outperforms standard kNN.
Dependencies: S02 (reads JSON, optional).

CV evaluation with multiple conditions: hybrid_unweighted, hybrid_weighted,
concat_unweighted, concat_weighted. Multiple k values. Stratified analysis
by cliff compounds (if ecfp4 available). Per-rep kNN baselines (the real baselines).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results,
    get_hybrid_features_and_weights,
)

from kirby.hybrid import create_hybrid, apply_feature_selection, compute_feature_importance


# =============================================================================
# LOCAL HELPERS
# =============================================================================

def identify_cliff_compounds(ecfp4, y, tanimoto_threshold=0.55, activity_k=2.0):
    """Identify activity cliff compounds using ECFP4 Tanimoto.

    Definition based on Bajorath standard for ECFP4 (Stumpfe & Bajorath 2019):
      - Structural similarity: Tanimoto(ECFP4) >= tanimoto_threshold (default 0.55)
      - Activity difference: |delta_y| > activity_k * MAD(y)

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

    # Per-compound cliff pair count
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
        'cliff_pair_counts': cliff_pair_counts,
    }

    return cliff_mask, info


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Downstream task validation: hybrid-weighted kNN outperforms standard kNN,
    especially on cliff molecules."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S05: kNN PREDICTION WITH HYBRID DISTANCE")
    print("=" * 70)

    k_values = kwargs.get('k_values', [5, 10])
    conditions = kwargs.get('conditions', ['hybrid_unweighted', 'hybrid_weighted'])
    shared_search = kwargs.get('shared_search', None)
    search_speed = config['search_speed']
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'conditions': {}, 'cliff_analysis': {}}

    # Pre-compute concat
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    # Get hybrid via find_best_hybrid -- extract blueprint for per-fold CV
    X_hybrid = None
    weights_flat = None
    hybrid_blueprint = None
    if any(c.startswith('hybrid') for c in conditions):
        try:
            if shared_search is not None:
                X_hybrid, weights_flat, search = shared_search
                print(f"  Using shared search (dim={X_hybrid.shape[1]})")
            else:
                X_hybrid, weights_flat, search = get_hybrid_features_and_weights(
                    reps, labels, groups, search_speed, 'lightgbm_gain')
            if search.best_overall:
                hybrid_blueprint = search.best_overall['blueprint']
        except Exception as e:
            print(f"  Hybrid search failed: {e}")

    # Identify cliff molecules for stratified analysis (representation-independent)
    cliff_indices = set()
    cliff_info = None
    if 'ecfp4' in reps:
        cliff_mask, cliff_info = identify_cliff_compounds(reps['ecfp4'], labels)
        cliff_indices = set(np.where(cliff_mask)[0].tolist())
        print(f"  Cliff molecules (ECFP4 Tanimoto): {len(cliff_indices)} / {len(labels)} "
              f"({cliff_info['pct_cliff_compounds']:.1f}%)")
    else:
        print("  WARNING: ecfp4 not available, skipping cliff stratification")

    # CV evaluation
    gkf = GroupKFold(n_splits=n_folds)
    sample_X = next(iter(reps.values()))

    for condition in conditions:
        print(f"\n  Condition: {condition}")

        for k in k_values:
            fold_scores = []
            fold_cliff_scores = []
            fold_noncliff_scores = []
            fold_errors = []

            for train_idx, val_idx in gkf.split(sample_X, labels, groups=groups):
                y_train, y_val = labels[train_idx], labels[val_idx]
                scaler = StandardScaler()

                if condition == 'concat_unweighted':
                    X_tr = scaler.fit_transform(X_concat[train_idx])
                    X_va = scaler.transform(X_concat[val_idx])
                    model = KNeighborsRegressor(n_neighbors=min(k, len(y_train) - 1))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'concat_weighted':
                    # Compute importance on train, weight concat features
                    train_reps = {name: rep[train_idx] for name, rep in reps.items()}
                    try:
                        imp = compute_feature_importance(train_reps, y_train,
                                                        method='lightgbm_gain')
                        w = np.concatenate([imp[name] for name in sorted(reps.keys())])
                        w = np.sqrt(np.maximum(w, 0))
                    except Exception:
                        w = np.ones(X_concat.shape[1])

                    X_tr = scaler.fit_transform(X_concat[train_idx]) * w
                    X_va = scaler.transform(X_concat[val_idx]) * w
                    model = KNeighborsRegressor(n_neighbors=min(k, len(y_train) - 1))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'hybrid_unweighted' and hybrid_blueprint is not None:
                    # Per-fold hybrid using search blueprint
                    train_reps = {name: rep[train_idx] for name, rep in reps.items()}
                    val_reps = {name: rep[val_idx] for name, rep in reps.items()}
                    try:
                        X_tr_h, feat_info = create_hybrid(
                            train_reps, y_train,
                            allocation_method=hybrid_blueprint.allocation_method,
                            budget=hybrid_blueprint.budget,
                            importance_method=hybrid_blueprint.importance_method,
                        )
                        X_va_h = apply_feature_selection(val_reps, feat_info)
                    except Exception:
                        continue

                    X_tr = scaler.fit_transform(X_tr_h)
                    X_va = scaler.transform(X_va_h)
                    model = KNeighborsRegressor(n_neighbors=min(k, len(y_train) - 1))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'hybrid_weighted' and hybrid_blueprint is not None:
                    # Per-fold hybrid using search blueprint + importance weighting
                    train_reps = {name: rep[train_idx] for name, rep in reps.items()}
                    val_reps = {name: rep[val_idx] for name, rep in reps.items()}
                    try:
                        X_tr_h, feat_info = create_hybrid(
                            train_reps, y_train,
                            allocation_method=hybrid_blueprint.allocation_method,
                            budget=hybrid_blueprint.budget,
                            importance_method=hybrid_blueprint.importance_method,
                        )
                        X_va_h = apply_feature_selection(val_reps, feat_info)

                        imp = compute_feature_importance(
                            train_reps, y_train, method='lightgbm_gain')
                        # Get weights for selected features
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
                    except Exception:
                        continue

                    X_tr = scaler.fit_transform(X_tr_h) * w
                    X_va = scaler.transform(X_va_h) * w
                    model = KNeighborsRegressor(n_neighbors=min(k, len(y_train) - 1))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'rf_baseline':
                    X_tr = scaler.fit_transform(X_concat[train_idx])
                    X_va = scaler.transform(X_concat[val_idx])
                    model = RandomForestRegressor(n_estimators=100, random_state=42,
                                                  n_jobs=-1)
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                else:
                    continue

                r2 = r2_score(y_val, y_pred)
                fold_scores.append(r2)

                # Stratified: cliff vs non-cliff
                val_set = set(val_idx.tolist())
                cliff_val = np.array([i for i, vi in enumerate(val_idx)
                                      if vi in cliff_indices])
                noncliff_val = np.array([i for i, vi in enumerate(val_idx)
                                         if vi not in cliff_indices])

                if len(cliff_val) > 1:
                    fold_cliff_scores.append(r2_score(y_val[cliff_val],
                                                      y_pred[cliff_val]))
                if len(noncliff_val) > 1:
                    fold_noncliff_scores.append(r2_score(y_val[noncliff_val],
                                                         y_pred[noncliff_val]))

                # Per-sample errors for SALI correlation
                fold_errors.extend(list(zip(val_idx.tolist(),
                                           np.abs(y_val - y_pred).tolist())))

            if not fold_scores:
                continue

            r2_mean = float(np.mean(fold_scores))
            r2_std = float(np.std(fold_scores))
            cliff_r2 = float(np.mean(fold_cliff_scores)) if fold_cliff_scores else float('nan')
            noncliff_r2 = float(np.mean(fold_noncliff_scores)) if fold_noncliff_scores else float('nan')

            key = f"{condition}_k{k}"
            results['conditions'][key] = {
                'r2_mean': r2_mean,
                'r2_std': r2_std,
                'cliff_r2': cliff_r2,
                'noncliff_r2': noncliff_r2,
                'k': k,
                'condition': condition,
            }
            print(f"    k={k}: R2={r2_mean:.4f}+/-{r2_std:.3f}, "
                  f"cliff_R2={cliff_r2:.4f}, noncliff_R2={noncliff_r2:.4f}")

    # Per-rep kNN baselines (individual representations -- the real baselines)
    print("\n  Individual representation kNN baselines:")
    results['per_rep_baselines'] = {}
    best_k = k_values[len(k_values) // 2]  # use middle k value
    gkf2 = GroupKFold(n_splits=n_folds)
    for rep_name, X_rep in reps.items():
        fold_r2s_uw = []
        fold_r2s_w = []
        for train_idx, val_idx in gkf2.split(sample_X, labels, groups=groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # Unweighted
            model = KNeighborsRegressor(n_neighbors=min(best_k, len(train_idx) - 1))
            model.fit(X_tr, labels[train_idx])
            fold_r2s_uw.append(r2_score(labels[val_idx], model.predict(X_va)))

            # Weighted by importance (computed on train only)
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[train_idx]}, labels[train_idx],
                    method='lightgbm_gain')
                w = np.sqrt(np.maximum(imp[rep_name], 0))
                X_tr_w = X_tr * w
                X_va_w = X_va * w
                model_w = KNeighborsRegressor(
                    n_neighbors=min(best_k, len(train_idx) - 1))
                model_w.fit(X_tr_w, labels[train_idx])
                fold_r2s_w.append(r2_score(labels[val_idx], model_w.predict(X_va_w)))
            except Exception:
                fold_r2s_w.append(float('nan'))

        r2_mean = float(np.mean(fold_r2s_uw))
        r2_std = float(np.std(fold_r2s_uw))
        r2_w_mean = float(np.nanmean(fold_r2s_w)) if fold_r2s_w else float('nan')
        r2_w_std = float(np.nanstd(fold_r2s_w)) if fold_r2s_w else float('nan')
        results['per_rep_baselines'][rep_name] = {
            'r2_mean': r2_mean, 'r2_std': r2_std,
            'r2_weighted_mean': r2_w_mean, 'r2_weighted_std': r2_w_std,
            'k': best_k, 'dim': X_rep.shape[1],
        }
        delta = r2_w_mean - r2_mean
        print(f"    {rep_name} (dim={X_rep.shape[1]}): "
              f"kNN R2={r2_mean:.4f}+/-{r2_std:.3f}, "
              f"weighted R2={r2_w_mean:.4f}+/-{r2_w_std:.3f} "
              f"(delta={delta:+.4f})")

    # Summary: best k per condition
    print("\n  Summary (best k per condition):")
    for cond in conditions:
        cond_results = {k: v for k, v in results['conditions'].items()
                       if v['condition'] == cond}
        if cond_results:
            best_key = max(cond_results, key=lambda k: cond_results[k]['r2_mean'])
            best = cond_results[best_key]
            print(f"    {cond}: best k={best['k']}, "
                  f"R2={best['r2_mean']:.4f}")

    # Flag best individual rep
    if results['per_rep_baselines']:
        best_rep = max(results['per_rep_baselines'],
                       key=lambda r: results['per_rep_baselines'][r]['r2_mean'])
        best_rep_r2 = results['per_rep_baselines'][best_rep]['r2_mean']
        print(f"\n  Best individual rep baseline: {best_rep} "
              f"(kNN R2={best_rep_r2:.4f})")

        # Compare hybrid_weighted to best individual rep
        hw_results = {k: v for k, v in results['conditions'].items()
                      if v['condition'] == 'hybrid_weighted'}
        if hw_results:
            best_hw = max(hw_results.values(), key=lambda v: v['r2_mean'])
            delta = best_hw['r2_mean'] - best_rep_r2
            print(f"  Hybrid weighted kNN R2={best_hw['r2_mean']:.4f} "
                  f"(delta vs {best_rep}: {delta:+.4f})")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S05: kNN Prediction with Hybrid Distance',
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
    save_experiment_results(results, args.output, 's05_knn_prediction', args.dataset)
