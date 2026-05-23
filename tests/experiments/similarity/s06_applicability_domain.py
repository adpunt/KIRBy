#!/usr/bin/env python3
"""S06: Applicability Domain -- hybrid distance tracks prediction uncertainty.

Paper: Similarity
Conclusion: Hybrid distance correlates with prediction error (AD).
Dependencies: s02 (reads JSON, optional -- falls back to fresh search).

Per-fold CV with per-fold hybrid construction:
  - kNN distances in concat, hybrid, and weighted hybrid spaces
  - RF prediction |errors|
  - Spearman correlation between distance and |error|
  - Reliability diagram (distance decile -> mean error)
  - Conformal calibration comparison: concat vs hybrid
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results, convert_numpy,
    get_fitted_hybrid, get_fitted_pool_hybrid, flatten_weights, flatten_fold_weights, make_cv_splitter,
    apply_feature_selection, compute_feature_importance,
    get_domain_metric, get_model_factories, build_fold_hybrid,
    get_best_for_downstream, extract_top3_base_reps,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)
from kirby.hybrid import create_hybrid


def run(data, config, output_dir, dataset_name, **kwargs):
    """Hybrid distance tracks prediction uncertainty better than standard distance."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S06: APPLICABILITY DOMAIN")
    print("=" * 70)

    n_folds = config['n_splits']
    search_speed = config['search_speed']
    k_values = kwargs.get('k_values', [5])

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    task_type = data['task_type']
    domain = data.get('domain', 'molecular')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    _rf_factory = get_model_factories(['rf'], task_type=task_type,
                                       n_samples=len(labels))['rf']

    results = {'distance_error_correlation': {}, 'reliability': {}, 'conformal': {}}

    augmentations = data.get('augmentations')
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    # Get hybrid + blueprint for per-fold CV
    try:
        budget = kwargs.get('budget', 100)
        augmentations_search = data.get('augmentations')
        X_hybrid, fi, search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='knn', budget=budget,
            augmentations=augmentations_search,
            speed=config.get('search_speed', 'full'),
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset_name, distance=True)
        weights_flat = flatten_fold_weights(fi) if fi else np.ones(X_hybrid.shape[1])
        best_config = get_best_for_downstream(search, 'knn')
        bp = best_config['blueprint'] if best_config and 'blueprint' in best_config else None
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            results['top3_base_reps'] = [(n, s) for n, s in top3]
    except Exception as e:
        print(f"  Hybrid search failed: {e}")
        return {'error': str(e)}

    splitter, use_groups = make_cv_splitter(n_folds, groups)
    sample_X = next(iter(reps.values()))

    for k in k_values:
        print(f"\n  k={k}:")
        all_d_concat = []
        all_d_concat_w = []
        all_d_hybrid = []
        all_d_weighted = []
        all_d_per_rep = {rname: [] for rname in reps}
        all_d_per_rep_w = {rname: [] for rname in reps}
        all_errors = []
        all_errors_neutral = []  # from RF on concat (removes tautology)

        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Per-fold hybrid using search blueprint
            train_reps_h = {name: rep[train_idx] for name, rep in reps.items()}
            val_reps_h = {name: rep[val_idx] for name, rep in reps.items()}
            aug_tr = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
            aug_va = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps_h, val_reps_h, y_train, bp,
                    augmentations_train=aug_tr, augmentations_val=aug_va)

                # Importance weights for this fold
                imp = compute_feature_importance(
                    train_reps_h, y_train, method='lightgbm_gain')
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

            # Train RF on hybrid features (existing — may have tautological AD advantage)
            scaler_h = StandardScaler()
            X_tr_s = scaler_h.fit_transform(X_tr_h)
            X_va_s = scaler_h.transform(X_va_h)
            rf = _rf_factory()
            rf.fit(X_tr_s, y_train)
            y_pred = rf.predict(X_va_s)
            # Classification: error = misclassification (0/1). Regression: |residual|.
            if task_type == 'classification':
                errors = (y_val != y_pred).astype(float)
            else:
                errors = np.abs(y_val - y_pred)

            # Neutral model: RF on concat (removes tautology — errors are
            # independent of hybrid space, so hybrid distance advantage is genuine)
            scaler_c = StandardScaler()
            X_tr_c = scaler_c.fit_transform(X_concat[train_idx])
            X_va_c = scaler_c.transform(X_concat[val_idx])
            rf_neutral = _rf_factory()
            rf_neutral.fit(X_tr_c, y_train)
            y_pred_neutral = rf_neutral.predict(X_va_c)
            if task_type == 'classification':
                errors_neutral = (y_val != y_pred_neutral).astype(float)
            else:
                errors_neutral = np.abs(y_val - y_pred_neutral)

            # Concat weights for this fold
            w_concat = np.concatenate([imp[name] for name in sorted(reps.keys())
                                       if name in imp])
            w_concat = np.sqrt(np.maximum(w_concat, 0))

            # Distance to k-NN in training set for each val molecule
            # Concat space (unweighted)
            scaler_c = StandardScaler()
            Xtr_c = scaler_c.fit_transform(X_concat[train_idx])
            Xva_c = scaler_c.transform(X_concat[val_idx])
            nn_c = NearestNeighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_c.fit(Xtr_c)
            dists_c, _ = nn_c.kneighbors(Xva_c)
            d_concat = dists_c.mean(axis=1)

            # Concat space (weighted)
            nn_cw = NearestNeighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_cw.fit(Xtr_c * w_concat)
            dists_cw, _ = nn_cw.kneighbors(Xva_c * w_concat)
            d_concat_w = dists_cw.mean(axis=1)

            # Per-rep distances (unweighted + weighted)
            for rname, X_rep in reps.items():
                scaler_r = StandardScaler()
                Xtr_r = scaler_r.fit_transform(X_rep[train_idx])
                Xva_r = scaler_r.transform(X_rep[val_idx])
                nn_r = NearestNeighbors(n_neighbors=min(k, len(y_train) - 1))
                nn_r.fit(Xtr_r)
                dr, _ = nn_r.kneighbors(Xva_r)
                all_d_per_rep[rname].extend(dr.mean(axis=1).tolist())

                if rname in imp:
                    w_r = np.sqrt(np.maximum(imp[rname], 0))
                    nn_rw = NearestNeighbors(n_neighbors=min(k, len(y_train) - 1))
                    nn_rw.fit(Xtr_r * w_r)
                    drw, _ = nn_rw.kneighbors(Xva_r * w_r)
                    all_d_per_rep_w[rname].extend(drw.mean(axis=1).tolist())

            # Hybrid space (unweighted)
            nn_h = NearestNeighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_h.fit(X_tr_s)
            dists_h, _ = nn_h.kneighbors(X_va_s)
            d_hybrid = dists_h.mean(axis=1)

            # Weighted hybrid space
            nn_w = NearestNeighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_w.fit(X_tr_s * w)
            dists_w, _ = nn_w.kneighbors(X_va_s * w)
            d_weighted = dists_w.mean(axis=1)

            all_d_concat.extend(d_concat.tolist())
            all_d_concat_w.extend(d_concat_w.tolist())
            all_d_hybrid.extend(d_hybrid.tolist())
            all_d_weighted.extend(d_weighted.tolist())
            all_errors.extend(errors.tolist())
            all_errors_neutral.extend(errors_neutral.tolist())

        if not all_errors:
            continue

        all_d_concat = np.array(all_d_concat)
        all_d_concat_w = np.array(all_d_concat_w)
        all_d_hybrid = np.array(all_d_hybrid)
        all_d_weighted = np.array(all_d_weighted)
        all_errors = np.array(all_errors)

        # Spearman correlations: distance vs |error| (hybrid-model errors)
        rho_c, _ = spearmanr(all_d_concat, all_errors)
        rho_cw, _ = spearmanr(all_d_concat_w, all_errors)
        rho_h, _ = spearmanr(all_d_hybrid, all_errors)
        rho_w, _ = spearmanr(all_d_weighted, all_errors)

        corr_result = {
            'concat': float(rho_c),
            'concat_weighted': float(rho_cw),
            'hybrid': float(rho_h),
            'hybrid_weighted': float(rho_w),
        }

        # Neutral-model correlations (errors from RF on concat — removes tautology)
        all_errors_neutral = np.array(all_errors_neutral)
        if len(all_errors_neutral) == len(all_d_concat):
            rho_c_n, _ = spearmanr(all_d_concat, all_errors_neutral)
            rho_h_n, _ = spearmanr(all_d_hybrid, all_errors_neutral)
            rho_w_n, _ = spearmanr(all_d_weighted, all_errors_neutral)
            corr_result['neutral_model'] = {
                'concat': float(rho_c_n),
                'hybrid': float(rho_h_n),
                'hybrid_weighted': float(rho_w_n),
            }
            print(f"    Neutral-model AD: concat={rho_c_n:.4f}, hybrid={rho_h_n:.4f}, "
                  f"hybrid_w={rho_w_n:.4f}")

        # Per-rep correlations
        for rname in reps:
            if all_d_per_rep[rname]:
                rho_r, _ = spearmanr(np.array(all_d_per_rep[rname]), all_errors)
                corr_result[rname] = float(rho_r)
            if all_d_per_rep_w[rname]:
                rho_rw, _ = spearmanr(np.array(all_d_per_rep_w[rname]), all_errors)
                corr_result[f'{rname}_weighted'] = float(rho_rw)

        results['distance_error_correlation'][f'k{k}'] = corr_result
        print(f"    Spearman(dist, |error|): concat={rho_c:.4f}, concat_w={rho_cw:.4f}, "
              f"hybrid={rho_h:.4f}, hybrid_w={rho_w:.4f}")

        # Reliability diagram: bin by distance decile
        for dist_name, dists in [('concat', all_d_concat),
                                  ('hybrid', all_d_hybrid),
                                  ('hybrid_weighted', all_d_weighted)]:
            deciles = np.percentile(dists, np.arange(0, 110, 10))
            decile_errors = []
            decile_counts = []
            for di in range(10):
                mask = (dists >= deciles[di]) & (dists < deciles[di + 1])
                if di == 9:  # include upper bound in last bin
                    mask = (dists >= deciles[di]) & (dists <= deciles[di + 1])
                if mask.sum() > 0:
                    decile_errors.append(float(np.mean(all_errors[mask])))
                    decile_counts.append(int(mask.sum()))
                else:
                    decile_errors.append(float('nan'))
                    decile_counts.append(0)

            results['reliability'].setdefault(f'k{k}', {})[dist_name] = {
                'decile_errors': decile_errors,
                'decile_counts': decile_counts,
            }

    # Conformal calibration comparison (regression only — residual-based intervals)
    if task_type == 'classification':
        print("\n  Conformal calibration: SKIP (regression-only, not applicable to classification)")
        results['conformal'] = {'skipped': True, 'reason': 'classification'}
    else:
        print("\n  Conformal calibration (CV):")
        try:
            splitter_conf, use_groups_conf = make_cv_splitter(n_folds, groups)
            conf_picp = {'concat': [], 'hybrid': []}
            conf_mpiw = {'concat': [], 'hybrid': []}
            alpha = 0.1

            for train_idx, val_idx in (splitter_conf.split(sample_X, labels, groups=groups) if use_groups_conf else splitter_conf.split(sample_X, labels)):
                y_train, y_val = labels[train_idx], labels[val_idx]

                # Per-fold hybrid using search blueprint (avoids data leakage)
                train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
                val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
                aug_tr_c = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                aug_va_c = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                try:
                    X_hybrid_tr, X_hybrid_va, fi = build_fold_hybrid(
                        train_reps_fold, val_reps_fold, y_train, bp,
                        augmentations_train=aug_tr_c, augmentations_val=aug_va_c)
                except Exception:
                    continue

                # Split train into proper-train and calibration (group-aware)
                train_groups = groups[train_idx] if groups is not None else None
                if train_groups is not None and len(np.unique(train_groups)) >= 2:
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                    tr_local, cal_local = next(gss.split(
                        np.arange(len(y_train)), y_train, groups=train_groups))
                else:
                    n_cal = max(int(len(y_train) * 0.2), 10)
                    rng = np.random.RandomState(42)
                    perm = rng.permutation(len(y_train))
                    cal_local, tr_local = perm[:n_cal], perm[n_cal:]

                for feat_name, X_tr_all, X_va in [
                    ('concat', X_concat[train_idx], X_concat[val_idx]),
                    ('hybrid', X_hybrid_tr, X_hybrid_va),
                ]:
                    scaler = StandardScaler()
                    X_tr_all_s = scaler.fit_transform(X_tr_all)
                    X_va_s = scaler.transform(X_va)

                    X_tr = X_tr_all_s[tr_local]
                    X_cal = X_tr_all_s[cal_local]
                    y_tr = y_train[tr_local]
                    y_cal = y_train[cal_local]

                    # Train RF, compute conformal quantile on calibration set
                    rf = _rf_factory()
                    rf.fit(X_tr, y_tr)

                    # Calibration: residuals normalized by tree variance
                    y_pred_cal = rf.predict(X_cal)
                    tree_preds_cal = np.array([t.predict(X_cal) for t in rf.estimators_])
                    cal_std = tree_preds_cal.std(axis=0) + 1e-10
                    conformity = np.abs(y_cal - y_pred_cal) / cal_std
                    q = np.quantile(conformity, 1 - alpha)

                    # Val: predict + interval
                    y_pred_va = rf.predict(X_va_s)
                    tree_preds_va = np.array([t.predict(X_va_s) for t in rf.estimators_])
                    va_std = tree_preds_va.std(axis=0)
                    interval_width = 2 * q * va_std

                    picp = float(np.mean(np.abs(y_val - y_pred_va) <= interval_width / 2))
                    mpiw = float(np.mean(interval_width))
                    conf_picp[feat_name].append(picp)
                    conf_mpiw[feat_name].append(mpiw)

            results['conformal'] = {}
            for feat_name in ['concat', 'hybrid']:
                if conf_picp[feat_name]:
                    picp_mean = float(np.mean(conf_picp[feat_name]))
                    mpiw_mean = float(np.mean(conf_mpiw[feat_name]))
                    results['conformal'][feat_name] = {
                        'picp': picp_mean, 'mpiw': mpiw_mean,
                    }
                    print(f"    {feat_name.capitalize()}:  PICP={picp_mean:.3f}, "
                          f"MPIW={mpiw_mean:.3f}")
        except Exception as e:
            print(f"    Conformal failed: {e}")
            results['conformal'] = {'error': str(e)}

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S06: Applicability Domain',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--k-values', type=int, nargs='+', default=[5],
                        help='k values for kNN distance (default: [5])')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"k values: {args.k_values}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  k_values=args.k_values)
    save_experiment_results(results, args.output, 's06_applicability_domain', args.dataset)
