#!/usr/bin/env python3
"""S08: Optimal Dimension for Similarity vs Prediction.

Paper: Similarity
Hypothesis: Optimal dimensionality may differ for similarity vs prediction.
Dependencies: None (standalone).

At each budget, runs get_hybrid_features_and_weights with specific budget and
computes: MLE intrinsic dim, SAC, violation rate, distance correlation with
full concat, RF R2 and kNN R2. Also computes per-rep baselines and concat
reference. Finds optimal budget for prediction (RF) vs similarity (SAC).

Includes Pareto frontier analysis to quantify the SAC-vs-prediction trade-off
and determine whether the optima genuinely differ or coincide.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_hybrid_features_and_weights, estimate_intrinsic_dim_mle, make_cv_splitter,
    get_domain_metric, get_model_factories,
    build_fold_hybrid,
)
from experiments.similarity.similarity_helpers import (
    similarity_activity_correlation, violation_rate,
)
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


def run(data, config, output_dir, dataset_name, **kwargs):
    """The optimal hybrid dimension may differ for prediction vs similarity.
    Similarity may peak at lower dim than prediction (curse of dimensionality)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S08: OPTIMAL DIMENSION FOR SIMILARITY VS PREDICTION")
    print("=" * 70)

    n_folds = config['n_splits']
    search_speed = config['search_speed']
    budgets = kwargs.get('budgets', config.get('budgets', [15, 25, 50, 75, 100, 150, 200]))
    n_sample = kwargs.get('n_sample', 5000)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data.get('task_type', 'regression')
    is_clf = task_type == 'classification'
    rf_factory = get_model_factories(['rf'], task_type=task_type,
                                      n_samples=len(labels))['rf']

    def _make_knn(n_train):
        k = min(5, n_train - 1)
        if is_clf:
            return KNeighborsClassifier(n_neighbors=k)
        return KNeighborsRegressor(n_neighbors=k)

    results = {'budgets': {}}

    # Full-dim reference
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    full_dim = X_concat.shape[1]

    splitter, use_groups = make_cv_splitter(n_folds, groups)
    sample_X = next(iter(reps.values()))

    # Pre-compute full-dim distance matrix + sampled pair indices (reused per budget)
    D_full = compute_distance_matrix(X_concat, standardize=True)
    n = len(labels)
    triu_i, triu_j = np.triu_indices(n, k=1)
    n_pairs = len(triu_i)
    if n_pairs > 10000:
        rng = np.random.RandomState(42)
        _idx = rng.choice(n_pairs, 10000, replace=False)
        triu_i_s, triu_j_s = triu_i[_idx], triu_j[_idx]
    else:
        triu_i_s, triu_j_s = triu_i, triu_j
    d_full_sampled = D_full[triu_i_s, triu_j_s]

    for budget in budgets:
        print(f"\n  Budget={budget}:")
        t0 = time.time()

        try:
            X_hybrid, weights_flat, search = get_hybrid_features_and_weights(
                reps, labels, groups, search_speed, 'lightgbm_gain',
                budget=budget,
                augmentations=data.get('augmentations'))
        except Exception as e:
            print(f"    Failed: {e}")
            results['budgets'][budget] = {'error': str(e)}
            continue

        actual_dim = X_hybrid.shape[1]

        # MLE intrinsic dimensionality of the hybrid
        try:
            X_hybrid_scaled = StandardScaler().fit_transform(X_hybrid)
            mle_id = float(estimate_intrinsic_dim_mle(X_hybrid_scaled, k=10))
        except Exception:
            mle_id = float('nan')

        # Similarity quality
        D_weighted = compute_distance_matrix(X_hybrid, feature_weights=weights_flat,
                                             standardize=True)
        sac_rho, _ = similarity_activity_correlation(D_weighted, labels, n_sample)
        viol = violation_rate(D_weighted, labels, k=5)

        # Distance correlation with full concat (using pre-computed pairs)
        d_hybrid_sampled = D_weighted[triu_i_s, triu_j_s]
        dist_corr, _ = spearmanr(d_full_sampled, d_hybrid_sampled)

        # Prediction quality: RF R2 (CV) -- use search blueprint's method
        bp = search.best_overall['blueprint'] if search.best_overall else None
        if bp is None:
            print(f"    No blueprint at budget={budget}, skipping prediction")
            continue

        rf_scores = []
        knn_scores = []
        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]

            train_reps_fold = {name: rep[train_idx] for name, rep in reps.items()}
            val_reps_fold = {name: rep[val_idx] for name, rep in reps.items()}
            aug_tr = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
            aug_va = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps_fold, val_reps_fold, y_train, bp,
                    augmentations_train=aug_tr, augmentations_val=aug_va)
            except Exception:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_h)
            X_va_s = scaler.transform(X_va_h)

            # RF
            rf = rf_factory()
            rf.fit(X_tr_s, y_train)
            rf_scores.append(metric_fn(y_val, rf.predict(X_va_s)))

            # kNN
            knn = _make_knn(len(y_train))
            knn.fit(X_tr_s, y_train)
            knn_scores.append(metric_fn(y_val, knn.predict(X_va_s)))

        rf_score = float(np.mean(rf_scores)) if rf_scores else float('nan')
        knn_score = float(np.mean(knn_scores)) if knn_scores else float('nan')

        elapsed = time.time() - t0

        results['budgets'][budget] = {
            'actual_dim': actual_dim,
            'mle_id': mle_id,
            'rf_score': rf_score,
            'knn_score': knn_score,
            'metric': metric_name,
            'sac_rho': sac_rho,
            'violation_rate': viol,
            'dist_corr_with_full': dist_corr,
            'time_seconds': elapsed,
        }
        print(f"    dim={actual_dim} (MLE_ID={mle_id:.1f}): RF_{metric_name}={rf_score:.4f}, "
              f"kNN_{metric_name}={knn_score:.4f}, "
              f"SAC={sac_rho:.4f}, viol={viol:.3f}, dist_corr={dist_corr:.3f}")

    # Per-rep baselines (unweighted + weighted)
    print("\n  Individual rep baselines:")
    results['per_rep'] = {}
    best_rep_sac = -1
    best_rep_name = None

    # Compute importance once for weighted baselines
    from kirby.hybrid import compute_feature_importance as _cfi
    try:
        importance = _cfi(reps, labels, method='lightgbm_gain')
    except Exception:
        importance = None

    for rep_name, X in reps.items():
        D_rep = compute_distance_matrix(X, standardize=True)
        sac_rep, _ = similarity_activity_correlation(D_rep, labels, n_sample)
        viol_rep = violation_rate(D_rep, labels, k=5)

        # Weighted SAC
        sac_rep_w = float('nan')
        if importance is not None and rep_name in importance:
            D_rep_w = compute_distance_matrix(X, feature_weights=importance[rep_name],
                                              standardize=True)
            sac_rep_w, _ = similarity_activity_correlation(D_rep_w, labels, n_sample)

        rf_rep = []
        knn_rep = []
        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_va = scaler.transform(X[val_idx])

            rf = rf_factory()
            rf.fit(X_tr, y_train)
            rf_rep.append(metric_fn(y_val, rf.predict(X_va)))

            knn = _make_knn(len(y_train))
            knn.fit(X_tr, y_train)
            knn_rep.append(metric_fn(y_val, knn.predict(X_va)))

        # MLE ID for individual rep
        try:
            rep_mle = float(estimate_intrinsic_dim_mle(
                StandardScaler().fit_transform(X), k=10))
        except Exception:
            rep_mle = float('nan')

        results['per_rep'][rep_name] = {
            'dim': X.shape[1],
            'mle_id': rep_mle,
            'rf_score': float(np.mean(rf_rep)),
            'knn_score': float(np.mean(knn_rep)),
            'metric': metric_name,
            'sac_rho': sac_rep,
            'sac_rho_weighted': sac_rep_w,
            'violation_rate': viol_rep,
        }
        if best_rep_name is None or sac_rep > best_rep_sac:
            best_rep_sac = sac_rep
            best_rep_name = rep_name
        print(f"    {rep_name} (dim={X.shape[1]}): RF_{metric_name}={np.mean(rf_rep):.4f}, "
              f"kNN_{metric_name}={np.mean(knn_rep):.4f}, SAC={sac_rep:.4f}, SAC_w={sac_rep_w:.4f}")

    results['best_individual_rep'] = best_rep_name
    print(f"  Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f})")

    # Concat (reference, not standard practice)
    print(f"\n  Concat reference (dim={full_dim}):")
    sac_full, _ = similarity_activity_correlation(D_full, labels, n_sample)
    viol_full = violation_rate(D_full, labels, k=5)

    rf_scores_full = []
    knn_scores_full = []
    for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
        y_train, y_val = labels[train_idx], labels[val_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_concat[train_idx])
        X_va = scaler.transform(X_concat[val_idx])

        rf = rf_factory()
        rf.fit(X_tr, y_train)
        rf_scores_full.append(metric_fn(y_val, rf.predict(X_va)))

        knn = _make_knn(len(y_train))
        knn.fit(X_tr, y_train)
        knn_scores_full.append(metric_fn(y_val, knn.predict(X_va)))

    concat_ref = {
        'dim': full_dim,
        'rf_score': float(np.mean(rf_scores_full)),
        'knn_score': float(np.mean(knn_scores_full)),
        'metric': metric_name,
        'sac_rho': sac_full,
        'violation_rate': viol_full,
    }
    # Weighted concat SAC
    if importance is not None:
        w_concat = np.concatenate([importance[k] for k in sorted(reps.keys())
                                   if k in importance])
        D_full_w = compute_distance_matrix(X_concat, feature_weights=w_concat,
                                           standardize=True)
        sac_full_w, _ = similarity_activity_correlation(D_full_w, labels, n_sample)
        concat_ref['sac_rho_weighted'] = sac_full_w
        print(f"    RF_{metric_name}={concat_ref['rf_score']:.4f}, "
              f"kNN_{metric_name}={concat_ref['knn_score']:.4f}, "
              f"SAC={sac_full:.4f}, SAC_w={sac_full_w:.4f}")
    else:
        print(f"    RF_{metric_name}={concat_ref['rf_score']:.4f}, "
              f"kNN_{metric_name}={concat_ref['knn_score']:.4f}, "
              f"SAC={sac_full:.4f}, viol={viol_full:.3f}")
    results['concat_reference'] = concat_ref

    # Find optimal budget for prediction vs similarity
    budget_list = sorted(results['budgets'].keys())
    if budget_list:
        _best_fn = min if lower_is_better else max
        _default = 999 if lower_is_better else -999
        best_rf_budget = _best_fn(budget_list,
                                   key=lambda b: results['budgets'][b].get('rf_score', _default))
        best_knn_budget = _best_fn(budget_list,
                                    key=lambda b: results['budgets'][b].get('knn_score', _default))
        # Highest SAC = best distance-activity correlation
        best_sac_budget = max(budget_list,
                              key=lambda b: results['budgets'][b].get('sac_rho', -999))

        results['optimal'] = {
            'best_rf_budget': best_rf_budget,
            'best_knn_budget': best_knn_budget,
            'best_sac_budget': best_sac_budget,
        }
        print(f"\n  Optimal budgets: RF={best_rf_budget}, "
              f"kNN={best_knn_budget}, SAC={best_sac_budget}")

        # ── Pareto frontier: SAC (higher=better) vs RF (direction depends on metric) ──
        # Both objectives should be "higher is better" for Pareto.
        # RF score is already in the right direction (metric_fn handles it).
        # For lower-is-better metrics (RMSE), negate so higher = better.
        pareto_points = []
        for b in budget_list:
            bd = results['budgets'][b]
            sac = bd.get('sac_rho', float('nan'))
            rf = bd.get('rf_score', float('nan'))
            if np.isnan(sac) or np.isnan(rf):
                continue
            rf_max = -rf if lower_is_better else rf  # convert to higher=better
            pareto_points.append((b, sac, rf_max, rf))

        pareto_front = []
        if pareto_points:
            # Sort by SAC descending; sweep for Pareto-optimal
            pareto_points.sort(key=lambda x: x[1], reverse=True)
            best_rf_seen = float('-inf')
            for budget_val, sac, rf_max, rf_orig in pareto_points:
                if rf_max >= best_rf_seen:
                    pareto_front.append({
                        'budget': budget_val,
                        'sac_rho': sac,
                        f'rf_{metric_name}': rf_orig,
                    })
                    best_rf_seen = rf_max

            # Quantify trade-off: how much SAC do you lose going from
            # SAC-optimal to RF-optimal budget, and vice versa?
            sac_at_rf_opt = results['budgets'][best_rf_budget].get('sac_rho', float('nan'))
            rf_at_sac_opt = results['budgets'][best_sac_budget].get('rf_score', float('nan'))
            sac_at_sac_opt = results['budgets'][best_sac_budget].get('sac_rho', float('nan'))
            rf_at_rf_opt = results['budgets'][best_rf_budget].get('rf_score', float('nan'))

            tradeoff = {
                'sac_loss_at_rf_optimum': float(sac_at_sac_opt - sac_at_rf_opt),
                'rf_loss_at_sac_optimum': float(rf_at_rf_opt - rf_at_sac_opt) if not lower_is_better
                    else float(rf_at_sac_opt - rf_at_rf_opt),
                'optima_differ': best_rf_budget != best_sac_budget,
                'budget_gap': abs(best_rf_budget - best_sac_budget),
            }

            results['pareto'] = {
                'frontier': pareto_front,
                'n_pareto_points': len(pareto_front),
                'tradeoff': tradeoff,
            }

            print(f"\n  Pareto frontier ({len(pareto_front)} points):")
            for pt in pareto_front:
                print(f"    budget={pt['budget']:4d}  "
                      f"SAC={pt['sac_rho']:.4f}  "
                      f"RF_{metric_name}={pt[f'rf_{metric_name}']:.4f}")
            if tradeoff['optima_differ']:
                print(f"  Trade-off: SAC-opt@{best_sac_budget} vs RF-opt@{best_rf_budget}")
                print(f"    SAC loss at RF-optimum: {tradeoff['sac_loss_at_rf_optimum']:.4f}")
                print(f"    RF loss at SAC-optimum: {tradeoff['rf_loss_at_sac_optimum']:.4f}")
            else:
                print(f"  No trade-off: both objectives peak at budget={best_rf_budget}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S08: Optimal Dimension for Similarity vs Prediction',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[15, 25, 50, 75, 100, 150, 200],
                        help='Budget values to test')
    parser.add_argument('--n-sample', type=int, default=5000,
                        help='Number of pairs to sample for SAC (default: 5000)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Budgets: {args.budgets}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  budgets=args.budgets, n_sample=args.n_sample)
    save_experiment_results(results, args.output, 's08_optimal_dimension', args.dataset)
