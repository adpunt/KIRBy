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

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_fitted_hybrid, flatten_fold_weights,
    estimate_intrinsic_dim_mle, make_cv_splitter,
    get_domain_metric, get_model_factories,
    build_fold_hybrid, extract_top3_base_reps,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DOWNSTREAM_MODELS,
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
    budgets = kwargs.get('budgets', config.get('budgets',
                         [10, 25, 50, 100, 250, 500, 1000, 2000]))
    n_sample = kwargs.get('n_sample', 5000)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data.get('task_type', 'regression')
    domain = data.get('domain', 'molecular')
    is_clf = task_type == 'classification'
    model_factories = get_model_factories(DOWNSTREAM_MODELS, task_type=task_type,
                                           n_samples=len(labels))
    print(f"  Models: {list(model_factories.keys())}")

    results = {'budgets': {}}
    _top3_reported = False

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
            X_hybrid, fi, search = get_fitted_hybrid(
                reps, labels, groups,
                aug_condition='uncapped' if data.get('augmentations') else 'none',
                augmentations=data.get('augmentations'),
                speed=search_speed,
                precomputed=data.get('precomputed_searches'),
                internal_evaluators=DISTANCE_EVALUATORS,
                budgets=[budget],
                metric=metric_name, task_type=task_type, domain=domain,
                downstream_model='knn')
            weights_flat = flatten_fold_weights(fi) if fi else np.ones(X_hybrid.shape[1])
            if not _top3_reported:
                top3 = extract_top3_base_reps(search)
                if top3:
                    print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
                    results['top3_base_reps'] = [(n, s) for n, s in top3]
                _top3_reported = True
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
        sac_rho, _ = similarity_activity_correlation(D_weighted, labels, n_sample, is_clf=is_clf)
        viol = violation_rate(D_weighted, labels, k=5, is_clf=is_clf)

        # Distance correlation with full concat (using pre-computed pairs)
        d_hybrid_sampled = D_weighted[triu_i_s, triu_j_s]
        dist_corr, _ = spearmanr(d_full_sampled, d_hybrid_sampled)

        # Prediction quality: RF R2 (CV) -- use search blueprint's method
        bp = search.best_by_model['knn']['blueprint'] if search.best_by_model.get('knn') else None
        if bp is None:
            print(f"    No blueprint at budget={budget}, skipping prediction")
            continue

        model_scores = {mname: [] for mname in model_factories}
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

            for mname, mfactory in model_factories.items():
                try:
                    m = mfactory()
                    m.fit(X_tr_s, y_train)
                    model_scores[mname].append(metric_fn(y_val, m.predict(X_va_s)))
                except Exception:
                    pass

        model_means = {mname: float(np.mean(s)) if s else float('nan')
                       for mname, s in model_scores.items()}

        elapsed = time.time() - t0

        budget_result = {
            'actual_dim': actual_dim,
            'mle_id': mle_id,
            'metric': metric_name,
            'sac_rho': sac_rho,
            'violation_rate': viol,
            'dist_corr_with_full': dist_corr,
            'time_seconds': elapsed,
            'model_scores': model_means,
        }
        # Keep rf_score/knn_score for backwards compat with Pareto/optimal analysis
        budget_result['rf_score'] = model_means.get('rf', float('nan'))
        budget_result['knn_score'] = model_means.get('knn', float('nan'))
        results['budgets'][budget] = budget_result

        scores_str = ', '.join(f"{mn}={model_means[mn]:.4f}" for mn in model_means
                               if not np.isnan(model_means[mn]))
        print(f"    dim={actual_dim} (MLE_ID={mle_id:.1f}): {scores_str}, "
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
        sac_rep, _ = similarity_activity_correlation(D_rep, labels, n_sample, is_clf=is_clf)
        viol_rep = violation_rate(D_rep, labels, k=5, is_clf=is_clf)

        # Weighted SAC
        sac_rep_w = float('nan')
        if importance is not None and rep_name in importance:
            D_rep_w = compute_distance_matrix(X, feature_weights=importance[rep_name],
                                              standardize=True)
            sac_rep_w, _ = similarity_activity_correlation(D_rep_w, labels, n_sample, is_clf=is_clf)

        rep_model_scores = {mname: [] for mname in model_factories}
        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_va = scaler.transform(X[val_idx])

            for mname, mfactory in model_factories.items():
                try:
                    m = mfactory()
                    m.fit(X_tr, y_train)
                    rep_model_scores[mname].append(metric_fn(y_val, m.predict(X_va)))
                except Exception:
                    pass

        rep_model_means = {mname: float(np.mean(s)) if s else float('nan')
                           for mname, s in rep_model_scores.items()}

        # MLE ID for individual rep
        try:
            rep_mle = float(estimate_intrinsic_dim_mle(
                StandardScaler().fit_transform(X), k=10))
        except Exception:
            rep_mle = float('nan')

        rep_result = {
            'dim': X.shape[1],
            'mle_id': rep_mle,
            'metric': metric_name,
            'sac_rho': sac_rep,
            'sac_rho_weighted': sac_rep_w,
            'violation_rate': viol_rep,
            'model_scores': rep_model_means,
            'rf_score': rep_model_means.get('rf', float('nan')),
            'knn_score': rep_model_means.get('knn', float('nan')),
        }
        results['per_rep'][rep_name] = rep_result
        if best_rep_name is None or sac_rep > best_rep_sac:
            best_rep_sac = sac_rep
            best_rep_name = rep_name
        scores_str = ', '.join(f"{mn}={rep_model_means[mn]:.4f}" for mn in rep_model_means
                               if not np.isnan(rep_model_means[mn]))
        print(f"    {rep_name} (dim={X.shape[1]}): {scores_str}, "
              f"SAC={sac_rep:.4f}, SAC_w={sac_rep_w:.4f}")

    results['best_individual_rep'] = best_rep_name
    print(f"  Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f})")

    # Top-3 baselines per model
    _best_fn = min if lower_is_better else max
    _default = 999 if lower_is_better else -999
    results['top3_by_model'] = {}
    for mname in model_factories:
        top3 = sorted(results['per_rep'].items(),
                       key=lambda x: x[1].get('model_scores', {}).get(mname, _default),
                       reverse=not lower_is_better)[:3]
        results['top3_by_model'][mname] = [
            (name, r['model_scores'].get(mname, float('nan'))) for name, r in top3]
    # Backwards compat
    results['top3_rf'] = results['top3_by_model'].get('rf', [])
    results['top3_knn'] = results['top3_by_model'].get('knn', [])
    for mname in list(model_factories.keys())[:3]:
        print(f"\n  Top-3 baselines by {mname} {metric_name}:")
        for name, score in results['top3_by_model'][mname]:
            print(f"    {name}: {score:.4f}")

    # Find optimal budget for prediction (per model) vs similarity
    budget_list = sorted(results['budgets'].keys())
    if budget_list:
        _best_fn = min if lower_is_better else max
        _default = 999 if lower_is_better else -999

        # Per-model optimal budget
        best_model_budgets = {}
        for mname in model_factories:
            best_model_budgets[mname] = _best_fn(
                budget_list,
                key=lambda b: results['budgets'][b].get('model_scores', {}).get(mname, _default))

        # Highest SAC = best distance-activity correlation
        best_sac_budget = max(budget_list,
                              key=lambda b: results['budgets'][b].get('sac_rho', -999))

        results['optimal'] = {
            'best_model_budgets': best_model_budgets,
            'best_sac_budget': best_sac_budget,
            # Backwards compat
            'best_rf_budget': best_model_budgets.get('rf'),
            'best_knn_budget': best_model_budgets.get('knn'),
        }
        best_rf_budget = best_model_budgets.get('rf', budget_list[0])
        budget_strs = ', '.join(f"{mn}={b}" for mn, b in best_model_budgets.items())
        print(f"\n  Optimal budgets: {budget_strs}, SAC={best_sac_budget}")

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
    parser.add_argument('--budgets', type=int, nargs='+', default=[10, 25, 50, 100, 250, 500, 1000, 2000],
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
