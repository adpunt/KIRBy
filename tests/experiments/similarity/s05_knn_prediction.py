#!/usr/bin/env python3
"""S05: kNN Prediction with Hybrid Distance.

Paper: Similarity
Conclusion: Hybrid-weighted kNN outperforms standard kNN when the hybrid
            is optimized with a distance-aware evaluator at appropriate budgets.
Dependencies: None (runs its own distance-aware hybrid search).

CV evaluation with per-budget hybrid conditions: hybrid_unweighted, hybrid_weighted.
Per-rep kNN baselines (the real baselines). Budget sweep over DISTANCE_BUDGETS
to show how dimensionality affects kNN performance.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_fitted_hybrid, get_best_for_downstream,
    flatten_weights, make_cv_splitter,
    get_domain_metric, build_fold_hybrid, extract_top3_base_reps,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)

from kirby.hybrid import apply_feature_selection, compute_feature_importance


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Downstream task validation: kNN with distance-aware hybrid selection
    across multiple budgets."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S05: kNN PREDICTION WITH HYBRID DISTANCE")
    print("=" * 70)

    k_values = kwargs.get('k_values', [5, 10])
    budgets = kwargs.get('budgets', DISTANCE_BUDGETS)
    n_folds = config['n_splits']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    search_speed = config.get('search_speed', 'full')

    task_type = data.get('task_type', 'regression')
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    results = {'per_budget': {}, 'per_rep_baselines': {}}

    # ── Distance-aware hybrid search (on-demand, disk-cached) ──
    # Uses kNN + tree evaluators competing, with lower budgets.
    print(f"  Running distance-aware hybrid search "
          f"(evaluators={DISTANCE_EVALUATORS}, budgets={budgets})...")
    try:
        # We only need the SearchResult, not the full-data hybrid matrix.
        # Per-budget blueprints are used with build_fold_hybrid in the CV loop.
        _X_unused, _fi_unused, search = get_fitted_hybrid(
            reps, labels, groups,
            aug_condition='uncapped', augmentations=augmentations,
            speed=search_speed,
            precomputed=data.get('precomputed_searches'),
            internal_evaluators=DISTANCE_EVALUATORS,
            budgets=budgets,
            metric=metric_name, task_type=task_type,
            domain=data.get('domain', 'molecular'),
            downstream_model='knn',
            dataset_name=dataset_name, tier='general_distance')

        # Report which evaluator won
        best_knn = get_best_for_downstream(search, 'knn')
        if best_knn and 'blueprint' in best_knn:
            bp = best_knn['blueprint']
            print(f"  Best kNN config: alloc={bp.allocation_method}, "
                  f"imp={bp.importance_method}, ev={bp.internal_evaluator}, "
                  f"budget={bp.budget}, score={best_knn.get('score_mean', 0):.4f}")
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            results['top3_base_reps'] = [(n, s) for n, s in top3]
    except Exception as e:
        print(f"  Distance-aware search failed: {e}")
        search = None

    # ── Per-budget hybrid kNN evaluation ──
    splitter, use_groups = make_cv_splitter(n_folds, groups)
    sample_X = next(iter(reps.values()))

    def _make_knn(k_val, n_train):
        nn = min(k_val, n_train - 1)
        if is_clf:
            return KNeighborsClassifier(n_neighbors=nn)
        return KNeighborsRegressor(n_neighbors=nn)

    if search is not None:
        for budget in budgets:
            best_at_budget = get_best_for_downstream(search, 'knn', budget=budget)
            if best_at_budget is None or 'blueprint' not in best_at_budget:
                print(f"\n  Budget={budget}: no config found, skipping")
                continue

            blueprint = best_at_budget['blueprint']
            print(f"\n  Budget={budget} (ev={blueprint.internal_evaluator}, "
                  f"alloc={blueprint.allocation_method}):")

            budget_results = {}

            # Create splitter ONCE per budget so all conditions share
            # the same fold splits — required for fair comparison.
            splitter_b, use_groups_b = make_cv_splitter(n_folds, groups)
            folds_b = list(
                splitter_b.split(sample_X, labels, groups=groups)
                if use_groups_b
                else splitter_b.split(sample_X, labels)
            )

            for condition in ['hybrid_unweighted', 'hybrid_weighted']:
                for k in k_values:
                    fold_scores = []

                    for train_idx, val_idx in folds_b:
                        y_train, y_val = labels[train_idx], labels[val_idx]

                        # Build per-fold hybrid from blueprint
                        train_reps = {name: rep[train_idx] for name, rep in reps.items()}
                        val_reps = {name: rep[val_idx] for name, rep in reps.items()}
                        aug_tr = ({k2: v[train_idx] for k2, v in augmentations.items()}
                                  if augmentations else None)
                        aug_va = ({k2: v[val_idx] for k2, v in augmentations.items()}
                                  if augmentations else None)
                        try:
                            X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                                train_reps, val_reps, y_train, blueprint,
                                augmentations_train=aug_tr,
                                augmentations_val=aug_va)
                        except Exception:
                            continue

                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_tr_h)
                        X_va = scaler.transform(X_va_h)

                        # Importance weighting for weighted condition
                        if condition == 'hybrid_weighted':
                            try:
                                imp = compute_feature_importance(
                                    train_reps, y_train,
                                    method='lightgbm_gain')
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
                                X_tr = X_tr * w
                                X_va = X_va * w
                            except Exception:
                                continue

                        model = _make_knn(k, len(y_train))
                        model.fit(X_tr, y_train)
                        y_pred = model.predict(X_va)
                        fold_scores.append(metric_fn(y_val, y_pred))

                    if not fold_scores:
                        continue

                    score_mean = float(np.mean(fold_scores))
                    score_std = float(np.std(fold_scores))
                    key = f"{condition}_k{k}"
                    budget_results[key] = {
                        'score_mean': score_mean,
                        'score_std': score_std,
                        'metric': metric_name,
                        'k': k,
                        'condition': condition,
                        'budget': budget,
                        'evaluator': blueprint.internal_evaluator,
                        'allocation': blueprint.allocation_method,
                    }
                    print(f"    {condition} k={k}: "
                          f"{metric_name}={score_mean:.4f}+/-{score_std:.3f}")

            results['per_budget'][budget] = budget_results

    # ── Per-rep kNN baselines (individual representations) ──
    # Try all k values and pick the best per metric direction.
    print(f"\n  Individual representation kNN baselines (metric={metric_name}):")
    for rep_name, X_rep in reps.items():
        best_k_score = None
        best_k_used = k_values[0]
        best_k_scores_uw = None
        best_k_scores_w = None
        best_k_std = 0.0
        best_k_w_mean = float('nan')

        for k_candidate in k_values:
            splitter2, use_groups2 = make_cv_splitter(n_folds, groups)
            fold_scores_uw = []
            fold_scores_w = []
            for train_idx, val_idx in (
                splitter2.split(sample_X, labels, groups=groups)
                if use_groups2
                else splitter2.split(sample_X, labels)
            ):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_rep[train_idx])
                X_va = scaler.transform(X_rep[val_idx])

                # Unweighted
                model = _make_knn(k_candidate, len(train_idx))
                model.fit(X_tr, labels[train_idx])
                fold_scores_uw.append(metric_fn(labels[val_idx], model.predict(X_va)))

                # Weighted by importance
                try:
                    imp = compute_feature_importance(
                        {rep_name: X_rep[train_idx]}, labels[train_idx],
                        method='lightgbm_gain')
                    w = np.sqrt(np.maximum(imp[rep_name], 0))
                    X_tr_w = X_tr * w
                    X_va_w = X_va * w
                    model_w = _make_knn(k_candidate, len(train_idx))
                    model_w.fit(X_tr_w, labels[train_idx])
                    fold_scores_w.append(metric_fn(labels[val_idx],
                                                   model_w.predict(X_va_w)))
                except Exception:
                    fold_scores_w.append(float('nan'))

            if not fold_scores_uw:
                continue
            sc_mean = float(np.mean(fold_scores_uw))
            if best_k_score is None or (sc_mean < best_k_score if lower_is_better else sc_mean > best_k_score):
                best_k_score = sc_mean
                best_k_used = k_candidate
                best_k_std = float(np.std(fold_scores_uw))
                best_k_scores_uw = fold_scores_uw
                best_k_w_mean = float(np.nanmean(fold_scores_w)) if fold_scores_w else float('nan')

        sc_mean = best_k_score if best_k_score is not None else float('nan')
        sc_std = best_k_std
        sc_w_mean = best_k_w_mean
        results['per_rep_baselines'][rep_name] = {
            'score_mean': sc_mean, 'score_std': sc_std,
            'score_weighted_mean': sc_w_mean,
            'metric': metric_name,
            'k': best_k_used, 'dim': X_rep.shape[1],
        }
        delta = sc_w_mean - sc_mean
        print(f"    {rep_name} (dim={X_rep.shape[1]}, k={best_k_used}): "
              f"kNN {metric_name}={sc_mean:.4f}+/-{sc_std:.3f}, "
              f"weighted {metric_name}={sc_w_mean:.4f} "
              f"(delta={delta:+.4f})")

    # ── Summary ──
    print(f"\n  Summary (best per budget):")
    for b in sorted(results['per_budget'].keys()):
        budget_res = results['per_budget'][b]
        if budget_res:
            best_key = (min if lower_is_better else max)(
                budget_res, key=lambda k: budget_res[k]['score_mean'])
            best = budget_res[best_key]
            print(f"    budget={b}: best={best['condition']} k={best['k']}, "
                  f"{metric_name}={best['score_mean']:.4f} "
                  f"(ev={best.get('evaluator', '?')})")

    if results['per_rep_baselines']:
        best_rep = (min if lower_is_better else max)(
            results['per_rep_baselines'],
            key=lambda r: results['per_rep_baselines'][r]['score_mean'])
        best_rep_score = results['per_rep_baselines'][best_rep]['score_mean']
        print(f"\n  Best individual rep baseline: {best_rep} "
              f"(kNN {metric_name}={best_rep_score:.4f})")

        # Compare best hybrid budget to best individual rep
        all_hybrid_scores = []
        for b_res in results['per_budget'].values():
            for v in b_res.values():
                all_hybrid_scores.append(v)
        if all_hybrid_scores:
            best_hybrid = (min if lower_is_better else max)(
                all_hybrid_scores, key=lambda v: v['score_mean'])
            delta = best_hybrid['score_mean'] - best_rep_score
            print(f"  Best hybrid kNN: budget={best_hybrid['budget']}, "
                  f"{best_hybrid['condition']} k={best_hybrid['k']}, "
                  f"{metric_name}={best_hybrid['score_mean']:.4f} "
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
