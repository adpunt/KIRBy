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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results,
    get_fitted_hybrid, flatten_weights, make_cv_splitter,
    get_domain_metric, get_model_factories, build_fold_hybrid,
)
from experiments.similarity.similarity_helpers import identify_cliff_compounds

from kirby.hybrid import apply_feature_selection, compute_feature_importance
from kirby.hybrid import create_hybrid


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
    conditions = kwargs.get('conditions', [
        'concat_unweighted', 'concat_weighted',
        'hybrid_unweighted', 'hybrid_weighted',
    ])
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'conditions': {}, 'cliff_analysis': {}}

    augmentations = data.get('augmentations')

    # Pre-compute concat
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    # Get hybrid via find_best_hybrid -- extract blueprint for per-fold CV
    X_hybrid = None
    weights_flat = None
    hybrid_blueprint = None
    if any(c.startswith('hybrid') for c in conditions):
        try:
            budget = kwargs.get('budget', 100)
            augmentations_search = data.get('augmentations')
            X_hybrid, fi, search = get_fitted_hybrid(
                reps, labels, groups, budget=budget,
                aug_condition='uncapped', augmentations=augmentations_search,
                speed=config.get('search_speed', 'fast'),
                precomputed=data.get('precomputed_searches'))
            fitted = search.fitted_result
            weights_flat = flatten_weights(fitted) if fitted else np.ones(X_hybrid.shape[1])
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
    splitter, use_groups = make_cv_splitter(n_folds, groups)
    sample_X = next(iter(reps.values()))

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data.get('task_type', 'regression')
    is_clf = task_type == 'classification'

    def _make_knn(k_val, n_train):
        nn = min(k_val, n_train - 1)
        if is_clf:
            return KNeighborsClassifier(n_neighbors=nn)
        return KNeighborsRegressor(n_neighbors=nn)

    for condition in conditions:
        print(f"\n  Condition: {condition}")

        for k in k_values:
            fold_scores = []
            fold_cliff_scores = []
            fold_noncliff_scores = []
            fold_errors = []

            for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
                y_train, y_val = labels[train_idx], labels[val_idx]
                scaler = StandardScaler()

                if condition == 'concat_unweighted':
                    X_tr = scaler.fit_transform(X_concat[train_idx])
                    X_va = scaler.transform(X_concat[val_idx])
                    model = _make_knn(k, len(y_train))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'concat_weighted':
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
                    model = _make_knn(k, len(y_train))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'hybrid_unweighted' and hybrid_blueprint is not None:
                    train_reps_h = {name: rep[train_idx] for name, rep in reps.items()}
                    val_reps_h = {name: rep[val_idx] for name, rep in reps.items()}
                    aug_tr = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                    aug_va = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                    try:
                        X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                            train_reps_h, val_reps_h, y_train, hybrid_blueprint,
                            augmentations_train=aug_tr, augmentations_val=aug_va)
                    except Exception:
                        continue

                    X_tr = scaler.fit_transform(X_tr_h)
                    X_va = scaler.transform(X_va_h)
                    model = _make_knn(k, len(y_train))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'hybrid_weighted' and hybrid_blueprint is not None:
                    train_reps_h = {name: rep[train_idx] for name, rep in reps.items()}
                    val_reps_h = {name: rep[val_idx] for name, rep in reps.items()}
                    aug_tr = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                    aug_va = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                    try:
                        X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                            train_reps_h, val_reps_h, y_train, hybrid_blueprint,
                            augmentations_train=aug_tr, augmentations_val=aug_va)

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

                    X_tr = scaler.fit_transform(X_tr_h) * w
                    X_va = scaler.transform(X_va_h) * w
                    model = _make_knn(k, len(y_train))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'rf_baseline':
                    X_tr = scaler.fit_transform(X_concat[train_idx])
                    X_va = scaler.transform(X_concat[val_idx])
                    rf_factory = get_model_factories(['rf'], task_type=task_type,
                                                      n_samples=len(labels))['rf']
                    model = rf_factory()
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                else:
                    continue

                score = metric_fn(y_val, y_pred)
                fold_scores.append(score)

                # Stratified: cliff vs non-cliff
                cliff_val = np.array([i for i, vi in enumerate(val_idx)
                                      if vi in cliff_indices])
                noncliff_val = np.array([i for i, vi in enumerate(val_idx)
                                         if vi not in cliff_indices])

                if len(cliff_val) > 1:
                    fold_cliff_scores.append(metric_fn(y_val[cliff_val],
                                                       y_pred[cliff_val]))
                if len(noncliff_val) > 1:
                    fold_noncliff_scores.append(metric_fn(y_val[noncliff_val],
                                                          y_pred[noncliff_val]))

                # Per-sample errors for SALI correlation
                fold_errors.extend(list(zip(val_idx.tolist(),
                                           np.abs(y_val - y_pred).tolist())))

            if not fold_scores:
                continue

            score_mean = float(np.mean(fold_scores))
            score_std = float(np.std(fold_scores))
            cliff_score = float(np.mean(fold_cliff_scores)) if fold_cliff_scores else float('nan')
            noncliff_score = float(np.mean(fold_noncliff_scores)) if fold_noncliff_scores else float('nan')

            key = f"{condition}_k{k}"
            results['conditions'][key] = {
                'score_mean': score_mean,
                'score_std': score_std,
                'cliff_score': cliff_score,
                'noncliff_score': noncliff_score,
                'metric': metric_name,
                'k': k,
                'condition': condition,
            }
            print(f"    k={k}: {metric_name}={score_mean:.4f}+/-{score_std:.3f}, "
                  f"cliff={cliff_score:.4f}, noncliff={noncliff_score:.4f}")

    # Per-rep kNN baselines (individual representations -- the real baselines)
    print(f"\n  Individual representation kNN baselines (metric={metric_name}):")
    results['per_rep_baselines'] = {}
    best_k = k_values[len(k_values) // 2]  # use middle k value
    splitter2, use_groups2 = make_cv_splitter(n_folds, groups)
    for rep_name, X_rep in reps.items():
        fold_scores_uw = []
        fold_scores_w = []
        for train_idx, val_idx in (splitter2.split(sample_X, labels, groups=groups) if use_groups2 else splitter2.split(sample_X, labels)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # Unweighted
            model = _make_knn(best_k, len(train_idx))
            model.fit(X_tr, labels[train_idx])
            fold_scores_uw.append(metric_fn(labels[val_idx], model.predict(X_va)))

            # Weighted by importance (computed on train only)
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[train_idx]}, labels[train_idx],
                    method='lightgbm_gain')
                w = np.sqrt(np.maximum(imp[rep_name], 0))
                X_tr_w = X_tr * w
                X_va_w = X_va * w
                model_w = _make_knn(best_k, len(train_idx))
                model_w.fit(X_tr_w, labels[train_idx])
                fold_scores_w.append(metric_fn(labels[val_idx], model_w.predict(X_va_w)))
            except Exception:
                fold_scores_w.append(float('nan'))

        sc_mean = float(np.mean(fold_scores_uw))
        sc_std = float(np.std(fold_scores_uw))
        sc_w_mean = float(np.nanmean(fold_scores_w)) if fold_scores_w else float('nan')
        sc_w_std = float(np.nanstd(fold_scores_w)) if fold_scores_w else float('nan')
        results['per_rep_baselines'][rep_name] = {
            'score_mean': sc_mean, 'score_std': sc_std,
            'score_weighted_mean': sc_w_mean, 'score_weighted_std': sc_w_std,
            'metric': metric_name,
            'k': best_k, 'dim': X_rep.shape[1],
        }
        delta = sc_w_mean - sc_mean
        print(f"    {rep_name} (dim={X_rep.shape[1]}): "
              f"kNN {metric_name}={sc_mean:.4f}+/-{sc_std:.3f}, "
              f"weighted {metric_name}={sc_w_mean:.4f}+/-{sc_w_std:.3f} "
              f"(delta={delta:+.4f})")

    # Summary: best k per condition
    print("\n  Summary (best k per condition):")
    for cond in conditions:
        cond_results = {k: v for k, v in results['conditions'].items()
                       if v['condition'] == cond}
        if cond_results:
            best_key = (min if lower_is_better else max)(
                cond_results, key=lambda k: cond_results[k]['score_mean'])
            best = cond_results[best_key]
            print(f"    {cond}: best k={best['k']}, "
                  f"{metric_name}={best['score_mean']:.4f}")

    # Flag best individual rep
    if results['per_rep_baselines']:
        best_rep = (min if lower_is_better else max)(
            results['per_rep_baselines'],
            key=lambda r: results['per_rep_baselines'][r]['score_mean'])
        best_rep_score = results['per_rep_baselines'][best_rep]['score_mean']
        print(f"\n  Best individual rep baseline: {best_rep} "
              f"(kNN {metric_name}={best_rep_score:.4f})")

        # Compare hybrid_weighted to best individual rep
        hw_results = {k: v for k, v in results['conditions'].items()
                      if v['condition'] == 'hybrid_weighted'}
        if hw_results:
            best_hw = (min if lower_is_better else max)(
                hw_results.values(), key=lambda v: v['score_mean'])
            delta = best_hw['score_mean'] - best_rep_score
            print(f"  Hybrid weighted kNN {metric_name}={best_hw['score_mean']:.4f} "
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
