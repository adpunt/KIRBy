#!/usr/bin/env python3
"""C19: Importance Weights as Training Signal.

Paper: Complexity
Core question: How should hybrid importance scores enter the training process?
    Tests 5 weighting strategies across models, on both raw concat and hybrid
    features. The key question: does importance weighting help MORE on raw
    features (compensating for noise) or on the hybrid (further refining)?

Conditions:
  0. unweighted (baseline)
  1. feature_weighted: X * sqrt(importance) — rescale features
  2. sample_weighted: reliability scores as sample_weight in fit
  3. combined: feature_weighted + sample_weighted
  4. importance_pruned: zero out features with importance < median

Usage:
    python c19_importance_weighted_training.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories,
    compute_feature_importance,
)
from experiments._shared import get_fitted_hybrid, cap_dimensions


# =========================================================================
# IMPORTANCE WEIGHT TRANSFORMS
# =========================================================================

def _compute_importance_weights(X, y, method='lightgbm_gain'):
    """Compute normalised sqrt-importance weights for features."""
    imp_dict = compute_feature_importance({'_features': X}, y, method=method)
    imp = imp_dict['_features']
    w = np.sqrt(np.maximum(imp, 0))
    w = w / (w.sum() + 1e-10)
    return w


def _compute_reliability_scores(X, y, importance_weights, k=10):
    """Per-sample reliability based on neighbor label agreement in importance-weighted space.

    reliability_i = 1 - |y_i - mean(y_neighbors)| / range(y), clamped to [0.1, 1.0].
    """
    X_weighted = X * importance_weights
    k_actual = min(k, len(y) - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1)
    nn.fit(X_weighted)
    _, indices = nn.kneighbors(X_weighted)

    y_range = float(np.max(y) - np.min(y))
    if y_range < 1e-10:
        return np.ones(len(y))

    reliability = np.zeros(len(y))
    for i in range(len(y)):
        nbr_idx = indices[i, 1:]  # exclude self
        nbr_mean = np.mean(y[nbr_idx])
        reliability[i] = 1.0 - abs(y[i] - nbr_mean) / y_range

    return np.clip(reliability, 0.1, 1.0)


# =========================================================================
# MODELS THAT SUPPORT SAMPLE WEIGHT
# =========================================================================

WEIGHTED_MODELS = {'rf', 'lightgbm', 'ridge', 'xgboost'}


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """C19: Importance-weighted training."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C19: IMPORTANCE-WEIGHTED TRAINING")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(labels)

    model_names = [m for m in config['models'] if m in WEIGHTED_MODELS]
    if not model_names:
        model_names = ['rf', 'ridge']
    models = get_model_factories(model_names, task_type=data.get('task_type', 'regression'),
                                 n_samples=n_samples)

    budget = kwargs.get('budget', 100)

    conditions = ['unweighted', 'feature_weighted', 'sample_weighted',
                  'combined', 'importance_pruned']

    # Two input spaces: raw concat and hybrid
    input_spaces = {}

    # 1. Raw concat
    X_concat = np.hstack([reps[n] for n in sorted(reps.keys())])
    input_spaces['concat'] = X_concat

    # 2. Hybrid
    augmentations = data.get('augmentations')
    try:
        X_hybrid, fi, _search = get_fitted_hybrid(
            reps, labels, groups, downstream_model='rf',
            budget=budget,
            aug_condition='capped', augmentations=augmentations,
            speed=config.get('search_speed', 'fast'),
            precomputed=data.get('precomputed_searches'),
        )
        input_spaces['hybrid'] = X_hybrid
    except Exception as e:
        print(f"  Hybrid creation failed: {e}")

    print(f"  Input spaces: {list(input_spaces.keys())}")
    print(f"  Models: {list(models.keys())}")
    print(f"  Conditions: {conditions}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'budget': budget,
        'conditions': conditions,
        'per_space': {},
    }

    for space_name, X_full in input_spaces.items():
        print(f"\n  === {space_name} (dim={X_full.shape[1]}) ===")
        space_results = {}

        for model_name, model_factory in models.items():
            print(f"\n    {model_name}:")
            model_results = {}

            for cond in conditions:
                fold_scores = []
                sample_X = X_full

                for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                                   labels, mode=mode):
                    X_tr = X_full[train_idx]
                    X_va = X_full[val_idx]
                    y_train = labels[train_idx]
                    y_val = labels[val_idx]

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_va_s = scaler.transform(X_va)
                    X_tr_s, X_va_s = cap_dimensions(X_tr_s, X_va_s)

                    # Compute importance on training data
                    imp_weights = _compute_importance_weights(X_tr_s, y_train)

                    # Apply condition transforms
                    fit_kwargs = {}

                    if cond == 'feature_weighted':
                        X_tr_s = X_tr_s * imp_weights
                        X_va_s = X_va_s * imp_weights

                    elif cond == 'sample_weighted':
                        reliability = _compute_reliability_scores(
                            X_tr_s, y_train, imp_weights, k=10)
                        fit_kwargs['sample_weight'] = reliability

                    elif cond == 'combined':
                        X_tr_s = X_tr_s * imp_weights
                        X_va_s = X_va_s * imp_weights
                        reliability = _compute_reliability_scores(
                            X_tr_s, y_train, imp_weights, k=10)
                        fit_kwargs['sample_weight'] = reliability

                    elif cond == 'importance_pruned':
                        mask = imp_weights >= np.median(imp_weights)
                        if mask.sum() < 2:
                            mask[:2] = True
                        X_tr_s = X_tr_s[:, mask]
                        X_va_s = X_va_s[:, mask]

                    try:
                        model = model_factory()
                        model.fit(X_tr_s, y_train, **fit_kwargs)
                        fold_scores.append(metric_fn(y_val, model.predict(X_va_s)))
                    except Exception:
                        # Some models don't support sample_weight
                        if 'sample_weight' in fit_kwargs:
                            try:
                                model = model_factory()
                                model.fit(X_tr_s, y_train)
                                fold_scores.append(metric_fn(y_val, model.predict(X_va_s)))
                            except Exception:
                                continue
                        else:
                            continue

                score = float(np.mean(fold_scores)) if fold_scores else float('nan')
                model_results[cond] = score
                print(f"      {cond}: {score:.4f}")

            # Compute deltas vs unweighted
            baseline = model_results.get('unweighted', float('nan'))
            deltas = {}
            for cond in conditions:
                if cond == 'unweighted':
                    continue
                s = model_results.get(cond, float('nan'))
                if not np.isnan(s) and not np.isnan(baseline):
                    deltas[cond] = float(s - baseline)

            space_results[model_name] = {
                'scores': model_results,
                'deltas': deltas,
                'best_condition': max(model_results, key=lambda k: model_results[k]
                                      if not np.isnan(model_results[k]) else -1e10),
            }

        results['per_space'][space_name] = convert_numpy(space_results)

    # Summary: which condition wins most often?
    condition_wins = {c: 0 for c in conditions}
    for space_name, space_res in results['per_space'].items():
        for model_name, model_res in space_res.items():
            best = model_res.get('best_condition', 'unweighted')
            condition_wins[best] = condition_wins.get(best, 0) + 1

    results['summary'] = {
        'condition_wins': condition_wins,
        'best_overall': max(condition_wins, key=condition_wins.get),
    }
    print(f"\n  Condition wins: {condition_wins}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser(
        'C19: Importance-Weighted Training')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output,
                            'c19_importance_weighted_training', args.dataset)
