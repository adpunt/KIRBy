#!/usr/bin/env python3
"""C11: Allocation strategy comparison — does smarter budget allocation help?

Paper: Complexity
Conclusion: Whether softmax/power-law/thresholding outperforms linear-proportional.
Dependencies: None (standalone).

Tests 6 weighting schemes across models and budgets using create_hybrid directly
for fine-grained control over weighting_scheme / weighting_params.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, INTERMEDIARY, FULL, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split,
    create_hybrid, apply_feature_selection,
)


# Strategy grid: (label, allocation_method, weighting_scheme, weighting_params)
STRATEGY_GRID = [
    # Greedy baseline (no weighting scheme)
    ('greedy', 'greedy', None, None),
    # Linear (current default)
    ('linear', 'performance_weighted', 'linear', None),
    # Softmax with varying temperatures
    ('softmax_T0.1', 'performance_weighted', 'softmax', {'temperature': 0.1}),
    ('softmax_T0.25', 'performance_weighted', 'softmax', {'temperature': 0.25}),
    ('softmax_T0.5', 'performance_weighted', 'softmax', {'temperature': 0.5}),
    ('softmax_T1.0', 'performance_weighted', 'softmax', {'temperature': 1.0}),
    ('softmax_T2.0', 'performance_weighted', 'softmax', {'temperature': 2.0}),
    ('softmax_T5.0', 'performance_weighted', 'softmax', {'temperature': 5.0}),
    ('softmax_T10.0', 'performance_weighted', 'softmax', {'temperature': 10.0}),
    # Power law
    ('power_0.5', 'performance_weighted', 'power_law', {'power': 0.5}),
    ('power_1.5', 'performance_weighted', 'power_law', {'power': 1.5}),
    ('power_2.0', 'performance_weighted', 'power_law', {'power': 2.0}),
    ('power_3.0', 'performance_weighted', 'power_law', {'power': 3.0}),
    ('power_5.0', 'performance_weighted', 'power_law', {'power': 5.0}),
    # Thresholded
    ('thresh_0.3', 'performance_weighted', 'thresholded', {'alpha': 0.3}),
    ('thresh_0.5', 'performance_weighted', 'thresholded', {'alpha': 0.5}),
    ('thresh_0.7', 'performance_weighted', 'thresholded', {'alpha': 0.7}),
    # Top-k
    ('top_2', 'performance_weighted', 'top_k', {'k': 2}),
    ('top_3', 'performance_weighted', 'top_k', {'k': 3}),
    # Min floor
    ('floor_0.03', 'performance_weighted', 'min_floor', {'floor': 0.03}),
    ('floor_0.05', 'performance_weighted', 'min_floor', {'floor': 0.05}),
    ('floor_0.10', 'performance_weighted', 'min_floor', {'floor': 0.10}),
]


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run allocation strategy comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C11: ALLOCATION STRATEGY COMPARISON")
    print("=" * 70)

    models = get_model_factories(config['models'])
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    importance_method = 'lightgbm_gain'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    budgets = [100, 500]

    n_strategies = len(STRATEGY_GRID)
    total = n_strategies * len(budgets) * len(models)
    print(f"  {n_strategies} strategies x {len(budgets)} budgets x {len(models)} models "
          f"= {total} evaluations")
    print(f"  importance: {importance_method}, folds: {n_folds}\n")

    sample_X = next(iter(reps.values()))
    configs = []
    idx = 0

    for strat_label, alloc_method, w_scheme, w_params in STRATEGY_GRID:
        for budget in budgets:
            for model_name, model_factory in models.items():
                idx += 1
                fold_scores = []
                last_n_features = 0

                for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                                   labels, mode=mode):
                    train_reps = {k: v[train_idx] for k, v in reps.items()}
                    val_reps = {k: v[val_idx] for k, v in reps.items()}
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    try:
                        create_kwargs = dict(
                            allocation_method=alloc_method,
                            budget=budget,
                            importance_method=importance_method,
                        )
                        if w_scheme is not None:
                            create_kwargs['weighting_scheme'] = w_scheme
                        if w_params is not None:
                            create_kwargs['weighting_params'] = w_params

                        X_train_h, feature_info = create_hybrid(
                            train_reps, y_train, **create_kwargs,
                        )
                        X_val_h = apply_feature_selection(val_reps, feature_info)
                    except Exception as e:
                        if idx <= 5:
                            print(f"  [{idx}/{total}] {strat_label}/{model_name}/b={budget}: "
                                  f"HYBRID FAILED - {e}")
                        continue

                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train_h)
                    X_val_s = scaler.transform(X_val_h)
                    last_n_features = X_train_h.shape[1]

                    model = model_factory()
                    model.fit(X_train_s, y_train)
                    fold_scores.append(r2_score(y_val, model.predict(X_val_s)))

                if not fold_scores:
                    r2_mean, r2_std = float('nan'), float('nan')
                    n_features = 0
                else:
                    r2_mean = float(np.mean(fold_scores))
                    r2_std = float(np.std(fold_scores))
                    n_features = last_n_features

                entry = {
                    'strategy': strat_label,
                    'allocation': alloc_method,
                    'weighting_scheme': w_scheme or 'n/a',
                    'weighting_params': w_params,
                    'model': model_name,
                    'budget': budget,
                    'importance': importance_method,
                    'r2_mean': r2_mean,
                    'r2_std': r2_std,
                    'fold_scores': fold_scores,
                    'n_features': n_features,
                }
                configs.append(entry)

                print(f"  [{idx}/{total}] {strat_label}/{model_name}/b={budget}: "
                      f"R2={r2_mean:.4f}+/-{r2_std:.3f} dim={n_features}")

    # Best by model
    best_by_model = {}
    for model_name in models:
        model_configs = [c for c in configs if c['model'] == model_name
                         and not np.isnan(c['r2_mean'])]
        if model_configs:
            best = max(model_configs, key=lambda c: c['r2_mean'])
            best_by_model[model_name] = best

    # Heatmap: strategy x model -> best R2 (across budgets)
    heatmap = {}
    for strat_label, _, _, _ in STRATEGY_GRID:
        heatmap[strat_label] = {}
        strat_configs = [c for c in configs if c['strategy'] == strat_label
                         and not np.isnan(c['r2_mean'])]
        for model_name in models:
            model_strat = [c for c in strat_configs if c['model'] == model_name]
            if model_strat:
                heatmap[strat_label][model_name] = max(c['r2_mean'] for c in model_strat)

    # Strategy ranking (avg best R2 across models)
    strategy_avg = {}
    for strat_label in heatmap:
        scores = list(heatmap[strat_label].values())
        strategy_avg[strat_label] = float(np.mean(scores)) if scores else float('nan')

    # Print summary
    print("\n  BEST BY MODEL:")
    for model_name, best in best_by_model.items():
        print(f"  >> {model_name}: {best['strategy']}/b={best['budget']} "
              f"-> R2={best['r2_mean']:.4f}")

    print("\n  STRATEGY RANKING (avg best R2 across models):")
    for strat, avg in sorted(strategy_avg.items(), key=lambda x: -x[1]):
        print(f"    {strat:<20} {avg:.4f}")

    # Heatmap table
    print("\n  HEATMAP (strategy x model -> best R2):")
    model_names = list(models.keys())
    header = f"  {'Strategy':<20}" + "".join(f"{m:>12}" for m in model_names)
    print(header)
    print("  " + "-" * (20 + 12 * len(model_names)))
    for strat_label, _, _, _ in STRATEGY_GRID:
        row = f"  {strat_label:<20}"
        for model_name in model_names:
            val = heatmap.get(strat_label, {}).get(model_name)
            row += f"{val:>12.4f}" if val is not None else f"{'--':>12}"
        print(row)

    return {
        'configs': configs,
        'best_by_model': best_by_model,
        'heatmap': heatmap,
        'strategy_ranking': strategy_avg,
    }


if __name__ == '__main__':
    parser = create_experiment_parser('C11: Allocation Strategy Comparison',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {config['models']}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c11_allocation_strategies', args.dataset)
