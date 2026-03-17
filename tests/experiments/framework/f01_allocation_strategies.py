#!/usr/bin/env python3
"""F01: Allocation strategy comparison — which allocation method wins?

Paper: Framework
Question: Do smarter budget allocations (softmax, power-law, thresholded,
    greedy variants) outperform simple greedy or linear-proportional?

Tests all 17 NAMED_ALLOCATIONS × budgets × models with domain-aware metrics.
Logs allocation dicts (which reps got how many features) and wall time.

Usage:
    python f01_allocation_strategies.py --dataset logd --reps all --mode full
    python f01_allocation_strategies.py --dataset flab_thermo_tresanco --reps all --mode full
    python f01_allocation_strategies.py --dataset tcga_luad_vs_lusc --reps all --mode full
    python f01_allocation_strategies.py --dataset buchwald_hartwig --reps all --mode full

    # Custom importance method (default: treeshap):
    python f01_allocation_strategies.py --dataset logd --importance-method lightgbm_gain
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split,
    create_hybrid, apply_feature_selection,
    parse_rep_args, FRAMEWORK_TASKS,
)
from kirby.hybrid import NAMED_ALLOCATIONS
from kirby.find_best_hybrid import _score_to_comparable


# All 17 named allocations from hybrid.py
STRATEGY_NAMES = sorted(NAMED_ALLOCATIONS.keys())

BUDGETS = [50, 100, 200, 500]


def run(data, config, output_dir, dataset_name, importance_method='treeshap',
        **kwargs):
    """Run allocation strategy comparison across all named allocations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F01: ALLOCATION STRATEGY COMPARISON")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(data['labels'])
    task_type = data.get('task_type', 'regression')
    models = get_model_factories(config['models'], task_type=task_type,
                                 n_samples=n_samples)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')

    n_strategies = len(STRATEGY_NAMES)
    total = n_strategies * len(BUDGETS) * len(models)
    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  {n_strategies} strategies x {len(BUDGETS)} budgets x {len(models)} models "
          f"= {total} evaluations")
    print(f"  Importance: {importance_method}, folds: {n_folds}\n", flush=True)

    sample_X = next(iter(reps.values()))
    configs = []
    idx = 0

    for strat_name in STRATEGY_NAMES:
        for budget in BUDGETS:
            for model_name, model_factory in models.items():
                idx += 1
                fold_scores = []
                last_n_features = 0
                last_allocation = {}
                t0 = time.time()

                for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                                   labels, mode=mode):
                    train_reps = {k: v[train_idx] for k, v in reps.items()}
                    val_reps = {k: v[val_idx] for k, v in reps.items()}
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    try:
                        X_train_h, feature_info = create_hybrid(
                            train_reps, y_train,
                            allocation_method=strat_name,
                            budget=budget,
                            importance_method=importance_method,
                        )
                        X_val_h = apply_feature_selection(val_reps, feature_info)
                    except Exception as e:
                        if idx <= 5:
                            print(f"  [{idx}/{total}] {strat_name}/{model_name}/b={budget}: "
                                  f"HYBRID FAILED - {e}", flush=True)
                        continue

                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train_h)
                    X_val_s = scaler.transform(X_val_h)
                    last_n_features = X_train_h.shape[1]
                    last_allocation = feature_info.get('allocation', {})

                    try:
                        model = model_factory()
                        model.fit(X_train_s, y_train)
                        fold_scores.append(metric_fn(y_val, model.predict(X_val_s)))
                    except Exception as e:
                        if idx <= 5:
                            print(f"  [{idx}/{total}] {strat_name}/{model_name}/b={budget}: "
                                  f"MODEL FAILED - {e}", flush=True)
                        continue

                wall_time = time.time() - t0

                if not fold_scores:
                    score_mean, score_std = float('nan'), float('nan')
                    n_features = 0
                else:
                    score_mean = float(np.mean(fold_scores))
                    score_std = float(np.std(fold_scores))
                    n_features = last_n_features

                entry = {
                    'strategy': strat_name,
                    'model': model_name,
                    'budget': budget,
                    'importance': importance_method,
                    'score_mean': score_mean,
                    'score_std': score_std,
                    'fold_scores': fold_scores,
                    'n_features': n_features,
                    'allocation_dict': {k: int(v) for k, v in last_allocation.items()},
                    'wall_time': wall_time,
                }
                configs.append(entry)

                if idx % 10 == 0 or idx <= 5:
                    print(f"  [{idx}/{total}] {strat_name}/{model_name}/b={budget}: "
                          f"{metric_name}={score_mean:.4f}+/-{score_std:.3f} "
                          f"dim={n_features} t={wall_time:.1f}s", flush=True)

    # --- Analysis ---
    valid = [c for c in configs if not np.isnan(c['score_mean'])]

    # Best per model (across all strategies/budgets)
    best_by_model = {}
    for model_name in models:
        model_cfgs = [c for c in valid if c['model'] == model_name]
        if model_cfgs:
            if lower_is_better:
                best = min(model_cfgs, key=lambda c: c['score_mean'])
            else:
                best = max(model_cfgs, key=lambda c: c['score_mean'])
            best_by_model[model_name] = best

    # Strategy ranking: average best-per-model score across models
    heatmap = {}
    for strat_name in STRATEGY_NAMES:
        heatmap[strat_name] = {}
        strat_cfgs = [c for c in valid if c['strategy'] == strat_name]
        for model_name in models:
            model_strat = [c for c in strat_cfgs if c['model'] == model_name]
            if model_strat:
                comparables = [_score_to_comparable(c['score_mean'], lower_is_better)
                               for c in model_strat]
                best_idx = int(np.argmax(comparables))
                heatmap[strat_name][model_name] = model_strat[best_idx]['score_mean']

    strategy_avg = {}
    for strat_name in heatmap:
        scores = list(heatmap[strat_name].values())
        if scores:
            comparables = [_score_to_comparable(s, lower_is_better) for s in scores]
            strategy_avg[strat_name] = float(np.mean(comparables))
        else:
            strategy_avg[strat_name] = float('nan')

    # --- Print summary ---
    print(f"\n  BEST BY MODEL ({metric_name}):")
    for model_name, best in best_by_model.items():
        print(f"  >> {model_name}: {best['strategy']}/b={best['budget']} "
              f"-> {metric_name}={best['score_mean']:.4f}")

    print(f"\n  STRATEGY RANKING (avg comparable score across models):")
    for strat, avg in sorted(strategy_avg.items(), key=lambda x: -x[1]):
        print(f"    {strat:<25} {avg:.4f}")

    # Heatmap table
    print(f"\n  HEATMAP (strategy x model -> best {metric_name}):")
    model_names = list(models.keys())
    header = f"  {'Strategy':<25}" + "".join(f"{m:>12}" for m in model_names)
    print(header)
    print("  " + "-" * (25 + 12 * len(model_names)))
    for strat_name in STRATEGY_NAMES:
        row = f"  {strat_name:<25}"
        for model_name in model_names:
            val = heatmap.get(strat_name, {}).get(model_name)
            row += f"{val:>12.4f}" if val is not None else f"{'--':>12}"
        print(row)

    # Build structured output
    summary = {
        'best_by_model': best_by_model,
        'strategy_ranking': strategy_avg,
        'heatmap': heatmap,
    }

    return make_structured_output(
        experiment_id='f01',
        dataset=dataset_name,
        data=data,
        config={
            'n_folds': n_folds,
            'models': list(models.keys()),
            'budgets': BUDGETS,
            'strategies': STRATEGY_NAMES,
            'importance_method': importance_method,
        },
        results={'configs': configs, 'summary': summary},
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser('F01: Allocation Strategy Comparison',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--importance-method', type=str, default='treeshap',
                        help='Importance method for allocation (default: treeshap)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {config['models']}")
    print(f"Importance: {args.importance_method}")

    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  importance_method=args.importance_method)
    save_experiment_results(results, args.output, 'f01_allocation_strategies',
                            args.dataset)
