#!/usr/bin/env python3
"""C13: Greedy step size and patience — how granular should greedy search be?

Paper: Complexity
Conclusion: Whether fine-grained greedy (step=1) outperforms coarse (step=50).
Dependencies: None (standalone).

Tests step_size x patience x budget x model grid using create_hybrid directly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split,
    create_hybrid, apply_feature_selection,
)


STEP_SIZES = [1, 2, 5, 10, 25, 50]
PATIENCE_VALUES = [2, 3, 5, None]  # None = no early stopping
BUDGETS = [50, 100, 200, 500]


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run greedy step size comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C13: GREEDY STEP SIZE & PATIENCE")
    print("=" * 70)

    models = get_model_factories(config['models'])
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    total = len(STEP_SIZES) * len(PATIENCE_VALUES) * len(BUDGETS) * len(models)
    print(f"  {len(STEP_SIZES)} step_sizes x {len(PATIENCE_VALUES)} patience x "
          f"{len(BUDGETS)} budgets x {len(models)} models = {total} evaluations\n")

    sample_X = next(iter(reps.values()))
    configs = []
    idx = 0

    for step_size in STEP_SIZES:
        for patience in PATIENCE_VALUES:
            for budget in BUDGETS:
                for model_name, model_factory in models.items():
                    idx += 1
                    fold_scores = []
                    last_n_features = 0
                    last_feature_info = {}
                    t0 = time.time()

                    for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                                       labels, mode=mode):
                        train_reps = {k: v[train_idx] for k, v in reps.items()}
                        val_reps = {k: v[val_idx] for k, v in reps.items()}
                        y_train, y_val = labels[train_idx], labels[val_idx]

                        try:
                            X_train_h, feature_info = create_hybrid(
                                train_reps, y_train,
                                allocation_method='greedy',
                                budget=budget,
                                step_size=step_size,
                                patience=patience,
                                importance_method='lightgbm_gain',
                            )
                            X_val_h = apply_feature_selection(val_reps, feature_info)
                        except Exception as e:
                            if idx <= 5:
                                print(f"  [{idx}/{total}] step={step_size}/pat={patience}/"
                                      f"b={budget}/{model_name}: FAILED - {e}")
                            continue

                        scaler = StandardScaler()
                        X_train_s = scaler.fit_transform(X_train_h)
                        X_val_s = scaler.transform(X_val_h)
                        last_n_features = X_train_h.shape[1]
                        last_feature_info = feature_info

                        model = model_factory()
                        model.fit(X_train_s, y_train)
                        fold_scores.append(r2_score(y_val, model.predict(X_val_s)))

                    elapsed = time.time() - t0

                    if not fold_scores:
                        r2_mean, r2_std = float('nan'), float('nan')
                        n_features = 0
                        stop_reason = 'no_valid_rep'
                        n_iterations = 0
                        utilization = 0.0
                        alloc_dict = {}
                    else:
                        r2_mean = float(np.mean(fold_scores))
                        r2_std = float(np.std(fold_scores))
                        n_features = last_n_features

                        # Extract greedy diagnostics from last fold
                        history = last_feature_info.get('greedy_history', [])
                        alloc_dict = last_feature_info.get('allocation', {})
                        n_iterations = len(history)
                        utilization = n_features / budget if budget > 0 else 0.0

                        # Infer stop reason
                        total_allocated = sum(alloc_dict.values())
                        if n_iterations == 0:
                            stop_reason = 'no_valid_rep'
                        elif total_allocated >= budget:
                            stop_reason = 'budget'
                        elif patience is not None and n_iterations >= 2:
                            scores = [h['val_r2'] for h in history]
                            best_score = max(scores)
                            best_idx = scores.index(best_score)
                            if n_iterations - best_idx - 1 >= patience:
                                stop_reason = 'patience'
                            else:
                                stop_reason = 'reps_maxed'
                        else:
                            stop_reason = 'reps_maxed'

                    entry = {
                        'step_size': step_size,
                        'patience': patience,
                        'budget': budget,
                        'model': model_name,
                        'r2_mean': r2_mean,
                        'r2_std': r2_std,
                        'fold_scores': fold_scores,
                        'n_features': n_features,
                        'elapsed_s': round(elapsed, 2),
                        'n_iterations': n_iterations,
                        'stop_reason': stop_reason,
                        'utilization': utilization,
                        'allocation': dict(alloc_dict),
                    }
                    configs.append(entry)

                    pat_str = str(patience) if patience is not None else 'off'
                    print(f"  [{idx}/{total}] step={step_size}/pat={pat_str}/"
                          f"b={budget}/{model_name}: "
                          f"R2={r2_mean:.4f} dim={n_features} util={utilization:.0%} "
                          f"stop={stop_reason} ({elapsed:.1f}s)")

    # --- Analysis ---

    # Best by model
    best_by_model = {}
    for model_name in models:
        model_configs = [c for c in configs if c['model'] == model_name
                         and not np.isnan(c['r2_mean'])]
        if model_configs:
            best_by_model[model_name] = max(model_configs, key=lambda c: c['r2_mean'])

    # Best step_size per budget (averaged across models)
    step_budget_avg = {}
    for step_size in STEP_SIZES:
        step_budget_avg[step_size] = {}
        for budget in BUDGETS:
            scores = [c['r2_mean'] for c in configs
                      if c['step_size'] == step_size and c['budget'] == budget
                      and not np.isnan(c['r2_mean'])]
            step_budget_avg[step_size][budget] = float(np.mean(scores)) if scores else float('nan')

    # Speed vs quality: avg elapsed per step_size
    step_speed = {}
    for step_size in STEP_SIZES:
        times = [c['elapsed_s'] for c in configs if c['step_size'] == step_size]
        scores = [c['r2_mean'] for c in configs
                  if c['step_size'] == step_size and not np.isnan(c['r2_mean'])]
        step_speed[step_size] = {
            'avg_time_s': float(np.mean(times)),
            'avg_r2': float(np.mean(scores)) if scores else float('nan'),
        }

    # Patience effect: avg R2 per patience value
    patience_avg = {}
    for pat in PATIENCE_VALUES:
        scores = [c['r2_mean'] for c in configs
                  if c['patience'] == pat and not np.isnan(c['r2_mean'])]
        patience_avg[str(pat)] = float(np.mean(scores)) if scores else float('nan')

    # Print summary
    print("\n  BEST BY MODEL:")
    for model_name, best in best_by_model.items():
        pat_str = str(best['patience']) if best['patience'] is not None else 'off'
        print(f"  >> {model_name}: step={best['step_size']}/pat={pat_str}/"
              f"b={best['budget']} -> R2={best['r2_mean']:.4f} ({best['elapsed_s']:.1f}s)")

    print("\n  STEP SIZE vs BUDGET (avg R2 across models):")
    header = f"  {'Step':>6}" + "".join(f"  b={b:>4}" for b in BUDGETS) + "  avg_time"
    print(header)
    print("  " + "-" * (6 + 8 * len(BUDGETS) + 10))
    for step_size in STEP_SIZES:
        row = f"  {step_size:>6}"
        for budget in BUDGETS:
            val = step_budget_avg[step_size][budget]
            row += f"  {val:>6.4f}" if not np.isnan(val) else f"  {'--':>6}"
        row += f"  {step_speed[step_size]['avg_time_s']:>7.1f}s"
        print(row)

    print("\n  PATIENCE EFFECT (avg R2):")
    for pat, avg in patience_avg.items():
        print(f"    patience={pat:<5}: {avg:.4f}")

    # --- Utilization analysis ---

    # Utilization heatmap: step_size x patience -> avg utilization %
    utilization_heatmap = {}
    print("\n  BUDGET UTILIZATION (step × patience → avg %):")
    pat_header = "".join(f"  pat={str(p) if p is not None else 'off':>5}" for p in PATIENCE_VALUES)
    print(f"  {'step':>6}{pat_header}")
    print("  " + "-" * (6 + 10 * len(PATIENCE_VALUES)))
    for step_size in STEP_SIZES:
        utilization_heatmap[step_size] = {}
        row = f"  {step_size:>6}"
        for pat in PATIENCE_VALUES:
            subset = [c for c in configs if c['step_size'] == step_size
                      and c['patience'] == pat and not np.isnan(c['r2_mean'])]
            avg_util = float(np.mean([c['utilization'] for c in subset])) if subset else float('nan')
            utilization_heatmap[step_size][str(pat)] = avg_util
            row += f"  {avg_util:>8.0%}" if not np.isnan(avg_util) else f"  {'--':>8}"
        print(row)

    # Under-utilization alert
    under_util = [c for c in configs if not np.isnan(c['r2_mean']) and c['utilization'] < 0.3]
    total_valid = len([c for c in configs if not np.isnan(c['r2_mean'])])
    if under_util:
        print(f"\n  ** UNDER-UTILIZATION: {len(under_util)}/{total_valid} configs "
              f"({len(under_util)/total_valid:.0%}) use <30% of budget")

    # Stop reason breakdown by step_size
    stop_reason_counts = {}
    print("\n  STOP REASONS (by step_size):")
    for step_size in STEP_SIZES:
        reasons = Counter(c['stop_reason'] for c in configs if c['step_size'] == step_size)
        stop_reason_counts[step_size] = dict(reasons)
        total_step = sum(reasons.values())
        parts = ", ".join(f"{r}={n}/{total_step}" for r, n in reasons.most_common())
        print(f"    step={step_size:>2}: {parts}")

    # Stop reason breakdown by patience
    print("\n  STOP REASONS (by patience):")
    for pat in PATIENCE_VALUES:
        reasons = Counter(c['stop_reason'] for c in configs if c['patience'] == pat)
        total_pat = sum(reasons.values())
        parts = ", ".join(f"{r}={n}/{total_pat}" for r, n in reasons.most_common())
        pat_str = str(pat) if pat is not None else 'off'
        print(f"    pat={pat_str:>3}: {parts}")

    # --- Model dependency analysis ---

    optimal_per_model = {}
    print("\n  OPTIMAL CONFIG PER MODEL (best R2):")
    for model_name in models:
        model_configs = [c for c in configs if c['model'] == model_name
                         and not np.isnan(c['r2_mean'])]
        if model_configs:
            best = max(model_configs, key=lambda c: c['r2_mean'])
            optimal_per_model[model_name] = {
                'step_size': best['step_size'],
                'patience': best['patience'],
                'budget': best['budget'],
                'n_features': best['n_features'],
                'r2_mean': best['r2_mean'],
                'stop_reason': best['stop_reason'],
                'utilization': best['utilization'],
            }
            pat_str = str(best['patience']) if best['patience'] is not None else 'off'
            print(f"    {model_name:>10}: step={best['step_size']:>2} pat={pat_str:>3} "
                  f"b={best['budget']:>3} dim={best['n_features']:>3} "
                  f"R2={best['r2_mean']:.4f} stop={best['stop_reason']}")

    # Check if optimal step_size varies by model
    optimal_steps = set()
    for model_name in models:
        mc = [c for c in configs if c['model'] == model_name and not np.isnan(c['r2_mean'])]
        if mc:
            optimal_steps.add(max(mc, key=lambda c: c['r2_mean'])['step_size'])

    is_model_dependent = len(optimal_steps) > 1
    if is_model_dependent:
        print(f"\n  ** STEP_SIZE IS MODEL-DEPENDENT: optimal steps vary: {sorted(optimal_steps)}")
        print(f"  ** Recommendation: include step_size in search space")
    elif optimal_steps:
        print(f"\n  ** STEP_SIZE is NOT model-dependent: optimal = {optimal_steps.pop()}")

    return {
        'configs': configs,
        'best_by_model': best_by_model,
        'step_budget_heatmap': step_budget_avg,
        'step_speed': step_speed,
        'patience_effect': patience_avg,
        'utilization_heatmap': {str(k): v for k, v in utilization_heatmap.items()},
        'stop_reason_counts': {str(k): v for k, v in stop_reason_counts.items()},
        'optimal_per_model': optimal_per_model,
        'is_model_dependent': is_model_dependent,
    }


if __name__ == '__main__':
    parser = create_experiment_parser('C13: Greedy Step Size & Patience',
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
    save_experiment_results(results, args.output, 'c13_greedy_step_size', args.dataset)
