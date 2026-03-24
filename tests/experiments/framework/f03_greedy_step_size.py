#!/usr/bin/env python3
"""F03: Greedy step size and patience — how granular should greedy search be?

Paper: Framework
Question: Whether fine-grained greedy (step=1) outperforms coarse (step=50),
    and how patience interacts with step size across greedy-family allocations.

Tests step_size × patience × budget × model × allocation_method grid using
create_hybrid directly. Covers all three step-size-sensitive allocations:
greedy, dominant_greedy, and greedy_feature (patience only, step_size ignored).

Usage:
    python f03_greedy_step_size.py --dataset logd --reps all --mode full
    python f03_greedy_step_size.py --dataset flab_thermo_tresanco --reps all --mode full
    python f03_greedy_step_size.py --dataset tcga_luad_vs_lusc --reps all --mode full
    python f03_greedy_step_size.py --dataset buchwald_hartwig --reps all --mode full

    # Custom importance method (default: treeshap):
    python f03_greedy_step_size.py --dataset logd --importance-method lightgbm_gain
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_model_factories, get_domain_metric, cv_split,
    create_hybrid, apply_feature_selection,
    parse_rep_args, FRAMEWORK_TASKS,
)
from kirby.find_best_hybrid import _score_to_comparable


# Greedy-family allocations that are step-size/patience sensitive.
# greedy and dominant_greedy use step_size + patience.
# greedy_feature uses patience only (step_size=1 implicitly).
GREEDY_ALLOCATIONS = ['greedy', 'dominant_greedy', 'greedy_feature']

STEP_SIZES = [1, 2, 5, 10, 25, 50]
PATIENCE_VALUES = [2, 3, 5, None]  # None = no early stopping
BUDGETS = [50, 100, 200, 500]


def _best_config(configs, lower_is_better):
    """Pick best config from list using comparable scores."""
    comparables = [_score_to_comparable(c['score_mean'], lower_is_better)
                   for c in configs]
    return configs[int(np.argmax(comparables))]


def run(data, config, output_dir, dataset_name, importance_method='treeshap',
        **kwargs):
    """Run greedy step size comparison across greedy-family allocations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F03: GREEDY STEP SIZE & PATIENCE")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(data['labels'])
    task_type = data.get('task_type', 'regression')
    models = get_model_factories(config['models'], task_type=task_type,
                                 n_samples=n_samples)
    n_folds = config['n_splits']
    max_folds = config.get('max_folds', n_folds)
    mode = 'intermediary' if n_folds == 1 else 'full'
    budgets = config.get('budgets', BUDGETS)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')

    # greedy_feature ignores step_size, so it only runs once per
    # (patience, budget, model) — effectively step_sizes=[1].
    n_step_sensitive = len([a for a in GREEDY_ALLOCATIONS if a != 'greedy_feature'])
    n_greedy_feature = 1  # runs once per (patience, budget, model)
    total = (n_step_sensitive * len(STEP_SIZES) + n_greedy_feature) \
        * len(PATIENCE_VALUES) * len(budgets) * len(models)

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Allocations: {GREEDY_ALLOCATIONS}")
    print(f"  {len(STEP_SIZES)} step_sizes × {len(PATIENCE_VALUES)} patience × "
          f"{len(budgets)} budgets × {len(models)} models = {total} evaluations")
    print(f"  Importance: {importance_method}, folds: {max_folds}/{n_folds}\n", flush=True)

    sample_X = next(iter(reps.values()))
    all_configs = []
    idx = 0

    for alloc_method in GREEDY_ALLOCATIONS:
        # greedy_feature doesn't use step_size — run with step_size=1 only
        effective_steps = [1] if alloc_method == 'greedy_feature' else STEP_SIZES

        for step_size in effective_steps:
            for patience in PATIENCE_VALUES:
                for budget in budgets:
                    for model_name, model_factory in models.items():
                        idx += 1
                        fold_scores = []
                        last_n_features = 0
                        last_feature_info = {}
                        t0 = time.time()

                        for fold_i, (train_idx, val_idx) in enumerate(cv_split(
                            groups, n_folds, sample_X, labels, mode=mode
                        )):
                            if fold_i >= max_folds:
                                break
                            train_reps = {k: v[train_idx] for k, v in reps.items()}
                            val_reps = {k: v[val_idx] for k, v in reps.items()}
                            y_train, y_val = labels[train_idx], labels[val_idx]

                            hybrid_kwargs = dict(
                                allocation_method=alloc_method,
                                budget=budget,
                                patience=patience,
                                importance_method=importance_method,
                            )
                            # Only pass step_size for allocations that use it
                            if alloc_method != 'greedy_feature':
                                hybrid_kwargs['step_size'] = step_size

                            try:
                                X_train_h, feature_info = create_hybrid(
                                    train_reps, y_train, **hybrid_kwargs,
                                )
                                X_val_h = apply_feature_selection(
                                    val_reps, feature_info
                                )
                            except Exception as e:
                                if idx <= 5:
                                    print(
                                        f"  [{idx}/{total}] {alloc_method}/"
                                        f"step={step_size}/pat={patience}/"
                                        f"b={budget}/{model_name}: "
                                        f"HYBRID FAILED - {e}", flush=True
                                    )
                                continue

                            scaler = StandardScaler()
                            X_train_s = scaler.fit_transform(X_train_h)
                            X_val_s = scaler.transform(X_val_h)
                            last_n_features = X_train_h.shape[1]
                            last_feature_info = feature_info

                            try:
                                model = model_factory()
                                model.fit(X_train_s, y_train)
                                fold_scores.append(
                                    metric_fn(y_val, model.predict(X_val_s))
                                )
                            except Exception as e:
                                if idx <= 5:
                                    print(
                                        f"  [{idx}/{total}] {alloc_method}/"
                                        f"{model_name}: MODEL FAILED - {e}",
                                        flush=True,
                                    )
                                continue

                        elapsed = time.time() - t0

                        if not fold_scores:
                            score_mean = float('nan')
                            score_std = float('nan')
                            n_features = 0
                            stop_reason = 'no_valid_rep'
                            n_iterations = 0
                            utilization = 0.0
                            alloc_dict = {}
                        else:
                            score_mean = float(np.mean(fold_scores))
                            score_std = float(np.std(fold_scores))
                            n_features = last_n_features

                            # Extract greedy diagnostics from last fold.
                            # History key differs per allocation method.
                            history = (
                                last_feature_info.get('greedy_history')
                                or last_feature_info.get('dominant_greedy_history')
                                or last_feature_info.get('greedy_feature_history')
                                or []
                            )
                            alloc_dict = last_feature_info.get(
                                'allocation', {}
                            )
                            n_iterations = len(history)
                            utilization = (
                                n_features / budget if budget > 0 else 0.0
                            )

                            # Infer stop reason
                            total_allocated = sum(alloc_dict.values())
                            if n_iterations == 0:
                                stop_reason = 'no_valid_rep'
                            elif total_allocated >= budget:
                                stop_reason = 'budget'
                            elif patience is not None and n_iterations >= 2:
                                # Check if patience was the stopping criterion
                                # History entries may use 'val_r2' or 'val_score'
                                score_key = None
                                if history:
                                    if 'val_score' in history[0]:
                                        score_key = 'val_score'
                                    elif 'val_r2' in history[0]:
                                        score_key = 'val_r2'
                                if score_key:
                                    hist_scores = [h[score_key]
                                                   for h in history]
                                    # Internal evaluator always uses R²/accuracy
                                    # (higher=better), regardless of domain metric.
                                    best_idx = int(np.argmax(hist_scores))
                                    tail = n_iterations - best_idx - 1
                                    if tail >= patience:
                                        stop_reason = 'patience'
                                    else:
                                        stop_reason = 'reps_maxed'
                                else:
                                    stop_reason = 'reps_maxed'
                            else:
                                stop_reason = 'reps_maxed'

                        entry = {
                            'allocation': alloc_method,
                            'step_size': step_size,
                            'patience': patience,
                            'budget': budget,
                            'model': model_name,
                            'importance': importance_method,
                            'score_mean': score_mean,
                            'score_std': score_std,
                            'fold_scores': fold_scores,
                            'n_features': n_features,
                            'elapsed_s': round(elapsed, 2),
                            'n_iterations': n_iterations,
                            'stop_reason': stop_reason,
                            'utilization': utilization,
                            'allocation_dict': {
                                k: int(v) for k, v in alloc_dict.items()
                            },
                        }
                        all_configs.append(entry)

                        pat_str = (
                            str(patience) if patience is not None else 'off'
                        )
                        if idx % 10 == 0 or idx <= 5:
                            print(
                                f"  [{idx}/{total}] {alloc_method}/"
                                f"step={step_size}/pat={pat_str}/"
                                f"b={budget}/{model_name}: "
                                f"{metric_name}={score_mean:.4f} "
                                f"dim={n_features} util={utilization:.0%} "
                                f"stop={stop_reason} ({elapsed:.1f}s)",
                                flush=True,
                            )

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    valid = [c for c in all_configs if not np.isnan(c['score_mean'])]

    # --- Best by model (across all alloc/step/patience/budget) ---
    best_by_model = {}
    for model_name in models:
        model_cfgs = [c for c in valid if c['model'] == model_name]
        if model_cfgs:
            best_by_model[model_name] = _best_config(model_cfgs, lower_is_better)

    # --- Best by allocation method (averaged across models) ---
    alloc_ranking = {}
    for alloc_method in GREEDY_ALLOCATIONS:
        alloc_cfgs = [c for c in valid if c['allocation'] == alloc_method]
        if alloc_cfgs:
            # For each model, find the best config under this allocation
            per_model_best = []
            for model_name in models:
                mc = [c for c in alloc_cfgs if c['model'] == model_name]
                if mc:
                    best = _best_config(mc, lower_is_better)
                    per_model_best.append(
                        _score_to_comparable(best['score_mean'], lower_is_better)
                    )
            if per_model_best:
                alloc_ranking[alloc_method] = float(np.mean(per_model_best))

    # --- Step size × budget heatmap (greedy + dominant_greedy only) ---
    step_budget_avg = {}
    for step_size in STEP_SIZES:
        step_budget_avg[step_size] = {}
        for budget in BUDGETS:
            scores = [
                _score_to_comparable(c['score_mean'], lower_is_better)
                for c in valid
                if c['step_size'] == step_size and c['budget'] == budget
                and c['allocation'] != 'greedy_feature'
            ]
            step_budget_avg[step_size][budget] = (
                float(np.mean(scores)) if scores else float('nan')
            )

    # --- Speed vs quality per step_size ---
    step_speed = {}
    for step_size in STEP_SIZES:
        step_cfgs = [c for c in all_configs
                     if c['step_size'] == step_size
                     and c['allocation'] != 'greedy_feature']
        times = [c['elapsed_s'] for c in step_cfgs]
        scores = [
            _score_to_comparable(c['score_mean'], lower_is_better)
            for c in step_cfgs if not np.isnan(c['score_mean'])
        ]
        step_speed[step_size] = {
            'avg_time_s': float(np.mean(times)) if times else float('nan'),
            'avg_comparable': float(np.mean(scores)) if scores else float('nan'),
        }

    # --- Patience effect per patience value ---
    patience_avg = {}
    for pat in PATIENCE_VALUES:
        scores = [
            _score_to_comparable(c['score_mean'], lower_is_better)
            for c in valid if c['patience'] == pat
        ]
        patience_avg[str(pat)] = (
            float(np.mean(scores)) if scores else float('nan')
        )

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print(f"\n  ALLOCATION METHOD RANKING (avg comparable score):")
    for alloc, avg in sorted(alloc_ranking.items(), key=lambda x: -x[1]):
        print(f"    {alloc:<20} {avg:.4f}")

    print(f"\n  BEST BY MODEL ({metric_name}):")
    for model_name, best in best_by_model.items():
        pat_str = str(best['patience']) if best['patience'] is not None else 'off'
        print(f"  >> {model_name}: {best['allocation']}/step={best['step_size']}/"
              f"pat={pat_str}/b={best['budget']} -> "
              f"{metric_name}={best['score_mean']:.4f} ({best['elapsed_s']:.1f}s)")

    print(f"\n  STEP SIZE vs BUDGET (avg comparable, greedy+dominant_greedy):")
    header = f"  {'Step':>6}" + "".join(f"  b={b:>4}" for b in BUDGETS) + "  avg_time"
    print(header)
    print("  " + "-" * (6 + 8 * len(BUDGETS) + 10))
    for step_size in STEP_SIZES:
        row = f"  {step_size:>6}"
        for budget in BUDGETS:
            val = step_budget_avg[step_size][budget]
            row += f"  {val:>6.4f}" if not np.isnan(val) else f"  {'--':>6}"
        avg_t = step_speed[step_size]['avg_time_s']
        row += f"  {avg_t:>7.1f}s" if not np.isnan(avg_t) else f"  {'--':>7}s"
        print(row)

    print(f"\n  PATIENCE EFFECT (avg comparable score):")
    for pat, avg in patience_avg.items():
        print(f"    patience={pat:<5}: {avg:.4f}")

    # --- Utilization analysis ---
    utilization_heatmap = {}
    print(f"\n  BUDGET UTILIZATION (step × patience → avg %):")
    pat_header = "".join(
        f"  pat={str(p) if p is not None else 'off':>5}" for p in PATIENCE_VALUES
    )
    print(f"  {'step':>6}{pat_header}")
    print("  " + "-" * (6 + 10 * len(PATIENCE_VALUES)))
    for step_size in STEP_SIZES:
        utilization_heatmap[step_size] = {}
        row = f"  {step_size:>6}"
        for pat in PATIENCE_VALUES:
            subset = [c for c in valid
                      if c['step_size'] == step_size and c['patience'] == pat
                      and c['allocation'] != 'greedy_feature']
            avg_util = (
                float(np.mean([c['utilization'] for c in subset]))
                if subset else float('nan')
            )
            utilization_heatmap[step_size][str(pat)] = avg_util
            row += (
                f"  {avg_util:>8.0%}" if not np.isnan(avg_util)
                else f"  {'--':>8}"
            )
        print(row)

    # Under-utilization alert
    under_util = [c for c in valid if c['utilization'] < 0.3]
    if under_util:
        print(f"\n  ** UNDER-UTILIZATION: {len(under_util)}/{len(valid)} configs "
              f"({len(under_util)/len(valid):.0%}) use <30% of budget")

    # --- Stop reason breakdowns ---
    stop_reason_by_alloc = {}
    print(f"\n  STOP REASONS (by allocation):")
    for alloc_method in GREEDY_ALLOCATIONS:
        reasons = Counter(
            c['stop_reason'] for c in all_configs
            if c['allocation'] == alloc_method
        )
        stop_reason_by_alloc[alloc_method] = dict(reasons)
        total_a = sum(reasons.values())
        parts = ", ".join(f"{r}={n}/{total_a}" for r, n in reasons.most_common())
        print(f"    {alloc_method:<20}: {parts}")

    stop_reason_by_step = {}
    print(f"\n  STOP REASONS (by step_size, greedy+dominant_greedy):")
    for step_size in STEP_SIZES:
        reasons = Counter(
            c['stop_reason'] for c in all_configs
            if c['step_size'] == step_size
            and c['allocation'] != 'greedy_feature'
        )
        stop_reason_by_step[step_size] = dict(reasons)
        total_s = sum(reasons.values())
        parts = ", ".join(f"{r}={n}/{total_s}" for r, n in reasons.most_common())
        print(f"    step={step_size:>2}: {parts}")

    print(f"\n  STOP REASONS (by patience):")
    for pat in PATIENCE_VALUES:
        reasons = Counter(
            c['stop_reason'] for c in all_configs if c['patience'] == pat
        )
        total_p = sum(reasons.values())
        parts = ", ".join(f"{r}={n}/{total_p}" for r, n in reasons.most_common())
        pat_str = str(pat) if pat is not None else 'off'
        print(f"    pat={pat_str:>3}: {parts}")

    # --- Optimal config per model ---
    optimal_per_model = {}
    print(f"\n  OPTIMAL CONFIG PER MODEL (best {metric_name}):")
    for model_name in models:
        mc = [c for c in valid if c['model'] == model_name]
        if mc:
            best = _best_config(mc, lower_is_better)
            optimal_per_model[model_name] = {
                'allocation': best['allocation'],
                'step_size': best['step_size'],
                'patience': best['patience'],
                'budget': best['budget'],
                'n_features': best['n_features'],
                'score_mean': best['score_mean'],
                'stop_reason': best['stop_reason'],
                'utilization': best['utilization'],
            }
            pat_str = str(best['patience']) if best['patience'] is not None else 'off'
            print(f"    {model_name:>10}: {best['allocation']}/step={best['step_size']:>2} "
                  f"pat={pat_str:>3} b={best['budget']:>3} dim={best['n_features']:>3} "
                  f"{metric_name}={best['score_mean']:.4f} stop={best['stop_reason']}")

    # Model dependency: does optimal step_size vary by model?
    optimal_steps = set()
    for model_name in models:
        mc = [c for c in valid if c['model'] == model_name
              and c['allocation'] != 'greedy_feature']
        if mc:
            best = _best_config(mc, lower_is_better)
            optimal_steps.add(best['step_size'])

    is_model_dependent = len(optimal_steps) > 1
    if is_model_dependent:
        print(f"\n  ** STEP_SIZE IS MODEL-DEPENDENT: optimal steps vary: {sorted(optimal_steps)}")
        print(f"  ** Recommendation: include step_size in search space")
    elif optimal_steps:
        print(f"\n  ** STEP_SIZE is NOT model-dependent: optimal = {optimal_steps.pop()}")

    # =========================================================================
    # BUILD STRUCTURED OUTPUT
    # =========================================================================
    summary = {
        'alloc_ranking': alloc_ranking,
        'best_by_model': best_by_model,
        'step_budget_heatmap': step_budget_avg,
        'step_speed': step_speed,
        'patience_effect': patience_avg,
        'utilization_heatmap': {str(k): v for k, v in utilization_heatmap.items()},
        'stop_reasons_by_alloc': stop_reason_by_alloc,
        'stop_reasons_by_step': {str(k): v for k, v in stop_reason_by_step.items()},
        'optimal_per_model': optimal_per_model,
        'is_model_dependent': is_model_dependent,
    }

    return make_structured_output(
        experiment_id='f03',
        dataset=dataset_name,
        data=data,
        config={
            'n_folds': n_folds,
            'models': list(models.keys()),
            'budgets': BUDGETS,
            'step_sizes': STEP_SIZES,
            'patience_values': PATIENCE_VALUES,
            'allocations': GREEDY_ALLOCATIONS,
            'importance_method': importance_method,
        },
        results={'configs': all_configs, 'summary': summary},
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser('F03: Greedy Step Size & Patience',
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
    save_experiment_results(results, args.output, 'f03_greedy_step_size',
                            args.dataset)
