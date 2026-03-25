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

# Non-greedy baseline for comparison — ignores step_size and patience.
BASELINE_ALLOCATIONS = ['performance_weighted']

STEP_SIZES = [1, 2, 5, 10, 25, 50]
PATIENCE_VALUES = [2, 3, 5, None]  # None = no early stopping
BUDGETS = [50, 100, 200, 500]

# Tier 1 defaults — used to screen step_size with everything else held fixed.
SCREEN_PATIENCE = 5
SCREEN_BUDGET = 150
SCREEN_MODEL = 'rf'


def _best_config(configs, lower_is_better):
    """Pick best config from list using comparable scores."""
    comparables = [_score_to_comparable(c['score_mean'], lower_is_better)
                   for c in configs]
    return configs[int(np.argmax(comparables))]


def _eval_one(alloc_method, step_size, patience, budget, model_name,
              model_factory, reps, labels, groups, n_folds, max_folds,
              metric_fn, importance_method):
    """Run a single (alloc, step, patience, budget, model) configuration.

    Returns a result dict or None on total failure.
    """
    mode = 'intermediary' if n_folds == 1 else 'full'
    sample_X = next(iter(reps.values()))
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
        if alloc_method != 'greedy_feature' and step_size is not None:
            hybrid_kwargs['step_size'] = step_size

        try:
            X_train_h, feature_info = create_hybrid(
                train_reps, y_train, **hybrid_kwargs,
            )
            X_val_h = apply_feature_selection(val_reps, feature_info)
        except Exception:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_h)
        X_val_s = scaler.transform(X_val_h)
        last_n_features = X_train_h.shape[1]
        last_feature_info = feature_info

        try:
            model = model_factory()
            model.fit(X_train_s, y_train)
            fold_scores.append(metric_fn(y_val, model.predict(X_val_s)))
        except Exception:
            continue

    elapsed = time.time() - t0

    if not fold_scores:
        return {
            'allocation': alloc_method, 'step_size': step_size,
            'patience': patience, 'budget': budget, 'model': model_name,
            'importance': importance_method,
            'score_mean': float('nan'), 'score_std': float('nan'),
            'fold_scores': [], 'n_features': 0, 'elapsed_s': round(elapsed, 2),
            'n_iterations': 0, 'stop_reason': 'no_valid_rep',
            'utilization': 0.0, 'allocation_dict': {},
        }

    score_mean = float(np.mean(fold_scores))
    score_std = float(np.std(fold_scores))
    n_features = last_n_features

    history = (
        last_feature_info.get('greedy_history')
        or last_feature_info.get('dominant_greedy_history')
        or last_feature_info.get('greedy_feature_history')
        or []
    )
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
        score_key = None
        if history:
            if 'val_score' in history[0]:
                score_key = 'val_score'
            elif 'val_r2' in history[0]:
                score_key = 'val_r2'
        if score_key:
            hist_scores = [h[score_key] for h in history]
            best_idx = int(np.argmax(hist_scores))
            tail = n_iterations - best_idx - 1
            stop_reason = 'patience' if tail >= patience else 'reps_maxed'
        else:
            stop_reason = 'reps_maxed'
    else:
        stop_reason = 'reps_maxed'

    return {
        'allocation': alloc_method, 'step_size': step_size,
        'patience': patience, 'budget': budget, 'model': model_name,
        'importance': importance_method,
        'score_mean': score_mean, 'score_std': score_std,
        'fold_scores': fold_scores, 'n_features': n_features,
        'elapsed_s': round(elapsed, 2), 'n_iterations': n_iterations,
        'stop_reason': stop_reason, 'utilization': utilization,
        'allocation_dict': {k: int(v) for k, v in alloc_dict.items()},
    }


def run(data, config, output_dir, dataset_name, importance_method='treeshap',
        **kwargs):
    """Run tiered greedy step size experiment.

    Tier 1 — Step size screening: test all step_sizes with fixed defaults
        (patience=5, budget=150, model=rf) per allocation.  Picks the best
        step_size per allocation.

    Tier 2 — Full search: sweep patience × budget × model using the winning
        step_size from Tier 1.  greedy_feature (step-insensitive) goes
        straight to Tier 2.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT F03: GREEDY STEP SIZE & PATIENCE (tiered)")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(data['labels'])
    task_type = data.get('task_type', 'regression')
    models = get_model_factories(config['models'], task_type=task_type,
                                 n_samples=n_samples)
    n_folds = config['n_splits']
    max_folds = config.get('max_folds', n_folds)
    budgets = config.get('budgets', BUDGETS)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')

    # Resolve screening model
    screen_model_name = SCREEN_MODEL
    if screen_model_name not in models:
        screen_model_name = next(iter(models))
    screen_factory = models[screen_model_name]

    # Pick screening budget: use SCREEN_BUDGET if it's in the budget list,
    # otherwise the middle value.
    screen_budget = SCREEN_BUDGET
    if screen_budget not in budgets:
        screen_budget = sorted(budgets)[len(budgets) // 2]

    # Step-sensitive allocations (greedy, dominant_greedy)
    step_allocs = [a for a in GREEDY_ALLOCATIONS if a != 'greedy_feature']
    n_tier1 = len(step_allocs) * len(STEP_SIZES)
    n_tier2_per_alloc = len(PATIENCE_VALUES) * len(budgets) * len(models)
    n_tier2 = len(GREEDY_ALLOCATIONS) * n_tier2_per_alloc
    n_baseline = len(BASELINE_ALLOCATIONS) * len(budgets) * len(models)

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Allocations: {GREEDY_ALLOCATIONS}")
    print(f"  Tier 1: {n_tier1} evals — screen step_size "
          f"(pat={SCREEN_PATIENCE}, b={screen_budget}, model={screen_model_name})")
    print(f"  Tier 2: up to {n_tier2} evals — "
          f"{len(PATIENCE_VALUES)} patience × {len(budgets)} budgets × {len(models)} models"
          f" (best step_size per alloc)")
    print(f"  Baseline: {n_baseline} evals")
    print(f"  Importance: {importance_method}, folds: {max_folds}/{n_folds}\n", flush=True)

    all_configs = []
    idx = 0

    # =========================================================================
    # TIER 1: Screen step_size per allocation
    # =========================================================================
    print("  --- Tier 1: Step size screening ---", flush=True)
    best_step_per_alloc = {}

    for alloc_method in step_allocs:
        tier1_results = []
        for step_size in STEP_SIZES:
            idx += 1
            entry = _eval_one(
                alloc_method, step_size, SCREEN_PATIENCE, screen_budget,
                screen_model_name, screen_factory,
                reps, labels, groups, n_folds, max_folds,
                metric_fn, importance_method,
            )
            # Tag as tier 1
            entry['tier'] = 1
            all_configs.append(entry)
            tier1_results.append(entry)
            print(
                f"  [{idx}] {alloc_method}/step={step_size}: "
                f"{metric_name}={entry['score_mean']:.4f} "
                f"dim={entry['n_features']} stop={entry['stop_reason']} "
                f"({entry['elapsed_s']:.1f}s)",
                flush=True,
            )

        # Pick best step_size for this allocation
        valid_t1 = [r for r in tier1_results if not np.isnan(r['score_mean'])]
        if valid_t1:
            best = _best_config(valid_t1, lower_is_better)
            best_step_per_alloc[alloc_method] = best['step_size']
            print(f"  >> {alloc_method} best step_size = {best['step_size']} "
                  f"({metric_name}={best['score_mean']:.4f})\n", flush=True)
        else:
            # Fallback
            best_step_per_alloc[alloc_method] = 10
            print(f"  >> {alloc_method}: no valid results, defaulting step_size=10\n",
                  flush=True)

    # =========================================================================
    # TIER 2: Full search with winning step_size
    # =========================================================================
    print("  --- Tier 2: Patience × budget × model search ---", flush=True)

    for alloc_method in GREEDY_ALLOCATIONS:
        if alloc_method in step_allocs:
            fixed_step = best_step_per_alloc[alloc_method]
            print(f"  {alloc_method}: using step_size={fixed_step} from Tier 1",
                  flush=True)
        else:
            fixed_step = 1  # greedy_feature ignores step_size

        for patience in PATIENCE_VALUES:
            for budget in budgets:
                for model_name, model_factory in models.items():
                    # Skip if this exact config was already run in Tier 1
                    if (alloc_method in step_allocs
                            and patience == SCREEN_PATIENCE
                            and budget == screen_budget
                            and model_name == screen_model_name):
                        continue

                    idx += 1
                    entry = _eval_one(
                        alloc_method, fixed_step, patience, budget,
                        model_name, model_factory,
                        reps, labels, groups, n_folds, max_folds,
                        metric_fn, importance_method,
                    )
                    entry['tier'] = 2
                    all_configs.append(entry)

                    pat_str = str(patience) if patience is not None else 'off'
                    if idx % 10 == 0 or idx <= 3:
                        print(
                            f"  [{idx}] {alloc_method}/"
                            f"step={fixed_step}/pat={pat_str}/"
                            f"b={budget}/{model_name}: "
                            f"{metric_name}={entry['score_mean']:.4f} "
                            f"dim={entry['n_features']} "
                            f"util={entry['utilization']:.0%} "
                            f"stop={entry['stop_reason']} "
                            f"({entry['elapsed_s']:.1f}s)",
                            flush=True,
                        )

    # =========================================================================
    # BASELINE: non-greedy allocations for comparison
    # =========================================================================
    print(f"\n  --- Baseline allocations (no step_size/patience) ---", flush=True)
    for alloc_method in BASELINE_ALLOCATIONS:
        for budget in budgets:
            for model_name, model_factory in models.items():
                idx += 1
                entry = _eval_one(
                    alloc_method, None, None, budget,
                    model_name, model_factory,
                    reps, labels, groups, n_folds, max_folds,
                    metric_fn, importance_method,
                )
                entry['tier'] = 0
                entry['stop_reason'] = 'baseline'
                all_configs.append(entry)

                print(f"  [{idx}] {alloc_method}/b={budget}/{model_name}: "
                      f"{metric_name}={entry['score_mean']:.4f} "
                      f"({entry['elapsed_s']:.1f}s)", flush=True)

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    valid = [c for c in all_configs if not np.isnan(c['score_mean'])]

    # --- Best by model (across all configs) ---
    best_by_model = {}
    for model_name in models:
        model_cfgs = [c for c in valid if c['model'] == model_name]
        if model_cfgs:
            best_by_model[model_name] = _best_config(model_cfgs, lower_is_better)

    # --- Best by allocation method ---
    alloc_ranking = {}
    for alloc_method in GREEDY_ALLOCATIONS + BASELINE_ALLOCATIONS:
        alloc_cfgs = [c for c in valid if c['allocation'] == alloc_method]
        if alloc_cfgs:
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

    # --- Tier 1: step_size speed vs quality ---
    tier1_configs = [c for c in all_configs if c.get('tier') == 1]
    step_speed = {}
    for step_size in STEP_SIZES:
        step_cfgs = [c for c in tier1_configs if c['step_size'] == step_size]
        times = [c['elapsed_s'] for c in step_cfgs]
        scores = [
            _score_to_comparable(c['score_mean'], lower_is_better)
            for c in step_cfgs if not np.isnan(c['score_mean'])
        ]
        step_speed[step_size] = {
            'avg_time_s': float(np.mean(times)) if times else float('nan'),
            'avg_comparable': float(np.mean(scores)) if scores else float('nan'),
        }

    # --- Patience effect (from Tier 2) ---
    tier2_configs = [c for c in valid if c.get('tier') == 2]
    patience_avg = {}
    for pat in PATIENCE_VALUES:
        scores = [
            _score_to_comparable(c['score_mean'], lower_is_better)
            for c in tier2_configs if c['patience'] == pat
        ]
        patience_avg[str(pat)] = (
            float(np.mean(scores)) if scores else float('nan')
        )

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print(f"\n  TIER 1 — STEP SIZE SCREENING (speed vs quality):")
    print(f"  {'Step':>6}  {'score':>8}  {'time':>8}")
    print(f"  {'-'*28}")
    for step_size in STEP_SIZES:
        s = step_speed[step_size]
        score_str = f"{s['avg_comparable']:.4f}" if not np.isnan(s['avg_comparable']) else '--'
        time_str = f"{s['avg_time_s']:.1f}s" if not np.isnan(s['avg_time_s']) else '--'
        print(f"  {step_size:>6}  {score_str:>8}  {time_str:>8}")

    print(f"\n  BEST STEP_SIZE PER ALLOCATION:")
    for alloc, step in best_step_per_alloc.items():
        print(f"    {alloc:<20} -> step_size={step}")

    print(f"\n  ALLOCATION METHOD RANKING (avg comparable score):")
    for alloc, avg in sorted(alloc_ranking.items(), key=lambda x: -x[1]):
        print(f"    {alloc:<20} {avg:.4f}")

    print(f"\n  PATIENCE EFFECT (avg comparable, Tier 2):")
    for pat, avg in patience_avg.items():
        print(f"    patience={pat:<5}: {avg:.4f}")

    print(f"\n  BEST BY MODEL ({metric_name}):")
    for model_name, best in best_by_model.items():
        pat_str = str(best['patience']) if best['patience'] is not None else 'off'
        print(f"  >> {model_name}: {best['allocation']}/step={best['step_size']}/"
              f"pat={pat_str}/b={best['budget']} -> "
              f"{metric_name}={best['score_mean']:.4f} ({best['elapsed_s']:.1f}s)")

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
    elif optimal_steps:
        print(f"\n  ** STEP_SIZE is NOT model-dependent: optimal = {optimal_steps.pop()}")

    # --- Stop reason breakdown ---
    print(f"\n  STOP REASONS (by allocation):")
    stop_reason_by_alloc = {}
    for alloc_method in GREEDY_ALLOCATIONS:
        reasons = Counter(
            c['stop_reason'] for c in all_configs
            if c['allocation'] == alloc_method
        )
        stop_reason_by_alloc[alloc_method] = dict(reasons)
        total_a = sum(reasons.values())
        parts = ", ".join(f"{r}={n}/{total_a}" for r, n in reasons.most_common())
        print(f"    {alloc_method:<20}: {parts}")

    # =========================================================================
    # BUILD STRUCTURED OUTPUT
    # =========================================================================
    summary = {
        'best_step_per_alloc': best_step_per_alloc,
        'alloc_ranking': alloc_ranking,
        'best_by_model': best_by_model,
        'step_speed': step_speed,
        'patience_effect': patience_avg,
        'optimal_per_model': optimal_per_model,
        'is_model_dependent': is_model_dependent,
        'stop_reasons_by_alloc': stop_reason_by_alloc,
    }

    return make_structured_output(
        experiment_id='f03',
        dataset=dataset_name,
        data=data,
        config={
            'n_folds': n_folds,
            'models': list(models.keys()),
            'budgets': budgets,
            'step_sizes': STEP_SIZES,
            'patience_values': PATIENCE_VALUES,
            'allocations': GREEDY_ALLOCATIONS + BASELINE_ALLOCATIONS,
            'importance_method': importance_method,
            'screen_defaults': {
                'patience': SCREEN_PATIENCE,
                'budget': screen_budget,
                'model': screen_model_name,
            },
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
