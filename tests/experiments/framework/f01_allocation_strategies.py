#!/usr/bin/env python3
"""F01: Allocation strategy comparison — which allocation method wins?

Paper: Framework
Question: Do smarter budget allocations (softmax, power-law, thresholded,
    greedy variants, rank-based, entropy-weighted) outperform simple greedy
    or linear-proportional?

Tiered design:
  Tier 1 — Screen all allocation families with fixed defaults (budget=150,
      model=rf).  One entry per family (21 strategies).
  Tier 2 — For the top-N families that have tunable parameters, sweep the
      parameter grid across all budgets × models.

Usage:
    python f01_allocation_strategies.py --dataset logd --framework --mode full
    python f01_allocation_strategies.py --dataset logd --framework --mode full --models ridge,rf,xgboost,lightgbm
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
    save_experiment_results, make_structured_output,
    get_model_factories, get_domain_metric, cv_split,
    create_hybrid, apply_feature_selection,
    parse_rep_args, FRAMEWORK_TASKS,
)
from kirby.hybrid import NAMED_ALLOCATIONS
from kirby.find_best_hybrid import _score_to_comparable


# All named allocations from hybrid.py — one entry per family for screening.
STRATEGY_NAMES = sorted(NAMED_ALLOCATIONS.keys())

BUDGETS = [50, 100, 200, 500]

# Tier 1 defaults — screen strategy with everything else fixed.
SCREEN_BUDGET = 150
SCREEN_MODEL = 'rf'

# Tier 2 top-N: how many winning families to expand in parameter tuning.
TIER2_TOP_N = 5

# Parameter grids for tunable families.  Keys are the *base* strategy name
# (the one that appears in NAMED_ALLOCATIONS).  Each grid entry is
# (display_name, allocation_method_name, extra_kwargs) where extra_kwargs
# are passed to create_hybrid.
#
# Families without a grid here are not tunable and only appear in Tier 1.
PARAM_GRIDS = {
    'softmax': [
        ('softmax_T0.5', 'softmax_T0.5', {}),
        ('softmax_T1',   'softmax_T1',   {}),
        ('softmax_T2',   'softmax_T2',   {}),
        ('softmax_T5',   'softmax_T5',   {}),
        ('softmax_T10',  'softmax_T10',  {}),
    ],
    'power': [
        ('power_1.5', 'performance_weighted',
         {'weighting_scheme': 'power_law', 'weighting_params': {'power': 1.5}}),
        ('power_2',   'power_2', {}),
        ('power_3',   'power_3', {}),
        ('power_5',   'performance_weighted',
         {'weighting_scheme': 'power_law', 'weighting_params': {'power': 5.0}}),
    ],
    'thresholded': [
        ('thresh_0.3', 'performance_weighted',
         {'weighting_scheme': 'thresholded', 'weighting_params': {'alpha': 0.3}}),
        ('thresh_0.5', 'thresh_0.5', {}),
        ('thresh_0.7', 'performance_weighted',
         {'weighting_scheme': 'thresholded', 'weighting_params': {'alpha': 0.7}}),
    ],
    'top_k': [
        ('top_2', 'performance_weighted',
         {'weighting_scheme': 'top_k', 'weighting_params': {'k': 2}}),
        ('top_3', 'top_3', {}),
        ('top_5', 'performance_weighted',
         {'weighting_scheme': 'top_k', 'weighting_params': {'k': 5}}),
    ],
    'rank_exp': [
        ('rank_exp_d0.1', 'performance_weighted',
         {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.1}}),
        ('rank_exp_d0.3', 'rank_exp', {}),
        ('rank_exp_d0.5', 'performance_weighted',
         {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.5}}),
        ('rank_exp_d1.0', 'performance_weighted',
         {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 1.0}}),
    ],
    'score_gap': [
        ('score_gap_0.2', 'performance_weighted',
         {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.2}}),
        ('score_gap_0.3', 'score_gap', {}),
        ('score_gap_0.5', 'performance_weighted',
         {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.5}}),
        ('score_gap_0.7', 'performance_weighted',
         {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.7}}),
    ],
    'entropy': [
        ('entropy_p0.5', 'performance_weighted',
         {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 0.5}}),
        ('entropy_p1.0', 'entropy', {}),
        ('entropy_p2.0', 'performance_weighted',
         {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 2.0}}),
        ('entropy_p3.0', 'performance_weighted',
         {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 3.0}}),
    ],
}

# Map each NAMED_ALLOCATION to its tunable family (if any).
# Strategies not in a family are non-tunable (greedy, fixed, etc.).
_STRATEGY_TO_FAMILY = {}
# softmax family: all softmax_T* entries
for name in NAMED_ALLOCATIONS:
    if name.startswith('softmax_T'):
        _STRATEGY_TO_FAMILY[name] = 'softmax'
    elif name.startswith('power_'):
        _STRATEGY_TO_FAMILY[name] = 'power'
    elif name == 'thresh_0.5':
        _STRATEGY_TO_FAMILY[name] = 'thresholded'
    elif name == 'top_3':
        _STRATEGY_TO_FAMILY[name] = 'top_k'
    elif name == 'rank_exp':
        _STRATEGY_TO_FAMILY[name] = 'rank_exp'
    elif name == 'score_gap':
        _STRATEGY_TO_FAMILY[name] = 'score_gap'
    elif name == 'entropy':
        _STRATEGY_TO_FAMILY[name] = 'entropy'


def _eval_one(strat_name, budget, model_name, model_factory,
              reps, labels, groups, n_folds, max_folds,
              metric_fn, importance_method, extra_kwargs=None):
    """Run a single (strategy, budget, model) evaluation. Returns result dict."""
    mode = 'intermediary' if n_folds == 1 else 'full'
    sample_X = next(iter(reps.values()))
    fold_scores = []
    last_n_features = 0
    last_allocation = {}
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
            allocation_method=strat_name,
            budget=budget,
            importance_method=importance_method,
        )
        if extra_kwargs:
            hybrid_kwargs.update(extra_kwargs)

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
        last_allocation = feature_info.get('allocation', {})

        try:
            model = model_factory()
            model.fit(X_train_s, y_train)
            fold_scores.append(metric_fn(y_val, model.predict(X_val_s)))
        except Exception:
            continue

    wall_time = time.time() - t0

    if not fold_scores:
        score_mean, score_std = float('nan'), float('nan')
        n_features = 0
    else:
        score_mean = float(np.mean(fold_scores))
        score_std = float(np.std(fold_scores))
        n_features = last_n_features

    return {
        'strategy': strat_name,
        'model': model_name,
        'budget': budget,
        'importance': importance_method,
        'score_mean': score_mean,
        'score_std': score_std,
        'fold_scores': fold_scores,
        'n_features': n_features,
        'allocation_dict': {k: int(v) for k, v in last_allocation.items()},
        'wall_time': round(wall_time, 2),
    }


def run(data, config, output_dir, dataset_name, importance_method='treeshap',
        **kwargs):
    """Run tiered allocation strategy comparison.

    Tier 1 — Screen all named allocations with fixed defaults.
    Tier 2 — For top-N tunable families, sweep parameter grids × budgets × models.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT F01: ALLOCATION STRATEGY COMPARISON (tiered)")
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

    screen_budget = SCREEN_BUDGET
    if screen_budget not in budgets:
        screen_budget = sorted(budgets)[len(budgets) // 2]

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Strategies: {len(STRATEGY_NAMES)}")
    print(f"  Tier 1: {len(STRATEGY_NAMES)} evals — screen strategies "
          f"(b={screen_budget}, model={screen_model_name})")
    print(f"  Tier 2: top {TIER2_TOP_N} tunable families — "
          f"param grid × {len(budgets)} budgets × {len(models)} models")
    print(f"  Importance: {importance_method}, folds: {max_folds}/{n_folds}\n",
          flush=True)

    all_configs = []
    idx = 0

    # =========================================================================
    # TIER 1: Screen all strategies with fixed defaults
    # =========================================================================
    print("  --- Tier 1: Strategy screening ---", flush=True)
    tier1_scores = {}  # {strategy_name: comparable_score}

    for strat_name in STRATEGY_NAMES:
        idx += 1
        entry = _eval_one(
            strat_name, screen_budget, screen_model_name, screen_factory,
            reps, labels, groups, n_folds, max_folds,
            metric_fn, importance_method,
        )
        entry['tier'] = 1
        all_configs.append(entry)

        comp = _score_to_comparable(entry['score_mean'], lower_is_better)
        tier1_scores[strat_name] = comp

        print(f"  [{idx}] {strat_name}: "
              f"{metric_name}={entry['score_mean']:.4f} "
              f"dim={entry['n_features']} ({entry['wall_time']:.1f}s)",
              flush=True)

    # Rank strategies
    ranked = sorted(tier1_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  TIER 1 RANKING:")
    for rank, (name, comp) in enumerate(ranked, 1):
        print(f"    {rank:>2}. {name:<25} {comp:.4f}")

    # Identify top-N tunable families for tier 2
    seen_families = set()
    tier2_families = []
    for strat_name, _ in ranked:
        family = _STRATEGY_TO_FAMILY.get(strat_name)
        if family and family not in seen_families:
            seen_families.add(family)
            tier2_families.append(family)
        if len(tier2_families) >= TIER2_TOP_N:
            break

    print(f"\n  Tier 2 families: {tier2_families}")

    # =========================================================================
    # TIER 2: Parameter tuning for winning families
    # =========================================================================
    print("\n  --- Tier 2: Parameter tuning ---", flush=True)

    for family in tier2_families:
        grid = PARAM_GRIDS.get(family, [])
        if not grid:
            continue

        print(f"\n  >> Family: {family} ({len(grid)} param settings "
              f"× {len(budgets)} budgets × {len(models)} models)", flush=True)

        for display_name, alloc_method, extra_kw in grid:
            for budget in budgets:
                for model_name, model_factory in models.items():
                    # Skip if this exact config was already run in tier 1
                    if (alloc_method == display_name
                            and budget == screen_budget
                            and model_name == screen_model_name
                            and not extra_kw):
                        continue

                    idx += 1
                    entry = _eval_one(
                        alloc_method, budget, model_name, model_factory,
                        reps, labels, groups, n_folds, max_folds,
                        metric_fn, importance_method,
                        extra_kwargs=extra_kw if extra_kw else None,
                    )
                    # Override strategy name for display/analysis
                    entry['strategy'] = display_name
                    entry['family'] = family
                    entry['tier'] = 2
                    all_configs.append(entry)

                    if idx % 20 == 0 or idx <= 3:
                        print(
                            f"  [{idx}] {display_name}/b={budget}/{model_name}: "
                            f"{metric_name}={entry['score_mean']:.4f} "
                            f"({entry['wall_time']:.1f}s)",
                            flush=True,
                        )

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    valid = [c for c in all_configs if not np.isnan(c['score_mean'])]

    # Best per model (across all tiers)
    best_by_model = {}
    for model_name in models:
        model_cfgs = [c for c in valid if c['model'] == model_name]
        if model_cfgs:
            comparables = [_score_to_comparable(c['score_mean'], lower_is_better)
                           for c in model_cfgs]
            best_by_model[model_name] = model_cfgs[int(np.argmax(comparables))]

    # Strategy ranking from tier 1
    tier1_ranking = {name: comp for name, comp in ranked}

    # Tier 2: best param per family
    tier2_best = {}
    tier2_configs = [c for c in valid if c.get('tier') == 2]
    for family in tier2_families:
        family_cfgs = [c for c in tier2_configs
                       if c.get('family') == family]
        if family_cfgs:
            comparables = [_score_to_comparable(c['score_mean'], lower_is_better)
                           for c in family_cfgs]
            best = family_cfgs[int(np.argmax(comparables))]
            tier2_best[family] = best

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print(f"\n  BEST BY MODEL ({metric_name}):")
    for model_name, best in best_by_model.items():
        print(f"  >> {model_name}: {best['strategy']}/b={best['budget']} "
              f"-> {metric_name}={best['score_mean']:.4f}")

    if tier2_best:
        print(f"\n  TIER 2 — BEST PARAM PER FAMILY:")
        for family, best in tier2_best.items():
            print(f"    {family:<15}: {best['strategy']}/b={best['budget']}/"
                  f"{best['model']} -> {metric_name}={best['score_mean']:.4f}")

    # Build structured output
    summary = {
        'tier1_ranking': tier1_ranking,
        'best_by_model': best_by_model,
        'tier2_best_per_family': {
            f: {'strategy': b['strategy'], 'budget': b['budget'],
                'model': b['model'], 'score_mean': b['score_mean']}
            for f, b in tier2_best.items()
        },
        'tier2_families': tier2_families,
    }

    return make_structured_output(
        experiment_id='f01',
        dataset=dataset_name,
        data=data,
        config={
            'n_folds': n_folds,
            'models': list(models.keys()),
            'budgets': budgets,
            'strategies': STRATEGY_NAMES,
            'importance_method': importance_method,
            'screen_defaults': {
                'budget': screen_budget,
                'model': screen_model_name,
            },
            'tier2_top_n': TIER2_TOP_N,
        },
        results={'configs': all_configs, 'summary': summary},
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
