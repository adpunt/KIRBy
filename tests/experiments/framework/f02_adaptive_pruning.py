#!/usr/bin/env python3
"""F02: Adaptive pruning analysis — what would pruning cut, and at what cost?

Paper: Framework
Question: Can aggressive pruning halve search time without losing the best config?

Simulates adaptive pruning on pre-computed search results at multiple
thresholds, pruning all three dimensions (models, importance methods,
allocation methods) — matching the real pruning in find_best_hybrid._run_search.

Can consume pre-computed results JSON from f01/f03 (--from-json), or run
its own full grid search via find_best_hybrid.

Usage:
    # Own search (default):
    python f02_adaptive_pruning.py --dataset logd --reps all --mode full

    # From f01 results:
    python f02_adaptive_pruning.py --from-json tests/results/f01_allocation_strategies/logd.json

    # From f03 results:
    python f02_adaptive_pruning.py --from-json tests/results/f03_greedy_step_size/logd.json

    # Custom thresholds:
    python f02_adaptive_pruning.py --dataset logd --thresholds 0.3 0.5 0.7
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, load_experiment_results,
    find_best_hybrid, parse_rep_args, FRAMEWORK_TASKS,
    INTERNAL_EVALUATORS, DOWNSTREAM_MODELS, extract_top3_base_reps,
)
from kirby.find_best_hybrid import _score_to_comparable


def _should_prune(best_comparable: float, global_best_comparable: float,
                  threshold: float) -> bool:
    """Whether a search dimension would be pruned at this threshold.

    Prunes when the gap from the global best exceeds (1 - threshold) times the
    size of the global best. Works for scores that count up (R2, accuracy) and
    for negated ones (RMSE, MAE), because both arrive here as comparables where
    larger is better.

    This lived in find_best_hybrid until commit d2d41b2 removed the pruning it
    served, taking the function with it and leaving this file unable to import.
    It is reproduced here verbatim from that commit rather than restored to the
    engine: this file exists to simulate what the old rule WOULD have pruned,
    from search results that already ran, and nothing else calls it.
    """
    if abs(global_best_comparable) < 1e-10:
        return False  # global best near zero -- pruning meaningless
    gap = global_best_comparable - best_comparable
    return gap > (1 - threshold) * abs(global_best_comparable)


PRUNING_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# =========================================================================
# NORMALISE RESULTS FROM DIFFERENT SOURCES
# =========================================================================

def _normalise_result(r, source='search'):
    """Map result dict from different sources to a common schema.

    Common schema keys: model, allocation_method, importance_method,
    budget, score_mean, score_std.
    """
    if source == 'f01':
        return {
            'model': r['model'],
            'allocation_method': r.get('strategy', r.get('allocation_method', '')),
            'importance_method': r.get('importance', ''),
            'budget': r.get('budget', 0),
            'score_mean': r['score_mean'],
            'score_std': r.get('score_std', 0.0),
        }
    elif source == 'f03':
        return {
            'model': r['model'],
            'allocation_method': r.get('allocation', r.get('allocation_method', '')),
            'importance_method': r.get('importance', ''),
            'budget': r.get('budget', 0),
            'score_mean': r['score_mean'],
            'score_std': r.get('score_std', 0.0),
        }
    else:
        # From _run_search / find_best_hybrid.all_results
        return {
            'model': r['model'],
            'allocation_method': r.get('allocation_method', ''),
            'importance_method': r.get('importance_method', ''),
            'budget': r.get('budget', 0),
            'score_mean': r['score_mean'],
            'score_std': r.get('score_std', 0.0),
        }


def _detect_source(json_data):
    """Detect whether JSON came from f01, f03, or raw search."""
    exp = json_data.get('experiment', '')
    if exp == 'f01':
        return 'f01'
    elif exp == 'f03':
        return 'f03'
    return 'search'


def _load_results_from_json(filepath):
    """Load and normalise results from a pre-computed JSON file.

    Returns (normalised_results, metric_name, lower_is_better, dataset, domain).
    """
    with open(filepath) as f:
        data = json.load(f)

    source = _detect_source(data)
    metric_name = data.get('primary_metric', 'r2')
    lower_is_better = data.get('lower_is_better', False)
    dataset = data.get('dataset', 'unknown')
    domain = data.get('domain', 'molecular')

    # Extract the raw config list
    results_section = data.get('results', {})
    raw_configs = results_section.get('configs', [])
    if not raw_configs:
        raise ValueError(f"No 'results.configs' found in {filepath}")

    normalised = [_normalise_result(r, source) for r in raw_configs]
    return normalised, metric_name, lower_is_better, dataset, domain


# =========================================================================
# THREE-DIMENSIONAL PRUNING SIMULATION
# =========================================================================

def simulate_pruning(all_results, lower_is_better, thresholds=None):
    """Simulate adaptive pruning on completed search results.

    Replays the evaluation order and, for each threshold, prunes all three
    dimensions (models, importance methods, allocation methods) using the
    same _should_prune logic as find_best_hybrid._run_search.

    Returns dict keyed by threshold with pruning analysis.
    """
    if thresholds is None:
        thresholds = PRUNING_THRESHOLDS

    # Compute true best using comparable scores
    valid = [r for r in all_results if not np.isnan(r['score_mean'])]
    if not valid:
        return {}

    comparables = [_score_to_comparable(r['score_mean'], lower_is_better) for r in valid]
    true_best = valid[int(np.argmax(comparables))]
    true_best_model = true_best['model']
    true_best_comparable = max(comparables)

    # Unique dimension values
    model_names = sorted(set(r['model'] for r in all_results))
    alloc_names = sorted(set(r['allocation_method'] for r in all_results))
    imp_names = sorted(set(r['importance_method'] for r in all_results))

    # Per-dimension true best (ground truth)
    true_best_by_model = {}
    for m in model_names:
        m_results = [r for r in valid if r['model'] == m]
        if m_results:
            m_comp = [_score_to_comparable(r['score_mean'], lower_is_better)
                      for r in m_results]
            true_best_by_model[m] = m_results[int(np.argmax(m_comp))]['score_mean']

    true_best_by_alloc = {}
    for a in alloc_names:
        a_results = [r for r in valid if r['allocation_method'] == a]
        if a_results:
            a_comp = [_score_to_comparable(r['score_mean'], lower_is_better)
                      for r in a_results]
            true_best_by_alloc[a] = a_results[int(np.argmax(a_comp))]['score_mean']

    true_best_by_imp = {}
    for i in imp_names:
        i_results = [r for r in valid if r['importance_method'] == i]
        if i_results:
            i_comp = [_score_to_comparable(r['score_mean'], lower_is_better)
                      for r in i_results]
            true_best_by_imp[i] = i_results[int(np.argmax(i_comp))]['score_mean']

    analysis = {}

    for threshold in thresholds:
        # Pruning state
        pruned_models = set()
        pruned_allocs = set()
        pruned_imps = set()

        model_best_comp = {}
        alloc_best_comp = {}
        imp_best_comp = {}
        global_best_comp = -np.inf

        # Track when each dimension first gets fully seen
        first_model_sweep = False
        first_alloc_sweep = False
        first_imp_sweep = False

        pruned_at_step = {}  # dimension:name -> step
        surviving_results = []
        total_skipped = 0

        for step, r in enumerate(all_results):
            mn = r['model']
            an = r['allocation_method']
            im = r['importance_method']

            # Skip if any dimension is pruned
            if mn in pruned_models or an in pruned_allocs or im in pruned_imps:
                total_skipped += 1
                continue

            surviving_results.append(r)
            score = r['score_mean']
            if np.isnan(score):
                continue

            comp = _score_to_comparable(score, lower_is_better)

            # Update per-dimension bests
            if mn not in model_best_comp or comp > model_best_comp[mn]:
                model_best_comp[mn] = comp
            if an not in alloc_best_comp or comp > alloc_best_comp[an]:
                alloc_best_comp[an] = comp
            if im not in imp_best_comp or comp > imp_best_comp[im]:
                imp_best_comp[im] = comp
            if comp > global_best_comp:
                global_best_comp = comp

            # Check sweep completion
            if not first_model_sweep and len(model_best_comp) >= len(model_names):
                first_model_sweep = True
            if not first_alloc_sweep and len(alloc_best_comp) >= len(alloc_names):
                first_alloc_sweep = True
            if not first_imp_sweep and len(imp_best_comp) >= len(imp_names):
                first_imp_sweep = True

            # Prune models (after first full sweep)
            if first_model_sweep and global_best_comp > -np.inf:
                for m in model_names:
                    if m in pruned_models:
                        continue
                    best = model_best_comp.get(m, -np.inf)
                    if _should_prune(best, global_best_comp, threshold):
                        pruned_models.add(m)
                        pruned_at_step[f'model:{m}'] = step

            # Prune allocation methods
            if first_alloc_sweep and global_best_comp > -np.inf:
                for a in alloc_names:
                    if a in pruned_allocs:
                        continue
                    best = alloc_best_comp.get(a, -np.inf)
                    if _should_prune(best, global_best_comp, threshold):
                        pruned_allocs.add(a)
                        pruned_at_step[f'alloc:{a}'] = step

            # Prune importance methods
            if first_imp_sweep and global_best_comp > -np.inf:
                for i in imp_names:
                    if i in pruned_imps:
                        continue
                    best = imp_best_comp.get(i, -np.inf)
                    if _should_prune(best, global_best_comp, threshold):
                        pruned_imps.add(i)
                        pruned_at_step[f'imp:{i}'] = step

        # Find best from surviving results
        surviving_valid = [r for r in surviving_results
                          if not np.isnan(r['score_mean'])]
        if surviving_valid:
            surv_comp = [_score_to_comparable(r['score_mean'], lower_is_better)
                         for r in surviving_valid]
            pruned_best = surviving_valid[int(np.argmax(surv_comp))]
            pruned_best_score = pruned_best['score_mean']
            pruned_best_model = pruned_best['model']
        else:
            pruned_best_score = float('nan')
            pruned_best_model = None

        # Score delta (always: pruned - true, sign depends on metric)
        score_delta = pruned_best_score - true_best['score_mean']

        # Regret as comparable: how much worse is the pruned best?
        pruned_best_comp = (_score_to_comparable(pruned_best_score, lower_is_better)
                            if not np.isnan(pruned_best_score) else -np.inf)
        regret = true_best_comparable - pruned_best_comp

        analysis[threshold] = {
            # Models
            'pruned_models': sorted(pruned_models),
            'n_pruned_models': len(pruned_models),
            'n_models_total': len(model_names),
            'lost_models': {
                m: true_best_by_model[m]
                for m in pruned_models if m in true_best_by_model
            },
            # Allocations
            'pruned_allocs': sorted(pruned_allocs),
            'n_pruned_allocs': len(pruned_allocs),
            'n_allocs_total': len(alloc_names),
            'lost_allocs': {
                a: true_best_by_alloc[a]
                for a in pruned_allocs if a in true_best_by_alloc
            },
            # Importance methods
            'pruned_imps': sorted(pruned_imps),
            'n_pruned_imps': len(pruned_imps),
            'n_imps_total': len(imp_names),
            'lost_imps': {
                i: true_best_by_imp[i]
                for i in pruned_imps if i in true_best_by_imp
            },
            # Pruning timing
            'pruned_at_step': pruned_at_step,
            # Overall
            'total_evals': len(all_results),
            'evals_run': len(surviving_results),
            'evals_skipped': total_skipped,
            'speedup_pct': round(100 * total_skipped / len(all_results), 1)
                           if all_results else 0.0,
            'true_best_survives': true_best_model not in pruned_models,
            'true_best_score': true_best['score_mean'],
            'pruned_best_score': pruned_best_score,
            'score_delta': score_delta,
            'regret': regret,
            'pruned_best_model': pruned_best_model,
        }

    return analysis


# =========================================================================
# RUN FUNCTION
# =========================================================================

def run(data, config, output_dir, dataset_name, thresholds=None, **kwargs):
    """Run full grid search then simulate pruning at multiple thresholds."""
    if thresholds is None:
        thresholds = PRUNING_THRESHOLDS

    print("\n" + "=" * 70)
    print("EXPERIMENT F02: ADAPTIVE PRUNING ANALYSIS")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(data['labels'])
    task_type = data.get('task_type', 'regression')
    models = get_model_factories(config['models'], task_type=task_type,
                                 n_samples=n_samples)
    n_folds = config['n_splits']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')

    budgets = config['budgets']

    model_instances = {name: factory() for name, factory in models.items()}

    total_configs = (len(model_instances) * len(budgets)
                     * len(config['allocations']) * len(config['importance_methods']))
    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Running full grid: {total_configs} configs")
    print(f"  Then simulating pruning at thresholds: {thresholds}\n", flush=True)

    # Run full search (no pruning — we need all results for simulation)
    search = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        models=model_instances,
        budgets=budgets,
        allocation_methods=config['allocations'],
        importance_methods=config['importance_methods'],
        internal_evaluators=INTERNAL_EVALUATORS,
        n_folds=n_folds,
        augmentations=data.get('augmentations'),
        verbose=True,
        cache=True,
        dataset_name=dataset_name,
        tier='general',
        adaptive_pruning=False,  # Disable pruning to get complete results
        metric=metric_name,
    )

    all_results = search.all_results
    print(f"\n  Full search: {len(all_results)} results collected")

    # Extract top 3 base reps for reporting
    top3 = extract_top3_base_reps(search)
    if top3:
        print(f"  Top 3 base reps: {[(r, f'{s:.4f}') for r, s in top3]}")

    # Normalise results
    normalised = [_normalise_result(r, 'search') for r in all_results]

    # Simulate pruning
    return _run_simulation(normalised, metric_name, lower_is_better,
                           thresholds, data, dataset_name, config)


def run_from_json(filepath, thresholds=None, output_dir='tests/results'):
    """Run pruning simulation on pre-computed results from f01 or f03."""
    if thresholds is None:
        thresholds = PRUNING_THRESHOLDS

    print("\n" + "=" * 70)
    print("EXPERIMENT F02: ADAPTIVE PRUNING ANALYSIS (from pre-computed)")
    print("=" * 70)
    print(f"  Source: {filepath}")

    normalised, metric_name, lower_is_better, dataset, domain = \
        _load_results_from_json(filepath)

    print(f"  Dataset: {dataset} ({domain})")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Loaded {len(normalised)} results")
    print(f"  Simulating pruning at thresholds: {thresholds}\n", flush=True)

    # Build minimal data dict for make_structured_output
    data = {
        'domain': domain,
        'labels': np.zeros(0),  # unknown
        'reps': {},
        'task_type': 'classification' if metric_name in ('accuracy', 'f1_weighted') else 'regression',
    }

    config = {'source_file': filepath}

    return _run_simulation(normalised, metric_name, lower_is_better,
                           thresholds, data, dataset, config)


def _run_simulation(normalised, metric_name, lower_is_better,
                    thresholds, data, dataset_name, config):
    """Core simulation + output (shared between run() and run_from_json())."""
    pruning_analysis = simulate_pruning(normalised, lower_is_better, thresholds)

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("\n" + "-" * 70)
    print("  PRUNING SIMULATION RESULTS")
    print("-" * 70)

    header = (f"  {'Threshold':>10} {'Models':>10} {'Allocs':>10} {'Imps':>10} "
              f"{'Skipped':>9} {'Speedup':>9} {'Best OK?':>9} {'Regret':>10}")
    print(header)
    print("  " + "-" * 80)

    for t in thresholds:
        a = pruning_analysis[t]
        ok = "YES" if a['true_best_survives'] else "NO"
        print(f"  {t:>10.1f}"
              f" {a['n_pruned_models']:>4}/{a['n_models_total']}"
              f" {a['n_pruned_allocs']:>4}/{a['n_allocs_total']}"
              f" {a['n_pruned_imps']:>4}/{a['n_imps_total']}"
              f" {a['evals_skipped']:>9}"
              f" {a['speedup_pct']:>8.1f}%"
              f" {ok:>9}"
              f" {a['regret']:>10.4f}")

    # Detail per threshold — what got pruned and when
    for t in thresholds:
        a = pruning_analysis[t]
        pruned_any = (a['pruned_models'] or a['pruned_allocs'] or a['pruned_imps'])
        if not pruned_any:
            continue

        print(f"\n  Threshold {t}:")
        if a['pruned_models']:
            print(f"    Pruned models: {a['pruned_models']}")
            for m, score in a['lost_models'].items():
                step = a['pruned_at_step'].get(f'model:{m}', '?')
                print(f"      {m}: best {metric_name}={score:.4f}, "
                      f"pruned at step {step}/{a['total_evals']}")
        if a['pruned_allocs']:
            print(f"    Pruned allocations: {a['pruned_allocs']}")
            for al, score in a['lost_allocs'].items():
                step = a['pruned_at_step'].get(f'alloc:{al}', '?')
                print(f"      {al}: best {metric_name}={score:.4f}, "
                      f"pruned at step {step}/{a['total_evals']}")
        if a['pruned_imps']:
            print(f"    Pruned importance methods: {a['pruned_imps']}")
            for im, score in a['lost_imps'].items():
                step = a['pruned_at_step'].get(f'imp:{im}', '?')
                print(f"      {im}: best {metric_name}={score:.4f}, "
                      f"pruned at step {step}/{a['total_evals']}")

    # Pareto frontier: threshold with best speedup where regret < 0.01
    safe_thresholds = [
        (t, pruning_analysis[t])
        for t in thresholds
        if pruning_analysis[t]['regret'] < 0.01
        and pruning_analysis[t]['true_best_survives']
    ]
    if safe_thresholds:
        best_safe = max(safe_thresholds, key=lambda x: x[1]['speedup_pct'])
        print(f"\n  RECOMMENDATION: threshold={best_safe[0]} "
              f"(speedup={best_safe[1]['speedup_pct']:.1f}%, "
              f"regret={best_safe[1]['regret']:.4f}, best survives)")
    else:
        print(f"\n  WARNING: No threshold achieves <1% regret while keeping best config")
        # Show the least-regret option
        min_regret = min(thresholds, key=lambda t: pruning_analysis[t]['regret'])
        a = pruning_analysis[min_regret]
        print(f"  Least regret: threshold={min_regret} "
              f"(regret={a['regret']:.4f}, speedup={a['speedup_pct']:.1f}%)")

    # =========================================================================
    # STRUCTURED OUTPUT
    # =========================================================================
    results_out = {
        'n_input_results': len(normalised),
        'thresholds_tested': thresholds,
        'pruning_analysis': pruning_analysis,
    }

    return make_structured_output(
        experiment_id='f02',
        dataset=dataset_name,
        data=data,
        config=convert_numpy(config) if isinstance(config, dict) else {'source': str(config)},
        results=results_out,
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser('F02: Adaptive Pruning Analysis',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--from-json', type=str, default=None,
                        help='Path to pre-computed results JSON from f01/f03')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=PRUNING_THRESHOLDS,
                        help='Pruning thresholds to test (default: 0.3-0.7)')
    parser.add_argument('--framework', action='store_true',
                        help='Use curated FRAMEWORK_TASKS rep lists')
    args = parser.parse_args()

    if args.from_json:
        # Run from pre-computed JSON
        results = run_from_json(args.from_json, thresholds=args.thresholds,
                                output_dir=args.output)
        # Extract dataset name from the JSON for saving
        with open(args.from_json) as f:
            src = json.load(f)
        ds = src.get('dataset', 'unknown')
        save_experiment_results(results, args.output, 'f02_adaptive_pruning', ds)
    else:
        # Run own search
        config = get_config(args)
        print(f"Mode: {args.mode}")
        print(f"Dataset: {args.dataset}")
        print(f"Reps: {args.reps}")

        rep_preset, rep_names = parse_rep_args(args)
        data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                         delta=getattr(args, 'delta', False),
                         augment=getattr(args, 'augment', False))
        results = run(data, config, args.output, args.dataset,
                      thresholds=args.thresholds)
        save_experiment_results(results, args.output, 'f02_adaptive_pruning',
                                args.dataset)
