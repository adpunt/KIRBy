#!/usr/bin/env python3
"""C12: Adaptive pruning analysis — what would pruning cut, and at what cost?

Paper: Complexity
Conclusion: Whether aggressive pruning can halve search time without losing best configs.
Dependencies: None (standalone, reuses cached search results from find_best_hybrid).

Runs a full grid search (identical to C02), then simulates adaptive pruning post-hoc
at multiple thresholds. Reports: which models would be pruned, how early, and whether
the best overall config survives each threshold.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories,
    find_best_hybrid,
)


PRUNING_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def simulate_pruning(all_results, thresholds=None):
    """Simulate adaptive pruning on completed search results.

    Replays the evaluation order and, for each threshold, tracks:
    - When each model would first become prune-eligible
    - Which models get pruned and at what step
    - Whether the globally best config survives

    Returns dict keyed by threshold with pruning analysis.
    """
    if thresholds is None:
        thresholds = PRUNING_THRESHOLDS

    # Find the true best config
    valid = [r for r in all_results if not np.isnan(r['score_mean'])]
    if not valid:
        return {}
    true_best = max(valid, key=lambda r: r['score_mean'])
    true_best_model = true_best['model']
    true_best_score = true_best['r2_mean']

    # Per-model best (ground truth)
    model_names = sorted(set(r['model'] for r in all_results))
    true_best_by_model = {}
    for m in model_names:
        m_results = [r for r in valid if r['model'] == m]
        if m_results:
            true_best_by_model[m] = max(m_results, key=lambda r: r['score_mean'])

    analysis = {}

    for threshold in thresholds:
        pruned_models = set()
        model_best_scores = {}
        global_best_score = -np.inf
        first_sweep_done = False

        pruned_at_step = {}  # model -> step when pruned
        surviving_results = []
        total_skipped = 0

        for step, r in enumerate(all_results):
            model_name = r['model']

            # Would this be skipped?
            if model_name in pruned_models:
                total_skipped += 1
                continue

            surviving_results.append(r)
            score = r['score_mean']
            if not np.isnan(score):
                if model_name not in model_best_scores or score > model_best_scores[model_name]:
                    model_best_scores[model_name] = score
                if score > global_best_score:
                    global_best_score = score

            # Check for pruning after first sweep (every model seen at least once)
            if not first_sweep_done:
                if len(model_best_scores) >= len(model_names):
                    first_sweep_done = True
                continue

            if global_best_score > 0:
                for mn in model_names:
                    if mn in pruned_models:
                        continue
                    best = model_best_scores.get(mn, -np.inf)
                    if best < threshold * global_best_score:
                        pruned_models.add(mn)
                        pruned_at_step[mn] = step

        # Find best from surviving results only
        surviving_valid = [r for r in surviving_results if not np.isnan(r['score_mean'])]
        if surviving_valid:
            pruned_best = max(surviving_valid, key=lambda r: r['score_mean'])
            pruned_best_score = pruned_best['r2_mean']
            pruned_best_model = pruned_best['model']
        else:
            pruned_best_score = float('nan')
            pruned_best_model = None

        # Per-model: what survives?
        surviving_best_by_model = {}
        for m in model_names:
            m_results = [r for r in surviving_valid if r['model'] == m]
            if m_results:
                surviving_best_by_model[m] = max(r['score_mean'] for r in m_results)

        analysis[threshold] = {
            'pruned_models': sorted(pruned_models),
            'pruned_at_step': pruned_at_step,
            'n_pruned': len(pruned_models),
            'n_models_total': len(model_names),
            'total_evals': len(all_results),
            'evals_run': len(surviving_results),
            'evals_skipped': total_skipped,
            'speedup_pct': round(100 * total_skipped / len(all_results), 1),
            'true_best_survives': true_best_model not in pruned_models,
            'true_best_score': true_best_score,
            'pruned_best_score': pruned_best_score,
            'score_delta': pruned_best_score - true_best_score,
            'pruned_best_model': pruned_best_model,
            'surviving_best_by_model': surviving_best_by_model,
            'lost_models': {
                m: true_best_by_model[m]['r2_mean']
                for m in pruned_models if m in true_best_by_model
            },
        }

    return analysis


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run full grid search then simulate pruning at multiple thresholds."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C12: ADAPTIVE PRUNING ANALYSIS")
    print("=" * 70)

    models = get_model_factories(config['models'])
    budgets = config['budgets']
    allocation_methods = config['allocations']
    importance_methods = config['importance_methods']
    n_folds = config['n_splits']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    total_dim = sum(v.shape[1] for v in reps.values())
    full_budgets = budgets + [total_dim]

    model_instances = {name: factory() for name, factory in models.items()}

    total_configs = (len(model_instances) * len(full_budgets)
                     * len(allocation_methods) * len(importance_methods))
    print(f"  Running full grid: {total_configs} configs")
    print(f"  Then simulating pruning at thresholds: {PRUNING_THRESHOLDS}\n")

    # Run full search (uses cache if available)
    augs = data.get('augmentations')
    search = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        models=model_instances,
        budgets=full_budgets,
        allocation_methods=allocation_methods,
        importance_methods=importance_methods,
        n_folds=n_folds,
        augmentations=augs,
        verbose=True,
        cache=True,
    )

    all_results = search.all_results
    print(f"\n  Full search: {len(all_results)} results collected")

    # Simulate pruning
    print(f"\n  Simulating adaptive pruning...")
    pruning_analysis = simulate_pruning(all_results, PRUNING_THRESHOLDS)

    # Print summary
    print("\n" + "-" * 70)
    print("  PRUNING SIMULATION RESULTS")
    print("-" * 70)

    header = f"  {'Threshold':>10} {'Pruned':>8} {'Skipped':>9} {'Speedup':>9} {'Best OK?':>9} {'Score Δ':>10}"
    print(header)
    print("  " + "-" * 58)

    for t in PRUNING_THRESHOLDS:
        a = pruning_analysis[t]
        ok = "YES" if a['true_best_survives'] else "NO"
        delta = a['score_delta']
        print(f"  {t:>10.1f} {a['n_pruned']:>5}/{a['n_models_total']}"
              f" {a['evals_skipped']:>9} {a['speedup_pct']:>8.1f}%"
              f" {ok:>9} {delta:>+10.4f}")

    # Detail per threshold
    for t in PRUNING_THRESHOLDS:
        a = pruning_analysis[t]
        if a['pruned_models']:
            print(f"\n  Threshold {t}: pruned {a['pruned_models']}")
            for m, score in a['lost_models'].items():
                step = a['pruned_at_step'].get(m, '?')
                print(f"    {m}: best R2={score:.4f}, pruned at step {step}/{a['total_evals']}")

    # Best by model comparison
    print(f"\n  BEST BY MODEL (full vs pruned at 0.5):")
    if 0.5 in pruning_analysis:
        a = pruning_analysis[0.5]
        model_names = sorted(set(r['model'] for r in all_results))
        for m in model_names:
            full = [r for r in all_results if r['model'] == m and not np.isnan(r['score_mean'])]
            full_best = max(r['score_mean'] for r in full) if full else float('nan')
            pruned_best = a['surviving_best_by_model'].get(m, float('nan'))
            status = "PRUNED" if m in a['pruned_models'] else "kept"
            print(f"    {m:<12} full={full_best:>8.4f}  pruned={pruned_best:>8.4f}  [{status}]")

    return {
        'full_search_n_results': len(all_results),
        'pruning_analysis': pruning_analysis,
        'thresholds_tested': PRUNING_THRESHOLDS,
    }


if __name__ == '__main__':
    parser = create_experiment_parser('C12: Adaptive Pruning Analysis',
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
    save_experiment_results(results, args.output, 'c12_adaptive_pruning', args.dataset)
