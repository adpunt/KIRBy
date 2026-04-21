#!/usr/bin/env python3
"""C06: Within-Family Complexity Gradient — vary capacity within model families.

Paper: Complexity
Conclusion: Within-family capacity gradient confirms complexity matching.
Dependencies: None (standalone).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_family_variants, evaluate_cv_with_train,
    find_best_hybrid, get_domain_metric,
    extract_top3_base_reps,
    INTERNAL_EVALUATORS,
)


def run(data, config, output_dir, dataset_name, families=None):
    """Exp 6: Vary complexity within each model family to show optimal budget shifts.

    For each model variant, runs find_best_hybrid() with that variant as the
    sole model. The library searches importance methods, allocation methods, and
    budgets to find the true best hybrid for that specific model capacity.
    Then we also collect train R² for bias-variance analysis (Exp 9d).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: WITHIN-FAMILY MODEL COMPLEXITY GRADIENT")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    budgets = config['budgets']
    search_speed = config['search_speed']
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    all_families = get_model_family_variants()
    if families:
        all_families = {k: v for k, v in all_families.items() if k in families}

    n_samples = len(labels)

    # Size guards for expensive model families
    _FAMILY_MAX_SAMPLES = {'GP': 2000, 'SVM': 10000}
    for fam_name, max_n in _FAMILY_MAX_SAMPLES.items():
        if fam_name in all_families and n_samples > max_n:
            print(f"  Skipping {fam_name}: n_samples={n_samples} > {max_n}")
            del all_families[fam_name]

    results = {'families': {}, 'search_speed': search_speed}

    for fam_name, variants in all_families.items():
        print(f"\n  Family: {fam_name} ({len(variants)} variants, search_speed={search_speed})")
        fam_result = {'heatmap': {}, 'optimal_budget': {}, 'train_val_gap': {}}

        for var_name, var_factory, capacity_idx in variants:
            print(f"\n    Variant: {var_name} (capacity={capacity_idx})")
            fam_result['heatmap'][var_name] = {}

            # Run find_best_hybrid with this variant as the sole model
            augs = data.get('augmentations')
            try:
                search = find_best_hybrid(
                    reps, labels, groups=groups,
                    models={var_name: var_factory()},
                    speed=search_speed,
                    budgets=budgets,
                    augmentations=augs,
                    augmentation_strategy='capped',
                    internal_evaluators=INTERNAL_EVALUATORS,
                    metric=metric_name,
                    task_type=data['task_type'],
                    verbose=False,
                    cache=True,
                    dataset_name=f"{dataset_name}_{var_name}",
                    tier='general',
                )
            except Exception as e:
                print(f"      find_best_hybrid failed: {e}")
                fam_result['optimal_budget'][var_name] = {
                    'budget': None, 'r2': float('nan'), 'capacity_index': capacity_idx,
                }
                continue

            top3 = extract_top3_base_reps(search, model=var_name)
            if top3:
                print(f"      Top-3 base reps (by {var_name}): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

            # Extract best score at each budget from all_results
            for budget in budgets:
                budget_results = [r for r in search.all_results
                                  if r['budget'] == budget and not np.isnan(r['score_mean'])]
                if budget_results:
                    best_at_budget = (min if lower_is_better else max)(budget_results, key=lambda r: r['score_mean'])
                    bp = best_at_budget['blueprint']

                    # Get train R² for bias-variance analysis
                    try:
                        val_r2, val_std, train_r2, train_std, n_feat = evaluate_cv_with_train(
                            reps, labels, groups, var_factory, n_folds,
                            allocation_method=bp.allocation_method,
                            importance_method=bp.importance_method,
                            budget=budget,
                            mode=mode,
                            metric_fn=metric_fn,
                        )
                    except Exception:
                        val_r2 = best_at_budget['score_mean']
                        val_std = best_at_budget['score_std']
                        train_r2, train_std = float('nan'), float('nan')
                        n_feat = best_at_budget.get('n_features', 0)

                    fam_result['heatmap'][var_name][budget] = {
                        'val_r2': val_r2, 'val_std': val_std,
                        'train_r2': train_r2, 'train_std': train_std,
                        'n_features': n_feat,
                        'importance': bp.importance_method,
                        'allocation': bp.allocation_method,
                    }
                    gap = train_r2 - val_r2 if not np.isnan(train_r2) else float('nan')
                    fam_result['train_val_gap'][f"{var_name}_b{budget}"] = gap

                    print(f"      b={budget}: {metric_name}={val_r2:.4f}±{val_std:.3f}, "
                          f"train_{metric_name}={train_r2:.4f}, gap={gap:.3f} "
                          f"[{bp.importance_method}/{bp.allocation_method}]")

            # Optimal budget comes from find_best_hybrid's own search
            # Single-model search: var_name is the only model
            best = search.best_by_model.get(var_name)
            if best:
                fam_result['optimal_budget'][var_name] = {
                    'budget': best['blueprint'].budget,
                    'r2': best['score_mean'],
                    'capacity_index': capacity_idx,
                    'importance': best['blueprint'].importance_method,
                    'allocation': best['blueprint'].allocation_method,
                }
                print(f"      >> optimal: b={best['blueprint'].budget}, {metric_name}={best['score_mean']:.4f} "
                      f"[{best['blueprint'].importance_method}/{best['blueprint'].allocation_method}]")
            else:
                fam_result['optimal_budget'][var_name] = {
                    'budget': None, 'r2': float('nan'), 'capacity_index': capacity_idx,
                }

        # Correlation: capacity_index vs optimal_budget
        capacities = []
        opt_budgets = []
        for var_name, info in fam_result['optimal_budget'].items():
            if info.get('budget') is not None:
                capacities.append(info['capacity_index'])
                opt_budgets.append(info['budget'])

        if len(capacities) >= 3:
            rho, pval = spearmanr(capacities, opt_budgets)
            fam_result['capacity_budget_correlation'] = {'rho': float(rho), 'p': float(pval)}
            print(f"\n  >> {fam_name}: capacity vs optimal_budget: rho={rho:.3f}, p={pval:.3f}")
        else:
            fam_result['capacity_budget_correlation'] = None

        results['families'][fam_name] = fam_result

    # Summary across all families
    print("\n  SUMMARY: optimal budget by variant")
    for fam_name, fam_result in results['families'].items():
        for var_name, info in fam_result['optimal_budget'].items():
            b = info.get('budget', '?')
            r2 = info.get('r2', float('nan'))
            imp = info.get('importance', '?')
            print(f"    {fam_name}/{var_name}: optimal_budget={b}, {metric_name}={r2:.4f} [{imp}]")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C06: Within-Family Complexity Gradient',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--families', type=str, nargs='+', default=None,
                        help='Model families to test (RF, Ridge, kNN, MLP, XGBoost)')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, families=args.families)
    save_experiment_results(results, args.output, 'c06_within_family_gradient', args.dataset)
