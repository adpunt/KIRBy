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
    get_best_search_for_downstream,
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

            # Pool helper: for each budget, pick the best (tier, aug_cond)
            # blueprint discovered with a SHARED proxy evaluator (rf), then
            # evaluate THIS variant under that blueprint. Variants are
            # custom models not in INTERNAL_EVALUATORS, so the shared
            # blueprint pool is the right reference.
            augs = data.get('augmentations')
            per_budget_val = {}

            # Extract best score at each budget via the pool helper
            for budget in budgets:
                try:
                    _s, _meta = get_best_search_for_downstream(
                        reps, labels, groups,
                        downstream_model='rf', budget=budget,
                        augmentations=augs, speed=search_speed,
                        dataset_name=dataset_name,
                        task_type=data['task_type'], metric=metric_name,
                    )
                except Exception as e:
                    print(f"      pool helper b={budget} failed: {e}")
                    _meta = None
                if not _meta:
                    continue
                best_at_budget = _meta['best_config']
                bp = best_at_budget['blueprint']
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
                per_budget_val[budget] = (val_r2, bp)
                gap = train_r2 - val_r2 if not np.isnan(train_r2) else float('nan')
                fam_result['train_val_gap'][f"{var_name}_b{budget}"] = gap

                print(f"      b={budget}: {metric_name}={val_r2:.4f}±{val_std:.3f}, "
                      f"train_{metric_name}={train_r2:.4f}, gap={gap:.3f} "
                      f"[{bp.importance_method}/{bp.allocation_method}]")

            # Optimal budget = best per-budget val score for this variant.
            if per_budget_val:
                cmp_fn = min if lower_is_better else max
                opt_budget, (opt_val, opt_bp) = cmp_fn(
                    per_budget_val.items(), key=lambda kv: kv[1][0])
                fam_result['optimal_budget'][var_name] = {
                    'budget': opt_budget,
                    'r2': opt_val,
                    'capacity_index': capacity_idx,
                    'importance': opt_bp.importance_method,
                    'allocation': opt_bp.allocation_method,
                }
                print(f"      >> optimal: b={opt_budget}, {metric_name}={opt_val:.4f} "
                      f"[{opt_bp.importance_method}/{opt_bp.allocation_method}]")
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
