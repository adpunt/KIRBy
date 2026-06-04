#!/usr/bin/env python3
"""C06: Within-Family Complexity Gradient — vary capacity within model families.

Paper: Complexity
Conclusion: Within-family capacity gradient confirms complexity matching.
Dependencies: None (standalone).

For each variant, runs find_best_hybrid() with that variant as the sole
downstream model. The library searches importance methods, allocation methods,
and budgets to find the variant-optimal hybrid recipe. Optimal budget per
variant is the budget where val score peaks. Spearman correlation between
capacity index and optimal budget tests whether capacity and budget are coupled.

Train scores / bias-variance gap are not collected here (find_best_hybrid does
not return train scores). Moved to a dedicated experiment if/when needed.
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
    get_model_family_variants,
    find_best_hybrid, get_domain_metric,
    INTERNAL_EVALUATORS,
)
from kirby.find_best_hybrid import _resolve_models


# Map c06 family labels to canonical names in DEFAULT_MODELS. List-form
# _resolve_models is the only way to trigger _max_samples + _DOMAIN_MODEL_SKIPS
# + _DATASET_STRATEGY_SKIPS; the dict form is a passthrough.
_FAMILY_TO_CANONICAL = {
    'RF': 'rf', 'Ridge': 'ridge', 'kNN': 'knn', 'MLP': 'mlp',
    'XGBoost': 'xgboost', 'SVM': 'svm', 'GP': 'gp',
}


def run(data, config, output_dir, dataset_name, families=None):
    """Exp 6: vary complexity within each model family."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: WITHIN-FAMILY MODEL COMPLEXITY GRADIENT")
    print("=" * 70)

    task_type = data.get('task_type')
    if task_type != 'regression':
        # get_model_family_variants() hardcodes regressor variants.
        print(f"  Skipping: c06 is regression-only (task_type={task_type}).")
        return {'error': 'regression_only', 'task_type': task_type}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    budgets = config['budgets']
    search_speed = config['search_speed']
    n_folds = config['n_splits']
    n_samples = len(labels)

    # Canonical budget filter (matches find_best_hybrid.py:3934).
    eff_budgets = sorted(b for b in budgets if b <= n_samples)
    dropped = sorted(set(budgets) - set(eff_budgets))
    if dropped:
        print(f"  Dropped budgets > n_samples={n_samples}: {dropped}")

    # The capacity<->budget correlation is undefined unless several DISTINCT
    # budgets are viable. On small datasets a coarse grid (e.g. [50,500,...])
    # collapses to a single value <= n_samples, pinning every variant's optimal
    # budget and forcing a NaN correlation. Guarantee a data-relative spread of
    # at least 4 distinct budgets below n_samples (and below the feature dim).
    # See COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
    max_dim = max((v.shape[1] for v in reps.values()), default=n_samples)
    viable_cap = min(n_samples, max_dim)
    if len(eff_budgets) < 4:
        data_rel = sorted({max(5, int(viable_cap * f))
                           for f in (0.05, 0.1, 0.2, 0.35, 0.5)})
        eff_budgets = sorted({b for b in (eff_budgets + data_rel)
                              if 5 <= b <= viable_cap})
        print(f"  Small-data budget grid widened to {eff_budgets}")

    all_families = get_model_family_variants()
    if families:
        all_families = {k: v for k, v in all_families.items() if k in families}

    # Compute aug_strategy first so we can pass it to the family gate
    # (matches the strategy each per-variant find_best_hybrid call will use).
    augs = data.get('augmentations')
    aug_strategy = 'uncapped' if augs else 'none'

    # Family-level canonical gate via _resolve_models (LIST form).
    # Replaces the old hardcoded `_FAMILY_MAX_SAMPLES = {'GP': 2000, 'SVM': 10000}`
    # which diverged from DEFAULT_MODELS (gp: 1000, svm: 10000) and didn't
    # inherit domain/dataset skips.
    for fam_name in list(all_families.keys()):
        canonical = _FAMILY_TO_CANONICAL.get(fam_name)
        if canonical is None:
            continue  # unknown family, leave alone
        # Pass strategy so _DATASET_STRATEGY_SKIPS / _DOMAIN_STRATEGY_SKIPS
        # are applied (e.g., 'gp' skipped for flab_thermo_tresanco/uncapped).
        resolved = _resolve_models(
            [canonical], task_type='regression', speed=search_speed,
            n_samples=n_samples, dataset_name=dataset_name,
            strategy=aug_strategy)
        if not resolved:
            print(f"  Skipping {fam_name}: gated by _resolve_models "
                  f"(canonical={canonical}, n_samples={n_samples}, "
                  f"strategy={aug_strategy})")
            del all_families[fam_name]

    # Fetch precompute's importance_cache once for the dataset and reuse
    # across all per-variant find_best_hybrid calls. Skips the per-fold
    # importance rebuild (lightgbm_gain × n_reps × n_folds) inside each
    # variant search. Splits align because c06 passes seed=42 + same
    # n_folds + same task_type as precompute. If precompute SearchResult
    # is missing/empty, falls back to building importance from scratch
    # (find_best_hybrid handles `importance_cache=None` as the no-op default).
    _precompute_importance = None
    try:
        from experiments._shared import get_precomputed_search
        _pc_search = get_precomputed_search(
            reps, labels, groups,
            # Fetch the cache that matches the actual augmentation condition
            # we'll be using (otherwise we may fall through to a from-scratch
            # find_best_hybrid run when the 'none' cache doesn't exist).
            aug_condition=aug_strategy, augmentations=augs,
            speed=search_speed,
            task_type='regression', metric=metric_name,
            dataset_name=dataset_name, tier='general',
        )
        _pc_imp = getattr(_pc_search, 'importance_cache', None) or {}
        if _pc_imp:
            _precompute_importance = _pc_imp
            print(f"  Reusing precompute importance_cache: "
                  f"{len(_precompute_importance)} (fold, method) entries")
    except Exception as _e:
        print(f"  Could not load precompute importance_cache: {_e}")

    results = {'families': {}, 'search_speed': search_speed}

    for fam_name, variants in all_families.items():
        print(f"\n  Family: {fam_name} ({len(variants)} variants, "
              f"search_speed={search_speed})")
        fam_result = {'heatmap': {}, 'optimal_budget': {}}

        for var_name, var_factory, capacity_idx in variants:
            print(f"\n    Variant: {var_name} (capacity={capacity_idx})")
            fam_result['heatmap'][var_name] = {}

            # Per-variant search: this variant is the SOLE downstream model.
            # find_best_hybrid searches its full grid (alloc × imp × budget)
            # against this specific capacity. seed=42 matches precompute so
            # CV splits align with any reused cache state.
            try:
                search = find_best_hybrid(
                    base_reps=reps,
                    labels=labels,
                    groups=groups,
                    models={var_name: var_factory()},
                    budgets=eff_budgets,
                    task_type='regression',
                    metric=metric_name,
                    speed=search_speed,
                    seed=42,
                    n_folds=n_folds,
                    cache=True,
                    # Use real dataset_name + condition_suffix so the
                    # _DATASET_STRATEGY_SKIPS lookup still matches (the
                    # mangled dataset_name would never hit those entries).
                    dataset_name=dataset_name,
                    condition_suffix=f"c06_{fam_name}_{var_name}",
                    tier='c06_variant',
                    augmentations=augs,
                    augmentation_strategy=aug_strategy,
                    internal_evaluators=INTERNAL_EVALUATORS,
                    verbose=False,
                    importance_cache=_precompute_importance,
                )
            except Exception as e:
                print(f"      search failed: {e}")
                fam_result['optimal_budget'][var_name] = {
                    'budget': None, 'r2': float('nan'),
                    'capacity_index': capacity_idx,
                }
                continue

            per_budget = search.best_by_model_budget.get(var_name, {})
            per_budget_val = {}
            # Iterate keys actually present (find_best_hybrid may append n_samples
            # as an at-n boundary point even when it wasn't in eff_budgets).
            for budget in sorted(per_budget.keys()):
                best = per_budget.get(budget)
                if best is None:
                    continue
                bp = best['blueprint']
                val_score = best['score_mean']
                val_std = best.get('score_std', float('nan'))
                n_feat = best.get('n_features', 0)
                fam_result['heatmap'][var_name][budget] = {
                    'val_r2': val_score,
                    'val_std': val_std,
                    'n_features': n_feat,
                    'importance': bp.importance_method,
                    'allocation': bp.allocation_method,
                }
                per_budget_val[budget] = (val_score, bp)
                print(f"      b={budget}: {metric_name}={val_score:.4f}±{val_std:.3f} "
                      f"[{bp.importance_method}/{bp.allocation_method}]")

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
                    'budget': None, 'r2': float('nan'),
                    'capacity_index': capacity_idx,
                }

        # Correlation: capacity_index vs optimal_budget within family.
        capacities = []
        opt_budgets = []
        for var_name, info in fam_result['optimal_budget'].items():
            if info.get('budget') is not None:
                capacities.append(info['capacity_index'])
                opt_budgets.append(info['budget'])

        # spearmanr is NaN unless BOTH variables vary. On small/low-signal data
        # every capacity variant can optimise at the same budget -> no variance
        # -> NaN. Distinguish that genuine "no budget variance" case from a real
        # measured correlation so the result is interpretable (not a bare NaN).
        n_distinct_budgets = len(set(opt_budgets))
        if len(capacities) >= 3 and n_distinct_budgets >= 2:
            rho, pval = spearmanr(capacities, opt_budgets)
            fam_result['capacity_budget_correlation'] = {
                'rho': float(rho), 'p': float(pval),
                'n_distinct_budgets': n_distinct_budgets,
                'status': 'measured',
            }
            print(f"\n  >> {fam_name}: capacity vs optimal_budget: "
                  f"rho={rho:.3f}, p={pval:.3f}")
        else:
            status = ('insufficient_variants' if len(capacities) < 3
                      else 'no_budget_variance')
            fam_result['capacity_budget_correlation'] = {
                'rho': float('nan'), 'p': float('nan'),
                'n_distinct_budgets': n_distinct_budgets,
                'n_variants': len(capacities),
                'status': status,
            }
            print(f"\n  >> {fam_name}: capacity vs optimal_budget UNDEFINED "
                  f"({status}: {n_distinct_budgets} distinct budget(s) over "
                  f"{len(capacities)} variants)")

        results['families'][fam_name] = fam_result

    # Summary across all families.
    print("\n  SUMMARY: optimal budget by variant")
    for fam_name, fam_result in results['families'].items():
        for var_name, info in fam_result['optimal_budget'].items():
            b = info.get('budget', '?')
            r2 = info.get('r2', float('nan'))
            imp = info.get('importance', '?')
            print(f"    {fam_name}/{var_name}: optimal_budget={b}, "
                  f"{metric_name}={r2:.4f} [{imp}]")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C06: Within-Family Complexity Gradient',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--families', type=str, nargs='+', default=None,
                        help='Model families to test (RF, Ridge, kNN, MLP, XGBoost, SVM, GP)')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, families=args.families)
    save_experiment_results(results, args.output, 'c06_within_family_gradient', args.dataset)
