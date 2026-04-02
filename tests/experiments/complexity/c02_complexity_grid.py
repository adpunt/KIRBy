#!/usr/bin/env python3
"""C02: Model-budget complexity grid -- tests optimal pairing hypothesis (CORE).

Paper: Complexity
Conclusion: Different models have different optimal representation budgets.
Dependencies: None (standalone).

Central experiment. Delegates core grid search to find_best_hybrid() with
configurable models, budgets, allocations, and importance methods.
Also computes PCA baselines, builds model x budget heatmap, and tracks
importance method wins.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split, _cv_split,
    find_best_hybrid, get_domain_metric,
    extract_top3_base_reps,
    INTERNAL_EVALUATORS,
)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run model x budget complexity grid via find_best_hybrid."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C02: COMPLEXITY MATCHING GRID")
    print("=" * 70)

    models = get_model_factories(config['models'],
                                    task_type=data.get('task_type', 'regression'),
                                    n_samples=len(data['labels']))
    budgets = config['budgets']
    allocation_methods = config['allocations']
    importance_methods = config['importance_methods']
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    total_dim = sum(v.shape[1] for v in reps.values())
    # Include larger budgets up to total dim for dimension-matched comparison
    full_budgets = sorted(set(budgets + [1000, 1500, 2000, total_dim]))

    # Convert model factories to instances for the library
    model_instances = {name: factory() for name, factory in models.items()}

    augs = data.get('augmentations')

    # --- Top-tier reps only (from DATASET_REPS, if available) ---
    from experiments.complexity.complexity_helpers import DATASET_REPS
    ds_reps_config = DATASET_REPS.get(dataset_name, {})
    top_tier_names = ds_reps_config.get('top_tier', [])
    top_tier_reps = {k: v for k, v in reps.items() if k in top_tier_names}

    search_top_tier = None
    if len(top_tier_reps) >= 2:
        print(f"\n  --- Top-tier reps only ({len(top_tier_reps)}/{len(reps)}: "
              f"{list(top_tier_reps.keys())}) ---")
        search_top_tier = find_best_hybrid(
            base_reps=top_tier_reps,
            labels=labels,
            groups=groups,
            models=model_instances,
            budgets=full_budgets,
            allocation_methods=allocation_methods,
            importance_methods=importance_methods,
            n_folds=n_folds,
            augmentations=None,
            internal_evaluators=INTERNAL_EVALUATORS,
            verbose=True,
            cache=True,
        )

    # --- All reps, no augmentation ---
    print(f"\n  --- All reps, no augmentation ({len(reps)} reps) ---")
    print(f"  {len(models)} models x {len(full_budgets)} budgets x "
          f"{len(allocation_methods)} allocations x {len(importance_methods)} importance")

    search_no_aug = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        models=model_instances,
        budgets=full_budgets,
        allocation_methods=allocation_methods,
        importance_methods=importance_methods,
        n_folds=n_folds,
        augmentations=None,
        internal_evaluators=INTERNAL_EVALUATORS,
        verbose=True,
        cache=True,
    )

    # --- All reps, uncapped augmentation ---
    search_uncapped = search_no_aug  # fallback if no augmentations
    if augs:
        print(f"\n  --- Uncapped augmentation ({len(augs)} aug reps) ---")
        search_uncapped = find_best_hybrid(
            base_reps=reps,
            labels=labels,
            groups=groups,
            models=model_instances,
            budgets=full_budgets,
            allocation_methods=allocation_methods,
            importance_methods=importance_methods,
            n_folds=n_folds,
            augmentations=augs,
            augmentation_strategy='uncapped',
            internal_evaluators=INTERNAL_EVALUATORS,
            verbose=True,
            cache=True,
        )

    # --- All reps, capped augmentation ---
    search_capped = None
    if augs:
        print(f"\n  --- Capped augmentation ({len(augs)} aug reps) ---")
        search_capped = find_best_hybrid(
            base_reps=reps,
            labels=labels,
            groups=groups,
            models=model_instances,
            budgets=full_budgets,
            allocation_methods=allocation_methods,
            importance_methods=importance_methods,
            n_folds=n_folds,
            augmentations=augs,
            augmentation_strategy='capped',
            internal_evaluators=INTERNAL_EVALUATORS,
            verbose=True,
            cache=True,
        )

    # TODO: Frozen vs fine-tuned comparison. When FT reps are available as
    # cached .npy files, load them here and run a fifth search pass:
    #   ft_reps = _load_ft_reps(dataset_name, reps)  # frozen + FT merged
    #   search_ft = find_best_hybrid(base_reps=ft_reps, ...)
    # Compare search_ft.best_by_model vs search_no_aug.best_by_model to
    # answer: does hybrid(frozen+FT) beat hybrid(frozen_only)?
    search_ft = None

    # Use uncapped as the primary search (best overall)
    search = search_uncapped

    top3 = extract_top3_base_reps(search)
    if top3:
        print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

    # --- Reshape SearchResult to downstream format ---
    metric_label = search.metric_name

    # Configs: fields for Exp 9a ANOVA compatibility
    configs = []
    for r in search.all_results:
        configs.append({
            'model': r['model'],
            'allocation': r['allocation_method'],
            'importance': r['importance_method'],
            'budget': r['budget'],
            'r2_mean': r['score_mean'],
            'r2_std': r['score_std'],
            'fold_scores': r.get('fold_scores', []),
            'n_features': r['n_features'],
        })

    # Best by model
    best_by_model = {}
    for model_name, r in search.best_by_model.items():
        best_by_model[model_name] = {
            'model': model_name,
            'allocation': r['allocation_method'],
            'importance': r['importance_method'],
            'budget': r['budget'],
            'r2_mean': r['score_mean'],
            'r2_std': r['score_std'],
            'n_features': r['n_features'],
        }

    # Baselines: reshape library format {rep: {model: score}} to Exp 2 format
    baselines = {}
    lib_baselines = search.baselines
    _lower = search.lower_is_better
    for model_name in models:
        best_single_rep = None
        best_single_score = np.inf if _lower else -np.inf
        for rep_name, rep_scores in lib_baselines.items():
            rep_score = rep_scores.get(model_name, np.inf if _lower else -np.inf)
            if (_lower and rep_score < best_single_score) or \
               (not _lower and rep_score > best_single_score):
                best_single_score = rep_score
                best_single_rep = rep_name
        baselines[model_name] = {
            'best_single_rep': best_single_rep,
            'best_single_r2': best_single_score,
        }
        print(f"  {model_name}: best_rep={best_single_rep} {metric_label}={best_single_score:.4f}")

    # PCA baselines (library doesn't compute these -- keep manual)
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    print("\n  Computing PCA baselines...")
    pca_baselines = {}
    for budget in budgets:
        n_comp = min(budget, X_concat.shape[1], X_concat.shape[0] - 1)
        pca_baselines[budget] = {}
        for model_name, model_factory in models.items():
            pca_scores = []
            for train_idx, val_idx in cv_split(groups, n_folds, X_concat, labels, mode=mode):
                pca = PCA(n_components=n_comp)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_concat[train_idx])
                X_val_s = scaler.transform(X_concat[val_idx])
                X_train_p = pca.fit_transform(X_train_s)
                X_val_p = pca.transform(X_val_s)
                m = model_factory()
                m.fit(X_train_p, labels[train_idx])
                pca_scores.append(metric_fn(labels[val_idx], m.predict(X_val_p)))
            pca_baselines[budget][model_name] = float(np.mean(pca_scores))

    # Heatmap: model x budget -> best score
    heatmap = {}
    for model_name in models:
        heatmap[model_name] = {}
        model_configs = [c for c in configs if c['model'] == model_name]
        for budget in full_budgets:
            budget_configs = [c for c in model_configs if c['budget'] == budget]
            if budget_configs:
                best = max(budget_configs, key=lambda c: c['r2_mean'])
                heatmap[model_name][budget] = best['r2_mean']

    # Importance method wins
    importance_wins = {imp: 0 for imp in importance_methods}
    for model_name, best in best_by_model.items():
        imp = best['importance']
        if imp in importance_wins:
            importance_wins[imp] += 1

    # Print summary
    print(f"\n  BEST BY MODEL (metric: {metric_label}):")
    for model_name, best in best_by_model.items():
        baseline_score = baselines[model_name]['best_single_r2']
        baseline_rep = baselines[model_name]['best_single_rep']
        delta = best['r2_mean'] - baseline_score
        print(f"  >> {model_name}: {best['allocation']}/{best['importance']}"
              f"/b={best['budget']} -> {metric_label}={best['r2_mean']:.4f} "
              f"(vs {baseline_rep} {baseline_score:.4f}, delta={delta:+.4f})")

    print("\n  IMPORTANCE METHOD WINS:")
    for imp, count in sorted(importance_wins.items(), key=lambda x: -x[1]):
        print(f"    {imp}: {count}/{len(models)}")

    # Heatmap table
    print(f"\n  HEATMAP (model x budget -> best {metric_label}):")
    budget_strs = [str(b) for b in full_budgets]
    header = f"  {'Model':<12}" + "".join(f"{b:>8}" for b in budget_strs)
    print(header)
    print("  " + "-" * (12 + 8 * len(full_budgets)))
    for model_name in models:
        row = f"  {model_name:<12}"
        for budget in full_budgets:
            val = heatmap.get(model_name, {}).get(budget)
            row += f"{val:>8.4f}" if val is not None else f"{'--':>8}"
        print(row)

    # --- Condition comparison summary ---
    def _extract_best(search_obj):
        if search_obj is None:
            return {}
        out = {}
        for model_name, r in search_obj.best_by_model.items():
            out[model_name] = {
                'model': model_name,
                'allocation': r['allocation_method'],
                'importance': r['importance_method'],
                'budget': r['budget'],
                'r2_mean': r['score_mean'],
                'r2_std': r['score_std'],
                'n_features': r['n_features'],
            }
        return out

    conditions = {
        'no_aug': _extract_best(search_no_aug),
        'uncapped': _extract_best(search_uncapped),
    }
    if search_capped is not None:
        conditions['capped'] = _extract_best(search_capped)
    if search_top_tier is not None:
        conditions['top_tier_only'] = _extract_best(search_top_tier)
    if search_ft is not None:
        conditions['frozen_plus_ft'] = _extract_best(search_ft)

    # Print condition comparison
    print("\n  CONDITION COMPARISON (best score per model):")
    for model_name in models:
        parts = []
        for cond_name, cond_best in conditions.items():
            if model_name in cond_best:
                score = cond_best[model_name]['r2_mean']
                parts.append(f"{cond_name}={score:.4f}")
        if parts:
            print(f"    {model_name}: {', '.join(parts)}")

    return {
        'configs': configs,
        'best_by_model': best_by_model,
        'baselines': baselines,
        'pca_baselines': pca_baselines,
        'heatmap': heatmap,
        'importance_wins': importance_wins,
        'conditions': conditions,
        'has_augmentation': augs is not None,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
    }


if __name__ == '__main__':
    parser = create_experiment_parser('C02: Complexity Matching Grid',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {config['models']}")
    print(f"Budgets: {config['budgets']}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c02_complexity_matching_grid', args.dataset)
