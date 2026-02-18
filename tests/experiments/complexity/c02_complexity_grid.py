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
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, INTERMEDIARY, FULL, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split, _cv_split,
    find_best_hybrid,
)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run model x budget complexity grid via find_best_hybrid."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C02: COMPLEXITY MATCHING GRID")
    print("=" * 70)

    models = get_model_factories(config['models'])
    budgets = config['budgets']
    allocation_methods = config['allocations']
    importance_methods = config['importance_methods']
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    total_dim = sum(v.shape[1] for v in reps.values())
    full_budgets = budgets + [total_dim]

    # Convert model factories to instances for the library
    model_instances = {name: factory() for name, factory in models.items()}

    print(f"  {len(models)} models x {len(full_budgets)} budgets x "
          f"{len(allocation_methods)} allocations x {len(importance_methods)} importance")
    print(f"  Delegating to find_best_hybrid()...\n")

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

    # --- Reshape SearchResult to downstream format ---

    # Configs: rename fields for Exp 9a ANOVA compatibility
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

    # Best by model: rename fields for Exp 8 compatibility
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
    for model_name in models:
        best_single_rep = None
        best_single_r2 = -np.inf
        for rep_name, rep_scores in lib_baselines.items():
            if rep_name == 'all_concat':
                continue
            rep_r2 = rep_scores.get(model_name, -np.inf)
            if rep_r2 > best_single_r2:
                best_single_r2 = rep_r2
                best_single_rep = rep_name
        baselines[model_name] = {
            'best_single_rep': best_single_rep,
            'best_single_r2': best_single_r2,
        }
        print(f"  {model_name}: best_rep={best_single_rep} R2={best_single_r2:.4f}")

    # PCA baselines (library doesn't compute these -- keep manual)
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
                pca_scores.append(r2_score(labels[val_idx], m.predict(X_val_p)))
            pca_baselines[budget][model_name] = float(np.mean(pca_scores))

    # Heatmap: model x budget -> best R2
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
    print("\n  BEST BY MODEL:")
    for model_name, best in best_by_model.items():
        baseline_r2 = baselines[model_name]['best_single_r2']
        baseline_rep = baselines[model_name]['best_single_rep']
        delta = best['r2_mean'] - baseline_r2
        print(f"  >> {model_name}: {best['allocation']}/{best['importance']}"
              f"/b={best['budget']} -> R2={best['r2_mean']:.4f} "
              f"(vs {baseline_rep} {baseline_r2:.4f}, delta={delta:+.4f})")

    print("\n  IMPORTANCE METHOD WINS:")
    for imp, count in sorted(importance_wins.items(), key=lambda x: -x[1]):
        print(f"    {imp}: {count}/{len(models)}")

    # Heatmap table
    print("\n  HEATMAP (model x budget -> best R2):")
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

    return {
        'configs': configs,
        'best_by_model': best_by_model,
        'baselines': baselines,
        'pca_baselines': pca_baselines,
        'heatmap': heatmap,
        'importance_wins': importance_wins,
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
    save_experiment_results(results, args.output, 'c02_complexity_grid', args.dataset)
