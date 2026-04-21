#!/usr/bin/env python3
"""C03: Blueprint transfer across datasets.

Paper: Complexity
Conclusion: Optimal configs transfer across datasets (context-dependent).
Dependencies: c02 (reads JSON from multiple datasets, or runs own searches).

Loads multiple datasets, computes complexity profiles (Lorena et al. 2018),
runs find_best_hybrid on each, and compares optimal configurations.
Includes novel model transfer (SVR, ElasticNet), budget variance decomposition,
and correlation of complexity metrics with optimal budget.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from copy import deepcopy

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split, _cv_split,
    compute_dataset_complexity, get_domain_metric,
    find_best_hybrid, create_hybrid, apply_feature_selection,
    get_best_for_downstream,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)
from kirby.hybrid import apply_augmentation_selection


def run(datasets, config, output_dir, dataset_name=None, **kwargs):
    """Cross-context analysis across multiple datasets.

    Parameters
    ----------
    datasets : dict
        {dataset_name: data_dict} for each dataset.
    config : dict
        Config dict (fast or full).
    output_dir : str
        Output directory path.
    dataset_name : str, optional
        Ignored (we use datasets dict keys instead).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C03: CROSS-CONTEXT ANALYSIS")
    print("=" * 70)

    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    results = {}

    for ds_name, data in datasets.items():
        print(f"\n  --- {ds_name} ---")
        metric_name_ds, _, _ = get_domain_metric(data)

        # Compute dataset complexity profile
        print(f"  Computing complexity metrics...")
        complexity = compute_dataset_complexity(data['reps'], data['labels'])
        mle_str = f", MLE_ID={complexity['mle_id']:.1f}" if complexity.get('mle_id') else ""
        print(f"    erank={complexity['erank']:.1f}, PR={complexity['participation_ratio']:.1f}, "
              f"n/d={complexity['n_over_d']:.2f}, n/erank={complexity['n_over_erank']:.1f}{mle_str}")
        print(f"    C1={complexity['C1']:.3f}, C2={complexity['C2']:.3f}, "
              f"L1={complexity['L1']:.3f}, S3={complexity['S3']:.3f}")

        # Run hybrid search
        print(f"  Running find_best_hybrid...")
        start = time.time()

        try:
            from sklearn.svm import SVR
            from sklearn.linear_model import ElasticNet

            novel_models = {
                'svr': SVR(kernel='rbf', C=1.0),
                'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            }

            search_result = find_best_hybrid(
                base_reps=data['reps'],
                labels=data['labels'],
                groups=data['groups'],
                speed='full',
                augmentations=data.get('augmentations'),
                augmentation_strategy='uncapped' if data.get('augmentations') else 'none',
                verbose=False,
                cache=True,
                dataset_name=ds_name,
                tier='general_distance',
                internal_evaluators=DISTANCE_EVALUATORS,
                budgets=DISTANCE_BUDGETS,
                metric=metric_name_ds,
                task_type=data['task_type'],
            )

            # Prefer SVM-evaluated config for SVR downstream, fall back to RF
            best = get_best_for_downstream(search_result, 'svm')
            if best is None or 'blueprint' not in best:
                best = search_result.best_by_model.get('rf') or (
                    next(iter(search_result.best_by_model.values()), None)
                    if search_result.best_by_model else None
                )
            metric_name, metric_fn, _ = get_domain_metric(data)
            augs = data.get('augmentations')
            if best:
                bp = best['blueprint']
                novel_results = {}
                for novel_name, novel_model in novel_models.items():
                    try:
                        sample_X = next(iter(data['reps'].values()))
                        novel_scores = []

                        for train_idx, val_idx in cv_split(
                                data['groups'], n_folds, sample_X, data['labels'], mode=mode):
                            train_reps = {k: v[train_idx] for k, v in data['reps'].items()}
                            val_reps = {k: v[val_idx] for k, v in data['reps'].items()}
                            y_train = data['labels'][train_idx]
                            y_val = data['labels'][val_idx]

                            train_augs = {k: v[train_idx] for k, v in augs.items()} if augs else None
                            X_train, fi = create_hybrid(
                                train_reps, y_train,
                                allocation_method=bp.allocation_method,
                                budget=bp.budget,
                                importance_method=bp.importance_method,
                                internal_evaluator=bp.internal_evaluator,
                                augmentations=train_augs,
                                augmentation_strategy=bp.augmentation_strategy,
                            )
                            X_val = apply_feature_selection(val_reps, fi)
                            aug_info = fi.get('augmentation_info')
                            if aug_info and augs:
                                val_augs = {k: v[val_idx] for k, v in augs.items()}
                                X_aug_val = apply_augmentation_selection(val_augs, aug_info)
                                if X_aug_val is not None:
                                    X_val = np.hstack([X_val, X_aug_val])

                            scaler = StandardScaler()
                            X_train_s = scaler.fit_transform(X_train)
                            X_val_s = scaler.transform(X_val)

                            m = deepcopy(novel_model)
                            m.fit(X_train_s, y_train)
                            novel_scores.append(metric_fn(y_val, m.predict(X_val_s)))

                        novel_results[novel_name] = float(np.mean(novel_scores))
                        print(f"    {novel_name} on {ds_name}: {metric_name}={np.mean(novel_scores):.4f}")
                    except Exception as e:
                        print(f"    {novel_name} on {ds_name}: FAILED - {e}")

            elapsed = time.time() - start

            ds_result = {
                'n_samples': len(data['labels']),
                'elapsed_seconds': elapsed,
                'complexity': complexity,
                'metric': metric_name_ds,
                'best_model': best['model'] if best else None,
                'best_score': best['score_mean'] if best else None,
                'best_allocation': best['blueprint'].allocation_method if best else None,
                'best_importance': best['blueprint'].importance_method if best else None,
                'best_budget': best['blueprint'].budget if best else None,
                'best_dim': best['n_features'] if best else None,
                'baselines': search_result.baselines,
                'best_by_model': {
                    m: {'budget': r['budget'], 'score': r['score_mean'], 'dim': r['n_features']}
                    for m, r in search_result.best_by_model.items()
                },
                'performance_matrix': search_result.performance_matrix,
                'novel_models': novel_results if best else {},
            }
            results[ds_name] = ds_result

            print(f"    Best: {ds_result['best_model']}, {metric_name_ds}={ds_result['best_score']:.4f}, "
                  f"alloc={ds_result['best_allocation']}, imp={ds_result['best_importance']}, "
                  f"budget={ds_result['best_budget']}")

        except Exception as e:
            print(f"    {ds_name}: FAILED - {e}")
            results[ds_name] = {'error': str(e)}

    # Cross-dataset comparison table
    print("\n  CROSS-CONTEXT COMPARISON:")
    print(f"  {'Dataset':<10} {'N':<6} {'Best Model':<12} {'Importance':<15} {'Budget':<8} {'Score':<8}")
    print("  " + "-" * 65)
    for ds_name, r in results.items():
        if 'error' not in r:
            print(f"  {ds_name:<10} {r['n_samples']:<6} {r['best_model']:<12} "
                  f"{r['best_importance']:<15} {r['best_budget']:<8} {r['best_score']:.4f}")

    # --- Dataset complexity vs optimal hybrid dimension ---
    valid = {ds: r for ds, r in results.items()
             if 'error' not in r and r.get('best_budget') is not None}
    if len(valid) >= 2:
        print("\n  DATASET COMPLEXITY vs OPTIMAL BUDGET:")
        print(f"  {'Dataset':<10} {'erank':>7} {'PR':>7} {'n/erank':>8} {'C1':>6} {'C2':>6} "
              f"{'L1':>6} {'S3':>6} {'Budget':>7} {'Dim':>5}")
        print("  " + "-" * 82)

        complexity_keys = ['erank', 'participation_ratio', 'n_over_erank', 'C1', 'C2', 'L1', 'S3']
        ds_names = sorted(valid.keys())
        budgets_arr = np.array([valid[ds]['best_budget'] for ds in ds_names])
        dims = np.array([valid[ds]['best_dim'] for ds in ds_names])

        for ds in ds_names:
            c = valid[ds]['complexity']
            print(f"  {ds:<10} {c['erank']:>7.1f} {c['participation_ratio']:>7.1f} "
                  f"{c['n_over_erank']:>8.1f} {c['C1']:>6.3f} {c['C2']:>6.3f} "
                  f"{c['L1']:>6.3f} {c['S3']:>6.3f} "
                  f"{valid[ds]['best_budget']:>7} {valid[ds]['best_dim']:>5}")

        # Correlation of each complexity metric with optimal budget
        correlations = {}
        print("\n  Spearman correlation (complexity metric -> optimal budget):")
        for key in complexity_keys:
            values = np.array([valid[ds]['complexity'][key] for ds in ds_names])
            if np.std(values) < 1e-10:
                continue
            rho, p = spearmanr(values, budgets_arr)
            correlations[key] = {'rho_vs_budget': float(rho), 'p_vs_budget': float(p)}
            # Also correlate with actual achieved dimension
            rho_dim, p_dim = spearmanr(values, dims)
            correlations[key]['rho_vs_dim'] = float(rho_dim)
            correlations[key]['p_vs_dim'] = float(p_dim)
            print(f"    {key:<20} -> budget: rho={rho:+.3f} (p={p:.3f}), "
                  f"dim: rho={rho_dim:+.3f} (p={p_dim:.3f})")

        results['_complexity_correlation'] = correlations

        # Key insight: does higher intrinsic complexity -> higher optimal budget?
        eranks = np.array([valid[ds]['complexity']['erank'] for ds in ds_names])
        rho_erank, _ = spearmanr(eranks, budgets_arr)
        print(f"\n  KEY FINDING: erank -> optimal budget correlation: rho={rho_erank:+.3f}")
        if rho_erank > 0.3:
            print("    -> Higher intrinsic dimensionality correlates with larger optimal hybrid")
        elif rho_erank < -0.3:
            print("    -> Higher intrinsic dimensionality correlates with smaller optimal hybrid")
        else:
            print("    -> Weak correlation -- optimal budget depends on more than raw dimensionality")

        # Per-model optimal budget comparison across datasets
        model_names = set()
        for ds in ds_names:
            model_names.update(valid[ds].get('best_by_model', {}).keys())
        if model_names:
            print("\n  PER-MODEL OPTIMAL BUDGET (cross-dataset):")
            print(f"  {'Model':<12}" + "".join(f"{ds:>14}" for ds in ds_names))
            print("  " + "-" * (12 + 14 * len(ds_names)))
            for model in sorted(model_names):
                row = f"  {model:<12}"
                for ds in ds_names:
                    bm = valid[ds].get('best_by_model', {}).get(model)
                    if bm:
                        row += f"b={bm['budget']:>3} score={bm['score']:.3f} "
                    else:
                        row += f"{'--':>14}"
                print(row)

            # MODEL x DATASET INTERACTION ANALYSIS
            model_list = sorted(model_names)
            budget_matrix = []
            score_matrix = []
            for model in model_list:
                model_budgets = []
                model_scores = []
                for ds in ds_names:
                    bm = valid[ds].get('best_by_model', {}).get(model)
                    if bm:
                        model_budgets.append(bm['budget'])
                        model_scores.append(bm['score'])
                    else:
                        model_budgets.append(np.nan)
                        model_scores.append(np.nan)
                budget_matrix.append(model_budgets)
                score_matrix.append(model_scores)
            budget_matrix = np.array(budget_matrix, dtype=float)
            score_matrix = np.array(score_matrix, dtype=float)

            # Budget variance decomposition
            if budget_matrix.shape[0] >= 2 and budget_matrix.shape[1] >= 2:
                valid_mask = ~np.isnan(budget_matrix)
                if valid_mask.all():
                    grand_mean = np.nanmean(budget_matrix)
                    row_means = np.nanmean(budget_matrix, axis=1)
                    col_means = np.nanmean(budget_matrix, axis=0)
                    ss_model = len(ds_names) * np.sum((row_means - grand_mean) ** 2)
                    ss_dataset = len(model_list) * np.sum((col_means - grand_mean) ** 2)
                    ss_total = np.sum((budget_matrix - grand_mean) ** 2)
                    ss_interaction = ss_total - ss_model - ss_dataset

                    if ss_total > 0:
                        print(f"\n  BUDGET VARIANCE DECOMPOSITION:")
                        print(f"    Model effect:       {ss_model/ss_total*100:5.1f}% of variance")
                        print(f"    Dataset effect:     {ss_dataset/ss_total*100:5.1f}% of variance")
                        print(f"    Interaction effect: {ss_interaction/ss_total*100:5.1f}% of variance")

                    results['_budget_variance'] = {
                        'ss_model': float(ss_model),
                        'ss_dataset': float(ss_dataset),
                        'ss_interaction': float(ss_interaction),
                        'ss_total': float(ss_total),
                        'pct_model': float(ss_model / ss_total * 100) if ss_total > 0 else 0,
                        'pct_dataset': float(ss_dataset / ss_total * 100) if ss_total > 0 else 0,
                        'pct_interaction': float(ss_interaction / ss_total * 100) if ss_total > 0 else 0,
                    }

                    # Does budget range differ across models?
                    print(f"\n  MODEL DIMENSION SENSITIVITY:")
                    print(f"  {'Model':<12} {'Budget Range':<15} {'Budget Std':<12} {'Score Range':<12}")
                    print("  " + "-" * 51)
                    for i, model in enumerate(model_list):
                        b_row = budget_matrix[i]
                        score_row = score_matrix[i]
                        if not np.any(np.isnan(b_row)):
                            print(f"  {model:<12} {int(b_row.min())}-{int(b_row.max()):<10} "
                                  f"{np.std(b_row):>8.1f}    {np.ptp(score_row):>8.4f}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C03: Cross-Dataset Transfer',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--datasets', type=str, nargs='+', default=['logd', 'caco2', 'herg_ki'],
                        help='Dataset names to compare (at least 2)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Datasets: {args.datasets}")
    print(f"Reps: {args.reps}")

    # Load all datasets
    all_datasets = {}
    for ds_name in args.datasets:
        print(f"\nLoading {ds_name}...")
        try:
            all_datasets[ds_name] = load_data(
                ds_name, args.reps,
                delta=getattr(args, 'delta', False),
                augment=getattr(args, 'augment', False))
        except Exception as e:
            print(f"  FAILED: {e}")

    if len(all_datasets) < 2:
        print("Need at least 2 datasets for cross-context analysis.")
        sys.exit(1)

    results = run(all_datasets, config, args.output)
    save_experiment_results(results, args.output, 'c03_cross_dataset_transfer', 'multi')
