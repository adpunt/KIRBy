#!/usr/bin/env python3
"""C04: Feature importance as learned metric space.

Paper: Complexity
Conclusion: Feature importance doubles as a learned similarity metric.
Dependencies: None (standalone). Requires benchmark_importance_aggregation.

Computes importance scores with each method, then evaluates them as kNN
weights and similarity retrieval weights. Cross-tests selection by method A
with weighting by method B. Low-dim subset evaluation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    compute_feature_importance,
    evaluate_weighted_knn, evaluate_similarity_retrieval,
)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Evaluate importance scores as learned metric spaces."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C04: IMPORTANCE WEIGHTS AS LEARNED METRICS")
    print("=" * 70)

    importance_methods = config['importance_methods']
    n_folds = config['n_splits']
    budget = kwargs.get('budget', 25)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    print(f"  Concatenated features: {X_concat.shape}")

    results = {'methods': {}, 'cross_test': {}, 'low_dim': {}}

    # Compute importance with each method
    importances = {}
    for method in importance_methods:
        print(f"  Computing {method}...", end=' ', flush=True)
        start = time.time()
        try:
            imp_dict = compute_feature_importance(reps, labels, method=method)
            imp_concat = np.concatenate([imp_dict[rep] for rep in sorted(imp_dict.keys())])
            importances[method] = imp_concat
            elapsed = time.time() - start
            print(f"done ({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    # Evaluate each method: unweighted kNN vs weighted kNN
    print("\n  Evaluating as kNN weights...")
    for method, imp_scores in importances.items():
        try:
            knn_result = evaluate_weighted_knn(X_concat, labels, imp_scores, groups, n_folds)
            sim_result = evaluate_similarity_retrieval(X_concat, labels, imp_scores, groups, n_folds)

            k10 = knn_result.get(10, knn_result.get(5, {}))
            results['methods'][method] = {
                'knn': knn_result,
                'similarity': sim_result,
                'knn_delta_k10': k10.get('delta', 0),
                'sim_delta_r2': sim_result.get('delta_r2', 0),
            }

            print(f"    {method}: kNN delta={k10.get('delta', 0):+.4f}, "
                  f"sim delta={sim_result.get('delta_r2', 0):+.4f}")
        except Exception as e:
            print(f"    {method}: FAILED - {e}")

    # Cross-test: selection by method A, weighting by method B
    print("\n  Cross-test matrix (select x weight)...")
    method_names = list(importances.keys())
    if len(method_names) >= 2:
        for select_method in method_names[:4]:
            select_scores = importances[select_method]
            top_idx = np.argsort(select_scores)[-budget:]
            X_subset = X_concat[:, top_idx]

            for weight_method in method_names[:4]:
                weight_scores = importances[weight_method][top_idx]
                try:
                    sim_result = evaluate_similarity_retrieval(
                        X_subset, labels, weight_scores, groups, n_folds)
                    key = f"{select_method}__x__{weight_method}"
                    results['cross_test'][key] = sim_result['weighted_r2']
                except Exception:
                    pass

    # Low-dim: budget=25 selected features + importance weighting
    print(f"\n  Low-dim (budget={budget}) selected + weighted...")
    for method, imp_scores in importances.items():
        top_idx = np.argsort(imp_scores)[-budget:]
        X_subset = X_concat[:, top_idx]
        subset_scores = imp_scores[top_idx]
        try:
            sim_result = evaluate_similarity_retrieval(
                X_subset, labels, subset_scores, groups, n_folds)
            results['low_dim'][method] = {
                'weighted_r2': sim_result['weighted_r2'],
                'unweighted_r2': sim_result['unweighted_r2'],
                'delta_r2': sim_result['delta_r2'],
            }
            print(f"    {method}: weighted R2={sim_result['weighted_r2']:.4f}, "
                  f"unweighted R2={sim_result['unweighted_r2']:.4f}")
        except Exception as e:
            print(f"    {method}: FAILED - {e}")

    # Summary: which method produces the best metric space?
    if results['methods']:
        best_knn = max(results['methods'].items(),
                       key=lambda x: x[1].get('knn_delta_k10', -999))
        best_sim = max(results['methods'].items(),
                       key=lambda x: x[1].get('sim_delta_r2', -999))
        print(f"\n  Best kNN weight source: {best_knn[0]} "
              f"(delta={best_knn[1]['knn_delta_k10']:+.4f})")
        print(f"  Best similarity weight source: {best_sim[0]} "
              f"(delta={best_sim[1]['sim_delta_r2']:+.4f})")
        results['best_knn_method'] = best_knn[0]
        results['best_sim_method'] = best_sim[0]

    # Cross-test summary table
    if results['cross_test']:
        print("\n  CROSS-TEST MATRIX (select x weight -> weighted R2):")
        select_methods = sorted(set(k.split('__x__')[0] for k in results['cross_test']))
        weight_methods = sorted(set(k.split('__x__')[1] for k in results['cross_test']))
        col_label = "select\\weight"
        header = f"  {col_label:<20}" + "".join(f"{w:>14}" for w in weight_methods)
        print(header)
        for s in select_methods:
            row = f"  {s:<20}"
            for w in weight_methods:
                key = f"{s}__x__{w}"
                val = results['cross_test'].get(key)
                row += f"{val:>14.4f}" if val is not None else f"{'--':>14}"
            print(row)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C04: Importance Weights as Learned Metrics',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--budget', type=int, default=25,
                        help='Budget for low-dim feature subset evaluation')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Importance methods: {config['importance_methods']}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, budget=args.budget)
    save_experiment_results(results, args.output, 'c04_importance_as_metric', args.dataset)
