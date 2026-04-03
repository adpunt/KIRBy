#!/usr/bin/env python3
"""C07: Data Complexity via Scaffold Partitioning — optimal config by subpopulation.

Paper: Complexity
Conclusion: Optimal config depends on data subpopulation complexity.
Dependencies: c02 (reads JSON, optional).
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
    save_experiment_results, load_experiment_results, convert_numpy,
    get_model_factories, get_domain_metric,
    estimate_mutual_information_knn, compute_effective_rank,
    compute_intrinsic_dimensionality,
    find_best_hybrid,
    extract_top3_base_reps,
    INTERNAL_EVALUATORS,
)


def run(data, config, output_dir, dataset_name, partitions=None):
    """Exp 7: Optimal config depends on data subpopulation complexity.

    Uses find_best_hybrid() on each subpopulation so we get a properly-searched
    optimal config per partition, not just greedy/treeshap.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: DATA COMPLEXITY VIA SCAFFOLD PARTITIONING")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    budgets = config['budgets']
    search_speed = config['search_speed']
    metric_name, _, lower_is_better = get_domain_metric(data)

    # For molecular datasets, use smiles; for antibody, this partition may not apply
    smiles = data.get('smiles')

    all_partitions = partitions or ['scaffold_frequency', 'molecular_complexity', 'target_variance']
    results = {'partitions': {}, 'search_speed': search_speed}

    for partition_name in all_partitions:
        print(f"\n  Partition: {partition_name}")

        # Compute partition masks
        if partition_name == 'scaffold_frequency':
            # Common (top 20% scaffolds) vs rare (bottom 80%)
            unique_scaffolds, scaffold_counts = np.unique(groups, return_counts=True)
            count_threshold = np.percentile(scaffold_counts, 80)  # top 20% most frequent
            frequent_scaffolds = set(unique_scaffolds[scaffold_counts >= count_threshold])
            mask_easy = np.array([g in frequent_scaffolds for g in groups])
            mask_hard = ~mask_easy

        elif partition_name == 'molecular_complexity':
            # Heavy atom count: <= median vs > median
            if smiles is None:
                print("    No SMILES available (antibody dataset?), skipping molecular_complexity")
                continue
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                hac = np.array([Descriptors.HeavyAtomCount(Chem.MolFromSmiles(s))
                                if Chem.MolFromSmiles(s) else 0 for s in smiles])
                median_hac = np.median(hac[hac > 0])
                mask_easy = hac <= median_hac
                mask_hard = hac > median_hac
            except ImportError:
                print("    RDKit not available, skipping molecular_complexity")
                continue

        elif partition_name == 'target_variance':
            # Scaffolds with low target variance vs high
            unique_scaffolds = np.unique(groups)
            scaffold_var = {}
            for s in unique_scaffolds:
                s_labels = labels[groups == s]
                if len(s_labels) >= 2:
                    scaffold_var[s] = np.var(s_labels)
            if not scaffold_var:
                print("    Not enough scaffolds with 2+ members, skipping")
                continue
            median_var = np.median(list(scaffold_var.values()))
            mask_easy = np.array([scaffold_var.get(g, 0) <= median_var for g in groups])
            mask_hard = ~mask_easy
        else:
            print(f"    Unknown partition: {partition_name}, skipping")
            continue

        n_easy = mask_easy.sum()
        n_hard = mask_hard.sum()
        print(f"    easy={n_easy}, hard={n_hard}")

        if n_easy < 50 or n_hard < 50:
            print("    Subpopulation too small, skipping")
            continue

        part_result = {'easy': {}, 'hard': {}, 'n_easy': int(n_easy), 'n_hard': int(n_hard)}

        full_augs = data.get('augmentations')
        for subpop_name, mask in [('easy', mask_easy), ('hard', mask_hard)]:
            sub_reps = {k: v[mask] for k, v in reps.items()}
            sub_labels = labels[mask]
            sub_groups = groups[mask]
            sub_augs = {k: v[mask] for k, v in full_augs.items()} if full_augs else None
            n_unique_groups = len(np.unique(sub_groups))

            if n_unique_groups < 2:
                print(f"    {subpop_name}: not enough groups for CV, skipping")
                continue

            # Complexity metrics
            X_concat = np.hstack([sub_reps[k] for k in sorted(sub_reps.keys())])
            metrics = {}
            try:
                n_comp = min(50, X_concat.shape[1], X_concat.shape[0] - 1)
                X_reduced = PCA(n_components=n_comp).fit_transform(
                    StandardScaler().fit_transform(X_concat))
                metrics['mi'] = float(estimate_mutual_information_knn(X_reduced, sub_labels, k=5))
            except Exception:
                metrics['mi'] = None
            metrics['effective_rank'] = compute_effective_rank(X_concat)
            dim_info = compute_intrinsic_dimensionality(X_concat)
            metrics['pca_95'] = dim_info.get('pca_95')
            metrics['mle_dim'] = dim_info.get('mle')
            metrics['n_samples'] = int(mask.sum())

            print(f"    {subpop_name}: eff_rank={metrics['effective_rank']:.1f}, "
                  f"MI={metrics.get('mi', '?')}, PCA95={metrics.get('pca_95', '?')}")

            # Use find_best_hybrid for proper search
            top3 = []
            try:
                search = find_best_hybrid(
                    sub_reps, sub_labels, groups=sub_groups,
                    speed=search_speed,
                    budgets=budgets,
                    augmentations=sub_augs,
                    augmentation_strategy='capped',
                    internal_evaluators=INTERNAL_EVALUATORS,
                    metric=metric_name,
                    verbose=False,
                    cache=True,
                )
                top3 = extract_top3_base_reps(search)
                if top3:
                    print(f"    {subpop_name} Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

                # Pick the best result across all models
                best = None
                best_model_name = None
                for m_name, m_best in search.best_by_model.items():
                    if best is None or \
                       (lower_is_better and m_best['score_mean'] < best['score_mean']) or \
                       (not lower_is_better and m_best['score_mean'] > best['score_mean']):
                        best = m_best
                        best_model_name = m_name
                best_config = {
                    'model': best.get('model', best_model_name),
                    'budget': best['blueprint'].budget,
                    'r2': best['score_mean'],
                    'r2_std': best['score_std'],
                    'allocation': best['blueprint'].allocation_method,
                    'importance': best['blueprint'].importance_method,
                    'n_features': best['n_features'],
                } if best else None

                # Also extract best-by-model for richer comparison
                best_by_model = {}
                for model_name, mbest in search.best_by_model.items():
                    best_by_model[model_name] = {
                        'r2': mbest['score_mean'],
                        'budget': mbest['blueprint'].budget,
                        'importance': mbest['blueprint'].importance_method,
                    }

            except Exception as e:
                print(f"    {subpop_name}: find_best_hybrid failed - {e}")
                best_config = None
                best_by_model = {}

            part_result[subpop_name] = {
                'metrics': metrics,
                'best': best_config,
                'best_by_model': best_by_model,
                'baselines': search.baselines if best_config else {},
                'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
            }
            if best_config:
                print(f"    {subpop_name} best: {best_config['model']}/b={best_config['budget']} "
                      f"R²={best_config['r2']:.4f} "
                      f"[{best_config['importance']}/{best_config['allocation']}]")

        # Compare easy vs hard
        easy_best = part_result['easy'].get('best')
        hard_best = part_result['hard'].get('best')
        if easy_best and hard_best:
            part_result['delta_budget'] = hard_best['budget'] - easy_best['budget']
            part_result['config_differs'] = (
                easy_best['model'] != hard_best['model']
                or easy_best['budget'] != hard_best['budget']
                or easy_best.get('importance') != hard_best.get('importance')
            )
            print(f"    >> Configs differ: {part_result['config_differs']} "
                  f"(easy={easy_best['model']}/b={easy_best['budget']}/{easy_best.get('importance')}, "
                  f"hard={hard_best['model']}/b={hard_best['budget']}/{hard_best.get('importance')})")

        results['partitions'][partition_name] = part_result

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C07: Data Complexity via Scaffold Partitioning',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--partitions', type=str, nargs='+', default=None,
                        choices=['scaffold_frequency', 'molecular_complexity', 'target_variance'],
                        help='Which partitions to test (default: all)')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, partitions=args.partitions)
    save_experiment_results(results, args.output, 'c07_data_complexity_partitions', args.dataset)
