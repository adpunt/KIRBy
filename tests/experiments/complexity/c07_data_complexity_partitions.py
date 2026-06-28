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
    INTERNAL_EVALUATORS, TOP_CAPPED_MAX_N,
    get_best_search_for_downstream,
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

    # Size-gate (2026-06-28): the per-partition find_best_hybrid searches are
    # uncached and blow the 7-day wall on big datasets. Skip above the cutoff.
    from experiments._shared import LARGE_DATASET_N
    if len(labels) > LARGE_DATASET_N:
        print(f"  [size-gate] n={len(labels)} > {LARGE_DATASET_N}: skipping c07 "
              f"(partition searches too expensive at scale)")
        return {'skipped': f'n>{LARGE_DATASET_N}', 'partitions': {},
                'search_speed': search_speed}

    # For molecular datasets, use smiles; for antibody, this partition may not apply
    smiles = data.get('smiles')

    all_partitions = partitions or ['scaffold_frequency', 'molecular_complexity', 'target_variance']
    results = {'partitions': {}, 'search_speed': search_speed}

    # Second-pass scaffolding (per user 2026-05-23): in addition to per-partition
    # find_best_hybrid (the experiment's main signal), record what the
    # FULL-DATA pool-best blueprint scores when re-fit on each partition. Shows
    # how much partition-specific tuning matters beyond global tuning.
    fulldata_blueprints = {}
    full_augs_all = data.get('augmentations')
    for cand_model in ['rf', 'svm', 'lightgbm', 'mlp']:
        try:
            _s_fd, _m_fd = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=cand_model, budget=None,
                augmentations=full_augs_all, speed=search_speed,
                dataset_name=dataset_name,
                task_type=data['task_type'], metric=metric_name,
                data=data,
            )
            if _m_fd:
                fulldata_blueprints[cand_model] = (
                    _m_fd['best_config']['blueprint'],
                    _m_fd['tier'], _m_fd['aug_condition'],
                )
        except Exception:
            pass
    results['fulldata_blueprints_meta'] = {
        m: {'tier': t, 'aug_condition': a, 'budget': bp.budget,
            'allocation': bp.allocation_method, 'importance': bp.importance_method}
        for m, (bp, t, a) in fulldata_blueprints.items()
    }

    # c07 large-dataset cap (signed off 2026-06-05): the per-subpopulation
    # searches below are full, uncached find_best_hybrid runs (uncapped once a
    # half exceeds TOP_CAPPED_MAX_N rows), so on big datasets they grow into the
    # hundreds of hours — c07 was the one expensive experiment with no size
    # limit. Preselect a fixed subset for the partition searches. Selected by
    # WHOLE scaffold groups (not random rows) so the scaffold-based partitions
    # stay intact, no group is fragmented, and downstream CV stays group-aware
    # (NOT a random split). The full-data blueprints above used the real full
    # data (cached precompute). Deterministic (seed 42).
    C07_MAX_SAMPLES = 2000
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    _c07_augs = data.get('augmentations')
    if len(labels) > C07_MAX_SAMPLES:
        _rng = np.random.RandomState(42)
        _uniq = np.unique(groups)
        _rng.shuffle(_uniq)
        _chosen, _count = set(), 0
        for _g in _uniq:
            _chosen.add(_g)
            _count += int((groups == _g).sum())
            if _count >= C07_MAX_SAMPLES:
                break
        _idx = np.where(np.array([g in _chosen for g in groups]))[0]
        reps = {k: v[_idx] for k, v in reps.items()}
        labels = labels[_idx]
        groups = groups[_idx]
        if smiles is not None:
            smiles = ([smiles[i] for i in _idx] if isinstance(smiles, list)
                      else np.asarray(smiles)[_idx])
        if _c07_augs:
            _c07_augs = {k: v[_idx] for k, v in _c07_augs.items()}
        print(f"  c07: large dataset capped to {len(labels)} samples "
              f"({len(_chosen)} whole scaffold groups, seed 42) for partition searches")

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
                # Parse once per SMILES, track valid_mask so failed parses
                # don't silently land in "easy" via hac=0 below median.
                hac = np.full(len(smiles), -1, dtype=int)
                valid_mask = np.zeros(len(smiles), dtype=bool)
                for i, s in enumerate(smiles):
                    mol = Chem.MolFromSmiles(s)
                    if mol is not None:
                        hac[i] = Descriptors.HeavyAtomCount(mol)
                        valid_mask[i] = True
                if not valid_mask.any():
                    print("    No valid SMILES, skipping molecular_complexity")
                    continue
                median_hac = np.median(hac[valid_mask])
                mask_easy = valid_mask & (hac <= median_hac)
                mask_hard = valid_mask & (hac > median_hac)
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

        full_augs = _c07_augs  # c07 cap: subsampled augmentations (see C07_MAX_SAMPLES)
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

            # Use find_best_hybrid for proper search.
            # Subpop-specific cache key — won't collide with main precompute.
            # Capped is fine for small subpops but projects past the 7-day
            # partition when subpop n>TOP_CAPPED_MAX_N. Fall back to uncapped.
            n_sub = int(mask.sum())
            sub_aug_strategy = 'capped' if n_sub <= TOP_CAPPED_MAX_N else 'uncapped'
            top3 = []
            # Initialize before the try so the `search.baselines` read further
            # down (part_result['baselines']) can never NameError if a future
            # refactor makes best_config truthy on a path where the search
            # raised. Today it's guarded by the `if best_config` short-circuit,
            # but None is the safe default. (CXA-17)
            search = None
            try:
                search = find_best_hybrid(
                    sub_reps, sub_labels, groups=sub_groups,
                    speed=search_speed,
                    budgets=budgets,
                    augmentations=sub_augs,
                    augmentation_strategy=sub_aug_strategy,
                    internal_evaluators=INTERNAL_EVALUATORS,
                    metric=metric_name,
                    task_type=data['task_type'],
                    verbose=False,
                    cache=True,
                    dataset_name=f"{dataset_name}_{partition_name}_{subpop_name}",
                    tier='general',
                )
                top3 = extract_top3_base_reps(search)
                if top3:
                    print(f"    {subpop_name} Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

                # Pick the best result across all models. Skip NaN-scored
                # models (otherwise NaN can "win" by accident when iterated
                # first — NaN comparisons are always False).
                best = None
                best_model_name = None
                for m_name, m_best in search.best_by_model.items():
                    _sc = m_best.get('score_mean', float('nan'))
                    if np.isnan(_sc):
                        continue
                    if best is None or \
                       (lower_is_better and _sc < best['score_mean']) or \
                       (not lower_is_better and _sc > best['score_mean']):
                        best = m_best
                        best_model_name = m_name
                best_config = {
                    'model': best.get('model', best_model_name),
                    'budget': best['blueprint'].budget,
                    'score': best['score_mean'],
                    'score_std': best['score_std'],
                    'allocation': best['blueprint'].allocation_method,
                    'importance': best['blueprint'].importance_method,
                    'n_features': best['n_features'],
                } if best else None

                # Also extract best-by-model for richer comparison
                best_by_model = {}
                for model_name, mbest in search.best_by_model.items():
                    best_by_model[model_name] = {
                        'score': mbest['score_mean'],
                        'budget': mbest['blueprint'].budget,
                        'importance': mbest['blueprint'].importance_method,
                    }

            except Exception as e:
                print(f"    {subpop_name}: find_best_hybrid failed - {e}")
                best_config = None
                best_by_model = {}

            # Second pass: apply the full-data pool-best blueprint per model
            # to this subpopulation (re-fit on subpop data only) and record.
            fulldata_bp_scores = {}
            # FIX 4d: this partitioned sub-population can have fewer distinct
            # scaffold groups than the fold count, and cv_split RAISES rather
            # than fall back to a random split. Reduce the fold count to the
            # available groups (min 2); if the sub-population has fewer than 2
            # distinct groups, skip the second pass for it and record an
            # explicit flag rather than crash or silently emit nothing.
            fulldata_bp_skipped = None
            from sklearn.preprocessing import StandardScaler as _SS
            from experiments._shared import cv_split as _cv_sp7, get_model_factories as _gmf
            n_folds = config.get('n_splits', 5)
            n_sub_groups = len(np.unique(sub_groups))
            sub_folds = min(n_folds, max(2, n_sub_groups))
            sub_sample_X = next(iter(sub_reps.values()))
            if n_sub_groups < 2:
                fulldata_bp_skipped = (
                    f'insufficient_groups: {n_sub_groups} distinct group(s) '
                    f'< 2 required for fulldata-bp CV')
            for cand_model, (bp_fd, _t, _a) in (
                    fulldata_blueprints.items() if fulldata_bp_skipped is None else []):
                if bp_fd.budget is not None and bp_fd.budget > len(sub_labels):
                    continue
                try:
                    factory = _gmf([cand_model], task_type=data['task_type'],
                                   n_samples=len(sub_labels)).get(cand_model)
                    if factory is None:
                        continue
                    fold_scores = []
                    try:
                        _sp7_splits = list(_cv_sp7(sub_groups, sub_folds,
                                                   sub_sample_X, sub_labels))
                    except ValueError:
                        # Group count below the (reduced) fold count after
                        # StratifiedGroupKFold's own checks — flag + skip.
                        fulldata_bp_skipped = (
                            'insufficient_groups: cv_split rejected '
                            'fulldata-bp folds')
                        break
                    for tr_i, va_i in _sp7_splits:
                        tr_reps_f = {k: v[tr_i] for k, v in sub_reps.items()}
                        va_reps_f = {k: v[va_i] for k, v in sub_reps.items()}
                        tr_augs_f = ({k: v[tr_i] for k, v in sub_augs.items()}
                                     if sub_augs else None)
                        va_augs_f = ({k: v[va_i] for k, v in sub_augs.items()}
                                     if sub_augs else None)
                        from experiments._shared import build_fold_hybrid as _bfh
                        X_tr, X_va, _fi = _bfh(
                            tr_reps_f, va_reps_f, sub_labels[tr_i], bp_fd,
                            augmentations_train=tr_augs_f,
                            augmentations_val=va_augs_f,
                        )
                        sc = _SS()
                        X_tr_s = sc.fit_transform(X_tr)
                        X_va_s = sc.transform(X_va)
                        mdl = factory()
                        mdl.fit(X_tr_s, sub_labels[tr_i])
                        from experiments._shared import get_domain_metric as _gdm
                        _, metric_fn_part, _ = _gdm(data)
                        fold_scores.append(metric_fn_part(sub_labels[va_i], mdl.predict(X_va_s)))
                    if fold_scores:
                        fulldata_bp_scores[cand_model] = float(np.mean(fold_scores))
                except Exception:
                    pass

            part_result[subpop_name] = {
                'metrics': metrics,
                'best': best_config,
                'best_by_model': best_by_model,
                'baselines': search.baselines if (best_config and search is not None) else {},
                'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
                'fulldata_bp_scores': fulldata_bp_scores,
                # FIX 4d: None when the fulldata-bp second pass ran normally,
                # else a string explaining why it was skipped (too few groups).
                'skipped_insufficient_groups': fulldata_bp_skipped,
            }
            if best_config:
                print(f"    {subpop_name} best: {best_config['model']}/b={best_config['budget']} "
                      f"score={best_config['score']:.4f} "
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
