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
from kirby.hybrid import _graded_scores_for_clf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    own_search_kwargs,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results, convert_numpy,
    get_model_factories, get_domain_metric,
    estimate_mutual_information_knn, compute_effective_rank,
    compute_intrinsic_dimensionality,
    find_best_hybrid,
    extract_top3_base_reps,
    select_model_rep_pairs,
    INTERNAL_EVALUATORS,
    build_fold_reselection_input,
    get_rule_hybrids, top_hybrids, fold_hybrid_scoring_kwargs,
    DOWNSTREAM_MODELS, DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DISTANCE_MODELS,
    RULE_N_GENERAL, RULE_N_DISTANCE,
)


def _rule_from_searches(gen_search, dist_search, model, *, dist_error=None):
    """Owner's set (2026-10-02): top 2 general + top 1 distance hybrids for
    one downstream model, read off this file's own two searches on one
    subpopulation.

    Returns (entries, missing). Each entry is a top_hybrids 'selected' entry
    without its Blueprint, plus 'slot' ('general1', 'general2', 'distance1'),
    'lineage', 'tier' and 'objective'. Every slot asked for and not filled is
    a sentence in `missing`.
    """
    entries, missing = [], []
    for lineage, srch, n, tier in (
            ('general', gen_search, RULE_N_GENERAL, 'general'),
            ('distance', dist_search, RULE_N_DISTANCE, 'general_distance')):
        if lineage == 'distance' and model not in DISTANCE_MODELS:
            missing.append(f'distance for {model}: {model} is not one of the '
                           f'distance search models {DISTANCE_MODELS}, so it '
                           f'was never scored there')
            continue
        if srch is None:
            missing.append(f'{lineage} for {model}: no search '
                           f'({(dist_error if lineage == "distance" else None) or "it raised or did not run"})')
            continue
        try:
            th = top_hybrids(srch, model, n, require_full=False)
        except Exception as e:
            missing.append(f'{lineage} for {model}: top_hybrids raised '
                           f'{type(e).__name__}: {e}')
            continue
        if not th['complete']:
            missing.append(f'{lineage} for {model}: {th["shortfall_reason"]}')
        for e in th['selected']:
            e = {k: v for k, v in e.items() if k != 'blueprint'}
            e['slot'] = f"{lineage}{e['rank']}"
            e['lineage'] = lineage
            e['tier'] = tier
            e['objective'] = 'predictive'
            e['budget_relaxed'] = False
            entries.append(e)
    return entries, missing


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
    metric_name, _, lower_is_better = get_domain_metric(data)

    # 2026-10-05, owner delegated, 2026-10-05: a dataset with fewer than 10
    # scaffold/sequence groups in total (the groups array every partition below
    # splits by) is skipped. abagym_Ang2/EGFR/VEGF and the four ginkgo datasets
    # have 5. Halving 5 groups leaves too few groups per half to
    # cross-validate, and splitting inside a group would break the
    # no-random-split rule. The {'skipped': True} marker is what
    # classify_result_payload (_shared.py) reads as an intended skip.
    _n_groups_total = int(len(np.unique(np.asarray(groups))))
    if _n_groups_total < 10:
        print(f"  [skip] {_n_groups_total} groups < 10: partitions not run")
        return {
            'skipped': True,
            'reason': (f'{_n_groups_total} scaffold/sequence groups; whole-group '
                       'draws/halves cannot be cross-validated and splitting '
                       'inside groups would break the no-random-split rule'),
            'n_groups': _n_groups_total,
            'n_samples': int(len(labels)),
        }

    # For molecular datasets, use smiles; for antibody, this partition may not apply
    smiles = data.get('smiles')

    # 'feature_variance' is the label-free counterpart of 'target_variance':
    # it splits scaffold groups by how spread out their rows are in
    # representation space, so nothing about the property being predicted
    # decides which half a row lands in. 'target_variance' is kept, but its
    # easy/hard membership comes from the same labels the two halves are
    # scored on, so its gap is not a generalisation measure — every
    # target_variance block carries partition_uses_labels=True saying so.
    all_partitions = partitions or ['scaffold_frequency', 'molecular_complexity',
                                    'target_variance', 'feature_variance']
    results = {'partitions': {},
               # Why a partition is absent from 'partitions', added 2026-10-05.
               # {partition name: sentence}. 'partitions' came back {} on some
               # datasets with nothing saying which split failed which minimum.
               'partition_skips': {}}

    # Second-pass scaffolding (per user 2026-05-23): in addition to per-partition
    # find_best_hybrid (the experiment's main signal), record what a blueprint
    # chosen on EVERY row scores on each partition. Its allocation method,
    # importance method, internal evaluator and budget were picked using every
    # label, including the rows each partition later holds out; only the
    # feature selection is redone per partition. So the scores under
    # 'all_rows_bp_scores' are not held-out scores.
    all_rows_blueprints = {}
    full_augs_all = data.get('augmentations')
    # Owner's set (2026-10-02): top 2 general + top 1 distance hybrids per
    # model, per family, read for the whole dataset. This replaces the single
    # pool winner per model; general1 keeps the old keys.
    # {model: [(slot, blueprint, tier, aug_condition), ...]}
    all_rows_rule_bps = {}
    results['rule_missing'] = []
    results['rule'] = ('top 2 general + top 1 distance hybrids per model; '
                       'base side top 2 single representations')
    for cand_model in ['rf', 'svm', 'lightgbm', 'dnn']:
        try:
            _rule = get_rule_hybrids(
                reps, labels, groups,
                downstream_model=cand_model, budget=None,
                augmentations=full_augs_all,
                dataset_name=dataset_name,
                task_type=data['task_type'], metric=metric_name,
                data=data,
            )
        except Exception as _e:
            results['rule_missing'].append(
                f'all-rows pass for {cand_model}: the pool read raised '
                f'{type(_e).__name__}: {_e}')
            continue
        results['rule_missing'] += [f'all-rows pass: {m}'
                                    for m in _rule['missing']]
        all_rows_rule_bps[cand_model] = [
            (e['slot'], e['blueprint'], e['tier'], e['aug_condition'])
            for e in _rule['hybrids'] if e.get('blueprint') is not None]
        for _sl, _bp, _t, _a in all_rows_rule_bps[cand_model]:
            if _sl == 'general1':
                all_rows_blueprints[cand_model] = (_bp, _t, _a)
    results['all_rows_blueprints_meta'] = {
        m: {'tier': t, 'aug_condition': a, 'budget': bp.budget,
            'allocation': bp.allocation_method, 'importance': bp.importance_method}
        for m, (bp, t, a) in all_rows_blueprints.items()
    }
    # Every slot of the owner's set, same fields, per model.
    results['all_rows_rule_meta'] = {
        m: [{'slot': sl, 'tier': t, 'aug_condition': a, 'budget': bp.budget,
             'allocation': bp.allocation_method,
             'importance': bp.importance_method,
             'internal_evaluator': bp.internal_evaluator}
            for sl, bp, t, a in lst]
        for m, lst in all_rows_rule_bps.items()
    }

    # c07 large-dataset cap (signed off 2026-06-05): the per-subpopulation
    # searches below are full, uncached find_best_hybrid runs (uncapped once a
    # half is large), so on big datasets they grow into the
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
                results['partition_skips'][partition_name] = (
                    'no SMILES on this dataset, so no heavy-atom count')
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
                    results['partition_skips'][partition_name] = (
                        'no SMILES parsed with RDKit')
                    continue
                median_hac = np.median(hac[valid_mask])
                mask_easy = valid_mask & (hac <= median_hac)
                mask_hard = valid_mask & (hac > median_hac)
            except ImportError:
                print("    RDKit not available, skipping molecular_complexity")
                results['partition_skips'][partition_name] = (
                    'RDKit not importable')
                continue

        elif partition_name == 'target_variance':
            # Scaffolds with low target variance vs high. The label decides
            # which half a row goes to, and the label spread is also part of
            # what the two halves are then compared on, so the easy/hard gap
            # here measures the split as much as it measures generalisation.
            # Flagged in the output; see 'feature_variance' for the label-free
            # version of the same question.
            unique_scaffolds = np.unique(groups)
            scaffold_var = {}
            for s in unique_scaffolds:
                s_labels = labels[groups == s]
                if len(s_labels) >= 2:
                    scaffold_var[s] = np.var(s_labels)
            if not scaffold_var:
                print("    Not enough scaffolds with 2+ members, skipping")
                results['partition_skips'][partition_name] = (
                    'no scaffold group has 2 or more molecules, so no group '
                    'has a label variance')
                continue
            median_var = np.median(list(scaffold_var.values()))
            mask_easy = np.array([scaffold_var.get(g, 0) <= median_var for g in groups])
            mask_hard = ~mask_easy

        elif partition_name == 'feature_variance':
            # Scaffolds whose rows sit close together in representation space
            # vs spread out. Uses no label at any step, so membership carries
            # nothing about the property being predicted. Every representation
            # is z-scored first so a wide one does not set the spread on its
            # own, then a group's spread is the mean per-feature variance of
            # its rows.
            unique_scaffolds = np.unique(groups)
            X_all = np.hstack([reps[k] for k in sorted(reps.keys())])
            X_all = np.asarray(X_all, dtype=float)
            _sd = X_all.std(axis=0)
            _sd[_sd == 0] = 1.0
            X_all = (X_all - X_all.mean(axis=0)) / _sd
            scaffold_spread = {}
            for s in unique_scaffolds:
                rows = X_all[groups == s]
                if len(rows) >= 2:
                    scaffold_spread[s] = float(np.mean(np.var(rows, axis=0)))
            if not scaffold_spread:
                print("    Not enough scaffolds with 2+ members, skipping")
                results['partition_skips'][partition_name] = (
                    'no scaffold group has 2 or more molecules, so no group '
                    'has a feature spread')
                continue
            median_spread = np.median(list(scaffold_spread.values()))
            mask_easy = np.array(
                [scaffold_spread.get(g, 0.0) <= median_spread for g in groups])
            mask_hard = ~mask_easy
        else:
            print(f"    Unknown partition: {partition_name}, skipping")
            results['partition_skips'][partition_name] = 'unknown partition name'
            continue

        n_easy = mask_easy.sum()
        n_hard = mask_hard.sum()
        print(f"    easy={n_easy}, hard={n_hard}")

        if n_easy < 50 or n_hard < 50:
            print("    Subpopulation too small, skipping")
            results['partition_skips'][partition_name] = (
                f'easy half {int(n_easy)} rows, hard half {int(n_hard)} rows; '
                f'each half needs at least 50')
            continue

        part_result = {'easy': {}, 'hard': {}, 'n_easy': int(n_easy), 'n_hard': int(n_hard),
                       # True when the label decided which half each row went
                       # to. The easy/hard gap is then partly the split itself.
                       'partition_uses_labels': partition_name == 'target_variance'}

        full_augs = _c07_augs  # c07 cap: subsampled augmentations (see C07_MAX_SAMPLES)
        for subpop_name, mask in [('easy', mask_easy), ('hard', mask_hard)]:
            sub_reps = {k: v[mask] for k, v in reps.items()}
            sub_labels = labels[mask]
            sub_groups = groups[mask]
            sub_augs = {k: v[mask] for k, v in full_augs.items()} if full_augs else None
            n_unique_groups = len(np.unique(sub_groups))

            if n_unique_groups < 2:
                print(f"    {subpop_name}: not enough groups for CV, skipping")
                # Recorded 2026-10-05; this half was left out with no entry.
                part_result[subpop_name] = {
                    'skipped': (f'{int(n_unique_groups)} scaffold group in '
                                f'{int(mask.sum())} rows; group-aware folds '
                                f'need at least 2'),
                }
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
            # The 250-row gate was removed 2026-08-27. Always capped.
            n_sub = int(mask.sum())
            # Always capped; the 250-row gate was removed 2026-08-27.
            sub_aug_strategy = 'capped'
            top3 = []
            # Initialize before the try so the `search.baselines` read further
            # down (part_result['baselines']) can never NameError if a future
            # refactor makes best_config truthy on a path where the search
            # raised. Today it's guarded by the `if best_config` short-circuit,
            # but None is the safe default. (CXA-17)
            search = None
            # Read after this block (part_result below) but assigned only
            # inside the try, so a search that raised -- e.g. refused by
            # KIRBY_NO_COLD_SEARCH -- crashed c07 with "UnboundLocalError:
            # ... 'sel'" (server check, 2026-09-29).
            sel = None
            search_error = None
            try:
                search = find_best_hybrid(
                    sub_reps, sub_labels, groups=sub_groups,
                    budgets=budgets,
                    augmentations=sub_augs,
                    augmentation_strategy=sub_aug_strategy,
                    metric=metric_name,
                    task_type=data['task_type'],
                    verbose=False,
                    cache=True,
                    dataset_name=f"{dataset_name}_{partition_name}_{subpop_name}",
                    tier='general',
                    # Its own search on one subpopulation: fast grid, adaptive
                    # pruning, the five downstream models named, not refused by
                    # KIRBY_NO_COLD_SEARCH. Owner, 2026-09-29.
                    **own_search_kwargs(len(sub_labels)),
                )
                top3 = extract_top3_base_reps(search)
                if top3:
                    print(f"    {subpop_name} Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

                # ADDED 2026-09-18, beside top3 rather than instead of it.
                # `top3` is 3 base representations ranked for rf alone, and it
                # does not say so in the key it is stored under. `sel` is the
                # top 2 base representations AND the top 2 hybrids for EVERY
                # downstream model, each row carrying the model it belongs to.
                #
                # 0 new model fits: both halves read rows the search stored.
                #
                # top3 is left in place and still written, so `delta_budget` at
                # c07:296-306 and `config_differs` at c07:453-457 keep reading
                # the winner they read today and nothing on disk moves.
                sel = None
                try:
                    sel = select_model_rep_pairs(
                        search, n_base=2, n_hybrids=2, require_full=False,
                        context=f'c07 {dataset_name}/{partition_name}/{subpop_name}')
                except Exception as _e:
                    print(f"    {subpop_name}: top-2 selection unavailable "
                          f"({type(_e).__name__}: {_e})")

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
                # Saved 2026-10-05, the way distance_search_error below is.
                search_error = f'{type(e).__name__}: {e}'

            # Owner's set (2026-10-02): top 2 general + top 1 distance hybrids
            # per model. The distance counterpart of the search above, on the
            # same subpopulation rows: tier 'general_distance', the distance
            # evaluators and the distance budgets (10 to 500 features). The
            # five downstream models it scores are the ones the general search
            # above scores. The cross-model winner 'best' above is what the
            # experiment compares between halves and is left as it was.
            search_d = None
            dist_error = None
            _dist_models = [m for m in DOWNSTREAM_MODELS if m in DISTANCE_MODELS]
            try:
                _okw = own_search_kwargs(len(sub_labels), models=_dist_models)
                _okw['internal_evaluators'] = list(DISTANCE_EVALUATORS)
                search_d = find_best_hybrid(
                    sub_reps, sub_labels, groups=sub_groups,
                    budgets=list(DISTANCE_BUDGETS),
                    augmentations=sub_augs,
                    augmentation_strategy=sub_aug_strategy,
                    metric=metric_name,
                    task_type=data['task_type'],
                    verbose=False,
                    cache=True,
                    dataset_name=f"{dataset_name}_{partition_name}_{subpop_name}",
                    tier='general_distance',
                    **_okw,
                )
            except Exception as e:
                dist_error = f'{type(e).__name__}: {e}'
                print(f"    {subpop_name}: distance find_best_hybrid failed - {e}")
            rule_hybrids = {}
            for _m in DOWNSTREAM_MODELS:
                _ents, _miss = _rule_from_searches(search, search_d, _m,
                                                   dist_error=dist_error)
                rule_hybrids[_m] = _ents
                results['rule_missing'] += [
                    f'{partition_name}/{subpop_name}: {m}' for m in _miss]
            reselection_block_distance = None
            if search_d is not None:
                try:
                    reselection_block_distance = build_fold_reselection_input(
                        search_d)
                except Exception as _e:
                    print(f"    {subpop_name}: distance fold_reselection "
                          f"unavailable ({_e})")

            # Second pass: apply the all-rows-selected blueprint per model to
            # this subpopulation, re-selecting only its features here.
            all_rows_bp_scores = {}
            # Per-fold scores for the all-rows-bp second pass, kept PARALLEL to
            # the scalar means above (means unchanged for existing readers). Lets
            # the reader apply the same held-out-fold honest correction to this
            # arm too, per model — no cross-model collapse.
            all_rows_bp_fold_scores = {}
            # Owner's set (2026-10-02): the same pass once per slot.
            # {model: {slot: score}}; general1 also fills the two dicts above.
            all_rows_bp_scores_rule = {}
            all_rows_bp_fold_scores_rule = {}
            # FIX 4d: this partitioned sub-population can have fewer distinct
            # scaffold groups than the fold count, and cv_split RAISES rather
            # than fall back to a random split. Reduce the fold count to the
            # available groups (min 2); if the sub-population has fewer than 2
            # distinct groups, skip the second pass for it and record an
            # explicit flag rather than crash or silently emit nothing.
            all_rows_bp_skipped = None
            from sklearn.preprocessing import StandardScaler as _SS
            from experiments._shared import cv_split as _cv_sp7, get_model_factories as _gmf
            n_folds = config.get('n_splits', 5)
            n_sub_groups = len(np.unique(sub_groups))
            sub_folds = min(n_folds, max(2, n_sub_groups))
            sub_sample_X = next(iter(sub_reps.values()))
            if n_sub_groups < 2:
                all_rows_bp_skipped = (
                    f'insufficient_groups: {n_sub_groups} distinct group(s) '
                    f'< 2 required for all-rows-bp CV')
            # Owner's set (2026-10-02): every slot, not the single winner.
            _flat_slots = [(cm, sl, bp) for cm, lst in (
                all_rows_rule_bps.items() if all_rows_bp_skipped is None else [])
                for sl, bp, _t, _a in lst]
            # Features this half offers: every base representation's width
            # plus every augmentation block's. Used by the guard below.
            _n_feat_offered = (sum(int(v.shape[1]) for v in sub_reps.values())
                               + (sum(int(v.shape[1]) for v in sub_augs.values())
                                  if sub_augs else 0))
            # Why a slot has no all-rows score here. {model: {slot: sentence}}.
            all_rows_bp_slot_errors = {}
            # Features the rebuilt hybrid has per fold, beside its budget.
            all_rows_bp_widths_rule = {}
            for cand_model, _slot, bp_fd in _flat_slots:
                # Before 2026-10-05 this compared the budget, a feature count,
                # with len(sub_labels), a row count, and skipped silently. It
                # guards a budget the half cannot supply, so it now compares
                # features with features, and says so when it skips.
                # find_best_hybrid builds budgets above the row count since
                # 2026-09-10, so rows are not a limit here.
                if bp_fd.budget is not None and bp_fd.budget > _n_feat_offered:
                    all_rows_bp_slot_errors.setdefault(cand_model, {})[_slot] = (
                        f'budget {int(bp_fd.budget)} features is more than the '
                        f'{_n_feat_offered} features this half offers')
                    continue
                try:
                    factory = _gmf([cand_model], task_type=data['task_type'],
                                   n_samples=len(sub_labels)).get(cand_model)
                    if factory is None:
                        all_rows_bp_slot_errors.setdefault(cand_model, {})[
                            _slot] = ('get_model_factories returned no factory '
                                      f'at {len(sub_labels)} rows')
                        continue
                    fold_scores = []
                    fold_widths = []
                    try:
                        _sp7_splits = list(_cv_sp7(sub_groups, sub_folds,
                                                   sub_sample_X, sub_labels))
                    except ValueError:
                        # Group count below the (reduced) fold count after
                        # StratifiedGroupKFold's own checks — flag + skip.
                        all_rows_bp_skipped = (
                            'insufficient_groups: cv_split rejected '
                            'all-rows-bp folds')
                        break
                    for tr_i, va_i in _sp7_splits:
                        tr_reps_f = {k: v[tr_i] for k, v in sub_reps.items()}
                        va_reps_f = {k: v[va_i] for k, v in sub_reps.items()}
                        tr_augs_f = ({k: v[tr_i] for k, v in sub_augs.items()}
                                     if sub_augs else None)
                        va_augs_f = ({k: v[va_i] for k, v in sub_augs.items()}
                                     if sub_augs else None)
                        from experiments._shared import build_fold_hybrid as _bfh
                        from experiments._shared import get_domain_metric as _gdm
                        _, metric_fn_part, _lib_part = _gdm(data)
                        # Without groups, create_hybrid judges keep/drop on a
                        # RandomState(42) 20% split of these training rows, so
                        # a scaffold sits on both sides of it. Without the
                        # model and the dataset's own metric it judges them by
                        # r2_score from the internal evaluator instead of the
                        # estimator and metric this fold is scored with. Both
                        # change which features are chosen, so the blueprint
                        # rebuilt here stopped matching c05's rebuild.
                        X_tr, X_va, _fi = _bfh(
                            tr_reps_f, va_reps_f, sub_labels[tr_i], bp_fd,
                            augmentations_train=tr_augs_f,
                            augmentations_val=va_augs_f,
                            groups=sub_groups[tr_i],
                            **fold_hybrid_scoring_kwargs(
                                bp_fd, model=factory(),
                                metric_fn=metric_fn_part,
                                lower_is_better=_lib_part),
                        )
                        fold_widths.append(int(X_tr.shape[1]))
                        sc = _SS()
                        X_tr_s = sc.fit_transform(X_tr)
                        X_va_s = sc.transform(X_va)
                        # 2026-10-05, owner delegated, 2026-10-05: each
                        # validation feature is capped at the range it takes
                        # on the scaled training rows. On adme_hlm one
                        # validation molecule sat 83 million training standard
                        # deviations out and the dnn's rmse ran to thousands.
                        X_va_s = np.clip(X_va_s, X_tr_s.min(0), X_tr_s.max(0))
                        mdl = factory()
                        mdl.fit(X_tr_s, sub_labels[tr_i])
                        # Graded scores for ranking metrics on classification,
                        # as c05's second pass does; an AUROC from 0/1 labels is
                        # balanced accuracy (fixed 2026-10-05).
                        _hard = mdl.predict(X_va_s)
                        _y = (_graded_scores_for_clf(mdl, X_va_s, _hard)
                              if data.get('task_type') == 'classification'
                              else _hard)
                        fold_scores.append(metric_fn_part(sub_labels[va_i], _y))
                    if fold_scores:
                        all_rows_bp_scores_rule.setdefault(cand_model, {})[
                            _slot] = float(np.mean(fold_scores))
                        all_rows_bp_fold_scores_rule.setdefault(cand_model, {})[
                            _slot] = [float(s) for s in fold_scores]
                        all_rows_bp_widths_rule.setdefault(cand_model, {})[
                            _slot] = {'budget': bp_fd.budget,
                                      'n_features_per_fold': fold_widths}
                        if _slot == 'general1':
                            all_rows_bp_scores[cand_model] = float(np.mean(fold_scores))
                            all_rows_bp_fold_scores[cand_model] = [
                                float(s) for s in fold_scores]
                except Exception as _e:
                    # Recorded 2026-10-05; this was `pass`.
                    all_rows_bp_slot_errors.setdefault(cand_model, {})[_slot] = (
                        f'{type(_e).__name__}: {_e}')

            # Honest per-model rescore block (CONTRACT json_convention). Carries
            # per-fold scores for BOTH arms — hybrid candidate recipes AND
            # baseline reps — keyed per model, so the reader can apply the
            # leave-one-fold-out selection-optimism correction without re-running
            # a search. This is the PER-MODEL replacement for the single
            # cross-model argmax `best` above: every evaluated model is retained,
            # nothing collapses to one max cell. build_fold_reselection_input reads
            # search.baseline_fold_scores (per-fold baselines) and
            # search.all_results (per-fold hybrid candidates); on old caches that
            # lack per-fold baselines a model simply won't appear (soft miss).
            reselection_block = (build_fold_reselection_input(search)
                            if search is not None else None)

            part_result[subpop_name] = {
                'metrics': metrics,
                'best': best_config,
                'best_by_model': best_by_model,
                'baselines': search.baselines if (best_config and search is not None) else {},
                # Ranked for rf alone. The name does not say so; it is kept
                # for the readers that already parse it.
                'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
                # Top 2 base representations and top 2 hybrids PER downstream
                # model, each row naming its model. Added 2026-09-18.
                'top2_pairs_by_model': (sel['pairs'] if sel else []),
                'all_rows_bp_scores': all_rows_bp_scores,
                'all_rows_bp_fold_scores': all_rows_bp_fold_scores,
                # Owner's set (2026-10-02): top 2 general + top 1 distance
                # hybrids per model. The all-rows pass per slot, and this
                # half's own two searches read per slot, each entry with its
                # slot, tier and cross-validated score.
                'all_rows_bp_scores_rule': all_rows_bp_scores_rule,
                'all_rows_bp_fold_scores_rule': all_rows_bp_fold_scores_rule,
                # 2026-10-05: the width each slot's rebuild had per fold, and
                # why a slot has no score.
                'all_rows_bp_widths_rule': all_rows_bp_widths_rule,
                'all_rows_bp_slot_errors': all_rows_bp_slot_errors,
                'rule_hybrids': rule_hybrids,
                'fold_reselection_distance': reselection_block_distance,
                'search_error': search_error,
                'distance_search_error': dist_error,
                # Per-model per-fold reselection inputs (per-fold, both arms). Exact
                # CONTRACT keys inside: lower_is_better / candidates_by_model /
                # baselines_by_model. None if the search itself failed.
                'fold_reselection': reselection_block,
                # FIX 4d: None when the all-rows-bp second pass ran normally,
                # else a string explaining why it was skipped (too few groups).
                'skipped_insufficient_groups': all_rows_bp_skipped,
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
                        choices=['scaffold_frequency', 'molecular_complexity',
                                 'target_variance', 'feature_variance'],
                        help='Which partitions to test (default: all)')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, partitions=args.partitions)
    save_experiment_results(results, args.output, 'c07_data_complexity_partitions', args.dataset)
