#!/usr/bin/env python3
"""C05: Learning curves -- sample efficiency.

Paper: Complexity
Conclusion: Hybrids are more sample-efficient than individual reps.
Dependencies: None (standalone).

At each sample size fraction, runs find_best_hybrid on subsample to discover
the optimal config for that data regime. Tests whether the methodology adapts
to different data scales and whether the hybrid advantage persists.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, evaluate_single_rep_cv, fit_learning_curve,
    find_best_hybrid, get_domain_metric,
    extract_top3_base_reps,
    INTERNAL_EVALUATORS,
    get_best_search_for_downstream, build_fold_hybrid, cv_split,
    build_honest_rescore_input, honest_rescore,
    hr_exclude_folds,
)
from sklearn.preprocessing import StandardScaler


def run(data, config, output_dir, dataset_name, **kwargs):
    """Learning curves: hybrid advantage vs sample size."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C05: LEARNING CURVES")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    models = get_model_factories(config['models'],
                                    task_type=data['task_type'],
                                    n_samples=len(data['labels']))
    n_folds = config['n_splits']
    # Canonical default (FULL_BUDGETS) — old default [25, 50, 100, 200, 500] had
    # 200 which is NOT in FULL_BUDGETS, causing silent blueprint misses.
    from experiments._shared import FULL_BUDGETS
    sweep_budgets = config.get('budgets', FULL_BUDGETS)
    n_repeats = kwargs.get('n_repeats', 3)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_total = len(labels)
    model_names = list(models.keys())

    from experiments._shared import LARGE_DATASET_N, LARGE_SUBSAMPLE_K
    if n_total > LARGE_DATASET_N:
        # Large dataset: keep a multi-point curve but cap it at K rows so we
        # never run the giant full-data searches. Absolute sizes, not %-of-n.
        sample_sizes = [s for s in [250, 500, 1000, LARGE_SUBSAMPLE_K]
                        if s < n_total]
        print(f"  [size-gate] n={n_total} > {LARGE_DATASET_N}: learning curve "
              f"capped at {LARGE_SUBSAMPLE_K} rows ({sample_sizes})")
    else:
        fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
        sample_sizes = [max(50, int(n_total * f)) for f in fractions]
    # Deduplicate and cap at total
    sample_sizes = sorted(set(min(s, n_total) for s in sample_sizes))

    print(f"  Models: {model_names}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Budgets: {sweep_budgets}")
    print(f"  Repeats: {n_repeats}")
    print(f"  Search ")

    last_search = None
    results = {'sample_sizes': sample_sizes, 'budgets': sweep_budgets,
                'models': {},
                # FIX 4d: explicit log of (n, model) all-rows-blueprint
                # second-pass points that had too few scaffold groups for the
                # requested fold count. cv_split raises (no random-split
                # fallback) on groups < fold count, so rather than let that
                # propagate or be swallowed into an empty score we reduce folds
                # to the available groups (min 2) or record a skip here.
                'skipped_insufficient_groups': [],
                # Honest per-fold rescore blocks (CONTRACT shape) nested by
                # sample size. For each sample size we store one block PER
                # subsampling seed (subsamples differ across seeds, so their
                # per-fold scores must not be merged into one candidate list).
                # Each block is exactly what build_honest_rescore_input returns
                # (candidates_by_model + baselines_by_model + lower_is_better),
                # so a reader can feed each straight to honest_rescore. Purely
                # additive: the mean-based 'hybrid'/'best_single' arrays are
                # UNCHANGED.
                'honest_rescore_by_n': {},
                # The sample-efficiency curve to read. At each sample size,
                # each held-out fold's hybrid recipe and single representation
                # are picked on the other folds only, so both sides pay the
                # same tax for being a maximum. The 'hybrid'/'best_single'
                # arrays below are each a maximum taken on the folds they are
                # then reported on, over ~600 recipes against ~10
                # representations, so their difference runs high.
                'honest_curve': {}}

    # Initialize per-model result structure
    for model_name in model_names:
        results['models'][model_name] = {
            'hybrid': [], 'best_single': [],
            # Leave-one-fold-out arrays, aligned with sample_sizes. Read these.
            'hybrid_honest': [], 'best_single_honest': [],
            'honest_delta_mean_of_folds': [],
            'best_configs': [],
            # Second pass: the all-rows-selected blueprint applied at each n.
            'hybrid_all_rows_bp': [],
            'all_rows_bp_meta': [],
        }

    # The allocation method, importance method, internal evaluator and budget
    # below are chosen once by scoring every row of the dataset, so every label
    # a subsample later holds out was used to pick them. At each subsample only
    # the feature selection is redone, on that subsample's rows. So the scores
    # under 'hybrid_all_rows_bp' are not held-out scores: they answer "how does
    # a recipe picked with all the labels do at size n", not "how does a recipe
    # picked at size n generalise".
    all_rows_blueprints = {}
    all_rows_meta = {}
    full_augs_top = data.get('augmentations')
    for model_name in model_names:
        try:
            _s_fd, _m_fd = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=model_name, budget=None,
                augmentations=full_augs_top, 
                dataset_name=dataset_name,
                task_type=data['task_type'], metric=metric_name,
                data=data,
            )
            if _m_fd:
                all_rows_blueprints[model_name] = _m_fd['best_config']['blueprint']
                all_rows_meta[model_name] = {
                    'tier': _m_fd['tier'], 'aug_condition': _m_fd['aug_condition'],
                    'budget': _m_fd['best_config']['blueprint'].budget,
                }
                print(f"  All-rows-selected BP for {model_name}: tier={_m_fd['tier']} "
                      f"aug={_m_fd['aug_condition']} budget={_m_fd['best_config']['blueprint'].budget}")
        except Exception as e:
            print(f"  All-rows-selected BP for {model_name}: pool helper failed ({e})")

    for n_sub in sample_sizes:
        print(f"\n  n={n_sub}:")
        # Collect scores across repeats per model
        hybrid_by_model = {m: [] for m in model_names}
        single_by_model = {m: [] for m in model_names}
        configs_by_model = {m: [] for m in model_names}
        # Second pass: the all-rows-selected blueprint, features re-selected
        # on the subsample. The recipe itself still came from every label.
        hybrid_fd_by_model = {m: [] for m in model_names}
        # One CONTRACT-shape honest_rescore block per seed at this sample size.
        honest_blocks_this_n = []

        last_search = None
        full_augs = data.get('augmentations')
        # Pre-compute group-level pool so subsampling preserves scaffold
        # integrity (subsampling individual rows shatters scaffold groups
        # across train/val of the same fold).
        _unique_groups = np.unique(groups)
        _group_to_idx = {g: np.where(groups == g)[0] for g in _unique_groups}
        # FIX 3h: at the full-data point the group sampling below picks EVERY
        # group regardless of seed, so all n_repeats seeds draw the identical
        # dataset and run the identical (expensive) search under different cache
        # names. The mean of n_repeats identical runs is just the value, so
        # collapse to a single seed here. <100% fractions still use n_repeats.
        n_repeats_eff = 1 if n_sub >= n_total else n_repeats
        for seed in range(n_repeats_eff):
            rng = np.random.RandomState(seed)
            # Draw whole scaffold groups until we hit ~n_sub samples (the
            # exact size is approximate; group sizes vary).
            shuffled_groups = rng.permutation(_unique_groups)
            picked, picked_count = [], 0
            for g in shuffled_groups:
                gi = _group_to_idx[g]
                if picked_count + len(gi) > n_sub and picked:
                    break
                picked.extend(gi.tolist())
                picked_count += len(gi)
                if picked_count >= n_sub:
                    break
            idx = np.array(sorted(picked))
            sub_reps = {k: v[idx] for k, v in reps.items()}
            sub_labels = labels[idx]
            sub_groups = groups[idx]
            sub_augs = {k: v[idx] for k, v in full_augs.items()} if full_augs else None
            n_unique_groups = len(np.unique(sub_groups))

            if n_unique_groups < 2:
                continue

            # Re-resolve model factories at this subsample size so the
            # _max_samples gate fires consistently — SVM is excluded forever
            # if we use the full-N factory dict on every subsample.
            sub_models = get_model_factories(
                config['models'], task_type=data['task_type'],
                n_samples=len(idx))

            sub_folds = min(n_folds, n_unique_groups)

            # find_best_hybrid discovers optimal config per model at this sample size.
            # Cap budgets to the TRAINING partition, not the whole subsample: a
            # budget > train rows yields a degenerate selection -> NaN (this was
            # the all-NaN-at-small-n bug, e.g. budget 50 at n_sub=50 leaves only
            # ~40 train rows). See COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
            train_size = (int(n_sub * (sub_folds - 1) / sub_folds)
                          if sub_folds > 1 else int(n_sub * 0.8))
            sub_budgets = [b for b in sweep_budgets if b <= train_size]
            if not sub_budgets:
                # smallest sweep budget that still fits the training partition
                sub_budgets = [min((b for b in sweep_budgets if b <= train_size),
                                   default=max(2, train_size))]
            # Subsample-specific cache key — won't collide with main precompute
            # cache. Capped strategy is fine for small subsamples but projects
            # The 250-row gate was removed 2026-08-27. Always capped;
            # uncapped when the subsample is over the threshold.
            # Always capped. The 250-row gate was removed 2026-08-27; the
            # rebuilt capped costs about a tenth of the version it was
            # written for.
            sub_aug_strategy = 'capped'
            try:
                search = find_best_hybrid(
                    sub_reps, sub_labels, groups=sub_groups,
                    models=model_names,
                    budgets=sub_budgets,
                    n_folds=sub_folds,
                    augmentations=sub_augs,
                    augmentation_strategy=sub_aug_strategy,
                    internal_evaluators=INTERNAL_EVALUATORS,
                    metric=metric_name,
                    task_type=data['task_type'],
                    verbose=False,
                    cache=True,
                    dataset_name=f"{dataset_name}_n{n_sub}_s{seed}",
                    tier='general',
                )
                last_search = search
                # Per-fold scores for BOTH arms (hybrid candidates +
                # per-rep baselines) in the CONTRACT shape. Derives models
                # only from what the search actually evaluated, so model
                # gating is preserved automatically.
                try:
                    honest_blocks_this_n.append(
                        build_honest_rescore_input(search))
                except Exception as _he:
                    print(f"    seed={seed}: honest_rescore build failed - {_he}")
                for model_name in model_names:
                    if model_name in search.best_by_model:
                        mbest = search.best_by_model[model_name]
                        hybrid_by_model[model_name].append(mbest['score_mean'])
                        configs_by_model[model_name].append({
                            'budget': mbest['blueprint'].budget,
                            'importance': mbest['blueprint'].importance_method,
                            'allocation': mbest['blueprint'].allocation_method,
                        })
            except Exception as e:
                print(f"    seed={seed}: find_best_hybrid failed - {e}")

            # Baselines per model (use sub-sample resolved factories so SVM/GP
            # gates apply consistently with the hybrid search's n_samples=n_sub).
            for model_name, model_factory in sub_models.items():
                # Best single rep
                best_single = np.inf if lower_is_better else -np.inf
                for rep_name, X in sub_reps.items():
                    try:
                        r2, _ = evaluate_single_rep_cv(
                            X, sub_labels, sub_groups, model_factory,
                            n_folds=sub_folds,
                            mode='intermediary' if sub_folds == 1 else 'full',
                            metric_fn=metric_fn,
                        )
                        if (lower_is_better and r2 < best_single) or \
                           (not lower_is_better and r2 > best_single):
                            best_single = r2
                    except Exception:
                        pass
                if best_single != (np.inf if lower_is_better else -np.inf):
                    single_by_model[model_name].append(best_single)

            # Second pass: reapply the all-rows-selected blueprint at this
            # subsample. alloc/imp/eval/budget/strategy stay fixed from the
            # whole-dataset selection above; only the feature selection runs on
            # subsample rows. The resulting score is NOT held out from the rows
            # that chose the recipe.
            for model_name, model_factory in models.items():
                bp_fd = all_rows_blueprints.get(model_name)
                if bp_fd is None:
                    continue
                bp_fd_budget = bp_fd.budget
                if bp_fd_budget is not None and bp_fd_budget > n_sub:
                    # Blueprint budget exceeds subsample — skip this n.
                    continue
                # FIX 4d: cv_split RAISES (no random-split fallback) when the
                # subsample has fewer distinct groups than the requested fold
                # count. Reduce the fold count to the available groups (min 2);
                # if fewer than 2 distinct groups exist, record an explicit skip
                # flag instead of crashing or silently emitting nothing.
                n_sub_groups = len(np.unique(sub_groups))
                if n_sub_groups < 2:
                    results['skipped_insufficient_groups'].append(
                        {'n': int(n_sub), 'model': model_name, 'seed': int(seed),
                         'n_groups': int(n_sub_groups),
                         'reason': 'fewer than 2 distinct groups for all-rows-bp CV'})
                    continue
                sub_folds_fd = min(sub_folds, n_sub_groups)
                try:
                    fold_scores = []
                    sample_sub_X = next(iter(sub_reps.values()))
                    try:
                        _fd_splits = list(cv_split(sub_groups, sub_folds_fd,
                                                   sample_sub_X, sub_labels))
                    except ValueError:
                        # Group count fell below the (already-reduced) fold
                        # count after StratifiedGroupKFold's own checks — skip
                        # cleanly with a flag rather than propagate.
                        results['skipped_insufficient_groups'].append(
                            {'n': int(n_sub), 'model': model_name, 'seed': int(seed),
                             'n_groups': int(n_sub_groups),
                             'reason': 'cv_split rejected all-rows-bp folds'})
                        continue
                    for tr_i, va_i in _fd_splits:
                        train_reps_fold = {k: v[tr_i] for k, v in sub_reps.items()}
                        val_reps_fold = {k: v[va_i] for k, v in sub_reps.items()}
                        train_augs_fold = ({k: v[tr_i] for k, v in sub_augs.items()}
                                           if sub_augs else None)
                        val_augs_fold = ({k: v[va_i] for k, v in sub_augs.items()}
                                         if sub_augs else None)
                        X_tr, X_va, _fi = build_fold_hybrid(
                            train_reps_fold, val_reps_fold,
                            sub_labels[tr_i], bp_fd,
                            augmentations_train=train_augs_fold,
                            augmentations_val=val_augs_fold,
                            groups=(sub_groups[tr_i] if sub_groups is not None else None),
                            model=model_factory(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better)
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_tr)
                        X_va_s = scaler.transform(X_va)
                        mdl = model_factory()
                        mdl.fit(X_tr_s, sub_labels[tr_i])
                        fold_scores.append(metric_fn(sub_labels[va_i], mdl.predict(X_va_s)))
                    if fold_scores:
                        hybrid_fd_by_model[model_name].append(float(np.mean(fold_scores)))
                except Exception:
                    pass

        if honest_blocks_this_n:
            results['honest_rescore_by_n'][str(n_sub)] = honest_blocks_this_n

        # Leave-one-fold-out re-pick at this sample size, per seed, then
        # averaged over seeds. Pure arithmetic on per-fold scores that already
        # exist, so it costs no fits. Subsamples differ between seeds, so each
        # seed's block is rescored on its own and only the resulting numbers
        # are averaged.
        honest_by_model_this_n = {m: {'h': [], 's': [], 'd': []}
                                  for m in model_names}
        for _blk in honest_blocks_this_n:
            try:
                _hr = honest_rescore(
                    _blk.get('candidates_by_model', {}),
                    _blk.get('baselines_by_model', {}),
                    _blk.get('lower_is_better'),
                    metric_name=_blk.get('metric_name'),
                    exclude_folds=hr_exclude_folds(_blk))
            except Exception as _e:
                print(f"    honest rescore unavailable at n={n_sub}: {_e}")
                continue
            for _m, _v in _hr.items():
                if _m not in honest_by_model_this_n:
                    honest_by_model_this_n[_m] = {'h': [], 's': [], 'd': []}
                for _k, _f in (('h', 'hybrid_honest'),
                               ('s', 'best_single_honest'),
                               ('d', 'delta_mean_of_folds')):
                    _val = _v.get(_f)
                    if _val is not None and not np.isnan(_val):
                        honest_by_model_this_n[_m][_k].append(float(_val))

        if last_search is not None:
            top3 = extract_top3_base_reps(last_search)
            if top3:
                print(f"    Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

        # Aggregate across repeats
        for model_name in model_names:
            h = (float(np.mean(hybrid_by_model[model_name]))
                 if hybrid_by_model[model_name] else float('nan'))
            s = (float(np.mean(single_by_model[model_name]))
                 if single_by_model[model_name] else float('nan'))

            h_fd = (float(np.mean(hybrid_fd_by_model[model_name]))
                    if hybrid_fd_by_model[model_name] else float('nan'))
            results['models'][model_name]['hybrid'].append(h)
            results['models'][model_name]['best_single'].append(s)
            results['models'][model_name]['best_configs'].append(
                configs_by_model[model_name] if configs_by_model[model_name] else None)
            _hb = honest_by_model_this_n.get(model_name, {'h': [], 's': [], 'd': []})
            for _k, _field in (('h', 'hybrid_honest'),
                               ('s', 'best_single_honest'),
                               ('d', 'honest_delta_mean_of_folds')):
                results['models'][model_name][_field].append(
                    float(np.mean(_hb[_k])) if _hb[_k] else float('nan'))
            results['models'][model_name]['hybrid_all_rows_bp'].append(h_fd)
            results['models'][model_name]['all_rows_bp_meta'].append(
                all_rows_meta.get(model_name))

            delta = (s - h if lower_is_better else h - s) if not (np.isnan(h) or np.isnan(s)) else float('nan')
            hd = results['models'][model_name]['honest_delta_mean_of_folds'][-1]
            print(f"    {model_name}: hybrid={h:.4f}, hybrid_all_rows_bp={h_fd:.4f}, "
                  f"best_single={s:.4f}, same-fold delta={delta:+.4f}, "
                  f"leave-one-fold-out delta={hd:+.4f}")

    # Top-3 from the last (full-data) subsample search
    top3_full = extract_top3_base_reps(last_search) if last_search is not None else []

    # Fit learning curves per model
    for model_name in model_names:
        model_result = results['models'][model_name]
        valid = [i for i, h in enumerate(model_result['hybrid']) if not np.isnan(h)]
        if len(valid) >= 3:
            n_arr = np.array([sample_sizes[i] for i in valid])
            h_arr = np.array([model_result['hybrid'][i] for i in valid])
            model_result['hybrid_fit'] = fit_learning_curve(
                n_arr, h_arr, lower_is_better=lower_is_better)
        valid_h = [i for i, h in enumerate(model_result['hybrid_honest'])
                   if not np.isnan(h)]
        if len(valid_h) >= 3:
            n_arr = np.array([sample_sizes[i] for i in valid_h])
            h_arr = np.array([model_result['hybrid_honest'][i] for i in valid_h])
            model_result['hybrid_honest_fit'] = fit_learning_curve(
                n_arr, h_arr, lower_is_better=lower_is_better)

    # Summary table. Each model gets its same-fold pair and the
    # leave-one-fold-out delta beside it; the two differ by how much the
    # hybrid side gains from maximising over ~600 recipes on the folds it is
    # then scored on.
    print("\n" + "-" * 70)
    print("LEARNING CURVE SUMMARY")
    print("-" * 70)
    header = f"  {'n':>6}"
    for m in model_names:
        header += f"  {m+'_h':>10} {m+'_s':>10} {'delta':>8} {'lofo_d':>8}"
    print(header)
    print("  " + "-" * (6 + (10 + 10 + 8 + 8 + 5) * len(model_names)))
    for i, n_sub in enumerate(sample_sizes):
        row = f"  {n_sub:>6}"
        for m in model_names:
            h = results['models'][m]['hybrid'][i]
            s = results['models'][m]['best_single'][i]
            d = (s - h if lower_is_better else h - s) if not (np.isnan(h) or np.isnan(s)) else float('nan')
            hd = results['models'][m]['honest_delta_mean_of_folds'][i]
            row += f"  {h:>10.4f} {s:>10.4f} {d:>+8.4f} {hd:>+8.4f}"
        print(row)

    # Learning curve fit summary
    print(f"\n  LEARNING CURVE FITS (score_max = asymptotic {metric_name}):")
    for model_name in model_names:
        fit = results['models'][model_name].get('hybrid_fit', {})
        if 'r2_max' in fit and 'error' not in fit:
            print(f"    {model_name}: {metric_name}_max={fit['r2_max']:.4f}, "
                  f"a={fit['a']:.3f}, b={fit['b']:.3f}, fit_R2={fit['fit_r2']:.3f}")
        elif 'r2_max' in fit:
            print(f"    {model_name}: {metric_name}_max~{fit['r2_max']:.4f} (fit failed: {fit.get('error', '?')})")

    results['top3_base_reps'] = [(n, s) for n, s in top3_full] if top3_full else []
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C05: Learning Curves',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--n_repeats', type=int, default=3,
                        help='Number of random subsampling repeats per size')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Repeats: {args.n_repeats}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, n_repeats=args.n_repeats)
    save_experiment_results(results, args.output, 'c05_learning_curves', args.dataset)
