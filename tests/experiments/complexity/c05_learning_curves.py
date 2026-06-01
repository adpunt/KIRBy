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
    INTERNAL_EVALUATORS, TOP_CAPPED_MAX_N,
    get_best_search_for_downstream, build_fold_hybrid, cv_split,
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
    search_speed = config.get('search_speed', 'full')
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

    fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
    sample_sizes = [max(50, int(n_total * f)) for f in fractions]
    # Deduplicate and cap at total
    sample_sizes = sorted(set(min(s, n_total) for s in sample_sizes))

    print(f"  Models: {model_names}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Budgets: {sweep_budgets}")
    print(f"  Repeats: {n_repeats}")
    print(f"  Search speed: {search_speed}")

    last_search = None
    results = {'sample_sizes': sample_sizes, 'budgets': sweep_budgets,
                'models': {}, 'search_speed': search_speed}

    # Initialize per-model result structure
    for model_name in model_names:
        results['models'][model_name] = {
            'hybrid': [], 'best_single': [],
            'best_configs': [],
            # Second pass: full-data pool-best blueprint applied at each n.
            # Compares "best AT each n" vs "full-data optimal at each n".
            'hybrid_fulldata_bp': [],
            'fulldata_bp_meta': [],
        }

    # Pre-compute the full-data pool-best blueprint per model (once). At each
    # subsample, this blueprint is *re-fit* on the subsample only (no leakage)
    # using build_fold_hybrid — the BLUEPRINT (alloc/imp/eval/budget/strategy)
    # is fixed, the feature selection runs on subsample data.
    fulldata_blueprints = {}
    fulldata_meta = {}
    full_augs_top = data.get('augmentations')
    for model_name in model_names:
        try:
            _s_fd, _m_fd = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=model_name, budget=None,
                augmentations=full_augs_top, speed=search_speed,
                dataset_name=dataset_name,
                task_type=data['task_type'], metric=metric_name,
                data=data,
            )
            if _m_fd:
                fulldata_blueprints[model_name] = _m_fd['best_config']['blueprint']
                fulldata_meta[model_name] = {
                    'tier': _m_fd['tier'], 'aug_condition': _m_fd['aug_condition'],
                    'budget': _m_fd['best_config']['blueprint'].budget,
                }
                print(f"  Full-data BP for {model_name}: tier={_m_fd['tier']} "
                      f"aug={_m_fd['aug_condition']} budget={_m_fd['best_config']['blueprint'].budget}")
        except Exception as e:
            print(f"  Full-data BP for {model_name}: pool helper failed ({e})")

    for n_sub in sample_sizes:
        print(f"\n  n={n_sub}:")
        # Collect scores across repeats per model
        hybrid_by_model = {m: [] for m in model_names}
        single_by_model = {m: [] for m in model_names}
        configs_by_model = {m: [] for m in model_names}
        # Second pass: full-data blueprint re-fit on subsample.
        hybrid_fd_by_model = {m: [] for m in model_names}

        last_search = None
        full_augs = data.get('augmentations')
        # Pre-compute group-level pool so subsampling preserves scaffold
        # integrity (subsampling individual rows shatters scaffold groups
        # across train/val of the same fold).
        _unique_groups = np.unique(groups)
        _group_to_idx = {g: np.where(groups == g)[0] for g in _unique_groups}
        for seed in range(n_repeats):
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

            # find_best_hybrid discovers optimal config per model at this sample size
            # Cap budgets to subsample size (budget > n makes no sense)
            sub_budgets = [b for b in sweep_budgets if b <= n_sub]
            if not sub_budgets:
                sub_budgets = [min(sweep_budgets)]
            # Subsample-specific cache key — won't collide with main precompute
            # cache. Capped strategy is fine for small subsamples but projects
            # past the 7-day partition for n>TOP_CAPPED_MAX_N. Fall back to
            # uncapped when the subsample is over the threshold.
            sub_aug_strategy = 'capped' if n_sub <= TOP_CAPPED_MAX_N else 'uncapped'
            try:
                search = find_best_hybrid(
                    sub_reps, sub_labels, groups=sub_groups,
                    models=model_names,
                    budgets=sub_budgets,
                    speed=search_speed,
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

            # Second pass: re-fit the full-data pool-best blueprint on
            # subsample. Blueprint (alloc/imp/eval/budget/strategy) is fixed
            # per model from the full-data pool helper above; we just
            # reapply it. CV evaluation on the subsample using build_fold_hybrid.
            for model_name, model_factory in models.items():
                bp_fd = fulldata_blueprints.get(model_name)
                if bp_fd is None:
                    continue
                bp_fd_budget = bp_fd.budget
                if bp_fd_budget is not None and bp_fd_budget > n_sub:
                    # Blueprint budget exceeds subsample — skip this n.
                    continue
                try:
                    fold_scores = []
                    sample_sub_X = next(iter(sub_reps.values()))
                    for tr_i, va_i in cv_split(sub_groups, sub_folds,
                                                sample_sub_X, sub_labels,
                                                mode='intermediary' if sub_folds == 1 else 'full'):
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
                        )
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
            results['models'][model_name]['hybrid_fulldata_bp'].append(h_fd)
            results['models'][model_name]['fulldata_bp_meta'].append(
                fulldata_meta.get(model_name))

            delta = (s - h if lower_is_better else h - s) if not (np.isnan(h) or np.isnan(s)) else float('nan')
            print(f"    {model_name}: hybrid={h:.4f}, hybrid_fd_bp={h_fd:.4f}, "
                  f"best_single={s:.4f}, delta={delta:+.4f}")

    # Top-3 from the last (full-data) subsample search
    top3_full = extract_top3_base_reps(last_search) if last_search is not None else []

    # Fit learning curves per model
    for model_name in model_names:
        model_result = results['models'][model_name]
        valid = [i for i, h in enumerate(model_result['hybrid']) if not np.isnan(h)]
        if len(valid) >= 3:
            n_arr = np.array([sample_sizes[i] for i in valid])
            h_arr = np.array([model_result['hybrid'][i] for i in valid])
            model_result['hybrid_fit'] = fit_learning_curve(n_arr, h_arr)

    # Summary table
    print("\n" + "-" * 70)
    print("LEARNING CURVE SUMMARY")
    print("-" * 70)
    header = f"  {'n':>6}"
    for m in model_names:
        header += f"  {m+'_h':>10} {m+'_s':>10} {'delta':>8}"
    print(header)
    print("  " + "-" * (6 + (10 + 10 + 8 + 4) * len(model_names)))
    for i, n_sub in enumerate(sample_sizes):
        row = f"  {n_sub:>6}"
        for m in model_names:
            h = results['models'][m]['hybrid'][i]
            s = results['models'][m]['best_single'][i]
            d = (s - h if lower_is_better else h - s) if not (np.isnan(h) or np.isnan(s)) else float('nan')
            row += f"  {h:>10.4f} {s:>10.4f} {d:>+8.4f}"
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
