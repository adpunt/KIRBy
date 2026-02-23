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
    load_data, INTERMEDIARY, FULL, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, evaluate_single_rep_cv, fit_learning_curve,
    find_best_hybrid,
)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Learning curves: hybrid advantage vs sample size."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C05: LEARNING CURVES")
    print("=" * 70)

    models = get_model_factories(config['models'])
    n_folds = config['n_splits']
    # TODO: sweep budget (e.g. 100, 300, 500) — current fixed budget=100 is
    #  suboptimal for most models (C02 shows Ridge peaks at 500, MLP at 300).
    search_speed = config.get('search_speed', 'fast')
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
    print(f"  Repeats: {n_repeats}")
    print(f"  Search speed: {search_speed}")

    results = {'sample_sizes': sample_sizes, 'models': {}, 'search_speed': search_speed}

    # Initialize per-model result structure
    for model_name in model_names:
        results['models'][model_name] = {
            'hybrid': [], 'best_single': [],
            'best_configs': [],
        }

    for n_sub in sample_sizes:
        print(f"\n  n={n_sub}:")
        # Collect scores across repeats per model
        hybrid_by_model = {m: [] for m in model_names}
        single_by_model = {m: [] for m in model_names}
        configs_by_model = {m: [] for m in model_names}

        full_augs = data.get('augmentations')
        for seed in range(n_repeats):
            np.random.seed(seed)
            idx = np.random.choice(n_total, n_sub, replace=False)
            sub_reps = {k: v[idx] for k, v in reps.items()}
            sub_labels = labels[idx]
            sub_groups = groups[idx]
            sub_augs = {k: v[idx] for k, v in full_augs.items()} if full_augs else None
            n_unique_groups = len(np.unique(sub_groups))

            if n_unique_groups < 2:
                continue

            sub_folds = min(n_folds, n_unique_groups)

            # find_best_hybrid discovers optimal config per model at this sample size
            try:
                search = find_best_hybrid(
                    sub_reps, sub_labels, groups=sub_groups,
                    models=model_names,
                    speed=search_speed,
                    n_folds=sub_folds,
                    augmentations=sub_augs,
                    verbose=False,
                    cache=True,
                )
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

            # Baselines per model
            for model_name, model_factory in models.items():
                # Best single rep
                best_single = -np.inf
                for rep_name, X in sub_reps.items():
                    try:
                        r2, _ = evaluate_single_rep_cv(
                            X, sub_labels, sub_groups, model_factory,
                            n_folds=sub_folds,
                            mode='intermediary' if sub_folds == 1 else 'full',
                        )
                        best_single = max(best_single, r2)
                    except Exception:
                        pass
                if best_single > -np.inf:
                    single_by_model[model_name].append(best_single)

        # Aggregate across repeats
        for model_name in model_names:
            h = (float(np.mean(hybrid_by_model[model_name]))
                 if hybrid_by_model[model_name] else float('nan'))
            s = (float(np.mean(single_by_model[model_name]))
                 if single_by_model[model_name] else float('nan'))

            results['models'][model_name]['hybrid'].append(h)
            results['models'][model_name]['best_single'].append(s)
            results['models'][model_name]['best_configs'].append(
                configs_by_model[model_name] if configs_by_model[model_name] else None)

            delta = h - s if not (np.isnan(h) or np.isnan(s)) else float('nan')
            print(f"    {model_name}: hybrid={h:.4f}, best_single={s:.4f}, delta={delta:+.4f}")

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
            d = h - s if not (np.isnan(h) or np.isnan(s)) else float('nan')
            row += f"  {h:>10.4f} {s:>10.4f} {d:>+8.4f}"
        print(row)

    # Learning curve fit summary
    print("\n  LEARNING CURVE FITS (R2_max = asymptotic R2):")
    for model_name in model_names:
        fit = results['models'][model_name].get('hybrid_fit', {})
        if 'r2_max' in fit and 'error' not in fit:
            print(f"    {model_name}: R2_max={fit['r2_max']:.4f}, "
                  f"a={fit['a']:.3f}, b={fit['b']:.3f}, fit_R2={fit['fit_r2']:.3f}")
        elif 'r2_max' in fit:
            print(f"    {model_name}: R2_max~{fit['r2_max']:.4f} (fit failed: {fit.get('error', '?')})")

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
