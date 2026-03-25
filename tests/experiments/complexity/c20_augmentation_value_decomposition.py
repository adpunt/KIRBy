#!/usr/bin/env python3
"""C20: Augmentation Value Decomposition.

Paper: Complexity
Conclusion: Hybrid feature selection is what makes augmentation valuable —
    raw concatenation of augmentation reps adds noise, while hybrid selection
    extracts the useful signal.

Dependencies: None (standalone).

Design: For each (base_rep, aug_rep) pair, compare three conditions:
    1. Base alone:       model on base_rep
    2. Base + aug concat: model on hstack(base_rep, aug_rep)
    3. Hybrid(base+aug): create_hybrid({base, aug}, budget=base.d) then model

The budget for condition 3 is set to the base rep's dimensionality so that
the hybrid and the raw rep have the same number of features. This isolates
whether hybrid *feature selection* helps vs just *more features*.

Key comparison:
    - concat - base  = does augmentation help at all (raw)?
    - hybrid - base  = does augmentation help via hybrid selection?
    - If concat ≈ base but hybrid > base, the hybrid's selection is the value.

Usage:
    python c20_augmentation_value_decomposition.py --dataset herg --reps all --mode intermediary
    python c20_augmentation_value_decomposition.py --dataset flab_thermo_tresanco --reps all --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split, get_domain_metric,
)
from kirby.hybrid import create_hybrid, apply_feature_selection


def run(data, config, output_dir, dataset_name, **kwargs):
    """C20: Augmentation Value Decomposition."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C20: AUGMENTATION VALUE DECOMPOSITION")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data.get('task_type', 'regression')
    n_samples = len(labels)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")

    # Use a small set of models — this is a comparison experiment, not a grid search
    model_names = ['rf', 'ridge']
    models = get_model_factories(model_names, task_type=task_type, n_samples=n_samples)
    print(f"  Models: {list(models.keys())}")

    # Identify augmentation reps from DATASET_REPS
    from experiments.complexity.complexity_helpers import DATASET_REPS
    ds_config = DATASET_REPS.get(dataset_name, {})
    aug_rep_names = ds_config.get('augmentation', [])
    general_rep_names = ds_config.get('general', [])

    # Load augmentation reps that aren't already in data (run_all only loads general)
    missing_aug = [r for r in aug_rep_names if r not in reps]
    if missing_aug:
        print(f"  Loading {len(missing_aug)} augmentation reps: {missing_aug}")
        aug_data = load_data(dataset_name, rep_preset='all', rep_names=missing_aug)
        for rname, arr in aug_data['reps'].items():
            reps[rname] = arr

    # Filter to reps that are actually loaded
    aug_reps_available = [r for r in aug_rep_names if r in reps]
    base_reps_available = [r for r in general_rep_names if r in reps]

    if not aug_reps_available:
        print(f"  No augmentation reps available for {dataset_name} — skipping C20")
        return {'error': 'no_augmentation_reps', 'dataset': dataset_name}

    if not base_reps_available:
        print(f"  No base reps available for {dataset_name} — skipping C20")
        return {'error': 'no_base_reps', 'dataset': dataset_name}

    print(f"  Base reps ({len(base_reps_available)}): {base_reps_available}")
    print(f"  Aug reps ({len(aug_reps_available)}): {aug_reps_available}")

    # ── Evaluate all (base, aug) pairs across folds ──
    pair_results = []

    for base_name in base_reps_available:
        X_base = reps[base_name]
        base_dim = X_base.shape[1]

        for aug_name in aug_reps_available:
            X_aug = reps[aug_name]
            aug_dim = X_aug.shape[1]
            X_concat = np.hstack([X_base, X_aug])

            print(f"\n  {base_name} ({base_dim}D) + {aug_name} ({aug_dim}D)")

            for model_name, model_factory in models.items():
                scores_base = []
                scores_concat = []
                scores_hybrid = []

                for fold_idx, (train_idx, val_idx) in enumerate(
                    cv_split(groups, n_folds, X_base, labels, mode=mode)
                ):
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    # ── Condition 1: base alone ──
                    scaler_b = StandardScaler()
                    Xtr_b = scaler_b.fit_transform(X_base[train_idx])
                    Xva_b = scaler_b.transform(X_base[val_idx])
                    m1 = model_factory()
                    m1.fit(Xtr_b, y_train)
                    scores_base.append(float(metric_fn(y_val, m1.predict(Xva_b))))

                    # ── Condition 2: base + aug concat ──
                    scaler_c = StandardScaler()
                    Xtr_c = scaler_c.fit_transform(X_concat[train_idx])
                    Xva_c = scaler_c.transform(X_concat[val_idx])
                    m2 = model_factory()
                    m2.fit(Xtr_c, y_train)
                    scores_concat.append(float(metric_fn(y_val, m2.predict(Xva_c))))

                    # ── Condition 3: hybrid at base dimensionality ──
                    train_reps_pair = {
                        base_name: X_base[train_idx],
                        aug_name: X_aug[train_idx],
                    }
                    val_reps_pair = {
                        base_name: X_base[val_idx],
                        aug_name: X_aug[val_idx],
                    }
                    try:
                        Xtr_h, feature_info = create_hybrid(
                            base_reps=train_reps_pair,
                            labels=y_train,
                            allocation_method='performance_weighted',
                            budget=base_dim,
                            importance_method='treeshap',
                        )
                        Xva_h = apply_feature_selection(val_reps_pair, feature_info)

                        scaler_h = StandardScaler()
                        Xtr_h = scaler_h.fit_transform(Xtr_h)
                        Xva_h = scaler_h.transform(Xva_h)
                        m3 = model_factory()
                        m3.fit(Xtr_h, y_train)
                        scores_hybrid.append(float(metric_fn(y_val, m3.predict(Xva_h))))
                    except Exception as e:
                        print(f"    Hybrid failed fold {fold_idx}: {e}")
                        scores_hybrid.append(float('nan'))

                result = {
                    'base_rep': base_name,
                    'aug_rep': aug_name,
                    'model': model_name,
                    'base_dim': base_dim,
                    'aug_dim': aug_dim,
                    'concat_dim': base_dim + aug_dim,
                    'hybrid_budget': base_dim,
                    'base_scores': scores_base,
                    'concat_scores': scores_concat,
                    'hybrid_scores': scores_hybrid,
                    'base_mean': float(np.nanmean(scores_base)),
                    'concat_mean': float(np.nanmean(scores_concat)),
                    'hybrid_mean': float(np.nanmean(scores_hybrid)),
                    'delta_concat': float(np.nanmean(scores_concat) - np.nanmean(scores_base)),
                    'delta_hybrid': float(np.nanmean(scores_hybrid) - np.nanmean(scores_base)),
                }
                pair_results.append(result)

                sign = -1 if lower_is_better else 1
                print(f"    {model_name}: base={result['base_mean']:.4f}  "
                      f"concat={result['concat_mean']:.4f} ({sign * result['delta_concat']:+.4f})  "
                      f"hybrid={result['hybrid_mean']:.4f} ({sign * result['delta_hybrid']:+.4f})")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    n_pairs = len(pair_results)
    if n_pairs > 0:
        sign = -1 if lower_is_better else 1
        concat_wins = sum(1 for r in pair_results if sign * r['delta_concat'] > 0)
        hybrid_wins = sum(1 for r in pair_results if sign * r['delta_hybrid'] > 0)
        hybrid_beats_concat = sum(
            1 for r in pair_results
            if sign * (r['hybrid_mean'] - r['concat_mean']) > 0
        )
        print(f"  Total pairs evaluated: {n_pairs}")
        print(f"  Concat improves over base:  {concat_wins}/{n_pairs}")
        print(f"  Hybrid improves over base:  {hybrid_wins}/{n_pairs}")
        print(f"  Hybrid beats concat:        {hybrid_beats_concat}/{n_pairs}")

    return convert_numpy({
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_folds': n_folds,
        'n_samples': n_samples,
        'pair_results': pair_results,
        'summary': {
            'n_pairs': n_pairs,
            'concat_improves': concat_wins if n_pairs > 0 else 0,
            'hybrid_improves': hybrid_wins if n_pairs > 0 else 0,
            'hybrid_beats_concat': hybrid_beats_concat if n_pairs > 0 else 0,
        },
    })


if __name__ == '__main__':
    parser = create_experiment_parser('C20: Augmentation Value Decomposition',
                                     has_delta=False, has_augment=False)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c20_augmentation_value_decomposition',
                            args.dataset)
