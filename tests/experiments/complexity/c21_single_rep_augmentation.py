#!/usr/bin/env python3
"""C21: Single-Rep Augmentation Test.

Paper: Complexity
Conclusion: Uncapped augmentation can improve individual reps without
    hybridisation — small structural/engineered features carry signal that
    even high-dimensional PLM embeddings miss.

Dependencies: None (standalone).

Design: For each base rep in DATASET_REPS[dataset]['general']:
    Condition A: base rep alone → model CV score
    Condition B: create_hybrid(base_reps={rep}, augmentations=aug_dict,
                               augmentation_strategy='uncapped') → model CV score

Reports per-rep improvement, which aug features were kept, and how many
features were added beyond the base dimensionality.

Usage:
    python c21_single_rep_augmentation.py --dataset herg --reps all --mode intermediary
    python c21_single_rep_augmentation.py --dataset flab_thermo_tresanco --reps all --mode full
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
from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection


def run(data, config, output_dir, dataset_name, **kwargs):
    """C21: Single-Rep Augmentation Test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C21: SINGLE-REP AUGMENTATION")
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

    model_names = ['rf', 'ridge', 'xgboost']
    models = get_model_factories(model_names, task_type=task_type, n_samples=n_samples)
    print(f"  Models: {list(models.keys())}")

    # Identify augmentation reps from DATASET_REPS
    from experiments.complexity.complexity_helpers import DATASET_REPS
    ds_config = DATASET_REPS.get(dataset_name, {})
    aug_rep_names = ds_config.get('augmentation', [])
    general_rep_names = ds_config.get('general', [])

    # Load augmentation reps that aren't already in data
    missing_aug = [r for r in aug_rep_names if r not in reps]
    if missing_aug:
        print(f"  Loading {len(missing_aug)} augmentation reps: {missing_aug}")
        aug_data = load_data(dataset_name, rep_preset='all', rep_names=missing_aug)
        for rname, arr in aug_data['reps'].items():
            reps[rname] = arr

    aug_reps_available = [r for r in aug_rep_names if r in reps]
    base_reps_available = [r for r in general_rep_names if r in reps]

    if not aug_reps_available:
        print(f"  No augmentation reps available for {dataset_name} — skipping C21")
        return {'error': 'no_augmentation_reps', 'dataset': dataset_name}

    if not base_reps_available:
        print(f"  No base reps available for {dataset_name} — skipping C21")
        return {'error': 'no_base_reps', 'dataset': dataset_name}

    # Build augmentation dict (shared across all base reps)
    aug_dict = {rn: reps[rn] for rn in aug_reps_available}
    aug_dims = {rn: reps[rn].shape[1] for rn in aug_reps_available}

    print(f"  Base reps ({len(base_reps_available)}): {base_reps_available}")
    print(f"  Aug reps ({len(aug_reps_available)}): "
          f"{', '.join(f'{k}={v}D' for k, v in aug_dims.items())}")

    # ── Evaluate each base rep ──
    rep_results = []

    for base_name in base_reps_available:
        X_base = reps[base_name]
        base_dim = X_base.shape[1]
        print(f"\n  {base_name} ({base_dim}D)")

        for model_name, model_factory in models.items():
            scores_base = []
            scores_aug = []
            fold_aug_info = []

            for fold_idx, (train_idx, val_idx) in enumerate(
                cv_split(groups, n_folds, X_base, labels, mode=mode)
            ):
                y_train, y_val = labels[train_idx], labels[val_idx]

                # ── Condition A: base rep alone ──
                scaler_b = StandardScaler()
                Xtr_b = scaler_b.fit_transform(X_base[train_idx])
                Xva_b = scaler_b.transform(X_base[val_idx])
                m1 = model_factory()
                m1.fit(Xtr_b, y_train)
                scores_base.append(float(metric_fn(y_val, m1.predict(Xva_b))))

                # ── Condition B: base + uncapped augmentation ──
                train_base = {base_name: X_base[train_idx]}
                val_base = {base_name: X_base[val_idx]}
                train_augs = {rn: aug_dict[rn][train_idx] for rn in aug_reps_available}
                val_augs = {rn: aug_dict[rn][val_idx] for rn in aug_reps_available}

                try:
                    Xtr_h, feature_info = create_hybrid(
                        base_reps=train_base,
                        labels=y_train,
                        allocation_method='performance_weighted',
                        budget=base_dim,
                        importance_method='treeshap',
                        augmentations=train_augs,
                        augmentation_strategy='uncapped',
                    )

                    # Apply same selection to validation
                    Xva_h = apply_feature_selection(val_base, feature_info)

                    # Append augmentation features if any were kept
                    aug_info = feature_info.get('augmentation_info', {})
                    kept = feature_info.get('kept_augmentations', [])
                    if kept and aug_info:
                        aug_features_val = apply_augmentation_selection(
                            val_augs, aug_info
                        )
                        if aug_features_val is not None:
                            Xva_h = np.hstack([Xva_h, aug_features_val])

                    scaler_h = StandardScaler()
                    Xtr_h = scaler_h.fit_transform(Xtr_h)
                    Xva_h = scaler_h.transform(Xva_h)
                    m2 = model_factory()
                    m2.fit(Xtr_h, y_train)
                    scores_aug.append(float(metric_fn(y_val, m2.predict(Xva_h))))

                    n_added = Xtr_h.shape[1] - base_dim
                    fold_aug_info.append({
                        'kept': kept,
                        'n_added': max(0, n_added),
                        'final_dim': Xtr_h.shape[1],
                    })
                except Exception as e:
                    print(f"    {model_name} fold {fold_idx}: augmentation failed - {e}")
                    scores_aug.append(float('nan'))
                    fold_aug_info.append({'kept': [], 'n_added': 0, 'final_dim': base_dim})

            base_mean = float(np.nanmean(scores_base))
            aug_mean = float(np.nanmean(scores_aug))
            delta = aug_mean - base_mean

            # Aggregate aug info across folds
            all_kept = set()
            total_added = []
            for fi in fold_aug_info:
                all_kept.update(fi['kept'])
                total_added.append(fi['n_added'])

            result = {
                'base_rep': base_name,
                'model': model_name,
                'base_dim': base_dim,
                'base_mean': base_mean,
                'base_std': float(np.nanstd(scores_base)),
                'base_scores': scores_base,
                'aug_mean': aug_mean,
                'aug_std': float(np.nanstd(scores_aug)),
                'aug_scores': scores_aug,
                'delta': delta,
                'relative_improvement': delta / abs(base_mean) if base_mean != 0 else 0.0,
                'kept_augmentations': sorted(all_kept),
                'mean_features_added': float(np.mean(total_added)),
                'fold_aug_info': fold_aug_info,
            }
            rep_results.append(result)

            sign = -1 if lower_is_better else 1
            status = "+" if sign * delta > 0 else "-" if sign * delta < 0 else "="
            print(f"    {model_name}: base={base_mean:.4f}  aug={aug_mean:.4f}  "
                  f"delta={sign * delta:+.4f} [{status}]  "
                  f"kept={sorted(all_kept)}  added={np.mean(total_added):.1f}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    n_tests = len(rep_results)
    if n_tests > 0:
        sign = -1 if lower_is_better else 1
        improved = sum(1 for r in rep_results if sign * r['delta'] > 0)
        hurt = sum(1 for r in rep_results if sign * r['delta'] < 0)
        mean_delta = float(np.mean([sign * r['delta'] for r in rep_results]))
        print(f"  Total tests: {n_tests} ({len(base_reps_available)} reps x {len(models)} models)")
        print(f"  Augmentation improved: {improved}/{n_tests}")
        print(f"  Augmentation hurt:     {hurt}/{n_tests}")
        print(f"  Mean delta:            {mean_delta:+.4f}")

        # Best improvement
        best = max(rep_results, key=lambda r: sign * r['delta'])
        print(f"  Best: {best['base_rep']}/{best['model']} "
              f"delta={sign * best['delta']:+.4f} "
              f"kept={best['kept_augmentations']}")

    return convert_numpy({
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_folds': n_folds,
        'n_samples': n_samples,
        'aug_reps': aug_reps_available,
        'aug_dims': aug_dims,
        'rep_results': rep_results,
        'summary': {
            'n_tests': n_tests,
            'improved': improved if n_tests > 0 else 0,
            'hurt': hurt if n_tests > 0 else 0,
            'mean_delta': mean_delta if n_tests > 0 else 0.0,
        },
    })


if __name__ == '__main__':
    parser = create_experiment_parser('C21: Single-Rep Augmentation',
                                     has_delta=False, has_augment=False)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c21_single_rep_augmentation',
                            args.dataset)
