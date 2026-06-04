#!/usr/bin/env python3
"""C21: Single-Rep Augmentation Test.

Paper: Complexity
Conclusion: Uncapped augmentation can improve individual reps without
    hybridisation — small structural/engineered features carry signal that
    even high-dimensional PLM embeddings miss.

Dependencies: None (standalone).

Design: For each base rep in DATASET_REPS[dataset]['general']:
    Condition A: base rep alone → model CV score
    Condition B: find_best_hybrid(base_reps={rep}, augmentations=aug_dict,
                                  augmentation_strategy='uncapped',
                                  budgets=[base_dim]) → best config per-fold CV score

The search over importance_method × internal_evaluator discovers the best
way to select/weight features for that specific rep + augmentation combination.
Allocation is trivial (1 base rep → 100%).

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
    build_fold_hybrid,
    extract_top3_base_reps,
    INTERNAL_EVALUATORS,
)
from kirby.find_best_hybrid import find_best_hybrid, find_best_single_rep_augmented
from experiments.complexity.complexity_helpers import get_aug_groups


def run(data, config, output_dir, dataset_name, **kwargs):
    """C21: Single-Rep Augmentation Test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C21: SINGLE-REP AUGMENTATION")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    n_samples = len(labels)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")

    model_names = ['rf', 'knn', 'xgboost']
    models = get_model_factories(model_names, task_type=task_type, n_samples=n_samples)
    print(f"  Models: {list(models.keys())}")

    # Identify augmentation reps from DATASET_REPS
    from experiments.complexity.complexity_helpers import DATASET_REPS
    ds_config = DATASET_REPS.get(dataset_name, {})
    aug_rep_names = ds_config.get('augmentation', [])
    general_rep_names = ds_config.get('general', [])

    # Small general reps (≤256D) also serve as augmentation sources.
    # E.g. MACCs (167D), PDV (200D), aac, blosum — these carry complementary
    # structural/engineered signal that can augment larger reps.
    SMALL_REP_THRESHOLD = 256
    small_general = [r for r in general_rep_names
                     if r in reps and reps[r].shape[1] <= SMALL_REP_THRESHOLD]
    # Merge: augmentation reps + small general reps (deduplicated)
    all_aug_candidates = list(dict.fromkeys(aug_rep_names + small_general))

    # Load augmentation reps that aren't already in data
    missing_aug = [r for r in all_aug_candidates if r not in reps]
    if missing_aug:
        print(f"  Loading {len(missing_aug)} augmentation reps: {missing_aug}")
        aug_data = load_data(dataset_name, rep_preset='all', rep_names=missing_aug)
        for rname, arr in aug_data['reps'].items():
            reps[rname] = arr

    aug_reps_available = [r for r in all_aug_candidates if r in reps]
    base_reps_available = [r for r in general_rep_names if r in reps]

    if not aug_reps_available:
        print(f"  No augmentation reps available for {dataset_name} — skipping C21")
        return {'error': 'no_augmentation_reps', 'dataset': dataset_name}

    if not base_reps_available:
        print(f"  No base reps available for {dataset_name} — skipping C21")
        return {'error': 'no_base_reps', 'dataset': dataset_name}

    print(f"  Small general reps added to aug pool: {small_general}")

    aug_dims = {rn: reps[rn].shape[1] for rn in aug_reps_available}
    print(f"  Base reps ({len(base_reps_available)}): {base_reps_available}")
    print(f"  Aug pool ({len(aug_reps_available)}): "
          f"{', '.join(f'{k}={v}D' for k, v in aug_dims.items())}")

    search_speed = config.get('search_speed', 'full')

    # ── Build augmentation group conditions ──
    # Condition A = baseline (no aug), B = all augs, C-F = domain-specific groups
    aug_group_defs = get_aug_groups(dataset_name, set(reps.keys()))
    print(f"  Aug groups: {', '.join(f'{k}({len(v)})' for k, v in aug_group_defs.items())}")

    # ── Evaluate each base rep × augmentation condition ──
    rep_results = []

    n_samples = len(labels)
    for base_name in base_reps_available:
        X_base = reps[base_name]
        base_dim = X_base.shape[1]
        print(f"\n  {base_name} ({base_dim}D)")
        if base_dim > n_samples:
            print(f"    skipping {base_name}: base_dim={base_dim} > n_samples={n_samples} "
                  f"(find_best_single_rep_augmented default budget=base_dim would be silently dropped)")
            continue

        # Full augmentation pool (exclude self)
        full_aug = {rn: reps[rn] for rn in aug_reps_available if rn != base_name}

        # Build conditions: A=none, B=all, then each domain group
        conditions = {'A_baseline': None, 'B_all': full_aug}
        for gname, grep_list in aug_group_defs.items():
            grp_dict = {rn: reps[rn] for rn in grep_list
                        if rn != base_name and rn in reps}
            if grp_dict:
                conditions[f'C_{gname}'] = grp_dict

        for cond_name, aug_dict in conditions.items():
            is_baseline = aug_dict is None

            # Search for best augmentation config (skip for baseline)
            best_by_model = None
            bp = None
            if not is_baseline:
                try:
                    search = find_best_single_rep_augmented(
                        primary_rep=base_name,
                        primary_features=X_base,
                        augmentation_pool=aug_dict,
                        labels=labels,
                        groups=groups,
                        task_type=task_type,
                        metric=metric_name,
                        n_folds=n_folds,
                        speed=search_speed,
                        internal_evaluators=INTERNAL_EVALUATORS,
                        cache=True,
                        dataset_name=dataset_name,
                        tier=f'c21_{base_name}',
                        condition_suffix=cond_name,
                        verbose=False,
                    )
                    best_by_model = search.best_by_model
                    best = (best_by_model.get('rf') or
                            next(iter(best_by_model.values()), None))
                    bp = best['blueprint'] if best else None
                    if bp is None:
                        # Diagnostic: search returned but found no winning config
                        # -> every augmented fold scores NaN -> aug_mean NaN.
                        # This was the cause of the NaN c21 results on
                        # VEGF/Ang2/abcov_kd. Surface it instead of silently NaN.
                        print(f"    {cond_name}: search returned no config "
                              f"(best_by_model has {len(best_by_model)} entries)")
                except Exception as e:
                    print(f"    {cond_name}: search FAILED - {e}")
                    raise

            for model_name, model_factory in models.items():
                # Use this downstream model's own blueprint when available
                # (knn/xgboost should not be scored against rf-optimal feature
                # selection). Falls back to rf when the search didn't run
                # (baseline) or didn't include this model.
                bp_model = bp
                if best_by_model is not None:
                    _best = (best_by_model.get(model_name)
                             or best_by_model.get('rf')
                             or next(iter(best_by_model.values()), None))
                    bp_model = _best['blueprint'] if _best else bp

                scores_base = []
                scores_aug = []
                fold_aug_info = []

                for fold_idx, (train_idx, val_idx) in enumerate(
                    cv_split(groups, n_folds, X_base, labels, mode=mode)
                ):
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    # Baseline: base rep alone
                    scaler_b = StandardScaler()
                    Xtr_b = scaler_b.fit_transform(X_base[train_idx])
                    Xva_b = scaler_b.transform(X_base[val_idx])
                    m1 = model_factory()
                    m1.fit(Xtr_b, y_train)
                    scores_base.append(float(metric_fn(y_val, m1.predict(Xva_b))))

                    # Augmented condition
                    if is_baseline or bp is None:
                        scores_aug.append(float('nan'))
                        fold_aug_info.append({'kept': [], 'n_added': 0,
                                             'final_dim': base_dim})
                        continue

                    train_base = {base_name: X_base[train_idx]}
                    val_base = {base_name: X_base[val_idx]}
                    train_augs = {rn: aug_dict[rn][train_idx]
                                  for rn in aug_dict}
                    val_augs = {rn: aug_dict[rn][val_idx]
                                for rn in aug_dict}

                    try:
                        Xtr_h, Xva_h, feature_info = build_fold_hybrid(
                            train_base, val_base, y_train, bp_model,
                            augmentations_train=train_augs,
                            augmentations_val=val_augs)

                        scaler_h = StandardScaler()
                        Xtr_h = scaler_h.fit_transform(Xtr_h)
                        Xva_h = scaler_h.transform(Xva_h)
                        m2 = model_factory()
                        m2.fit(Xtr_h, y_train)
                        scores_aug.append(float(metric_fn(y_val, m2.predict(Xva_h))))

                        aug_info = feature_info.get('augmentation_info', {})
                        kept = list(aug_info.keys()) if aug_info else []
                        n_added = Xtr_h.shape[1] - base_dim
                        fold_aug_info.append({
                            'kept': kept,
                            'n_added': max(0, n_added),
                            'final_dim': Xtr_h.shape[1],
                        })
                    except Exception as e:
                        print(f"    {model_name} fold {fold_idx}: augmentation failed - {e}")
                        scores_aug.append(float('nan'))
                        fold_aug_info.append({'kept': [], 'n_added': 0,
                                             'final_dim': base_dim})

                base_mean = float(np.nanmean(scores_base))
                # BRK-7/8: when the augmentation search returned no winning
                # config (bp is None), every augmented fold is NaN. Skip cleanly
                # with an explicit flag instead of emitting a bare NaN aug_mean
                # (which nan-saturated all the powered c21 results — VEGF/Ang2/
                # abcov_kd). A real all-NaN fold set (e.g. every fold raised) is
                # treated the same: flagged, not silently averaged.
                if is_baseline:
                    aug_status = 'baseline'
                elif bp is None:
                    aug_status = 'search_no_config'
                elif np.all(np.isnan(scores_aug)):
                    aug_status = 'all_folds_failed'
                else:
                    aug_status = 'ok'

                if aug_status in ('baseline', 'search_no_config', 'all_folds_failed'):
                    aug_mean = float('nan')
                    delta = 0.0 if is_baseline else float('nan')
                else:
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
                    'condition': cond_name,
                    'model': model_name,
                    'base_dim': base_dim,
                    'base_mean': base_mean,
                    'base_std': float(np.nanstd(scores_base)),
                    'base_scores': scores_base,
                    'aug_mean': aug_mean,
                    'aug_std': float(np.nanstd(scores_aug)) if not is_baseline else 0.0,
                    'aug_scores': scores_aug,
                    'delta': delta,
                    'aug_status': aug_status,
                    # Sign-normalise: positive == improvement regardless of
                    # metric direction.  Without this, RMSE/MAE wins are
                    # stored as negative relative_improvement and any
                    # downstream filter on positive sign rejects them.
                    'relative_improvement': (
                        (-1 if lower_is_better else 1) * delta / abs(base_mean)
                        if (base_mean != 0 and not np.isnan(delta)) else float('nan')),
                    'kept_augmentations': sorted(all_kept),
                    'mean_features_added': float(np.mean(total_added)),
                    'fold_aug_info': fold_aug_info,
                    'aug_group_reps': list(aug_dict.keys()) if aug_dict else [],
                    'blueprint': bp.to_dict() if bp else None,
                }
                rep_results.append(result)

                sign = -1 if lower_is_better else 1
                if aug_status not in ('ok', 'baseline'):
                    print(f"    {cond_name}/{model_name}: base={base_mean:.4f}  "
                          f"SKIPPED ({aug_status})")
                else:
                    status = "+" if sign * delta > 0 else "-" if sign * delta < 0 else "="
                    print(f"    {cond_name}/{model_name}: base={base_mean:.4f}  aug={aug_mean:.4f}  "
                          f"delta={sign * delta:+.4f} [{status}]  "
                      f"kept={sorted(all_kept)}  added={np.mean(total_added):.1f}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Exclude baseline condition from aug analysis. BRK-7/8: only aggregate
    # over results that were actually evaluable (aug_status == 'ok'); skipped
    # ones (search returned no config / all folds failed) carry a NaN delta and
    # would poison the mean / argmax. Count them separately so a NaN-saturated
    # run reads as "N skipped" rather than a meaningless NaN mean_delta.
    aug_results_all = [r for r in rep_results if r['condition'] != 'A_baseline']
    aug_results = [r for r in aug_results_all if r.get('aug_status') == 'ok']
    n_skipped = len(aug_results_all) - len(aug_results)
    n_tests = len(aug_results)
    improved = hurt = 0
    mean_delta = 0.0
    if n_skipped:
        skip_reasons = {}
        for r in aug_results_all:
            if r.get('aug_status') != 'ok':
                skip_reasons[r['aug_status']] = skip_reasons.get(r['aug_status'], 0) + 1
        print(f"  Skipped (not evaluable): {n_skipped} ({skip_reasons})")
    if n_tests > 0:
        sign = -1 if lower_is_better else 1
        improved = sum(1 for r in aug_results if sign * r['delta'] > 0)
        hurt = sum(1 for r in aug_results if sign * r['delta'] < 0)
        mean_delta = float(np.mean([sign * r['delta'] for r in aug_results]))
        all_conditions = sorted(set(r['condition'] for r in aug_results))
        print(f"  Total evaluable tests: {n_tests} "
              f"({len(base_reps_available)} reps x {len(models)} models x "
              f"{len(all_conditions)} conditions)")
        print(f"  Augmentation improved: {improved}/{n_tests}")
        print(f"  Augmentation hurt:     {hurt}/{n_tests}")
        print(f"  Mean delta:            {mean_delta:+.4f}")

        # Best improvement per condition
        for cond in sorted(set(r['condition'] for r in aug_results)):
            cond_results = [r for r in aug_results if r['condition'] == cond]
            cond_improved = sum(1 for r in cond_results if sign * r['delta'] > 0)
            cond_mean = float(np.mean([sign * r['delta'] for r in cond_results]))
            best_c = max(cond_results, key=lambda r: sign * r['delta'])
            print(f"  {cond}: {cond_improved}/{len(cond_results)} improved, "
                  f"mean={cond_mean:+.4f}, "
                  f"best={best_c['base_rep']}/{best_c['model']} "
                  f"delta={sign * best_c['delta']:+.4f}")
    else:
        print(f"  No evaluable augmentation results "
              f"(all {len(aug_results_all)} non-baseline tests were skipped).")

    return convert_numpy({
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_folds': n_folds,
        'n_samples': n_samples,
        'aug_reps': aug_reps_available,
        'aug_dims': aug_dims,
        'aug_groups': {k: v for k, v in aug_group_defs.items()},
        'rep_results': rep_results,
        'summary': {
            'n_tests': n_tests,          # evaluable (aug_status == 'ok') only
            'n_skipped': n_skipped,       # search_no_config / all_folds_failed
            'improved': improved if n_tests > 0 else 0,
            'hurt': hurt if n_tests > 0 else 0,
            'mean_delta': mean_delta if n_tests > 0 else 0.0,
            # When every non-baseline test was skipped the experiment produced
            # no augmentation signal — flag it so a NaN-saturated run is not
            # mistaken for a "no improvement" result.
            'all_skipped': (n_tests == 0 and n_skipped > 0),
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
