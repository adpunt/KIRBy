#!/usr/bin/env python3
"""C21: Single-Rep Augmentation Test.

Paper: Complexity
Question: a top-tier standalone rep already does well and hybridisation adds
    little. Can a handful of INDIVIDUAL extra features nudge it further?

Design: for each base rep in DATASET_REPS[dataset]['top_tier']:
    A_baseline      base rep alone
    B_aug           base + the dataset's augmentation-category reps, uncapped.
                    These are the small engineered feature sets the surveys
                    assigned to 'augmentation' (PIPELINE_READINESS_TODO.md).
    C_topreps       base + the top-N features of every OTHER top_tier rep,
                    each capped to N by per-fold RF importance.
    D_both          B and C together (aug reps uncapped, top reps capped to N).

Every condition uses augmentation_strategy='uncapped': candidates are ranked,
then added ONE AT A TIME and kept only if the downstream score improves. That
is the original augmentation concept — individual features, keep what helps.

Conditions are skipped, not faked, when their pool is empty: on datasets whose
survey found no augmentation rep helps, `augmentation` is legitimately [] and
B/D do not run. C still runs, and is the condition that answers "can other top
reps donate a few good features".

Usage:
    python c21_single_rep_augmentation.py --dataset herg --mode intermediary
    python c21_single_rep_augmentation.py --dataset abcov_kd --top-n 5
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
    INTERNAL_EVALUATORS,
)
from kirby.find_best_hybrid import find_best_single_rep_augmented
from experiments.complexity.complexity_helpers import DATASET_REPS

# Floor on the denominator of relative_improvement (= delta / |base_score|).
# On antibody tasks the base Spearman rho routinely lands near zero, and an
# unfloored |base_score| manufactures absurd relative improvements (e.g. a
# base rho of 0.004 turned a modest +0.12 delta into a bogus +27.8). Below
# this magnitude the base score is not a meaningful signal, so we clamp the
# denominator to keep the ratio finite and interpretable. Base scores at or
# above the floor are unaffected.
REL_IMPROVEMENT_DENOM_FLOOR = 0.1

# Features taken from each OTHER top_tier rep, chosen per fold by RF importance.
DEFAULT_TOP_N = 5


def _build_conditions(base_name, top_tier_reps, aug_reps, top_n):
    """Return {condition_name: (aug_pool, per_source_budget)}.

    per_source_budget is a dict so the wide top-tier reps get capped to top_n
    while the small engineered augmentation reps stay uncapped — a source
    absent from the dict contributes all of its columns.
    """
    other_top = {k: v for k, v in top_tier_reps.items() if k != base_name}
    aug_only = {k: v for k, v in aug_reps.items() if k != base_name}

    conditions = {'A_baseline': (None, None)}
    if aug_only:
        conditions['B_aug'] = (aug_only, None)
    if other_top:
        conditions[f'C_topreps_n{top_n}'] = (
            other_top, {k: top_n for k in other_top})
    if aug_only and other_top:
        both = {**aug_only, **other_top}
        conditions[f'D_both_n{top_n}'] = (
            both, {k: top_n for k in other_top})
    return conditions


def run(data, config, output_dir, dataset_name, top_n=DEFAULT_TOP_N, **kwargs):
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

    ds_config = DATASET_REPS.get(dataset_name, {})
    top_tier_names = ds_config.get('top_tier', [])
    aug_rep_names = ds_config.get('augmentation', [])

    # Base reps are TOP TIER. c21 asks "can the BEST rep be nudged further" —
    # the general pool is the hybrid search pool, not the answer to that.
    # Augmentation reps live outside 'general', so they are loaded separately.
    missing = [r for r in (top_tier_names + aug_rep_names) if r not in reps]
    if missing:
        print(f"  Loading {len(missing)} reps not in the general pool: {missing}")
        extra = load_data(dataset_name, rep_preset='all', rep_names=missing)
        for rname, arr in extra['reps'].items():
            reps[rname] = arr

    top_tier_reps = {r: reps[r] for r in top_tier_names if r in reps}
    aug_reps = {r: reps[r] for r in aug_rep_names if r in reps}

    if not top_tier_reps:
        print(f"  No top_tier reps available for {dataset_name} — skipping C21")
        return {'error': 'no_top_tier_reps', 'dataset': dataset_name}

    print(f"  Base reps (top_tier, {len(top_tier_reps)}): "
          f"{', '.join(f'{k}={v.shape[1]}D' for k, v in top_tier_reps.items())}")
    if aug_reps:
        print(f"  Augmentation reps ({len(aug_reps)}): "
              f"{', '.join(f'{k}={v.shape[1]}D' for k, v in aug_reps.items())}")
    else:
        # Not a defect: the survey tested augmentation on this dataset and none
        # helped, so PIPELINE_READINESS_TODO records "Augmentation | none".
        print(f"  Augmentation reps: NONE configured for {dataset_name} "
              f"(survey found none helped) — B/D conditions will be skipped")
    print(f"  Top-N features taken from each other top_tier rep: {top_n}")

    search_speed = config.get('search_speed', 'full')
    rep_results = []

    for base_name, X_base in top_tier_reps.items():
        base_dim = X_base.shape[1]
        print(f"\n  {base_name} ({base_dim}D)")

        conditions = _build_conditions(base_name, top_tier_reps, aug_reps, top_n)
        if len(conditions) == 1:
            print(f"    only baseline available (no aug reps, no other top reps)")

        for cond_name, (aug_dict, per_source) in conditions.items():
            is_baseline = aug_dict is None

            best_by_model = None
            bp = None
            if not is_baseline:
                # allow_budget_exceeds_n: give the base rep its FULL native
                # dimensionality even when base_dim > n_samples. Without it the
                # engine clamps the budget to n_samples and the augmented arm
                # silently competes with fewer base features than the baseline.
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
                    aug_per_source_budget=per_source,
                    allow_budget_exceeds_n=True,
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
                    print(f"    {cond_name}: search returned no config "
                          f"(best_by_model has {len(best_by_model)} entries)")

            for model_name, model_factory in models.items():
                # Use this downstream model's own blueprint when available
                # (knn/xgboost should not be scored against rf-optimal feature
                # selection). Falls back to rf when the search didn't run.
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

                    scaler_b = StandardScaler()
                    Xtr_b = scaler_b.fit_transform(X_base[train_idx])
                    Xva_b = scaler_b.transform(X_base[val_idx])
                    m1 = model_factory()
                    m1.fit(Xtr_b, y_train)
                    scores_base.append(float(metric_fn(y_val, m1.predict(Xva_b))))

                    if is_baseline or bp is None:
                        scores_aug.append(float('nan'))
                        fold_aug_info.append({'kept': [], 'n_added': 0,
                                              'final_dim': base_dim})
                        continue

                    train_base = {base_name: X_base[train_idx]}
                    val_base = {base_name: X_base[val_idx]}
                    train_augs = {rn: aug_dict[rn][train_idx] for rn in aug_dict}
                    val_augs = {rn: aug_dict[rn][val_idx] for rn in aug_dict}

                    try:
                        # per_source must match the search — the blueprint does
                        # not carry it, so an omission here would rebuild the
                        # fold with every column pooled.
                        Xtr_h, Xva_h, feature_info = build_fold_hybrid(
                            train_base, val_base, y_train, bp_model,
                            augmentations_train=train_augs,
                            augmentations_val=val_augs,
                            aug_per_source_budget=per_source)

                        scaler_h = StandardScaler()
                        Xtr_h = scaler_h.fit_transform(Xtr_h)
                        Xva_h = scaler_h.transform(Xva_h)
                        m2 = model_factory()
                        m2.fit(Xtr_h, y_train)
                        scores_aug.append(float(metric_fn(y_val, m2.predict(Xva_h))))

                        aug_info = feature_info.get('augmentation_info', {})
                        # Per-source detail: which rep donated how many columns.
                        kept_detail = {
                            k: int(v.get('n_features', 0))
                            for k, v in (aug_info or {}).items()
                        }
                        n_added = Xtr_h.shape[1] - base_dim
                        fold_aug_info.append({
                            'kept': sorted(kept_detail),
                            'kept_detail': kept_detail,
                            'n_added': max(0, n_added),
                            'final_dim': Xtr_h.shape[1],
                        })
                    except Exception as e:
                        print(f"    {model_name} fold {fold_idx}: augmentation failed - {e}")
                        scores_aug.append(float('nan'))
                        fold_aug_info.append({'kept': [], 'n_added': 0,
                                              'final_dim': base_dim})

                base_mean = float(np.nanmean(scores_base))
                # A search that found no winning config makes every augmented
                # fold NaN. Flag it explicitly rather than emitting a bare NaN
                # aug_mean, which is what nan-saturated the old c21 results.
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

                all_kept = set()
                total_added = []
                for fi in fold_aug_info:
                    all_kept.update(fi['kept'])
                    total_added.append(fi['n_added'])

                result = {
                    'base_rep': base_name,
                    'base_tier': 'top_tier',
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
                    # metric direction. Without this, RMSE/MAE wins are stored
                    # as negative relative_improvement and any downstream filter
                    # on positive sign rejects them.
                    'relative_improvement': (
                        (-1 if lower_is_better else 1) * delta
                        / max(abs(base_mean), REL_IMPROVEMENT_DENOM_FLOOR)
                        if not np.isnan(delta) else float('nan')),
                    'kept_augmentations': sorted(all_kept),
                    'mean_features_added': float(np.mean(total_added)),
                    'fold_aug_info': fold_aug_info,
                    'aug_pool_reps': sorted(aug_dict) if aug_dict else [],
                    'per_source_budget': per_source,
                    'top_n': top_n,
                    'blueprint': bp.to_dict() if bp else None,
                }
                rep_results.append(result)

                sign = -1 if lower_is_better else 1
                if aug_status not in ('ok', 'baseline'):
                    print(f"    {cond_name}/{model_name}: base={base_mean:.4f}  "
                          f"SKIPPED ({aug_status})")
                elif is_baseline:
                    print(f"    {cond_name}/{model_name}: base={base_mean:.4f}")
                else:
                    status = "+" if sign * delta > 0 else "-" if sign * delta < 0 else "="
                    print(f"    {cond_name}/{model_name}: base={base_mean:.4f}  "
                          f"aug={aug_mean:.4f}  delta={sign * delta:+.4f} [{status}]  "
                          f"kept={sorted(all_kept)}  added={np.mean(total_added):.1f}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Only aggregate over results that were actually evaluable. Skipped ones
    # carry a NaN delta and would poison the mean / argmax; count them
    # separately so a NaN-saturated run reads as "N skipped" rather than a
    # meaningless NaN mean_delta.
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
              f"({len(top_tier_reps)} top-tier reps x {len(models)} models x "
              f"{len(all_conditions)} conditions)")
        print(f"  Augmentation improved: {improved}/{n_tests}")
        print(f"  Augmentation hurt:     {hurt}/{n_tests}")
        print(f"  Mean delta:            {mean_delta:+.4f}")

        for cond in all_conditions:
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
        'top_n': top_n,
        'base_reps': sorted(top_tier_reps),
        'base_dims': {k: int(v.shape[1]) for k, v in top_tier_reps.items()},
        'aug_reps': sorted(aug_reps),
        'aug_dims': {k: int(v.shape[1]) for k, v in aug_reps.items()},
        'rep_results': rep_results,
        'summary': {
            'n_tests': n_tests,          # evaluable (aug_status == 'ok') only
            'n_skipped': n_skipped,      # search_no_config / all_folds_failed
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
    parser.add_argument('--top-n', type=int, default=DEFAULT_TOP_N,
                        help='Features taken from each other top_tier rep '
                             f'(default {DEFAULT_TOP_N})')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset, top_n=args.top_n)
    save_experiment_results(results, args.output, 'c21_single_rep_augmentation',
                            args.dataset)
