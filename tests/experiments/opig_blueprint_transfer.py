#!/usr/bin/env python3
"""OPIG C5: Cross-Dataset Blueprint Transfer — Pretrained Hybrid Evaluation.

OPIG Category C: Models and Evaluation Protocols.
Conclusion: Hybrid blueprints discovered on a large source dataset can
    transfer to a smaller target dataset, enabling publishable "pretrained"
    hybrid configurations.

Dependencies: None (standalone). Uses find_best_hybrid with cache=True.

Design:
    Source dataset: large dataset (e.g., logd ~7K molecules)
    Target dataset: small dataset (e.g., herg_ki ~1.4K molecules)
    Shared reps: intersection of both datasets' DATASET_REPS general lists.

    Conditions evaluated on target (per downstream model, per CV fold):
        A. base_best:        Best individual rep (no hybrid) — baseline.
        B. native:           Hybrid from target's own search blueprint — upper bound.
        C. recipe_transfer:  Source blueprint recipe (allocation, importance, budget,
                             evaluator) applied to target data. Feature selection
                             recomputed on target training data.
        D. full_transfer:    Source's exact feature indices applied directly to
                             target data. No recomputation — the "pretrained" hybrid.

    Also evaluates hybrid vs base reps on source (sanity check).

    Key questions:
        1. full_transfer > base_best?   -> Blueprints have direct transfer value.
        2. recipe_transfer ~ native?    -> Recipe generalises; no search needed.
        3. full_transfer ~ native?      -> Publish exact feature lists!

Usage:
    python opig_blueprint_transfer.py --source logd --target herg_ki --mode full
    python opig_blueprint_transfer.py --source adme_hlm --target caco2 --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, save_experiment_results, convert_numpy,
    get_model_factories, _cv_split, build_fold_hybrid,
    get_domain_metric,
    get_best_for_downstream,
    DOWNSTREAM_MODELS,
    FULL_BUDGETS, INTERNAL_EVALUATORS,
)
from kirby.find_best_hybrid import find_best_hybrid
from kirby.hybrid import create_hybrid, apply_feature_selection


def run(source_data, target_data, config, output_dir, source_name, target_name,
        pruning_mode='default_pruning'):
    """OPIG C5: Cross-dataset blueprint transfer."""
    print("\n" + "=" * 70)
    print("OPIG C5: CROSS-DATASET BLUEPRINT TRANSFER")
    print(f"  Source: {source_name} ({len(source_data['labels'])} samples)")
    print(f"  Target: {target_name} ({len(target_data['labels'])} samples)")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(target_data)
    cmp = (lambda a, b: a < b) if lower_is_better else (lambda a, b: a > b)

    source_reps = source_data['reps']
    source_labels = source_data['labels']
    source_groups = source_data['groups']

    target_reps = target_data['reps']
    target_labels = target_data['labels']
    target_groups = target_data['groups']

    n_folds = config['n_splits']
    task_type = target_data['task_type']

    # ── Domain consistency check ───────────────────────────────────────
    src_domain = source_data.get('domain', 'molecular')
    tgt_domain = target_data.get('domain', 'molecular')
    if src_domain != tgt_domain:
        print(f"  WARNING: domain mismatch (source={src_domain}, target={tgt_domain}). "
              f"Metrics and feature semantics may not be comparable.")

    # ── Validate rep compatibility ──────────────────────────────────────
    shared_rep_names = sorted(set(source_reps.keys()) & set(target_reps.keys()))
    dim_mismatches = []
    for rn in shared_rep_names:
        if source_reps[rn].shape[1] != target_reps[rn].shape[1]:
            dim_mismatches.append(rn)
            print(f"  WARNING: {rn} dim mismatch: source={source_reps[rn].shape[1]}, "
                  f"target={target_reps[rn].shape[1]} -- excluded")
    shared_rep_names = [r for r in shared_rep_names if r not in dim_mismatches]

    if len(shared_rep_names) < 2:
        print("  ERROR: Need at least 2 shared reps for meaningful comparison")
        return {'error': 'insufficient_shared_reps'}

    src = {k: source_reps[k] for k in shared_rep_names}
    tgt = {k: target_reps[k] for k in shared_rep_names}

    print(f"\n  Shared reps ({len(shared_rep_names)}): {', '.join(shared_rep_names)}")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")

    from experiments._shared import resolve_pruning_kwargs
    _src_pkw = resolve_pruning_kwargs(pruning_mode, len(source_labels),
                                       dataset_label=source_name)
    _tgt_pkw = resolve_pruning_kwargs(pruning_mode, len(target_labels),
                                       dataset_label=target_name)

    # ── Step 1: Hybrid search on source (single strategy, cached) ─────
    # Source/target are arbitrary OPIG datasets — no shared precompute.
    # Hardcode (tier=general, aug=none) per empirical winning trend; no
    # augmentations in this experiment.
    print(f"\n  Step 1: Hybrid search on source ({source_name})")
    source_search = find_best_hybrid(
        base_reps=src, labels=source_labels, groups=source_groups,
        speed=config['search_speed'],
        budgets=FULL_BUDGETS,
        augmentation_strategy='none',
        internal_evaluators=INTERNAL_EVALUATORS,
        metric=metric_name, task_type=task_type,
        verbose=False, cache=True,
        dataset_name=source_name, tier='general',
        adaptive_pruning=0.25,
        **_src_pkw,
    )

    print(f"\n  Step 2: Hybrid search on target ({target_name})")
    target_search = find_best_hybrid(
        base_reps=tgt, labels=target_labels, groups=target_groups,
        speed=config['search_speed'],
        budgets=FULL_BUDGETS,
        augmentation_strategy='none',
        internal_evaluators=INTERNAL_EVALUATORS,
        metric=metric_name, task_type=task_type,
        verbose=False, cache=True,
        dataset_name=target_name, tier='general',
        adaptive_pruning=0.25,
        **_tgt_pkw,
    )

    # ── Step 3: Per-model evaluation ────────────────────────────────────
    print(f"\n  Step 3: Per-model evaluation ({n_folds}-fold CV)")

    model_names = config.get('models', DOWNSTREAM_MODELS)
    model_factories_tgt = get_model_factories(
        model_names, task_type=task_type, n_samples=len(target_labels))
    model_factories_src = get_model_factories(
        model_names, task_type=task_type, n_samples=len(source_labels))

    target_folds = list(_cv_split(
        target_groups, n_folds,
        next(iter(tgt.values())), target_labels))
    source_folds = list(_cv_split(
        source_groups, n_folds,
        next(iter(src.values())), source_labels))

    per_model = {}

    for model_name in model_names:
        if model_name not in model_factories_tgt or model_name not in model_factories_src:
            continue

        source_best = get_best_for_downstream(source_search, model_name)
        target_best = get_best_for_downstream(target_search, model_name)

        if source_best is None or target_best is None:
            print(f"\n    {model_name}: skipped (missing search config)")
            continue

        source_bp = source_best['blueprint']
        target_bp = target_best['blueprint']

        print(f"\n    {model_name}:")
        print(f"      Source BP: budget={source_bp.budget}, "
              f"alloc={source_bp.allocation_method}, "
              f"imp={source_bp.importance_method}, "
              f"eval={source_bp.internal_evaluator}")
        print(f"      Target BP: budget={target_bp.budget}, "
              f"alloc={target_bp.allocation_method}, "
              f"imp={target_bp.importance_method}, "
              f"eval={target_bp.internal_evaluator}")

        # Build source hybrid on FULL source data -> feature_info for full_transfer
        try:
            _, fi_source = create_hybrid(
                src, source_labels,
                allocation_method=source_bp.allocation_method,
                budget=source_bp.budget,
                importance_method=source_bp.importance_method,
                internal_evaluator=source_bp.internal_evaluator,
                # Explicit: this experiment has no augmentations. Default of
                # 'uncapped' would silently mismatch the search recipe if
                # any future caller introduces augs without updating here.
                augmentations=None,
                augmentation_strategy='none',
            )
        except Exception as e:
            print(f"      Source hybrid build failed: {e}")
            continue

        # Report which features the source blueprint selected
        print(f"      Source blueprint feature selection:")
        for rn in sorted(fi_source.keys()):
            info = fi_source[rn]
            if isinstance(info, dict) and 'selected_indices' in info:
                n_sel = len(info['selected_indices'])
                total = src[rn].shape[1]
                print(f"        {rn}: {n_sel}/{total}")

        # Identify best single rep for this model from search baselines
        # (selected via search's internal CV — no leakage into fold eval)
        tgt_baselines = getattr(target_search, 'baselines', {})
        best_tgt_rep = None
        best_tgt_bl = None
        for rn, mscores in tgt_baselines.items():
            if rn.startswith('_') or rn not in shared_rep_names:
                continue
            s = mscores.get(model_name)
            if s is not None and not np.isnan(s):
                if best_tgt_bl is None or cmp(s, best_tgt_bl):
                    best_tgt_bl, best_tgt_rep = s, rn
        if best_tgt_rep is None:
            best_tgt_rep = shared_rep_names[0]

        src_baselines = getattr(source_search, 'baselines', {})
        best_src_rep = None
        best_src_bl = None
        for rn, mscores in src_baselines.items():
            if rn.startswith('_') or rn not in shared_rep_names:
                continue
            s = mscores.get(model_name)
            if s is not None and not np.isnan(s):
                if best_src_bl is None or cmp(s, best_src_bl):
                    best_src_bl, best_src_rep = s, rn
        if best_src_rep is None:
            best_src_rep = shared_rep_names[0]

        print(f"      Best base rep (target): {best_tgt_rep}")
        print(f"      Best base rep (source): {best_src_rep}")

        # ── 3a: Evaluate on TARGET ──────────────────────────────────────
        tgt_fold_scores = {
            'base_best': [],
            'native': [], 'recipe_transfer': [], 'full_transfer': [],
        }

        print(f"      Target evaluation ({target_name}):")
        for fold_i, (train_idx, val_idx) in enumerate(target_folds):
            tr = {k: v[train_idx] for k, v in tgt.items()}
            va = {k: v[val_idx] for k, v in tgt.items()}
            y_tr = target_labels[train_idx]
            y_va = target_labels[val_idx]
            mk = model_factories_tgt[model_name]

            # A: Best single rep (fixed across folds from search baselines)
            sc = StandardScaler()
            X_tr = sc.fit_transform(tr[best_tgt_rep])
            X_va = sc.transform(va[best_tgt_rep])
            m = mk()
            m.fit(X_tr, y_tr)
            tgt_fold_scores['base_best'].append(
                float(metric_fn(y_va, m.predict(X_va))))

            # B: Native hybrid
            try:
                X_tr_h, X_va_h, _ = build_fold_hybrid(tr, va, y_tr, target_bp)
                sc = StandardScaler()
                X_tr_h = sc.fit_transform(X_tr_h)
                X_va_h = sc.transform(X_va_h)
                m = mk()
                m.fit(X_tr_h, y_tr)
                tgt_fold_scores['native'].append(
                    float(metric_fn(y_va, m.predict(X_va_h))))
            except Exception as e:
                print(f"        Fold {fold_i} native failed: {e}")
                tgt_fold_scores['native'].append(float('nan'))

            # C: Recipe transfer (source recipe, target data)
            try:
                X_tr_h, X_va_h, _ = build_fold_hybrid(tr, va, y_tr, source_bp)
                sc = StandardScaler()
                X_tr_h = sc.fit_transform(X_tr_h)
                X_va_h = sc.transform(X_va_h)
                m = mk()
                m.fit(X_tr_h, y_tr)
                tgt_fold_scores['recipe_transfer'].append(
                    float(metric_fn(y_va, m.predict(X_va_h))))
            except Exception as e:
                print(f"        Fold {fold_i} recipe failed: {e}")
                tgt_fold_scores['recipe_transfer'].append(float('nan'))

            # D: Full transfer (source feature indices on target)
            try:
                X_tr_f = apply_feature_selection(tr, fi_source)
                X_va_f = apply_feature_selection(va, fi_source)
                sc = StandardScaler()
                X_tr_f = sc.fit_transform(X_tr_f)
                X_va_f = sc.transform(X_va_f)
                m = mk()
                m.fit(X_tr_f, y_tr)
                tgt_fold_scores['full_transfer'].append(
                    float(metric_fn(y_va, m.predict(X_va_f))))
            except Exception as e:
                print(f"        Fold {fold_i} full failed: {e}")
                tgt_fold_scores['full_transfer'].append(float('nan'))

            print(f"        Fold {fold_i}: base={tgt_fold_scores['base_best'][-1]:.4f} "
                  f"({best_tgt_rep}), native={tgt_fold_scores['native'][-1]:.4f}, "
                  f"recipe={tgt_fold_scores['recipe_transfer'][-1]:.4f}, "
                  f"full={tgt_fold_scores['full_transfer'][-1]:.4f}")

        # ── 3b: Evaluate on SOURCE (context) ────────────────────────────
        src_fold_scores = {'base_best': [], 'hybrid': []}

        print(f"      Source evaluation ({source_name}):")
        for fold_i, (train_idx, val_idx) in enumerate(source_folds):
            tr = {k: v[train_idx] for k, v in src.items()}
            va = {k: v[val_idx] for k, v in src.items()}
            y_tr = source_labels[train_idx]
            y_va = source_labels[val_idx]
            mk = model_factories_src[model_name]

            # Best single rep (fixed across folds from search baselines)
            sc = StandardScaler()
            X_tr = sc.fit_transform(tr[best_src_rep])
            X_va = sc.transform(va[best_src_rep])
            m = mk()
            m.fit(X_tr, y_tr)
            src_fold_scores['base_best'].append(
                float(metric_fn(y_va, m.predict(X_va))))

            # Hybrid
            try:
                X_tr_h, X_va_h, _ = build_fold_hybrid(tr, va, y_tr, source_bp)
                sc = StandardScaler()
                X_tr_h = sc.fit_transform(X_tr_h)
                X_va_h = sc.transform(X_va_h)
                m = mk()
                m.fit(X_tr_h, y_tr)
                src_fold_scores['hybrid'].append(
                    float(metric_fn(y_va, m.predict(X_va_h))))
            except Exception as e:
                print(f"        Fold {fold_i} source hybrid failed: {e}")
                src_fold_scores['hybrid'].append(float('nan'))

            print(f"        Fold {fold_i}: base={src_fold_scores['base_best'][-1]:.4f}, "
                  f"hybrid={src_fold_scores['hybrid'][-1]:.4f}")

        # ── Aggregate for this model ────────────────────────────────────
        def _agg(scores):
            return {'mean': float(np.nanmean(scores)),
                    'std': float(np.nanstd(scores)),
                    'per_fold': scores}

        model_res = {
            'target': {},
            'source': {
                'base_best': _agg(src_fold_scores['base_best']),
                'hybrid': _agg(src_fold_scores['hybrid']),
            },
            'source_blueprint': {
                'budget': source_bp.budget,
                'allocation': source_bp.allocation_method,
                'importance': source_bp.importance_method,
                'evaluator': source_bp.internal_evaluator,
            },
            'target_blueprint': {
                'budget': target_bp.budget,
                'allocation': target_bp.allocation_method,
                'importance': target_bp.importance_method,
                'evaluator': target_bp.internal_evaluator,
            },
        }
        for cond in ['base_best', 'native', 'recipe_transfer', 'full_transfer']:
            model_res['target'][cond] = _agg(tgt_fold_scores[cond])
        model_res['target']['base_rep'] = best_tgt_rep
        model_res['source']['base_rep'] = best_src_rep

        # Deltas vs target base
        base_mean = model_res['target']['base_best']['mean']
        for cond in ['native', 'recipe_transfer', 'full_transfer']:
            cond_mean = model_res['target'][cond]['mean']
            delta = (base_mean - cond_mean) if lower_is_better else (cond_mean - base_mean)
            model_res['target'][cond]['delta_vs_base'] = float(delta)

        per_model[model_name] = model_res

    # ── Summary table ───────────────────────────────────────────────────
    if not per_model:
        print("\n  No models produced results.")
        return {'error': 'no_results'}

    arrow = '\u2193' if lower_is_better else '\u2191'
    print(f"\n  {'='*90}")
    print(f"  CROSS-DATASET BLUEPRINT TRANSFER SUMMARY")
    print(f"  Source: {source_name} -> Target: {target_name}")
    print(f"  Metric: {metric_name} ({arrow} better)")
    print(f"  {'='*90}")

    # Source context
    print(f"\n  Source ({source_name}):")
    print(f"  {'Model':<12} {'Base':>10} {'Hybrid':>10} {'Delta':>10}")
    print(f"  {'-'*42}")
    for mn, res in per_model.items():
        sb = res['source']['base_best']['mean']
        sh = res['source']['hybrid']['mean']
        d = (sb - sh) if lower_is_better else (sh - sb)
        print(f"  {mn:<12} {sb:>10.4f} {sh:>10.4f} {d:>+10.4f}")

    # Target
    print(f"\n  Target ({target_name}):")
    print(f"  {'Model':<12} {'Base':>10} {'Native':>10} {'Recipe':>10} "
          f"{'Full':>10} {'d(Rec)':>8} {'d(Full)':>8}")
    print(f"  {'-'*78}")
    for mn, res in per_model.items():
        t = res['target']
        print(f"  {mn:<12} "
              f"{t['base_best']['mean']:>10.4f} "
              f"{t['native']['mean']:>10.4f} "
              f"{t['recipe_transfer']['mean']:>10.4f} "
              f"{t['full_transfer']['mean']:>10.4f} "
              f"{t['recipe_transfer']['delta_vs_base']:>+8.4f} "
              f"{t['full_transfer']['delta_vs_base']:>+8.4f}")

    # Aggregate interpretation
    d_recipe = [r['target']['recipe_transfer']['delta_vs_base']
                for r in per_model.values()]
    d_full = [r['target']['full_transfer']['delta_vs_base']
              for r in per_model.values()]
    mean_d_recipe = float(np.mean(d_recipe))
    mean_d_full = float(np.mean(d_full))
    n_recipe_win = sum(1 for d in d_recipe if d > 0.001)
    n_full_win = sum(1 for d in d_full if d > 0.001)
    n_models = len(per_model)

    print(f"\n  Mean delta recipe: {mean_d_recipe:+.4f}  "
          f"(wins {n_recipe_win}/{n_models} models)")
    print(f"  Mean delta full:   {mean_d_full:+.4f}  "
          f"(wins {n_full_win}/{n_models} models)")

    if n_full_win > n_models // 2:
        print(f"\n  RESULT: Blueprint FULLY TRANSFERS. "
              f"Exact feature indices from {source_name} improve "
              f"performance on {target_name}.")
    elif n_recipe_win > n_models // 2:
        print(f"\n  RESULT: Blueprint RECIPE transfers. "
              f"Rerun feature selection with the source recipe on target data.")
    else:
        print(f"\n  RESULT: Blueprint does NOT transfer reliably. "
              f"Hybrid is dataset-specific for this pair.")

    # ── Compile results ─────────────────────────────────────────────────
    results = {
        'source_dataset': source_name,
        'target_dataset': target_name,
        'source_n_samples': len(source_labels),
        'target_n_samples': len(target_labels),
        'shared_reps': shared_rep_names,
        'n_shared_reps': len(shared_rep_names),
        'n_folds': n_folds,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'per_model': convert_numpy(per_model),
        'aggregate': {
            'mean_delta_recipe': mean_d_recipe,
            'mean_delta_full': mean_d_full,
            'n_recipe_wins': n_recipe_win,
            'n_full_wins': n_full_win,
            'n_models': n_models,
        },
    }

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='OPIG C5: Cross-Dataset Blueprint Transfer')
    parser.add_argument('--source', type=str, default='logd',
                        help='Source (large) dataset name (default: logd)')
    parser.add_argument('--target', type=str, default='herg_ki',
                        help='Target (small) dataset name (default: herg_ki)')
    parser.add_argument('--mode', choices=['intermediary', 'full'],
                        default='intermediary',
                        help='intermediary (1 split) or full (5-fold CV)')
    parser.add_argument('--reps', type=str, default=None,
                        help='Comma-separated rep list (default: intersection of DATASET_REPS)')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated downstream model list')
    parser.add_argument('--budgets', type=str, default=None,
                        help='Comma-separated budget list')
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Max folds to evaluate (stops early)')
    parser.add_argument('--augment', action='store_true',
                        help='Include augmentation reps')
    parser.add_argument('--pruning-mode', type=str, default='default_pruning',
                        help="Adaptive pruning mode (default 'default_pruning'). "
                             "'aggressive_pruning' = F16-validated multi-stage pruning "
                             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")
    args = parser.parse_args()

    # Compute shared reps from DATASET_REPS intersection
    if args.reps:
        shared_rep_names = [r.strip() for r in args.reps.split(',')]
    else:
        try:
            from experiments.complexity.complexity_helpers import DATASET_REPS
            src_gen = DATASET_REPS.get(args.source, {}).get('general', [])
            tgt_gen = DATASET_REPS.get(args.target, {}).get('general', [])
            if src_gen and tgt_gen:
                shared_rep_names = sorted(set(src_gen) & set(tgt_gen))
                print(f"  Auto-intersected reps: {len(shared_rep_names)} shared "
                      f"(source={len(src_gen)}, target={len(tgt_gen)})")
            else:
                shared_rep_names = None
                print(f"  WARNING: DATASET_REPS not found for both datasets, "
                      f"using default preset")
        except ImportError:
            shared_rep_names = None

    # Load both datasets with the shared rep set
    source_data = load_data(args.source, rep_names=shared_rep_names,
                            augment=args.augment)
    target_data = load_data(args.target, rep_names=shared_rep_names,
                            augment=args.augment)

    config = get_config(args)

    results = run(source_data, target_data, config, args.output,
                  args.source, args.target,
                  pruning_mode=args.pruning_mode)
    save_experiment_results(results, args.output,
                           'opig_blueprint_transfer',
                           f'{args.source}_to_{args.target}')
