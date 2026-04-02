#!/usr/bin/env python3
"""F06: Augmentation strategy test — core reps vs augmentation pathways.

Paper: Framework
Question: When reps are split into "core" (top N by single-rep baseline) and
    "candidate" (rest), is it better to treat candidates as base reps (standard
    hybrid) or as augmentations (post-hoc greedy ablation)?  How much does
    augmentation_budget matter?

For each task:
    1. Rank reps by single-rep RF baseline (domain-aware metric).
    2. Split into core (top N) and candidates (rest).
    3. Test 4 conditions:
        a. all-as-base:       All reps are base_reps, no augmentations.
        b. core+greedy_ablation: Core = base_reps, candidates = augmentations
                                 with strategy='greedy_ablation'.
        c. core+all:          Core = base_reps, candidates = augmentations
                                 with strategy='all'.
        d. core-only:         Core = base_reps only, no candidates.
    4. Sweep augmentation_budget: [10, 20, 50].
    5. Compare scores and which augmentations were kept.

Usage:
    python f06_augmentation_test.py --dataset logd --reps all --mode full
    python f06_augmentation_test.py --dataset flab_thermo_tresanco --reps all --mode full
    python f06_augmentation_test.py --dataset tcga_luad_vs_lusc --reps all --mode full
    python f06_augmentation_test.py --dataset buchwald_hartwig --reps all --mode full

    # Custom core size and importance method:
    python f06_augmentation_test.py --dataset logd --n-core 5 --importance-method lightgbm_gain
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_model_factories, get_domain_metric, cv_split,
    evaluate_single_rep_cv, parse_rep_args, FRAMEWORK_TASKS,
    INTERNAL_EVALUATORS, DOWNSTREAM_MODELS, extract_top3_base_reps,
)
from kirby.hybrid import (
    create_hybrid, apply_feature_selection, apply_augmentation_selection,
)


# Budget sweep for augmentations (only applies to conditions b and c)
AUG_BUDGETS = [10, 20, 50]

# Hybrid budget for create_hybrid (base reps allocation)
HYBRID_BUDGETS = [100, 200]

# Models to evaluate downstream
EVAL_MODELS = DOWNSTREAM_MODELS

# Default number of core reps
DEFAULT_N_CORE = 5


def _rank_reps_by_baseline(reps, labels, groups, metric_fn, lower_is_better,
                           task_type, n_folds=3):
    """Rank reps by single-rep RF baseline. Returns sorted list of (name, score)."""
    factories = get_model_factories(['rf'], task_type=task_type,
                                    n_samples=len(labels))
    if 'rf' not in factories:
        # Fallback to ridge if RF unavailable
        factories = get_model_factories(['ridge'], task_type=task_type,
                                        n_samples=len(labels))
    model_name = next(iter(factories))
    model_factory = factories[model_name]

    baselines = []
    for rep_name, X_rep in reps.items():
        mean_score, _ = evaluate_single_rep_cv(
            X_rep, labels, groups, model_factory,
            n_folds=n_folds, mode='full', metric_fn=metric_fn,
        )
        baselines.append((rep_name, mean_score))

    # Sort: higher is better for higher-is-better, lower for lower-is-better
    baselines.sort(key=lambda x: x[1], reverse=(not lower_is_better))
    return baselines


def _evaluate_condition(base_reps, labels, groups, model_factory, metric_fn,
                        n_folds, allocation_method, importance_method,
                        hybrid_budget, augmentations=None,
                        augmentation_strategy='none', augmentation_budget=20,
                        lower_is_better=False):
    """Evaluate a single condition via per-fold CV.

    Returns dict with mean_score, std_score, n_features, fold_scores,
    and per-fold augmentation details if applicable.
    """
    sample_X = next(iter(base_reps.values()))
    fold_scores = []
    fold_n_features = []
    fold_aug_details = []

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels,
                                       mode='full'):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Split augmentations per fold
        train_augs = None
        val_augs = None
        if augmentations is not None and augmentation_strategy != 'none':
            train_augs = {k: v[train_idx] for k, v in augmentations.items()}
            val_augs = {k: v[val_idx] for k, v in augmentations.items()}

        try:
            # For iterative strategy, pass the downstream model so augmentation
            # decisions are model-aware (fill-then-replace A/B testing).
            iter_kwargs = {}
            if augmentation_strategy == 'iterative':
                iter_kwargs = dict(
                    model=model_factory(),
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better,
                )

            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=hybrid_budget,
                importance_method=importance_method,
                augmentations=train_augs,
                augmentation_strategy=augmentation_strategy,
                augmentation_budget=augmentation_budget,
                **iter_kwargs,
            )
            X_val = apply_feature_selection(val_reps, feature_info)

            # Apply augmentation selection to val set
            aug_info = feature_info.get('augmentation_info')
            if aug_info and val_augs is not None:
                aug_val_features = apply_augmentation_selection(
                    val_augs, aug_info)
                if aug_val_features is not None:
                    X_val = np.hstack([X_val, aug_val_features])
        except Exception as e:
            import traceback
            print(f"\n      WARNING: create_hybrid failed: {e}")
            traceback.print_exc()
            fold_aug_details.append({'error': str(e)})
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=1e10,
                                  neginf=-1e10)
        X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=1e10, neginf=-1e10)

        model = model_factory()
        model.fit(X_train_s, y_train)
        score = metric_fn(y_val, model.predict(X_val_s))
        fold_scores.append(score)
        fold_n_features.append(X_train.shape[1])

        # Record augmentation details
        aug_detail = {'n_features': X_train.shape[1]}
        if aug_info:
            aug_detail['kept_augmentations'] = list(aug_info.keys())
            aug_detail['aug_features'] = sum(
                info.get('n_features', 0) for info in aug_info.values()
            )
        fold_aug_details.append(aug_detail)

    if not fold_scores:
        return {
            'mean_score': float('nan'),
            'std_score': float('nan'),
            'n_features': 0,
            'fold_scores': [],
            'aug_details': fold_aug_details,
        }

    return {
        'mean_score': float(np.mean(fold_scores)),
        'std_score': float(np.std(fold_scores)),
        'n_features': int(np.mean(fold_n_features)),
        'fold_scores': [float(s) for s in fold_scores],
        'aug_details': fold_aug_details,
    }


def run(data, config, output_dir, dataset_name, importance_method='treeshap',
        n_core=DEFAULT_N_CORE, **kwargs):
    """Run augmentation strategy comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F06: AUGMENTATION STRATEGY TEST")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    domain = data.get('domain', 'molecular')
    task_type = data.get('task_type', 'regression')
    n_samples = len(labels)
    n_folds = config['n_splits']

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Reps: {len(reps)} total")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Core size: {n_core}")
    print(f"  Importance method: {importance_method}")
    print(f"  N folds: {n_folds}")
    print(f"  Hybrid budgets: {HYBRID_BUDGETS}")
    print(f"  Aug budgets: {AUG_BUDGETS}")
    print(f"  Models: {EVAL_MODELS}\n", flush=True)

    # ------------------------------------------------------------------
    # Step 1: Rank reps by single-rep baseline
    # ------------------------------------------------------------------
    print("  Step 1: Ranking reps by single-rep baseline...", flush=True)
    t0 = time.time()
    ranked = _rank_reps_by_baseline(
        reps, labels, groups, metric_fn, lower_is_better,
        task_type, n_folds=min(n_folds, 3),
    )
    t_rank = time.time() - t0

    print(f"  Ranking done ({t_rank:.1f}s). Top reps:")
    for i, (name, score) in enumerate(ranked):
        tag = " [CORE]" if i < n_core else ""
        print(f"    {i+1:3d}. {name:<30} {metric_name}={score:.4f}{tag}")

    # Clamp n_core to available reps (need at least 2 base reps for hybrid)
    n_core = min(n_core, len(ranked) - 1)
    n_core = max(n_core, 2)  # need at least 2 for create_hybrid

    core_names = [name for name, _ in ranked[:n_core]]
    candidate_names = [name for name, _ in ranked[n_core:]]

    core_reps = {k: reps[k] for k in core_names}
    candidate_reps = {k: reps[k] for k in candidate_names}

    print(f"\n  Core ({n_core}): {core_names}")
    print(f"  Candidates ({len(candidate_names)}): {candidate_names}")

    if len(candidate_names) == 0:
        print("  WARNING: No candidate reps — only core-only and all-as-base "
              "conditions are meaningful.")

    # ------------------------------------------------------------------
    # Step 2: Get model factories
    # ------------------------------------------------------------------
    factories = get_model_factories(EVAL_MODELS, task_type=task_type,
                                    n_samples=n_samples)
    print(f"  Models: {sorted(factories.keys())}\n", flush=True)

    # ------------------------------------------------------------------
    # Step 3: Run conditions
    # ------------------------------------------------------------------
    all_results = []

    # Define conditions
    # Each condition: (name, base_reps_dict, augmentations_dict_or_None,
    #                  augmentation_strategy, uses_aug_budget)
    conditions = [
        ('all_as_base', reps, None, 'none', False),
        ('core_only', core_reps, None, 'none', False),
    ]
    if candidate_names:
        conditions.extend([
            ('core+greedy_ablation', core_reps, candidate_reps,
             'greedy_ablation', True),
            ('core+iterative', core_reps, candidate_reps,
             'iterative', True),
            ('core+all', core_reps, candidate_reps, 'all', True),
        ])

    total_configs = 0
    for _, _, _, _, uses_aug_budget in conditions:
        n_aug = len(AUG_BUDGETS) if uses_aug_budget else 1
        total_configs += len(HYBRID_BUDGETS) * len(factories) * n_aug

    print(f"  Total configurations: {total_configs}\n", flush=True)

    config_idx = 0
    for cond_name, base, augs, aug_strategy, uses_aug_budget in conditions:
        print(f"\n  --- Condition: {cond_name} ---")
        print(f"      Base reps: {len(base)}, "
              f"Augmentations: {len(augs) if augs else 0}, "
              f"Strategy: {aug_strategy}")

        aug_budget_list = AUG_BUDGETS if uses_aug_budget else [0]

        for hybrid_budget in HYBRID_BUDGETS:
            for aug_budget in aug_budget_list:
                for model_name, model_factory in sorted(factories.items()):
                    config_idx += 1
                    label = (f"[{config_idx}/{total_configs}] "
                             f"{cond_name} | budget={hybrid_budget} | "
                             f"aug_budget={aug_budget} | model={model_name}")
                    print(f"    {label}...", end=" ", flush=True)

                    t0 = time.time()
                    result = _evaluate_condition(
                        base_reps=base,
                        labels=labels,
                        groups=groups,
                        model_factory=model_factory,
                        metric_fn=metric_fn,
                        n_folds=n_folds,
                        allocation_method='greedy',
                        importance_method=importance_method,
                        hybrid_budget=hybrid_budget,
                        augmentations=augs,
                        augmentation_strategy=aug_strategy,
                        augmentation_budget=aug_budget,
                        lower_is_better=lower_is_better,
                    )
                    elapsed = time.time() - t0

                    result_entry = {
                        'condition': cond_name,
                        'hybrid_budget': hybrid_budget,
                        'augmentation_budget': aug_budget,
                        'augmentation_strategy': aug_strategy,
                        'model': model_name,
                        'importance_method': importance_method,
                        'n_base_reps': len(base),
                        'n_augmentation_reps': len(augs) if augs else 0,
                        'wall_time': elapsed,
                        **result,
                    }
                    all_results.append(result_entry)

                    ms = result['mean_score']
                    ss = result['std_score']
                    nf = result['n_features']
                    print(f"{metric_name}={ms:.4f}±{ss:.4f} "
                          f"({nf} feats, {elapsed:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Step 4: Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Group by condition and compute mean score across models/budgets
    from collections import defaultdict
    cond_scores = defaultdict(list)
    for r in all_results:
        key = r['condition']
        if r['augmentation_budget'] > 0:
            key += f" (aug_b={r['augmentation_budget']})"
        cond_scores[key].append(r['mean_score'])

    print(f"\n  Mean {metric_name} by condition "
          f"({'lower' if lower_is_better else 'higher'} is better):")
    for key in sorted(cond_scores.keys()):
        scores = [s for s in cond_scores[key] if not np.isnan(s)]
        if scores:
            mean = np.mean(scores)
            print(f"    {key:<45} {mean:.4f} (n={len(scores)})")

    # Best per model
    print(f"\n  Best {metric_name} per model:")
    for model_name in sorted(factories.keys()):
        model_results = [r for r in all_results if r['model'] == model_name]
        if not model_results:
            continue
        if lower_is_better:
            best = min(model_results,
                       key=lambda r: r['mean_score']
                       if not np.isnan(r['mean_score']) else float('inf'))
        else:
            best = max(model_results,
                       key=lambda r: r['mean_score']
                       if not np.isnan(r['mean_score']) else float('-inf'))
        print(f"    {model_name:<12} {best['condition']:<30} "
              f"budget={best['hybrid_budget']} "
              f"aug_b={best['augmentation_budget']} "
              f"{metric_name}={best['mean_score']:.4f}")

    # Augmentation retention (for greedy_ablation and iterative conditions)
    ga_results = [r for r in all_results
                  if r['augmentation_strategy'] in ('greedy_ablation', 'iterative')]
    if ga_results:
        print(f"\n  Augmentation retention:")
        for r in ga_results:
            kept_per_fold = []
            for d in r.get('aug_details', []):
                if 'kept_augmentations' in d:
                    kept_per_fold.append(d['kept_augmentations'])
            if kept_per_fold:
                # Count how often each candidate was kept
                from collections import Counter
                counts = Counter()
                for kept_list in kept_per_fold:
                    counts.update(kept_list)
                n_f = len(kept_per_fold)
                kept_str = ", ".join(
                    f"{name}({c}/{n_f})"
                    for name, c in counts.most_common(5)
                )
                print(f"    {r['condition']} | budget={r['hybrid_budget']} | "
                      f"aug_b={r['augmentation_budget']} | "
                      f"model={r['model']}: {kept_str}")

    # ------------------------------------------------------------------
    # Build structured output
    # ------------------------------------------------------------------
    return make_structured_output(
        experiment_id='f06',
        dataset=dataset_name,
        data=data,
        config={
            'importance_method': importance_method,
            'n_core': n_core,
            'hybrid_budgets': HYBRID_BUDGETS,
            'aug_budgets': AUG_BUDGETS,
            'eval_models': EVAL_MODELS,
            'n_folds': n_folds,
            'core_reps': core_names,
            'candidate_reps': candidate_names,
        },
        results={
            'rep_ranking': [{'rep': name, metric_name: score}
                            for name, score in ranked],
            'conditions': all_results,
            'ranking_time': t_rank,
        },
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser('F06: Augmentation Strategy Test',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--importance-method', type=str, default='treeshap',
                        help='Importance method for hybrid allocation '
                             '(default: treeshap)')
    parser.add_argument('--n-core', type=int, default=DEFAULT_N_CORE,
                        help=f'Number of core reps (default: {DEFAULT_N_CORE})')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Importance method: {args.importance_method}")
    print(f"N core: {args.n_core}")

    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  importance_method=args.importance_method,
                  n_core=args.n_core)
    save_experiment_results(results, args.output, 'f06_augmentation_test',
                            args.dataset)
