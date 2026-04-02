#!/usr/bin/env python3
"""F07: Internal evaluator comparison — RF vs XGBoost vs Ridge inside greedy loops.

Paper: Framework
Question: The greedy allocation loop uses `_evaluate_allocation` (RF by default)
    to score trial allocations. Does changing the internal evaluator to XGBoost
    or Ridge change which allocations greedy picks? Does it change downstream
    hybrid performance?

For each task:
    1. Run greedy allocation with each internal evaluator (RF, XGBoost, Ridge).
    2. Compare: resulting allocation dicts, wall time, final hybrid scores
       evaluated with multiple downstream models.

The monkey-patch replaces `hybrid._evaluate_allocation` for the duration of
each condition. This affects all calls inside `allocate_greedy`,
`allocate_dominant_greedy`, `allocate_greedy_feature`, and augmentation
evaluation.

Usage:
    python f07_internal_evaluator.py --dataset logd --reps all --mode full
    python f07_internal_evaluator.py --dataset flab_thermo_tresanco --reps all --mode full
    python f07_internal_evaluator.py --dataset tcga_luad_vs_lusc --reps all --mode full
    python f07_internal_evaluator.py --dataset buchwald_hartwig --reps all --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_model_factories, get_domain_metric, cv_split,
    parse_rep_args, FRAMEWORK_TASKS,
    INTERNAL_EVALUATORS as SHARED_INTERNAL_EVALUATORS,
    DOWNSTREAM_MODELS, extract_top3_base_reps,
)
from kirby.hybrid import create_hybrid, apply_feature_selection
import kirby.hybrid as hybrid_module


# Internal evaluator models to test (f07-specific list for monkey-patching)
F07_INTERNAL_EVALUATORS = ['rf', 'xgboost', 'ridge']

# Greedy-based allocation methods (the ones that call _evaluate_allocation)
GREEDY_ALLOCATIONS = ['greedy', 'dominant_greedy']

# Downstream models for final hybrid evaluation
F07_DOWNSTREAM_MODELS = DOWNSTREAM_MODELS

# Budgets to test
BUDGETS = [100, 200]

# Greedy search settings
STEP_SIZE = 10
PATIENCE = 3


def _make_evaluator(evaluator_name, task_type):
    """Return a replacement for hybrid._evaluate_allocation using the given model.

    The returned function has the same signature:
        (X_train, X_val, y_train, y_val) -> float (higher is better)
    """
    is_clf = task_type == 'classification'

    def _evaluate_allocation(X_train, X_val, y_train, y_val):
        if X_train.shape[1] == 0:
            return -np.inf

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if evaluator_name == 'rf':
            if is_clf:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1)

        elif evaluator_name == 'xgboost':
            if is_clf:
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=100, max_depth=6, random_state=42,
                    n_jobs=-1, verbosity=0)
            else:
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=100, max_depth=6, random_state=42,
                    n_jobs=-1, verbosity=0)

        elif evaluator_name == 'ridge':
            if is_clf:
                from sklearn.linear_model import RidgeClassifier
                model = RidgeClassifier(alpha=1.0)
            else:
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)

        else:
            raise ValueError(f"Unknown evaluator: {evaluator_name}")

        model.fit(X_train_scaled, y_train)

        if is_clf:
            return model.score(X_val_scaled, y_val)
        else:
            y_pred = model.predict(X_val_scaled)
            return r2_score(y_val, y_pred)

    return _evaluate_allocation


def _extract_allocation(feature_info):
    """Extract {rep_name: n_features} from create_hybrid feature_info."""
    alloc = {}
    for rep_name, info in feature_info.items():
        if isinstance(info, dict) and 'selected_indices' in info:
            alloc[rep_name] = len(info['selected_indices'])
    return alloc


def _allocation_similarity(alloc_a, alloc_b):
    """Compute similarity between two allocation dicts.

    Returns:
        dict with cosine similarity and L1 distance between allocation vectors.
    """
    all_reps = sorted(set(alloc_a.keys()) | set(alloc_b.keys()))
    vec_a = np.array([alloc_a.get(r, 0) for r in all_reps], dtype=float)
    vec_b = np.array([alloc_b.get(r, 0) for r in all_reps], dtype=float)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a > 0 and norm_b > 0:
        cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    else:
        cosine = 0.0

    return {
        'cosine_similarity': cosine,
        'l1_distance': float(np.sum(np.abs(vec_a - vec_b))),
        'total_features_a': int(np.sum(vec_a)),
        'total_features_b': int(np.sum(vec_b)),
    }


def run(data, config, output_dir, dataset_name, importance_method='treeshap',
        **kwargs):
    """Run internal evaluator comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F07: INTERNAL EVALUATOR COMPARISON")
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
    print(f"  Reps: {len(reps)} ({sum(v.shape[1] for v in reps.values())} "
          f"total features)")
    print(f"  Metric: {metric_name} "
          f"({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Importance method: {importance_method}")
    print(f"  Internal evaluators: {F07_INTERNAL_EVALUATORS}")
    print(f"  Allocation methods: {GREEDY_ALLOCATIONS}")
    print(f"  Budgets: {BUDGETS}")
    print(f"  Downstream models: {F07_DOWNSTREAM_MODELS}")
    print(f"  N folds: {n_folds}\n", flush=True)

    # Get downstream model factories
    factories = get_model_factories(F07_DOWNSTREAM_MODELS, task_type=task_type,
                                    n_samples=n_samples)
    print(f"  Available downstream models: {sorted(factories.keys())}\n",
          flush=True)

    # Save the original _evaluate_allocation to restore after each condition
    original_evaluate = hybrid_module._evaluate_allocation

    total_configs = (len(F07_INTERNAL_EVALUATORS) * len(GREEDY_ALLOCATIONS)
                     * len(BUDGETS))
    all_results = []
    config_idx = 0

    for evaluator_name in F07_INTERNAL_EVALUATORS:
        for alloc_method in GREEDY_ALLOCATIONS:
            for budget in BUDGETS:
                config_idx += 1
                label = (f"[{config_idx}/{total_configs}] "
                         f"evaluator={evaluator_name} | "
                         f"alloc={alloc_method} | budget={budget}")
                print(f"\n  --- {label} ---", flush=True)

                # Monkey-patch the internal evaluator
                hybrid_module._evaluate_allocation = _make_evaluator(
                    evaluator_name, task_type)

                # Per-fold CV: create_hybrid on train, evaluate on val
                fold_results = []
                t0 = time.time()

                try:
                    for fold_idx, (train_idx, val_idx) in enumerate(
                        cv_split(groups, n_folds,
                                 next(iter(reps.values())), labels,
                                 mode='full')
                    ):
                        train_reps = {k: v[train_idx]
                                      for k, v in reps.items()}
                        val_reps = {k: v[val_idx]
                                    for k, v in reps.items()}
                        y_train = labels[train_idx]
                        y_val = labels[val_idx]

                        # Run hybrid creation with the patched evaluator
                        try:
                            t_fold = time.time()
                            X_train, feature_info = create_hybrid(
                                train_reps, y_train,
                                allocation_method=alloc_method,
                                budget=budget,
                                importance_method=importance_method,
                                step_size=STEP_SIZE,
                                patience=PATIENCE,
                            )
                            X_val = apply_feature_selection(
                                val_reps, feature_info)
                            hybrid_time = time.time() - t_fold

                            allocation = _extract_allocation(feature_info)

                            # Evaluate with each downstream model
                            downstream_scores = {}
                            scaler = StandardScaler()
                            X_train_s = scaler.fit_transform(X_train)
                            X_val_s = scaler.transform(X_val)

                            for model_name, model_factory in sorted(
                                factories.items()
                            ):
                                model = model_factory()
                                model.fit(X_train_s, y_train)
                                score = metric_fn(
                                    y_val, model.predict(X_val_s))
                                downstream_scores[model_name] = float(score)

                            fold_results.append({
                                'fold': fold_idx,
                                'allocation': allocation,
                                'n_features': X_train.shape[1],
                                'hybrid_time': hybrid_time,
                                'downstream_scores': downstream_scores,
                            })
                        except Exception as e:
                            print(f"    Fold {fold_idx} FAILED: {e}")
                            fold_results.append({
                                'fold': fold_idx,
                                'error': str(e),
                            })
                finally:
                    # Always restore original evaluator
                    hybrid_module._evaluate_allocation = original_evaluate

                total_time = time.time() - t0

                # Aggregate fold results
                valid_folds = [f for f in fold_results if 'error' not in f]
                if valid_folds:
                    # Mean allocation across folds
                    all_reps_names = sorted(
                        set().union(*(f['allocation'].keys()
                                      for f in valid_folds)))
                    mean_alloc = {
                        r: float(np.mean([f['allocation'].get(r, 0)
                                          for f in valid_folds]))
                        for r in all_reps_names
                    }

                    # Mean downstream scores
                    mean_downstream = {}
                    for model_name in sorted(factories.keys()):
                        scores = [f['downstream_scores'].get(model_name)
                                  for f in valid_folds
                                  if model_name in f.get('downstream_scores',
                                                         {})]
                        if scores:
                            mean_downstream[model_name] = {
                                'mean': float(np.mean(scores)),
                                'std': float(np.std(scores)),
                            }

                    # Print summary
                    n_feats = np.mean([f['n_features'] for f in valid_folds])
                    print(f"    Features: {n_feats:.0f} | "
                          f"Time: {total_time:.1f}s | "
                          f"Folds OK: {len(valid_folds)}/{n_folds}")
                    print(f"    Allocation: {mean_alloc}")
                    for mn, ms in sorted(mean_downstream.items()):
                        print(f"      {mn}: {metric_name}="
                              f"{ms['mean']:.4f}±{ms['std']:.4f}")
                else:
                    mean_alloc = {}
                    mean_downstream = {}
                    print(f"    ALL FOLDS FAILED")

                result_entry = {
                    'internal_evaluator': evaluator_name,
                    'allocation_method': alloc_method,
                    'budget': budget,
                    'importance_method': importance_method,
                    'step_size': STEP_SIZE,
                    'patience': PATIENCE,
                    'wall_time': total_time,
                    'n_folds_ok': len(valid_folds),
                    'mean_allocation': mean_alloc,
                    'mean_downstream': mean_downstream,
                    'fold_results': fold_results,
                }
                all_results.append(result_entry)

    # ------------------------------------------------------------------
    # Summary: compare evaluators
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Group by (alloc_method, budget) and compare evaluators
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = (r['allocation_method'], r['budget'])
        grouped[key].append(r)

    for (alloc_method, budget), entries in sorted(grouped.items()):
        print(f"\n  {alloc_method} | budget={budget}:")
        print(f"    {'Evaluator':<12} {'Time':>8}  ", end="")
        # Print header for each downstream model
        for mn in sorted(factories.keys()):
            print(f"  {mn:>12}", end="")
        print()
        print("    " + "-" * (12 + 8 + 2 + 14 * len(factories)))

        for entry in entries:
            ev = entry['internal_evaluator']
            t = entry['wall_time']
            print(f"    {ev:<12} {t:>7.1f}s  ", end="")
            for mn in sorted(factories.keys()):
                ms = entry['mean_downstream'].get(mn)
                if ms:
                    print(f"  {ms['mean']:>12.4f}", end="")
                else:
                    print(f"  {'N/A':>12}", end="")
            print()

    # Allocation similarity across evaluators
    print(f"\n  ALLOCATION SIMILARITY (cosine) across evaluators:")
    for (alloc_method, budget), entries in sorted(grouped.items()):
        evaluator_allocs = {e['internal_evaluator']: e['mean_allocation']
                           for e in entries}
        ev_names = sorted(evaluator_allocs.keys())
        print(f"\n    {alloc_method} | budget={budget}:")
        for i, ev_a in enumerate(ev_names):
            for ev_b in ev_names[i + 1:]:
                sim = _allocation_similarity(
                    evaluator_allocs[ev_a], evaluator_allocs[ev_b])
                print(f"      {ev_a} vs {ev_b}: "
                      f"cosine={sim['cosine_similarity']:.3f}, "
                      f"L1={sim['l1_distance']:.0f}")

    # Timing comparison
    print(f"\n  TIMING (mean across alloc methods and budgets):")
    timing_by_ev = defaultdict(list)
    for r in all_results:
        timing_by_ev[r['internal_evaluator']].append(r['wall_time'])
    for ev in F07_INTERNAL_EVALUATORS:
        times = timing_by_ev.get(ev, [])
        if times:
            print(f"    {ev:<12} mean={np.mean(times):.1f}s  "
                  f"total={np.sum(times):.1f}s")

    # ------------------------------------------------------------------
    # Build structured output
    # ------------------------------------------------------------------
    return make_structured_output(
        experiment_id='f07',
        dataset=dataset_name,
        data=data,
        config={
            'internal_evaluators': F07_INTERNAL_EVALUATORS,
            'allocation_methods': GREEDY_ALLOCATIONS,
            'budgets': BUDGETS,
            'downstream_models': F07_DOWNSTREAM_MODELS,
            'importance_method': importance_method,
            'step_size': STEP_SIZE,
            'patience': PATIENCE,
            'n_folds': n_folds,
        },
        results={
            'conditions': all_results,
        },
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser('F07: Internal Evaluator Comparison',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--importance-method', type=str, default='treeshap',
                        help='Importance method for hybrid allocation '
                             '(default: treeshap)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Importance method: {args.importance_method}")

    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  importance_method=args.importance_method)
    save_experiment_results(results, args.output, 'f07_internal_evaluator',
                            args.dataset)
