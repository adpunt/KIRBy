#!/usr/bin/env python3
"""F04: Two-phase search validation — does two-phase match full grid?

Paper: Framework
Conclusion: Two-phase (coarse grid + step_size refinement) recovers full-grid
quality with ~75% fewer evaluations.
Dependencies: None (standalone).

Compares:
  - Full grid: allocations x budgets x step_sizes x models (manual per-fold CV)
  - Two-phase: find_best_hybrid(two_phase=True) with same allocations/budgets/models

Both paths use identical model sets (from DEFAULT_MODELS) and domain-aware
metrics, fixing the old mismatch where the full grid and two-phase used
different model catalogs.

Usage:
    python f04_two_phase_search.py --dataset logd --reps all --mode full
    python f04_two_phase_search.py --dataset flab_thermo_tresanco --reps all --mode full
    python f04_two_phase_search.py --dataset tcga_luad_vs_lusc --reps all --mode full
    python f04_two_phase_search.py --dataset buchwald_hartwig --reps all --mode full

    # Custom importance method (default: treeshap):
    python f04_two_phase_search.py --dataset logd --importance-method lightgbm_gain
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split,
    create_hybrid, apply_feature_selection,
    parse_rep_args, FRAMEWORK_TASKS,
)
from kirby.find_best_hybrid import find_best_hybrid, _score_to_comparable


ALLOCATIONS = ['greedy', 'softmax_T5', 'dominant_greedy']
BUDGETS = [50, 100, 200, 500]
STEP_SIZES = [1, 2, 5, 10, 25]
REFINEMENT_STEP_SIZES = [1, 2, 5, 25]  # exclude 10 (Phase 1 default)


def _run_full_grid(data, config, importance_method, metric_name, metric_fn,
                   lower_is_better):
    """Full grid: all allocs x budgets x step_sizes x models."""
    n_samples = len(data['labels'])
    task_type = data.get('task_type', 'regression')
    models = get_model_factories(config['models'], task_type=task_type,
                                 n_samples=n_samples)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    sample_X = next(iter(reps.values()))

    total = len(ALLOCATIONS) * len(BUDGETS) * len(STEP_SIZES) * len(models)
    print(f"\n  Full grid: {len(ALLOCATIONS)} allocs x {len(BUDGETS)} budgets x "
          f"{len(STEP_SIZES)} step_sizes x {len(models)} models = {total} evaluations")

    results = []
    idx = 0
    t0_total = time.time()

    for alloc in ALLOCATIONS:
        for budget in BUDGETS:
            for step_size in STEP_SIZES:
                for model_name, model_factory in models.items():
                    idx += 1
                    fold_scores = []
                    last_n_features = 0
                    last_allocation = {}
                    t0 = time.time()

                    for train_idx, val_idx in cv_split(groups, n_folds,
                                                       sample_X, labels,
                                                       mode=mode):
                        train_reps = {k: v[train_idx] for k, v in reps.items()}
                        val_reps = {k: v[val_idx] for k, v in reps.items()}
                        y_train, y_val = labels[train_idx], labels[val_idx]

                        try:
                            X_train_h, feature_info = create_hybrid(
                                train_reps, y_train,
                                allocation_method=alloc,
                                budget=budget,
                                step_size=step_size,
                                importance_method=importance_method,
                            )
                            X_val_h = apply_feature_selection(val_reps,
                                                              feature_info)
                        except Exception:
                            continue

                        scaler = StandardScaler()
                        X_train_s = scaler.fit_transform(X_train_h)
                        X_val_s = scaler.transform(X_val_h)
                        last_n_features = X_train_h.shape[1]
                        last_allocation = feature_info.get('allocation', {})

                        try:
                            model = model_factory()
                            model.fit(X_train_s, y_train)
                            fold_scores.append(
                                metric_fn(y_val, model.predict(X_val_s))
                            )
                        except Exception:
                            continue

                    elapsed = time.time() - t0

                    if not fold_scores:
                        score_mean, score_std = float('nan'), float('nan')
                        n_features = 0
                    else:
                        score_mean = float(np.mean(fold_scores))
                        score_std = float(np.std(fold_scores))
                        n_features = last_n_features

                    entry = {
                        'allocation_method': alloc,
                        'budget': budget,
                        'step_size': step_size,
                        'model': model_name,
                        'importance_method': importance_method,
                        'score_mean': score_mean,
                        'score_std': score_std,
                        'fold_scores': fold_scores,
                        'n_features': n_features,
                        'allocation_dict': {k: int(v)
                                            for k, v in last_allocation.items()},
                        'elapsed_s': round(elapsed, 2),
                    }
                    results.append(entry)

                    if idx % 20 == 0 or idx == total or idx <= 3:
                        print(f"  [{idx}/{total}] {alloc}/b={budget}/"
                              f"step={step_size}/{model_name}: "
                              f"{metric_name}={score_mean:.4f} ({elapsed:.1f}s)",
                              flush=True)

    total_time = time.time() - t0_total
    return results, total_time, list(models.keys())


def _run_two_phase(data, config, importance_method, metric_name):
    """Two-phase: find_best_hybrid with two_phase=True."""
    n_samples = len(data['labels'])
    task_type = data.get('task_type', 'regression')
    models = get_model_factories(config['models'], task_type=task_type,
                                 n_samples=n_samples)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    # Build model instances dict for find_best_hybrid
    model_instances = {name: factory() for name, factory in models.items()}

    t0 = time.time()
    search = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        models=model_instances,
        metric=metric_name,
        speed='fast',
        allocation_methods=ALLOCATIONS,
        budgets=BUDGETS,
        importance_methods=[importance_method],
        two_phase=True,
        refinement_top_n=5,
        refinement_step_sizes=REFINEMENT_STEP_SIZES,
        verbose=True,
        cache=False,
    )
    elapsed = time.time() - t0

    return search, elapsed


def run(data, config, output_dir, dataset_name, importance_method='treeshap',
        **kwargs):
    """Run two-phase validation experiment."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F04: TWO-PHASE SEARCH VALIDATION")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_samples = len(data['labels'])
    domain = data.get('domain', 'molecular')

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  Importance: {importance_method}")

    # --- Full grid ---
    print("\n--- Phase A: Full Grid ---")
    full_results, full_time, model_names = _run_full_grid(
        data, config, importance_method, metric_name, metric_fn,
        lower_is_better,
    )
    valid_full = [r for r in full_results if not np.isnan(r['score_mean'])]

    if not valid_full:
        print("  WARNING: No valid full-grid results.")
        return make_structured_output(
            experiment_id='f04', dataset=dataset_name, data=data,
            config={}, results={'error': 'no valid full-grid results'},
            primary_metric=metric_name, lower_is_better=lower_is_better,
        )

    # Find best using comparable scores (handles lower-is-better correctly)
    full_comparables = [_score_to_comparable(r['score_mean'], lower_is_better)
                        for r in valid_full]
    full_best = valid_full[int(np.argmax(full_comparables))]

    print(f"\n  Full grid best: {full_best['allocation_method']}/b={full_best['budget']}/"
          f"step={full_best['step_size']}/{full_best['model']} "
          f"-> {metric_name}={full_best['score_mean']:.4f} "
          f"({len(full_results)} evals, {full_time:.1f}s)")

    # --- Two-phase ---
    print("\n--- Phase B: Two-Phase Search ---")
    search, two_phase_time = _run_two_phase(data, config, importance_method,
                                             metric_name)
    n_two_phase = len(search.all_results)

    two_phase_best = search.best_overall
    print(f"\n  Two-phase best: {two_phase_best['allocation_method']}/"
          f"b={two_phase_best['budget']}/"
          f"step={two_phase_best.get('step_size', 10)}/{two_phase_best['model']} "
          f"-> {metric_name}={two_phase_best['score_mean']:.4f} "
          f"({n_two_phase} evals, {two_phase_time:.1f}s)")

    # --- Comparison ---
    print("\n--- Comparison ---")
    score_diff = two_phase_best['score_mean'] - full_best['score_mean']
    eval_reduction = 1.0 - n_two_phase / len(full_results)
    time_ratio = two_phase_time / full_time if full_time > 0 else None

    # Regret in comparable terms
    full_best_comp = _score_to_comparable(full_best['score_mean'], lower_is_better)
    tp_best_comp = _score_to_comparable(two_phase_best['score_mean'], lower_is_better)
    regret = full_best_comp - tp_best_comp

    # Rank correlation: for (model, alloc, budget) keys in both, compare rankings
    full_config_scores = {}
    for r in valid_full:
        key = (r['model'], r['allocation_method'], r['budget'])
        comp = _score_to_comparable(r['score_mean'], lower_is_better)
        if key not in full_config_scores or comp > full_config_scores[key]:
            full_config_scores[key] = comp

    two_phase_config_scores = {}
    for r in search.all_results:
        key = (r['model'], r['allocation_method'], r['budget'])
        comp = _score_to_comparable(r['score_mean'], lower_is_better)
        if key not in two_phase_config_scores or comp > two_phase_config_scores[key]:
            two_phase_config_scores[key] = comp

    shared_keys = sorted(set(full_config_scores) & set(two_phase_config_scores))
    if len(shared_keys) >= 3:
        full_scores = [full_config_scores[k] for k in shared_keys]
        tp_scores = [two_phase_config_scores[k] for k in shared_keys]
        rank_corr, rank_p = spearmanr(full_scores, tp_scores)
    else:
        rank_corr, rank_p = float('nan'), float('nan')

    print(f"  Score diff (two_phase - full): {score_diff:+.4f}")
    print(f"  Regret (comparable): {regret:.4f}")
    print(f"  Eval reduction: {eval_reduction:.1%} ({n_two_phase} vs {len(full_results)})")
    print(f"  Rank correlation: {rank_corr:.3f} (p={rank_p:.4f})"
          if not np.isnan(rank_corr) else "  Rank correlation: N/A (too few shared configs)")
    print(f"  Time: full={full_time:.1f}s, two_phase={two_phase_time:.1f}s"
          + (f" ({time_ratio:.1%})" if time_ratio else ""))

    # --- Per-model comparison ---
    print(f"\n  PER-MODEL COMPARISON ({metric_name}):")
    per_model = {}
    for model_name in model_names:
        # Full grid best for this model
        model_full = [r for r in valid_full if r['model'] == model_name]
        if not model_full:
            continue
        mf_comps = [_score_to_comparable(r['score_mean'], lower_is_better)
                    for r in model_full]
        mf_best = model_full[int(np.argmax(mf_comps))]

        # Two-phase best for this model
        model_tp = [r for r in search.all_results if r['model'] == model_name
                    and not np.isnan(r['score_mean'])]
        if model_tp:
            mt_comps = [_score_to_comparable(r['score_mean'], lower_is_better)
                        for r in model_tp]
            mt_best = model_tp[int(np.argmax(mt_comps))]
            delta = mt_best['score_mean'] - mf_best['score_mean']
        else:
            mt_best = None
            delta = float('nan')

        per_model[model_name] = {
            'full_best': mf_best['score_mean'],
            'full_config': f"{mf_best['allocation_method']}/b={mf_best['budget']}/step={mf_best['step_size']}",
            'two_phase_best': mt_best['score_mean'] if mt_best else float('nan'),
            'two_phase_config': (
                f"{mt_best['allocation_method']}/b={mt_best['budget']}/step={mt_best.get('step_size', '?')}"
                if mt_best else 'N/A'
            ),
            'delta': delta,
        }

        print(f"    {model_name:>10}: full={mf_best['score_mean']:.4f} "
              f"tp={mt_best['score_mean']:.4f if mt_best else 'N/A':>7} "
              f"delta={delta:+.4f}" if mt_best else
              f"    {model_name:>10}: full={mf_best['score_mean']:.4f} tp=N/A")

    # --- Verdict ---
    if regret < 0.01:
        verdict = "PASS: two-phase recovers full-grid quality (regret < 0.01)"
    elif regret < 0.05:
        verdict = "MARGINAL: two-phase close but regret 0.01-0.05"
    else:
        verdict = f"FAIL: two-phase regret = {regret:.4f} (> 0.05)"
    print(f"\n  VERDICT: {verdict}")

    # --- Structured output ---
    summary = {
        'full_grid_best': full_best,
        'two_phase_best': convert_numpy(two_phase_best),
        'score_diff': round(score_diff, 6),
        'regret': round(regret, 6),
        'eval_reduction_pct': round(eval_reduction * 100, 1),
        'rank_correlation': round(rank_corr, 4) if not np.isnan(rank_corr) else None,
        'rank_p_value': round(rank_p, 6) if not np.isnan(rank_p) else None,
        'time_ratio': round(time_ratio, 3) if time_ratio else None,
        'per_model': per_model,
        'verdict': verdict,
    }

    results = {
        'full_grid': {
            'best': full_best,
            'n_evals': len(full_results),
            'elapsed_s': round(full_time, 2),
            'all_results': full_results,
        },
        'two_phase': {
            'best': convert_numpy(two_phase_best),
            'n_evals': n_two_phase,
            'elapsed_s': round(two_phase_time, 2),
        },
        'summary': summary,
    }

    return make_structured_output(
        experiment_id='f04',
        dataset=dataset_name,
        data=data,
        config={
            'n_folds': config['n_splits'],
            'models': model_names,
            'allocations': ALLOCATIONS,
            'budgets': BUDGETS,
            'step_sizes': STEP_SIZES,
            'refinement_step_sizes': REFINEMENT_STEP_SIZES,
            'importance_method': importance_method,
        },
        results=results,
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


if __name__ == '__main__':
    parser = create_experiment_parser('F04: Two-Phase Search Validation',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--importance-method', type=str, default='treeshap',
                        help='Importance method for allocation (default: treeshap)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {config['models']}")
    print(f"Importance: {args.importance_method}")

    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  importance_method=args.importance_method)
    save_experiment_results(results, args.output, 'f04_two_phase_search',
                            args.dataset)
