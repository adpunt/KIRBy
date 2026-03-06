#!/usr/bin/env python3
"""C14: Two-phase search validation — does two-phase match full grid?

Paper: Complexity
Conclusion: Two-phase (coarse grid + step_size refinement) recovers full-grid
quality with ~75% fewer evaluations.
Dependencies: None (standalone).

TODO: This is a one-off validation experiment, not a recurring benchmark.
      Its purpose is to empirically confirm that the two-phase search shortcut
      (now built into find_best_hybrid via two_phase=True) recovers the same
      best configuration as an exhaustive grid over step_sizes. Run once per
      domain (molecular, antibody, TCGA) to build confidence, then archive.

Compares:
  - Full grid: 3 allocs x 4 budgets x 5 step_sizes x 6 models = 360 evals
  - Two-phase: Phase 1 (3 x 4 x 6 = 72, step=10) + Phase 2 (top-5 x 4 = 20) = 92 evals
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split,
    create_hybrid, apply_feature_selection,
)
from kirby.find_best_hybrid import find_best_hybrid


ALLOCATIONS = ['greedy', 'softmax_T5', 'dominant_greedy']
BUDGETS = [50, 100, 200, 500]
STEP_SIZES = [1, 2, 5, 10, 25]
REFINEMENT_STEP_SIZES = [1, 2, 5, 25]  # exclude 10 (Phase 1 default)


def _run_full_grid(data, config):
    """Full grid: all allocs x budgets x step_sizes x models."""
    models = get_model_factories(config['models'])
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
                    t0 = time.time()

                    for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                                       labels, mode=mode):
                        train_reps = {k: v[train_idx] for k, v in reps.items()}
                        val_reps = {k: v[val_idx] for k, v in reps.items()}
                        y_train, y_val = labels[train_idx], labels[val_idx]

                        try:
                            X_train_h, feature_info = create_hybrid(
                                train_reps, y_train,
                                allocation_method=alloc,
                                budget=budget,
                                step_size=step_size,
                                importance_method='lightgbm_gain',
                            )
                            X_val_h = apply_feature_selection(val_reps, feature_info)
                        except Exception:
                            continue

                        scaler = StandardScaler()
                        X_train_s = scaler.fit_transform(X_train_h)
                        X_val_s = scaler.transform(X_val_h)
                        last_n_features = X_train_h.shape[1]

                        model = model_factory()
                        model.fit(X_train_s, y_train)
                        fold_scores.append(r2_score(y_val, model.predict(X_val_s)))

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
                        'score_mean': score_mean,
                        'score_std': score_std,
                        'n_features': n_features,
                        'elapsed_s': round(elapsed, 2),
                    }
                    results.append(entry)

                    if idx % 20 == 0 or idx == total:
                        print(f"  [{idx}/{total}] {alloc}/b={budget}/step={step_size}/"
                              f"{model_name}: R2={score_mean:.4f} ({elapsed:.1f}s)")

    total_time = time.time() - t0_total
    return results, total_time


def _run_two_phase(data, config):
    """Two-phase: find_best_hybrid with two_phase=True."""
    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    t0 = time.time()
    search = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        speed='fast',
        allocation_methods=ALLOCATIONS,
        budgets=BUDGETS,
        two_phase=True,
        refinement_top_n=5,
        refinement_step_sizes=REFINEMENT_STEP_SIZES,
        verbose=True,
        cache=False,
    )
    elapsed = time.time() - t0

    return search, elapsed


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run two-phase validation experiment."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C14: TWO-PHASE SEARCH VALIDATION")
    print("=" * 70)

    # --- Full grid ---
    print("\n--- Phase A: Full Grid ---")
    full_results, full_time = _run_full_grid(data, config)
    valid_full = [r for r in full_results if not np.isnan(r['score_mean'])]

    if not valid_full:
        print("  WARNING: No valid full-grid results.")
        return {'error': 'no valid full-grid results'}

    full_best = max(valid_full, key=lambda r: r['score_mean'])
    print(f"\n  Full grid best: {full_best['allocation_method']}/b={full_best['budget']}/"
          f"step={full_best['step_size']}/{full_best['model']} "
          f"-> R2={full_best['score_mean']:.4f} ({len(full_results)} evals, {full_time:.1f}s)")

    # --- Two-phase ---
    print("\n--- Phase B: Two-Phase Search ---")
    search, two_phase_time = _run_two_phase(data, config)
    n_two_phase = len(search.all_results)

    two_phase_best = search.best_overall
    print(f"\n  Two-phase best: {two_phase_best['allocation_method']}/b={two_phase_best['budget']}/"
          f"step={two_phase_best.get('step_size', 10)}/{two_phase_best['model']} "
          f"-> score={two_phase_best['score_mean']:.4f} "
          f"({n_two_phase} evals, {two_phase_time:.1f}s)")

    # --- Comparison ---
    print("\n--- Comparison ---")
    score_diff = two_phase_best['score_mean'] - full_best['score_mean']
    eval_reduction = 1.0 - n_two_phase / len(full_results)

    # Rank correlation: for configs evaluated in both, compare rankings
    # Build (model, alloc, budget) -> best_score for full grid
    full_config_scores = {}
    for r in valid_full:
        key = (r['model'], r['allocation_method'], r['budget'])
        if key not in full_config_scores or r['score_mean'] > full_config_scores[key]:
            full_config_scores[key] = r['score_mean']

    two_phase_config_scores = {}
    for r in search.all_results:
        key = (r['model'], r['allocation_method'], r['budget'])
        if key not in two_phase_config_scores or r['score_mean'] > two_phase_config_scores[key]:
            two_phase_config_scores[key] = r['score_mean']

    # Compute rank correlation on shared configs
    shared_keys = sorted(set(full_config_scores) & set(two_phase_config_scores))
    if len(shared_keys) >= 3:
        full_scores = [full_config_scores[k] for k in shared_keys]
        tp_scores = [two_phase_config_scores[k] for k in shared_keys]
        rank_corr, rank_p = spearmanr(full_scores, tp_scores)
    else:
        rank_corr, rank_p = float('nan'), float('nan')

    print(f"  Score diff (two_phase - full): {score_diff:+.4f}")
    print(f"  Eval reduction: {eval_reduction:.1%} ({n_two_phase} vs {len(full_results)})")
    print(f"  Rank correlation: {rank_corr:.3f} (p={rank_p:.4f})")
    print(f"  Time: full={full_time:.1f}s, two_phase={two_phase_time:.1f}s "
          f"({two_phase_time/full_time:.1%})")

    return {
        'full_grid': {
            'best': convert_numpy(full_best),
            'n_evals': len(full_results),
            'elapsed_s': round(full_time, 2),
            'all_results': convert_numpy(full_results),
        },
        'two_phase': {
            'best': convert_numpy(two_phase_best),
            'n_evals': n_two_phase,
            'elapsed_s': round(two_phase_time, 2),
        },
        'comparison': {
            'score_diff': round(score_diff, 6),
            'eval_reduction_pct': round(eval_reduction * 100, 1),
            'rank_correlation': round(rank_corr, 4) if not np.isnan(rank_corr) else None,
            'rank_p_value': round(rank_p, 6) if not np.isnan(rank_p) else None,
            'time_ratio': round(two_phase_time / full_time, 3) if full_time > 0 else None,
        },
    }


if __name__ == '__main__':
    parser = create_experiment_parser('C14: Two-Phase Search Validation',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c14_two_phase_search', args.dataset)
