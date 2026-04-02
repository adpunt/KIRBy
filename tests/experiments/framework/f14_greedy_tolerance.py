#!/usr/bin/env python3
"""F14: Greedy Allocation Size Analysis.

Paper: Framework
Question: Does greedy allocation fill the budget, or does it stop too early
    and produce undersized hybrids? Should we keep greedy at all, or replace
    it with greedy_fill / greedy_forced / non-greedy allocation methods?

Dependencies: None (standalone).

Design: For each dataset × budget, directly call create_hybrid with each
    greedy variant and measure:
    - n_features_selected vs budget (fill ratio)
    - n_reps_contributing (how many reps got features)
    - validation score
    Then compare greedy variants:
    - greedy (patience=20, may stop early)
    - greedy_fill_s25 (greedy + backfill to budget, used in fast tier)
    - greedy_forced_s25 (no patience, fills budget)
    - greedy + tolerance (patience with tolerance)
    - Non-greedy baselines: entropy_1.0, top_3 (always fill exactly)

Key metric: fill ratio = n_features_selected / budget.
If greedy consistently has fill_ratio << 1.0, it's producing undersized reps.

Usage:
    python tests/experiments/framework/f14_greedy_tolerance.py --dataset herg_ki
    python tests/experiments/framework/f14_greedy_tolerance.py --dataset tcga_brca_pam50
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_domain_metric, convert_numpy,
    get_model_factories,
    INTERNAL_EVALUATORS, DOWNSTREAM_MODELS, extract_top3_base_reps,
)


# Allocation variants to compare
ALLOC_CONFIGS = {
    # Greedy variants (the ones under scrutiny)
    'greedy_patience20':  {'method': 'greedy', 'step_size': 25, 'patience': 20,
                           'greedy_tolerance': 0.0},
    'greedy_tol_0.005':   {'method': 'greedy', 'step_size': 25, 'patience': 20,
                           'greedy_tolerance': 0.005},
    'greedy_tol_0.01':    {'method': 'greedy', 'step_size': 25, 'patience': 20,
                           'greedy_tolerance': 0.01},
    'greedy_tol_0.02':    {'method': 'greedy', 'step_size': 25, 'patience': 20,
                           'greedy_tolerance': 0.02},
    'greedy_fill_s25':    {'method': 'greedy_fill_s25'},
    'greedy_forced_s25':  {'method': 'greedy_forced_s25'},
    # Non-greedy baselines (always produce exactly budget features)
    'entropy_1.0':        {'method': 'entropy_1.0'},
    'top_3':              {'method': 'top_3'},
}

BUDGETS = [500, 1000, 2000, 4000]


def _count_features(feature_info):
    """Count actual features selected from feature_info dict."""
    n = 0
    reps_used = 0
    for k, v in feature_info.items():
        if isinstance(v, dict) and 'selected_indices' in v:
            count = len(v['selected_indices'])
            n += count
            if count > 0:
                reps_used += 1
    return n, reps_used


def run_allocation_analysis(dataset_name: str, budgets=None):
    """Compare allocation strategies on one dataset."""
    from kirby.hybrid import create_hybrid, apply_feature_selection

    data = load_data(dataset_name, rep_preset='all')
    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    n_samples = len(labels)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    if budgets is None:
        budgets = BUDGETS

    # Available feature budget (total features across all reps)
    total_available = sum(X.shape[1] for X in reps.values())

    print(f"\n{'=' * 80}")
    print(f"F14: GREEDY ALLOCATION SIZE ANALYSIS — {dataset_name}")
    print(f"{'=' * 80}")
    print(f"  Metric: {metric_name}")
    print(f"  Reps: {len(reps)} ({total_available} total features)")
    for rn, X in sorted(reps.items(), key=lambda x: -x[1].shape[1]):
        print(f"    {rn}: {X.shape[1]}D")
    print(f"  Samples: {n_samples}")
    print(f"  Budgets: {budgets}")

    # Single train/val split for speed
    if groups is not None:
        cv = GroupKFold(n_splits=5)
        train_idx, val_idx = next(cv.split(labels, labels, groups))
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(cv.split(labels))

    y_train, y_val = labels[train_idx], labels[val_idx]
    print(f"  Train/val split: {len(train_idx)}/{len(val_idx)}")

    # Get a model for downstream evaluation
    models = get_model_factories(['rf'], task_type=data.get('task_type', 'regression'),
                                 n_samples=len(train_idx))
    model_name, model_factory = next(iter(models.items()))

    results = []

    for budget in budgets:
        print(f"\n  Budget = {budget}")
        print(f"  {'Allocation':<25s}  {'Features':>8s}  {'Fill%':>6s}  "
              f"{'Reps':>5s}  {'Score':>10s}  {'Time':>6s}")
        print(f"  {'-'*25}  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*10}  {'-'*6}")

        for alloc_name, cfg in ALLOC_CONFIGS.items():
            method = cfg['method']
            extra_kwargs = {k: v for k, v in cfg.items() if k != 'method'}

            t0 = time.time()
            try:
                X_train, feature_info = create_hybrid(
                    {k: v[train_idx] for k, v in reps.items()},
                    y_train,
                    allocation_method=method,
                    budget=budget,
                    importance_method='treeshap',
                    internal_evaluator='rf',
                    **extra_kwargs,
                )
                X_val = apply_feature_selection(
                    {k: v[val_idx] for k, v in reps.items()},
                    feature_info,
                )

                n_feat, n_reps = _count_features(feature_info)
                fill_pct = 100.0 * n_feat / budget if budget > 0 else 0

                # Evaluate with RF
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train)
                Xva = scaler.transform(X_val)
                m = model_factory()
                m.fit(Xtr, y_train)
                score = float(metric_fn(y_val, m.predict(Xva)))
                elapsed = time.time() - t0

                tag = ' ***' if fill_pct < 80 else ''
                print(f"  {alloc_name:<25s}  {n_feat:>8d}  {fill_pct:>5.1f}%  "
                      f"{n_reps:>5d}  {score:>10.4f}  {elapsed:>5.1f}s{tag}")

                # Per-rep breakdown
                per_rep = {}
                for rn, info in feature_info.items():
                    if isinstance(info, dict) and 'selected_indices' in info:
                        per_rep[rn] = len(info['selected_indices'])

                results.append({
                    'budget': budget,
                    'allocation': alloc_name,
                    'method': method,
                    'n_features': n_feat,
                    'fill_ratio': n_feat / budget if budget > 0 else 0,
                    'n_reps': n_reps,
                    'score': score,
                    'elapsed': elapsed,
                    'per_rep': per_rep,
                    **{k: v for k, v in extra_kwargs.items()
                       if k in ('greedy_tolerance', 'patience', 'step_size')},
                })

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  {alloc_name:<25s}  {'FAIL':>8s}  {'':>6s}  "
                      f"{'':>5s}  {'':>10s}  {elapsed:>5.1f}s  {e}")
                results.append({
                    'budget': budget,
                    'allocation': alloc_name,
                    'error': str(e),
                })

    # ── Summary: fill ratio table ──
    print(f"\n{'=' * 80}")
    print("FILL RATIO SUMMARY (features selected / budget)")
    print(f"{'=' * 80}")
    alloc_names = list(ALLOC_CONFIGS.keys())
    header = f"  {'Budget':>7s}  " + "  ".join(f"{a:>15s}" for a in alloc_names)
    print(header)
    print(f"  {'-'*7}  " + "  ".join(f"{'-'*15}" for _ in alloc_names))

    for budget in budgets:
        row = f"  {budget:>7d}  "
        for alloc_name in alloc_names:
            match = [r for r in results
                     if r.get('budget') == budget and r.get('allocation') == alloc_name]
            if match and 'fill_ratio' in match[0]:
                ratio = match[0]['fill_ratio']
                row += f"  {ratio:>14.1%}"
            else:
                row += f"  {'FAIL':>15s}"
        print(row)

    # Score comparison
    print(f"\n{'=' * 80}")
    print(f"SCORE SUMMARY ({metric_name}, RF)")
    print(f"{'=' * 80}")
    print(header)
    print(f"  {'-'*7}  " + "  ".join(f"{'-'*15}" for _ in alloc_names))

    for budget in budgets:
        row = f"  {budget:>7d}  "
        for alloc_name in alloc_names:
            match = [r for r in results
                     if r.get('budget') == budget and r.get('allocation') == alloc_name]
            if match and 'score' in match[0]:
                row += f"  {match[0]['score']:>15.4f}"
            else:
                row += f"  {'FAIL':>15s}"
        print(row)

    return {
        'dataset': dataset_name,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_reps': len(reps),
        'total_features_available': total_available,
        'rep_dims': {k: v.shape[1] for k, v in reps.items()},
        'n_samples': n_samples,
        'budgets': budgets,
        'results': results,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='F14: Greedy Allocation Size Analysis — '
                    'does greedy fill the budget or stop too early?')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g. herg_ki, tcga_brca_pam50)')
    parser.add_argument('--budgets', type=str, default=None,
                        help='Comma-separated budgets (default: 500,1000,2000,4000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(',')] if args.budgets else None

    results = run_allocation_analysis(args.dataset, budgets=budgets)

    output_path = args.output or f'tests/results/framework/f14_greedy_tolerance_{args.dataset}.json'
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\nResults saved: {p}")
