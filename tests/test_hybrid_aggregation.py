#!/usr/bin/env python3
"""
Experiment: Aggregated importance for hybrid feature selection.

Tests whether aggregating importance scores from multiple non-correlated
methods (borda, avg, RRF) improves hybrid feature selection compared to
single-method selection.

Benchmark finding: single methods >> aggregated for kNN/similarity *weights*,
but borda_all was most consistent for *selection*. This experiment tests
that in the full hybrid pipeline with CV evaluation.

Approach:
  1. Compute importance per-rep from each fast method
  2. Aggregate using borda, simple_avg, RRF
  3. Select top-N features per rep based on aggregated ranking
  4. Evaluate with CV against single-method baselines

Usage:
    python tests/test_hybrid_aggregation.py logd --reps fast --budget 100
    python tests/test_hybrid_aggregation.py caco2 --budgets 50 100 200
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import warnings
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from kirby.hybrid import compute_feature_importance, create_hybrid, apply_feature_selection
from find_best_hybrid_molecular import load_dataset, get_representations, assign_scaffold_groups

# Import aggregation functions from benchmark
from benchmark_importance_aggregation import (
    normalize_scores,
    aggregate_simple_average,
    aggregate_borda,
    aggregate_rrf,
)

warnings.filterwarnings('ignore')


# =============================================================================
# CORE: AGGREGATED FEATURE SELECTION
# =============================================================================

FAST_METHODS = ['treeshap', 'xgboost_gain', 'lightgbm_gain', 'deeplift']
ALL_CLUSTER_REPS = ['treeshap', 'xgboost_gain', 'lightgbm_gain', 'deeplift',
                    'permutation', 'kernelshap']

AGGREGATION_STRATEGIES = {
    'borda_all': lambda results: aggregate_borda(results),
    'borda_fast4': lambda results: aggregate_borda(results, methods=FAST_METHODS),
    'avg_all': lambda results: aggregate_simple_average(results),
    'avg_fast4': lambda results: aggregate_simple_average(results, methods=FAST_METHODS),
    'rrf_all': lambda results: aggregate_rrf(results),
    'rrf_fast4': lambda results: aggregate_rrf(results, methods=FAST_METHODS),
}


def compute_all_importance(base_reps, labels, methods):
    """Compute importance scores from each method for all reps.

    Returns dict in the format expected by aggregation functions:
        {method: {'success': bool, 'scores': np.ndarray (concatenated across reps)}}
    Also returns per-rep importance dicts for manual selection.
    """
    all_results = {}
    per_rep_results = {}

    for method in methods:
        print(f"  Computing {method}...", end=' ', flush=True)
        try:
            imp = compute_feature_importance(base_reps, labels, method=method)
            # Concatenate across reps in sorted order for aggregation
            rep_names = sorted(base_reps.keys())
            concat_scores = np.concatenate([imp[r] for r in rep_names])
            all_results[method] = {'success': True, 'scores': concat_scores}
            per_rep_results[method] = {r: imp[r] for r in rep_names}
            print("done")
        except Exception as e:
            all_results[method] = {'success': False, 'scores': None}
            print(f"FAILED: {e}")

    return all_results, per_rep_results


def select_features_by_scores(base_reps, scores_concat, budget):
    """Select top features from each rep proportional to rep size, using aggregated scores.

    Allocates budget proportional to number of features in each rep,
    then selects the top features per rep by score.

    Returns dict of {rep_name: selected_indices}.
    """
    rep_names = sorted(base_reps.keys())
    rep_dims = {r: base_reps[r].shape[1] for r in rep_names}
    total_dim = sum(rep_dims.values())

    # Proportional allocation
    allocation = {}
    remaining = budget
    for i, r in enumerate(rep_names):
        if i == len(rep_names) - 1:
            allocation[r] = remaining
        else:
            n = max(1, round(budget * rep_dims[r] / total_dim))
            allocation[r] = min(n, rep_dims[r])
            remaining -= allocation[r]

    # Split concatenated scores back to per-rep
    offset = 0
    selected = {}
    for r in rep_names:
        dim = rep_dims[r]
        rep_scores = scores_concat[offset:offset + dim]
        n_select = min(allocation[r], dim)
        top_idx = np.argsort(rep_scores)[-n_select:]
        selected[r] = np.sort(top_idx)
        offset += dim

    return selected


def build_hybrid_from_selection(base_reps, selected_indices):
    """Build hybrid feature matrix from selected indices per rep."""
    rep_names = sorted(base_reps.keys())
    parts = []
    for r in rep_names:
        if r in selected_indices and len(selected_indices[r]) > 0:
            parts.append(base_reps[r][:, selected_indices[r]])
    return np.hstack(parts) if parts else np.empty((base_reps[rep_names[0]].shape[0], 0))


def evaluate_cv(X, y, model_fn, groups, n_folds=5, seed=42):
    """Evaluate with CV. Returns (mean_r2, std_r2)."""
    if groups is not None:
        splitter = GroupKFold(n_splits=n_folds)
        split_args = (X, y, groups)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_args = (X, y)

    scores = []
    for train_idx, val_idx in splitter.split(*split_args):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        m = model_fn()
        m.fit(X_tr_s, y_tr)
        y_pred = m.predict(X_val_s)
        scores.append(r2_score(y_val, y_pred))

    return float(np.mean(scores)), float(np.std(scores))


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment(base_reps, labels, groups, budgets, methods, n_folds=5, seed=42):
    """Run the full aggregation vs single-method comparison."""

    model_fn = lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rep_names = sorted(base_reps.keys())

    # --- Baselines (full dimension) ---
    print("\n=== BASELINES (full dimension) ===")
    baselines = {}
    for r in rep_names:
        mean, std = evaluate_cv(base_reps[r], labels, model_fn, groups, n_folds, seed)
        baselines[r] = (mean, std)
        print(f"  {r} (dim={base_reps[r].shape[1]}): R²={mean:.4f}±{std:.3f}")

    X_concat = np.hstack([base_reps[r] for r in rep_names])
    mean, std = evaluate_cv(X_concat, labels, model_fn, groups, n_folds, seed)
    baselines['all_concat'] = (mean, std)
    print(f"  all_concat (dim={X_concat.shape[1]}): R²={mean:.4f}±{std:.3f}")

    # --- Compute importance from all methods ---
    print(f"\n=== COMPUTING IMPORTANCE ({len(methods)} methods) ===")
    agg_results, per_rep_results = compute_all_importance(base_reps, labels, methods)

    successful = [m for m, r in agg_results.items() if r['success']]
    print(f"  Successful: {len(successful)}/{len(methods)}: {successful}")

    # --- Evaluate each budget ---
    for budget in budgets:
        print(f"\n{'='*80}")
        print(f"BUDGET = {budget}")
        print(f"{'='*80}")

        results_table = []

        # Single-method baselines (using create_hybrid pipeline)
        print(f"\n--- Single-method selection (via create_hybrid, greedy) ---")
        for method in successful:
            try:
                # Use create_hybrid for fair comparison — same allocation logic
                splitter = GroupKFold(n_splits=n_folds) if groups is not None else \
                           KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                split_args = (X_concat, labels, groups) if groups is not None else (X_concat, labels)

                fold_scores = []
                for train_idx, val_idx in splitter.split(*split_args):
                    train_reps = {k: v[train_idx] for k, v in base_reps.items()}
                    val_reps = {k: v[val_idx] for k, v in base_reps.items()}
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    X_train, feature_info = create_hybrid(
                        train_reps, y_train,
                        allocation_method='greedy',
                        budget=budget,
                        importance_method=method,
                    )
                    X_val = apply_feature_selection(val_reps, feature_info)

                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_val_s = scaler.transform(X_val)

                    m = model_fn()
                    m.fit(X_train_s, y_train)
                    y_pred = m.predict(X_val_s)
                    fold_scores.append(r2_score(y_val, y_pred))

                mean_s = float(np.mean(fold_scores))
                std_s = float(np.std(fold_scores))
                dim = budget  # approximate
                results_table.append(('single', method, mean_s, std_s, dim))
                print(f"  {method:<20}: R²={mean_s:.4f}±{std_s:.3f}")
            except Exception as e:
                print(f"  {method:<20}: FAILED - {e}")

        # Aggregated selection
        print(f"\n--- Aggregated selection (proportional allocation) ---")
        for agg_name, agg_fn in AGGREGATION_STRATEGIES.items():
            try:
                agg_scores = agg_fn(agg_results)
                selected = select_features_by_scores(base_reps, agg_scores, budget)
                total_selected = sum(len(v) for v in selected.values())

                # CV evaluation
                fold_scores = []
                splitter = GroupKFold(n_splits=n_folds) if groups is not None else \
                           KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                split_args = (X_concat, labels, groups) if groups is not None else (X_concat, labels)

                for train_idx, val_idx in splitter.split(*split_args):
                    # Recompute importance on train fold, aggregate, select
                    train_reps = {k: v[train_idx] for k, v in base_reps.items()}
                    val_reps = {k: v[val_idx] for k, v in base_reps.items()}
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    # Compute importance on train fold
                    fold_agg_results = {}
                    for method in successful:
                        try:
                            imp = compute_feature_importance(train_reps, y_train, method=method)
                            concat = np.concatenate([imp[r] for r in rep_names])
                            fold_agg_results[method] = {'success': True, 'scores': concat}
                        except Exception:
                            fold_agg_results[method] = {'success': False, 'scores': None}

                    fold_agg_scores = agg_fn(fold_agg_results)
                    fold_selected = select_features_by_scores(train_reps, fold_agg_scores, budget)

                    # Build hybrid
                    X_train = np.hstack([train_reps[r][:, fold_selected[r]] for r in rep_names
                                         if r in fold_selected and len(fold_selected[r]) > 0])
                    X_val = np.hstack([val_reps[r][:, fold_selected[r]] for r in rep_names
                                       if r in fold_selected and len(fold_selected[r]) > 0])

                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_val_s = scaler.transform(X_val)

                    m = model_fn()
                    m.fit(X_train_s, y_train)
                    y_pred = m.predict(X_val_s)
                    fold_scores.append(r2_score(y_val, y_pred))

                mean_a = float(np.mean(fold_scores))
                std_a = float(np.std(fold_scores))
                results_table.append(('aggregated', agg_name, mean_a, std_a, total_selected))
                print(f"  {agg_name:<20}: R²={mean_a:.4f}±{std_a:.3f} (dim={total_selected})")
            except Exception as e:
                print(f"  {agg_name:<20}: FAILED - {e}")

        # Summary table
        print(f"\n--- Summary (budget={budget}) ---")
        print(f"{'Type':<12} {'Method':<20} {'R²':<12} {'Dim':<6}")
        print("-" * 55)
        results_table.sort(key=lambda x: -x[2])
        for typ, method, mean, std, dim in results_table:
            print(f"{typ:<12} {method:<20} {mean:.4f}±{std:.3f}  {dim}")

        # Best baseline for comparison
        best_bl_name = max(baselines, key=lambda k: baselines[k][0])
        best_bl = baselines[best_bl_name]
        print(f"\nBest baseline: {best_bl_name} R²={best_bl[0]:.4f}±{best_bl[1]:.3f}")

        if results_table:
            best = results_table[0]
            improvement = best[2] - best_bl[0]
            print(f"Best hybrid:  {best[1]} R²={best[2]:.4f} ({improvement:+.4f} vs baseline)")


def main():
    parser = argparse.ArgumentParser(
        description='Test aggregated importance for hybrid feature selection')
    parser.add_argument('dataset', type=str, default='logd',
                        help="'logd', 'caco2', 'herg', or path to CSV")
    parser.add_argument('--reps', type=str, default='fast')
    parser.add_argument('--budgets', type=int, nargs='+', default=[50, 100, 200])
    parser.add_argument('--methods', type=str, default='all_cluster',
                        choices=['fast4', 'all_cluster'],
                        help="Which importance methods to compute")
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    methods = FAST_METHODS if args.methods == 'fast4' else ALL_CLUSTER_REPS

    print(f"Loading dataset: {args.dataset}")
    smiles, labels, task_type = load_dataset(args.dataset)
    print(f"  {len(smiles)} molecules, task: {task_type}")

    print("Assigning scaffold groups...")
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  {n_scaffolds} scaffolds")

    base_reps = get_representations(smiles, args.reps)

    run_experiment(
        base_reps, labels, groups,
        budgets=args.budgets,
        methods=methods,
        n_folds=args.n_folds,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
