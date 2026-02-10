#!/usr/bin/env python3
"""
Experiment: Low-dimensional hybrid optimization (dim 25-50).

Tests which importance methods, allocation strategies, and budgets produce
the best hybrids at very low dimensions suitable for downstream tasks
like molecular similarity, clustering, and visualization.

Hypothesis: Different methods win at very low budgets vs standard budgets.
At dim 25-50, rep diversity matters more — methods that spread features
across reps (like performance_weighted, greedy_feature) should outperform
greedy which may concentrate on one rep.

Evaluates:
  - Multiple models (RF, kNN, ridge) — kNN tests similarity-usefulness
  - Standard R² regression metric
  - Also reports pairwise distance correlation as a proxy for similarity quality

Usage:
    python tests/test_hybrid_low_dim.py logd --reps fast
    python tests/test_hybrid_low_dim.py caco2 --budgets 15 25 50
    python tests/test_hybrid_low_dim.py herg --reps fast --models rf knn ridge
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import warnings
from pathlib import Path
from copy import deepcopy
from typing import Dict, List

import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from kirby.find_best_hybrid import find_best_hybrid, SearchResult
from find_best_hybrid_molecular import load_dataset, get_representations, assign_scaffold_groups

warnings.filterwarnings('ignore')


# =============================================================================
# SIMILARITY / CLUSTERING QUALITY METRICS
# =============================================================================

def distance_correlation(X_low, X_full, n_sample=1000, seed=42):
    """Spearman correlation between pairwise distances in low-dim vs full-dim.

    High correlation means the low-dim hybrid preserves the distance structure
    of the full concatenated representation — useful for similarity search.
    """
    rng = np.random.RandomState(seed)
    n = X_low.shape[0]

    if n > n_sample:
        idx = rng.choice(n, n_sample, replace=False)
        X_low = X_low[idx]
        X_full = X_full[idx]

    d_low = pdist(X_low, metric='euclidean')
    d_full = pdist(X_full, metric='euclidean')

    corr, _ = spearmanr(d_low, d_full)
    return corr


def rep_diversity_score(feature_info):
    """Measure how evenly features are spread across representations.

    Returns Shannon entropy normalized to [0, 1]. Higher = more diverse.
    1.0 = perfectly even allocation, 0.0 = all features from one rep.
    """
    counts = []
    for rep_name, info in feature_info.items():
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue
        counts.append(len(info['selected_indices']))

    if not counts or sum(counts) == 0:
        return 0.0

    total = sum(counts)
    probs = np.array(counts) / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


# =============================================================================
# LOW-DIM SEARCH
# =============================================================================

LOW_DIM_BUDGETS = [15, 25, 50]

LOW_DIM_IMPORTANCE = ['treeshap', 'xgboost_gain', 'lightgbm_gain', 'deeplift',
                      'permutation', 'kernelshap']

LOW_DIM_ALLOCATIONS = ['greedy', 'greedy_feature', 'performance_weighted']

MODELS = {
    'rf': lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'knn': lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    'ridge': lambda: Ridge(alpha=1.0),
}


def evaluate_config_cv(base_reps, labels, groups, alloc, imp, budget,
                       model_fn, n_folds=5, seed=42):
    """Evaluate one config with CV. Returns (mean_r2, std_r2, mean_dim, diversity)."""
    from kirby.hybrid import create_hybrid, apply_feature_selection

    X_ref = next(iter(base_reps.values()))
    if groups is not None:
        splitter = GroupKFold(n_splits=n_folds)
        split_args = (X_ref, labels, groups)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_args = (X_ref, labels)

    fold_scores = []
    fold_dims = []
    fold_diversities = []

    for train_idx, val_idx in splitter.split(*split_args):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        try:
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=alloc,
                budget=budget,
                importance_method=imp,
            )
            X_val = apply_feature_selection(val_reps, feature_info)
        except Exception:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        m = model_fn()
        m.fit(X_train_s, y_train)
        y_pred = m.predict(X_val_s)
        fold_scores.append(r2_score(y_val, y_pred))
        fold_dims.append(X_train.shape[1])
        fold_diversities.append(rep_diversity_score(feature_info))

    if not fold_scores:
        return None, None, None, None

    return (float(np.mean(fold_scores)), float(np.std(fold_scores)),
            int(np.mean(fold_dims)), float(np.mean(fold_diversities)))


def run_low_dim_experiment(base_reps, labels, groups, budgets, model_names,
                           n_folds=5, seed=42, verbose=True):
    """Run the full low-dim experiment."""

    rep_names = sorted(base_reps.keys())
    X_full = np.hstack([base_reps[r] for r in rep_names])

    # --- Baselines ---
    print("\n=== BASELINES ===")
    baselines = {}
    for model_name in model_names:
        model_fn = MODELS[model_name]

        if groups is not None:
            splitter = GroupKFold(n_splits=n_folds)
            split_args = (X_full, labels, groups)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_args = (X_full, labels)

        scores = []
        for train_idx, val_idx in splitter.split(*split_args):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_full[train_idx])
            X_val = scaler.transform(X_full[val_idx])
            m = model_fn()
            m.fit(X_tr, labels[train_idx])
            scores.append(r2_score(labels[val_idx], m.predict(X_val)))

        baselines[model_name] = (float(np.mean(scores)), float(np.std(scores)))
        print(f"  {model_name} (dim={X_full.shape[1]}): R²={baselines[model_name][0]:.4f}"
              f"±{baselines[model_name][1]:.3f}")

    # --- Search ---
    all_results = []

    for budget in budgets:
        total = len(LOW_DIM_ALLOCATIONS) * len(LOW_DIM_IMPORTANCE) * len(model_names)
        print(f"\n{'='*80}")
        print(f"BUDGET = {budget} ({total} configs)")
        print(f"{'='*80}")

        budget_results = []
        idx = 0

        for alloc in LOW_DIM_ALLOCATIONS:
            for imp in LOW_DIM_IMPORTANCE:
                for model_name in model_names:
                    idx += 1
                    model_fn = MODELS[model_name]

                    if verbose:
                        print(f"  [{idx}/{total}] {model_name}/{alloc}/{imp}/b={budget}...",
                              end=' ', flush=True)

                    try:
                        mean_r2, std_r2, dim, diversity = evaluate_config_cv(
                            base_reps, labels, groups, alloc, imp, budget,
                            model_fn, n_folds, seed
                        )

                        if mean_r2 is not None:
                            result = {
                                'budget': budget, 'allocation': alloc, 'importance': imp,
                                'model': model_name, 'r2_mean': mean_r2, 'r2_std': std_r2,
                                'dim': dim, 'diversity': diversity,
                            }
                            budget_results.append(result)
                            all_results.append(result)

                            if verbose:
                                print(f"R²={mean_r2:.4f}±{std_r2:.3f} dim={dim} "
                                      f"div={diversity:.2f}")
                        else:
                            if verbose:
                                print("all folds failed")
                    except Exception as e:
                        if verbose:
                            print(f"FAILED - {e}")

        # Budget summary
        if budget_results:
            print(f"\n--- Top 10 configs (budget={budget}) ---")
            budget_results.sort(key=lambda x: -x['r2_mean'])
            print(f"{'Model':<8} {'Alloc':<22} {'Importance':<18} {'R²':<12} "
                  f"{'Dim':<6} {'Diversity'}")
            print("-" * 80)
            for r in budget_results[:10]:
                print(f"{r['model']:<8} {r['allocation']:<22} {r['importance']:<18} "
                      f"{r['r2_mean']:.4f}±{r['r2_std']:.3f}  {r['dim']:<6} "
                      f"{r['diversity']:.2f}")

            # Analysis: which allocation wins?
            print(f"\n--- Allocation method comparison (budget={budget}) ---")
            for alloc in LOW_DIM_ALLOCATIONS:
                subset = [r for r in budget_results if r['allocation'] == alloc]
                if subset:
                    mean = np.mean([r['r2_mean'] for r in subset])
                    best = max(subset, key=lambda x: x['r2_mean'])
                    mean_div = np.mean([r['diversity'] for r in subset])
                    print(f"  {alloc:<22}: mean R²={mean:.4f}, "
                          f"best={best['r2_mean']:.4f} ({best['importance']}/{best['model']}), "
                          f"mean diversity={mean_div:.2f}")

            # Analysis: which importance wins?
            print(f"\n--- Importance method comparison (budget={budget}) ---")
            for imp in LOW_DIM_IMPORTANCE:
                subset = [r for r in budget_results if r['importance'] == imp]
                if subset:
                    mean = np.mean([r['r2_mean'] for r in subset])
                    best = max(subset, key=lambda x: x['r2_mean'])
                    print(f"  {imp:<18}: mean R²={mean:.4f}, "
                          f"best={best['r2_mean']:.4f} ({best['allocation']}/{best['model']})")

            # Baseline comparison
            for model_name in model_names:
                model_results = [r for r in budget_results if r['model'] == model_name]
                if model_results:
                    best = max(model_results, key=lambda x: x['r2_mean'])
                    bl = baselines[model_name]
                    diff = best['r2_mean'] - bl[0]
                    print(f"\n  {model_name}: best hybrid (dim={best['dim']}) = "
                          f"{best['r2_mean']:.4f} vs baseline (dim={X_full.shape[1]}) = "
                          f"{bl[0]:.4f} ({diff:+.4f})")

    # --- Distance correlation for best configs ---
    if all_results:
        print(f"\n{'='*80}")
        print("DISTANCE PRESERVATION (best config per budget)")
        print(f"{'='*80}")

        from kirby.hybrid import create_hybrid

        for budget in budgets:
            budget_results = [r for r in all_results if r['budget'] == budget]
            if not budget_results:
                continue
            best = max(budget_results, key=lambda x: x['r2_mean'])

            try:
                X_hybrid, _ = create_hybrid(
                    base_reps, labels,
                    allocation_method=best['allocation'],
                    budget=budget,
                    importance_method=best['importance'],
                )
                dcorr = distance_correlation(X_hybrid, X_full)
                print(f"  budget={budget}: {best['allocation']}/{best['importance']} "
                      f"→ dist_corr={dcorr:.4f} (dim={X_hybrid.shape[1]})")
            except Exception as e:
                print(f"  budget={budget}: FAILED - {e}")

    # --- Grand summary ---
    if all_results:
        print(f"\n{'='*80}")
        print("GRAND SUMMARY: Best config per budget × model")
        print(f"{'='*80}")
        print(f"{'Budget':<8} {'Model':<8} {'Alloc':<22} {'Importance':<18} "
              f"{'R²':<12} {'Dim':<6} {'Div'}")
        print("-" * 85)
        for budget in budgets:
            for model_name in model_names:
                subset = [r for r in all_results
                          if r['budget'] == budget and r['model'] == model_name]
                if subset:
                    best = max(subset, key=lambda x: x['r2_mean'])
                    print(f"{budget:<8} {model_name:<8} {best['allocation']:<22} "
                          f"{best['importance']:<18} {best['r2_mean']:.4f}±{best['r2_std']:.3f}  "
                          f"{best['dim']:<6} {best['diversity']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Low-dimensional hybrid optimization (dim 25-50)')
    parser.add_argument('dataset', type=str, default='logd',
                        help="'logd', 'caco2', 'herg', or path to CSV")
    parser.add_argument('--reps', type=str, default='fast')
    parser.add_argument('--budgets', type=int, nargs='+', default=LOW_DIM_BUDGETS)
    parser.add_argument('--models', type=str, nargs='+', default=['rf', 'knn', 'ridge'],
                        choices=list(MODELS.keys()))
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    smiles, labels, task_type = load_dataset(args.dataset)
    print(f"  {len(smiles)} molecules, task: {task_type}")

    print("Assigning scaffold groups...")
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  {n_scaffolds} scaffolds")

    base_reps = get_representations(smiles, args.reps)

    run_low_dim_experiment(
        base_reps, labels, groups,
        budgets=args.budgets,
        model_names=args.models,
        n_folds=args.n_folds,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
