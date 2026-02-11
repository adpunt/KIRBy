#!/usr/bin/env python3
"""
Graph Kernel Investigation

One-time analysis of WL histogram representation (graph_kernel) to determine:
  A. What the ~35-72K features actually look like (sparsity, PCA dim, importance)
  B. Whether PCA or importance-based reduction preserves signal
  C. Which allocation methods work with reduced graph_kernel in hybrids

Results inform whether graph_kernel should be included in standard presets
and how to reduce it if so.

Usage:
    python tests/graph_kernel_analysis.py --dataset logd --output tests/results/gk_analysis
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from kirby.representations.molecular import create_graph_kernel
from kirby.utils.feature_filtering import apply_quality_filters
from kirby.hybrid import create_hybrid, apply_feature_selection
from kirby.find_best_hybrid import find_best_hybrid
from find_best_hybrid_molecular import (
    load_dataset, get_representations, assign_scaffold_groups
)

warnings.filterwarnings('ignore')


# =============================================================================
# EXP A: Characterize graph_kernel
# =============================================================================

def characterize_graph_kernel(
    smiles: List[str], labels: np.ndarray, groups: np.ndarray
) -> Dict[str, Any]:
    """Understand what graph_kernel features look like."""
    print("\n" + "=" * 70)
    print("EXP A: Characterize graph_kernel")
    print("=" * 70)

    exp_start = time.time()
    t0 = time.time()
    X_gk = create_graph_kernel(smiles)
    gen_time = time.time() - t0
    n, d = X_gk.shape
    print(f"  Shape: {n} x {d}, generation time: {gen_time:.1f}s")

    results = {
        'n_samples': n, 'n_features': d, 'generation_time_s': round(gen_time, 1),
    }

    # 1. Sparsity profile
    sparsity = float((X_gk == 0).sum() / X_gk.size)
    sparsity_per_feature = (X_gk == 0).sum(axis=0) / n
    features_gt95_sparse = int((sparsity_per_feature > 0.95).sum())
    features_gt99_sparse = int((sparsity_per_feature > 0.99).sum())
    print(f"  Overall sparsity: {sparsity:.3f}")
    print(f"  Features >95% sparse: {features_gt95_sparse}/{d}")
    print(f"  Features >99% sparse: {features_gt99_sparse}/{d}")
    results['sparsity'] = round(sparsity, 4)
    results['features_gt95_sparse'] = features_gt95_sparse
    results['features_gt99_sparse'] = features_gt99_sparse

    # 2. Quality filtering baseline
    t0 = time.time()
    X_filt, mask = apply_quality_filters(X_gk)
    filt_time = time.time() - t0
    n_survived = int(mask.sum())
    print(f"  After quality filters: {n_survived}/{d} features survive ({filt_time:.1f}s)")
    results['filtered_features'] = n_survived
    results['filter_time_s'] = round(filt_time, 1)

    # 3. PCA effective dimensionality (randomized SVD — full SVD on 71K is too slow)
    print("  Computing PCA effective dimensionality (TruncatedSVD)...")
    t0 = time.time()
    max_comp = min(500, n - 1)
    svd = TruncatedSVD(n_components=max_comp, random_state=42)
    svd.fit(StandardScaler().fit_transform(X_gk))
    cumvar = np.cumsum(svd.explained_variance_ratio_)
    pca_time = time.time() - t0

    pca_dims = {}
    for threshold in [0.80, 0.85, 0.90, 0.95]:
        idx = np.searchsorted(cumvar, threshold)
        pca_dims[f'pca_{int(threshold*100)}'] = int(min(idx + 1, max_comp)) if idx < len(cumvar) else max_comp
    print(f"  PCA dims: {pca_dims} (computed in {pca_time:.1f}s)")
    print(f"  Variance at 500 components: {cumvar[-1]:.3f}")
    results['pca_dims'] = pca_dims
    results['variance_at_500'] = round(float(cumvar[-1]), 4)
    results['pca_time_s'] = round(pca_time, 1)

    # 4. Feature importance distribution
    print("  Computing LightGBM importance distribution...")
    t0 = time.time()
    from lightgbm import LGBMRegressor
    lgb = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(X_gk, labels)
    importances = lgb.feature_importances_.astype(float)
    imp_time = time.time() - t0

    # Sort descending
    sorted_imp = np.sort(importances)[::-1]
    total_imp = sorted_imp.sum()
    if total_imp > 0:
        cum_imp = np.cumsum(sorted_imp) / total_imp
        nonzero_features = int((importances > 0).sum())

        top_k_coverage = {}
        for k in [50, 100, 200, 500]:
            if k <= len(cum_imp):
                top_k_coverage[f'top_{k}'] = round(float(cum_imp[k-1]), 4)
        print(f"  Non-zero importance features: {nonzero_features}/{d}")
        print(f"  Cumulative importance: {top_k_coverage}")

        # Gini coefficient
        n_feat = len(sorted_imp)
        index = np.arange(1, n_feat + 1)
        gini = float(2 * np.sum(index * sorted_imp) / (n_feat * total_imp) - (n_feat + 1) / n_feat)
    else:
        nonzero_features = 0
        top_k_coverage = {}
        gini = 1.0

    results['nonzero_importance_features'] = nonzero_features
    results['importance_coverage'] = top_k_coverage
    results['importance_gini'] = round(gini, 4)
    results['importance_time_s'] = round(imp_time, 1)

    # 5. Per-iteration analysis (WL iterations 0..5)
    print("  Per-iteration breakdown...")
    # Feature names are like "0_C", "3_C_C_N_O" — prefix is iteration number
    iteration_stats = {}
    for it in range(6):
        prefix = f"{it}_"
        mask_it = np.array([True] * d)  # can't get feature names from array
        # Since we don't have feature names, estimate by feature index ranges
        # WL iterations produce groups of features — we'll compute per-group
        # variance instead (features within each iteration share structural depth)
    # Skip per-iteration if we can't get feature names from the array
    results['note_per_iteration'] = "Per-iteration analysis requires feature names (not stored in numpy array)"

    print(f"  EXP A complete ({time.time() - exp_start:.0f}s total, {gen_time:.0f}s gen)")
    return results


# =============================================================================
# EXP B: Compare reduction methods
# =============================================================================

def compare_reductions(
    smiles: List[str], labels: np.ndarray, groups: np.ndarray, n_folds: int = 3
) -> Dict[str, Any]:
    """Compare PCA vs importance-based reduction of graph_kernel.

    Tests 8 conditions: full, filtered, pca_{100,200,500}, imp_{100,200,500}.
    For each: single-rep RF R² and hybrid RF R² (gk + fast reps, budget=100).
    """
    print("\n" + "=" * 70)
    print("EXP B: Compare reduction methods")
    print("=" * 70)

    X_gk = create_graph_kernel(smiles)
    other_reps = get_representations(smiles, 'fast')
    n, d = X_gk.shape

    # Pre-compute quality filter mask
    _, filt_mask = apply_quality_filters(X_gk)

    gkf = GroupKFold(n_splits=n_folds)
    sample_X = np.zeros((n, 1))  # dummy for split

    results = {}

    conditions = {
        'full': ('identity', None),
        'filtered': ('filter', None),
        'pca_500': ('pca', 500),
        'pca_200': ('pca', 200),
        'pca_100': ('pca', 100),
        'imp_500': ('importance', 500),
        'imp_200': ('importance', 200),
        'imp_100': ('importance', 100),
    }

    for cond_name, (method, param) in conditions.items():
        print(f"\n  --- {cond_name} ---")
        t0 = time.time()

        single_scores = []
        hybrid_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(sample_X, labels, groups)):
            X_gk_train = X_gk[train_idx]
            X_gk_test = X_gk[test_idx]
            y_train = labels[train_idx]
            y_test = labels[test_idx]

            # Apply reduction (fit on train, transform test)
            if method == 'identity':
                X_train_r = X_gk_train
                X_test_r = X_gk_test
            elif method == 'filter':
                X_train_r = X_gk_train[:, filt_mask]
                X_test_r = X_gk_test[:, filt_mask]
            elif method == 'pca':
                n_comp = min(param, X_gk_train.shape[0] - 1)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_gk_train)
                X_test_scaled = scaler.transform(X_gk_test)
                pca = TruncatedSVD(n_components=n_comp, random_state=42)
                X_train_r = pca.fit_transform(X_train_scaled)
                X_test_r = pca.transform(X_test_scaled)
            elif method == 'importance':
                # Per-fold: compute importance on train, select top-N
                lgb_fold = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
                lgb_fold.fit(X_gk_train, y_train)
                top_idx = np.argsort(lgb_fold.feature_importances_)[-param:]
                X_train_r = X_gk_train[:, top_idx]
                X_test_r = X_gk_test[:, top_idx]

            effective_dim = X_train_r.shape[1]

            # Single-rep RF
            rf = RandomForestRegressor(100, random_state=42, n_jobs=-1)
            rf.fit(X_train_r, y_train)
            single_scores.append(r2_score(y_test, rf.predict(X_test_r)))

            # Hybrid RF: gk_reduced + fast reps, budget=100, greedy
            fold_reps_train = {k: v[train_idx] for k, v in other_reps.items()}
            fold_reps_train['graph_kernel'] = X_train_r
            fold_reps_test = {k: v[test_idx] for k, v in other_reps.items()}
            fold_reps_test['graph_kernel'] = X_test_r

            try:
                X_hybrid_train, feature_info = create_hybrid(
                    fold_reps_train, y_train,
                    allocation_method='greedy',
                    budget=100,
                    importance_method='treeshap',
                )
                X_hybrid_test = apply_feature_selection(fold_reps_test, feature_info)

                rf_h = RandomForestRegressor(100, random_state=42, n_jobs=-1)
                rf_h.fit(X_hybrid_train, y_train)
                hybrid_scores.append(r2_score(y_test, rf_h.predict(X_hybrid_test)))
            except Exception as e:
                print(f"    Fold {fold_idx} hybrid failed: {e}")
                hybrid_scores.append(np.nan)

        elapsed = time.time() - t0
        single_mean = float(np.nanmean(single_scores))
        single_std = float(np.nanstd(single_scores))
        hybrid_mean = float(np.nanmean(hybrid_scores))
        hybrid_std = float(np.nanstd(hybrid_scores))

        print(f"    dim={effective_dim}, single RF R²={single_mean:.4f}±{single_std:.3f}, "
              f"hybrid RF R²={hybrid_mean:.4f}±{hybrid_std:.3f}, time={elapsed:.0f}s")

        results[cond_name] = {
            'effective_dim': effective_dim,
            'single_r2_mean': round(single_mean, 4),
            'single_r2_std': round(single_std, 4),
            'hybrid_r2_mean': round(hybrid_mean, 4),
            'hybrid_r2_std': round(hybrid_std, 4),
            'time_s': round(elapsed, 1),
        }

    return results


# =============================================================================
# EXP C: Test allocation methods with best reduction
# =============================================================================

def test_allocations(
    smiles: List[str], labels: np.ndarray, groups: np.ndarray,
    reduction: str = 'imp_200', n_folds: int = 3,
) -> Dict[str, Any]:
    """Test each allocation method with reduced graph_kernel.

    Uses importance-based reduction by default (from Exp B results).
    """
    print("\n" + "=" * 70)
    print(f"EXP C: Test allocation methods (reduction={reduction})")
    print("=" * 70)

    X_gk = create_graph_kernel(smiles)
    other_reps = get_representations(smiles, 'fast')

    # Reduce graph_kernel
    method, param = reduction.split('_')
    param = int(param)

    if method == 'pca':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_gk)
        reducer = TruncatedSVD(n_components=min(param, X_gk.shape[0] - 1), random_state=42)
        X_gk_reduced = reducer.fit_transform(X_scaled)
    elif method == 'imp':
        from lightgbm import LGBMRegressor
        lgb = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        lgb.fit(X_gk, labels)
        top_idx = np.argsort(lgb.feature_importances_)[-param:]
        X_gk_reduced = X_gk[:, top_idx]
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    print(f"  Reduced graph_kernel: {X_gk.shape[1]} -> {X_gk_reduced.shape[1]}")

    reps = {**other_reps, 'graph_kernel': X_gk_reduced}

    methods = ['greedy', 'greedy_feature', 'performance_weighted', 'two_stage']
    results = {}

    for alloc_method in methods:
        print(f"\n  --- {alloc_method} ---")
        t0 = time.time()
        try:
            result = find_best_hybrid(
                base_reps=reps,
                labels=labels,
                groups=groups,
                allocation_methods=[alloc_method],
                importance_methods=['treeshap'],
                budgets=[100],
                models=['rf'],
                n_folds=n_folds,
                speed='fast',
                seed=42,
                return_fitted=True,
            )
            elapsed = time.time() - t0

            best = result.best_overall
            if best is not None:
                bp = best['blueprint']
                # Check how many features came from graph_kernel via fitted result
                gk_alloc = 'N/A'
                if result.fitted_result and hasattr(result.fitted_result, 'selected_indices'):
                    gk_idx = result.fitted_result.selected_indices.get('graph_kernel', [])
                    gk_alloc = len(gk_idx)
                print(f"    R²={best['score_mean']:.4f}±{best['score_std']:.3f}, "
                      f"dim={best['n_features']}, gk_alloc={gk_alloc}, time={elapsed:.0f}s")
                results[alloc_method] = {
                    'r2_mean': round(best['score_mean'], 4),
                    'r2_std': round(best['score_std'], 4),
                    'n_features': best['n_features'],
                    'gk_allocation': gk_alloc,
                    'blueprint': bp.to_dict() if hasattr(bp, 'to_dict') else str(bp),
                    'time_s': round(elapsed, 1),
                }
            else:
                print(f"    No result, time={elapsed:.0f}s")
                results[alloc_method] = {'error': 'no result', 'time_s': round(elapsed, 1)}

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    FAILED: {e} ({elapsed:.0f}s)")
            results[alloc_method] = {'error': str(e), 'time_s': round(elapsed, 1)}

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Graph kernel investigation')
    parser.add_argument('--dataset', type=str, default='logd',
                        help='Dataset: logd, caco2, herg, qm9')
    parser.add_argument('--output', type=str, default='tests/results/gk_analysis',
                        help='Output path prefix (no extension)')
    parser.add_argument('--experiments', type=str, nargs='+', default=['A', 'B', 'C'],
                        help='Which experiments to run: A, B, C')
    parser.add_argument('--reduction', type=str, default='imp_200',
                        help='Reduction for Exp C (e.g. imp_200, pca_500)')
    parser.add_argument('--folds', type=int, default=3,
                        help='Number of CV folds (default: 3)')
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Graph Kernel Analysis — {args.dataset}")
    print(f"Experiments: {args.experiments}")
    print(f"Output: {args.output}")

    # Load data
    print(f"\nLoading {args.dataset}...")
    smiles, labels, task_type = load_dataset(args.dataset)
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  {len(smiles)} samples, {n_scaffolds} scaffolds")

    all_results = {
        'metadata': {
            'dataset': args.dataset,
            'n_samples': len(smiles),
            'n_scaffolds': n_scaffolds,
            'timestamp': datetime.now().isoformat(),
            'experiments': args.experiments,
        }
    }

    if 'A' in args.experiments:
        all_results['exp_a'] = characterize_graph_kernel(smiles, labels, groups)

    if 'B' in args.experiments:
        all_results['exp_b'] = compare_reductions(smiles, labels, groups, n_folds=args.folds)

    if 'C' in args.experiments:
        all_results['exp_c'] = test_allocations(
            smiles, labels, groups, reduction=args.reduction, n_folds=args.folds
        )

    # Save results
    output_path = f"{args.output}_{args.dataset}.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if 'exp_a' in all_results:
        a = all_results['exp_a']
        print(f"\n  graph_kernel: {a['n_features']} features, sparsity={a['sparsity']}")
        print(f"  After filtering: {a['filtered_features']} features")
        print(f"  Non-zero importance: {a['nonzero_importance_features']} features")
        if a.get('pca_dims'):
            print(f"  PCA dims: {a['pca_dims']}")
        if a.get('importance_coverage'):
            print(f"  Importance coverage: {a['importance_coverage']}")

    if 'exp_b' in all_results:
        b = all_results['exp_b']
        print(f"\n  {'Condition':<12} {'Dim':<8} {'Single R²':<14} {'Hybrid R²':<14} {'Time'}")
        print("  " + "-" * 60)
        for name, r in b.items():
            print(f"  {name:<12} {r['effective_dim']:<8} "
                  f"{r['single_r2_mean']:.4f}±{r['single_r2_std']:.3f}  "
                  f"{r['hybrid_r2_mean']:.4f}±{r['hybrid_r2_std']:.3f}  "
                  f"{r['time_s']:.0f}s")

    if 'exp_c' in all_results:
        c = all_results['exp_c']
        print(f"\n  {'Method':<22} {'R²':<14} {'Dim':<6} {'GK alloc':<10} {'Time'}")
        print("  " + "-" * 60)
        for name, r in c.items():
            if 'error' in r:
                print(f"  {name:<22} FAILED: {r['error']}")
            else:
                print(f"  {name:<22} {r['r2_mean']:.4f}±{r['r2_std']:.3f}  "
                      f"{r['n_features']:<6} {str(r['gk_allocation']):<10} {r['time_s']:.0f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
