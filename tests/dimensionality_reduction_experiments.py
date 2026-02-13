#!/usr/bin/env python3
"""
Dimensionality Reduction Experiments

Systematic comparison of reduction methods across molecular representations,
both as single-rep evaluation and within the hybrid pipeline. Design follows
van der Maaten et al. (2009) benchmark structure and pyDRMetrics evaluation
protocol (Zhang et al. 2021).

Methods (organized by type):
  Baselines:
    random_subset     - Random feature selection (null baseline)
    variance_top_k    - Keep highest-variance features (simplest filter)
    select_k_best     - Univariate f_regression filter
  Projections:
    truncated_svd     - PCA on sparse data (linear)
    sparse_rp         - Sparse random projection (JL lemma, fastest)
    gaussian_rp       - Gaussian random projection (JL lemma, denser)
  Factorizations:
    nmf               - Non-negative matrix factorization
    fast_ica          - Independent component analysis
  Clustering:
    feature_agglom    - Hierarchical feature clustering
  Supervised:
    importance        - LightGBM top-K by split gain (selection, not reduction)

Evaluation metrics:
  - Downstream RF R^2 (mean +/- std across folds)
  - Performance retention ratio (R^2_reduced / R^2_full)
  - Pairwise distance correlation (Spearman) -- JL preservation metric
  - Trustworthiness at k=10 -- neighborhood preservation (sklearn)
  - Wall-clock time

Experiments:
  A: Per-rep reduction comparison (method x dim -> metrics)
  B: Hybrid with pre-reduced reps (reduce all reps, then find_best_hybrid)
  C: Selective reduction (only reduce high-dim reps, pass low-dim through)

References:
  - van der Maaten et al. 2009, "Dimensionality Reduction: A Comparative Review"
  - Zhang et al. 2021, pyDRMetrics (reconstruction, residual variance, trustworthiness)
  - Mao et al. 2023, Nature Comms (CyTOF benchmark, 21 methods)
  - Sun et al. 2019, Genome Biology (sparse high-dim scRNA-seq benchmark)

Usage:
    python tests/dimensionality_reduction_experiments.py --dataset logd --reps fast --experiments A
    python tests/dimensionality_reduction_experiments.py --dataset logd --reps all_no_gk --experiments B C
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import TruncatedSVD, FastICA, NMF
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.manifold import trustworthiness
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from lightgbm import LGBMRegressor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from kirby.find_best_hybrid import find_best_hybrid
from find_best_hybrid_molecular import (
    load_dataset, get_representations, assign_scaffold_groups, REP_PRESETS
)

warnings.filterwarnings('ignore')

# -- Method groups (order: baselines, projections, factorizations, clustering, supervised)
REDUCTION_METHODS = [
    'random_subset', 'variance_top_k', 'select_k_best',
    'truncated_svd', 'sparse_rp', 'gaussian_rp',
    'nmf', 'fast_ica',
    'feature_agglom',
    'importance',
]

# Log-spaced sweep to show elbow curves (van der Maaten 2009 design)
TARGET_DIMS = [10, 25, 50, 100, 200, 500]

# Quality metric subsample cap (pdist is O(n^2))
_QUALITY_SUBSAMPLE = 500


# =============================================================================
# QUALITY METRICS
# =============================================================================

def compute_quality_metrics(X_orig, X_reduced, seed=42):
    """Compute distance/neighborhood preservation metrics.

    Returns dict with:
      dist_corr:           Spearman corr of pairwise Euclidean distances
      trustworthiness_k10: sklearn trustworthiness at k=10

    Subsamples to _QUALITY_SUBSAMPLE rows if n is large (pdist is O(n^2)).
    """
    n = X_orig.shape[0]
    if n > _QUALITY_SUBSAMPLE:
        idx = np.random.default_rng(seed).choice(n, _QUALITY_SUBSAMPLE, replace=False)
        X_orig = X_orig[idx]
        X_reduced = X_reduced[idx]

    d_orig = pdist(X_orig, metric='euclidean')
    d_reduced = pdist(X_reduced, metric='euclidean')
    dist_corr = float(spearmanr(d_orig, d_reduced).correlation)

    k = min(10, X_orig.shape[0] - 2)
    trust = float(trustworthiness(X_orig, X_reduced, n_neighbors=k))

    return {
        'dist_corr': round(dist_corr, 4),
        'trustworthiness_k10': round(trust, 4),
    }


def characterize_reps(reps):
    """Print characterization table: dim, sparsity, dtype."""
    print(f"\n{'Rep':<18} {'Dim':>6} {'Sparsity':>10} {'Min':>10} {'Max':>10} {'Type':<10}")
    print("-" * 70)
    info = {}
    for name, X in sorted(reps.items()):
        sparsity = float((X == 0).mean())
        is_nonneg = bool(np.all(X >= -1e-10))
        is_binary = bool(np.isin(X, [0, 1]).all())
        dtype = 'binary' if is_binary else ('nonneg' if is_nonneg else 'real')
        print(f"  {name:<18} {X.shape[1]:>5} {sparsity:>9.1%} {X.min():>10.2f} {X.max():>10.2f} {dtype:<10}")
        info[name] = {
            'dim': X.shape[1], 'sparsity': round(sparsity, 4),
            'nonneg': is_nonneg, 'dtype': dtype,
        }
    return info


# =============================================================================
# REDUCTION HELPER
# =============================================================================

def fit_transform_reduce(X_train, X_test, y_train, method, n_components):
    """Fit a reducer on X_train and transform both splits.

    Returns (X_train_reduced, X_test_reduced) or None if the method
    cannot be applied (e.g. NMF on negative data, ICA convergence failure).
    """
    n_samples, n_features = X_train.shape
    n_comp = min(n_components, n_features - 1, n_samples - 1)
    if n_comp < 2:
        return None

    # --- Baselines ---
    if method == 'random_subset':
        idx = np.random.default_rng(42).choice(n_features, n_comp, replace=False)
        return X_train[:, idx], X_test[:, idx]

    elif method == 'variance_top_k':
        variances = np.var(X_train, axis=0)
        top_idx = np.argsort(variances)[-n_comp:]
        return X_train[:, top_idx], X_test[:, top_idx]

    elif method == 'select_k_best':
        k = min(n_comp, n_features)
        selector = SelectKBest(f_regression, k=k)
        try:
            return selector.fit_transform(X_train, y_train), selector.transform(X_test)
        except Exception:
            return None

    # --- Projections ---
    elif method == 'truncated_svd':
        reducer = TruncatedSVD(n_components=n_comp, random_state=42)
        return reducer.fit_transform(X_train), reducer.transform(X_test)

    elif method == 'sparse_rp':
        reducer = SparseRandomProjection(n_components=n_comp, random_state=42)
        return reducer.fit_transform(X_train), reducer.transform(X_test)

    elif method == 'gaussian_rp':
        reducer = GaussianRandomProjection(n_components=n_comp, random_state=42)
        return reducer.fit_transform(X_train), reducer.transform(X_test)

    # --- Factorizations ---
    elif method == 'nmf':
        if np.any(X_train < -1e-10) or np.any(X_test < -1e-10):
            return None
        X_tr = np.clip(X_train, 0, None)
        X_te = np.clip(X_test, 0, None)
        try:
            reducer = NMF(n_components=n_comp, random_state=42, max_iter=300)
            return reducer.fit_transform(X_tr), reducer.transform(X_te)
        except Exception:
            return None

    elif method == 'fast_ica':
        n_ica = min(n_comp, n_samples, n_features)
        try:
            reducer = FastICA(n_components=n_ica, random_state=42, max_iter=500)
            return reducer.fit_transform(X_train), reducer.transform(X_test)
        except Exception:
            return None

    # --- Clustering ---
    elif method == 'feature_agglom':
        n_clust = min(n_comp, n_features)
        try:
            reducer = FeatureAgglomeration(n_clusters=n_clust)
            return reducer.fit_transform(X_train), reducer.transform(X_test)
        except Exception:
            return None

    # --- Supervised selection ---
    elif method == 'importance':
        lgb = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        lgb.fit(X_train, y_train)
        top_idx = np.argsort(lgb.feature_importances_)[-n_comp:]
        return X_train[:, top_idx], X_test[:, top_idx]

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# EXPERIMENT A: Per-Rep Reduction Comparison
# =============================================================================

def experiment_a(reps, labels, groups, n_folds=3):
    """Compare reduction methods per-representation.

    For each (rep, method, target_dim), evaluate:
      - RF R^2 (mean +/- std, GroupKFold)
      - Performance retention ratio vs full-dim
      - Pairwise distance correlation (Spearman)
      - Trustworthiness at k=10
      - Wall-clock time
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Per-Rep Reduction Comparison")
    print("=" * 70)

    gkf = GroupKFold(n_splits=n_folds)
    results = {}

    for rep_name, X in sorted(reps.items()):
        n_features = X.shape[1]
        sparsity = float((X == 0).mean())
        print(f"\n--- {rep_name} ({n_features}d, {sparsity:.0%} sparse) ---")

        # Full-dim baseline
        baseline_scores = []
        for train_idx, test_idx in gkf.split(X, labels, groups):
            rf = RandomForestRegressor(100, random_state=42, n_jobs=-1)
            rf.fit(X[train_idx], labels[train_idx])
            baseline_scores.append(r2_score(labels[test_idx], rf.predict(X[test_idx])))
        baseline_mean = float(np.mean(baseline_scores))
        baseline_std = float(np.std(baseline_scores))
        print(f"  full ({n_features}d): R²={baseline_mean:.4f}+/-{baseline_std:.3f}")

        results[rep_name] = {
            'n_features': n_features,
            'sparsity': round(sparsity, 4),
            'baseline_r2_mean': round(baseline_mean, 4),
            'baseline_r2_std': round(baseline_std, 4),
            'reductions': {},
        }

        for method in REDUCTION_METHODS:
            for target_dim in TARGET_DIMS:
                if target_dim >= n_features:
                    continue

                fold_scores = []
                fold_quality = []
                skipped = False
                t0 = time.time()

                for train_idx, test_idx in gkf.split(X, labels, groups):
                    pair = fit_transform_reduce(
                        X[train_idx], X[test_idx], labels[train_idx],
                        method, target_dim,
                    )
                    if pair is None:
                        skipped = True
                        break
                    X_tr, X_te = pair

                    # Downstream R^2
                    rf = RandomForestRegressor(100, random_state=42, n_jobs=-1)
                    rf.fit(X_tr, labels[train_idx])
                    fold_scores.append(r2_score(labels[test_idx], rf.predict(X_te)))

                    # Quality metrics on test fold
                    qm = compute_quality_metrics(X[test_idx], X_te)
                    fold_quality.append(qm)

                elapsed = time.time() - t0
                key = f"{method}_{target_dim}"

                if skipped or not fold_scores:
                    results[rep_name]['reductions'][key] = None
                    print(f"  {method} @ {target_dim}d: SKIPPED")
                else:
                    mean_r2 = float(np.mean(fold_scores))
                    std_r2 = float(np.std(fold_scores))
                    retention = mean_r2 / baseline_mean if baseline_mean > 0 else 0
                    avg_dist_corr = float(np.mean([q['dist_corr'] for q in fold_quality]))
                    avg_trust = float(np.mean([q['trustworthiness_k10'] for q in fold_quality]))

                    results[rep_name]['reductions'][key] = {
                        'r2_mean': round(mean_r2, 4),
                        'r2_std': round(std_r2, 4),
                        'retention': round(retention, 4),
                        'dist_corr': round(avg_dist_corr, 4),
                        'trustworthiness': round(avg_trust, 4),
                        'time_s': round(elapsed, 1),
                    }
                    print(f"  {method} @ {target_dim}d: "
                          f"R²={mean_r2:.4f}+/-{std_r2:.3f} "
                          f"(ret={retention:.0%}) "
                          f"dist_corr={avg_dist_corr:.3f} "
                          f"trust={avg_trust:.3f} "
                          f"[{elapsed:.1f}s]")

    # Summary table: best method per rep at dim=100
    print(f"\n{'='*70}")
    print("EXPERIMENT A SUMMARY: Best method per rep (dim=100, by R²)")
    print(f"{'Rep':<15} {'Full R²':>8} {'Method':<18} {'Red R²':>8} {'Ret':>6} {'DistCorr':>9} {'Trust':>6}")
    print("-" * 75)
    for rep_name, data in sorted(results.items()):
        best_key, best_r2 = None, -999
        for key, val in data['reductions'].items():
            if val is not None and '_100' in key and val['r2_mean'] > best_r2:
                best_r2 = val['r2_mean']
                best_key = key
        if best_key is not None:
            v = data['reductions'][best_key]
            mname = best_key.replace('_100', '')
            print(f"  {rep_name:<15} {data['baseline_r2_mean']:>7.4f}  {mname:<18} "
                  f"{v['r2_mean']:>7.4f} {v['retention']:>5.0%} {v['dist_corr']:>8.3f} {v['trustworthiness']:>6.3f}")
        else:
            print(f"  {rep_name:<15} {data['baseline_r2_mean']:>7.4f}  {'(dim<=100)':<18}")

    # Summary: best method by metric type
    print(f"\n  Best overall by metric at dim=100:")
    for metric_key, metric_label in [('r2_mean', 'R²'), ('dist_corr', 'DistCorr'), ('trustworthiness', 'Trust')]:
        best_m, best_v = None, -999
        for rep_name, data in results.items():
            for key, val in data['reductions'].items():
                if val is not None and '_100' in key:
                    v = val[metric_key]
                    if v > best_v:
                        best_v = v
                        best_m = f"{rep_name}/{key.replace('_100','')}"
        if best_m:
            print(f"    {metric_label}: {best_m} = {best_v:.4f}")

    return results


# =============================================================================
# EXPERIMENT B: Hybrid with Pre-Reduced Reps
# =============================================================================

def experiment_b(reps, labels, groups, n_folds=3, target_dim=100, seed=42):
    """Reduce ALL reps to fixed dim, then run find_best_hybrid.

    Compare hybrid R^2 when built on reduced reps vs the standard pipeline
    (importance-based selection at the same budget).
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT B: Hybrid with Pre-Reduced Reps (target_dim={target_dim})")
    print("=" * 70)

    # Standard hybrid baseline (no pre-reduction)
    print("\n--- Baseline: standard hybrid (importance selection) ---")
    t0 = time.time()
    baseline_result = find_best_hybrid(
        base_reps=reps, labels=labels, groups=groups,
        speed='fast', seed=seed, n_folds=n_folds,
        verbose=False, cache=True,
    )
    baseline_time = time.time() - t0
    baseline_r2 = baseline_result.best_overall['score_mean'] if baseline_result.best_overall else 0
    print(f"  Baseline hybrid R²={baseline_r2:.4f} [{baseline_time:.1f}s]")

    results = {
        'baseline_hybrid_r2': round(baseline_r2, 4),
        'baseline_time_s': round(baseline_time, 1),
        'target_dim': target_dim,
        'methods': {},
    }

    for method in REDUCTION_METHODS:
        print(f"\n--- {method} (reduce all to {target_dim}d) ---")
        t0 = time.time()

        reduced_reps = {}
        skipped_reps = []
        for rep_name, X in sorted(reps.items()):
            if X.shape[1] <= target_dim:
                reduced_reps[rep_name] = X
                continue
            pair = fit_transform_reduce(X, X, labels, method, target_dim)
            if pair is None:
                skipped_reps.append(rep_name)
                continue
            reduced_reps[rep_name] = pair[0]

        if skipped_reps:
            print(f"  Skipped reps (incompatible): {skipped_reps}")

        if len(reduced_reps) < 2:
            print(f"  Too few reps after reduction, skipping")
            results['methods'][method] = None
            continue

        dims = {k: v.shape[1] for k, v in reduced_reps.items()}
        print(f"  Dims after reduction: {dims}")

        try:
            result = find_best_hybrid(
                base_reps=reduced_reps, labels=labels, groups=groups,
                speed='fast', seed=seed, n_folds=n_folds,
                verbose=False, cache=True,
            )
            hybrid_r2 = result.best_overall['score_mean'] if result.best_overall else 0
            elapsed = time.time() - t0
            delta = hybrid_r2 - baseline_r2
            print(f"  Hybrid R²={hybrid_r2:.4f} (delta={delta:+.4f}) [{elapsed:.1f}s]")

            results['methods'][method] = {
                'hybrid_r2': round(hybrid_r2, 4),
                'delta_vs_baseline': round(delta, 4),
                'skipped_reps': skipped_reps,
                'time_s': round(elapsed, 1),
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            results['methods'][method] = {'error': str(e)}

    # Summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT B SUMMARY (baseline hybrid R²={baseline_r2:.4f})")
    print(f"{'Method':<20} {'Hybrid R²':>10} {'Delta':>8} {'Time':>6}")
    print("-" * 48)
    for method, data in results['methods'].items():
        if data is None or 'error' in data:
            print(f"  {method:<20} {'FAIL':>10}")
        else:
            print(f"  {method:<20} {data['hybrid_r2']:>10.4f} "
                  f"{data['delta_vs_baseline']:>+8.4f} {data['time_s']:>5.0f}s")

    return results


# =============================================================================
# EXPERIMENT C: Selective Reduction (High-Dim Only)
# =============================================================================

def experiment_c(reps, labels, groups, n_folds=3, seed=42):
    """Only reduce reps above a dimension threshold.

    Low-dim reps pass through unchanged. Tests the practical scenario:
    you wouldn't reduce 167-dim MACCS but might reduce 4800-dim Grover.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Selective Reduction (High-Dim Only)")
    print("=" * 70)

    thresholds = [200, 500, 1000]
    methods_to_test = ['truncated_svd', 'sparse_rp', 'nmf', 'importance']
    reduce_to = 200

    # Baseline
    print("\n--- Baseline: standard hybrid ---")
    t0 = time.time()
    baseline_result = find_best_hybrid(
        base_reps=reps, labels=labels, groups=groups,
        speed='fast', seed=seed, n_folds=n_folds,
        verbose=False, cache=True,
    )
    baseline_time = time.time() - t0
    baseline_r2 = baseline_result.best_overall['score_mean'] if baseline_result.best_overall else 0
    print(f"  Baseline hybrid R²={baseline_r2:.4f} [{baseline_time:.1f}s]")

    results = {
        'baseline_hybrid_r2': round(baseline_r2, 4),
        'reduce_to': reduce_to,
        'conditions': {},
    }

    for threshold in thresholds:
        high_dim = [k for k, v in reps.items() if v.shape[1] > threshold]
        low_dim = [k for k, v in reps.items() if v.shape[1] <= threshold]
        print(f"\n=== Threshold > {threshold}d ===")
        print(f"  Reduce: {high_dim}, Keep: {low_dim}")

        if not high_dim:
            print(f"  No reps above threshold, skipping")
            results['conditions'][f'threshold_{threshold}'] = {'note': 'no high-dim reps'}
            continue

        threshold_results = {}
        for method in methods_to_test:
            print(f"\n  --- {method} (threshold={threshold}, reduce_to={reduce_to}) ---")
            t0 = time.time()

            mixed_reps = {}
            skipped = []
            for rep_name, X in sorted(reps.items()):
                if X.shape[1] <= threshold:
                    mixed_reps[rep_name] = X
                else:
                    target = min(reduce_to, X.shape[1])
                    pair = fit_transform_reduce(X, X, labels, method, target)
                    if pair is None:
                        skipped.append(rep_name)
                        continue
                    mixed_reps[rep_name] = pair[0]

            if len(mixed_reps) < 2:
                print(f"    Too few reps, skipping")
                threshold_results[method] = None
                continue

            dims = {k: v.shape[1] for k, v in mixed_reps.items()}
            print(f"    Dims: {dims}")

            try:
                result = find_best_hybrid(
                    base_reps=mixed_reps, labels=labels, groups=groups,
                    speed='fast', seed=seed, n_folds=n_folds,
                    verbose=False, cache=True,
                )
                hybrid_r2 = result.best_overall['score_mean'] if result.best_overall else 0
                elapsed = time.time() - t0
                delta = hybrid_r2 - baseline_r2
                print(f"    Hybrid R²={hybrid_r2:.4f} (delta={delta:+.4f}) [{elapsed:.1f}s]")

                threshold_results[method] = {
                    'hybrid_r2': round(hybrid_r2, 4),
                    'delta_vs_baseline': round(delta, 4),
                    'skipped_reps': skipped,
                    'time_s': round(elapsed, 1),
                }
            except Exception as e:
                print(f"    FAILED: {e}")
                threshold_results[method] = {'error': str(e)}

        results['conditions'][f'threshold_{threshold}'] = threshold_results

    # Summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT C SUMMARY (baseline R²={baseline_r2:.4f})")
    print(f"{'Threshold':<14} {'Method':<18} {'R²':>8} {'Delta':>8}")
    print("-" * 52)
    for cond_name, cond_data in results['conditions'].items():
        if isinstance(cond_data, dict) and 'note' in cond_data:
            continue
        for method, data in cond_data.items():
            if data is None or 'error' in data:
                print(f"  {cond_name:<14} {method:<18} {'FAIL':>8}")
            else:
                print(f"  {cond_name:<14} {method:<18} {data['hybrid_r2']:>8.4f} {data['delta_vs_baseline']:>+8.4f}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dimensionality reduction experiments for hybrid representations.',
    )
    parser.add_argument('--dataset', type=str, default='logd',
                        help="'logd', 'caco2', 'herg', 'qm9', or path to CSV")
    parser.add_argument('--reps', type=str, default='fast',
                        help="Preset name or comma-separated rep names")
    parser.add_argument('--experiments', nargs='+', default=['A', 'B', 'C'],
                        choices=['A', 'B', 'C'],
                        help="Which experiments to run")
    parser.add_argument('--output', type=str, default=None,
                        help="Output JSON path (default: tests/results/reduction_{dataset}_{reps}.json)")
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"=== Dimensionality Reduction Experiments ===")
    print(f"Dataset: {args.dataset}, Reps: {args.reps}, Experiments: {args.experiments}")
    print(f"Date: {datetime.now().isoformat()}")

    # Load data
    print(f"\nLoading dataset: {args.dataset}")
    smiles, labels, task_type = load_dataset(args.dataset)
    print(f"  {len(smiles)} molecules, task: {task_type}")

    print("  Assigning scaffold groups...")
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  {n_scaffolds} unique scaffolds")

    # Generate representations
    reps = get_representations(smiles, args.reps)

    # Characterize input representations
    rep_info = characterize_reps(reps)

    all_results = {
        'metadata': {
            'dataset': args.dataset,
            'reps_preset': args.reps,
            'rep_info': rep_info,
            'n_molecules': len(smiles),
            'n_scaffolds': n_scaffolds,
            'n_folds': args.folds,
            'seed': args.seed,
            'methods': REDUCTION_METHODS,
            'target_dims': TARGET_DIMS,
            'timestamp': datetime.now().isoformat(),
        },
    }

    if 'A' in args.experiments:
        all_results['experiment_a'] = experiment_a(reps, labels, groups, args.folds)

    if 'B' in args.experiments:
        all_results['experiment_b'] = experiment_b(
            reps, labels, groups, args.folds, seed=args.seed,
        )

    if 'C' in args.experiments:
        all_results['experiment_c'] = experiment_c(
            reps, labels, groups, args.folds, seed=args.seed,
        )

    # Save
    if args.output:
        output_path = args.output
    else:
        output_path = f"tests/results/reduction_{args.dataset}_{args.reps}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
