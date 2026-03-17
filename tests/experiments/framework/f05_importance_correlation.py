#!/usr/bin/env python3
"""F05: Importance method correlation — are the 8 methods redundant?

Paper: Framework
Question: How correlated are the 8 importance methods? If several produce
    near-identical rankings, the search can drop them to save time.
    Also: which methods are fastest?

For each task:
    1. Concatenate all reps into one feature matrix.
    2. Compute feature importance with all 8 methods.
    3. Compute pairwise Spearman rank correlation of importance vectors.
    4. Measure wall time per method.
    5. Output: 8×8 correlation matrix + timing table.

Usage:
    python f05_importance_correlation.py --dataset logd --reps all --mode full
    python f05_importance_correlation.py --dataset flab_thermo_tresanco --reps all --mode full
    python f05_importance_correlation.py --dataset tcga_luad_vs_lusc --reps all --mode full
    python f05_importance_correlation.py --dataset buchwald_hartwig --reps all --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
from scipy.stats import spearmanr

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_domain_metric, parse_rep_args, FRAMEWORK_TASKS,
)
from kirby.hybrid import compute_feature_importance


ALL_IMPORTANCE_METHODS = [
    'treeshap',
    'xgboost_gain',
    'lightgbm_gain',
    'deeplift',
    'permutation',
    'kernelshap',
    'integrated_gradients',
    'gradientshap',
]


def _flatten_importance(importance_dict, rep_order):
    """Flatten per-rep importance dicts into a single 1-D array.

    Args:
        importance_dict: {rep_name: np.ndarray} from compute_feature_importance.
        rep_order: sorted list of rep names (for consistent ordering).

    Returns:
        1-D np.ndarray of importance scores, one per feature across all reps.
    """
    parts = []
    for rep_name in rep_order:
        arr = importance_dict.get(rep_name)
        if arr is None:
            raise KeyError(f"Missing importance for rep '{rep_name}'")
        parts.append(np.asarray(arr).ravel())
    return np.concatenate(parts)


def run(data, config, output_dir, dataset_name, methods=None, **kwargs):
    """Compute all importance methods and their pairwise rank correlations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT F05: IMPORTANCE METHOD CORRELATION")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    domain = data.get('domain', 'molecular')
    n_samples = len(labels)
    rep_order = sorted(reps.keys())
    total_features = sum(v.shape[1] for v in reps.values())

    if methods is None:
        methods = list(ALL_IMPORTANCE_METHODS)

    print(f"  Dataset: {dataset_name} ({domain}, {n_samples} samples)")
    print(f"  Reps: {len(reps)} ({total_features} total features)")
    print(f"  Methods to test: {len(methods)}")
    print(f"  Metric: {metric_name}\n", flush=True)

    # --- Step 1: Compute importance with each method ---
    # Incremental save: after each method, write a checkpoint so that slow
    # methods (kernelshap, integrated_gradients) don't lose earlier results
    # if the job crashes.
    importance_vectors = {}  # method -> flat 1-D array
    timings = {}             # method -> wall seconds
    errors = {}              # method -> error message

    for method in methods:
        print(f"  Computing {method}...", end=" ", flush=True)
        t0 = time.time()
        try:
            imp_dict = compute_feature_importance(reps, labels, method=method)
            flat = _flatten_importance(imp_dict, rep_order)
            elapsed = time.time() - t0
            importance_vectors[method] = flat
            timings[method] = elapsed
            print(f"done ({elapsed:.1f}s, "
                  f"min={flat.min():.4f}, max={flat.max():.4f}, "
                  f"nonzero={np.count_nonzero(flat)}/{len(flat)})", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            timings[method] = elapsed
            errors[method] = str(e)
            print(f"FAILED ({elapsed:.1f}s): {e}", flush=True)

        # Checkpoint: save partial results after each method so slow methods
        # don't cause total data loss on crash
        _save_checkpoint(importance_vectors, timings, errors, rep_order,
                         methods, data, dataset_name, output_dir,
                         metric_name, lower_is_better, total_features)

    successful = sorted(importance_vectors.keys(),
                        key=lambda m: methods.index(m))
    n_ok = len(successful)

    if n_ok < 2:
        print(f"\n  Only {n_ok} methods succeeded — cannot compute correlations.")
        return make_structured_output(
            experiment_id='f05',
            dataset=dataset_name,
            data=data,
            config={'methods': methods},
            results={
                'timings': timings,
                'errors': errors,
                'correlation_matrix': {},
                'method_stats': {},
            },
            primary_metric=metric_name,
            lower_is_better=lower_is_better,
        )

    # --- Step 2: Pairwise Spearman rank correlation ---
    print(f"\n  Computing {n_ok}×{n_ok} pairwise Spearman correlations...", flush=True)

    corr_matrix = {}
    for i, m1 in enumerate(successful):
        corr_matrix[m1] = {}
        for j, m2 in enumerate(successful):
            if i == j:
                corr_matrix[m1][m2] = 1.0
            elif j < i:
                # Symmetric — reuse
                corr_matrix[m1][m2] = corr_matrix[m2][m1]
            else:
                rho, _ = spearmanr(importance_vectors[m1], importance_vectors[m2])
                corr_matrix[m1][m2] = float(rho) if not np.isnan(rho) else 0.0

    # --- Step 3: Per-method statistics ---
    method_stats = {}
    for method in successful:
        v = importance_vectors[method]
        method_stats[method] = {
            'wall_time': timings[method],
            'n_features': len(v),
            'n_nonzero': int(np.count_nonzero(v)),
            'frac_nonzero': float(np.count_nonzero(v) / len(v)),
            'mean': float(np.mean(v)),
            'std': float(np.std(v)),
            'min': float(np.min(v)),
            'max': float(np.max(v)),
            'median': float(np.median(v)),
            'p90': float(np.percentile(v, 90)),
            'gini': float(_gini(np.abs(v))),
        }

    # --- Step 4: Identify redundant pairs (rho > 0.95) ---
    redundant_pairs = []
    for i, m1 in enumerate(successful):
        for j, m2 in enumerate(successful):
            if j <= i:
                continue
            rho = corr_matrix[m1][m2]
            if rho > 0.95:
                # Keep the faster one
                faster = m1 if timings[m1] <= timings[m2] else m2
                slower = m2 if faster == m1 else m1
                redundant_pairs.append({
                    'pair': [m1, m2],
                    'rho': rho,
                    'recommend_drop': slower,
                    'recommend_keep': faster,
                    'speedup': timings[slower] / timings[faster]
                                if timings[faster] > 0 else float('inf'),
                })

    # --- Print summary ---
    print(f"\n  TIMING (sorted by wall time):")
    for method in sorted(successful, key=lambda m: timings[m]):
        t = timings[method]
        print(f"    {method:<25} {t:>8.1f}s")
    for method in sorted(errors.keys()):
        print(f"    {method:<25} {'FAILED':>8}  ({errors[method][:60]})")

    print(f"\n  CORRELATION MATRIX (Spearman ρ):")
    header = f"  {'':25}" + "".join(f"{m[:10]:>12}" for m in successful)
    print(header)
    print("  " + "-" * (25 + 12 * n_ok))
    for m1 in successful:
        row = f"  {m1:<25}"
        for m2 in successful:
            rho = corr_matrix[m1][m2]
            row += f"{rho:>12.3f}"
        print(row)

    if redundant_pairs:
        print(f"\n  REDUNDANT PAIRS (ρ > 0.95):")
        for rp in redundant_pairs:
            print(f"    {rp['pair'][0]} <-> {rp['pair'][1]}: "
                  f"ρ={rp['rho']:.3f}, drop {rp['recommend_drop']} "
                  f"({rp['speedup']:.1f}x slower)")
    else:
        print(f"\n  No redundant pairs found (all ρ ≤ 0.95).")

    # Sparsity comparison
    print(f"\n  SPARSITY (Gini coefficient of |importance|, higher = sparser):")
    for method in sorted(successful,
                         key=lambda m: method_stats[m]['gini'], reverse=True):
        g = method_stats[method]['gini']
        nz = method_stats[method]['frac_nonzero']
        print(f"    {method:<25} gini={g:.3f}  nonzero={nz:.1%}")

    # Build structured output (omit raw importance_vectors — they are huge
    # and the correlation_matrix + method_stats capture everything f08 needs)
    results = {
        'correlation_matrix': corr_matrix,
        'timings': timings,
        'errors': errors,
        'method_stats': method_stats,
        'redundant_pairs': redundant_pairs,
        'successful_methods': successful,
    }

    return make_structured_output(
        experiment_id='f05',
        dataset=dataset_name,
        data=data,
        config={
            'methods': methods,
            'n_features': total_features,
            'rep_order': rep_order,
        },
        results=results,
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )


def _save_checkpoint(importance_vectors, timings, errors, rep_order,
                     methods, data, dataset_name, output_dir,
                     metric_name, lower_is_better, total_features):
    """Save partial results after each method completes.

    Computes correlations on whatever methods have succeeded so far.
    Overwrites the same output file each time.
    """
    successful = sorted(importance_vectors.keys(),
                        key=lambda m: methods.index(m))
    n_ok = len(successful)

    corr_matrix = {}
    if n_ok >= 2:
        for i, m1 in enumerate(successful):
            corr_matrix[m1] = {}
            for j, m2 in enumerate(successful):
                if i == j:
                    corr_matrix[m1][m2] = 1.0
                elif j < i:
                    corr_matrix[m1][m2] = corr_matrix[m2][m1]
                else:
                    rho, _ = spearmanr(importance_vectors[m1],
                                       importance_vectors[m2])
                    corr_matrix[m1][m2] = float(rho) if not np.isnan(rho) else 0.0

    method_stats = {}
    for method in successful:
        v = importance_vectors[method]
        method_stats[method] = {
            'wall_time': timings[method],
            'n_features': len(v),
            'n_nonzero': int(np.count_nonzero(v)),
            'frac_nonzero': float(np.count_nonzero(v) / len(v)),
        }

    checkpoint = make_structured_output(
        experiment_id='f05',
        dataset=dataset_name,
        data=data,
        config={
            'methods': methods,
            'n_features': total_features,
            'rep_order': rep_order,
            'checkpoint': True,
            'methods_completed': len(timings),
        },
        results={
            'correlation_matrix': corr_matrix,
            'timings': timings,
            'errors': errors,
            'method_stats': method_stats,
            'successful_methods': successful,
        },
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )
    save_experiment_results(checkpoint, output_dir,
                            'f05_importance_correlation', dataset_name)


def _gini(x):
    """Gini coefficient of array x (0 = perfect equality, 1 = max inequality)."""
    x = np.abs(x).ravel()
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n)


if __name__ == '__main__':
    parser = create_experiment_parser('F05: Importance Method Correlation',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--methods', type=str, nargs='+',
                        default=None,
                        help='Subset of importance methods to test '
                             '(default: all 8)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    methods = args.methods if args.methods else ALL_IMPORTANCE_METHODS
    print(f"Methods: {methods}")

    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, methods=methods)
    save_experiment_results(results, args.output, 'f05_importance_correlation',
                            args.dataset)
