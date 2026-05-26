#!/usr/bin/env python3
"""S07: Noise Detection Bridge (Supplementary) -- cliff/noise FP analysis.

Paper: Similarity
Conclusion: Hybrid distances reduce false-positive cliff-as-noise rate.
Dependencies: None.

Uses REPRESENTATION-INDEPENDENT ground-truth cliffs:
  - Antibody data: edit distance (Hamming) on heavy-chain sequences
  - Molecular data: ECFP4 Tanimoto (Bajorath standard)
Then measures kNN noise detection FP rate (ground-truth cliffs flagged as noise)
in hybrid space vs top-3 individual rep baselines.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results, convert_numpy,
    get_fitted_hybrid, get_fitted_pool_hybrid, flatten_fold_weights, extract_top3_base_reps,
    get_domain_metric,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)

from kirby.utils.noise_mitigation_methods import (
    compute_distance_matrix, get_k_neighbors_from_distances,
    ActivityCliffDetector, KNNConsistencyFilter, NeighborhoodConsensusFilter,
)
from kirby.utils.noise_estimation import (
    ActivityCliffAwareEstimator, ConformalEstimator,
)
from experiments.similarity.similarity_helpers import (
    identify_cliff_compounds, identify_cliff_compounds_editdist,
)


def _knn_noise_fp_rate(X, labels, ground_truth_cliffs, k=5, standardize=True, **kwargs):
    """Compute kNN noise detection cliff-contamination rate against ground-truth cliffs.

    Returns:
        dict with n_noise_flagged, n_fp_cliff_in_noise, cliff_contamination_rate.
        Note: cliff_contamination_rate = |cliffs flagged as noise| / |noise flagged|,
        i.e. the precision of cliff-as-noise (NOT a classical FPR which would divide
        by |ground-truth cliffs|).
    """
    k = kwargs.get('k', k)
    standardize = kwargs.get('standardize', standardize)
    D = compute_distance_matrix(X, standardize=standardize)
    knn_filter = KNNConsistencyFilter(k=k, precomputed_distances=D)
    noise_flagged = set(knn_filter.identify_noise(X, labels).tolist())
    fp = noise_flagged & ground_truth_cliffs
    cliff_contamination_rate = len(fp) / len(noise_flagged) if noise_flagged else 0.0
    return {
        'n_noise_flagged': len(noise_flagged),
        'n_fp_cliff_in_noise': len(fp),
        'cliff_contamination_rate': cliff_contamination_rate,
    }


def run(data, config, output_dir, dataset_name, **kwargs):
    """Noise detectors flag activity cliffs as noise (FP).
    Hybrid + per-rep baselines compared against ground-truth cliffs."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S07: NOISE DETECTION BRIDGE (Supplementary)")
    print("=" * 70)

    search_speed = config['search_speed']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']
    domain = data.get('domain', 'molecular')
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    results = {}

    # ── Step 1: Identify ground-truth cliffs (representation-independent) ──
    ground_truth_cliffs = set()
    cliff_info = {}

    # Antibody: edit distance on heavy-chain sequences
    if 'heavy_seqs' in data and data['heavy_seqs'] is not None:
        heavy_seqs = data['heavy_seqs']
        max_edits = kwargs.get('max_edits', 5)
        cliff_mask, cliff_info = identify_cliff_compounds_editdist(
            heavy_seqs, labels, max_edits=max_edits, is_clf=is_clf)
        ground_truth_cliffs = set(np.where(cliff_mask)[0].tolist())
        print(f"\n  Ground-truth cliffs (edit distance, max_edits={max_edits}):")
        print(f"    {cliff_info['n_cliff_compounds']} compounds "
              f"({cliff_info['pct_cliff_compounds']:.1f}%), "
              f"{cliff_info['n_cliff_pairs']} pairs")
        print(f"    Median pairwise edit dist: {cliff_info['median_edit_dist']:.0f}")
        results['cliff_method'] = 'edit_distance'
        results['cliff_info'] = {k: v for k, v in cliff_info.items()
                                 if k != 'cliff_pair_counts'}

    # Molecular: ECFP4 Tanimoto
    elif 'ecfp4' in reps:
        cliff_mask, cliff_info = identify_cliff_compounds(
            reps['ecfp4'], labels, is_clf=is_clf)
        ground_truth_cliffs = set(np.where(cliff_mask)[0].tolist())
        print(f"\n  Ground-truth cliffs (ECFP4 Tanimoto >= {cliff_info['tanimoto_threshold']}):")
        print(f"    {cliff_info['n_cliff_compounds']} compounds "
              f"({cliff_info['pct_cliff_compounds']:.1f}%), "
              f"{cliff_info['n_cliff_pairs']} pairs")
        results['cliff_method'] = 'ecfp4_tanimoto'
        results['cliff_info'] = {k: v for k, v in cliff_info.items()
                                 if k != 'cliff_pair_counts'}
    else:
        print("\n  WARNING: No ground-truth cliff source available "
              "(no heavy_seqs or ecfp4). Using representation-dependent fallback.")
        results['cliff_method'] = 'none'

    if not ground_truth_cliffs:
        print("  No ground-truth cliffs found. Skipping FP analysis.")
        results.setdefault('summary', {
            'n_ground_truth_cliffs': 0,
            'best_rep': None,
            'best_rep_cliff_contamination_rate': float('nan'),
            'top3_reps': [],
            'delta_vs_best_rep': float('nan'),
        })
        results.setdefault('per_rep_baselines', {})
        results.setdefault('hybrid_unweighted', {})
        results.setdefault('hybrid_weighted', {'skipped': 'no_ground_truth_cliffs'})
        return results

    # ── Step 2: Get hybrid ──
    augmentations = data.get('augmentations')
    try:
        X_hybrid, fi, search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='knn', budget=None,
            augmentations=augmentations,
            speed=search_speed,
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset_name, distance=True)
        if fi:
            weights_flat = flatten_fold_weights(fi)
            if len(weights_flat) != X_hybrid.shape[1]:
                weights_flat = None
        else:
            weights_flat = None
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            results['top3_base_reps'] = [(n, s) for n, s in top3]
    except Exception as e:
        import traceback
        print(f"  Hybrid search failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        X_hybrid = None
        weights_flat = None

    # ── Step 3: Compute FP rates for hybrid ──
    if X_hybrid is not None:
        # Hybrid unweighted
        hybrid_uw = _knn_noise_fp_rate(X_hybrid, labels, ground_truth_cliffs)
        results['hybrid_unweighted'] = hybrid_uw
        print(f"\n  Hybrid (unweighted): "
              f"cliff contamination={hybrid_uw['cliff_contamination_rate']:.3f} "
              f"({hybrid_uw['n_fp_cliff_in_noise']}/{hybrid_uw['n_noise_flagged']})")

        # Hybrid weighted — only if importance weights are available.
        # Standardize first, then apply weights; pass standardize=False to avoid
        # re-standardizing after weighting.
        if weights_flat is None:
            results['hybrid_weighted'] = {'skipped': 'no_importance_available'}
            hybrid_w = None
            print("  Hybrid (weighted):   skipped (no importance weights available)")
        else:
            w_clean = np.nan_to_num(weights_flat, nan=0.0)
            n_nan = int(np.isnan(weights_flat).sum())
            n_neg = int((w_clean < 0).sum())
            if n_nan or n_neg:
                print(f"  WARN: weights had {n_nan} NaN and {n_neg} negative entries; "
                      f"clamped to 0.")
            w_clean = np.maximum(w_clean, 0)
            X_hybrid_w = StandardScaler().fit_transform(X_hybrid) * np.sqrt(w_clean)
            hybrid_w = _knn_noise_fp_rate(X_hybrid_w, labels, ground_truth_cliffs,
                                          standardize=False)
            results['hybrid_weighted'] = hybrid_w
            print(f"  Hybrid (weighted):   "
                  f"cliff contamination={hybrid_w['cliff_contamination_rate']:.3f} "
                  f"({hybrid_w['n_fp_cliff_in_noise']}/{hybrid_w['n_noise_flagged']})")
    else:
        hybrid_uw = None
        hybrid_w = None

    # ── Step 4: Per-rep baselines — top 3 by lowest cliff contamination rate ──
    print(f"\n  Per-rep baselines (kNN cliff-as-noise contamination vs ground-truth cliffs):")
    per_rep_results = {}
    for rep_name, X_rep in reps.items():
        rep_fp = _knn_noise_fp_rate(X_rep, labels, ground_truth_cliffs)
        per_rep_results[rep_name] = rep_fp

    # Sort by cliff contamination rate (lower is better)
    sorted_reps = sorted(per_rep_results.items(),
                         key=lambda x: x[1]['cliff_contamination_rate'])
    for rep_name, rep_fp in sorted_reps[:5]:
        print(f"    {rep_name}: cliff contamination={rep_fp['cliff_contamination_rate']:.3f} "
              f"({rep_fp['n_fp_cliff_in_noise']}/{rep_fp['n_noise_flagged']})")
    if len(sorted_reps) > 5:
        print(f"    ... ({len(sorted_reps) - 5} more)")

    results['per_rep_baselines'] = per_rep_results

    # ── Step 5: Summary ──
    top3 = sorted_reps[:3]
    best_rep_name, best_rep_fp = top3[0]
    print(f"\n  Summary:")
    print(f"    Ground-truth cliffs: {len(ground_truth_cliffs)}")
    print(f"    Best individual rep: {best_rep_name} "
          f"(cliff contamination={best_rep_fp['cliff_contamination_rate']:.3f})")
    delta_uw = None
    delta_w = None
    if X_hybrid is not None:
        delta_uw = (hybrid_uw['cliff_contamination_rate']
                    - best_rep_fp['cliff_contamination_rate'])
        print(f"    Hybrid unweighted:  cliff contamination="
              f"{hybrid_uw['cliff_contamination_rate']:.3f} "
              f"(delta vs {best_rep_name}: {delta_uw:+.3f})")
        if hybrid_w is not None:
            delta_w = (hybrid_w['cliff_contamination_rate']
                       - best_rep_fp['cliff_contamination_rate'])
            print(f"    Hybrid weighted:    cliff contamination="
                  f"{hybrid_w['cliff_contamination_rate']:.3f} "
                  f"(delta vs {best_rep_name}: {delta_w:+.3f})")

    results['summary'] = {
        'n_ground_truth_cliffs': len(ground_truth_cliffs),
        'best_rep': best_rep_name,
        'best_rep_cliff_contamination_rate': best_rep_fp['cliff_contamination_rate'],
        'top3_reps': [(name, r['cliff_contamination_rate']) for name, r in top3],
        'delta_vs_best_rep': delta_uw if delta_uw is not None else float('nan'),
        'delta_vs_best_rep_weighted': delta_w if delta_w is not None else float('nan'),
    }

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S07: Noise Detection Bridge (Supplementary)',
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
    save_experiment_results(results, args.output, 's07_noise_bridge', args.dataset)
