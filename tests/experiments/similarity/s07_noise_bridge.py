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
    get_fitted_hybrid, flatten_fold_weights,
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


def _knn_noise_fp_rate(X, labels, ground_truth_cliffs, k=5):
    """Compute kNN noise detection FP rate against ground-truth cliffs.

    Returns:
        dict with n_noise_flagged, n_fp (cliffs flagged as noise), fp_rate.
    """
    D = compute_distance_matrix(X, standardize=True)
    knn_filter = KNNConsistencyFilter(k=k, precomputed_distances=D)
    noise_flagged = set(knn_filter.identify_noise(X, labels).tolist())
    fp = noise_flagged & ground_truth_cliffs
    fp_rate = len(fp) / len(noise_flagged) if noise_flagged else 0.0
    return {
        'n_noise_flagged': len(noise_flagged),
        'n_fp_cliff_in_noise': len(fp),
        'fp_rate': fp_rate,
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
    task_type = data.get('task_type', 'regression')
    is_clf = task_type == 'classification'

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
        return results

    # ── Step 2: Get hybrid ──
    augmentations = data.get('augmentations')
    try:
        X_hybrid, fi, search = get_fitted_hybrid(
            reps, labels, groups,
            aug_condition='uncapped', augmentations=augmentations,
            speed=search_speed,
            precomputed=data.get('precomputed_searches'),
            internal_evaluators=DISTANCE_EVALUATORS,
            budgets=DISTANCE_BUDGETS,
            downstream_model='knn')
        weights_flat = flatten_fold_weights(fi) if fi else np.ones(X_hybrid.shape[1])
        if len(weights_flat) != X_hybrid.shape[1]:
            weights_flat = np.ones(X_hybrid.shape[1])
    except Exception as e:
        print(f"  Hybrid search failed: {e}")
        X_hybrid = None
        weights_flat = None

    # ── Step 3: Compute FP rates for hybrid ──
    if X_hybrid is not None:
        # Hybrid unweighted
        hybrid_uw = _knn_noise_fp_rate(X_hybrid, labels, ground_truth_cliffs)
        results['hybrid_unweighted'] = hybrid_uw
        print(f"\n  Hybrid (unweighted): "
              f"FP rate={hybrid_uw['fp_rate']:.3f} "
              f"({hybrid_uw['n_fp_cliff_in_noise']}/{hybrid_uw['n_noise_flagged']})")

        # Hybrid weighted
        X_hybrid_w = StandardScaler().fit_transform(X_hybrid) * np.sqrt(
            np.maximum(weights_flat, 0))
        hybrid_w = _knn_noise_fp_rate(X_hybrid_w, labels, ground_truth_cliffs)
        results['hybrid_weighted'] = hybrid_w
        print(f"  Hybrid (weighted):   "
              f"FP rate={hybrid_w['fp_rate']:.3f} "
              f"({hybrid_w['n_fp_cliff_in_noise']}/{hybrid_w['n_noise_flagged']})")

    # ── Step 4: Per-rep baselines — top 3 by lowest FP rate ──
    print(f"\n  Per-rep baselines (kNN noise FP rate vs ground-truth cliffs):")
    per_rep_results = {}
    for rep_name, X_rep in reps.items():
        X_s = StandardScaler().fit_transform(X_rep)
        rep_fp = _knn_noise_fp_rate(X_s, labels, ground_truth_cliffs)
        per_rep_results[rep_name] = rep_fp

    # Sort by FP rate (lower is better)
    sorted_reps = sorted(per_rep_results.items(), key=lambda x: x[1]['fp_rate'])
    for rep_name, rep_fp in sorted_reps[:5]:
        print(f"    {rep_name}: FP rate={rep_fp['fp_rate']:.3f} "
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
          f"(FP rate={best_rep_fp['fp_rate']:.3f})")
    if X_hybrid is not None:
        delta_uw = hybrid_uw['fp_rate'] - best_rep_fp['fp_rate']
        print(f"    Hybrid unweighted:  FP rate={hybrid_uw['fp_rate']:.3f} "
              f"(delta vs {best_rep_name}: {delta_uw:+.3f})")

    results['summary'] = {
        'n_ground_truth_cliffs': len(ground_truth_cliffs),
        'best_rep': best_rep_name,
        'best_rep_fp_rate': best_rep_fp['fp_rate'],
        'top3_reps': [(name, r['fp_rate']) for name, r in top3],
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
