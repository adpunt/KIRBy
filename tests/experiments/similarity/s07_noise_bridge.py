#!/usr/bin/env python3
"""S07: Noise Detection Bridge (Supplementary) -- standard detectors flag cliffs as noise FPs.

Paper: Similarity
Conclusion: Hybrid distance reduces noise false positives.
Dependencies: s02 (reads JSON, optional -- falls back to fresh search).

Compares concat vs hybrid distance matrices for:
  - Activity cliff detection
  - kNN noise detection (KNNConsistencyFilter)
  - Neighborhood consensus (NeighborhoodConsensusFilter)
  - Cross-reference: noise flagged that are actually cliffs = false positives
  - ActivityCliffAwareEstimator comparison
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
    get_fitted_hybrid, flatten_weights,
)

from kirby.utils.noise_mitigation_methods import (
    compute_distance_matrix, get_k_neighbors_from_distances,
    ActivityCliffDetector, KNNConsistencyFilter, NeighborhoodConsensusFilter,
)
from kirby.utils.noise_estimation import (
    ActivityCliffAwareEstimator, ConformalEstimator,
)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Standard noise detectors flag activity cliffs as noise (false positives).
    Hybrid distances reduce this FP rate."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S07: NOISE DETECTION BRIDGE (Supplementary)")
    print("=" * 70)

    search_speed = config['search_speed']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {}

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    # Get hybrid
    budget = kwargs.get('budget', 100)
    augmentations = data.get('augmentations')
    try:
        X_hybrid, fi, search = get_fitted_hybrid(
            reps, labels, groups, budget=budget,
            aug_condition='uncapped', augmentations=augmentations,
            speed=config.get('search_speed', 'fast'),
            precomputed=data.get('precomputed_searches'))
        fitted = search.fitted_result
        weights_flat = flatten_weights(fitted) if fitted else np.ones(X_hybrid.shape[1])
    except Exception as e:
        print(f"  Hybrid search failed: {e}")
        return {'error': str(e)}

    # Compute distance matrices
    D_concat = compute_distance_matrix(X_concat, standardize=True)
    D_hybrid = compute_distance_matrix(X_hybrid, feature_weights=weights_flat,
                                       standardize=True)

    # --- Activity cliff detection ---
    print("\n  Activity cliff detection:")
    detector_concat = ActivityCliffDetector(precomputed_distances=D_concat)
    cliffs_concat = set(detector_concat.identify_noise(X_concat, labels).tolist())

    detector_hybrid = ActivityCliffDetector(precomputed_distances=D_hybrid)
    cliffs_hybrid = set(detector_hybrid.identify_noise(X_hybrid, labels).tolist())

    print(f"    Concat cliffs: {len(cliffs_concat)}")
    print(f"    Hybrid cliffs: {len(cliffs_hybrid)}")

    # --- Noise detection (kNN consistency) ---
    print("\n  kNN noise detection:")
    knn_concat = KNNConsistencyFilter(k=5, precomputed_distances=D_concat)
    noise_concat = set(knn_concat.identify_noise(X_concat, labels).tolist())

    knn_hybrid = KNNConsistencyFilter(k=5, precomputed_distances=D_hybrid)
    noise_hybrid = set(knn_hybrid.identify_noise(X_hybrid, labels).tolist())

    print(f"    Concat noise flagged: {len(noise_concat)}")
    print(f"    Hybrid noise flagged: {len(noise_hybrid)}")

    # --- Neighborhood consensus: distinguish noise from cliffs ---
    print("\n  Neighborhood consensus (noise vs cliff discrimination):")
    ncf_concat = NeighborhoodConsensusFilter(
        k=10, precomputed_distances=D_concat)
    pure_noise_concat = set(ncf_concat.identify_noise(X_concat, labels).tolist())
    consensus_cliffs_concat = set(ncf_concat.get_activity_cliff_indices().tolist())

    ncf_hybrid = NeighborhoodConsensusFilter(
        k=10, precomputed_distances=D_hybrid)
    pure_noise_hybrid = set(ncf_hybrid.identify_noise(X_hybrid, labels).tolist())
    consensus_cliffs_hybrid = set(ncf_hybrid.get_activity_cliff_indices().tolist())

    # Cross-reference: noise flagged that are actually cliffs = FP
    fp_concat = noise_concat & cliffs_concat  # flagged as noise, but is cliff
    fp_hybrid = noise_hybrid & cliffs_hybrid

    fp_rate_concat = len(fp_concat) / len(noise_concat) if noise_concat else 0
    fp_rate_hybrid = len(fp_hybrid) / len(noise_hybrid) if noise_hybrid else 0
    fp_reduction = ((fp_rate_concat - fp_rate_hybrid) / fp_rate_concat * 100
                    if fp_rate_concat > 0 else 0)

    results['concat'] = {
        'n_noise_flagged': len(noise_concat),
        'n_cliff_flagged': len(cliffs_concat),
        'n_consensus_noise': len(pure_noise_concat),
        'n_consensus_cliff': len(consensus_cliffs_concat),
        'fp_cliff_in_noise': len(fp_concat),
        'fp_rate': fp_rate_concat,
    }
    results['hybrid'] = {
        'n_noise_flagged': len(noise_hybrid),
        'n_cliff_flagged': len(cliffs_hybrid),
        'n_consensus_noise': len(pure_noise_hybrid),
        'n_consensus_cliff': len(consensus_cliffs_hybrid),
        'fp_cliff_in_noise': len(fp_hybrid),
        'fp_rate': fp_rate_hybrid,
    }
    results['fp_rate_reduction_pct'] = fp_reduction

    print(f"\n  FP rate (cliff-in-noise):")
    print(f"    Concat: {fp_rate_concat:.3f} ({len(fp_concat)} / {len(noise_concat)})")
    print(f"    Hybrid: {fp_rate_hybrid:.3f} ({len(fp_hybrid)} / {len(noise_hybrid)})")
    print(f"    Reduction: {fp_reduction:.1f}%")

    # --- Cliff-aware estimation ---
    print("\n  ActivityCliffAwareEstimator:")
    try:
        scaler_c = StandardScaler()
        X_concat_s = scaler_c.fit_transform(X_concat)
        est_concat = ActivityCliffAwareEstimator(k=10, metric='euclidean')
        est_concat.estimate(X_concat_s, labels)
        n_cliff_est_concat = int(est_concat.is_likely_cliff_.sum())

        scaler_h = StandardScaler()
        X_hybrid_s = scaler_h.fit_transform(X_hybrid)
        est_hybrid = ActivityCliffAwareEstimator(k=10, metric='euclidean')
        est_hybrid.estimate(X_hybrid_s, labels)
        n_cliff_est_hybrid = int(est_hybrid.is_likely_cliff_.sum())

        results['cliff_aware_estimator'] = {
            'concat_likely_cliffs': n_cliff_est_concat,
            'hybrid_likely_cliffs': n_cliff_est_hybrid,
        }
        print(f"    Concat likely cliffs: {n_cliff_est_concat}")
        print(f"    Hybrid likely cliffs: {n_cliff_est_hybrid}")
    except Exception as e:
        print(f"    Failed: {e}")
        results['cliff_aware_estimator'] = {'error': str(e)}

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
