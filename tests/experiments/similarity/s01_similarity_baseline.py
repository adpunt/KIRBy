#!/usr/bin/env python3
"""S01: Similarity-Activity Correlation Baseline -- how poorly standard distances work.

Paper: Similarity
Conclusion: Per-rep distances poorly correlate with activity.
Dependencies: None.

For each representation, compute distance matrix and measure:
- Similarity-Activity Correlation (SAC): Spearman(distance, |delta_y|)
- Violation rate: fraction of k-NN with large |delta_y|
- SALI: Structure-Activity Landscape Index
Also computes concatenated representation baseline.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
)

from kirby.utils.noise_mitigation_methods import (
    compute_distance_matrix, get_k_neighbors_from_distances,
)


# =============================================================================
# LOCAL METRIC FUNCTIONS
# =============================================================================

def similarity_activity_correlation(D, y, n_sample=5000):
    """Spearman correlation between pairwise distance and |delta_y|.

    Positive = good (large distance -> large activity difference means the
    representation preserves the structure-activity relationship).
    Returns (rho, p_value).
    """
    n = len(y)
    triu_i, triu_j = np.triu_indices(n, k=1)
    n_pairs = len(triu_i)

    if n_pairs > n_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_pairs, n_sample, replace=False)
        triu_i, triu_j = triu_i[idx], triu_j[idx]

    dists = D[triu_i, triu_j]
    delta_y = np.abs(y[triu_i] - y[triu_j])

    rho, p = spearmanr(dists, delta_y)
    return float(rho), float(p)


def compute_sali(D, y, eps=1e-6):
    """SALI: |y_i - y_j| / (1 - sim(i,j)), sim = 1/(1+d).

    Returns per-sample max SALI and summary stats.
    """
    n = len(y)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0)

    delta_y = np.abs(y[:, None] - y[None, :])
    denom = 1.0 - sim + eps
    sali_matrix = delta_y / denom
    np.fill_diagonal(sali_matrix, 0)

    per_sample_max = sali_matrix.max(axis=1)
    per_sample_mean = sali_matrix.mean(axis=1)

    return {
        'mean': float(np.mean(per_sample_max)),
        'max': float(np.max(per_sample_max)),
        'std': float(np.std(per_sample_max)),
        'median': float(np.median(per_sample_max)),
        'per_sample_mean_avg': float(np.mean(per_sample_mean)),
    }


def violation_rate(D, y, k=5):
    """Fraction of k-NN pairs where |delta_y| > median(|delta_y|)."""
    n = len(y)
    k = min(k, n - 1)

    indices, _ = get_k_neighbors_from_distances(D, k)

    triu_i, triu_j = np.triu_indices(n, k=1)
    if len(triu_i) > 50000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(triu_i), 50000, replace=False)
        all_delta = np.abs(y[triu_i[sample_idx]] - y[triu_j[sample_idx]])
    else:
        all_delta = np.abs(y[triu_i] - y[triu_j])
    median_delta = np.median(all_delta)

    n_violations = 0
    n_total = 0
    for i in range(n):
        for j_idx in range(k):
            j = indices[i, j_idx]
            if np.abs(y[i] - y[j]) > median_delta:
                n_violations += 1
            n_total += 1

    return float(n_violations / n_total) if n_total > 0 else 0.0


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Establish the problem: how poorly standard distances correlate with
    property similarity across representations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S01: SIMILARITY-ACTIVITY CORRELATION BASELINE")
    print("=" * 70)

    n_sample = kwargs.get('n_sample', 5000)
    reps = data['reps']
    labels = data['labels']

    results = {}

    # Per-rep analysis
    for rep_name, X in reps.items():
        print(f"\n  {rep_name} (dim={X.shape[1]}):")
        D = compute_distance_matrix(X, standardize=True)
        sac_rho, sac_p = similarity_activity_correlation(D, labels, n_sample)
        viol = violation_rate(D, labels, k=5)
        sali = compute_sali(D, labels)

        results[rep_name] = {
            'dim': X.shape[1],
            'sac_rho': sac_rho,
            'sac_p': sac_p,
            'violation_rate': viol,
            'sali_mean': sali['mean'],
            'sali_max': sali['max'],
            'sali_std': sali['std'],
        }
        print(f"    SAC rho={sac_rho:.4f} (p={sac_p:.2e}), "
              f"violation_rate={viol:.3f}, SALI_mean={sali['mean']:.3f}")

    # Concatenated baseline
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    print(f"\n  all_concat (dim={X_concat.shape[1]}):")
    D_concat = compute_distance_matrix(X_concat, standardize=True)
    sac_rho, sac_p = similarity_activity_correlation(D_concat, labels, n_sample)
    viol = violation_rate(D_concat, labels, k=5)
    sali = compute_sali(D_concat, labels)

    results['all_concat'] = {
        'dim': X_concat.shape[1],
        'sac_rho': sac_rho,
        'sac_p': sac_p,
        'violation_rate': viol,
        'sali_mean': sali['mean'],
        'sali_max': sali['max'],
        'sali_std': sali['std'],
    }
    print(f"    SAC rho={sac_rho:.4f} (p={sac_p:.2e}), "
          f"violation_rate={viol:.3f}, SALI_mean={sali['mean']:.3f}")

    # Summary table
    print("\n  Summary:")
    print(f"  {'Rep':<15} {'Dim':>5} {'SAC rho':>9} {'Viol%':>7} {'SALI':>8}")
    print(f"  {'-'*15} {'-'*5} {'-'*9} {'-'*7} {'-'*8}")
    for name, r in results.items():
        print(f"  {name:<15} {r['dim']:>5} {r['sac_rho']:>9.4f} "
              f"{r['violation_rate']:>7.3f} {r['sali_mean']:>8.3f}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S01: Similarity-Activity Correlation Baseline',
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
    save_experiment_results(results, args.output, 's01_similarity_baseline', args.dataset)
