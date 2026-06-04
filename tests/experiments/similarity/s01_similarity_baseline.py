#!/usr/bin/env python3
"""S01: Similarity-Activity Correlation Baseline -- how poorly standard distances work.

Paper: Similarity
Conclusion: Per-rep distances poorly correlate with activity.
Dependencies: None.

For each representation, compute distance matrix and measure:
- Similarity-Activity Correlation (SAC): Spearman(distance, |delta_y|)
- Violation rate: fraction of k-NN with large |delta_y|
- SALI: Structure-Activity Landscape Index
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
)
from experiments.similarity.similarity_helpers import (
    similarity_activity_correlation, compute_sali, violation_rate,
    run_antibody_cliff_analysis,
)

from kirby.utils.noise_mitigation_methods import compute_distance_matrix


# Above this many samples, the per-rep dense path is skipped. s01 builds a full
# n x n Euclidean distance matrix (compute_distance_matrix) AND a full n x n
# SALI matrix (compute_sali) for EVERY representation, before any subsampling.
# At n ~ 30k each float64 n x n matrix is ~7 GB, several are alive at once, and
# this repeats per rep in the loop -> OOM/crash on large datasets (nucleotide,
# large TCGA). Datasets at or below this bound are unaffected.
S01_MAX_DENSE_N = 15000


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
    n_samples = len(labels)

    task_type = data['task_type']
    is_classification = task_type == 'classification'
    if is_classification:
        print("  NOTE: classification labels — SAC uses |delta_y| on 0/1, "
              "violation rate still well-defined")

    results = {}

    # Large-n out-of-memory guard. The per-rep loop below allocates multiple
    # n x n dense matrices per representation; skip that whole path for very
    # large datasets and record a flag instead of crashing. Sequence-based
    # cliff analysis further down is independent of n x n matrices and still
    # runs. Behaviour for n <= S01_MAX_DENSE_N is unchanged.
    if n_samples > S01_MAX_DENSE_N:
        print(f"\n  SKIP dense per-rep distance/SALI path: n={n_samples} > "
              f"S01_MAX_DENSE_N={S01_MAX_DENSE_N} (n x n matrices per rep would "
              f"OOM). Recording 'skipped_large_n' flag.")
        results['_skipped_large_n'] = {
            'skipped': True,
            'reason': 'n_exceeds_S01_MAX_DENSE_N',
            'n_samples': int(n_samples),
            'max_dense_n': S01_MAX_DENSE_N,
            'n_reps': len(reps),
        }
    else:
        # Per-rep analysis
        for rep_name, X in reps.items():
            _run_per_rep(rep_name, X, labels, results, n_sample,
                         is_classification)

        # Summary table
        print("\n  Summary:")
        print(f"  {'Rep':<15} {'Dim':>5} {'SAC rho':>9} {'Viol%':>7} {'SALI':>8}")
        print(f"  {'-'*15} {'-'*5} {'-'*9} {'-'*7} {'-'*8}")
        for name, r in results.items():
            if not isinstance(r, dict) or 'sac_rho' not in r:
                continue
            print(f"  {name:<15} {r['dim']:>5} {r['sac_rho']:>9.4f} "
                  f"{r['violation_rate']:>7.3f} {r['sali_mean']:>8.3f}")

    # Sequence-based cliff sweep (antibody datasets + nucleotide) — independent
    # of any representation choice. Tests whether close pairs in sequence space
    # carry label flips (classification) or large activity diffs (regression).
    # dataset_name is passed so CDR extraction is cached per-dataset.
    cliff_results = run_antibody_cliff_analysis(
        data, is_clf=is_classification, dataset_name=dataset_name)
    if cliff_results is not None:
        results['_sequence_cliffs'] = cliff_results

    return results


def _run_per_rep(rep_name, X, labels, results, n_sample, is_classification):
    """Compute SAC / violation rate / SALI for one representation.

    Extracted unchanged from the original per-rep loop body so the large-n
    guard in run() can bypass it wholesale without altering its behaviour for
    n <= S01_MAX_DENSE_N.
    """
    print(f"\n  {rep_name} (dim={X.shape[1]}):")
    D = compute_distance_matrix(X, standardize=True)
    sac_rho, sac_p = similarity_activity_correlation(D, labels, n_sample, is_clf=is_classification)
    viol = violation_rate(D, labels, k=5, is_clf=is_classification)
    sali = compute_sali(D, labels, is_clf=is_classification)

    # Permutation test: is SAC significantly above random?
    n_perm = 200
    rng_perm = np.random.RandomState(42)
    perm_sacs = []
    for _ in range(n_perm):
        y_perm = rng_perm.permutation(labels)
        perm_rho, _ = similarity_activity_correlation(D, y_perm, n_sample, is_clf=is_classification)
        perm_sacs.append(perm_rho)
    perm_sacs = np.array(perm_sacs)
    sac_perm_p = float(np.mean(np.abs(perm_sacs) >= abs(sac_rho)))

    rep_result = {
        'dim': X.shape[1],
        'sac_rho': sac_rho,
        'sac_p': sac_p,
        'sac_permutation_p': sac_perm_p,
        'sac_significant': sac_perm_p < 0.05,
        'violation_rate': viol,
        'sali_mean': sali['mean'],
        'sali_max': sali['max'],
        'sali_std': sali['std'],
    }

    print(f"    SAC rho={sac_rho:.4f} (p={sac_p:.2e}), "
          f"violation_rate={viol:.3f}, SALI_mean={sali['mean']:.3f}")

    results[rep_name] = rep_result


if __name__ == '__main__':
    parser = create_experiment_parser('S01: Similarity-Activity Correlation Baseline',
                                     has_delta=True)
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
