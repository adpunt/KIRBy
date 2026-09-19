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
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
)
from experiments.similarity.similarity_helpers import (
    compute_sali, violation_rate,
    run_antibody_cliff_analysis,
    similarity_activity_correlation_scalable, compute_sali_scalable,
    violation_rate_scalable, _pairwise_delta, _sample_offdiag_pairs,
    sac_pair_sample, permuted_delta_ranks, sac_permutation_p,
)

from kirby.utils.noise_mitigation_methods import compute_distance_matrix


# Method switch (NOT a skip). The dense per-rep path builds a full n x n
# distance matrix (compute_distance_matrix) under the configured metric —
# manhattan by default since 2026-09-04 (distance_config.py:68), whatever
# --distance-metric / KIRBY_DISTANCE_METRIC names otherwise — AND a full n x n
# SALI matrix (compute_sali) per representation; at n ~ 30k each float64 n x n matrix
# is ~7 GB and several are alive at once -> OOM on large datasets (nucleotide,
# large TCGA). At or below this bound we keep the exact dense path (numbers
# unchanged). Above it we compute the SAME three scores via a nearest-neighbor
# index + sampled pairs (_run_per_rep_scalable), which never materializes the
# full matrix and so scales to any n.
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

    # n x n dense matrices OOM above S01_MAX_DENSE_N. At/below the bound: exact
    # dense path (_run_per_rep, numbers unchanged). Above it: the scalable
    # nearest-neighbor path computes the SAME three scores without ever
    # building the full matrix, so it runs at any n.
    if n_samples > S01_MAX_DENSE_N:
        print(f"\n  n={n_samples} > S01_MAX_DENSE_N={S01_MAX_DENSE_N}: scalable "
              f"nearest-neighbor path (no dense n x n matrix).")
        per_rep_fn = _run_per_rep_scalable
        _scalable = True
    else:
        per_rep_fn = _run_per_rep
        _scalable = False

    # The pairs and the 200 shuffled activity columns, once for the dataset.
    # Neither depends on the representation, and rebuilding them inside each
    # one is what made the shuffles expensive enough to hold the pair count
    # down to a sample too small to separate two representations.
    _t_null = time.time()
    pairs = sac_pair_sample(n_samples, n_sample, scalable=_scalable)
    perm_ranks = permuted_delta_ranks(labels, pairs[0], pairs[1],
                                      is_clf=is_classification)
    results['_sac_sampling'] = {
        'n_pairs': int(len(pairs[0])),
        'n_pairs_available': int(n_samples * (n_samples - 1) // 2),
        'n_permutations': int(perm_ranks.shape[0]),
        'note': ('every sac_rho below is one Spearman over these molecule '
                 'pairs, and every sac_permutation_p counts these shuffles'),
    }
    print(f"\n  {len(pairs[0])} molecule pairs of "
          f"{n_samples * (n_samples - 1) // 2}, and {perm_ranks.shape[0]} "
          f"shuffled activity columns over them, built in "
          f"{time.time() - _t_null:.1f}s")

    for rep_name, X in reps.items():
        per_rep_fn(rep_name, X, labels, results, n_sample, is_classification,
                   pairs=pairs, perm_ranks=perm_ranks)

    # Summary table (both paths populate results identically)
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


def _violation_reference(labels, is_classification, scalable=False):
    """The activity difference violation_rate counts neighbours against.

    violation_rate returns the proportion and keeps the threshold to itself, so
    a reader of the result file could not tell what "a very different activity"
    was worth on this dataset. Recomputed here rather than changing the helper,
    and drawn the same way each path draws it: the dense path samples 50,000
    upper-triangle pairs under RandomState(42) (similarity_helpers.py:169-176),
    the scalable path samples 50,000 off-diagonal pairs the same seeded way
    (similarity_helpers.py:243-244). Classification has no threshold — the rate
    counts neighbours in a different class — so it returns None there.
    """
    if is_classification:
        return None
    y = np.asarray(labels)
    n = len(y)
    if scalable:
        pi, pj = _sample_offdiag_pairs(n, 50000, 42)
        return float(np.median(np.abs(y[pi] - y[pj])))
    triu_i, triu_j = np.triu_indices(n, k=1)
    if len(triu_i) > 50000:
        sample_idx = np.random.RandomState(42).choice(len(triu_i), 50000,
                                                      replace=False)
        return float(np.median(np.abs(y[triu_i[sample_idx]] -
                                      y[triu_j[sample_idx]])))
    return float(np.median(np.abs(y[triu_i] - y[triu_j])))


def _run_per_rep(rep_name, X, labels, results, n_sample, is_classification,
                 pairs=None, perm_ranks=None):
    """Compute SAC / violation rate / SALI for one representation.

    Extracted unchanged from the original per-rep loop body so the large-n
    guard in run() can bypass it wholesale without altering its behaviour for
    n <= S01_MAX_DENSE_N.
    """
    labels = np.asarray(labels)
    print(f"\n  {rep_name} (dim={X.shape[1]}):")
    D = compute_distance_matrix(X, standardize=True)

    # The pairs and the 200 shuffled activity columns come from run(), which
    # builds them once for the dataset: neither depends on the representation.
    pair_i, pair_j = pairs
    pair_dists = D[pair_i, pair_j]
    _rho, _p = spearmanr(
        pair_dists,
        _pairwise_delta(labels[pair_i], labels[pair_j], is_clf=is_classification))
    sac_rho, sac_p = float(_rho), float(_p)

    viol = violation_rate(D, labels, k=5, is_clf=is_classification)
    viol_ref = _violation_reference(labels, is_classification)
    sali = compute_sali(D, labels, is_clf=is_classification)

    # Is that correlation distinguishable from a shuffled activity column?
    sac_perm_p, _perm_rhos = sac_permutation_p(pair_dists, perm_ranks, sac_rho)

    rep_result = {
        'dim': X.shape[1],
        'sac_rho': sac_rho,
        'sac_p': sac_p,
        'sac_permutation_p': sac_perm_p,
        # How far a shuffled activity column got on these same pairs. Without
        # it a reader cannot tell whether a difference between two
        # representations' sac_rho is bigger than the noise: on a 1,412
        # molecule dataset that spread is about 0.015, and raising the pair
        # count barely moves it, because every pair is drawn from the same
        # molecules.
        'sac_permutation_null_sd': float(np.std(_perm_rhos))
                                   if len(_perm_rhos) else float('nan'),
        'sac_permutation_null_max_abs': float(np.max(np.abs(_perm_rhos)))
                                        if len(_perm_rhos) else float('nan'),
        'sac_significant': sac_perm_p < 0.05,
        'violation_rate': viol,
        # The activity difference violation_rate thresholds on, so a reader can
        # see what "a very different activity" was worth on this dataset
        # instead of only the proportion that cleared it. Same unit as the
        # activity column; None on a classification dataset, where the rate
        # counts neighbours in a different class and no threshold exists.
        'violation_threshold_median_delta_y': viol_ref,
        'sali_mean': sali['mean'],
        'sali_max': sali['max'],
        'sali_std': sali['std'],
    }

    print(f"    SAC rho={sac_rho:.4f} (p={sac_p:.2e}), "
          f"violation_rate={viol:.3f}, SALI_mean={sali['mean']:.3f}")

    results[rep_name] = rep_result


def _run_per_rep_scalable(rep_name, X, labels, results, n_sample,
                          is_classification, pairs=None, perm_ranks=None):
    """Large-n version of _run_per_rep: the SAME three scores (SAC / violation
    rate / SALI) computed via a nearest-neighbor index + sampled pairs instead
    of a full n x n matrix, so it never OOMs. Used when n > S01_MAX_DENSE_N."""
    print(f"\n  {rep_name} (dim={X.shape[1]}) [scalable nn path]:")
    sac_rho, sac_p, pi, pj, pair_dists = similarity_activity_correlation_scalable(
        X, labels, n_sample, is_clf=is_classification)
    viol = violation_rate_scalable(X, labels, k=5, is_clf=is_classification)
    viol_ref = _violation_reference(labels, is_classification, scalable=True)
    sali = compute_sali_scalable(X, labels, is_clf=is_classification)

    # Same pairs, same 200 shuffled activity columns run() built for the
    # dataset.
    sac_perm_p, _perm_rhos = sac_permutation_p(pair_dists, perm_ranks, sac_rho)

    rep_result = {
        'dim': X.shape[1],
        'sac_rho': sac_rho,
        'sac_p': sac_p,
        'sac_permutation_p': sac_perm_p,
        # How far a shuffled activity column got on these same pairs. Without
        # it a reader cannot tell whether a difference between two
        # representations' sac_rho is bigger than the noise: on a 1,412
        # molecule dataset that spread is about 0.015, and raising the pair
        # count barely moves it, because every pair is drawn from the same
        # molecules.
        'sac_permutation_null_sd': float(np.std(_perm_rhos))
                                   if len(_perm_rhos) else float('nan'),
        'sac_permutation_null_max_abs': float(np.max(np.abs(_perm_rhos)))
                                        if len(_perm_rhos) else float('nan'),
        'sac_significant': sac_perm_p < 0.05,
        'violation_rate': viol,
        # Same reference as the dense path records — see _violation_reference.
        'violation_threshold_median_delta_y': viol_ref,
        'sali_mean': sali['mean'],
        'sali_max': sali['max'],
        'sali_std': sali['std'],
        'scalable_nn_path': True,
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
