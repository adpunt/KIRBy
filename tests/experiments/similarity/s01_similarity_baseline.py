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

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
)
from experiments.similarity.similarity_helpers import (
    similarity_activity_correlation, compute_sali, violation_rate,
    run_antibody_cliff_analysis,
)

from kirby.hybrid import compute_feature_importance
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


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

    task_type = data['task_type']
    is_classification = task_type == 'classification'
    if is_classification:
        print("  NOTE: classification labels — SAC uses |delta_y| on 0/1, "
              "violation rate still well-defined")

    results = {}

    # Compute importance weights for weighted baselines
    print("  Computing feature importance (lgbm_gain)...")
    try:
        importance = compute_feature_importance(reps, labels, method='lightgbm_gain')
    except Exception as e:
        print(f"  WARNING: importance computation failed ({e}), weighted baselines unavailable")
        importance = None

    # Per-rep analysis (unweighted + weighted)
    for rep_name, X in reps.items():
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
        sac_perm_p = float(np.mean(perm_sacs >= sac_rho))

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

        # Weighted version: importance-weighted distance on this rep
        if importance is not None and rep_name in importance:
            D_w = compute_distance_matrix(X, feature_weights=importance[rep_name],
                                          standardize=True)
            sac_w, sac_w_p = similarity_activity_correlation(D_w, labels, n_sample, is_clf=is_classification)
            viol_w = violation_rate(D_w, labels, k=5, is_clf=is_classification)
            sali_w = compute_sali(D_w, labels, is_clf=is_classification)
            rep_result['sac_rho_weighted'] = sac_w
            rep_result['sac_p_weighted'] = sac_w_p
            rep_result['violation_rate_weighted'] = viol_w
            rep_result['sali_mean_weighted'] = sali_w['mean']
            print(f"    SAC rho={sac_rho:.4f}, weighted={sac_w:.4f}, "
                  f"viol={viol:.3f}/{viol_w:.3f}, SALI={sali['mean']:.3f}/{sali_w['mean']:.3f}")
        else:
            print(f"    SAC rho={sac_rho:.4f} (p={sac_p:.2e}), "
                  f"violation_rate={viol:.3f}, SALI_mean={sali['mean']:.3f}")

        results[rep_name] = rep_result

    # Summary table
    print("\n  Summary:")
    has_weighted = any('sac_rho_weighted' in r for r in results.values())
    if has_weighted:
        print(f"  {'Rep':<15} {'Dim':>5} {'SAC':>9} {'SAC_w':>9} {'Viol%':>7} {'Viol_w%':>7}")
        print(f"  {'-'*15} {'-'*5} {'-'*9} {'-'*9} {'-'*7} {'-'*7}")
        for name, r in results.items():
            sac_w = r.get('sac_rho_weighted', float('nan'))
            viol_w = r.get('violation_rate_weighted', float('nan'))
            print(f"  {name:<15} {r['dim']:>5} {r['sac_rho']:>9.4f} "
                  f"{sac_w:>9.4f} {r['violation_rate']:>7.3f} {viol_w:>7.3f}")
    else:
        print(f"  {'Rep':<15} {'Dim':>5} {'SAC rho':>9} {'Viol%':>7} {'SALI':>8}")
        print(f"  {'-'*15} {'-'*5} {'-'*9} {'-'*7} {'-'*8}")
        for name, r in results.items():
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
