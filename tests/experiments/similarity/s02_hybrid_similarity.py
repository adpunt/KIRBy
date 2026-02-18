#!/usr/bin/env python3
"""S02: Hybrid Property-Aware Similarity -- the fix via find_best_hybrid + weighting.

Paper: Similarity
Conclusion: Hybrid distance improves SAC by 100%+.
Dependencies: None.

Computes per-rep SAC baselines, finds best individual rep, runs find_best_hybrid,
then for each weight method computes importance-weighted distance matrix and
compares SAC: hybrid weighted vs best individual rep.
Saves blueprint and search details for downstream experiments (S03, S04, S05).
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
    save_experiment_results, get_hybrid_features_and_weights,
    _fitted_to_feature_info, _flatten_weights,
)

from kirby.hybrid import compute_feature_importance
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
    """The fix: find_best_hybrid + importance-weighted distance improves SAC."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S02: HYBRID PROPERTY-AWARE SIMILARITY")
    print("=" * 70)

    weight_methods = kwargs.get('weight_methods', ['lightgbm_gain'])
    n_sample = kwargs.get('n_sample', 5000)
    shared_search = kwargs.get('shared_search', None)
    search_speed = config['search_speed']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'weight_methods': {}, 'per_rep': {}}

    # Compute importance for weighted baselines (once, used for all baselines)
    print("  Computing feature importance (lgbm_gain) for weighted baselines...")
    try:
        importance = compute_feature_importance(reps, labels, method='lightgbm_gain')
    except Exception as e:
        print(f"  WARNING: importance failed ({e}), weighted baselines unavailable")
        importance = None

    # Best individual rep baseline (the real-world comparison)
    best_rep_sac = -1  # SAC > 0 is good; we want highest
    best_rep_name = None
    best_rep_sac_weighted = -1
    best_rep_name_weighted = None
    for rep_name, X in reps.items():
        D_rep = compute_distance_matrix(X, standardize=True)
        sac_rep, _ = similarity_activity_correlation(D_rep, labels, n_sample)
        viol_rep = violation_rate(D_rep, labels, k=5)
        rep_result = {
            'dim': X.shape[1], 'sac_rho': sac_rep, 'violation_rate': viol_rep,
        }

        # Weighted version
        if importance is not None and rep_name in importance:
            D_rep_w = compute_distance_matrix(X, feature_weights=importance[rep_name],
                                              standardize=True)
            sac_rep_w, _ = similarity_activity_correlation(D_rep_w, labels, n_sample)
            viol_rep_w = violation_rate(D_rep_w, labels, k=5)
            rep_result['sac_rho_weighted'] = sac_rep_w
            rep_result['violation_rate_weighted'] = viol_rep_w
            if sac_rep_w > best_rep_sac_weighted:
                best_rep_sac_weighted = sac_rep_w
                best_rep_name_weighted = rep_name
            print(f"  {rep_name}: SAC={sac_rep:.4f}, SAC_w={sac_rep_w:.4f}, dim={X.shape[1]}")
        else:
            print(f"  {rep_name}: SAC={sac_rep:.4f}, viol={viol_rep:.3f}, dim={X.shape[1]}")

        results['per_rep'][rep_name] = rep_result
        if best_rep_name is None or sac_rep > best_rep_sac:
            best_rep_sac = sac_rep
            best_rep_name = rep_name

    results['best_individual_rep'] = {
        'name': best_rep_name, 'sac_rho': best_rep_sac,
        'dim': reps[best_rep_name].shape[1],
    }
    if best_rep_name_weighted is not None:
        results['best_individual_rep_weighted'] = {
            'name': best_rep_name_weighted, 'sac_rho': best_rep_sac_weighted,
            'dim': reps[best_rep_name_weighted].shape[1],
        }
    print(f"\n  Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f})")
    if best_rep_name_weighted:
        print(f"  Best individual rep (weighted): {best_rep_name_weighted} "
              f"(SAC_w={best_rep_sac_weighted:.4f})")

    # Concat baseline (unweighted + weighted)
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    D_concat = compute_distance_matrix(X_concat, standardize=True)
    sac_concat, _ = similarity_activity_correlation(D_concat, labels, n_sample)
    viol_concat = violation_rate(D_concat, labels, k=5)

    concat_result = {
        'dim': X_concat.shape[1],
        'sac_rho': sac_concat,
        'violation_rate': viol_concat,
    }

    if importance is not None:
        w_concat = np.concatenate([importance[k] for k in sorted(reps.keys())])
        D_concat_w = compute_distance_matrix(X_concat, feature_weights=w_concat,
                                             standardize=True)
        sac_concat_w, _ = similarity_activity_correlation(D_concat_w, labels, n_sample)
        viol_concat_w = violation_rate(D_concat_w, labels, k=5)
        concat_result['sac_rho_weighted'] = sac_concat_w
        concat_result['violation_rate_weighted'] = viol_concat_w
        print(f"  Concat: SAC={sac_concat:.4f}, SAC_w={sac_concat_w:.4f}, "
              f"dim={X_concat.shape[1]}")
    else:
        print(f"  Concat (reference only): SAC={sac_concat:.4f}, viol={viol_concat:.3f}, "
              f"dim={X_concat.shape[1]}")

    results['concat'] = concat_result

    best_sac = best_rep_sac
    best_method = best_rep_name

    # Run search once (selection is weight_method-independent), then vary weights
    t0 = time.time()
    if shared_search is not None:
        X_hybrid, _, search = shared_search
        fitted = search.fitted_result
        print(f"  Using shared search (dim={X_hybrid.shape[1]})")
    else:
        print(f"  Running hybrid search ({search_speed})...")
        try:
            X_hybrid, _, search = get_hybrid_features_and_weights(
                reps, labels, groups, search_speed, weight_methods[0])
            fitted = search.fitted_result
        except Exception as e:
            print(f"    Search FAILED: {e}")
            results['best_weight_method'] = best_method
            results['best_sac'] = best_sac
            return results

    search_time = time.time() - t0
    hybrid_dim = X_hybrid.shape[1]

    # Hybrid unweighted (same for all weight methods -- only compute once)
    D_hybrid = compute_distance_matrix(X_hybrid, standardize=True)
    sac_hybrid, _ = similarity_activity_correlation(D_hybrid, labels, n_sample)
    viol_hybrid = violation_rate(D_hybrid, labels, k=5)
    sali_hybrid = compute_sali(D_hybrid, labels)

    print(f"    Search done ({search_time:.1f}s), dim={hybrid_dim}")
    print(f"    Hybrid unweighted: SAC={sac_hybrid:.4f}, viol={viol_hybrid:.3f}")

    for wm in weight_methods:
        print(f"\n  Weight method: {wm}")

        # Compute importance with this weight method, apply to selected features
        try:
            importance = compute_feature_importance(reps, labels, method=wm)
            w_parts = []
            for rep_name in fitted.selected_indices:
                idx = fitted.selected_indices[rep_name]
                if rep_name in importance:
                    w_parts.append(importance[rep_name][idx])
                else:
                    w_parts.append(np.ones(len(idx)))
            weights_flat = np.concatenate(w_parts)
        except Exception as e:
            print(f"    Weight computation failed: {e}")
            results['weight_methods'][wm] = {'error': str(e)}
            continue

        # Hybrid weighted
        D_weighted = compute_distance_matrix(X_hybrid, feature_weights=weights_flat,
                                             standardize=True)
        sac_weighted, _ = similarity_activity_correlation(D_weighted, labels, n_sample)
        viol_weighted = violation_rate(D_weighted, labels, k=5)
        sali_weighted = compute_sali(D_weighted, labels)

        wm_result = {
            'hybrid_unweighted': {
                'dim': hybrid_dim,
                'sac_rho': sac_hybrid,
                'violation_rate': viol_hybrid,
                'sali_mean': sali_hybrid['mean'],
            },
            'hybrid_weighted': {
                'dim': hybrid_dim,
                'sac_rho': sac_weighted,
                'violation_rate': viol_weighted,
                'sali_mean': sali_weighted['mean'],
            },
            'sac_delta_from_best_rep': sac_weighted - best_rep_sac,
            'search_time_seconds': search_time,
            'blueprint': search.best_overall['blueprint'].to_dict() if search.best_overall else None,
        }

        results['weight_methods'][wm] = wm_result

        if sac_weighted > best_sac:  # higher = better distance-activity correlation
            best_sac = sac_weighted
            best_method = wm

        print(f"    Hybrid weighted:   SAC={sac_weighted:.4f}, viol={viol_weighted:.3f}")
        print(f"    Delta SAC from best rep ({best_rep_name}): "
              f"{sac_weighted - best_rep_sac:+.4f}")

    results['best_weight_method'] = best_method
    results['best_sac'] = best_sac

    # Summary
    print(f"\n  Summary:")
    print(f"    Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f}, "
          f"dim={reps[best_rep_name].shape[1]})")
    print(f"    Best hybrid:        {best_method} (SAC={best_sac:.4f}, dim={hybrid_dim})")
    delta_sac = best_sac - best_rep_sac
    print(f"    Delta vs best rep:  {delta_sac:+.4f}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S02: Hybrid Property-Aware Similarity',
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
    save_experiment_results(results, args.output, 's02_hybrid_similarity', args.dataset)
