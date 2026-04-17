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

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_fitted_hybrid, flatten_weights,
    extract_top3_base_reps,
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
    """The fix: find_best_hybrid + importance-weighted distance improves SAC."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S02: HYBRID PROPERTY-AWARE SIMILARITY")
    print("=" * 70)

    weight_methods = kwargs.get('weight_methods', ['lightgbm_gain'])
    n_sample = kwargs.get('n_sample', 5000)
    search_speed = config.get('search_speed', 'full')
    task_type = data['task_type']
    is_classification = task_type == 'classification'

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
        sac_rep, _ = similarity_activity_correlation(D_rep, labels, n_sample, is_clf=is_classification)
        viol_rep = violation_rate(D_rep, labels, k=5, is_clf=is_classification)
        rep_result = {
            'dim': X.shape[1], 'sac_rho': sac_rep, 'violation_rate': viol_rep,
        }

        # Weighted version
        if importance is not None and rep_name in importance:
            D_rep_w = compute_distance_matrix(X, feature_weights=importance[rep_name],
                                              standardize=True)
            sac_rep_w, _ = similarity_activity_correlation(D_rep_w, labels, n_sample, is_clf=is_classification)
            viol_rep_w = violation_rate(D_rep_w, labels, k=5, is_clf=is_classification)
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
    sac_concat, _ = similarity_activity_correlation(D_concat, labels, n_sample, is_clf=is_classification)
    viol_concat = violation_rate(D_concat, labels, k=5, is_clf=is_classification)

    concat_result = {
        'dim': X_concat.shape[1],
        'sac_rho': sac_concat,
        'violation_rate': viol_concat,
    }

    if importance is not None:
        w_concat = np.concatenate([importance[k] for k in sorted(reps.keys())])
        D_concat_w = compute_distance_matrix(X_concat, feature_weights=w_concat,
                                             standardize=True)
        sac_concat_w, _ = similarity_activity_correlation(D_concat_w, labels, n_sample, is_clf=is_classification)
        viol_concat_w = violation_rate(D_concat_w, labels, k=5, is_clf=is_classification)
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
    budget = kwargs.get('budget', 100)
    augmentations = data.get('augmentations')
    print(f"  Running hybrid search (budget={budget}, speed={search_speed})...")
    try:
        X_hybrid, fi, search = get_fitted_hybrid(
            reps, labels, groups,
            aug_condition='uncapped', augmentations=augmentations,
            speed=search_speed,
            precomputed=data.get('precomputed_searches'),
            downstream_model='knn',
            dataset_name=dataset_name, tier='general', task_type=task_type)
        fitted = search.fitted_results.get('knn')
    except Exception as e:
        print(f"    Search FAILED: {e}")
        results['best_weight_method'] = best_method
        results['best_sac'] = best_sac
        return results

    search_time = time.time() - t0
    hybrid_dim = X_hybrid.shape[1]

    top3 = extract_top3_base_reps(search)
    if top3:
        print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        results['top3_base_reps'] = [(n, s) for n, s in top3]

    # Hybrid unweighted (same for all weight methods -- only compute once)
    D_hybrid = compute_distance_matrix(X_hybrid, standardize=True)
    sac_hybrid, _ = similarity_activity_correlation(D_hybrid, labels, n_sample, is_clf=is_classification)
    viol_hybrid = violation_rate(D_hybrid, labels, k=5, is_clf=is_classification)
    sali_hybrid = compute_sali(D_hybrid, labels, is_clf=is_classification)

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
        sac_weighted, _ = similarity_activity_correlation(D_weighted, labels, n_sample, is_clf=is_classification)
        viol_weighted = violation_rate(D_weighted, labels, k=5, is_clf=is_classification)
        sali_weighted = compute_sali(D_weighted, labels, is_clf=is_classification)

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
            'blueprint': (search.best_by_model['knn']['blueprint'].to_dict()
                          if search.best_by_model.get('knn', {}).get('blueprint') else None),
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

    # Sequence-based cliff sweep (antibody datasets + nucleotide) — independent
    # of any representation choice. Same sweep as s01 for cross-experiment
    # comparability; CDR indices cached per-dataset for repeat runs.
    cliff_results = run_antibody_cliff_analysis(
        data, is_clf=is_classification, dataset_name=dataset_name)
    if cliff_results is not None:
        results['_sequence_cliffs'] = cliff_results

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
