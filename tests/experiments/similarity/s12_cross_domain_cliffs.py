#!/usr/bin/env python3
"""S12: Cross-Domain Activity Cliff Analysis — generic cliff detection for all domains.

Paper: Similarity
Companion to S04 (which uses ECFP4 Tanimoto, molecular-only). S12 uses Euclidean
similarity on concatenated reps, making it applicable to antibody, TCGA, and molecular
datasets alike. Same approach as N01's non-molecular branch.

Cliff definition: pairs where sim >= threshold AND |delta_y| > k * MAD(y).
For Euclidean: sim = 1/(1+d) on standardised concat.

Usage:
    python s12_cross_domain_cliffs.py --dataset flab_thermo_tresanco --mode intermediary
    python s12_cross_domain_cliffs.py --dataset tcga_luad_vs_lusc --mode intermediary
    python s12_cross_domain_cliffs.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, get_precomputed_search,
    build_fold_hybrid,
)
from kirby.hybrid import create_hybrid
from kirby.hybrid import apply_feature_selection
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


# =========================================================================
# CLIFF DETECTION (domain-generic, mirrors N01's identify_cliff_pairs)
# =========================================================================

def identify_cliff_pairs_euclidean(X_concat, y, sim_threshold=0.55, activity_k=2.0,
                                   max_compounds=5000):
    """Identify activity cliff pairs using Euclidean similarity.

    sim = 1/(1+d) on standardised features. Cliff = high structural similarity
    AND high activity difference (|delta_y| > k * MAD(y)).

    Returns (cliff_mask, info) where cliff_mask is boolean array marking cliff compounds.
    """
    n = len(y)
    if n > max_compounds:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_compounds, replace=False)
        X_sub = X_concat[idx]
        y_sub = y[idx]
    else:
        X_sub = X_concat
        y_sub = y
        idx = np.arange(n)

    D = compute_distance_matrix(X_sub, standardize=True)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)

    mad_y = float(np.median(np.abs(y_sub - np.median(y_sub))))
    activity_threshold = activity_k * mad_y

    delta_y = np.abs(y_sub[:, None] - y_sub[None, :])
    cliff_pair_mask = (sim >= sim_threshold) & (delta_y > activity_threshold)
    np.fill_diagonal(cliff_pair_mask, False)

    cliff_mask_sub = cliff_pair_mask.any(axis=1)
    n_cliff_pairs = int(cliff_pair_mask.sum()) // 2
    n_cliff_compounds = int(cliff_mask_sub.sum())

    # Map back to full indices
    cliff_indices = set(idx[cliff_mask_sub].tolist())

    info = {
        'sim_threshold': sim_threshold,
        'activity_k': activity_k,
        'activity_threshold': float(activity_threshold),
        'mad_y': float(mad_y),
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': float(100.0 * n_cliff_compounds / len(y_sub)),
        'n_analyzed': len(y_sub),
        'similarity_type': 'euclidean',
    }

    return cliff_indices, info


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """S12: Cross-domain activity cliff analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S12: CROSS-DOMAIN ACTIVITY CLIFFS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    domain = data.get('domain', 'molecular')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    search_speed = config['search_speed']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data.get('task_type', 'regression')
    _rf_factory = get_model_factories(['rf'], task_type=task_type,
                                       n_samples=len(labels))['rf']

    sim_threshold = kwargs.get('sim_threshold', 0.55)
    activity_k = kwargs.get('activity_k', 2.0)

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    print(f"  Domain: {domain}, n={len(labels)}, dim={X_concat.shape[1]}")

    # Identify cliffs
    cliff_indices, cliff_info = identify_cliff_pairs_euclidean(
        X_concat, labels, sim_threshold, activity_k)
    print(f"  Cliffs: {cliff_info['n_cliff_compounds']}/{cliff_info['n_analyzed']} "
          f"({cliff_info['pct_cliff_compounds']:.1f}%), "
          f"{cliff_info['n_cliff_pairs']} pairs")

    if not cliff_indices:
        print("  No cliffs found — adjusting threshold...")
        # Auto-threshold: find threshold that gives ~10% cliff compounds
        for t in [0.4, 0.3, 0.2, 0.1]:
            cliff_indices, cliff_info = identify_cliff_pairs_euclidean(
                X_concat, labels, t, activity_k)
            if cliff_info['pct_cliff_compounds'] >= 5:
                print(f"  Auto-threshold {t}: {cliff_info['n_cliff_compounds']} cliffs "
                      f"({cliff_info['pct_cliff_compounds']:.1f}%)")
                break

    if not cliff_indices:
        return {'error': 'no_cliffs_found', 'cliff_info': cliff_info}

    # Get hybrid blueprint
    budget = kwargs.get('budget', 100)
    try:
        search = get_precomputed_search(
            reps, labels, groups, budget=budget,
            aug_condition='uncapped', augmentations=data.get('augmentations'),
            speed=search_speed)
        bp = search.best_overall['blueprint']
    except Exception as e:
        print(f"  Hybrid search failed: {e}")
        return {'error': str(e), 'cliff_info': cliff_info}

    # CV evaluation: per-rep baselines + hybrid on cliff vs non-cliff
    sample_X = next(iter(reps.values()))

    # Per-rep baselines (best single rep)
    best_rep_cliff_score = None
    best_rep_name = None
    per_rep_results = {}

    for rep_name, X_rep in reps.items():
        cliff_scores = []
        noncliff_scores = []
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            rf = _rf_factory()
            rf.fit(X_tr, labels[train_idx])
            y_pred = rf.predict(X_va)

            val_cliff = [i for i, vi in enumerate(val_idx) if vi in cliff_indices]
            val_noncliff = [i for i, vi in enumerate(val_idx) if vi not in cliff_indices]

            if len(val_cliff) >= 2:
                cliff_scores.append(metric_fn(labels[val_idx[val_cliff]], y_pred[val_cliff]))
            if len(val_noncliff) >= 2:
                noncliff_scores.append(metric_fn(labels[val_idx[val_noncliff]], y_pred[val_noncliff]))

        per_rep_results[rep_name] = {
            'cliff_score': float(np.mean(cliff_scores)) if cliff_scores else float('nan'),
            'noncliff_score': float(np.mean(noncliff_scores)) if noncliff_scores else float('nan'),
        }

    # Hybrid evaluation
    hybrid_cliff_scores = []
    hybrid_noncliff_scores = []
    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
        train_reps_h = {k: v[train_idx] for k, v in reps.items()}
        val_reps_h = {k: v[val_idx] for k, v in reps.items()}
        aug_tr = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
        aug_va = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

        try:
            X_tr_h, X_va_h, fi = build_fold_hybrid(
                train_reps_h, val_reps_h, labels[train_idx], bp,
                augmentations_train=aug_tr, augmentations_val=aug_va)
        except Exception:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_h)
        X_va_s = scaler.transform(X_va_h)

        rf = _rf_factory()
        rf.fit(X_tr_s, labels[train_idx])
        y_pred = rf.predict(X_va_s)

        val_cliff = [i for i, vi in enumerate(val_idx) if vi in cliff_indices]
        val_noncliff = [i for i, vi in enumerate(val_idx) if vi not in cliff_indices]

        if len(val_cliff) >= 2:
            hybrid_cliff_scores.append(metric_fn(labels[val_idx[val_cliff]], y_pred[val_cliff]))
        if len(val_noncliff) >= 2:
            hybrid_noncliff_scores.append(metric_fn(labels[val_idx[val_noncliff]], y_pred[val_noncliff]))

    results = {
        'domain': domain,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'cliff_info': convert_numpy(cliff_info),
        'per_rep': convert_numpy(per_rep_results),
        'hybrid': {
            'cliff_score': float(np.mean(hybrid_cliff_scores)) if hybrid_cliff_scores else float('nan'),
            'noncliff_score': float(np.mean(hybrid_noncliff_scores)) if hybrid_noncliff_scores else float('nan'),
        },
    }

    # Summary
    h_cliff = results['hybrid']['cliff_score']
    best_rep_cliff = max((r['cliff_score'] for r in per_rep_results.values()
                          if not np.isnan(r['cliff_score'])), default=float('nan'))
    if not np.isnan(h_cliff) and not np.isnan(best_rep_cliff):
        results['hybrid_cliff_delta'] = float(h_cliff - best_rep_cliff)
        print(f"\n  Hybrid cliff {metric_name}: {h_cliff:.4f} "
              f"(best rep: {best_rep_cliff:.4f}, delta: {h_cliff - best_rep_cliff:+.4f})")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S12: Cross-Domain Activity Cliffs')
    parser.add_argument('--sim-threshold', type=float, default=0.55)
    parser.add_argument('--activity-threshold', type=float, default=2.0)
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset,
                  sim_threshold=args.sim_threshold,
                  activity_k=args.activity_threshold)
    save_experiment_results(results, args.output, 's12_cross_domain_cliffs', args.dataset)
