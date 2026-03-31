#!/usr/bin/env python3
"""S13: Clustering Quality & Retrieval Precision.

Paper: Similarity
If hybrid distance is a better metric, it should produce better clusters
(members have similar labels) and better retrieval (query → similar-activity
neighbors). Tests this across per-rep, concat, PCA-matched, and hybrid spaces.

Two sub-analyses:
  13a. Clustering: k-means label coherence (within-cluster label variance)
  13b. Retrieval: mean |y_query - y_neighbor| for k-NN retrieval

Usage:
    python s13_clustering_retrieval.py --dataset logd --mode intermediary
    python s13_clustering_retrieval.py --dataset flab_thermo_tresanco --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_hybrid_features_and_weights,
    compute_feature_importance,
    get_precomputed_search,
)
from kirby.hybrid import create_hybrid, apply_feature_selection


# =========================================================================
# 13a: CLUSTERING QUALITY
# =========================================================================

def _is_classification(y):
    """Heuristic: classification if <=20 unique values and integer-like."""
    unique = np.unique(y)
    return len(unique) <= 20 and np.allclose(y, np.round(y))


def _cluster_label_coherence(X, y, n_clusters_list=(3, 5, 10)):
    """Cluster in X-space, measure label coherence.

    For regression: 1 - mean(within-cluster std(y)) / std(y_global).
    For classification: mean class purity (majority class fraction per cluster).
    Higher = clusters group similar-label samples together.
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    is_clf = _is_classification(y)

    if not is_clf:
        y_std_global = np.std(y)
        if y_std_global < 1e-10:
            return {}

    results = {}
    for n_clusters in n_clusters_list:
        if n_clusters >= len(y):
            continue
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        cluster_labels = km.fit_predict(X_s)

        if is_clf:
            # Classification: mean purity (majority class fraction per cluster)
            purities = []
            for c in range(n_clusters):
                mask = cluster_labels == c
                if mask.sum() >= 1:
                    counts = np.bincount(y[mask].astype(int))
                    purities.append(counts.max() / mask.sum())
            mean_purity = float(np.mean(purities)) if purities else 0.0
            label_coherence = mean_purity
        else:
            # Regression: within-cluster std ratio
            cluster_stds = []
            for c in range(n_clusters):
                mask = cluster_labels == c
                if mask.sum() >= 2:
                    cluster_stds.append(np.std(y[mask]))
            mean_cluster_std = np.mean(cluster_stds) if cluster_stds else y_std_global
            label_coherence = 1.0 - mean_cluster_std / y_std_global

        # Silhouette (geometric quality)
        try:
            sil = silhouette_score(X_s, cluster_labels,
                                   sample_size=min(2000, len(y)))
        except Exception:
            sil = float('nan')

        result_entry = {
            'label_coherence': float(label_coherence),
            'silhouette': float(sil),
        }
        if is_clf:
            result_entry['mean_purity'] = mean_purity
        else:
            result_entry['mean_cluster_std'] = float(mean_cluster_std)
        results[n_clusters] = result_entry

    return results


# =========================================================================
# 13b: RETRIEVAL PRECISION
# =========================================================================

def _retrieval_quality(X, y, k=5, importance_weights=None):
    """k-NN retrieval: mean |y_query - y_neighbor| across all queries.

    Lower = retrieved neighbors have more similar labels = better metric space.
    If importance_weights provided, pre-weight features.
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    if importance_weights is not None:
        w = np.sqrt(np.maximum(importance_weights, 0))
        w = w / (w.sum() + 1e-10)
        X_s = X_s * w

    k_actual = min(k, len(y) - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1)
    nn.fit(X_s)
    _, indices = nn.kneighbors(X_s)

    # Exclude self (first neighbor)
    nbr_idx = indices[:, 1:]
    delta_y = np.abs(y[:, None] - y[nbr_idx])
    mean_delta = float(np.mean(delta_y))

    # For classification: precision@k (fraction of neighbors with same label)
    is_clf = len(np.unique(y)) <= 10 and np.issubdtype(np.asarray(y).dtype, np.integer)
    precision_at_k = None
    if is_clf:
        same_label = (y[:, None] == y[nbr_idx]).astype(float)
        precision_at_k = float(np.mean(same_label))

    return {
        'mean_delta_y': mean_delta,
        'precision_at_k': precision_at_k,
    }


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """S13: Clustering quality & retrieval precision."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S13: CLUSTERING & RETRIEVAL")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    search_speed = config['search_speed']
    budget = kwargs.get('budget', 100)
    k_retrieval = kwargs.get('k_retrieval', 5)
    n_clusters_list = kwargs.get('n_clusters_list', [3, 5, 10])

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    total_dim = X_concat.shape[1]

    # Build spaces to compare
    spaces = {}

    # Per-rep (top 3 by dimension)
    sorted_reps = sorted(reps.items(), key=lambda x: x[1].shape[1], reverse=True)
    for rname, X_rep in sorted_reps[:3]:
        spaces[f'rep_{rname}'] = X_rep

    # Concat
    spaces['concat'] = X_concat

    # PCA-matched (same dim as hybrid budget)
    pca_dim = min(budget, total_dim, len(labels) - 1)
    if pca_dim >= 2:
        X_pca = PCA(n_components=pca_dim, random_state=42).fit_transform(
            StandardScaler().fit_transform(X_concat))
        spaces['pca_matched'] = X_pca

    # Hybrid — use held-out split to avoid label leakage:
    # fit feature selection on 80% of data, transform ALL data for analysis.
    try:
        search = get_precomputed_search(
            reps, labels, groups, budget=budget,
            aug_condition='none', speed=config.get('search_speed', 'fast'),
            precomputed=data.get('precomputed_searches'))
        bp = search.best_overall['blueprint']

        from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
        if groups is not None and len(np.unique(groups)) >= 2:
            _splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            fit_idx, _ = next(_splitter.split(labels, labels, groups=groups))
        else:
            _splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            fit_idx, _ = next(_splitter.split(labels, labels))
        fit_reps = {k: v[fit_idx] for k, v in reps.items()}
        _, fi = create_hybrid(
            fit_reps, labels[fit_idx],
            allocation_method=bp.allocation_method,
            importance_method=bp.importance_method,
            internal_evaluator=bp.internal_evaluator,
            budget=bp.budget,
        )
        X_hybrid = apply_feature_selection(reps, fi)
        spaces['hybrid'] = X_hybrid
    except Exception as e:
        print(f"  Hybrid creation failed: {e}")

    # Importance weights for weighted retrieval
    # compute_feature_importance expects {rep_name: array} dict, not raw ndarray
    X_concat_s = StandardScaler().fit_transform(X_concat)
    imp_dict = compute_feature_importance(
        {'concat': X_concat_s}, labels, method='lightgbm_gain')
    imp_concat = imp_dict.get('concat', np.ones(X_concat.shape[1]))

    print(f"  Spaces: {list(spaces.keys())}")

    # --- 13a: Clustering ---
    print("\n  13a: Clustering quality")
    clustering_results = {}
    for space_name, X in spaces.items():
        cluster_res = _cluster_label_coherence(X, labels, n_clusters_list)
        clustering_results[space_name] = cluster_res
        if cluster_res:
            best_k = max(cluster_res, key=lambda k: cluster_res[k]['label_coherence'])
            print(f"    {space_name}: best coherence={cluster_res[best_k]['label_coherence']:.4f} "
                  f"(k={best_k}), silhouette={cluster_res[best_k]['silhouette']:.4f}")

    # --- 13b: Retrieval ---
    print("\n  13b: Retrieval quality")
    retrieval_results = {}
    for space_name, X in spaces.items():
        # Unweighted
        ret_uw = _retrieval_quality(X, labels, k=k_retrieval)

        # Weighted (only for concat and hybrid)
        ret_w = None
        if space_name == 'concat':
            ret_w = _retrieval_quality(X, labels, k=k_retrieval,
                                       importance_weights=imp_concat)
        elif space_name == 'hybrid' and 'hybrid' in spaces:
            # Importance on hybrid features
            imp_h_dict = compute_feature_importance(
                {'hybrid': StandardScaler().fit_transform(X)}, labels, method='lightgbm_gain')
            imp_h = imp_h_dict.get('hybrid', np.ones(X.shape[1]))
            ret_w = _retrieval_quality(X, labels, k=k_retrieval,
                                       importance_weights=imp_h)

        retrieval_results[space_name] = {
            'unweighted': ret_uw,
            'weighted': ret_w,
        }
        w_str = f", weighted={ret_w['mean_delta_y']:.4f}" if ret_w else ""
        print(f"    {space_name}: mean_delta_y={ret_uw['mean_delta_y']:.4f}{w_str}")

    results = {
        'clustering': convert_numpy(clustering_results),
        'retrieval': convert_numpy(retrieval_results),
        'config': {
            'budget': budget,
            'k_retrieval': k_retrieval,
            'n_clusters_list': n_clusters_list,
            'spaces': list(spaces.keys()),
        },
    }

    # Summary: does hybrid win on label coherence AND retrieval?
    if 'hybrid' in clustering_results and 'concat' in clustering_results:
        h_coh = max((v['label_coherence'] for v in clustering_results['hybrid'].values()),
                    default=float('nan'))
        c_coh = max((v['label_coherence'] for v in clustering_results['concat'].values()),
                    default=float('nan'))
        results['summary'] = {
            'hybrid_clustering_wins': h_coh > c_coh if not any(
                np.isnan(x) for x in [h_coh, c_coh]) else None,
            'hybrid_retrieval_wins': (
                retrieval_results.get('hybrid', {}).get('unweighted', {}).get('mean_delta_y', 1e10) <
                retrieval_results.get('concat', {}).get('unweighted', {}).get('mean_delta_y', 1e10)
            ),
        }
        print(f"\n  Hybrid vs concat: clustering={'win' if results['summary']['hybrid_clustering_wins'] else 'loss'}, "
              f"retrieval={'win' if results['summary']['hybrid_retrieval_wins'] else 'loss'}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S13: Clustering & Retrieval')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's13_clustering_retrieval', args.dataset)
