#!/usr/bin/env python3
"""S13: Clustering Quality & Retrieval Precision.

Paper: Similarity
If hybrid distance is a better metric, it should produce better clusters
(members have similar labels) and better retrieval (query → similar-activity
neighbors). Tests this across per-rep, concat, PCA-matched, and hybrid spaces.

Two sub-analyses:
  13a. Clustering: k-means label coherence (regression: 1 - within-cluster
       std(y)/std(y); classification: mean cluster majority-class purity).
       Higher = better. Silhouette is recorded but NOT used for the win test.
  13b. Retrieval: mean |y_query - y_neighbor| (regression) or 1 - precision@k
       (classification) for k-NN retrieval. Lower = better.

Both metrics are computed identically for hybrid and every baseline (same KMeans
config, same k-list), and their win tests respect the correct direction.
Empirically (2026-06-03) this is a MIXED finding: the hybrid tends to WIN
retrieval but LOSE clustering coherence on the powered antibody sets -- a real
result, not a metric bug. `summary.metrics_agree` flags the divergence, and
`best_rep_retrieval` / `best_rep_coherence` name the (different) baseline rep
that each axis is compared against.

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
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_fitted_hybrid, get_fitted_pool_hybrid, extract_top3_base_reps,
    compute_feature_importance, get_domain_metric,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)


# =========================================================================
# 13a: CLUSTERING QUALITY
# =========================================================================

def _cluster_label_coherence(X, y, n_clusters_list=(3, 5, 10), is_clf=False):
    """Cluster in X-space, measure label coherence.

    For regression: 1 - mean(within-cluster std(y)) / std(y_global).
    For classification: mean class purity (majority class fraction per cluster).
    Higher = clusters group similar-label samples together.
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    if not is_clf:
        y_std_global = np.std(y)
        if y_std_global < 1e-10:
            return {}

    results = {}
    for n_clusters in n_clusters_list:
        if n_clusters >= len(y):
            continue
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X_s)

        if is_clf:
            # Classification: mean purity (majority class fraction per cluster)
            if (y < 0).any():
                return {}
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
                                   sample_size=min(2000, len(y)),
                                   random_state=42)
        except Exception as exc:
            # N9: log so a NaN silhouette reads as "silhouette threw" (e.g.
            # single-cluster labelling) rather than a genuine value.
            print(f"      silhouette failed: {type(exc).__name__}: {exc}")
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

def _retrieval_quality(X, y, k=5, importance_weights=None, is_clf=False):
    """k-NN retrieval quality.

    Regression: mean |y_query - y_neighbor| (lower = better).
    Classification: precision@k (fraction of same-class neighbors, higher = better).
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

    if is_clf:
        same_label = (y[:, None] == y[nbr_idx]).astype(float)
        precision_at_k = float(np.mean(same_label))
        # For classification, mean_delta_y is fraction of different-class neighbors
        mean_delta = 1.0 - precision_at_k
    else:
        delta_y = np.abs(y[:, None] - y[nbr_idx])
        mean_delta = float(np.mean(delta_y))
        precision_at_k = None

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
    task_type = data['task_type']
    domain = data.get('domain', 'molecular')
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    search_speed = config['search_speed']
    budget = kwargs.get('budget', 100)
    k_retrieval = kwargs.get('k_retrieval', 5)
    n_clusters_list = kwargs.get('n_clusters_list', [3, 5, 10])

    # Use distinct comprehension vars (rep_k / aug_k) for clarity; in Python 3
    # comprehension vars don't leak, but the audit (OTHER-32) flagged the reuse
    # of `k` across the function as confusing. Keep them locally scoped + named.
    X_concat = np.hstack([reps[rep_k] for rep_k in sorted(reps.keys())])
    augmentations = data.get('augmentations')
    if augmentations:
        X_concat = np.hstack([X_concat] + [augmentations[aug_k] for aug_k in sorted(augmentations.keys())])
    total_dim = X_concat.shape[1]

    # Build spaces to compare
    spaces = {}

    # All per-rep spaces (top-3 selected after evaluation)
    for rname, X_rep in reps.items():
        spaces[f'rep_{rname}'] = X_rep

    # PCA-matched (same dim as hybrid budget)
    pca_dim = min(budget, total_dim, len(labels) - 1)
    if pca_dim >= 2:
        X_pca = PCA(n_components=pca_dim, random_state=42).fit_transform(
            StandardScaler().fit_transform(X_concat))
        spaces['pca_matched'] = X_pca

    # Hybrid — NOTE: feature selection uses all labels (supervised), so hybrid
    # clustering/retrieval metrics have a tautological advantage over unsupervised
    # baselines (PCA, raw reps). Results should be interpreted as "how well does
    # the hybrid space organise labels" not "will this generalise to unseen data".
    hybrid_error = None
    try:
        X_hybrid, fi, search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='knn', budget=None,
            augmentations=augmentations,
            speed=config.get('search_speed', 'full'),
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset_name, distance=True)
        spaces['hybrid'] = X_hybrid
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    except Exception as e:
        # Record the failure explicitly so a missing 'hybrid'/'summary' block is
        # distinguishable from a dataset that was never run (see gb_* genomic
        # rows that produced no summary).
        hybrid_error = f"{type(e).__name__}: {e}"
        print(f"  Hybrid creation failed: {hybrid_error}")
        top3 = None

    print(f"  Spaces: {list(spaces.keys())}")

    # --- 13a: Clustering ---
    print("\n  13a: Clustering quality")
    clustering_results = {}
    for space_name, X in spaces.items():
        cluster_res = _cluster_label_coherence(X, labels, n_clusters_list,
                                               is_clf=is_clf)
        clustering_results[space_name] = cluster_res
        if cluster_res:
            best_k = max(cluster_res, key=lambda k: cluster_res[k]['label_coherence'])
            print(f"    {space_name}: best coherence={cluster_res[best_k]['label_coherence']:.4f} "
                  f"(k={best_k}), silhouette={cluster_res[best_k]['silhouette']:.4f}")

    # --- 13b: Retrieval ---
    print("\n  13b: Retrieval quality")
    retrieval_results = {}
    for space_name, X in spaces.items():
        ret_uw = _retrieval_quality(X, labels, k=k_retrieval, is_clf=is_clf)

        # Weighted retrieval for hybrid only
        ret_w = None
        if space_name == 'hybrid':
            imp_h_dict = compute_feature_importance(
                {'hybrid': StandardScaler().fit_transform(X)}, labels, method='lightgbm_gain')
            imp_h = imp_h_dict.get('hybrid', np.ones(X.shape[1]))
            ret_w = _retrieval_quality(X, labels, k=k_retrieval,
                                       importance_weights=imp_h,
                                       is_clf=is_clf)

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
    if top3:
        results['top3_base_reps'] = list(top3)
    if hybrid_error is not None:
        results['hybrid_error'] = hybrid_error

    # Summary: hybrid vs top-3 per-rep baselines
    # Top-3 by retrieval (lower mean_delta_y = better)
    rep_retrieval = {k: v['unweighted']['mean_delta_y']
                     for k, v in retrieval_results.items() if k.startswith('rep_')}
    top3_retrieval = sorted(rep_retrieval.items(), key=lambda x: x[1])[:3]
    # Top-3 by clustering coherence (higher = better)
    rep_clustering = {}
    for k, v in clustering_results.items():
        if k.startswith('rep_') and v:
            rep_clustering[k] = max(val['label_coherence'] for val in v.values())
    top3_clustering = sorted(rep_clustering.items(), key=lambda x: x[1], reverse=True)[:3]

    results['top3_retrieval'] = top3_retrieval
    results['top3_clustering'] = top3_clustering

    if 'hybrid' in clustering_results and 'hybrid' in retrieval_results:
        h_coh = max((v['label_coherence'] for v in clustering_results['hybrid'].values()),
                    default=float('nan'))
        h_ret = retrieval_results['hybrid']['unweighted']['mean_delta_y']
        best_rep_ret_name, best_rep_ret_val = top3_retrieval[0] if top3_retrieval else ('?', 1e10)
        best_rep_coh_name, best_rep_coh_val = top3_clustering[0] if top3_clustering else ('?', 0)

        results['summary'] = {
            'hybrid_coherence': h_coh,
            'hybrid_retrieval': h_ret,
            'best_rep_coherence': (best_rep_coh_name, best_rep_coh_val),
            'best_rep_retrieval': (best_rep_ret_name, best_rep_ret_val),
            'hybrid_clustering_wins': h_coh > best_rep_coh_val,
            'hybrid_retrieval_wins': h_ret < best_rep_ret_val,
            # The two axes have different best-baseline reps and frequently
            # disagree (hybrid tends to win retrieval, lose clustering). Flag the
            # divergence so the mixed result is not read as a clean win/loss.
            'metrics_agree': (h_coh > best_rep_coh_val) == (h_ret < best_rep_ret_val),
        }
        print(f"\n  Top-3 baselines (retrieval): "
              + ", ".join(f"{n}={v:.4f}" for n, v in top3_retrieval))
        print(f"  Top-3 baselines (clustering): "
              + ", ".join(f"{n}={v:.4f}" for n, v in top3_clustering))
        print(f"  Hybrid: coherence={h_coh:.4f} vs best_rep={best_rep_coh_val:.4f} "
              f"({'win' if h_coh > best_rep_coh_val else 'loss'}), "
              f"retrieval={h_ret:.4f} vs best_rep={best_rep_ret_val:.4f} "
              f"({'win' if h_ret < best_rep_ret_val else 'loss'})")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S13: Clustering & Retrieval')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's13_clustering_retrieval', args.dataset)
