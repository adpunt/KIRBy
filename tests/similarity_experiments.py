#!/usr/bin/env python3
"""
Similarity Experiments for Hybrid Molecular Representations

Investigates WHY the similarity assumption fails in standard QSAR (equal-weight
Euclidean distance poorly tracks property similarity) and HOW hybrid methodology
fixes it (importance-weighted distance, dimensionality reduction via feature
selection). Bridges the hybrid methodology paper to noise mitigation by showing
that standard noise detection flags activity cliffs as false positives, which
hybrid distances reduce.

Paper thesis: "When Neighbours Lie: Property-Aware Similarity for Robust
Molecular Property Prediction"

Experiments:
  1. Similarity-Activity Correlation Baseline — how poorly standard distances work
  2. Hybrid Property-Aware Similarity — the fix via find_best_hybrid + weighting
  3. Feature Concentration Analysis — WHY hybrid similarity works (MI density)
  4. Prediction Performance on Activity Cliffs — RMSE_cliff (MoleculeACE-style)
  5. kNN Prediction with Hybrid Distance — downstream task validation
  6. Applicability Domain — hybrid distance tracks prediction uncertainty
  7. Noise Detection Bridge (Supplementary) — standard detectors flag cliffs as noise FPs
  8. Optimal Dimension for Similarity vs Prediction — different optima
  9. Cross-Dataset Summary — consolidate findings across datasets

Datasets: logd, caco2, herg, qm9 (via load_dataset from molecular wrapper)

Usage:
    python tests/similarity_experiments.py --mode quick --experiments 1 2 4 5 8
    python tests/similarity_experiments.py --mode full --experiments 1 2 3 4 5 6 7 8 9
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.ensemble import RandomForestRegressor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from kirby.find_best_hybrid import find_best_hybrid, SearchResult
from kirby.hybrid import create_hybrid, apply_feature_selection, compute_feature_importance
from kirby.utils.noise_mitigation_methods import (
    compute_distance_matrix, get_k_neighbors_from_distances,
    ActivityCliffDetector, KNNConsistencyFilter, NeighborhoodConsensusFilter,
)
from kirby.utils.noise_estimation import (
    ActivityCliffAwareEstimator, ConformalEstimator,
)

# Import from find_best_hybrid_molecular
sys.path.insert(0, str(Path(__file__).parent))
from find_best_hybrid_molecular import load_dataset, get_representations, assign_scaffold_groups

warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING (reuse pattern from complexity experiments)
# =============================================================================

def load_data(dataset_name: str, rep_preset: str = 'fast') -> dict:
    """Load dataset, generate representations, assign scaffold groups.

    Returns dict with keys: smiles, labels, reps, groups, n_scaffolds, task_type.
    """
    smiles, labels, task_type = load_dataset(dataset_name)
    reps = get_representations(smiles, rep_preset)
    groups, n_scaffolds = assign_scaffold_groups(smiles)

    print(f"  {dataset_name}: {len(smiles)} samples, {n_scaffolds} scaffolds, "
          f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

    return {
        'smiles': smiles,
        'labels': labels,
        'reps': reps,
        'groups': groups,
        'n_scaffolds': n_scaffolds,
        'task_type': task_type,
    }


# =============================================================================
# SHARED METRICS
# =============================================================================

def similarity_activity_correlation(D: np.ndarray, y: np.ndarray,
                                    n_sample: int = 5000) -> Tuple[float, float]:
    """Spearman correlation between pairwise distance and |delta_y|.

    Positive = good (large distance → large activity difference means the
    representation preserves the structure-activity relationship).
    Returns (rho, p_value).
    """
    n = len(y)
    # Extract upper triangle indices
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


def compute_sali(D: np.ndarray, y: np.ndarray,
                 eps: float = 1e-6) -> Dict[str, Any]:
    """Structure-Activity Landscape Index: |y_i - y_j| / (1 - sim(i,j)).

    Uses sim = 1/(1+d). Returns per-sample max SALI and summary stats.
    """
    n = len(y)
    # Similarity from distance: sim = 1/(1+d)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0)  # exclude self-pairs

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


def violation_rate(D: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Fraction of k-NN pairs where |delta_y| > median(|delta_y|) dataset-wide.

    High = neighbors are NOT activity-similar.
    """
    n = len(y)
    k = min(k, n - 1)

    indices, _ = get_k_neighbors_from_distances(D, k)

    # Compute all pairwise |delta_y| to get dataset median
    triu_i, triu_j = np.triu_indices(n, k=1)
    if len(triu_i) > 50000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(triu_i), 50000, replace=False)
        all_delta = np.abs(y[triu_i[sample_idx]] - y[triu_j[sample_idx]])
    else:
        all_delta = np.abs(y[triu_i] - y[triu_j])
    median_delta = np.median(all_delta)

    # Check k-NN pairs
    n_violations = 0
    n_total = 0
    for i in range(n):
        for j_idx in range(k):
            j = indices[i, j_idx]
            if np.abs(y[i] - y[j]) > median_delta:
                n_violations += 1
            n_total += 1

    return float(n_violations / n_total) if n_total > 0 else 0.0


def identify_cliff_compounds(ecfp4: np.ndarray, y: np.ndarray,
                             tanimoto_threshold: float = 0.55,
                             activity_k: float = 2.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Identify activity cliff compounds using ECFP4 Tanimoto similarity.

    Definition based on Bajorath standard for ECFP4 (Stumpfe & Bajorath 2019):
      - Structural similarity: Tanimoto(ECFP4) >= tanimoto_threshold (default 0.55)
      - Activity difference: |Δy| > activity_k * MAD(y)

    Note: MoleculeACE uses Tc>=0.9 for curated binding-affinity datasets.
    For diverse property datasets (logD, logP), Tc>=0.55 is standard for ECFP4.
    The activity threshold adapts to each dataset via MAD(y).

    Returns:
        cliff_mask: boolean array, True for cliff compounds
        info: dict with thresholds, counts, per-compound cliff pair counts
    """
    n = len(y)

    # Adaptive activity threshold: k * MAD(y)
    mad_y = float(np.median(np.abs(y - np.median(y))))
    activity_threshold = activity_k * mad_y

    # Tanimoto similarity from binary ECFP4 (Tanimoto = 1 - Jaccard distance)
    jaccard_dist = scipy_cdist(ecfp4, ecfp4, metric='jaccard')
    tanimoto_sim = 1.0 - jaccard_dist
    np.fill_diagonal(tanimoto_sim, 0)

    # Activity difference matrix
    delta_y = np.abs(y[:, None] - y[None, :])

    # Cliff pair mask: high similarity AND large activity difference
    cliff_pair_mask = (tanimoto_sim >= tanimoto_threshold) & (delta_y > activity_threshold)

    # Cliff compounds: involved in at least one cliff pair
    cliff_mask = cliff_pair_mask.any(axis=1)

    # Per-compound cliff pair count (how many cliff partners each molecule has)
    cliff_pair_counts = cliff_pair_mask.sum(axis=1)

    n_cliff_pairs = int(cliff_pair_mask.sum()) // 2  # symmetric
    n_cliff_compounds = int(cliff_mask.sum())

    info = {
        'tanimoto_threshold': tanimoto_threshold,
        'activity_k': activity_k,
        'activity_threshold': activity_threshold,
        'mad_y': mad_y,
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': float(100.0 * n_cliff_compounds / n),
        'cliff_pair_counts': cliff_pair_counts,  # per-compound array
    }

    return cliff_mask, info



def get_hybrid_features_and_weights(
    reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    search_speed: str = 'fast',
    weight_method: str = 'lightgbm_gain',
    budget: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, SearchResult]:
    """Run find_best_hybrid and return hybrid features + flat importance weights.

    Returns (X_hybrid, weights_flat, search_result).
    """
    kwargs = dict(
        base_reps=reps,
        labels=labels,
        groups=groups,
        speed=search_speed,
        return_fitted=True,
        weight_method=weight_method,
        verbose=False,
        cache=True,
    )
    if budget is not None:
        kwargs['budgets'] = [budget]

    search = find_best_hybrid(**kwargs)
    fitted = search.fitted_result

    if fitted is None:
        raise RuntimeError("find_best_hybrid returned no fitted result")

    # Build hybrid feature matrix
    X_hybrid = apply_feature_selection(reps, _fitted_to_feature_info(fitted))

    # Build flat weight vector
    weights_flat = _flatten_weights(fitted)

    return X_hybrid, weights_flat, search


def _fitted_to_feature_info(fitted) -> dict:
    """Convert FittedResult to feature_info dict for apply_feature_selection."""
    feature_info = {}
    for rep_name, indices in fitted.selected_indices.items():
        feature_info[rep_name] = {'selected_indices': indices}
    return feature_info


def _flatten_weights(fitted) -> np.ndarray:
    """Flatten per-rep feature_weights into a single array matching X_hybrid columns."""
    parts = []
    for rep_name in fitted.selected_indices:
        if fitted.feature_weights and rep_name in fitted.feature_weights:
            parts.append(fitted.feature_weights[rep_name])
        else:
            # Uniform weights if no importance available
            parts.append(np.ones(len(fitted.selected_indices[rep_name])))
    return np.concatenate(parts)


def estimate_mutual_information_knn(X: np.ndarray, y: np.ndarray,
                                    k: int = 5) -> float:
    """Estimate I(X; Y) using KSG estimator (Kraskov et al. 2004)."""
    from scipy.special import digamma

    n_samples = X.shape[0]
    k = min(k, n_samples - 1)

    X_scaled = StandardScaler().fit_transform(X)
    y_scaled = (y - y.mean()) / (y.std() + 1e-10)

    XY = np.column_stack([X_scaled, y_scaled.reshape(-1, 1)])

    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev')
    nn_joint.fit(XY)
    distances_joint, _ = nn_joint.kneighbors(XY)
    eps = distances_joint[:, -1]

    nn_x = NearestNeighbors(metric='chebyshev')
    nn_x.fit(X_scaled)

    nn_y = NearestNeighbors(metric='chebyshev')
    nn_y.fit(y_scaled.reshape(-1, 1))

    n_x = np.zeros(n_samples)
    n_y = np.zeros(n_samples)

    for i in range(n_samples):
        n_x[i] = len(nn_x.radius_neighbors([X_scaled[i]], radius=eps[i],
                      return_distance=False)[0]) - 1
        n_y[i] = len(nn_y.radius_neighbors([y_scaled[i:i + 1]], radius=eps[i],
                      return_distance=False)[0]) - 1

    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n_samples)
    return max(0, mi)


def estimate_intrinsic_dim_mle(X: np.ndarray, k: int = 10) -> float:
    """MLE estimator of intrinsic dimensionality (Levina-Bickel 2004)."""
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # remove self

    estimates = []
    for i in range(n_samples):
        dists = distances[i]
        if dists[-1] > 0:
            log_ratios = np.log(dists[-1] / dists[:-1])
            if np.sum(log_ratios) > 0:
                estimates.append((k - 1) / np.sum(log_ratios))

    return np.mean(estimates) if estimates else np.nan


def compute_effective_rank(X: np.ndarray) -> float:
    """Effective rank: exp(entropy of normalised singular values)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n, d = X_scaled.shape

    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(500, n - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(X_scaled)
        s = svd.singular_values_
    else:
        _, s, _ = np.linalg.svd(X_scaled, full_matrices=False)

    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


# =============================================================================
# EXPERIMENT 1: SIMILARITY-ACTIVITY CORRELATION BASELINE
# =============================================================================

def run_similarity_baseline(data: dict, n_folds: int,
                            n_sample: int = 5000) -> Dict[str, Any]:
    """Establish the problem: how poorly standard distances correlate with
    property similarity across representations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: SIMILARITY-ACTIVITY CORRELATION BASELINE")
    print("=" * 70)

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


# =============================================================================
# EXPERIMENT 2: HYBRID PROPERTY-AWARE SIMILARITY
# =============================================================================

def run_hybrid_similarity(data: dict, n_folds: int,
                          search_speed: str = 'fast',
                          weight_methods: List[str] = None,
                          n_sample: int = 5000,
                          shared_search: Tuple = None) -> Dict[str, Any]:
    """The fix: find_best_hybrid + importance-weighted distance improves SAC."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: HYBRID PROPERTY-AWARE SIMILARITY")
    print("=" * 70)

    if weight_methods is None:
        weight_methods = ['lightgbm_gain']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'weight_methods': {}, 'per_rep': {}}

    # Best individual rep baseline (the real-world comparison)
    best_rep_sac = -1  # SAC > 0 is good; we want highest
    best_rep_name = None
    for rep_name, X in reps.items():
        D_rep = compute_distance_matrix(X, standardize=True)
        sac_rep, _ = similarity_activity_correlation(D_rep, labels, n_sample)
        viol_rep = violation_rate(D_rep, labels, k=5)
        results['per_rep'][rep_name] = {
            'dim': X.shape[1], 'sac_rho': sac_rep, 'violation_rate': viol_rep,
        }
        if best_rep_name is None or sac_rep > best_rep_sac:
            best_rep_sac = sac_rep
            best_rep_name = rep_name
        print(f"  {rep_name}: SAC={sac_rep:.4f}, viol={viol_rep:.3f}, dim={X.shape[1]}")

    results['best_individual_rep'] = {
        'name': best_rep_name, 'sac_rho': best_rep_sac,
        'dim': reps[best_rep_name].shape[1],
    }
    print(f"\n  Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f})")

    # Also compute concat as an additional data point (not standard practice)
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    D_concat = compute_distance_matrix(X_concat, standardize=True)
    sac_concat, _ = similarity_activity_correlation(D_concat, labels, n_sample)
    viol_concat = violation_rate(D_concat, labels, k=5)

    results['concat'] = {
        'dim': X_concat.shape[1],
        'sac_rho': sac_concat,
        'violation_rate': viol_concat,
    }
    print(f"  Concat (reference only): SAC={sac_concat:.4f}, viol={viol_concat:.3f}, "
          f"dim={X_concat.shape[1]}")

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

    # Hybrid unweighted (same for all weight methods — only compute once)
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


# =============================================================================
# EXPERIMENT 3: FEATURE CONCENTRATION ANALYSIS
# =============================================================================

def run_feature_concentration(data: dict, search_speed: str = 'fast',
                              weight_methods: List[str] = None,
                              shared_search: Tuple = None) -> Dict[str, Any]:
    """Explain WHY hybrid similarity works: higher MI density, better effective
    rank distribution in selected features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: FEATURE CONCENTRATION ANALYSIS")
    print("=" * 70)

    if weight_methods is None:
        weight_methods = ['lightgbm_gain']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'per_rep': {}, 'gini': {}}

    # Get hybrid selection
    if shared_search is not None:
        X_hybrid, weights_flat, search = shared_search
        fitted = search.fitted_result
        print(f"  Using shared search (dim={X_hybrid.shape[1]})")
    else:
        try:
            X_hybrid, weights_flat, search = get_hybrid_features_and_weights(
                reps, labels, groups, search_speed, weight_methods[0])
            fitted = search.fitted_result
        except Exception as e:
            print(f"  find_best_hybrid failed: {e}")
            return {'error': str(e)}

    rng = np.random.RandomState(42)

    for rep_name, X in reps.items():
        print(f"\n  {rep_name} (dim={X.shape[1]}):")
        rep_result = {}

        n_features = X.shape[1]
        n_samples = X.shape[0]

        # Full rep MI (use PCA-reduced for stability if high-dim)
        n_comp = min(50, n_features, n_samples - 1)
        X_scaled_full = StandardScaler().fit_transform(X)
        if n_features > 2000:
            from sklearn.decomposition import TruncatedSVD
            X_pca_full = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X_scaled_full)
        else:
            X_pca_full = PCA(n_components=n_comp).fit_transform(X_scaled_full)
        try:
            mi_full = estimate_mutual_information_knn(X_pca_full, labels, k=5)
        except Exception:
            mi_full = float('nan')
        rep_result['mi_full'] = mi_full
        rep_result['effective_rank_full'] = compute_effective_rank(X)

        # Selected features for this rep
        if fitted and rep_name in fitted.selected_indices:
            sel_idx = fitted.selected_indices[rep_name]
            n_selected = len(sel_idx)

            if n_selected > 0:
                X_sel = X[:, sel_idx]

                # MI of selected
                try:
                    mi_sel = estimate_mutual_information_knn(
                        StandardScaler().fit_transform(X_sel), labels, k=5)
                except Exception:
                    mi_sel = float('nan')
                rep_result['mi_selected'] = mi_sel
                rep_result['n_selected'] = n_selected
                rep_result['effective_rank_selected'] = compute_effective_rank(X_sel)

                # Random subset of same size
                rand_idx = rng.choice(n_features, n_selected, replace=False)
                X_rand = X[:, rand_idx]
                try:
                    mi_rand = estimate_mutual_information_knn(
                        StandardScaler().fit_transform(X_rand), labels, k=5)
                except Exception:
                    mi_rand = float('nan')
                rep_result['mi_random'] = mi_rand
                rep_result['effective_rank_random'] = compute_effective_rank(X_rand)

                # PCA to same dim
                n_pca_comp = min(n_selected, n_samples - 1)
                X_scaled_rep = StandardScaler().fit_transform(X)
                if n_features > 2000:
                    from sklearn.decomposition import TruncatedSVD
                    X_pca_match = TruncatedSVD(n_components=n_pca_comp, random_state=42).fit_transform(X_scaled_rep)
                else:
                    X_pca_match = PCA(n_components=n_pca_comp).fit_transform(X_scaled_rep)
                try:
                    mi_pca = estimate_mutual_information_knn(X_pca_match, labels, k=5)
                except Exception:
                    mi_pca = float('nan')
                rep_result['mi_pca'] = mi_pca
                rep_result['effective_rank_pca'] = compute_effective_rank(X_pca_match)

                # MI density = MI / n_features
                rep_result['mi_density_full'] = mi_full / n_features if n_features > 0 else 0
                rep_result['mi_density_selected'] = mi_sel / n_selected if n_selected > 0 else 0
                rep_result['mi_density_ratio'] = (
                    rep_result['mi_density_selected'] / rep_result['mi_density_full']
                    if rep_result['mi_density_full'] > 0 else float('nan')
                )

                print(f"    MI full={mi_full:.4f} ({n_features}d), "
                      f"selected={mi_sel:.4f} ({n_selected}d), "
                      f"random={mi_rand:.4f}, pca={mi_pca:.4f}")
                print(f"    MI density ratio (selected/full): "
                      f"{rep_result['mi_density_ratio']:.2f}x")
            else:
                print(f"    No features selected from this rep")
        else:
            print(f"    Not in hybrid selection")

        results['per_rep'][rep_name] = rep_result

    # Gini coefficient of importance scores per method
    for wm in weight_methods:
        print(f"\n  Gini coefficients ({wm}):")
        try:
            importance = compute_feature_importance(reps, labels, method=wm)
            for rep_name, scores in importance.items():
                scores_sorted = np.sort(np.abs(scores))
                n = len(scores_sorted)
                if n == 0 or scores_sorted.sum() == 0:
                    gini = 0.0
                else:
                    index = np.arange(1, n + 1)
                    gini = float((2 * np.sum(index * scores_sorted) /
                                 (n * np.sum(scores_sorted))) - (n + 1) / n)
                results['gini'].setdefault(wm, {})[rep_name] = gini
                print(f"    {rep_name}: Gini={gini:.3f}")
        except Exception as e:
            print(f"    {wm} failed: {e}")

    return results


# =============================================================================
# EXPERIMENT 4: REPRESENTATION-DEPENDENT ACTIVITY CLIFFS
# =============================================================================

def run_activity_cliff_analysis(data: dict, n_folds: int = 5,
                                search_speed: str = 'fast',
                                shared_search: Tuple = None,
                                tanimoto_threshold: float = 0.55,
                                activity_k: float = 2.0) -> Dict[str, Any]:
    """Prediction performance on activity cliff compounds.

    Uses representation-independent cliff definition (Bajorath ECFP4 standard):
      - ECFP4 Tanimoto >= tanimoto_threshold
      - |Δy| > activity_k * MAD(y)

    Reports RMSE_cliff, R²_cliff, RMSE_noncliff, R²_noncliff for:
      - Per-rep kNN baselines
      - Hybrid kNN (unweighted and weighted)
      - RF baseline
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: PREDICTION PERFORMANCE ON ACTIVITY CLIFFS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    # --- Identify cliff compounds using ECFP4 Tanimoto ---
    if 'ecfp4' not in reps:
        print("  ERROR: ecfp4 not in reps, cannot identify cliffs")
        return {'error': 'ecfp4 not available'}

    cliff_mask, cliff_info = identify_cliff_compounds(
        reps['ecfp4'], labels,
        tanimoto_threshold=tanimoto_threshold,
        activity_k=activity_k,
    )
    cliff_indices = set(np.where(cliff_mask)[0].tolist())

    print(f"  Cliff definition: Tanimoto>={tanimoto_threshold}, "
          f"|Δy|>{activity_k}*MAD(y) = {cliff_info['activity_threshold']:.3f}")
    print(f"  Cliff compounds: {cliff_info['n_cliff_compounds']} / {len(labels)} "
          f"({cliff_info['pct_cliff_compounds']:.1f}%)")
    print(f"  Cliff pairs: {cliff_info['n_cliff_pairs']}")

    results = {
        'cliff_definition': {
            'tanimoto_threshold': tanimoto_threshold,
            'activity_k': activity_k,
            'activity_threshold': cliff_info['activity_threshold'],
            'mad_y': cliff_info['mad_y'],
            'n_cliff_compounds': cliff_info['n_cliff_compounds'],
            'n_cliff_pairs': cliff_info['n_cliff_pairs'],
            'pct_cliff_compounds': cliff_info['pct_cliff_compounds'],
        },
        'per_rep': {},
        'hybrid': {},
    }

    # --- Get hybrid blueprint for per-fold hybrid construction ---
    hybrid_blueprint = None
    if shared_search is not None:
        X_hybrid, weights_flat, search = shared_search
        if search.best_overall:
            hybrid_blueprint = search.best_overall['blueprint']
        print(f"  Using shared search (dim={X_hybrid.shape[1]})")
    else:
        try:
            X_hybrid, weights_flat, search = get_hybrid_features_and_weights(
                reps, labels, groups, search_speed, 'lightgbm_gain')
            if search.best_overall:
                hybrid_blueprint = search.best_overall['blueprint']
        except Exception as e:
            print(f"  Hybrid search failed: {e}")

    # --- CV evaluation of prediction on cliff vs non-cliff ---
    gkf = GroupKFold(n_splits=n_folds)
    sample_X = next(iter(reps.values()))

    def _evaluate_condition(get_Xy_fn, pre_scaled=False):
        """Run CV for a condition, return cliff/noncliff metrics."""
        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_all, fold_rmse_cliff, fold_rmse_noncliff = [], [], []

        for train_idx, val_idx in gkf.split(sample_X, labels, groups=groups):
            y_train, y_val = labels[train_idx], labels[val_idx]
            try:
                X_tr, X_va = get_Xy_fn(train_idx, val_idx, y_train)
            except Exception:
                continue
            if X_tr is None:
                continue

            if pre_scaled:
                X_tr_s, X_va_s = X_tr, X_va
            else:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)

            model = KNeighborsRegressor(n_neighbors=min(5, len(y_train) - 1))
            model.fit(X_tr_s, y_train)
            y_pred = model.predict(X_va_s)

            # Overall
            fold_all.append(r2_score(y_val, y_pred))
            fold_rmse_all.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))

            # Stratify by cliff membership
            val_cliff = np.array([i for i, vi in enumerate(val_idx) if vi in cliff_indices])
            val_noncliff = np.array([i for i, vi in enumerate(val_idx) if vi not in cliff_indices])

            if len(val_cliff) > 1:
                fold_cliff.append(r2_score(y_val[val_cliff], y_pred[val_cliff]))
                fold_rmse_cliff.append(float(np.sqrt(mean_squared_error(
                    y_val[val_cliff], y_pred[val_cliff]))))
            if len(val_noncliff) > 1:
                fold_noncliff.append(r2_score(y_val[val_noncliff], y_pred[val_noncliff]))
                fold_rmse_noncliff.append(float(np.sqrt(mean_squared_error(
                    y_val[val_noncliff], y_pred[val_noncliff]))))

        if not fold_all:
            return None
        return {
            'r2_all': float(np.mean(fold_all)),
            'r2_cliff': float(np.mean(fold_cliff)) if fold_cliff else float('nan'),
            'r2_noncliff': float(np.mean(fold_noncliff)) if fold_noncliff else float('nan'),
            'rmse_all': float(np.mean(fold_rmse_all)),
            'rmse_cliff': float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan'),
            'rmse_noncliff': float(np.mean(fold_rmse_noncliff)) if fold_rmse_noncliff else float('nan'),
        }

    # --- Per-rep kNN baselines ---
    print("\n  Per-rep kNN baselines (k=5):")
    for rep_name, X_rep in reps.items():
        def _get_rep(tr, va, yt, _X=X_rep):
            return _X[tr], _X[va]
        res = _evaluate_condition(_get_rep)
        if res:
            results['per_rep'][rep_name] = res
            print(f"    {rep_name:<15} R²_all={res['r2_all']:.4f}  "
                  f"R²_cliff={res['r2_cliff']:.4f}  RMSE_cliff={res['rmse_cliff']:.4f}  "
                  f"R²_noncliff={res['r2_noncliff']:.4f}")

    # --- Hybrid kNN (unweighted) ---
    if hybrid_blueprint is not None:
        print("\n  Hybrid kNN (unweighted, k=5):")

        def _get_hybrid_uw(tr, va, yt):
            train_reps = {name: rep[tr] for name, rep in reps.items()}
            val_reps = {name: rep[va] for name, rep in reps.items()}
            X_tr_h, feat_info = create_hybrid(
                train_reps, yt,
                allocation_method=hybrid_blueprint.allocation_method,
                budget=hybrid_blueprint.budget,
                importance_method=hybrid_blueprint.importance_method,
            )
            X_va_h = apply_feature_selection(val_reps, feat_info)
            return X_tr_h, X_va_h

        res = _evaluate_condition(_get_hybrid_uw)
        if res:
            results['hybrid']['unweighted'] = res
            print(f"    hybrid_uw      R²_all={res['r2_all']:.4f}  "
                  f"R²_cliff={res['r2_cliff']:.4f}  RMSE_cliff={res['rmse_cliff']:.4f}  "
                  f"R²_noncliff={res['r2_noncliff']:.4f}")

        # --- Hybrid kNN (importance-weighted) ---
        print("  Hybrid kNN (weighted, k=5):")

        def _get_hybrid_w(tr, va, yt):
            train_reps = {name: rep[tr] for name, rep in reps.items()}
            val_reps = {name: rep[va] for name, rep in reps.items()}
            X_tr_h, feat_info = create_hybrid(
                train_reps, yt,
                allocation_method=hybrid_blueprint.allocation_method,
                budget=hybrid_blueprint.budget,
                importance_method=hybrid_blueprint.importance_method,
            )
            X_va_h = apply_feature_selection(val_reps, feat_info)
            imp = compute_feature_importance(train_reps, yt, method='lightgbm_gain')
            w_parts = []
            for rname in feat_info:
                if not isinstance(feat_info[rname], dict):
                    continue
                if 'selected_indices' not in feat_info[rname]:
                    continue
                idx = feat_info[rname]['selected_indices']
                if rname in imp:
                    w_parts.append(imp[rname][idx])
                else:
                    w_parts.append(np.ones(len(idx)))
            w = np.sqrt(np.maximum(np.concatenate(w_parts), 0))
            # Scale-then-weight (matches Exp 5 and compute_distance_matrix convention)
            scaler_w = StandardScaler()
            return scaler_w.fit_transform(X_tr_h) * w, scaler_w.transform(X_va_h) * w

        res = _evaluate_condition(_get_hybrid_w, pre_scaled=True)
        if res:
            results['hybrid']['weighted'] = res
            print(f"    hybrid_w       R²_all={res['r2_all']:.4f}  "
                  f"R²_cliff={res['r2_cliff']:.4f}  RMSE_cliff={res['rmse_cliff']:.4f}  "
                  f"R²_noncliff={res['r2_noncliff']:.4f}")

    # --- RF baseline (on best individual rep) ---
    print("\n  RF baseline (per-rep, best rep):")
    best_rep_name = max(results['per_rep'],
                        key=lambda r: results['per_rep'][r]['r2_all']) if results['per_rep'] else None
    if best_rep_name:
        X_best = reps[best_rep_name]

        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_cliff = []
        for train_idx, val_idx in gkf.split(sample_X, labels, groups=groups):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_best[train_idx])
            X_va = scaler.transform(X_best[val_idx])
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_train)
            y_pred = rf.predict(X_va)

            fold_all.append(r2_score(y_val, y_pred))
            val_cliff = np.array([i for i, vi in enumerate(val_idx) if vi in cliff_indices])
            val_noncliff = np.array([i for i, vi in enumerate(val_idx) if vi not in cliff_indices])
            if len(val_cliff) > 1:
                fold_cliff.append(r2_score(y_val[val_cliff], y_pred[val_cliff]))
                fold_rmse_cliff.append(float(np.sqrt(mean_squared_error(
                    y_val[val_cliff], y_pred[val_cliff]))))
            if len(val_noncliff) > 1:
                fold_noncliff.append(r2_score(y_val[val_noncliff], y_pred[val_noncliff]))

        results['rf_baseline'] = {
            'rep': best_rep_name,
            'r2_all': float(np.mean(fold_all)),
            'r2_cliff': float(np.mean(fold_cliff)) if fold_cliff else float('nan'),
            'r2_noncliff': float(np.mean(fold_noncliff)) if fold_noncliff else float('nan'),
            'rmse_cliff': float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan'),
        }
        print(f"    RF({best_rep_name}) R²_all={results['rf_baseline']['r2_all']:.4f}  "
              f"R²_cliff={results['rf_baseline']['r2_cliff']:.4f}  "
              f"RMSE_cliff={results['rf_baseline']['rmse_cliff']:.4f}")

    # --- Summary ---
    print(f"\n  --- Summary (RMSE_cliff comparison) ---")
    all_methods = []
    for rname, rv in results['per_rep'].items():
        all_methods.append((f"kNN({rname})", rv.get('rmse_cliff', float('nan')),
                           rv.get('r2_cliff', float('nan'))))
    if 'unweighted' in results.get('hybrid', {}):
        rv = results['hybrid']['unweighted']
        all_methods.append(("hybrid_kNN_uw", rv.get('rmse_cliff', float('nan')),
                           rv.get('r2_cliff', float('nan'))))
    if 'weighted' in results.get('hybrid', {}):
        rv = results['hybrid']['weighted']
        all_methods.append(("hybrid_kNN_w", rv.get('rmse_cliff', float('nan')),
                           rv.get('r2_cliff', float('nan'))))
    if 'rf_baseline' in results:
        rv = results['rf_baseline']
        all_methods.append((f"RF({rv['rep']})", rv.get('rmse_cliff', float('nan')),
                           rv.get('r2_cliff', float('nan'))))

    all_methods.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 999)
    print(f"  {'Method':<25} {'RMSE_cliff':<12} {'R²_cliff':<12}")
    print("  " + "-" * 49)
    for name, rmse, r2 in all_methods:
        print(f"  {name:<25} {rmse:<12.4f} {r2:<12.4f}")

    return results


# =============================================================================
# EXPERIMENT 5: kNN PREDICTION WITH HYBRID DISTANCE
# =============================================================================

def run_knn_prediction(data: dict, n_folds: int,
                       search_speed: str = 'fast',
                       k_values: List[int] = None,
                       conditions: List[str] = None,
                       shared_search: Tuple = None) -> Dict[str, Any]:
    """Downstream task validation: hybrid-weighted kNN outperforms standard kNN,
    especially on cliff molecules."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: kNN PREDICTION WITH HYBRID DISTANCE")
    print("=" * 70)

    if k_values is None:
        k_values = [5, 10]
    if conditions is None:
        conditions = ['hybrid_unweighted', 'hybrid_weighted']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'conditions': {}, 'cliff_analysis': {}}

    # Pre-compute concat
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    # Get hybrid via find_best_hybrid — extract blueprint for per-fold CV
    X_hybrid = None
    weights_flat = None
    hybrid_blueprint = None
    if any(c.startswith('hybrid') for c in conditions):
        try:
            if shared_search is not None:
                X_hybrid, weights_flat, search = shared_search
                print(f"  Using shared search (dim={X_hybrid.shape[1]})")
            else:
                X_hybrid, weights_flat, search = get_hybrid_features_and_weights(
                    reps, labels, groups, search_speed, 'lightgbm_gain')
            if search.best_overall:
                hybrid_blueprint = search.best_overall['blueprint']
        except Exception as e:
            print(f"  Hybrid search failed: {e}")

    # Identify cliff molecules for stratified analysis (representation-independent)
    if 'ecfp4' in reps:
        cliff_mask, cliff_info = identify_cliff_compounds(reps['ecfp4'], labels)
        cliff_indices = set(np.where(cliff_mask)[0].tolist())
        print(f"  Cliff molecules (ECFP4 Tanimoto): {len(cliff_indices)} / {len(labels)} "
              f"({cliff_info['pct_cliff_compounds']:.1f}%)")
    else:
        cliff_indices = set()
        print("  WARNING: ecfp4 not available, skipping cliff stratification")

    # CV evaluation
    gkf = GroupKFold(n_splits=n_folds)
    sample_X = next(iter(reps.values()))

    for condition in conditions:
        print(f"\n  Condition: {condition}")

        for k in k_values:
            fold_scores = []
            fold_cliff_scores = []
            fold_noncliff_scores = []
            fold_errors = []

            for train_idx, val_idx in gkf.split(sample_X, labels, groups=groups):
                y_train, y_val = labels[train_idx], labels[val_idx]
                scaler = StandardScaler()

                if condition == 'concat_unweighted':
                    X_tr = scaler.fit_transform(X_concat[train_idx])
                    X_va = scaler.transform(X_concat[val_idx])
                    model = KNeighborsRegressor(n_neighbors=min(k, len(y_train) - 1))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'concat_weighted':
                    # Compute importance on train, weight concat features
                    train_reps = {name: rep[train_idx] for name, rep in reps.items()}
                    try:
                        imp = compute_feature_importance(train_reps, y_train,
                                                        method='lightgbm_gain')
                        w = np.concatenate([imp[name] for name in sorted(reps.keys())])
                        w = np.sqrt(np.maximum(w, 0))
                    except Exception:
                        w = np.ones(X_concat.shape[1])

                    X_tr = scaler.fit_transform(X_concat[train_idx]) * w
                    X_va = scaler.transform(X_concat[val_idx]) * w
                    model = KNeighborsRegressor(n_neighbors=min(k, len(y_train) - 1))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'hybrid_unweighted' and hybrid_blueprint is not None:
                    # Per-fold hybrid using search blueprint
                    train_reps = {name: rep[train_idx] for name, rep in reps.items()}
                    val_reps = {name: rep[val_idx] for name, rep in reps.items()}
                    try:
                        X_tr_h, feat_info = create_hybrid(
                            train_reps, y_train,
                            allocation_method=hybrid_blueprint.allocation_method,
                            budget=hybrid_blueprint.budget,
                            importance_method=hybrid_blueprint.importance_method,
                        )
                        X_va_h = apply_feature_selection(val_reps, feat_info)
                    except Exception:
                        continue

                    X_tr = scaler.fit_transform(X_tr_h)
                    X_va = scaler.transform(X_va_h)
                    model = KNeighborsRegressor(n_neighbors=min(k, len(y_train) - 1))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'hybrid_weighted' and hybrid_blueprint is not None:
                    # Per-fold hybrid using search blueprint + importance weighting
                    train_reps = {name: rep[train_idx] for name, rep in reps.items()}
                    val_reps = {name: rep[val_idx] for name, rep in reps.items()}
                    try:
                        X_tr_h, feat_info = create_hybrid(
                            train_reps, y_train,
                            allocation_method=hybrid_blueprint.allocation_method,
                            budget=hybrid_blueprint.budget,
                            importance_method=hybrid_blueprint.importance_method,
                        )
                        X_va_h = apply_feature_selection(val_reps, feat_info)

                        imp = compute_feature_importance(
                            train_reps, y_train, method='lightgbm_gain')
                        # Get weights for selected features
                        w_parts = []
                        for rname in feat_info:
                            if not isinstance(feat_info[rname], dict):
                                continue
                            if 'selected_indices' not in feat_info[rname]:
                                continue
                            idx = feat_info[rname]['selected_indices']
                            if rname in imp:
                                w_parts.append(imp[rname][idx])
                            else:
                                w_parts.append(np.ones(len(idx)))
                        w = np.sqrt(np.maximum(np.concatenate(w_parts), 0))
                    except Exception:
                        continue

                    X_tr = scaler.fit_transform(X_tr_h) * w
                    X_va = scaler.transform(X_va_h) * w
                    model = KNeighborsRegressor(n_neighbors=min(k, len(y_train) - 1))
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                elif condition == 'rf_baseline':
                    X_tr = scaler.fit_transform(X_concat[train_idx])
                    X_va = scaler.transform(X_concat[val_idx])
                    model = RandomForestRegressor(n_estimators=100, random_state=42,
                                                  n_jobs=-1)
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_va)

                else:
                    continue

                r2 = r2_score(y_val, y_pred)
                fold_scores.append(r2)

                # Stratified: cliff vs non-cliff
                val_set = set(val_idx.tolist())
                cliff_val = np.array([i for i, vi in enumerate(val_idx)
                                      if vi in cliff_indices])
                noncliff_val = np.array([i for i, vi in enumerate(val_idx)
                                         if vi not in cliff_indices])

                if len(cliff_val) > 1:
                    fold_cliff_scores.append(r2_score(y_val[cliff_val],
                                                      y_pred[cliff_val]))
                if len(noncliff_val) > 1:
                    fold_noncliff_scores.append(r2_score(y_val[noncliff_val],
                                                         y_pred[noncliff_val]))

                # Per-sample errors for SALI correlation
                fold_errors.extend(list(zip(val_idx.tolist(),
                                           np.abs(y_val - y_pred).tolist())))

            if not fold_scores:
                continue

            r2_mean = float(np.mean(fold_scores))
            r2_std = float(np.std(fold_scores))
            cliff_r2 = float(np.mean(fold_cliff_scores)) if fold_cliff_scores else float('nan')
            noncliff_r2 = float(np.mean(fold_noncliff_scores)) if fold_noncliff_scores else float('nan')

            key = f"{condition}_k{k}"
            results['conditions'][key] = {
                'r2_mean': r2_mean,
                'r2_std': r2_std,
                'cliff_r2': cliff_r2,
                'noncliff_r2': noncliff_r2,
                'k': k,
                'condition': condition,
            }
            print(f"    k={k}: R²={r2_mean:.4f}+/-{r2_std:.3f}, "
                  f"cliff_R²={cliff_r2:.4f}, noncliff_R²={noncliff_r2:.4f}")

    # Per-rep kNN baselines (individual representations — the real baselines)
    print("\n  Individual representation kNN baselines:")
    results['per_rep_baselines'] = {}
    best_k = k_values[len(k_values) // 2]  # use middle k value
    gkf2 = GroupKFold(n_splits=n_folds)
    for rep_name, X_rep in reps.items():
        fold_r2s_uw = []
        fold_r2s_w = []
        for train_idx, val_idx in gkf2.split(sample_X, labels, groups=groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_rep[train_idx])
            X_va = scaler.transform(X_rep[val_idx])

            # Unweighted
            model = KNeighborsRegressor(n_neighbors=min(best_k, len(train_idx) - 1))
            model.fit(X_tr, labels[train_idx])
            fold_r2s_uw.append(r2_score(labels[val_idx], model.predict(X_va)))

            # Weighted by importance (computed on train only)
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[train_idx]}, labels[train_idx],
                    method='lightgbm_gain')
                w = np.sqrt(np.maximum(imp[rep_name], 0))
                X_tr_w = X_tr * w
                X_va_w = X_va * w
                model_w = KNeighborsRegressor(
                    n_neighbors=min(best_k, len(train_idx) - 1))
                model_w.fit(X_tr_w, labels[train_idx])
                fold_r2s_w.append(r2_score(labels[val_idx], model_w.predict(X_va_w)))
            except Exception:
                fold_r2s_w.append(float('nan'))

        r2_mean = float(np.mean(fold_r2s_uw))
        r2_std = float(np.std(fold_r2s_uw))
        r2_w_mean = float(np.nanmean(fold_r2s_w)) if fold_r2s_w else float('nan')
        r2_w_std = float(np.nanstd(fold_r2s_w)) if fold_r2s_w else float('nan')
        results['per_rep_baselines'][rep_name] = {
            'r2_mean': r2_mean, 'r2_std': r2_std,
            'r2_weighted_mean': r2_w_mean, 'r2_weighted_std': r2_w_std,
            'k': best_k, 'dim': X_rep.shape[1],
        }
        delta = r2_w_mean - r2_mean
        print(f"    {rep_name} (dim={X_rep.shape[1]}): "
              f"kNN R²={r2_mean:.4f}+/-{r2_std:.3f}, "
              f"weighted R²={r2_w_mean:.4f}+/-{r2_w_std:.3f} "
              f"(delta={delta:+.4f})")

    # Summary: best k per condition
    print("\n  Summary (best k per condition):")
    for cond in conditions:
        cond_results = {k: v for k, v in results['conditions'].items()
                       if v['condition'] == cond}
        if cond_results:
            best_key = max(cond_results, key=lambda k: cond_results[k]['r2_mean'])
            best = cond_results[best_key]
            print(f"    {cond}: best k={best['k']}, "
                  f"R²={best['r2_mean']:.4f}")

    # Flag best individual rep
    if results['per_rep_baselines']:
        best_rep = max(results['per_rep_baselines'],
                       key=lambda r: results['per_rep_baselines'][r]['r2_mean'])
        best_rep_r2 = results['per_rep_baselines'][best_rep]['r2_mean']
        print(f"\n  Best individual rep baseline: {best_rep} "
              f"(kNN R²={best_rep_r2:.4f})")

        # Compare hybrid_weighted to best individual rep
        hw_results = {k: v for k, v in results['conditions'].items()
                      if v['condition'] == 'hybrid_weighted'}
        if hw_results:
            best_hw = max(hw_results.values(), key=lambda v: v['r2_mean'])
            delta = best_hw['r2_mean'] - best_rep_r2
            print(f"  Hybrid weighted kNN R²={best_hw['r2_mean']:.4f} "
                  f"(delta vs {best_rep}: {delta:+.4f})")

    return results


# =============================================================================
# EXPERIMENT 6: APPLICABILITY DOMAIN
# =============================================================================

def run_applicability_domain(data: dict, n_folds: int,
                             search_speed: str = 'fast',
                             k_values: List[int] = None,
                             shared_search: Tuple = None) -> Dict[str, Any]:
    """Hybrid distance tracks prediction uncertainty better than standard
    distance. Molecules far in hybrid space have less reliable predictions."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: APPLICABILITY DOMAIN")
    print("=" * 70)

    if k_values is None:
        k_values = [5]

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'distance_error_correlation': {}, 'reliability': {}, 'conformal': {}}

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    # Get hybrid + blueprint for per-fold CV
    try:
        if shared_search is not None:
            X_hybrid, weights_flat, search = shared_search
            print(f"  Using shared search (dim={X_hybrid.shape[1]})")
        else:
            X_hybrid, weights_flat, search = get_hybrid_features_and_weights(
                reps, labels, groups, search_speed, 'lightgbm_gain')
        bp = search.best_overall['blueprint']
    except Exception as e:
        print(f"  Hybrid search failed: {e}")
        return {'error': str(e)}

    gkf = GroupKFold(n_splits=n_folds)
    sample_X = next(iter(reps.values()))

    for k in k_values:
        print(f"\n  k={k}:")
        all_d_concat = []
        all_d_hybrid = []
        all_d_weighted = []
        all_errors = []

        for train_idx, val_idx in gkf.split(sample_X, labels, groups=groups):
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Per-fold hybrid using search blueprint
            train_reps = {name: rep[train_idx] for name, rep in reps.items()}
            val_reps = {name: rep[val_idx] for name, rep in reps.items()}
            try:
                X_tr_h, feat_info = create_hybrid(
                    train_reps, y_train,
                    allocation_method=bp.allocation_method,
                    budget=bp.budget,
                    importance_method=bp.importance_method,
                )
                X_va_h = apply_feature_selection(val_reps, feat_info)

                # Importance weights for this fold
                imp = compute_feature_importance(
                    train_reps, y_train, method='lightgbm_gain')
                w_parts = []
                for rname in feat_info:
                    if not isinstance(feat_info[rname], dict):
                        continue
                    if 'selected_indices' not in feat_info[rname]:
                        continue
                    idx = feat_info[rname]['selected_indices']
                    if rname in imp:
                        w_parts.append(imp[rname][idx])
                    else:
                        w_parts.append(np.ones(len(idx)))
                w = np.sqrt(np.maximum(np.concatenate(w_parts), 0))
            except Exception:
                continue

            # Train RF on hybrid features
            scaler_h = StandardScaler()
            X_tr_s = scaler_h.fit_transform(X_tr_h)
            X_va_s = scaler_h.transform(X_va_h)
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr_s, y_train)
            y_pred = rf.predict(X_va_s)
            errors = np.abs(y_val - y_pred)

            # Distance to k-NN in training set for each val molecule
            # Concat space
            scaler_c = StandardScaler()
            Xtr_c = scaler_c.fit_transform(X_concat[train_idx])
            Xva_c = scaler_c.transform(X_concat[val_idx])
            nn_c = NearestNeighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_c.fit(Xtr_c)
            dists_c, _ = nn_c.kneighbors(Xva_c)
            d_concat = dists_c.mean(axis=1)

            # Hybrid space (unweighted)
            nn_h = NearestNeighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_h.fit(X_tr_s)
            dists_h, _ = nn_h.kneighbors(X_va_s)
            d_hybrid = dists_h.mean(axis=1)

            # Weighted hybrid space
            nn_w = NearestNeighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_w.fit(X_tr_s * w)
            dists_w, _ = nn_w.kneighbors(X_va_s * w)
            d_weighted = dists_w.mean(axis=1)

            all_d_concat.extend(d_concat.tolist())
            all_d_hybrid.extend(d_hybrid.tolist())
            all_d_weighted.extend(d_weighted.tolist())
            all_errors.extend(errors.tolist())

        if not all_errors:
            continue

        all_d_concat = np.array(all_d_concat)
        all_d_hybrid = np.array(all_d_hybrid)
        all_d_weighted = np.array(all_d_weighted)
        all_errors = np.array(all_errors)

        # Spearman correlations: distance vs |error|
        rho_c, _ = spearmanr(all_d_concat, all_errors)
        rho_h, _ = spearmanr(all_d_hybrid, all_errors)
        rho_w, _ = spearmanr(all_d_weighted, all_errors)

        results['distance_error_correlation'][f'k{k}'] = {
            'concat': float(rho_c),
            'hybrid': float(rho_h),
            'hybrid_weighted': float(rho_w),
        }
        print(f"    Spearman(dist, |error|): concat={rho_c:.4f}, "
              f"hybrid={rho_h:.4f}, weighted={rho_w:.4f}")

        # Reliability diagram: bin by distance decile
        for dist_name, dists in [('concat', all_d_concat),
                                  ('hybrid', all_d_hybrid),
                                  ('hybrid_weighted', all_d_weighted)]:
            deciles = np.percentile(dists, np.arange(0, 110, 10))
            decile_errors = []
            decile_counts = []
            for di in range(10):
                mask = (dists >= deciles[di]) & (dists < deciles[di + 1])
                if di == 9:  # include upper bound in last bin
                    mask = (dists >= deciles[di]) & (dists <= deciles[di + 1])
                if mask.sum() > 0:
                    decile_errors.append(float(np.mean(all_errors[mask])))
                    decile_counts.append(int(mask.sum()))
                else:
                    decile_errors.append(float('nan'))
                    decile_counts.append(0)

            results['reliability'].setdefault(f'k{k}', {})[dist_name] = {
                'decile_errors': decile_errors,
                'decile_counts': decile_counts,
            }

    # Conformal calibration comparison (proper CV — per-fold hybrid, no data leakage)
    print("\n  Conformal calibration (CV):")
    try:
        gkf_conf = GroupKFold(n_splits=n_folds)
        conf_picp = {'concat': [], 'hybrid': []}
        conf_mpiw = {'concat': [], 'hybrid': []}
        alpha = 0.1

        for train_idx, val_idx in gkf_conf.split(sample_X, labels, groups=groups):
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Per-fold hybrid using search blueprint (avoids data leakage)
            train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
            val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
            try:
                X_hybrid_tr, fi = create_hybrid(
                    train_reps_fold, y_train,
                    allocation_method=bp.allocation_method,
                    budget=bp.budget,
                    importance_method=bp.importance_method,
                )
                X_hybrid_va = apply_feature_selection(val_reps_fold, fi)
            except Exception:
                continue

            # Split train into proper-train and calibration
            n_cal = max(int(len(y_train) * 0.2), 10)
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(y_train))
            cal_local, tr_local = perm[:n_cal], perm[n_cal:]

            for feat_name, X_tr_all, X_va in [
                ('concat', X_concat[train_idx], X_concat[val_idx]),
                ('hybrid', X_hybrid_tr, X_hybrid_va),
            ]:
                scaler = StandardScaler()
                X_tr_all_s = scaler.fit_transform(X_tr_all)
                X_va_s = scaler.transform(X_va)

                X_tr = X_tr_all_s[tr_local]
                X_cal = X_tr_all_s[cal_local]
                y_tr = y_train[tr_local]
                y_cal = y_train[cal_local]

                # Train RF, compute conformal quantile on calibration set
                rf = RandomForestRegressor(n_estimators=100, random_state=42,
                                           n_jobs=-1)
                rf.fit(X_tr, y_tr)

                # Calibration: residuals normalized by tree variance
                y_pred_cal = rf.predict(X_cal)
                tree_preds_cal = np.array([t.predict(X_cal) for t in rf.estimators_])
                cal_std = tree_preds_cal.std(axis=0) + 1e-10
                conformity = np.abs(y_cal - y_pred_cal) / cal_std
                q = np.quantile(conformity, 1 - alpha)

                # Val: predict + interval
                y_pred_va = rf.predict(X_va_s)
                tree_preds_va = np.array([t.predict(X_va_s) for t in rf.estimators_])
                va_std = tree_preds_va.std(axis=0)
                interval_width = 2 * q * va_std

                picp = float(np.mean(np.abs(y_val - y_pred_va) <= interval_width / 2))
                mpiw = float(np.mean(interval_width))
                conf_picp[feat_name].append(picp)
                conf_mpiw[feat_name].append(mpiw)

        results['conformal'] = {}
        for feat_name in ['concat', 'hybrid']:
            if conf_picp[feat_name]:
                picp_mean = float(np.mean(conf_picp[feat_name]))
                mpiw_mean = float(np.mean(conf_mpiw[feat_name]))
                results['conformal'][feat_name] = {
                    'picp': picp_mean, 'mpiw': mpiw_mean,
                }
                print(f"    {feat_name.capitalize()}:  PICP={picp_mean:.3f}, "
                      f"MPIW={mpiw_mean:.3f}")
    except Exception as e:
        print(f"    Conformal failed: {e}")
        results['conformal'] = {'error': str(e)}

    return results


# =============================================================================
# EXPERIMENT 7: NOISE DETECTION BRIDGE
# =============================================================================

def run_noise_detection_bridge(data: dict, n_folds: int,
                               search_speed: str = 'fast',
                               shared_search: Tuple = None) -> Dict[str, Any]:
    """Standard noise detectors flag activity cliffs as noise (false positives).
    Hybrid distances reduce this FP rate."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: NOISE DETECTION BRIDGE (Supplementary)")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {}

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    # Get hybrid
    try:
        if shared_search is not None:
            X_hybrid, weights_flat, search = shared_search
            print(f"  Using shared search (dim={X_hybrid.shape[1]})")
        else:
            X_hybrid, weights_flat, search = get_hybrid_features_and_weights(
                reps, labels, groups, search_speed, 'lightgbm_gain')
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


# =============================================================================
# EXPERIMENT 8: OPTIMAL DIMENSION FOR SIMILARITY VS PREDICTION
# =============================================================================

def run_optimal_similarity_dimension(data: dict, n_folds: int,
                                     search_speed: str = 'fast',
                                     budgets: List[int] = None,
                                     n_sample: int = 5000) -> Dict[str, Any]:
    """The optimal hybrid dimension may differ for prediction vs similarity.
    Similarity may peak at lower dim than prediction (curse of dimensionality)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: OPTIMAL DIMENSION FOR SIMILARITY VS PREDICTION")
    print("=" * 70)

    if budgets is None:
        budgets = [25, 50, 100]

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'budgets': {}}

    # Full-dim reference
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    full_dim = X_concat.shape[1]

    gkf = GroupKFold(n_splits=n_folds)
    sample_X = next(iter(reps.values()))

    # Pre-compute full-dim distance matrix + sampled pair indices (reused per budget)
    D_full = compute_distance_matrix(X_concat, standardize=True)
    n = len(labels)
    triu_i, triu_j = np.triu_indices(n, k=1)
    n_pairs = len(triu_i)
    if n_pairs > 10000:
        rng = np.random.RandomState(42)
        _idx = rng.choice(n_pairs, 10000, replace=False)
        triu_i_s, triu_j_s = triu_i[_idx], triu_j[_idx]
    else:
        triu_i_s, triu_j_s = triu_i, triu_j
    d_full_sampled = D_full[triu_i_s, triu_j_s]

    for budget in budgets:
        print(f"\n  Budget={budget}:")
        t0 = time.time()

        try:
            X_hybrid, weights_flat, search = get_hybrid_features_and_weights(
                reps, labels, groups, search_speed, 'lightgbm_gain',
                budget=budget)
        except Exception as e:
            print(f"    Failed: {e}")
            results['budgets'][budget] = {'error': str(e)}
            continue

        actual_dim = X_hybrid.shape[1]

        # MLE intrinsic dimensionality of the hybrid
        try:
            X_hybrid_scaled = StandardScaler().fit_transform(X_hybrid)
            mle_id = float(estimate_intrinsic_dim_mle(X_hybrid_scaled, k=10))
        except Exception:
            mle_id = float('nan')

        # Similarity quality
        D_weighted = compute_distance_matrix(X_hybrid, feature_weights=weights_flat,
                                                 standardize=True)
        sac_rho, _ = similarity_activity_correlation(D_weighted, labels, n_sample)
        viol = violation_rate(D_weighted, labels, k=5)

        # Distance correlation with full concat (using pre-computed pairs)
        d_hybrid_sampled = D_weighted[triu_i_s, triu_j_s]
        dist_corr, _ = spearmanr(d_full_sampled, d_hybrid_sampled)

        # Prediction quality: RF R² (CV) — use search blueprint's method
        bp = search.best_overall['blueprint'] if search.best_overall else None
        alloc = bp.allocation_method if bp else 'greedy'
        imp_method = bp.importance_method if bp else 'lightgbm_gain'

        rf_scores = []
        knn_scores = []
        for train_idx, val_idx in gkf.split(sample_X, labels, groups=groups):
            y_train, y_val = labels[train_idx], labels[val_idx]

            train_reps_fold = {name: rep[train_idx] for name, rep in reps.items()}
            val_reps_fold = {name: rep[val_idx] for name, rep in reps.items()}

            try:
                X_tr_h, feat_info = create_hybrid(
                    train_reps_fold, y_train,
                    allocation_method=alloc,
                    budget=budget,
                    importance_method=imp_method,
                )
                X_va_h = apply_feature_selection(val_reps_fold, feat_info)
            except Exception:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_h)
            X_va_s = scaler.transform(X_va_h)

            # RF
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr_s, y_train)
            rf_scores.append(r2_score(y_val, rf.predict(X_va_s)))

            # kNN
            knn = KNeighborsRegressor(n_neighbors=min(5, len(y_train) - 1))
            knn.fit(X_tr_s, y_train)
            knn_scores.append(r2_score(y_val, knn.predict(X_va_s)))

        rf_r2 = float(np.mean(rf_scores)) if rf_scores else float('nan')
        knn_r2 = float(np.mean(knn_scores)) if knn_scores else float('nan')

        elapsed = time.time() - t0

        results['budgets'][budget] = {
            'actual_dim': actual_dim,
            'mle_id': mle_id,
            'rf_r2': rf_r2,
            'knn_r2': knn_r2,
            'sac_rho': sac_rho,
            'violation_rate': viol,
            'dist_corr_with_full': dist_corr,
            'time_seconds': elapsed,
        }
        print(f"    dim={actual_dim} (MLE_ID={mle_id:.1f}): RF_R²={rf_r2:.4f}, kNN_R²={knn_r2:.4f}, "
              f"SAC={sac_rho:.4f}, viol={viol:.3f}, dist_corr={dist_corr:.3f}")

    # Per-rep baselines (the real comparison: best individual rep)
    print("\n  Individual rep baselines:")
    results['per_rep'] = {}
    best_rep_sac = -1
    best_rep_name = None
    for rep_name, X in reps.items():
        D_rep = compute_distance_matrix(X, standardize=True)
        sac_rep, _ = similarity_activity_correlation(D_rep, labels, n_sample)
        viol_rep = violation_rate(D_rep, labels, k=5)

        rf_rep = []
        knn_rep = []
        for train_idx, val_idx in gkf.split(sample_X, labels, groups=groups):
            y_train, y_val = labels[train_idx], labels[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_va = scaler.transform(X[val_idx])

            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_train)
            rf_rep.append(r2_score(y_val, rf.predict(X_va)))

            knn = KNeighborsRegressor(n_neighbors=min(5, len(y_train) - 1))
            knn.fit(X_tr, y_train)
            knn_rep.append(r2_score(y_val, knn.predict(X_va)))

        # MLE ID for individual rep
        try:
            rep_mle = float(estimate_intrinsic_dim_mle(
                StandardScaler().fit_transform(X), k=10))
        except Exception:
            rep_mle = float('nan')

        results['per_rep'][rep_name] = {
            'dim': X.shape[1],
            'mle_id': rep_mle,
            'rf_r2': float(np.mean(rf_rep)),
            'knn_r2': float(np.mean(knn_rep)),
            'sac_rho': sac_rep,
            'violation_rate': viol_rep,
        }
        if best_rep_name is None or sac_rep > best_rep_sac:
            best_rep_sac = sac_rep
            best_rep_name = rep_name
        print(f"    {rep_name} (dim={X.shape[1]}): RF_R²={np.mean(rf_rep):.4f}, "
              f"kNN_R²={np.mean(knn_rep):.4f}, SAC={sac_rep:.4f}")

    results['best_individual_rep'] = best_rep_name
    print(f"  Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f})")

    # Concat (reference, not standard practice)
    print(f"\n  Concat reference (dim={full_dim}):")
    sac_full, _ = similarity_activity_correlation(D_full, labels, n_sample)
    viol_full = violation_rate(D_full, labels, k=5)

    rf_scores_full = []
    knn_scores_full = []
    for train_idx, val_idx in gkf.split(sample_X, labels, groups=groups):
        y_train, y_val = labels[train_idx], labels[val_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_concat[train_idx])
        X_va = scaler.transform(X_concat[val_idx])

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_train)
        rf_scores_full.append(r2_score(y_val, rf.predict(X_va)))

        knn = KNeighborsRegressor(n_neighbors=min(5, len(y_train) - 1))
        knn.fit(X_tr, y_train)
        knn_scores_full.append(r2_score(y_val, knn.predict(X_va)))

    results['concat_reference'] = {
        'dim': full_dim,
        'rf_r2': float(np.mean(rf_scores_full)),
        'knn_r2': float(np.mean(knn_scores_full)),
        'sac_rho': sac_full,
        'violation_rate': viol_full,
    }
    print(f"    RF_R²={results['concat_reference']['rf_r2']:.4f}, "
          f"kNN_R²={results['concat_reference']['knn_r2']:.4f}, "
          f"SAC={sac_full:.4f}, viol={viol_full:.3f}")

    # Find optimal budget for prediction vs similarity
    budget_list = sorted(results['budgets'].keys())
    if budget_list:
        best_rf_budget = max(budget_list,
                             key=lambda b: results['budgets'][b].get('rf_r2', -999))
        best_knn_budget = max(budget_list,
                              key=lambda b: results['budgets'][b].get('knn_r2', -999))
        # Highest SAC = best distance-activity correlation
        best_sac_budget = max(budget_list,
                              key=lambda b: results['budgets'][b].get('sac_rho', -999))

        results['optimal'] = {
            'best_rf_budget': best_rf_budget,
            'best_knn_budget': best_knn_budget,
            'best_sac_budget': best_sac_budget,
        }
        print(f"\n  Optimal budgets: RF={best_rf_budget}, "
              f"kNN={best_knn_budget}, SAC={best_sac_budget}")

    return results


# =============================================================================
# EXPERIMENT 9: CROSS-DATASET GENERALIZATION SUMMARY
# =============================================================================

def run_cross_dataset_summary(all_dataset_results: Dict[str, dict]) -> Dict[str, Any]:
    """Consolidate findings across datasets. Show improvement is consistent."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: CROSS-DATASET GENERALIZATION SUMMARY")
    print("=" * 70)

    results = {'per_dataset': {}, 'summary': {}}

    sac_deltas = []
    cliff_reductions = []
    knn_deltas = []
    ad_deltas = []
    fp_reductions = []
    dim_ratios = []

    for ds_name, ds_results in all_dataset_results.items():
        print(f"\n  {ds_name}:")
        ds_summary = {}

        # Exp 2: SAC improvement vs best individual rep
        exp2 = ds_results.get('exp2_hybrid_similarity')
        if exp2:
            best_rep_info = exp2.get('best_individual_rep', {})
            best_rep_sac = best_rep_info.get('sac_rho', float('nan'))
            best_rep_name = best_rep_info.get('name', '?')
            best_hybrid_sac = exp2.get('best_sac', float('nan'))
            delta_sac = best_hybrid_sac - best_rep_sac
            ds_summary['sac_delta'] = delta_sac
            ds_summary['best_rep'] = best_rep_name
            sac_deltas.append(delta_sac)
            print(f"    SAC delta: {delta_sac:+.4f} "
                  f"(best_rep {best_rep_name}={best_rep_sac:.4f} -> "
                  f"hybrid={best_hybrid_sac:.4f})")

        # Exp 4: RMSE_cliff improvement (hybrid vs best per-rep kNN)
        exp4 = ds_results.get('exp4_activity_cliffs')
        if exp4 and 'per_rep' in exp4 and 'hybrid' in exp4:
            per_rep_rmse = [v.get('rmse_cliff', float('nan'))
                           for v in exp4['per_rep'].values()]
            valid_rmse = [r for r in per_rep_rmse if not np.isnan(r)]
            best_rep_rmse = min(valid_rmse) if valid_rmse else float('nan')
            hybrid_w = exp4['hybrid'].get('weighted', {})
            hybrid_rmse = hybrid_w.get('rmse_cliff', float('nan'))
            if not np.isnan(best_rep_rmse) and not np.isnan(hybrid_rmse) and best_rep_rmse > 0:
                reduction = (best_rep_rmse - hybrid_rmse) / best_rep_rmse * 100
                ds_summary['cliff_rmse_reduction_pct'] = reduction
                cliff_reductions.append(reduction)
                print(f"    RMSE_cliff reduction: {reduction:.1f}% "
                      f"(best_rep={best_rep_rmse:.4f} -> hybrid={hybrid_rmse:.4f})")
            ds_summary['n_cliff_compounds'] = exp4.get('cliff_definition', {}).get('n_cliff_compounds', 0)

        # Exp 5: kNN improvement vs best individual rep
        exp5 = ds_results.get('exp5_knn_prediction')
        if exp5 and 'conditions' in exp5:
            conds = exp5['conditions']
            hw_scores = [v['r2_mean'] for k, v in conds.items()
                        if v.get('condition') == 'hybrid_weighted']
            # Compare against best individual rep kNN baseline
            per_rep = exp5.get('per_rep_baselines', {})
            if hw_scores and per_rep:
                best_rep_knn = max(v['r2_mean'] for v in per_rep.values())
                delta_knn = max(hw_scores) - best_rep_knn
                ds_summary['knn_delta_r2'] = delta_knn
                knn_deltas.append(delta_knn)
                print(f"    kNN delta R² (vs best rep): {delta_knn:+.4f}")

        # Exp 6: AD correlation (hybrid_weighted vs unweighted hybrid)
        exp6 = ds_results.get('exp6_applicability_domain')
        if exp6 and 'distance_error_correlation' in exp6:
            for k_key, corrs in exp6['distance_error_correlation'].items():
                if 'hybrid' in corrs and 'hybrid_weighted' in corrs:
                    delta_ad = corrs['hybrid_weighted'] - corrs['hybrid']
                    ds_summary['ad_corr_delta'] = delta_ad
                    ds_summary['ad_corr_weighted'] = corrs['hybrid_weighted']
                    ad_deltas.append(delta_ad)
                    print(f"    AD corr: weighted={corrs['hybrid_weighted']:.4f}, "
                          f"unweighted={corrs['hybrid']:.4f} "
                          f"(delta={delta_ad:+.4f})")
                    break

        # Exp 7: noise FP reduction
        exp7 = ds_results.get('exp7_noise_bridge')
        if exp7 and 'fp_rate_reduction_pct' in exp7:
            fp_red = exp7['fp_rate_reduction_pct']
            ds_summary['noise_fp_reduction_pct'] = fp_red
            fp_reductions.append(fp_red)
            print(f"    Noise FP reduction: {fp_red:.1f}%")

        # Exp 8: optimal dim ratio
        exp8 = ds_results.get('exp8_optimal_dimension')
        if exp8 and 'optimal' in exp8:
            opt = exp8['optimal']
            knn_opt = opt.get('best_knn_budget', 0)
            rf_opt = opt.get('best_rf_budget', 0)
            if rf_opt > 0:
                ratio = knn_opt / rf_opt
                ds_summary['optimal_dim_ratio'] = ratio
                dim_ratios.append(ratio)
                print(f"    Optimal dim ratio (kNN/RF): {ratio:.2f}")

        results['per_dataset'][ds_name] = ds_summary

    # Summary statistics
    print("\n  Aggregate summary:")
    for name, values in [('SAC_delta', sac_deltas),
                          ('RMSE_cliff_reduction_%', cliff_reductions),
                          ('kNN_delta_R2', knn_deltas),
                          ('AD_corr_delta', ad_deltas),
                          ('noise_FP_reduction_%', fp_reductions),
                          ('optimal_dim_ratio', dim_ratios)]:
        if values:
            arr = np.array(values)
            results['summary'][name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'n_datasets': len(arr),
                'values': [float(v) for v in arr],
            }
            print(f"    {name}: mean={np.mean(arr):.4f} +/- {np.std(arr):.4f} "
                  f"(n={len(arr)})")

            # Wilcoxon test if enough samples
            if len(arr) >= 3:
                try:
                    stat, p = wilcoxon(arr)
                    results['summary'][name]['wilcoxon_p'] = float(p)
                    print(f"      Wilcoxon p={p:.4f}")
                except Exception:
                    pass

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Similarity experiments for hybrid representations')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full'],
                        help='quick (~30min, logd only) or full (~6-8h, all datasets)')
    parser.add_argument('--experiments', type=int, nargs='+', default=None,
                        help='Which experiments to run (1-9). Default: mode-dependent.')
    parser.add_argument('--reps', type=str, default=None,
                        help='Representation preset (fast, fast+hf, all) or comma-separated list. '
                             'Default: fast for quick, fast+hf for full.')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Override dataset list (e.g., --datasets logd caco2)')
    parser.add_argument('--output', type=str, default='tests/results/similarity')

    args = parser.parse_args()

    # Configuration
    if args.mode == 'quick':
        dataset_names = ['logd']
        rep_preset = args.reps or 'fast'
        n_folds = 3
        search_speed = 'fast'
        experiments_default = [1, 2, 4, 5, 8]
        budgets = [25, 50, 100, 200]
        k_values = [5, 10]
        weight_methods = ['lightgbm_gain']
        n_sample = 3000
        knn_conditions = ['hybrid_unweighted', 'hybrid_weighted']
    else:  # full
        dataset_names = ['logd', 'caco2', 'herg', 'qm9']
        rep_preset = args.reps or 'fast+hf'
        n_folds = 5
        search_speed = 'medium'
        experiments_default = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        budgets = [15, 25, 50, 100, 200]
        k_values = [3, 5, 10, 20]
        weight_methods = ['lightgbm_gain', 'treeshap', 'permutation']
        n_sample = 5000
        knn_conditions = ['hybrid_unweighted', 'hybrid_weighted']

    if args.datasets:
        dataset_names = args.datasets

    experiments = args.experiments if args.experiments else experiments_default

    print(f"Mode: {args.mode}")
    print(f"Datasets: {dataset_names}")
    print(f"Reps: {rep_preset}")
    print(f"Experiments: {experiments}")
    print(f"Search speed: {search_speed}")
    print(f"Weight methods: {weight_methods}")

    # Load datasets
    datasets = {}
    for ds_name in dataset_names:
        print(f"\nLoading {ds_name}...")
        try:
            datasets[ds_name] = load_data(ds_name, rep_preset=rep_preset)
        except Exception as e:
            print(f"  FAILED: {e}")

    if not datasets:
        print("No datasets loaded, exiting.")
        return

    # Output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        'config': {
            'mode': args.mode,
            'datasets': dataset_names,
            'rep_preset': rep_preset,
            'experiments': experiments,
            'n_folds': n_folds,
            'search_speed': search_speed,
            'weight_methods': weight_methods,
            'budgets': budgets,
            'k_values': k_values,
            'timestamp': datetime.now().isoformat(),
        },
        'datasets': {},
    }

    for ds_name, data in datasets.items():
        print(f"\n{'#' * 70}")
        print(f"# DATASET: {ds_name.upper()} ({len(data['labels'])} samples)")
        print('#' * 70)

        ds_results = {
            'n_samples': len(data['labels']),
            'n_scaffolds': data['n_scaffolds'],
        }

        # Shared search for experiments 2-7 (avoid redundant find_best_hybrid calls)
        shared_search = None
        needs_search = any(e in experiments for e in [2, 3, 4, 5, 6, 7])
        if needs_search:
            print(f"\n  Running shared hybrid search ({search_speed})...")
            try:
                shared_search = get_hybrid_features_and_weights(
                    data['reps'], data['labels'], data['groups'],
                    search_speed, 'lightgbm_gain')
                print(f"  Shared search done (dim={shared_search[0].shape[1]})")
            except Exception as e:
                print(f"  Shared search FAILED: {e}")

        if 1 in experiments:
            ds_results['exp1_baseline'] = run_similarity_baseline(
                data, n_folds, n_sample=n_sample)

        if 2 in experiments:
            ds_results['exp2_hybrid_similarity'] = run_hybrid_similarity(
                data, n_folds, search_speed, weight_methods, n_sample,
                shared_search=shared_search)

        if 3 in experiments:
            ds_results['exp3_feature_concentration'] = run_feature_concentration(
                data, search_speed, weight_methods,
                shared_search=shared_search)

        if 4 in experiments:
            ds_results['exp4_activity_cliffs'] = run_activity_cliff_analysis(
                data, n_folds, search_speed,
                shared_search=shared_search)

        if 5 in experiments:
            ds_results['exp5_knn_prediction'] = run_knn_prediction(
                data, n_folds, search_speed, k_values, knn_conditions,
                shared_search=shared_search)

        if 6 in experiments:
            ds_results['exp6_applicability_domain'] = run_applicability_domain(
                data, n_folds, search_speed, k_values[:2],
                shared_search=shared_search)

        if 7 in experiments:
            ds_results['exp7_noise_bridge'] = run_noise_detection_bridge(
                data, n_folds, search_speed,
                shared_search=shared_search)

        if 8 in experiments:
            ds_results['exp8_optimal_dimension'] = run_optimal_similarity_dimension(
                data, n_folds, search_speed, budgets, n_sample)

        all_results['datasets'][ds_name] = ds_results

    # Experiment 9: cross-dataset summary (needs all datasets)
    if 9 in experiments and len(all_results['datasets']) >= 1:
        all_results['exp9_cross_dataset'] = run_cross_dataset_summary(
            all_results['datasets'])

    # Save results
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif hasattr(obj, 'to_dict'):
            return convert_numpy(obj.to_dict())
        elif hasattr(obj, '__dataclass_fields__'):
            from dataclasses import asdict
            return convert_numpy(asdict(obj))
        elif isinstance(obj, dict):
            return {str(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(v) for v in obj]
        return obj

    output_file = output_path / f"similarity_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print('=' * 70)


if __name__ == '__main__':
    main()
