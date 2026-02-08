#!/usr/bin/env python3
"""
Comprehensive benchmark for importance methods and aggregation strategies.

Tests on two datasets:
- LogD from OpenADMET (regression)
- HOMO-LUMO gap from QM9 (regression, 5k samples)

Models tested: RF, MLP, LightGBM

Goals:
1. Measure agreement between importance methods (if high, skip complex aggregation)
2. Benchmark each importance method (time, stability, downstream performance)
3. Test aggregation strategies (averaging, RRF, voting, subsets)
4. Test allocation strategies (diversity_greedy, two_stage, stability_selection)

Usage:
    python tests/benchmark_importance_aggregation.py --quick    # Fast sanity check
    python tests/benchmark_importance_aggregation.py --full     # Full benchmark
"""

import argparse
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =============================================================================
# IMPORTANCE METHODS
# =============================================================================

# All available methods from kirby.hybrid.compute_feature_importance
ALL_IMPORTANCE_METHODS = [
    'random_forest',
    'treeshap',
    'xgboost_gain',
    'xgboost_weight',
    'xgboost_cover',
    'lightgbm_gain',
    'lightgbm_split',
    'permutation',
    'integrated_gradients',
    'deeplift',
    'deepliftshap',
    'gradientshap',
    'kernelshap',
    'lime',
    'drop_column',
    'boruta',
]

# Categorized by type
METHOD_CATEGORIES = {
    'tree_based': ['random_forest', 'treeshap', 'xgboost_gain', 'xgboost_weight',
                   'xgboost_cover', 'lightgbm_gain', 'lightgbm_split'],
    'gradient_based': ['integrated_gradients', 'deeplift', 'deepliftshap', 'gradientshap'],
    'perturbation': ['permutation', 'drop_column'],
    'approximation': ['kernelshap', 'lime'],
    'selection': ['boruta'],
}

# Quick subset for sanity checks
QUICK_METHODS = ['random_forest', 'permutation', 'treeshap', 'xgboost_gain']


# =============================================================================
# DATA LOADING
# =============================================================================

def load_logd_dataset() -> Tuple[List[str], np.ndarray]:
    """Load LogD from OpenADMET (HuggingFace)."""
    import pandas as pd
    from rdkit import Chem

    # Check cache first
    cache_dir = Path.home() / '.cache' / 'kirby'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / 'openadmet_train.csv'

    if not cached.exists():
        url = ("https://huggingface.co/datasets/openadmet/"
               "openadmet-expansionrx-challenge-data/resolve/main/expansion_data_train.csv")
        print(f"  Downloading OpenADMET data from HuggingFace...")
        import urllib.request
        urllib.request.urlretrieve(url, cached)

    df = pd.read_csv(cached)

    # Find LogD column
    logd_col = next((c for c in df.columns if 'LogD' in c), None)
    if logd_col is None:
        raise ValueError("Cannot find LogD column in OpenADMET data")

    # Validate SMILES and filter NaN
    valid_smiles = []
    valid_labels = []
    for smi, label in zip(df['SMILES'], df[logd_col]):
        if pd.isna(label) or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
            valid_labels.append(label)

    print(f"  Loaded {len(valid_smiles)} molecules from OpenADMET-LogD")
    return valid_smiles, np.array(valid_labels)


def load_qm9_homolumo(n_samples: int = 5000) -> Tuple[List[str], np.ndarray]:
    """Load HOMO-LUMO gap from QM9 (subset)."""
    try:
        from kirby.datasets.qm9 import load_qm9
        # property_idx=4 is 'gap' (HOMO-LUMO gap)
        data = load_qm9(n_samples=n_samples, property_idx=4)
        smiles = data['smiles']
        labels = data['labels']
        return smiles, labels
    except ImportError as e:
        print(f"  QM9 loader requires torch-geometric: {e}")
        # Fallback to CSV if available
        import pandas as pd
        path = Path(__file__).parent.parent / 'data' / 'qm9_gap.csv'
        if path.exists():
            df = pd.read_csv(path).head(n_samples)
            return df['smiles'].tolist(), df['gap'].values
        raise ImportError(
            "Could not load QM9 dataset. Either install torch-geometric or provide data/qm9_gap.csv"
        )


def generate_representations(smiles: List[str], rep_names: List[str]) -> Dict[str, np.ndarray]:
    """Generate molecular representations."""
    from kirby.representations.molecular import (
        create_ecfp4, create_maccs, create_pdv,
    )

    generators = {
        'ecfp4': create_ecfp4,
        'maccs': create_maccs,
        'pdv': create_pdv,
    }

    reps = {}
    for name in rep_names:
        if name in generators:
            print(f"  Generating {name}...")
            reps[name] = generators[name](smiles)

    return reps


def get_scaffold_groups(smiles: List[str]) -> np.ndarray:
    """Get Murcko scaffold groups for CV splitting."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except:
                scaffold = smi
        else:
            scaffold = smi
        scaffolds.append(scaffold)

    # Map to integers
    unique = list(set(scaffolds))
    scaffold_to_idx = {s: i for i, s in enumerate(unique)}
    return np.array([scaffold_to_idx[s] for s in scaffolds])


# =============================================================================
# IMPORTANCE COMPUTATION
# =============================================================================

def compute_importance_single_method(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    timeout: float = 300
) -> Tuple[np.ndarray, float, bool]:
    """
    Compute feature importance using a single method.

    Returns:
        (importance_scores, time_seconds, success)
    """
    from kirby.hybrid import compute_feature_importance

    start = time.time()
    try:
        # Wrap in dict for compute_feature_importance API
        rep_dict = {'features': X}
        scores = compute_feature_importance(rep_dict, y, method=method)
        elapsed = time.time() - start
        return scores['features'], elapsed, True
    except Exception as e:
        elapsed = time.time() - start
        print(f"    {method} failed: {e}")
        return np.zeros(X.shape[1]), elapsed, False


def compute_all_importance_methods(
    X: np.ndarray,
    y: np.ndarray,
    methods: List[str],
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Compute importance for all methods.

    Returns:
        {method: {'scores': array, 'time': float, 'success': bool}}
    """
    results = {}

    for method in methods:
        if verbose:
            print(f"  Computing {method}...", end=' ', flush=True)

        scores, elapsed, success = compute_importance_single_method(X, y, method)
        results[method] = {
            'scores': scores,
            'time': elapsed,
            'success': success,
        }

        if verbose:
            status = f"{elapsed:.1f}s" if success else "FAILED"
            print(status)

    return results


# =============================================================================
# AGREEMENT METRICS
# =============================================================================

def compute_rank_correlation(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """Spearman rank correlation between two importance score vectors."""
    if len(scores1) != len(scores2):
        return np.nan
    # Handle edge cases
    if np.std(scores1) == 0 or np.std(scores2) == 0:
        return np.nan
    corr, _ = stats.spearmanr(scores1, scores2)
    return corr


def compute_top_k_overlap(scores1: np.ndarray, scores2: np.ndarray, k: int = 50) -> float:
    """Jaccard overlap of top-K features between two rankings."""
    k = min(k, len(scores1), len(scores2))
    top1 = set(np.argsort(scores1)[-k:])
    top2 = set(np.argsort(scores2)[-k:])
    intersection = len(top1 & top2)
    union = len(top1 | top2)
    return intersection / union if union > 0 else 0.0


def compute_agreement_matrix(
    importance_results: Dict[str, Dict[str, Any]],
    metric: str = 'spearman'
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise agreement matrix between all methods.

    Returns:
        (matrix, method_names)
    """
    methods = [m for m, r in importance_results.items() if r['success']]
    n = len(methods)
    matrix = np.eye(n)

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                s1 = importance_results[m1]['scores']
                s2 = importance_results[m2]['scores']

                if metric == 'spearman':
                    val = compute_rank_correlation(s1, s2)
                elif metric == 'top_k':
                    val = compute_top_k_overlap(s1, s2, k=50)
                else:
                    val = np.nan

                matrix[i, j] = val
                matrix[j, i] = val

    return matrix, methods


def summarize_agreement(
    importance_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute agreement summary statistics.

    Returns:
        {
            'mean_correlation': float,
            'min_correlation': float,
            'max_correlation': float,
            'high_agreement': bool,  # True if mean > 0.8
            'correlation_matrix': array,
            'top_k_matrix': array,
            'methods': list,
        }
    """
    corr_matrix, methods = compute_agreement_matrix(importance_results, 'spearman')
    topk_matrix, _ = compute_agreement_matrix(importance_results, 'top_k')

    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(len(methods), k=1)
    corr_values = corr_matrix[triu_idx]
    corr_values = corr_values[~np.isnan(corr_values)]

    return {
        'mean_correlation': float(np.mean(corr_values)) if len(corr_values) > 0 else np.nan,
        'min_correlation': float(np.min(corr_values)) if len(corr_values) > 0 else np.nan,
        'max_correlation': float(np.max(corr_values)) if len(corr_values) > 0 else np.nan,
        'std_correlation': float(np.std(corr_values)) if len(corr_values) > 0 else np.nan,
        'high_agreement': float(np.mean(corr_values)) > 0.8 if len(corr_values) > 0 else False,
        'correlation_matrix': corr_matrix,
        'top_k_matrix': topk_matrix,
        'methods': methods,
    }


# =============================================================================
# AGGREGATION STRATEGIES
# =============================================================================

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range."""
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val - min_val < 1e-10:
        return np.ones_like(scores) / len(scores)
    return (scores - min_val) / (max_val - min_val)


def aggregate_simple_average(
    importance_results: Dict[str, Dict[str, Any]],
    methods: Optional[List[str]] = None
) -> np.ndarray:
    """Average normalized scores across methods."""
    if methods is None:
        methods = [m for m, r in importance_results.items() if r['success']]

    normalized = []
    for m in methods:
        if m in importance_results and importance_results[m]['success']:
            normalized.append(normalize_scores(importance_results[m]['scores']))

    if not normalized:
        raise ValueError("No valid importance scores to aggregate")

    return np.mean(normalized, axis=0)


def aggregate_rrf(
    importance_results: Dict[str, Dict[str, Any]],
    methods: Optional[List[str]] = None,
    k: int = 60
) -> np.ndarray:
    """
    Reciprocal Rank Fusion.

    RRF_score(feature) = sum over methods: 1 / (k + rank(feature))

    k is a smoothing constant (typically 60).
    """
    if methods is None:
        methods = [m for m, r in importance_results.items() if r['success']]

    n_features = None
    rrf_scores = None

    for m in methods:
        if m not in importance_results or not importance_results[m]['success']:
            continue

        scores = importance_results[m]['scores']
        if n_features is None:
            n_features = len(scores)
            rrf_scores = np.zeros(n_features)

        # Rank: higher score = lower rank number (rank 1 = best)
        ranks = n_features - np.argsort(np.argsort(scores))
        rrf_scores += 1.0 / (k + ranks)

    return rrf_scores if rrf_scores is not None else np.array([])


def aggregate_borda(
    importance_results: Dict[str, Dict[str, Any]],
    methods: Optional[List[str]] = None
) -> np.ndarray:
    """
    Borda count aggregation.

    Each method gives points: n_features - rank + 1 (top feature gets n_features points).
    """
    if methods is None:
        methods = [m for m, r in importance_results.items() if r['success']]

    n_features = None
    borda_scores = None

    for m in methods:
        if m not in importance_results or not importance_results[m]['success']:
            continue

        scores = importance_results[m]['scores']
        if n_features is None:
            n_features = len(scores)
            borda_scores = np.zeros(n_features)

        # Points: higher score = more points
        ranks = np.argsort(np.argsort(scores))  # 0 = lowest, n-1 = highest
        borda_scores += ranks + 1  # 1 to n_features

    return borda_scores if borda_scores is not None else np.array([])


def aggregate_voting(
    importance_results: Dict[str, Dict[str, Any]],
    methods: Optional[List[str]] = None,
    top_k: int = 100,
    min_votes: int = 3
) -> np.ndarray:
    """
    Voting/consensus: score = number of methods where feature is in top-K.

    Features with < min_votes get score 0.
    """
    if methods is None:
        methods = [m for m, r in importance_results.items() if r['success']]

    # Only count methods that actually succeeded
    valid_methods = [m for m in methods
                     if m in importance_results and importance_results[m]['success']]
    n_valid = len(valid_methods)

    # Clamp min_votes to n_valid so we don't zero out everything
    effective_min_votes = min(min_votes, max(n_valid, 1))

    n_features = None
    votes = None

    for m in valid_methods:
        scores = importance_results[m]['scores']
        if n_features is None:
            n_features = len(scores)
            votes = np.zeros(n_features)

        top_indices = np.argsort(scores)[-top_k:]
        votes[top_indices] += 1

    if votes is not None:
        votes[votes < effective_min_votes] = 0

    return votes if votes is not None else np.array([])


def aggregate_weighted_average(
    importance_results: Dict[str, Dict[str, Any]],
    weights: Dict[str, float]
) -> np.ndarray:
    """Weighted average of normalized scores."""
    n_features = None
    weighted_sum = None
    total_weight = 0.0

    for m, w in weights.items():
        if m not in importance_results or not importance_results[m]['success']:
            continue

        scores = importance_results[m]['scores']
        if n_features is None:
            n_features = len(scores)
            weighted_sum = np.zeros(n_features)

        weighted_sum += w * normalize_scores(scores)
        total_weight += w

    if weighted_sum is not None and total_weight > 0:
        return weighted_sum / total_weight
    return np.array([])


def aggregate_median_rank(
    importance_results: Dict[str, Dict[str, Any]],
    methods: Optional[List[str]] = None
) -> np.ndarray:
    """
    Median rank aggregation (Kemeny approximation).

    For each feature, compute median rank across methods.
    Return negative median rank (so higher = better, like importance).
    """
    if methods is None:
        methods = [m for m, r in importance_results.items() if r['success']]

    n_features = None
    all_ranks = []

    for m in methods:
        if m not in importance_results or not importance_results[m]['success']:
            continue

        scores = importance_results[m]['scores']
        if n_features is None:
            n_features = len(scores)

        # Rank: 1 = best (highest score)
        ranks = n_features - np.argsort(np.argsort(scores))
        all_ranks.append(ranks)

    if not all_ranks:
        return np.array([])

    # Median rank for each feature
    rank_matrix = np.array(all_ranks)  # (n_methods, n_features)
    median_ranks = np.median(rank_matrix, axis=0)

    # Return negative so higher = better
    return -median_ranks


def aggregate_geometric_mean(
    importance_results: Dict[str, Dict[str, Any]],
    methods: Optional[List[str]] = None
) -> np.ndarray:
    """
    Geometric mean of normalized scores.

    More conservative than arithmetic mean - requires agreement.
    """
    if methods is None:
        methods = [m for m, r in importance_results.items() if r['success']]

    normalized = []
    for m in methods:
        if m in importance_results and importance_results[m]['success']:
            scores = normalize_scores(importance_results[m]['scores'])
            # Add small epsilon to avoid log(0)
            normalized.append(scores + 1e-10)

    if not normalized:
        return np.array([])

    # Geometric mean = exp(mean(log(x)))
    log_scores = np.log(np.array(normalized))
    return np.exp(np.mean(log_scores, axis=0))


def aggregate_trimmed_mean(
    importance_results: Dict[str, Dict[str, Any]],
    methods: Optional[List[str]] = None,
    trim_fraction: float = 0.2
) -> np.ndarray:
    """
    Trimmed mean - remove top and bottom trim_fraction before averaging.

    More robust to outlier methods.
    """
    if methods is None:
        methods = [m for m, r in importance_results.items() if r['success']]

    normalized = []
    for m in methods:
        if m in importance_results and importance_results[m]['success']:
            normalized.append(normalize_scores(importance_results[m]['scores']))

    if not normalized:
        return np.array([])

    score_matrix = np.array(normalized)  # (n_methods, n_features)
    n_methods = score_matrix.shape[0]
    n_trim = int(n_methods * trim_fraction)

    if n_trim == 0 or n_methods - 2 * n_trim < 1:
        return np.mean(score_matrix, axis=0)

    # For each feature, sort scores and trim
    trimmed_means = np.zeros(score_matrix.shape[1])
    for i in range(score_matrix.shape[1]):
        sorted_scores = np.sort(score_matrix[:, i])
        trimmed_means[i] = np.mean(sorted_scores[n_trim:-n_trim] if n_trim > 0 else sorted_scores)

    return trimmed_means


def aggregate_rank_product(
    importance_results: Dict[str, Dict[str, Any]],
    methods: Optional[List[str]] = None
) -> np.ndarray:
    """
    Rank product method (from bioinformatics).

    Product of ranks across methods - features consistently ranked high get low product.
    Return negative log rank product (so higher = better).
    """
    if methods is None:
        methods = [m for m, r in importance_results.items() if r['success']]

    n_features = None
    log_rank_sum = None

    for m in methods:
        if m not in importance_results or not importance_results[m]['success']:
            continue

        scores = importance_results[m]['scores']
        if n_features is None:
            n_features = len(scores)
            log_rank_sum = np.zeros(n_features)

        # Rank: 1 = best
        ranks = n_features - np.argsort(np.argsort(scores))
        log_rank_sum += np.log(ranks + 1)  # +1 to avoid log(0)

    if log_rank_sum is None:
        return np.array([])

    # Return negative (lower rank product = better)
    return -log_rank_sum


def compute_cv_importance(
    X: np.ndarray,
    y: np.ndarray,
    scaffold_groups: np.ndarray,
    method: str,
    n_folds: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute importance across CV folds for stability analysis.

    Returns:
        (mean_importance, std_importance)
    """
    gkf = GroupKFold(n_splits=n_folds)
    fold_importances = []

    for train_idx, _ in gkf.split(X, y, groups=scaffold_groups):
        X_fold = X[train_idx]
        y_fold = y[train_idx]

        scores, _, success = compute_importance_single_method(X_fold, y_fold, method)
        if success:
            fold_importances.append(normalize_scores(scores))

    if not fold_importances:
        return np.zeros(X.shape[1]), np.ones(X.shape[1])

    fold_matrix = np.array(fold_importances)
    return np.mean(fold_matrix, axis=0), np.std(fold_matrix, axis=0)


def aggregate_cv_stable(
    X: np.ndarray,
    y: np.ndarray,
    scaffold_groups: np.ndarray,
    methods: List[str],
    n_folds: int = 3,
    stability_weight: float = 0.5
) -> np.ndarray:
    """
    Aggregate importance weighted by cross-validation stability.

    Features that are consistently important across folds get higher weight.
    Score = mean_importance * (1 - stability_weight * cv_std)
    """
    n_features = X.shape[1]
    weighted_scores = np.zeros(n_features)
    total_weight = 0.0

    for method in methods:
        mean_imp, std_imp = compute_cv_importance(X, y, scaffold_groups, method, n_folds)

        if np.sum(mean_imp) > 0:
            # Weight by stability (lower std = more stable = higher weight)
            stability = 1.0 - stability_weight * (std_imp / (np.max(std_imp) + 1e-10))
            weighted_scores += mean_imp * stability
            total_weight += 1.0

    if total_weight > 0:
        return weighted_scores / total_weight
    return np.zeros(n_features)


def aggregate_disagreement_aware(
    importance_results: Dict[str, Dict[str, Any]],
    methods: Optional[List[str]] = None,
    disagreement_penalty: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregation that tracks and penalizes high-disagreement features.

    Returns:
        (aggregated_scores, disagreement_scores)

    Features where methods disagree strongly get lower final scores.
    """
    if methods is None:
        methods = [m for m, r in importance_results.items() if r['success']]

    normalized = []
    for m in methods:
        if m in importance_results and importance_results[m]['success']:
            normalized.append(normalize_scores(importance_results[m]['scores']))

    if not normalized:
        return np.array([]), np.array([])

    score_matrix = np.array(normalized)
    mean_scores = np.mean(score_matrix, axis=0)
    std_scores = np.std(score_matrix, axis=0)

    # Normalize std to [0, 1]
    max_std = np.max(std_scores) if np.max(std_scores) > 0 else 1.0
    disagreement = std_scores / max_std

    # Final score penalizes disagreement
    final_scores = mean_scores * (1.0 - disagreement_penalty * disagreement)

    return final_scores, disagreement


# =============================================================================
# ALLOCATION STRATEGIES
# =============================================================================

def allocate_diversity_greedy(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    importance_scores: np.ndarray,
    budget: int = 100,
    lambda_diversity: float = 0.5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Diversity-penalized greedy selection.

    At each step, select feature that maximizes:
        importance[i] - lambda * max_correlation_with_selected[i]

    Returns:
        (selected_indices, info_dict)
    """
    n_features = X_train.shape[1]
    budget = min(budget, n_features)

    # Normalize importance
    imp_norm = normalize_scores(importance_scores)

    # Precompute correlation matrix (expensive but done once)
    X_combined = np.vstack([X_train, X_val])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Use absolute correlation
    corr_matrix = np.abs(np.corrcoef(X_scaled.T))
    np.fill_diagonal(corr_matrix, 0)  # Don't correlate with self

    selected = []
    remaining = set(range(n_features))

    for _ in range(budget):
        if not remaining:
            break

        best_score = -np.inf
        best_idx = None

        for idx in remaining:
            # Diversity penalty: max correlation with already selected
            if selected:
                diversity_penalty = np.max(corr_matrix[idx, selected])
            else:
                diversity_penalty = 0.0

            score = imp_norm[idx] - lambda_diversity * diversity_penalty

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return np.array(selected), {
        'method': 'diversity_greedy',
        'lambda': lambda_diversity,
        'n_selected': len(selected),
    }


def allocate_two_stage(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    budget: int = 100,
    stage1_method: str = 'random_forest',
    stage2_method: str = 'permutation',
    stage1_multiplier: int = 3
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Two-stage allocation: fast screening then precise selection.

    Stage 1: Use fast method to reduce to stage1_multiplier * budget candidates
    Stage 2: Use slow method on candidates for final selection

    Returns:
        (selected_indices, info_dict)
    """
    n_features = X_train.shape[1]
    stage1_budget = min(stage1_multiplier * budget, n_features)

    # Stage 1: Fast screening
    stage1_scores, stage1_time, stage1_success = compute_importance_single_method(
        X_train, y_train, stage1_method
    )

    if not stage1_success:
        # Fallback to random
        stage1_candidates = np.random.choice(n_features, stage1_budget, replace=False)
    else:
        stage1_candidates = np.argsort(stage1_scores)[-stage1_budget:]

    # Stage 2: Precise selection on candidates only
    X_train_subset = X_train[:, stage1_candidates]

    stage2_scores, stage2_time, stage2_success = compute_importance_single_method(
        X_train_subset, y_train, stage2_method
    )

    if not stage2_success:
        # Fallback to stage1 ranking
        selected_local = np.argsort(stage1_scores[stage1_candidates])[-budget:]
    else:
        selected_local = np.argsort(stage2_scores)[-budget:]

    # Map back to original indices
    selected = stage1_candidates[selected_local]

    return selected, {
        'method': 'two_stage',
        'stage1_method': stage1_method,
        'stage2_method': stage2_method,
        'stage1_time': stage1_time,
        'stage2_time': stage2_time,
        'stage1_candidates': len(stage1_candidates),
    }


def allocate_stability_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    budget: int = 100,
    n_bootstrap: int = 20,
    importance_method: str = 'random_forest',
    selection_threshold: float = 0.6
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Stability selection via bootstrapping.

    Run importance on B bootstrap samples, keep features that appear
    in top-K for >= threshold fraction of samples.

    Returns:
        (selected_indices, info_dict)
    """
    n_samples, n_features = X_train.shape
    top_k = min(budget * 2, n_features)  # Consider 2x budget as "top"

    selection_counts = np.zeros(n_features)

    for b in range(n_bootstrap):
        # Bootstrap sample
        boot_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_train[boot_idx]
        y_boot = y_train[boot_idx]

        # Compute importance
        scores, _, success = compute_importance_single_method(X_boot, y_boot, importance_method)

        if success:
            top_indices = np.argsort(scores)[-top_k:]
            selection_counts[top_indices] += 1

    # Selection probability
    selection_prob = selection_counts / n_bootstrap

    # Select features above threshold, ranked by probability
    above_threshold = np.where(selection_prob >= selection_threshold)[0]

    if len(above_threshold) > budget:
        # Take top budget by selection probability
        prob_ranking = np.argsort(selection_prob[above_threshold])[::-1]
        selected = above_threshold[prob_ranking[:budget]]
    elif len(above_threshold) > 0:
        selected = above_threshold
    else:
        # Fallback: take top by selection probability regardless of threshold
        selected = np.argsort(selection_prob)[-budget:]

    return selected, {
        'method': 'stability_selection',
        'n_bootstrap': n_bootstrap,
        'importance_method': importance_method,
        'threshold': selection_threshold,
        'n_above_threshold': len(above_threshold),
        'selection_probabilities': selection_prob,
    }


# =============================================================================
# EVALUATION
# =============================================================================

def get_model(model_name: str):
    """Get model instance by name."""
    if model_name == 'rf':
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'mlp':
        return MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    elif model_name == 'lightgbm':
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1)
        except ImportError:
            print("  LightGBM not available, falling back to RF")
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    selected_indices: np.ndarray,
    scaffold_groups: np.ndarray,
    model_name: str,
    n_folds: int = 3
) -> Dict[str, float]:
    """
    Evaluate a feature selection via cross-validation.

    Returns:
        {'r2_mean': float, 'r2_std': float}
    """
    X_selected = X[:, selected_indices]

    gkf = GroupKFold(n_splits=n_folds)
    scores = []

    for train_idx, val_idx in gkf.split(X_selected, y, groups=scaffold_groups):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = get_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(r2_score(y_val, y_pred))

    return {
        'r2_mean': float(np.mean(scores)),
        'r2_std': float(np.std(scores)),
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    datasets: List[str] = field(default_factory=lambda: ['logd', 'qm9'])
    models: List[str] = field(default_factory=lambda: ['rf', 'mlp', 'lightgbm'])
    importance_methods: List[str] = field(default_factory=lambda: ALL_IMPORTANCE_METHODS)
    budgets: List[int] = field(default_factory=lambda: [50, 100, 200])
    n_folds: int = 3
    n_bootstrap: int = 20  # For stability selection
    qm9_samples: int = 5000
    reps: List[str] = field(default_factory=lambda: ['ecfp4', 'maccs', 'pdv'])


def run_benchmark(config: BenchmarkConfig, output_dir: str = 'results/importance_benchmark'):
    """Run full benchmark."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        'config': {
            'datasets': config.datasets,
            'models': config.models,
            'importance_methods': config.importance_methods,
            'budgets': config.budgets,
            'timestamp': datetime.now().isoformat(),
        },
        'datasets': {},
    }

    for dataset_name in config.datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name.upper()}")
        print('='*60)

        # Load data
        print("\nLoading data...")
        try:
            if dataset_name == 'logd':
                smiles, labels = load_logd_dataset()
            elif dataset_name == 'qm9':
                smiles, labels = load_qm9_homolumo(config.qm9_samples)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        except Exception as e:
            print(f"  Failed to load {dataset_name}: {e}")
            continue

        print(f"  Loaded {len(smiles)} samples")

        # Generate representations
        print("\nGenerating representations...")
        base_reps = generate_representations(smiles, config.reps)

        # Concatenate for single-feature-set analysis
        X = np.hstack([base_reps[r] for r in sorted(base_reps.keys())])
        y = labels
        print(f"  Combined feature matrix: {X.shape}")

        # Get scaffold groups
        print("\nComputing scaffold groups...")
        scaffold_groups = get_scaffold_groups(smiles)
        n_scaffolds = len(np.unique(scaffold_groups))
        print(f"  {n_scaffolds} unique scaffolds")

        dataset_results = {
            'n_samples': len(smiles),
            'n_features': X.shape[1],
            'n_scaffolds': n_scaffolds,
        }

        # =================================================================
        # PHASE 1: Compute all importance methods
        # =================================================================
        print("\n" + "-"*40)
        print("PHASE 1: Computing importance methods")
        print("-"*40)

        importance_results = compute_all_importance_methods(X, y, config.importance_methods)

        # Timing summary
        timing = {m: r['time'] for m, r in importance_results.items()}
        success = {m: r['success'] for m, r in importance_results.items()}

        print("\nTiming summary:")
        for m in sorted(timing.keys(), key=lambda x: timing[x]):
            status = "OK" if success[m] else "FAIL"
            print(f"  {m}: {timing[m]:.1f}s [{status}]")

        dataset_results['importance_timing'] = timing
        dataset_results['importance_success'] = success

        # =================================================================
        # PHASE 2: Compute agreement metrics
        # =================================================================
        print("\n" + "-"*40)
        print("PHASE 2: Computing agreement metrics")
        print("-"*40)

        agreement = summarize_agreement(importance_results)

        print(f"\nAgreement summary:")
        print(f"  Mean Spearman correlation: {agreement['mean_correlation']:.3f}")
        print(f"  Min correlation: {agreement['min_correlation']:.3f}")
        print(f"  Max correlation: {agreement['max_correlation']:.3f}")
        print(f"  Std correlation: {agreement['std_correlation']:.3f}")
        print(f"  High agreement (>0.8): {agreement['high_agreement']}")

        dataset_results['agreement'] = {
            'mean_correlation': agreement['mean_correlation'],
            'min_correlation': agreement['min_correlation'],
            'max_correlation': agreement['max_correlation'],
            'std_correlation': agreement['std_correlation'],
            'high_agreement': agreement['high_agreement'],
        }

        # =================================================================
        # PHASE 3: Test aggregation strategies
        # =================================================================
        print("\n" + "-"*40)
        print("PHASE 3: Testing aggregation strategies")
        print("-"*40)

        aggregation_results = {}

        # Define aggregation strategies to test
        strategies = {
            # === SIMPLE AVERAGING (different subsets) ===
            'avg_all': lambda: aggregate_simple_average(importance_results),
            'avg_tree': lambda: aggregate_simple_average(
                importance_results, METHOD_CATEGORIES['tree_based']),
            'avg_gradient': lambda: aggregate_simple_average(
                importance_results, METHOD_CATEGORIES['gradient_based']),
            'avg_perturbation': lambda: aggregate_simple_average(
                importance_results, METHOD_CATEGORIES['perturbation']),
            'avg_fast3': lambda: aggregate_simple_average(
                importance_results, ['random_forest', 'xgboost_gain', 'lightgbm_gain']),
            'avg_diverse3': lambda: aggregate_simple_average(
                importance_results, ['random_forest', 'permutation', 'treeshap']),

            # === RANK-BASED AGGREGATION ===
            'rrf_all': lambda: aggregate_rrf(importance_results),
            'rrf_tree': lambda: aggregate_rrf(importance_results, METHOD_CATEGORIES['tree_based']),
            'rrf_k30': lambda: aggregate_rrf(importance_results, k=30),  # More aggressive
            'rrf_k100': lambda: aggregate_rrf(importance_results, k=100),  # More conservative
            'borda_all': lambda: aggregate_borda(importance_results),
            'borda_tree': lambda: aggregate_borda(importance_results, METHOD_CATEGORIES['tree_based']),
            'median_rank_all': lambda: aggregate_median_rank(importance_results),
            'median_rank_tree': lambda: aggregate_median_rank(
                importance_results, METHOD_CATEGORIES['tree_based']),
            'rank_product_all': lambda: aggregate_rank_product(importance_results),

            # === VOTING/CONSENSUS ===
            'voting_k100_m3': lambda: aggregate_voting(importance_results, top_k=100, min_votes=3),
            'voting_k50_m3': lambda: aggregate_voting(importance_results, top_k=50, min_votes=3),
            'voting_k100_m5': lambda: aggregate_voting(importance_results, top_k=100, min_votes=5),
            'voting_k50_m5': lambda: aggregate_voting(importance_results, top_k=50, min_votes=5),

            # === ROBUST AVERAGING ===
            'geometric_mean_all': lambda: aggregate_geometric_mean(importance_results),
            'trimmed_mean_10': lambda: aggregate_trimmed_mean(importance_results, trim_fraction=0.1),
            'trimmed_mean_20': lambda: aggregate_trimmed_mean(importance_results, trim_fraction=0.2),

            # === DISAGREEMENT-AWARE ===
            'disagreement_aware_0.3': lambda: aggregate_disagreement_aware(
                importance_results, disagreement_penalty=0.3)[0],
            'disagreement_aware_0.5': lambda: aggregate_disagreement_aware(
                importance_results, disagreement_penalty=0.5)[0],
            'disagreement_aware_0.7': lambda: aggregate_disagreement_aware(
                importance_results, disagreement_penalty=0.7)[0],

            # === CV-STABLE (expensive but thorough) ===
            'cv_stable_tree': lambda: aggregate_cv_stable(
                X, y, scaffold_groups, METHOD_CATEGORIES['tree_based'][:3], n_folds=3),
            'cv_stable_fast': lambda: aggregate_cv_stable(
                X, y, scaffold_groups, ['random_forest', 'xgboost_gain'], n_folds=3),

            # === WEIGHTED AVERAGES (weights based on typical reliability) ===
            'weighted_tree_favor': lambda: aggregate_weighted_average(importance_results, {
                'random_forest': 1.0, 'treeshap': 1.0, 'xgboost_gain': 0.8,
                'permutation': 0.6, 'lightgbm_gain': 0.7,
            }),
            'weighted_shap_favor': lambda: aggregate_weighted_average(importance_results, {
                'treeshap': 1.0, 'kernelshap': 0.9, 'deepliftshap': 0.8,
                'random_forest': 0.6, 'permutation': 0.7,
            }),
        }

        # Also test single methods as baselines
        for method in config.importance_methods:
            if importance_results.get(method, {}).get('success', False):
                strategies[f'single_{method}'] = lambda m=method: importance_results[m]['scores']

        print(f"\nEvaluating {len(strategies)} aggregation strategies...")

        for budget in config.budgets:
            print(f"\n  Budget: {budget}")
            budget_results = {}

            for strategy_name, get_scores in strategies.items():
                try:
                    # Time the aggregation
                    agg_start = time.time()
                    agg_scores = get_scores()
                    agg_time = time.time() - agg_start

                    if len(agg_scores) == 0:
                        continue

                    # Select top features
                    selected = np.argsort(agg_scores)[-budget:]

                    # Evaluate with each model
                    model_scores = {}
                    for model_name in config.models:
                        eval_result = evaluate_feature_selection(
                            X, y, selected, scaffold_groups, model_name, config.n_folds
                        )
                        model_scores[model_name] = eval_result

                    budget_results[strategy_name] = {
                        'scores': model_scores,
                        'agg_time': agg_time,
                    }

                    # Print summary
                    avg_r2 = np.mean([m['r2_mean'] for m in model_scores.values()])
                    time_str = f", agg={agg_time:.1f}s" if agg_time > 0.1 else ""
                    print(f"    {strategy_name}: avg R²={avg_r2:.4f}{time_str}")

                except Exception as e:
                    print(f"    {strategy_name}: FAILED - {e}")

            aggregation_results[budget] = budget_results

        dataset_results['aggregation'] = aggregation_results

        # =================================================================
        # PHASE 4: Test allocation strategies
        # =================================================================
        print("\n" + "-"*40)
        print("PHASE 4: Testing allocation strategies")
        print("-"*40)

        allocation_results = {}

        # Need train/val split for allocation strategies
        gkf = GroupKFold(n_splits=config.n_folds)
        train_idx, val_idx = next(gkf.split(X, y, groups=scaffold_groups))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build per-rep train/val dicts for hybrid.py allocators
        rep_dict_train = {r: base_reps[r][train_idx] for r in sorted(base_reps.keys())}
        rep_dict_val = {r: base_reps[r][val_idx] for r in sorted(base_reps.keys())}

        # Compute rep offsets in concatenated X for index mapping
        rep_offsets = {}
        offset = 0
        for r in sorted(base_reps.keys()):
            rep_offsets[r] = offset
            offset += base_reps[r].shape[1]

        def allocation_to_global_indices(alloc_dict, rep_dict, labels):
            """Convert {rep: n_features} allocation to global indices in concatenated X."""
            from kirby.hybrid import compute_feature_importance
            importance = compute_feature_importance(rep_dict, labels, method='random_forest')
            global_indices = []
            for rep_name, n_feat in alloc_dict.items():
                if n_feat <= 0:
                    continue
                scores = importance[rep_name]
                n_feat = min(n_feat, len(scores))
                local_idx = np.argsort(scores)[-n_feat:]
                global_indices.extend(local_idx + rep_offsets[rep_name])
            return np.array(global_indices, dtype=int)

        def feature_info_to_global_indices(feature_info):
            """Convert feature_info with per-rep selected_indices to global indices."""
            global_indices = []
            for rep_name, info in feature_info.items():
                if rep_name == 'allocation' or not isinstance(info, dict):
                    continue
                if 'selected_indices' in info:
                    global_indices.extend(info['selected_indices'] + rep_offsets[rep_name])
            return np.array(global_indices, dtype=int)

        # Get base importance for diversity_greedy
        base_importance, _, _ = compute_importance_single_method(X_train, y_train, 'random_forest')

        # Import hybrid.py allocation functions
        from kirby.hybrid import (
            allocate_greedy_forward, allocate_greedy_feature,
            allocate_performance_weighted, allocate_mrmr,
        )

        for budget in config.budgets:
            print(f"\n  Budget: {budget}")
            budget_results = {}

            # --- Flat-X allocation strategies (work on concatenated features) ---
            flat_strategies = {
                'diversity_greedy_0.3': lambda b=budget: allocate_diversity_greedy(
                    X_train, X_val, y_train, y_val, base_importance, b, lambda_diversity=0.3),
                'diversity_greedy_0.5': lambda b=budget: allocate_diversity_greedy(
                    X_train, X_val, y_train, y_val, base_importance, b, lambda_diversity=0.5),
                'diversity_greedy_0.7': lambda b=budget: allocate_diversity_greedy(
                    X_train, X_val, y_train, y_val, base_importance, b, lambda_diversity=0.7),
                'two_stage_rf_perm': lambda b=budget: allocate_two_stage(
                    X_train, X_val, y_train, y_val, b, 'random_forest', 'permutation'),
                'two_stage_rf_shap': lambda b=budget: allocate_two_stage(
                    X_train, X_val, y_train, y_val, b, 'random_forest', 'treeshap'),
                'stability_selection': lambda b=budget: allocate_stability_selection(
                    X_train, X_val, y_train, y_val, b, config.n_bootstrap, 'random_forest'),
            }

            for strategy_name, run_strategy in flat_strategies.items():
                try:
                    start = time.time()
                    selected, info = run_strategy()
                    elapsed = time.time() - start

                    model_scores = {}
                    for model_name in config.models:
                        eval_result = evaluate_feature_selection(
                            X, y, selected, scaffold_groups, model_name, config.n_folds
                        )
                        model_scores[model_name] = eval_result

                    budget_results[strategy_name] = {
                        'scores': model_scores,
                        'time': elapsed,
                        'info': {k: v for k, v in info.items() if k != 'selection_probabilities'},
                    }

                    avg_r2 = np.mean([m['r2_mean'] for m in model_scores.values()])
                    print(f"    {strategy_name}: avg R²={avg_r2:.4f}, time={elapsed:.1f}s")

                except Exception as e:
                    print(f"    {strategy_name}: FAILED - {e}")

            # --- Multi-rep allocation strategies from hybrid.py ---
            rep_strategies = {
                'greedy_forward': lambda b=budget: ('feature_info',
                    allocate_greedy_forward(rep_dict_train, rep_dict_val, y_train, y_val,
                                           total_budget=b, step_size=10)),
                'greedy_feature': lambda b=budget: ('feature_info',
                    allocate_greedy_feature(rep_dict_train, rep_dict_val, y_train, y_val,
                                           total_budget=b)),
                'performance_weighted': lambda b=budget: ('allocation',
                    allocate_performance_weighted(rep_dict_train, rep_dict_val, y_train, y_val,
                                                 total_budget=b)),
                'mrmr': lambda b=budget: ('allocation',
                    allocate_mrmr(rep_dict_train, rep_dict_val, y_train, y_val,
                                  total_budget=b)),
            }

            for strategy_name, run_strategy in rep_strategies.items():
                try:
                    start = time.time()
                    result_type, result = run_strategy()
                    elapsed = time.time() - start

                    if result_type == 'feature_info':
                        alloc, history, feature_info = result
                        selected = feature_info_to_global_indices(feature_info)
                        info = {'method': strategy_name, 'allocation': alloc}
                    else:
                        alloc = result
                        selected = allocation_to_global_indices(alloc, rep_dict_train, y_train)
                        info = {'method': strategy_name, 'allocation': alloc}

                    if len(selected) == 0:
                        print(f"    {strategy_name}: FAILED - no features selected")
                        continue

                    model_scores = {}
                    for model_name in config.models:
                        eval_result = evaluate_feature_selection(
                            X, y, selected, scaffold_groups, model_name, config.n_folds
                        )
                        model_scores[model_name] = eval_result

                    budget_results[strategy_name] = {
                        'scores': model_scores,
                        'time': elapsed,
                        'info': info,
                    }

                    avg_r2 = np.mean([m['r2_mean'] for m in model_scores.values()])
                    print(f"    {strategy_name}: avg R²={avg_r2:.4f}, time={elapsed:.1f}s")

                except Exception as e:
                    print(f"    {strategy_name}: FAILED - {e}")

            # Baseline: simple RF importance on concatenated X
            baseline_selected = np.argsort(base_importance)[-budget:]
            baseline_scores = {}
            for model_name in config.models:
                baseline_scores[model_name] = evaluate_feature_selection(
                    X, y, baseline_selected, scaffold_groups, model_name, config.n_folds
                )
            budget_results['baseline_rf'] = {'scores': baseline_scores, 'time': 0}
            avg_r2 = np.mean([m['r2_mean'] for m in baseline_scores.values()])
            print(f"    baseline_rf: avg R²={avg_r2:.4f}")

            allocation_results[budget] = budget_results

        dataset_results['allocation'] = allocation_results

        all_results['datasets'][dataset_name] = dataset_results

    # =================================================================
    # Save results
    # =================================================================
    output_file = output_path / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print('='*60)

    # Print summary
    print_benchmark_summary(all_results)

    return all_results


def print_benchmark_summary(results: Dict[str, Any]):
    """Print human-readable summary of benchmark results."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    for dataset_name, dataset_results in results.get('datasets', {}).items():
        print(f"\n{dataset_name.upper()}")
        print("-"*40)

        # Agreement
        agreement = dataset_results.get('agreement', {})
        print(f"  Agreement: mean_corr={agreement.get('mean_correlation', 'N/A'):.3f}, "
              f"high_agreement={agreement.get('high_agreement', 'N/A')}")

        # Best aggregation strategy
        aggregation = dataset_results.get('aggregation', {})
        if aggregation:
            for budget, budget_results in aggregation.items():
                best_strategy = None
                best_r2 = -np.inf

                for strategy, result in budget_results.items():
                    if isinstance(result, dict):
                        # Handle both old format (direct scores) and new format ({'scores': ...})
                        scores_dict = result.get('scores', result) if 'scores' in result else result
                        if isinstance(scores_dict, dict):
                            r2_values = [m.get('r2_mean', 0) for m in scores_dict.values()
                                        if isinstance(m, dict) and 'r2_mean' in m]
                            if r2_values:
                                avg_r2 = np.mean(r2_values)
                                if avg_r2 > best_r2:
                                    best_r2 = avg_r2
                                    best_strategy = strategy

                print(f"  Aggregation (budget={budget}): best={best_strategy} (R²={best_r2:.4f})")

        # Best allocation strategy
        allocation = dataset_results.get('allocation', {})
        if allocation:
            for budget, budget_results in allocation.items():
                best_strategy = None
                best_r2 = -np.inf

                for strategy, result in budget_results.items():
                    if isinstance(result, dict) and 'scores' in result:
                        avg_r2 = np.mean([m.get('r2_mean', 0) for m in result['scores'].values()
                                         if isinstance(m, dict)])
                        if avg_r2 > best_r2:
                            best_r2 = avg_r2
                            best_strategy = strategy

                print(f"  Allocation (budget={budget}): best={best_strategy} (R²={best_r2:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Benchmark importance and aggregation methods')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer methods')
    parser.add_argument('--full', action='store_true', help='Full benchmark (default)')
    parser.add_argument('--output', type=str, default='tests/results/importance_benchmark',
                       help='Output directory')
    parser.add_argument('--datasets', type=str, default='logd,qm9',
                       help='Comma-separated list of datasets')
    parser.add_argument('--models', type=str, default='rf,mlp,lightgbm',
                       help='Comma-separated list of models')

    args = parser.parse_args()

    if args.quick:
        config = BenchmarkConfig(
            datasets=args.datasets.split(','),
            models=['rf'],  # Just RF for quick test
            importance_methods=QUICK_METHODS,
            budgets=[50],
            n_folds=2,
            n_bootstrap=5,
            qm9_samples=1000,
            reps=['ecfp4', 'pdv'],
        )
    else:
        config = BenchmarkConfig(
            datasets=args.datasets.split(','),
            models=args.models.split(','),
            importance_methods=ALL_IMPORTANCE_METHODS,
            budgets=[50, 100, 200],
            n_folds=3,
            n_bootstrap=20,
            qm9_samples=5000,
            reps=['ecfp4', 'maccs', 'pdv'],
        )

    run_benchmark(config, args.output)


if __name__ == '__main__':
    main()
