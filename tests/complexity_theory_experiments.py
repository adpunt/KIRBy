#!/usr/bin/env python3
"""
Complexity Theory Experiments for Hybrid Molecular Representations

Tests the core thesis: there's an optimal pairing between model complexity and
representation complexity — the hybrid methodology discovers it automatically.

Experiments:
  1. Representation Characterization — profile each rep's complexity
  2. Complexity Matching Grid — model complexity × rep complexity optimal pairings
  3. Cross-Context Analysis — optimal config changes across datasets
  4. Importance Weights as Learned Metrics — Phase 5 bridge
  5. Learning Curves — hybrid advantage vs sample size
  6. Within-Family Complexity Gradient — vary capacity within model families
  7. Data Complexity via Scaffold Partitioning — optimal config by subpopulation
  8. Uncertainty Calibration — conformal prediction at optimal vs suboptimal complexity
  9. Formal Complexity Analysis — ANOVA, information retention, effective rank, bias-variance
  10. Noise Robustness & Feature Stability — NDS across noise strategies, feature overlap vs noise

Datasets: logd, caco2, herg, qm9 (via load_dataset from molecular wrapper)
Models: RF, kNN, ridge, XGBoost, MLP, LightGBM (ordered by complexity)

Usage:
    python tests/complexity_theory_experiments.py --mode quick --experiments 1 2 6 9 10
    python tests/complexity_theory_experiments.py --mode full --experiments 1 2 3 4 5 6 7 8 9 10
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.special import digamma
from scipy.stats import spearmanr, f as f_dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from kirby.hybrid import create_hybrid, apply_feature_selection, compute_feature_importance
from kirby.find_best_hybrid import find_best_hybrid, SearchResult, _resolve_models

try:
    from noiseInject import NoiseInjectorRegression
    HAS_NOISEINJECT = True
except ImportError:
    HAS_NOISEINJECT = False

warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING (domain-aware: molecular or antibody)
# =============================================================================

# Import from molecular wrapper (default)
sys.path.insert(0, str(Path(__file__).parent))
from find_best_hybrid_molecular import (
    load_dataset as load_molecular_dataset,
    get_representations as get_molecular_representations,
    assign_scaffold_groups,
)

# Antibody dataset prefixes
_ANTIBODY_PREFIXES = ('flab_', 'abcov_', 'sabdab_', 'abagym_', 'abbind', 'ddg_',
                      'ginkgo_', 'tz_her2', 'humanness_', 'graphinity_')


def _is_antibody_dataset(name: str) -> bool:
    return any(name.lower().startswith(p) for p in _ANTIBODY_PREFIXES)


def load_data(dataset_name: str, rep_preset: str = 'fast',
              delta: bool = False, augment: bool = False) -> dict:
    """Load dataset, generate representations, assign groups.

    Domain-aware: detects antibody dataset names (flab_*, abcov_*, sabdab_*)
    and dispatches to the antibody wrapper. Otherwise uses molecular wrapper.

    Returns dict with keys: labels, reps, groups, n_groups, task_type,
    augmentations, plus domain-specific keys (smiles or heavy_seqs, etc.).
    """
    if _is_antibody_dataset(dataset_name):
        from antibody_baseline_survey import load_dataset as load_ab_dataset
        from kirby.representations.antibody import ANTIBODY_REPS, ANTIBODY_FAST_REPS

        data = load_ab_dataset(dataset_name)
        heavy_seqs = data['heavy_seqs']
        light_seqs = data['light_seqs']
        labels = data['labels']
        groups = data['groups']
        n_groups = data['n_groups']

        if rep_preset in ('fast', 'skip_lm'):
            rep_names = sorted(ANTIBODY_FAST_REPS)
        else:
            rep_names = sorted(ANTIBODY_REPS.keys())

        reps = {}
        for rep_name in rep_names:
            try:
                reps[rep_name] = ANTIBODY_REPS[rep_name](heavy_seqs, light_seqs)
            except Exception as e:
                print(f"  WARNING: {rep_name} failed: {e}")

        print(f"  {dataset_name}: {len(heavy_seqs)} antibodies, {n_groups} groups, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        return {
            'heavy_seqs': heavy_seqs,
            'light_seqs': light_seqs,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_scaffolds': n_groups,
            'n_groups': n_groups,
            'task_type': data['task_type'],
            'domain': 'antibody',
            'augmentations': None,
        }
    else:
        smiles, labels, task_type = load_molecular_dataset(dataset_name)
        reps = get_molecular_representations(smiles, rep_preset)
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
            'domain': 'molecular',
            'augmentations': None,
        }


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def compute_intrinsic_dimensionality(X: np.ndarray) -> Dict[str, Any]:
    """Compute intrinsic dimensionality using PCA and MLE methods."""
    results = {'nominal_dim': X.shape[1]}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n, d = X_scaled.shape

    if d > 2000:
        # Use randomized SVD for high-dim reps (e.g. graph_kernel ~71K)
        from sklearn.decomposition import TruncatedSVD
        max_comp = min(500, n - 1)
        svd = TruncatedSVD(n_components=max_comp, random_state=42)
        svd.fit(X_scaled)
        cumvar = np.cumsum(svd.explained_variance_ratio_)
        for threshold in [0.90, 0.95, 0.99]:
            key = f'pca_{int(threshold * 100)}'
            idx = np.searchsorted(cumvar, threshold)
            results[key] = int(min(idx + 1, max_comp)) if idx < len(cumvar) else max_comp
    else:
        # Original PCA path for manageable dims
        n_components = min(n, d)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        for threshold in [0.90, 0.95, 0.99]:
            key = f'pca_{int(threshold * 100)}'
            intrinsic_dim = np.searchsorted(cumvar, threshold) + 1
            results[key] = int(min(intrinsic_dim, d))

    # MLE: pre-reduce if high-dim
    X_for_mle = X_scaled
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        X_for_mle = TruncatedSVD(n_components=min(200, n - 1), random_state=42).fit_transform(X_scaled)
    try:
        results['mle'] = float(estimate_intrinsic_dim_mle(X_for_mle, k=10))
    except Exception:
        results['mle'] = None

    if results.get('pca_95'):
        results['compression_ratio'] = X.shape[1] / results['pca_95']

    return results


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


def estimate_mutual_information_knn(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Estimate I(X; Y) using KSG estimator (Kraskov et al. 2004)."""
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


def fit_learning_curve(n_samples: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Fit asymptotic model: R² = R²_max - a/n^b."""
    from scipy.optimize import curve_fit

    def model(n, r2_max, a, b):
        return r2_max - a / np.power(n, b)

    try:
        p0 = [np.max(scores), 1.0, 0.5]
        bounds = ([0, 0, 0.1], [1, 100, 2])
        popt, _ = curve_fit(model, n_samples, scores, p0=p0, bounds=bounds, maxfev=5000)
        r2_max, a, b = popt

        predicted = model(n_samples, *popt)
        residuals = scores - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((scores - np.mean(scores)) ** 2)
        fit_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {'r2_max': float(r2_max), 'a': float(a), 'b': float(b), 'fit_r2': float(fit_r2)}
    except Exception as e:
        return {'error': str(e), 'r2_max': float(np.max(scores))}


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


def compute_dataset_complexity(reps: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, float]:
    """Compute dataset complexity metrics (Lorena et al. 2018 inspired).

    Returns dict of complexity metrics for cross-dataset correlation analysis.
    Metrics:
        erank: effective rank of X_concat (Roy & Vetterli 2007)
        participation_ratio: (sum(λ))²/sum(λ²) of covariance eigenvalues
        n_over_d: sample-to-feature ratio
        n_over_erank: effective sample-to-feature ratio
        C1: max |Spearman(feature, y)| across features (Lorena C1)
        C2: mean |Spearman(feature, y)| across features (Lorena C2)
        L1: mean absolute residual of Ridge fit (Lorena L1 adapted)
        S3: 1-NN leave-one-out MAE normalised by y range (Lorena S3 adapted)
        target_entropy: Shannon entropy of discretised y (20 bins)
        label_std: standard deviation of y
    """
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    n, d = X_concat.shape
    y = labels

    metrics = {'n_samples': n, 'n_features': d}

    # --- Structural metrics ---
    erank = compute_effective_rank(X_concat)
    metrics['erank'] = erank
    metrics['n_over_d'] = n / d
    metrics['n_over_erank'] = n / erank

    # Participation ratio from covariance eigenvalues
    X_scaled = StandardScaler().fit_transform(X_concat)
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(500, n - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(X_scaled)
        s = svd.singular_values_
    else:
        _, s, _ = np.linalg.svd(X_scaled, full_matrices=False)
    eigvals = s ** 2 / n
    eigvals = eigvals[eigvals > 1e-10]
    metrics['participation_ratio'] = float((eigvals.sum()) ** 2 / (eigvals ** 2).sum())

    # --- Feature-target correlation (Lorena C1, C2) ---
    # Sample features if d > 1000 to keep computation tractable
    max_features = min(d, 1000)
    if d > max_features:
        rng = np.random.RandomState(42)
        feat_idx = rng.choice(d, max_features, replace=False)
        X_sub = X_concat[:, feat_idx]
    else:
        X_sub = X_concat

    abs_corrs = []
    for j in range(X_sub.shape[1]):
        col = X_sub[:, j]
        if col.std() < 1e-10:
            abs_corrs.append(0.0)
            continue
        rho, _ = spearmanr(col, y)
        abs_corrs.append(abs(rho) if not np.isnan(rho) else 0.0)
    abs_corrs = np.array(abs_corrs)
    metrics['C1'] = float(np.max(abs_corrs))  # best single feature
    metrics['C2'] = float(np.mean(abs_corrs))  # average feature quality

    # --- Linearity: Ridge fit residual (Lorena L1 adapted) ---
    # Pre-reduce if high-dim (Ridge needs O(d²) for d > n, or O(n*d) otherwise)
    X_for_model = X_scaled
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        X_for_model = TruncatedSVD(n_components=min(500, n - 1), random_state=42).fit_transform(X_scaled)
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_for_model, y)
    y_pred_linear = ridge.predict(X_for_model)
    y_range = y.max() - y.min()
    metrics['L1'] = float(np.mean(np.abs(y - y_pred_linear)) / y_range) if y_range > 0 else 0.0

    # --- Smoothness: 1-NN LOO error (Lorena S3 adapted) ---
    # Use pre-reduced X for kNN if high-dim (brute-force kNN is O(n²d))
    nn = NearestNeighbors(n_neighbors=2, n_jobs=-1)
    nn.fit(X_for_model)
    _, indices = nn.kneighbors(X_for_model)
    y_nn = y[indices[:, 1]]  # nearest neighbor (not self)
    metrics['S3'] = float(np.mean(np.abs(y - y_nn)) / y_range) if y_range > 0 else 0.0

    # --- Target complexity ---
    metrics['label_std'] = float(np.std(y))
    # Shannon entropy of discretised y
    counts, _ = np.histogram(y, bins=20)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    metrics['target_entropy'] = float(-np.sum(probs * np.log(probs)))

    # --- MLE intrinsic dimensionality ---
    X_for_mle = X_scaled
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        X_for_mle = TruncatedSVD(n_components=min(200, n - 1), random_state=42).fit_transform(X_scaled)
    try:
        metrics['mle_id'] = float(estimate_intrinsic_dim_mle(X_for_mle, k=10))
    except Exception:
        metrics['mle_id'] = None

    return metrics


def _cv_split(groups, n_folds, X, labels, seed=42):
    """Yield (train_idx, val_idx) using GroupKFold if groups available, else KFold."""
    if groups is not None and len(np.unique(groups)) >= n_folds:
        yield from GroupKFold(n_splits=n_folds).split(X, labels, groups=groups)
    else:
        yield from KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, labels)


def evaluate_cv(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_folds: int = 5,
    allocation_method: str = 'greedy',
    importance_method: str = 'treeshap',
    budget: int = 100,
    seed: int = 42,
) -> Tuple[float, float, int]:
    """Proper per-fold CV: create_hybrid on train, apply_feature_selection on val.

    Returns (mean_r2, std_r2, n_features).
    """
    sample_X = next(iter(base_reps.values()))
    scores = []
    n_feats = []

    for train_idx, val_idx in _cv_split(groups, n_folds, sample_X, labels, seed):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        try:
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                importance_method=importance_method,
            )
            X_val = apply_feature_selection(val_reps, feature_info)
        except Exception:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_train_s, y_train)
        scores.append(r2_score(y_val, model.predict(X_val_s)))
        n_feats.append(X_train.shape[1])

    if not scores:
        return float('nan'), float('nan'), 0
    return float(np.mean(scores)), float(np.std(scores)), int(np.mean(n_feats))


def evaluate_single_rep_cv(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_folds: int = 5,
) -> Tuple[float, float]:
    """Simple CV for a single representation matrix. Returns (mean_r2, std_r2)."""
    scores = []

    for train_idx, val_idx in _cv_split(groups, n_folds, X, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model = model_factory()
        model.fit(X_train, labels[train_idx])
        scores.append(r2_score(labels[val_idx], model.predict(X_val)))

    return float(np.mean(scores)), float(np.std(scores))


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def _get_model_factories(model_names: List[str]) -> Dict[str, callable]:
    """Return {name: factory_fn} for requested models."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor

    catalog = {
        'rf': lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'knn': lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        'ridge': lambda: Ridge(alpha=1.0),
        'mlp': lambda: MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
    }

    # Optional imports
    try:
        from xgboost import XGBRegressor
        catalog['xgboost'] = lambda: XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        catalog['lightgbm'] = lambda: LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        pass

    try:
        from sklearn.svm import SVR
        catalog['svr'] = lambda: SVR(kernel='rbf', C=1.0)
    except ImportError:
        pass

    try:
        from sklearn.linear_model import ElasticNet
        catalog['elasticnet'] = lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    except ImportError:
        pass

    factories = {}
    for name in model_names:
        if name in catalog:
            factories[name] = catalog[name]
        else:
            print(f"  WARNING: model '{name}' not available, skipping")
    return factories


# =============================================================================
# EXPERIMENT 1: REPRESENTATION CHARACTERIZATION
# =============================================================================

def run_characterization(data: dict, models: Dict[str, callable], n_folds: int) -> Dict[str, Any]:
    """Profile each rep's complexity before hybrid construction."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: REPRESENTATION CHARACTERIZATION")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {}

    for rep_name, X in reps.items():
        print(f"\n  {rep_name} (dim={X.shape[1]}):")
        rep_result = {}

        # Intrinsic dimensionality
        dim_info = compute_intrinsic_dimensionality(X)
        rep_result['intrinsic_dim'] = dim_info
        print(f"    PCA95={dim_info.get('pca_95', '?')}, MLE={dim_info.get('mle', '?'):.1f}"
              if dim_info.get('mle') else f"    PCA95={dim_info.get('pca_95', '?')}")

        # Mutual information (use PCA-reduced for stability)
        n_comp = min(50, X.shape[1], X.shape[0] - 1)
        X_reduced = PCA(n_components=n_comp).fit_transform(StandardScaler().fit_transform(X))
        try:
            mi = estimate_mutual_information_knn(X_reduced, labels, k=5)
            rep_result['mutual_info'] = float(mi)
            print(f"    MI={mi:.4f}")
        except Exception as e:
            rep_result['mutual_info'] = None
            print(f"    MI failed: {e}")

        # Effective rank
        rep_result['effective_rank'] = compute_effective_rank(X)
        print(f"    Effective rank={rep_result['effective_rank']:.1f}")

        # Per-model baseline R²
        rep_result['baselines'] = {}
        scores_str = []
        for model_name, model_factory in models.items():
            mean_r2, std_r2 = evaluate_single_rep_cv(X, labels, groups, model_factory, n_folds)
            rep_result['baselines'][model_name] = {'r2_mean': mean_r2, 'r2_std': std_r2}
            scores_str.append(f"{model_name}={mean_r2:.3f}")
        print(f"    Baselines: {', '.join(scores_str)}")

        results[rep_name] = rep_result

    return results


# =============================================================================
# EXPERIMENT 2: COMPLEXITY MATCHING GRID
# =============================================================================

def run_complexity_matching_grid(
    data: dict,
    models: Dict[str, callable],
    budgets: List[int],
    allocation_methods: List[str],
    importance_methods: List[str],
    n_folds: int,
) -> Dict[str, Any]:
    """Central experiment: model complexity × rep complexity optimal pairing.

    Delegates the core grid search to find_best_hybrid(), then reshapes the
    SearchResult into the format expected by downstream experiments (8, 9a).
    PCA baselines are computed separately (the library doesn't provide these).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: COMPLEXITY MATCHING GRID")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    total_dim = sum(v.shape[1] for v in reps.values())
    full_budgets = budgets + [total_dim]

    # Convert model factories to instances for the library
    model_instances = {name: factory() for name, factory in models.items()}

    print(f"  {len(models)} models x {len(full_budgets)} budgets x "
          f"{len(allocation_methods)} allocations x {len(importance_methods)} importance")
    print(f"  Delegating to find_best_hybrid()...\n")

    augs = data.get('augmentations')
    search = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        models=model_instances,
        budgets=full_budgets,
        allocation_methods=allocation_methods,
        importance_methods=importance_methods,
        n_folds=n_folds,
        augmentations=augs,
        verbose=True,
        cache=True,
        dataset_name=data.get('dataset_name', 'unknown'),
        tier='general',
    )

    # --- Reshape SearchResult to downstream format ---

    # Configs: rename fields for Exp 9a ANOVA compatibility
    configs = []
    for r in search.all_results:
        configs.append({
            'model': r['model'],
            'allocation': r['allocation_method'],
            'importance': r['importance_method'],
            'budget': r['budget'],
            'r2_mean': r['score_mean'],
            'r2_std': r['score_std'],
            'fold_scores': r.get('fold_scores', []),
            'n_features': r['n_features'],
        })

    # Best by model: rename fields for Exp 8 compatibility
    best_by_model = {}
    for model_name, r in search.best_by_model.items():
        best_by_model[model_name] = {
            'model': model_name,
            'allocation': r['allocation_method'],
            'importance': r['importance_method'],
            'budget': r['budget'],
            'r2_mean': r['score_mean'],
            'r2_std': r['score_std'],
            'n_features': r['n_features'],
        }

    # Baselines: reshape library format {rep: {model: score}} to Exp 2 format
    baselines = {}
    lib_baselines = search.baselines
    for model_name in models:
        best_single_rep = None
        best_single_r2 = -np.inf
        for rep_name, rep_scores in lib_baselines.items():
            if rep_name == 'all_concat':
                continue
            rep_r2 = rep_scores.get(model_name, -np.inf)
            if rep_r2 > best_single_r2:
                best_single_r2 = rep_r2
                best_single_rep = rep_name
        baselines[model_name] = {
            'best_single_rep': best_single_rep,
            'best_single_r2': best_single_r2,
        }
        print(f"  {model_name}: best_rep={best_single_rep} R²={best_single_r2:.4f}")

    # PCA baselines (library doesn't compute these — keep manual)
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    print("\n  Computing PCA baselines...")
    pca_baselines = {}
    for budget in budgets:
        n_comp = min(budget, X_concat.shape[1], X_concat.shape[0] - 1)
        pca_baselines[budget] = {}
        for model_name, model_factory in models.items():
            pca_scores = []
            for train_idx, val_idx in _cv_split(groups, n_folds, X_concat, labels):
                pca = PCA(n_components=n_comp)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_concat[train_idx])
                X_val_s = scaler.transform(X_concat[val_idx])
                X_train_p = pca.fit_transform(X_train_s)
                X_val_p = pca.transform(X_val_s)
                m = model_factory()
                m.fit(X_train_p, labels[train_idx])
                pca_scores.append(r2_score(labels[val_idx], m.predict(X_val_p)))
            pca_baselines[budget][model_name] = float(np.mean(pca_scores))

    # Heatmap: model x budget -> best R²
    heatmap = {}
    for model_name in models:
        heatmap[model_name] = {}
        model_configs = [c for c in configs if c['model'] == model_name]
        for budget in full_budgets:
            budget_configs = [c for c in model_configs if c['budget'] == budget]
            if budget_configs:
                best = max(budget_configs, key=lambda c: c['r2_mean'])
                heatmap[model_name][budget] = best['r2_mean']

    # Importance method wins
    importance_wins = {imp: 0 for imp in importance_methods}
    for model_name, best in best_by_model.items():
        imp = best['importance']
        if imp in importance_wins:
            importance_wins[imp] += 1

    # Print summary
    print("\n  BEST BY MODEL:")
    for model_name, best in best_by_model.items():
        baseline_r2 = baselines[model_name]['best_single_r2']
        baseline_rep = baselines[model_name]['best_single_rep']
        delta = best['r2_mean'] - baseline_r2
        print(f"  >> {model_name}: {best['allocation']}/{best['importance']}"
              f"/b={best['budget']} -> R²={best['r2_mean']:.4f} "
              f"(vs {baseline_rep} {baseline_r2:.4f}, delta={delta:+.4f})")

    print("\n  IMPORTANCE METHOD WINS:")
    for imp, count in sorted(importance_wins.items(), key=lambda x: -x[1]):
        print(f"    {imp}: {count}/{len(models)}")

    return {
        'configs': configs,
        'best_by_model': best_by_model,
        'baselines': baselines,
        'pca_baselines': pca_baselines,
        'heatmap': heatmap,
        'importance_wins': importance_wins,
    }


# =============================================================================
# EXPERIMENT 3: CROSS-CONTEXT ANALYSIS
# =============================================================================

def run_cross_context_analysis(
    datasets: Dict[str, dict],
    n_folds: int,
) -> Dict[str, Any]:
    """Optimal config changes across datasets — proving context-dependence.

    Includes dataset complexity profiling (Lorena et al. 2018 inspired) and
    correlation of complexity metrics with optimal hybrid budget.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CROSS-CONTEXT ANALYSIS")
    print("=" * 70)

    results = {}

    for ds_name, data in datasets.items():
        print(f"\n  --- {ds_name} ---")

        # Compute dataset complexity profile
        print(f"  Computing complexity metrics...")
        complexity = compute_dataset_complexity(data['reps'], data['labels'])
        mle_str = f", MLE_ID={complexity['mle_id']:.1f}" if complexity.get('mle_id') else ""
        print(f"    erank={complexity['erank']:.1f}, PR={complexity['participation_ratio']:.1f}, "
              f"n/d={complexity['n_over_d']:.2f}, n/erank={complexity['n_over_erank']:.1f}{mle_str}")
        print(f"    C1={complexity['C1']:.3f}, C2={complexity['C2']:.3f}, "
              f"L1={complexity['L1']:.3f}, S3={complexity['S3']:.3f}")

        # Run hybrid search
        print(f"  Running find_best_hybrid...")
        start = time.time()

        try:
            from sklearn.svm import SVR
            from sklearn.linear_model import ElasticNet

            novel_models = {
                'svr': SVR(kernel='rbf', C=1.0),
                'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            }

            search_result = find_best_hybrid(
                base_reps=data['reps'],
                labels=data['labels'],
                groups=data['groups'],
                speed='medium',
                augmentations=data.get('augmentations'),
                verbose=False,
                cache=True,
                dataset_name=data.get('dataset_name', ds_name),
                tier='general',
            )

            # Use first model's best config for blueprint
            first_model_best = next(iter(search_result.best_by_model.values()), None)
            if first_model_best:
                bp = first_model_best['blueprint']
                novel_results = {}
                _n_samples_here = len(data['labels'])
                for novel_name, novel_model in novel_models.items():
                    # Canonical gate: SVR has _max_samples=10000 in DEFAULT_MODELS.
                    # _resolve_models with list form looks up DEFAULT_MODELS[name]
                    # and drops the entry when n_samples exceeds the cap.
                    if novel_name == 'svr':
                        _resolved = _resolve_models(
                            ['svm'], task_type='regression', speed='full',
                            n_samples=_n_samples_here,
                            dataset_name=data.get('dataset_name', ds_name))
                        if not _resolved:
                            print(f"    {novel_name} on {ds_name}: skipped "
                                  f"(n_samples={_n_samples_here} > svm _max_samples)")
                            continue
                    try:
                        sample_X = next(iter(data['reps'].values()))
                        novel_scores = []

                        for train_idx, val_idx in _cv_split(data['groups'], n_folds, sample_X, data['labels']):
                            train_reps = {k: v[train_idx] for k, v in data['reps'].items()}
                            val_reps = {k: v[val_idx] for k, v in data['reps'].items()}
                            y_train = data['labels'][train_idx]
                            y_val = data['labels'][val_idx]

                            X_train, fi = create_hybrid(
                                train_reps, y_train,
                                allocation_method=bp.allocation_method,
                                budget=bp.budget,
                                importance_method=bp.importance_method,
                            )
                            X_val = apply_feature_selection(val_reps, fi)

                            scaler = StandardScaler()
                            X_train_s = scaler.fit_transform(X_train)
                            X_val_s = scaler.transform(X_val)

                            m = deepcopy(novel_model)
                            m.fit(X_train_s, y_train)
                            novel_scores.append(r2_score(y_val, m.predict(X_val_s)))

                        novel_results[novel_name] = float(np.mean(novel_scores))
                        print(f"    {novel_name} on {ds_name}: R²={np.mean(novel_scores):.4f}")
                    except Exception as e:
                        print(f"    {novel_name} on {ds_name}: FAILED - {e}")

            elapsed = time.time() - start

            ds_result = {
                'n_samples': len(data['labels']),
                'elapsed_seconds': elapsed,
                'complexity': complexity,
                'best_model': best['model'] if best else None,
                'best_r2': best['score_mean'] if best else None,
                'best_allocation': best['blueprint'].allocation_method if best else None,
                'best_importance': best['blueprint'].importance_method if best else None,
                'best_budget': best['blueprint'].budget if best else None,
                'best_dim': best['n_features'] if best else None,
                'baselines': search_result.baselines,
                'best_by_model': {
                    m: {'budget': r['budget'], 'r2': r['score_mean'], 'dim': r['n_features']}
                    for m, r in search_result.best_by_model.items()
                },
                'performance_matrix': search_result.performance_matrix,
                'novel_models': novel_results if best else {},
            }
            results[ds_name] = ds_result

            print(f"    Best: {ds_result['best_model']}, R²={ds_result['best_r2']:.4f}, "
                  f"alloc={ds_result['best_allocation']}, imp={ds_result['best_importance']}, "
                  f"budget={ds_result['best_budget']}")

        except Exception as e:
            print(f"    {ds_name}: FAILED - {e}")
            results[ds_name] = {'error': str(e)}

    # Cross-dataset comparison table
    print("\n  CROSS-CONTEXT COMPARISON:")
    print(f"  {'Dataset':<10} {'N':<6} {'Best Model':<12} {'Importance':<15} {'Budget':<8} {'R²':<8}")
    print("  " + "-" * 65)
    for ds_name, r in results.items():
        if 'error' not in r:
            print(f"  {ds_name:<10} {r['n_samples']:<6} {r['best_model']:<12} "
                  f"{r['best_importance']:<15} {r['best_budget']:<8} {r['best_r2']:.4f}")

    # --- Dataset complexity vs optimal hybrid dimension ---
    valid = {ds: r for ds, r in results.items() if 'error' not in r and r.get('best_budget') is not None}
    if len(valid) >= 2:
        print("\n  DATASET COMPLEXITY vs OPTIMAL BUDGET:")
        print(f"  {'Dataset':<10} {'erank':>7} {'PR':>7} {'n/erank':>8} {'C1':>6} {'C2':>6} "
              f"{'L1':>6} {'S3':>6} {'Budget':>7} {'Dim':>5}")
        print("  " + "-" * 82)

        complexity_keys = ['erank', 'participation_ratio', 'n_over_erank', 'C1', 'C2', 'L1', 'S3']
        ds_names = sorted(valid.keys())
        budgets = np.array([valid[ds]['best_budget'] for ds in ds_names])
        dims = np.array([valid[ds]['best_dim'] for ds in ds_names])

        for ds in ds_names:
            c = valid[ds]['complexity']
            print(f"  {ds:<10} {c['erank']:>7.1f} {c['participation_ratio']:>7.1f} "
                  f"{c['n_over_erank']:>8.1f} {c['C1']:>6.3f} {c['C2']:>6.3f} "
                  f"{c['L1']:>6.3f} {c['S3']:>6.3f} "
                  f"{valid[ds]['best_budget']:>7} {valid[ds]['best_dim']:>5}")

        # Correlation of each complexity metric with optimal budget
        correlations = {}
        print("\n  Spearman correlation (complexity metric → optimal budget):")
        for key in complexity_keys:
            values = np.array([valid[ds]['complexity'][key] for ds in ds_names])
            if np.std(values) < 1e-10:
                continue
            rho, p = spearmanr(values, budgets)
            correlations[key] = {'rho_vs_budget': float(rho), 'p_vs_budget': float(p)}
            # Also correlate with actual achieved dimension
            rho_dim, p_dim = spearmanr(values, dims)
            correlations[key]['rho_vs_dim'] = float(rho_dim)
            correlations[key]['p_vs_dim'] = float(p_dim)
            print(f"    {key:<20} → budget: rho={rho:+.3f} (p={p:.3f}), "
                  f"dim: rho={rho_dim:+.3f} (p={p_dim:.3f})")

        results['_complexity_correlation'] = correlations

        # Key insight: does higher intrinsic complexity → higher optimal budget?
        eranks = np.array([valid[ds]['complexity']['erank'] for ds in ds_names])
        rho_erank, _ = spearmanr(eranks, budgets)
        print(f"\n  KEY FINDING: erank → optimal budget correlation: rho={rho_erank:+.3f}")
        if rho_erank > 0.3:
            print("    → Higher intrinsic dimensionality correlates with larger optimal hybrid")
        elif rho_erank < -0.3:
            print("    → Higher intrinsic dimensionality correlates with smaller optimal hybrid")
        else:
            print("    → Weak correlation — optimal budget depends on more than raw dimensionality")

        # Per-model optimal budget comparison across datasets
        model_names = set()
        for ds in ds_names:
            model_names.update(valid[ds].get('best_by_model', {}).keys())
        if model_names:
            print("\n  PER-MODEL OPTIMAL BUDGET (cross-dataset):")
            print(f"  {'Model':<12}" + "".join(f"{ds:>14}" for ds in ds_names))
            print("  " + "-" * (12 + 14 * len(ds_names)))
            for model in sorted(model_names):
                row = f"  {model:<12}"
                for ds in ds_names:
                    bm = valid[ds].get('best_by_model', {}).get(model)
                    if bm:
                        row += f"b={bm['budget']:>3} R²={bm['r2']:.3f} "
                    else:
                        row += f"{'--':>14}"
                print(row)

            # MODEL × DATASET INTERACTION ANALYSIS
            # Quantify how much budget variation comes from model vs dataset vs interaction
            model_list = sorted(model_names)
            budget_matrix = []  # rows=models, cols=datasets
            r2_matrix = []
            for model in model_list:
                model_budgets = []
                model_r2s = []
                for ds in ds_names:
                    bm = valid[ds].get('best_by_model', {}).get(model)
                    if bm:
                        model_budgets.append(bm['budget'])
                        model_r2s.append(bm['r2'])
                    else:
                        model_budgets.append(np.nan)
                        model_r2s.append(np.nan)
                budget_matrix.append(model_budgets)
                r2_matrix.append(model_r2s)
            budget_matrix = np.array(budget_matrix, dtype=float)
            r2_matrix = np.array(r2_matrix, dtype=float)

            # Budget variance decomposition (simple: row means, col means, residual)
            if budget_matrix.shape[0] >= 2 and budget_matrix.shape[1] >= 2:
                valid_mask = ~np.isnan(budget_matrix)
                if valid_mask.all():
                    grand_mean = np.nanmean(budget_matrix)
                    row_means = np.nanmean(budget_matrix, axis=1)  # per-model
                    col_means = np.nanmean(budget_matrix, axis=0)  # per-dataset
                    ss_model = len(ds_names) * np.sum((row_means - grand_mean) ** 2)
                    ss_dataset = len(model_list) * np.sum((col_means - grand_mean) ** 2)
                    ss_total = np.sum((budget_matrix - grand_mean) ** 2)
                    ss_interaction = ss_total - ss_model - ss_dataset

                    print(f"\n  BUDGET VARIANCE DECOMPOSITION:")
                    print(f"    Model effect:       {ss_model/ss_total*100:5.1f}% of variance" if ss_total > 0 else "")
                    print(f"    Dataset effect:     {ss_dataset/ss_total*100:5.1f}% of variance" if ss_total > 0 else "")
                    print(f"    Interaction effect: {ss_interaction/ss_total*100:5.1f}% of variance" if ss_total > 0 else "")

                    results['_budget_variance'] = {
                        'ss_model': float(ss_model),
                        'ss_dataset': float(ss_dataset),
                        'ss_interaction': float(ss_interaction),
                        'ss_total': float(ss_total),
                        'pct_model': float(ss_model / ss_total * 100) if ss_total > 0 else 0,
                        'pct_dataset': float(ss_dataset / ss_total * 100) if ss_total > 0 else 0,
                        'pct_interaction': float(ss_interaction / ss_total * 100) if ss_total > 0 else 0,
                    }

                    # Does budget range differ across models? (some models are dimension-sensitive)
                    print(f"\n  MODEL DIMENSION SENSITIVITY:")
                    print(f"  {'Model':<12} {'Budget Range':<15} {'Budget Std':<12} {'R² Range':<12}")
                    print("  " + "-" * 51)
                    for i, model in enumerate(model_list):
                        b_row = budget_matrix[i]
                        r2_row = r2_matrix[i]
                        if not np.any(np.isnan(b_row)):
                            print(f"  {model:<12} {int(b_row.min())}-{int(b_row.max()):<10} "
                                  f"{np.std(b_row):>8.1f}    {np.ptp(r2_row):>8.4f}")

    return results


# =============================================================================
# EXPERIMENT 4: IMPORTANCE WEIGHTS AS LEARNED METRICS
# =============================================================================

def run_importance_weight_metrics(
    data: dict,
    importance_methods: List[str],
    n_folds: int,
    budget: int = 25,
) -> Dict[str, Any]:
    """Bridge Phase 5 benchmark findings: importance scores define learned metric spaces."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: IMPORTANCE WEIGHTS AS LEARNED METRICS")
    print("=" * 70)

    # Import evaluation functions from benchmark
    from benchmark_importance_aggregation import evaluate_weighted_knn, evaluate_similarity_retrieval

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    print(f"  Concatenated features: {X_concat.shape}")

    results = {'methods': {}, 'cross_test': {}, 'low_dim': {}}

    # Compute importance with each method
    importances = {}
    for method in importance_methods:
        print(f"  Computing {method}...", end=' ', flush=True)
        start = time.time()
        try:
            imp_dict = compute_feature_importance(reps, labels, method=method)
            imp_concat = np.concatenate([imp_dict[rep] for rep in sorted(imp_dict.keys())])
            importances[method] = imp_concat
            elapsed = time.time() - start
            print(f"done ({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    # Evaluate each method: unweighted kNN vs weighted kNN
    print("\n  Evaluating as kNN weights...")
    for method, imp_scores in importances.items():
        try:
            knn_result = evaluate_weighted_knn(X_concat, labels, imp_scores, groups, n_folds)
            sim_result = evaluate_similarity_retrieval(X_concat, labels, imp_scores, groups, n_folds)

            k10 = knn_result.get(10, knn_result.get(5, {}))
            results['methods'][method] = {
                'knn': knn_result,
                'similarity': sim_result,
                'knn_delta_k10': k10.get('delta', 0),
                'sim_delta_r2': sim_result.get('delta_r2', 0),
            }

            print(f"    {method}: kNN delta={k10.get('delta', 0):+.4f}, "
                  f"sim delta={sim_result.get('delta_r2', 0):+.4f}")
        except Exception as e:
            print(f"    {method}: FAILED - {e}")

    # Cross-test: selection by method A, weighting by method B
    print("\n  Cross-test matrix (select × weight)...")
    method_names = list(importances.keys())
    if len(method_names) >= 2:
        for select_method in method_names[:4]:  # limit to avoid explosion
            select_scores = importances[select_method]
            top_idx = np.argsort(select_scores)[-budget:]
            X_subset = X_concat[:, top_idx]

            for weight_method in method_names[:4]:
                weight_scores = importances[weight_method][top_idx]
                try:
                    sim_result = evaluate_similarity_retrieval(
                        X_subset, labels, weight_scores, groups, n_folds)
                    key = f"{select_method}__x__{weight_method}"
                    results['cross_test'][key] = sim_result['weighted_r2']
                except Exception:
                    pass

    # Low-dim: budget=25 selected features + importance weighting
    print(f"\n  Low-dim (budget={budget}) selected + weighted...")
    for method, imp_scores in importances.items():
        top_idx = np.argsort(imp_scores)[-budget:]
        X_subset = X_concat[:, top_idx]
        subset_scores = imp_scores[top_idx]
        try:
            sim_result = evaluate_similarity_retrieval(
                X_subset, labels, subset_scores, groups, n_folds)
            results['low_dim'][method] = {
                'weighted_r2': sim_result['weighted_r2'],
                'unweighted_r2': sim_result['unweighted_r2'],
                'delta_r2': sim_result['delta_r2'],
            }
            print(f"    {method}: weighted R²={sim_result['weighted_r2']:.4f}, "
                  f"unweighted R²={sim_result['unweighted_r2']:.4f}")
        except Exception as e:
            print(f"    {method}: FAILED - {e}")

    # Summary: which method produces the best metric space?
    if results['methods']:
        best_knn = max(results['methods'].items(), key=lambda x: x[1].get('knn_delta_k10', -999))
        best_sim = max(results['methods'].items(), key=lambda x: x[1].get('sim_delta_r2', -999))
        print(f"\n  Best kNN weight source: {best_knn[0]} (delta={best_knn[1]['knn_delta_k10']:+.4f})")
        print(f"  Best similarity weight source: {best_sim[0]} (delta={best_sim[1]['sim_delta_r2']:+.4f})")
        results['best_knn_method'] = best_knn[0]
        results['best_sim_method'] = best_sim[0]

    return results


# =============================================================================
# EXPERIMENT 6: WITHIN-FAMILY MODEL COMPLEXITY GRADIENT
# =============================================================================

def _get_model_family_variants() -> Dict[str, List[Tuple[str, callable, float]]]:
    """Return model families with complexity gradients.

    Each family: list of (variant_name, factory_fn, capacity_index).
    capacity_index is a monotonically increasing proxy for model capacity.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor

    families = {
        'RF': [
            ('depth=3', lambda: RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1), 1),
            ('depth=5', lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1), 2),
            ('depth=10', lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), 3),
            ('depth=None', lambda: RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1), 4),
        ],
        'Ridge': [
            ('alpha=100', lambda: Ridge(alpha=100.0), 1),
            ('alpha=10', lambda: Ridge(alpha=10.0), 2),
            ('alpha=1', lambda: Ridge(alpha=1.0), 3),
            ('alpha=0.01', lambda: Ridge(alpha=0.01), 4),
        ],
        'kNN': [
            ('k=20', lambda: KNeighborsRegressor(n_neighbors=20, n_jobs=-1), 1),
            ('k=10', lambda: KNeighborsRegressor(n_neighbors=10, n_jobs=-1), 2),
            ('k=5', lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1), 3),
            ('k=1', lambda: KNeighborsRegressor(n_neighbors=1, n_jobs=-1), 4),
        ],
        'MLP': [
            ('16', lambda: MLPRegressor(hidden_layer_sizes=(16,), random_state=42, max_iter=500), 1),
            ('64', lambda: MLPRegressor(hidden_layer_sizes=(64,), random_state=42, max_iter=500), 2),
            ('128,64', lambda: MLPRegressor(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500), 3),
            ('256,128,64', lambda: MLPRegressor(hidden_layer_sizes=(256, 128, 64), random_state=42, max_iter=500), 4),
        ],
    }

    try:
        from xgboost import XGBRegressor
        families['XGBoost'] = [
            ('depth=2', lambda: XGBRegressor(n_estimators=100, max_depth=2, random_state=42, n_jobs=-1, verbosity=0), 1),
            ('depth=4', lambda: XGBRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, verbosity=0), 2),
            ('depth=6', lambda: XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbosity=0), 3),
            ('depth=None', lambda: XGBRegressor(n_estimators=100, max_depth=0, random_state=42, n_jobs=-1, verbosity=0, tree_method='hist'), 4),
        ]
    except ImportError:
        pass

    return families


def evaluate_cv_with_train(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_folds: int = 5,
    allocation_method: str = 'greedy',
    importance_method: str = 'treeshap',
    budget: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float, float, int]:
    """Like evaluate_cv but also returns train R².

    Returns (val_r2_mean, val_r2_std, train_r2_mean, train_r2_std, n_features).
    """
    sample_X = next(iter(base_reps.values()))
    val_scores = []
    train_scores = []
    n_feats = []

    for train_idx, val_idx in _cv_split(groups, n_folds, sample_X, labels, seed):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        try:
            X_train, feature_info = create_hybrid(
                train_reps, y_train,
                allocation_method=allocation_method,
                budget=budget,
                importance_method=importance_method,
            )
            X_val = apply_feature_selection(val_reps, feature_info)
        except Exception:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_train_s, y_train)
        val_scores.append(r2_score(y_val, model.predict(X_val_s)))
        train_scores.append(r2_score(y_train, model.predict(X_train_s)))
        n_feats.append(X_train.shape[1])

    if not val_scores:
        return float('nan'), float('nan'), float('nan'), float('nan'), 0
    return (
        float(np.mean(val_scores)), float(np.std(val_scores)),
        float(np.mean(train_scores)), float(np.std(train_scores)),
        int(np.mean(n_feats)),
    )


def run_within_family_complexity(
    data: dict,
    budgets: List[int],
    n_folds: int,
    families: List[str] = None,
    search_speed: str = 'fast',
) -> Dict[str, Any]:
    """Exp 6: Vary complexity within each model family to show optimal budget shifts.

    For each model variant, runs find_best_hybrid() with that variant as the
    sole model. The library searches importance methods, allocation methods, and
    budgets to find the true best hybrid for that specific model capacity.
    Then we also collect train R² for bias-variance analysis (Exp 9d).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: WITHIN-FAMILY MODEL COMPLEXITY GRADIENT")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    all_families = _get_model_family_variants()
    if families:
        all_families = {k: v for k, v in all_families.items() if k in families}

    results = {'families': {}, 'search_speed': search_speed}

    for fam_name, variants in all_families.items():
        print(f"\n  Family: {fam_name} ({len(variants)} variants, search_speed={search_speed})")
        fam_result = {'heatmap': {}, 'optimal_budget': {}, 'train_val_gap': {}}

        for var_name, var_factory, capacity_idx in variants:
            print(f"\n    Variant: {var_name} (capacity={capacity_idx})")
            fam_result['heatmap'][var_name] = {}

            # Run find_best_hybrid with this variant as the sole model
            augs = data.get('augmentations')
            try:
                search = find_best_hybrid(
                    reps, labels, groups=groups,
                    models={var_name: var_factory()},
                    speed=search_speed,
                    budgets=budgets,
                    augmentations=augs,
                    verbose=False,
                    cache=True,
                    dataset_name=f"{data.get('dataset_name', 'unknown')}_{var_name}",
                    tier='general',
                )
            except Exception as e:
                print(f"      find_best_hybrid failed: {e}")
                fam_result['optimal_budget'][var_name] = {
                    'budget': None, 'r2': float('nan'), 'capacity_index': capacity_idx,
                }
                continue

            # Extract best score at each budget from all_results
            for budget in budgets:
                budget_results = [r for r in search.all_results
                                  if r['budget'] == budget and not np.isnan(r['score_mean'])]
                if budget_results:
                    best_at_budget = max(budget_results, key=lambda r: r['score_mean'])
                    bp = best_at_budget['blueprint']

                    # Get train R² for bias-variance analysis
                    try:
                        val_r2, val_std, train_r2, train_std, n_feat = evaluate_cv_with_train(
                            reps, labels, groups, var_factory, n_folds,
                            allocation_method=bp.allocation_method,
                            importance_method=bp.importance_method,
                            budget=budget,
                        )
                    except Exception:
                        val_r2 = best_at_budget['score_mean']
                        val_std = best_at_budget['score_std']
                        train_r2, train_std = float('nan'), float('nan')
                        n_feat = best_at_budget.get('n_features', 0)

                    fam_result['heatmap'][var_name][budget] = {
                        'val_r2': val_r2, 'val_std': val_std,
                        'train_r2': train_r2, 'train_std': train_std,
                        'n_features': n_feat,
                        'importance': bp.importance_method,
                        'allocation': bp.allocation_method,
                    }
                    gap = train_r2 - val_r2 if not np.isnan(train_r2) else float('nan')
                    fam_result['train_val_gap'][f"{var_name}_b{budget}"] = gap

                    print(f"      b={budget}: R²={val_r2:.4f}±{val_std:.3f}, "
                          f"train_R²={train_r2:.4f}, gap={gap:.3f} "
                          f"[{bp.importance_method}/{bp.allocation_method}]")

            # Optimal budget comes from find_best_hybrid's own search
            best = next(iter(search.best_by_model.values()), None)
            if best:
                fam_result['optimal_budget'][var_name] = {
                    'budget': best['blueprint'].budget,
                    'r2': best['score_mean'],
                    'capacity_index': capacity_idx,
                    'importance': best['blueprint'].importance_method,
                    'allocation': best['blueprint'].allocation_method,
                }
                print(f"      >> optimal: b={best['blueprint'].budget}, R²={best['score_mean']:.4f} "
                      f"[{best['blueprint'].importance_method}/{best['blueprint'].allocation_method}]")
            else:
                fam_result['optimal_budget'][var_name] = {
                    'budget': None, 'r2': float('nan'), 'capacity_index': capacity_idx,
                }

        # Correlation: capacity_index vs optimal_budget
        capacities = []
        opt_budgets = []
        for var_name, info in fam_result['optimal_budget'].items():
            if info.get('budget') is not None:
                capacities.append(info['capacity_index'])
                opt_budgets.append(info['budget'])

        if len(capacities) >= 3:
            rho, pval = spearmanr(capacities, opt_budgets)
            fam_result['capacity_budget_correlation'] = {'rho': float(rho), 'p': float(pval)}
            print(f"\n  >> {fam_name}: capacity vs optimal_budget: rho={rho:.3f}, p={pval:.3f}")
        else:
            fam_result['capacity_budget_correlation'] = None

        results['families'][fam_name] = fam_result

    # Summary across all families
    print("\n  SUMMARY: optimal budget by variant")
    for fam_name, fam_result in results['families'].items():
        for var_name, info in fam_result['optimal_budget'].items():
            b = info.get('budget', '?')
            r2 = info.get('r2', float('nan'))
            imp = info.get('importance', '?')
            print(f"    {fam_name}/{var_name}: optimal_budget={b}, R²={r2:.4f} [{imp}]")

    return results


# =============================================================================
# EXPERIMENT 7: DATA COMPLEXITY VIA SCAFFOLD PARTITIONING
# =============================================================================

def run_data_complexity_analysis(
    data: dict,
    models: Dict[str, callable],
    budgets: List[int],
    n_folds: int,
    partitions: List[str] = None,
    search_speed: str = 'fast',
) -> Dict[str, Any]:
    """Exp 7: Optimal config depends on data subpopulation complexity.

    Uses find_best_hybrid() on each subpopulation so we get a properly-searched
    optimal config per partition, not just greedy/treeshap.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: DATA COMPLEXITY VIA SCAFFOLD PARTITIONING")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    smiles = data['smiles']

    all_partitions = partitions or ['scaffold_frequency', 'molecular_complexity', 'target_variance']
    results = {'partitions': {}, 'search_speed': search_speed}

    for partition_name in all_partitions:
        print(f"\n  Partition: {partition_name}")

        # Compute partition masks
        if partition_name == 'scaffold_frequency':
            # Common (top 20% scaffolds) vs rare (bottom 80%)
            unique_scaffolds, scaffold_counts = np.unique(groups, return_counts=True)
            count_threshold = np.percentile(scaffold_counts, 80)  # top 20% most frequent
            frequent_scaffolds = set(unique_scaffolds[scaffold_counts >= count_threshold])
            mask_easy = np.array([g in frequent_scaffolds for g in groups])
            mask_hard = ~mask_easy

        elif partition_name == 'molecular_complexity':
            # Heavy atom count: <= median vs > median
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                hac = np.array([Descriptors.HeavyAtomCount(Chem.MolFromSmiles(s))
                                if Chem.MolFromSmiles(s) else 0 for s in smiles])
                median_hac = np.median(hac[hac > 0])
                mask_easy = hac <= median_hac
                mask_hard = hac > median_hac
            except ImportError:
                print("    RDKit not available, skipping molecular_complexity")
                continue

        elif partition_name == 'target_variance':
            # Scaffolds with low target variance vs high
            unique_scaffolds = np.unique(groups)
            scaffold_var = {}
            for s in unique_scaffolds:
                s_labels = labels[groups == s]
                if len(s_labels) >= 2:
                    scaffold_var[s] = np.var(s_labels)
            if not scaffold_var:
                print("    Not enough scaffolds with 2+ members, skipping")
                continue
            median_var = np.median(list(scaffold_var.values()))
            mask_easy = np.array([scaffold_var.get(g, 0) <= median_var for g in groups])
            mask_hard = ~mask_easy
        else:
            print(f"    Unknown partition: {partition_name}, skipping")
            continue

        n_easy = mask_easy.sum()
        n_hard = mask_hard.sum()
        print(f"    easy={n_easy}, hard={n_hard}")

        if n_easy < 50 or n_hard < 50:
            print("    Subpopulation too small, skipping")
            continue

        part_result = {'easy': {}, 'hard': {}, 'n_easy': int(n_easy), 'n_hard': int(n_hard)}

        full_augs = data.get('augmentations')
        for subpop_name, mask in [('easy', mask_easy), ('hard', mask_hard)]:
            sub_reps = {k: v[mask] for k, v in reps.items()}
            sub_labels = labels[mask]
            sub_groups = groups[mask]
            sub_augs = {k: v[mask] for k, v in full_augs.items()} if full_augs else None
            n_unique_groups = len(np.unique(sub_groups))

            if n_unique_groups < 2:
                print(f"    {subpop_name}: not enough groups for CV, skipping")
                continue

            # Complexity metrics
            X_concat = np.hstack([sub_reps[k] for k in sorted(sub_reps.keys())])
            metrics = {}
            try:
                n_comp = min(50, X_concat.shape[1], X_concat.shape[0] - 1)
                X_reduced = PCA(n_components=n_comp).fit_transform(
                    StandardScaler().fit_transform(X_concat))
                metrics['mi'] = float(estimate_mutual_information_knn(X_reduced, sub_labels, k=5))
            except Exception:
                metrics['mi'] = None
            metrics['effective_rank'] = compute_effective_rank(X_concat)
            dim_info = compute_intrinsic_dimensionality(X_concat)
            metrics['pca_95'] = dim_info.get('pca_95')
            metrics['mle_dim'] = dim_info.get('mle')
            metrics['n_samples'] = int(mask.sum())

            print(f"    {subpop_name}: eff_rank={metrics['effective_rank']:.1f}, "
                  f"MI={metrics.get('mi', '?')}, PCA95={metrics.get('pca_95', '?')}")

            # Use find_best_hybrid for proper search
            try:
                search = find_best_hybrid(
                    sub_reps, sub_labels, groups=sub_groups,
                    speed=search_speed,
                    budgets=budgets,
                    augmentations=sub_augs,
                    verbose=False,
                    cache=True,
                    dataset_name=f"{data.get('dataset_name', 'unknown')}_{partition_name}_{subpop_name}",
                    tier='general',
                )
                best = next(iter(search.best_by_model.values()), None)
                best_config = {
                    'model': best['model'],
                    'budget': best['blueprint'].budget,
                    'r2': best['score_mean'],
                    'r2_std': best['score_std'],
                    'allocation': best['blueprint'].allocation_method,
                    'importance': best['blueprint'].importance_method,
                    'n_features': best['n_features'],
                } if best else None

                # Also extract best-by-model for richer comparison
                best_by_model = {}
                for model_name, mbest in search.best_by_model.items():
                    best_by_model[model_name] = {
                        'r2': mbest['score_mean'],
                        'budget': mbest['blueprint'].budget,
                        'importance': mbest['blueprint'].importance_method,
                    }

            except Exception as e:
                print(f"    {subpop_name}: find_best_hybrid failed - {e}")
                best_config = None
                best_by_model = {}

            part_result[subpop_name] = {
                'metrics': metrics,
                'best': best_config,
                'best_by_model': best_by_model,
                'baselines': search.baselines if best_config else {},
            }
            if best_config:
                print(f"    {subpop_name} best: {best_config['model']}/b={best_config['budget']} "
                      f"R²={best_config['r2']:.4f} "
                      f"[{best_config['importance']}/{best_config['allocation']}]")

        # Compare easy vs hard
        easy_best = part_result['easy'].get('best')
        hard_best = part_result['hard'].get('best')
        if easy_best and hard_best:
            part_result['delta_budget'] = hard_best['budget'] - easy_best['budget']
            part_result['config_differs'] = (
                easy_best['model'] != hard_best['model']
                or easy_best['budget'] != hard_best['budget']
                or easy_best.get('importance') != hard_best.get('importance')
            )
            print(f"    >> Configs differ: {part_result['config_differs']} "
                  f"(easy={easy_best['model']}/b={easy_best['budget']}/{easy_best.get('importance')}, "
                  f"hard={hard_best['model']}/b={hard_best['budget']}/{hard_best.get('importance')})")

        results['partitions'][partition_name] = part_result

    return results


# =============================================================================
# EXPERIMENT 5: LEARNING CURVES
# =============================================================================

def run_learning_curves(
    data: dict,
    models: Dict[str, callable],
    n_folds: int,
    n_repeats: int = 3,
    search_speed: str = 'fast',
) -> Dict[str, Any]:
    """How hybrid advantage changes with sample size.

    At each sample size, runs find_best_hybrid() on the subsample to discover
    the optimal config for that data regime. This tests whether the methodology
    adapts to different data scales and whether the hybrid advantage persists.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: LEARNING CURVES")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_total = len(labels)
    model_names = list(models.keys())

    fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
    sample_sizes = [max(50, int(n_total * f)) for f in fractions]
    # Deduplicate and cap at total
    sample_sizes = sorted(set(min(s, n_total) for s in sample_sizes))

    results = {'sample_sizes': sample_sizes, 'models': {}, 'search_speed': search_speed}

    # Initialize per-model result structure
    for model_name in model_names:
        results['models'][model_name] = {
            'hybrid': [], 'best_single': [],
            'best_configs': [],
        }

    for n_sub in sample_sizes:
        print(f"\n  n={n_sub}:")
        # Collect scores across repeats per model
        hybrid_by_model = {m: [] for m in model_names}
        single_by_model = {m: [] for m in model_names}
        configs_by_model = {m: [] for m in model_names}

        full_augs = data.get('augmentations')
        for seed in range(n_repeats):
            np.random.seed(seed)
            idx = np.random.choice(n_total, n_sub, replace=False)
            sub_reps = {k: v[idx] for k, v in reps.items()}
            sub_labels = labels[idx]
            sub_groups = groups[idx]
            sub_augs = {k: v[idx] for k, v in full_augs.items()} if full_augs else None
            n_unique_groups = len(np.unique(sub_groups))

            if n_unique_groups < 2:
                continue

            sub_folds = min(n_folds, n_unique_groups)

            # find_best_hybrid discovers optimal config per model at this sample size
            try:
                search = find_best_hybrid(
                    sub_reps, sub_labels, groups=sub_groups,
                    models=model_names,
                    speed=search_speed,
                    n_folds=sub_folds,
                    augmentations=sub_augs,
                    verbose=False,
                    cache=True,
                    dataset_name=f"{data.get('dataset_name', 'unknown')}_n{n_sub}_s{seed}",
                    tier='general',
                )
                for model_name in model_names:
                    if model_name in search.best_by_model:
                        mbest = search.best_by_model[model_name]
                        hybrid_by_model[model_name].append(mbest['score_mean'])
                        configs_by_model[model_name].append({
                            'budget': mbest['blueprint'].budget,
                            'importance': mbest['blueprint'].importance_method,
                            'allocation': mbest['blueprint'].allocation_method,
                        })
            except Exception as e:
                print(f"    seed={seed}: find_best_hybrid failed - {e}")

            # Baselines per model
            for model_name, model_factory in models.items():
                # Best single rep
                best_single = -np.inf
                for rep_name, X in sub_reps.items():
                    try:
                        r2, _ = evaluate_single_rep_cv(
                            X, sub_labels, sub_groups, model_factory,
                            n_folds=sub_folds,
                        )
                        best_single = max(best_single, r2)
                    except Exception:
                        pass
                if best_single > -np.inf:
                    single_by_model[model_name].append(best_single)

        # Aggregate across repeats
        for model_name in model_names:
            h = float(np.mean(hybrid_by_model[model_name])) if hybrid_by_model[model_name] else float('nan')
            s = float(np.mean(single_by_model[model_name])) if single_by_model[model_name] else float('nan')

            results['models'][model_name]['hybrid'].append(h)
            results['models'][model_name]['best_single'].append(s)
            results['models'][model_name]['best_configs'].append(
                configs_by_model[model_name] if configs_by_model[model_name] else None)

            delta = h - s if not (np.isnan(h) or np.isnan(s)) else float('nan')
            print(f"    {model_name}: hybrid={h:.4f}, best_single={s:.4f}, delta={delta:+.4f}")

    # Fit learning curves per model
    for model_name in model_names:
        model_result = results['models'][model_name]
        valid = [i for i, h in enumerate(model_result['hybrid']) if not np.isnan(h)]
        if len(valid) >= 3:
            n_arr = np.array([sample_sizes[i] for i in valid])
            h_arr = np.array([model_result['hybrid'][i] for i in valid])
            model_result['hybrid_fit'] = fit_learning_curve(n_arr, h_arr)

    return results


# =============================================================================
# EXPERIMENT 8: UNCERTAINTY CALIBRATION
# =============================================================================

def run_uncertainty_calibration(
    data: dict,
    n_folds: int,
    conditions: List[str] = None,
    exp2_results: Dict[str, Any] = None,
    search_speed: str = 'fast',
) -> Dict[str, Any]:
    """Exp 8: Complexity matching produces better-calibrated uncertainty estimates.

    The 'optimal' condition uses find_best_hybrid() (or Exp 2 results) to
    discover the actual best config. Suboptimal conditions are deliberately
    constructed to show the contrast.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: UNCERTAINTY CALIBRATION AT OPTIMAL VS SUBOPTIMAL COMPLEXITY")
    print("=" * 70)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    total_dim = sum(v.shape[1] for v in reps.values())

    all_conditions = conditions or ['optimal', 'underfit', 'overfit', 'mismatched']

    # Discover optimal config via find_best_hybrid or Exp 2 results
    optimal_model_factory = None
    optimal_budget = 100
    optimal_alloc = 'greedy'
    optimal_imp = 'treeshap'
    optimal_desc = 'RF at budget=100 (default)'

    if exp2_results and exp2_results.get('best_by_model'):
        # Use best config from Exp 2
        all_best = exp2_results['best_by_model']
        overall_best = max(all_best.values(), key=lambda c: c['r2_mean'])
        optimal_budget = overall_best['budget']
        optimal_alloc = overall_best['allocation']
        optimal_imp = overall_best['importance']
        best_model = overall_best['model']
        optimal_model_factory = _get_model_factories([best_model]).get(best_model)
        optimal_desc = (f"{best_model} at budget={optimal_budget} "
                        f"[{optimal_imp}/{optimal_alloc}] (from Exp 2)")
        print(f"  Using Exp 2 optimal: {optimal_desc}")
    else:
        # Run find_best_hybrid to discover optimal
        print(f"  Running find_best_hybrid(speed={search_speed}) to find optimal...")
        augs = data.get('augmentations')
        try:
            search = find_best_hybrid(
                reps, labels, groups=groups,
                speed=search_speed,
                augmentations=augs,
                verbose=False,
                cache=True,
                dataset_name=data.get('dataset_name', 'unknown'),
                tier='general',
            )
            best = next(iter(search.best_by_model.values()), None)
            if best:
                optimal_budget = best['blueprint'].budget
                optimal_alloc = best['blueprint'].allocation_method
                optimal_imp = best['blueprint'].importance_method
                best_model = best['model']
                optimal_model_factory = _get_model_factories([best_model]).get(best_model)
                optimal_desc = (f"{best_model} at budget={optimal_budget} "
                                f"[{optimal_imp}/{optimal_alloc}] (from find_best_hybrid)")
                print(f"  Discovered optimal: {optimal_desc}")
        except Exception as e:
            print(f"  find_best_hybrid failed ({e}), using RF/budget=100 default")

    if optimal_model_factory is None:
        optimal_model_factory = lambda: RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1)

    # Define conditions: (model_factory, budget, alloc, imp, description)
    condition_specs = {
        'optimal': (optimal_model_factory, optimal_budget, optimal_alloc, optimal_imp,
                    optimal_desc),
        'underfit': (optimal_model_factory, 25, optimal_alloc, optimal_imp,
                     f'Same model at budget=25 (too compressed)'),
        'overfit': (optimal_model_factory, total_dim, optimal_alloc, optimal_imp,
                    f'Same model at full dim={total_dim} (too many features)'),
        'mismatched': (lambda: Ridge(alpha=0.01), total_dim, 'greedy', 'treeshap',
                       'Ridge at full dim (low-capacity model, high-dim input)'),
    }

    results = {'conditions': {}}

    for cond_name in all_conditions:
        if cond_name not in condition_specs:
            continue
        model_factory, budget, alloc, imp, desc = condition_specs[cond_name]
        print(f"\n  {cond_name}: {desc}")

        sample_X = next(iter(reps.values()))

        fold_metrics = []
        for fold_i, (train_idx, val_idx) in enumerate(_cv_split(groups, n_folds, sample_X, labels)):
            train_reps = {k: v[train_idx] for k, v in reps.items()}
            val_reps = {k: v[val_idx] for k, v in reps.items()}
            y_train, y_val = labels[train_idx], labels[val_idx]

            try:
                X_train, feature_info = create_hybrid(
                    train_reps, y_train,
                    allocation_method=alloc,
                    budget=budget,
                    importance_method=imp,
                )
                X_val = apply_feature_selection(val_reps, feature_info)
            except Exception as e:
                print(f"    fold {fold_i}: hybrid creation failed - {e}")
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # Train base model
            model = model_factory()
            model.fit(X_train_s, y_train)
            y_pred_val = model.predict(X_val_s)
            val_r2 = r2_score(y_val, y_pred_val)

            # Conformal calibration: split train into proper-train + cal
            n_train = len(y_train)
            n_cal = max(20, int(n_train * 0.2))
            cal_perm = np.random.RandomState(42 + fold_i).permutation(n_train)
            cal_idx = cal_perm[:n_cal]
            ptrain_idx = cal_perm[n_cal:]

            X_ptrain, y_ptrain = X_train_s[ptrain_idx], y_train[ptrain_idx]
            X_cal, y_cal = X_train_s[cal_idx], y_train[cal_idx]

            # Retrain on proper-train, calibrate on cal
            model_conf = model_factory()
            model_conf.fit(X_ptrain, y_ptrain)

            y_pred_cal = model_conf.predict(X_cal)
            cal_residuals = np.abs(y_cal - y_pred_cal)

            # Conformal quantile (90% coverage target)
            alpha = 0.1
            q = np.quantile(cal_residuals, 1 - alpha)

            # Predict on val with conformal intervals
            y_pred_conf = model_conf.predict(X_val_s)
            interval_lo = y_pred_conf - q
            interval_hi = y_pred_conf + q

            # Metrics
            covered = (y_val >= interval_lo) & (y_val <= interval_hi)
            picp = float(np.mean(covered))  # prediction interval coverage probability
            mpiw = float(np.mean(interval_hi - interval_lo))  # mean interval width
            target_range = float(np.ptp(y_val))
            sharpness_ratio = mpiw / target_range if target_range > 0 else float('nan')
            cal_error = abs(picp - (1 - alpha))

            # Uncertainty-error correlation (per-sample uncertainty from RF tree variance)
            errors = np.abs(y_val - y_pred_conf)
            ue_corr = float('nan')
            if hasattr(model_conf, 'estimators_') and len(errors) >= 3:
                tree_preds = np.array([t.predict(X_val_s) for t in model_conf.estimators_])
                per_sample_unc = tree_preds.std(axis=0)
                ue_corr, _ = spearmanr(per_sample_unc, errors)
                ue_corr = float(ue_corr)

            fold_result = {
                'val_r2': val_r2, 'picp': picp, 'mpiw': mpiw,
                'sharpness_ratio': sharpness_ratio, 'cal_error': cal_error,
                'conformal_q': float(q), 'n_val': len(y_val),
                'ue_corr': ue_corr,
            }
            fold_metrics.append(fold_result)
            print(f"    fold {fold_i}: R²={val_r2:.4f}, PICP={picp:.3f}, "
                  f"MPIW={mpiw:.3f}, cal_error={cal_error:.3f}")

        if fold_metrics:
            avg = {}
            for key in fold_metrics[0]:
                vals = [fm[key] for fm in fold_metrics if not np.isnan(fm[key])]
                avg[key] = float(np.mean(vals)) if vals else float('nan')
            results['conditions'][cond_name] = {
                'description': desc,
                'budget': budget,
                'avg': avg,
                'folds': fold_metrics,
            }
            print(f"  >> {cond_name} avg: R²={avg['val_r2']:.4f}, PICP={avg['picp']:.3f}, "
                  f"MPIW={avg['mpiw']:.3f}, cal_error={avg['cal_error']:.3f}")

    # Summary table
    print("\n  CALIBRATION SUMMARY:")
    print(f"  {'Condition':<14} {'R²':<8} {'PICP':<8} {'MPIW':<8} {'Cal Error':<10} {'Budget':<8}")
    print("  " + "-" * 56)
    for cond_name, cond_data in results['conditions'].items():
        a = cond_data['avg']
        print(f"  {cond_name:<14} {a['val_r2']:.4f}  {a['picp']:.3f}  "
              f"{a['mpiw']:.3f}  {a['cal_error']:.4f}    {cond_data['budget']}")

    return results


# =============================================================================
# EXPERIMENT 9: FORMAL COMPLEXITY ANALYSIS
# =============================================================================

def _compute_icc_check(configs: list) -> Dict[str, Any]:
    """Compute pairwise ICC(1,1) between ANOVA factor levels to detect redundancy.

    Following qsar_qm_models pattern: check whether model/budget pairs are
    redundant (ICC > 0.9) before running ANOVA.

    ICC(1,1) = (BMS - WMS) / (BMS + (k-1)*WMS)
    """
    models_set = sorted(set(c['model'] for c in configs))
    budgets_set = sorted(set(c['budget'] for c in configs))

    # Build mean R² matrix: rows = budgets, columns = models
    cell_means = {}
    for c in configs:
        key = (c['model'], c['budget'])
        fold_scores = c.get('fold_scores', [])
        vals = [s for s in fold_scores if not np.isnan(s)] if fold_scores else []
        if not vals and not np.isnan(c['r2_mean']):
            vals = [c['r2_mean']]
        if vals:
            cell_means[key] = float(np.mean(vals))

    result = {'model_pairs': {}, 'budget_pairs': {}, 'redundant_models': [], 'redundant_budgets': []}

    # --- Pairwise ICC between models (treating budgets as subjects) ---
    print("\n    ICC(1,1) between model pairs (budgets as subjects):")
    for i, m1 in enumerate(models_set):
        for m2 in models_set[i+1:]:
            # Get R² for each budget where both models have data
            rows = []
            for b in budgets_set:
                v1 = cell_means.get((m1, b))
                v2 = cell_means.get((m2, b))
                if v1 is not None and v2 is not None:
                    rows.append([v1, v2])

            if len(rows) < 2:
                continue

            data = np.array(rows)  # shape: (n_budgets, 2)
            n, k = data.shape
            grand_mean = data.mean()
            row_means = data.mean(axis=1)
            bms = k * ((row_means - grand_mean) ** 2).sum() / (n - 1)
            wms = ((data - row_means.reshape(-1, 1)) ** 2).sum() / (n * (k - 1))

            if (bms + (k - 1) * wms) == 0:
                icc = 1.0
            else:
                icc = (bms - wms) / (bms + (k - 1) * wms)

            pair_key = f"{m1}_vs_{m2}"
            result['model_pairs'][pair_key] = float(icc)

            flag = " ** REDUNDANT" if icc > 0.9 else ""
            print(f"      {m1} vs {m2}: ICC={icc:.4f}{flag}")

            if icc > 0.9:
                result['redundant_models'].append(pair_key)

    # --- Pairwise ICC between budgets (treating models as subjects) ---
    print("    ICC(1,1) between budget pairs (models as subjects):")
    for i, b1 in enumerate(budgets_set):
        for b2 in budgets_set[i+1:]:
            rows = []
            for m in models_set:
                v1 = cell_means.get((m, b1))
                v2 = cell_means.get((m, b2))
                if v1 is not None and v2 is not None:
                    rows.append([v1, v2])

            if len(rows) < 2:
                continue

            data = np.array(rows)
            n, k = data.shape
            grand_mean = data.mean()
            row_means = data.mean(axis=1)
            bms = k * ((row_means - grand_mean) ** 2).sum() / (n - 1)
            wms = ((data - row_means.reshape(-1, 1)) ** 2).sum() / (n * (k - 1))

            if (bms + (k - 1) * wms) == 0:
                icc = 1.0
            else:
                icc = (bms - wms) / (bms + (k - 1) * wms)

            pair_key = f"b{b1}_vs_b{b2}"
            result['budget_pairs'][pair_key] = float(icc)

            flag = " ** REDUNDANT" if icc > 0.9 else ""
            print(f"      b={b1} vs b={b2}: ICC={icc:.4f}{flag}")

            if icc > 0.9:
                result['redundant_budgets'].append(pair_key)

    if result['redundant_models']:
        print(f"    WARNING: {len(result['redundant_models'])} redundant model pair(s) detected")
    if result['redundant_budgets']:
        print(f"    WARNING: {len(result['redundant_budgets'])} redundant budget pair(s) detected")

    return result


def _run_anova_decomposition(exp2_results: Dict[str, Any]) -> Dict[str, Any]:
    """9a: ANOVA variance decomposition on Exp 2 grid results.

    Tests whether model×budget interaction explains significant variance in R².
    Uses per-fold CV scores as replicates for proper within-cell variance estimation.
    """
    print("\n  9a: ANOVA Variance Decomposition")

    configs = exp2_results.get('configs', [])
    if not configs:
        print("    No Exp 2 configs available")
        return {'error': 'no_data'}

    # ICC redundancy check (following qsar_qm_models pattern)
    icc_results = _compute_icc_check(configs)

    # Build data: each config is an observation with factors (model, budget) → R²
    models_set = sorted(set(c['model'] for c in configs))
    budgets_set = sorted(set(c['budget'] for c in configs))

    if len(models_set) < 2 or len(budgets_set) < 2:
        print("    Need at least 2 models and 2 budgets for ANOVA")
        return {'error': 'insufficient_factors'}

    # Collect R² values into cells: cell[model][budget] = list of r2 values
    # Use per-fold scores as replicates (real variance) instead of mean R²
    # which can produce zero within-cell variance with identical configs
    cells = {}
    for c in configs:
        key = (c['model'], c['budget'])
        if key not in cells:
            cells[key] = []
        fold_scores = c.get('fold_scores', [])
        if fold_scores:
            cells[key].extend([s for s in fold_scores if not np.isnan(s)])
        elif not np.isnan(c['r2_mean']):
            # Fallback: use mean if fold_scores not available (old cached results)
            cells[key].append(c['r2_mean'])

    # Compute means
    grand_mean = np.mean([r2 for vals in cells.values() for r2 in vals])
    n_total = sum(len(v) for v in cells.values())

    model_means = {}
    for m in models_set:
        vals = [r2 for (mm, b), vv in cells.items() for r2 in vv if mm == m]
        model_means[m] = np.mean(vals) if vals else grand_mean

    budget_means = {}
    for b in budgets_set:
        vals = [r2 for (mm, bb), vv in cells.items() for r2 in vv if bb == b]
        budget_means[b] = np.mean(vals) if vals else grand_mean

    cell_means = {}
    for key, vals in cells.items():
        cell_means[key] = np.mean(vals) if vals else grand_mean

    # Sum of squares
    ss_total = sum((r2 - grand_mean) ** 2 for vals in cells.values() for r2 in vals)

    ss_model = sum(
        len([r2 for (mm, b), vv in cells.items() for r2 in vv if mm == m])
        * (model_means[m] - grand_mean) ** 2
        for m in models_set
    )

    ss_budget = sum(
        len([r2 for (mm, b), vv in cells.items() for r2 in vv if b == bb])
        * (budget_means[bb] - grand_mean) ** 2
        for bb in budgets_set
    )

    ss_interaction = 0
    for (m, b), vals in cells.items():
        expected = model_means[m] + budget_means[b] - grand_mean
        ss_interaction += len(vals) * (cell_means[(m, b)] - expected) ** 2

    ss_residual = max(0, ss_total - ss_model - ss_budget - ss_interaction)

    # Eta-squared (% variance)
    eta2 = {}
    if ss_total > 0:
        eta2['model'] = float(ss_model / ss_total * 100)
        eta2['budget'] = float(ss_budget / ss_total * 100)
        eta2['interaction'] = float(ss_interaction / ss_total * 100)
        eta2['residual'] = float(ss_residual / ss_total * 100)
    else:
        eta2 = {'model': 0, 'budget': 0, 'interaction': 0, 'residual': 0}

    # Degrees of freedom and F-statistics
    df_model = len(models_set) - 1
    df_budget = len(budgets_set) - 1
    df_interaction = df_model * df_budget
    df_residual = max(1, n_total - len(models_set) * len(budgets_set))

    ms_model = ss_model / df_model if df_model > 0 else 0
    ms_budget = ss_budget / df_budget if df_budget > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 1e-10

    f_model = ms_model / ms_residual
    f_budget = ms_budget / ms_residual
    f_interaction = ms_interaction / ms_residual

    p_model = 1 - f_dist.cdf(f_model, df_model, df_residual)
    p_budget = 1 - f_dist.cdf(f_budget, df_budget, df_residual)
    p_interaction = 1 - f_dist.cdf(f_interaction, df_interaction, df_residual)

    result = {
        'n_observations': n_total,
        'n_models': len(models_set),
        'n_budgets': len(budgets_set),
        'grand_mean_r2': float(grand_mean),
        'eta2': eta2,
        'f_model': float(f_model), 'p_model': float(p_model),
        'f_budget': float(f_budget), 'p_budget': float(p_budget),
        'f_interaction': float(f_interaction), 'p_interaction': float(p_interaction),
        'ss_total': float(ss_total),
        'icc': icc_results,
    }

    print(f"    Grand mean R²: {grand_mean:.4f}")
    print(f"    η² model:       {eta2['model']:.1f}%  (F={f_model:.2f}, p={p_model:.4f})")
    print(f"    η² budget:      {eta2['budget']:.1f}%  (F={f_budget:.2f}, p={p_budget:.4f})")
    print(f"    η² interaction: {eta2['interaction']:.1f}%  (F={f_interaction:.2f}, p={p_interaction:.4f})")
    print(f"    η² residual:    {eta2['residual']:.1f}%")

    return result


def _run_information_retention(
    data: dict,
    budgets: List[int],
    search_speed: str = 'fast',
) -> Dict[str, Any]:
    """9b: MI(X_selected, Y) and MI(X_selected, X_full) vs budget.

    At each budget, uses find_best_hybrid to discover the best hybrid config,
    then profiles its information content.
    """
    print("\n  9b: Information Retention Curves")

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    results = {'budgets': [], 'mi_xy': [], 'mi_xx_full': [], 'best_configs': []}

    augs = data.get('augmentations')
    for budget in sorted(budgets):
        # Find best hybrid at this budget
        try:
            search = find_best_hybrid(
                reps, labels, groups=groups,
                speed=search_speed, budgets=[budget],
                augmentations=augs,
                verbose=False,
                cache=True,
                dataset_name=f"{data.get('dataset_name', 'unknown')}_b{budget}",
                tier='general',
            )
            best = next(iter(search.best_by_model.values()), None)
            if best:
                bp = best['blueprint']
            else:
                bp = None
        except Exception:
            bp = None

        # Create the best hybrid at this budget
        alloc = bp.allocation_method if bp else 'greedy'
        imp = bp.importance_method if bp else 'treeshap'
        try:
            X_hybrid, _ = create_hybrid(
                reps, labels,
                allocation_method=alloc,
                importance_method=imp,
                budget=budget,
            )
        except Exception as e:
            print(f"    budget={budget}: hybrid creation failed - {e}")
            continue

        results['best_configs'].append({'budget': budget, 'importance': imp, 'allocation': alloc})

        # MI(X_selected, Y)
        n_comp = min(50, X_hybrid.shape[1], X_hybrid.shape[0] - 1)
        X_red = PCA(n_components=n_comp).fit_transform(
            StandardScaler().fit_transform(X_hybrid))
        try:
            mi_xy = estimate_mutual_information_knn(X_red, labels, k=5)
        except Exception:
            mi_xy = float('nan')

        # MI(X_selected, X_full) — approximate via explained variance
        # Project X_full onto space spanned by X_hybrid columns
        scaler = StandardScaler()
        X_full_s = scaler.fit_transform(X_concat)
        X_hyb_s = StandardScaler().fit_transform(X_hybrid)
        # Use canonical correlation proxy: R² of predicting X_full PCs from X_hybrid
        n_pcs = min(20, X_hybrid.shape[1], X_full_s.shape[0] - 1)
        pca_full = PCA(n_components=n_pcs)
        X_full_pcs = pca_full.fit_transform(X_full_s)
        explained_var_retained = 0
        try:
            from sklearn.linear_model import Ridge as _Ridge
            for pc_i in range(n_pcs):
                reg = _Ridge(alpha=1.0)
                reg.fit(X_hyb_s, X_full_pcs[:, pc_i])
                r2_pc = max(0, r2_score(X_full_pcs[:, pc_i], reg.predict(X_hyb_s)))
                explained_var_retained += r2_pc * pca_full.explained_variance_ratio_[pc_i]
            mi_xx = float(explained_var_retained)
        except Exception:
            mi_xx = float('nan')

        results['budgets'].append(budget)
        results['mi_xy'].append(float(mi_xy))
        results['mi_xx_full'].append(mi_xx)

        print(f"    budget={budget}: MI(X,Y)={mi_xy:.4f}, info_retained={mi_xx:.4f}")

    # Find inflection: where MI(X,Y) gain drops below 10% of max gain
    if len(results['mi_xy']) >= 3:
        mi_arr = np.array(results['mi_xy'])
        gains = np.diff(mi_arr)
        max_gain = np.max(np.abs(gains)) if len(gains) > 0 else 0
        if max_gain > 0:
            plateau_mask = gains < 0.1 * max_gain
            if np.any(plateau_mask):
                inflection_idx = int(np.argmax(plateau_mask))
                results['inflection_budget'] = results['budgets'][inflection_idx + 1]
                print(f"    Inflection point: budget ~{results['inflection_budget']}")
            else:
                print("    No inflection detected (MI still increasing)")

    return results


def _run_effective_rank_analysis(
    data: dict,
    budgets: List[int],
    search_speed: str = 'fast',
) -> Dict[str, Any]:
    """9c: Effective rank of hybrid vs PCA vs random at each budget.

    At each budget, uses find_best_hybrid to discover the best config,
    then compares effective rank: best hybrid vs PCA vs random subset.
    """
    print("\n  9c: Effective Rank Profile")

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    results = {'budgets': [], 'hybrid_rank': [], 'pca_rank': [], 'random_rank': []}

    augs = data.get('augmentations')
    for budget in sorted(budgets):
        n_feat = min(budget, X_concat.shape[1], X_concat.shape[0] - 1)

        # Find best hybrid at this budget, then compute its effective rank
        try:
            search = find_best_hybrid(
                reps, labels, groups=groups,
                speed=search_speed, budgets=[budget],
                augmentations=augs,
                verbose=False,
                cache=True,
                dataset_name=f"{data.get('dataset_name', 'unknown')}_b{budget}",
                tier='general',
            )
            best = next(iter(search.best_by_model.values()), None)
            bp = best['blueprint'] if best else None
        except Exception:
            bp = None

        alloc = bp.allocation_method if bp else 'greedy'
        imp = bp.importance_method if bp else 'treeshap'
        try:
            X_hybrid, _ = create_hybrid(
                reps, labels,
                allocation_method=alloc,
                importance_method=imp,
                budget=budget,
            )
            hybrid_rank = compute_effective_rank(X_hybrid)
        except Exception:
            hybrid_rank = float('nan')

        # PCA
        try:
            pca = PCA(n_components=n_feat)
            X_pca = pca.fit_transform(StandardScaler().fit_transform(X_concat))
            pca_rank = compute_effective_rank(X_pca)
        except Exception:
            pca_rank = float('nan')

        # Random subset
        try:
            rng = np.random.RandomState(42)
            random_idx = rng.choice(X_concat.shape[1], n_feat, replace=False)
            X_random = X_concat[:, random_idx]
            random_rank = compute_effective_rank(X_random)
        except Exception:
            random_rank = float('nan')

        results['budgets'].append(budget)
        results['hybrid_rank'].append(float(hybrid_rank))
        results['pca_rank'].append(float(pca_rank))
        results['random_rank'].append(float(random_rank))

        print(f"    budget={budget}: hybrid={hybrid_rank:.1f}, pca={pca_rank:.1f}, random={random_rank:.1f}")

    return results


def _run_bias_variance_analysis(
    data: dict,
    models: Dict[str, callable],
    budgets: List[int],
    n_folds: int,
    exp6_results: Dict[str, Any] = None,
    search_speed: str = 'fast',
) -> Dict[str, Any]:
    """9d: Bias-variance proxy from train-val gap across budgets."""
    print("\n  9d: Bias-Variance Proxy")

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'models': {}}

    # If Exp 6 results are available, extract train-val gaps from there
    if exp6_results and 'families' in exp6_results:
        print("    Using Exp 6 data for bias-variance analysis")
        for fam_name, fam_data in exp6_results['families'].items():
            fam_bv = {}
            for var_name, budget_data in fam_data.get('heatmap', {}).items():
                for budget, metrics in budget_data.items():
                    if isinstance(metrics, dict) and 'train_r2' in metrics:
                        key = f"{var_name}_b{budget}"
                        fam_bv[key] = {
                            'train_r2': metrics['train_r2'],
                            'val_r2': metrics['val_r2'],
                            'gap': metrics['train_r2'] - metrics['val_r2'],
                            'bias_proxy': 1 - metrics['train_r2'],
                            'budget': int(budget) if isinstance(budget, (int, float)) else budget,
                        }
            if fam_bv:
                results['models'][fam_name] = fam_bv
        if results['models']:
            # Print summary
            for fam_name, fam_data in results['models'].items():
                print(f"    {fam_name}:")
                for key, vals in sorted(fam_data.items()):
                    print(f"      {key}: gap={vals['gap']:.3f}, bias={vals['bias_proxy']:.3f}")
            return results

    # Fallback: discover best config at each budget via find_best_hybrid, then
    # measure train-val gap with that config
    print("    Computing bias-variance from scratch (using find_best_hybrid per budget)")
    augs = data.get('augmentations')
    model_names = list(models.keys())
    for budget in budgets:
        # Find best config at this budget
        try:
            search = find_best_hybrid(
                reps, labels, groups=groups,
                models=model_names,
                speed=search_speed,
                budgets=[budget],
                augmentations=augs,
                verbose=False,
                cache=True,
                dataset_name=f"{data.get('dataset_name', 'unknown')}_b{budget}",
                tier='general',
            )
        except Exception as e:
            print(f"    b={budget}: find_best_hybrid failed - {e}")
            continue

        for model_name, model_factory in models.items():
            # Get best config for this model at this budget
            if model_name in search.best_by_model:
                bp = search.best_by_model[model_name]['blueprint']
                alloc = bp.allocation_method
                imp = bp.importance_method
            else:
                alloc, imp = 'greedy', 'treeshap'

            try:
                val_r2, val_std, train_r2, train_std, n_feat = evaluate_cv_with_train(
                    reps, labels, groups, model_factory, n_folds,
                    allocation_method=alloc,
                    importance_method=imp,
                    budget=budget,
                )
                if model_name not in results['models']:
                    results['models'][model_name] = {}
                results['models'][model_name][budget] = {
                    'train_r2': train_r2, 'val_r2': val_r2,
                    'gap': train_r2 - val_r2,
                    'bias_proxy': 1 - train_r2,
                    'budget': budget,
                    'importance': imp,
                    'allocation': alloc,
                }
                print(f"    {model_name}/b={budget}: gap={train_r2 - val_r2:.3f}, "
                      f"bias={1 - train_r2:.3f}, val_R²={val_r2:.4f} [{imp}/{alloc}]")
            except Exception as e:
                print(f"    {model_name}/b={budget}: FAILED - {e}")

    return results


def run_formal_complexity_analysis(
    data: dict,
    models: Dict[str, callable],
    budgets: List[int],
    n_folds: int,
    sub_analyses: List[str] = None,
    exp2_results: Dict[str, Any] = None,
    exp6_results: Dict[str, Any] = None,
    search_speed: str = 'fast',
) -> Dict[str, Any]:
    """Exp 9: Formal theoretical grounding with 4 sub-analyses."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: FORMAL COMPLEXITY ANALYSIS")
    print("=" * 70)

    all_subs = sub_analyses or ['anova', 'information', 'effective_rank', 'bias_variance']
    results = {}

    if 'anova' in all_subs and exp2_results:
        results['anova'] = _run_anova_decomposition(exp2_results)

    if 'information' in all_subs:
        results['information'] = _run_information_retention(data, budgets, search_speed)

    if 'effective_rank' in all_subs:
        results['effective_rank'] = _run_effective_rank_analysis(data, budgets, search_speed)

    if 'bias_variance' in all_subs:
        results['bias_variance'] = _run_bias_variance_analysis(
            data, models, budgets, n_folds, exp6_results, search_speed)

    return results


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# EXPERIMENT 10: NOISE ROBUSTNESS & FEATURE STABILITY
# =============================================================================

def run_noise_robustness(
    data: dict,
    models: Dict[str, callable],
    budgets: List[int],
    n_folds: int,
    sigmas: List[float] = None,
    strategies: List[str] = None,
    search_speed: str = 'fast',
) -> Dict[str, Any]:
    """Exp 10: Noise Degradation Slope (NDS) across noise strategies.

    Tests whether hybrids are more noise-robust than individual reps, and whether
    the optimal hybrid config changes under noise. Also tracks feature stability
    across noise levels (do the same features stay important when noise increases?).

    NDS = slope of R² vs sigma. More negative = more noise-sensitive.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: NOISE ROBUSTNESS & FEATURE STABILITY")
    print("=" * 70)

    if not HAS_NOISEINJECT:
        print("  NoiseInject not installed, skipping. Install: pip install noiseInject")
        return {'error': 'noiseInject not installed'}

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    if sigmas is None:
        sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    if strategies is None:
        strategies = ['legacy', 'quantile', 'hetero']

    results = {
        'sigmas': sigmas,
        'strategies': strategies,
        'individual_reps': {},
        'hybrids': {},
        'feature_stability': {},
    }

    sample_X = next(iter(reps.values()))

    from sklearn.ensemble import RandomForestRegressor as _RF

    # --- Phase 1: Individual rep noise robustness ---
    print("\n  Phase 1: Individual rep NDS")
    for rep_name, X_rep in reps.items():
        print(f"    {rep_name} (dim={X_rep.shape[1]}):")
        rep_results = {}

        for strategy in strategies:
            noise_injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
            sigma_scores = {}

            for sigma in sigmas:
                fold_scores = []
                for train_idx, val_idx in _cv_split(groups, n_folds, sample_X, labels):
                    y_train = labels[train_idx].copy()
                    y_val = labels[val_idx]

                    # Inject noise into training labels only
                    if sigma > 0:
                        y_train = noise_injector.inject(y_train, sigma=sigma)

                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_rep[train_idx])
                    X_va = scaler.transform(X_rep[val_idx])

                    rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X_tr, y_train)
                    fold_scores.append(r2_score(y_val, rf.predict(X_va)))

                sigma_scores[sigma] = float(np.mean(fold_scores))

            # Compute NDS: slope of R² vs sigma
            s_arr = np.array(list(sigma_scores.keys()))
            r2_arr = np.array(list(sigma_scores.values()))
            if len(s_arr) >= 2 and np.std(s_arr) > 0:
                slope, _ = np.polyfit(s_arr, r2_arr, 1)
                nds = float(slope)
            else:
                nds = 0.0

            baseline_r2 = sigma_scores.get(0.0, sigma_scores.get(min(sigmas)))
            max_noise_r2 = sigma_scores.get(max(sigmas), float('nan'))
            retention = float(max_noise_r2 / baseline_r2 * 100) if baseline_r2 > 0 else 0.0

            rep_results[strategy] = {
                'scores_by_sigma': sigma_scores,
                'nds': nds,
                'baseline_r2': baseline_r2,
                'retention_pct': retention,
            }
            print(f"      {strategy}: NDS={nds:.4f}, base_R²={baseline_r2:.4f}, "
                  f"retention={retention:.1f}%")

        results['individual_reps'][rep_name] = rep_results

    # --- Phase 2: Hybrid noise robustness at each budget ---
    print("\n  Phase 2: Hybrid NDS by budget")

    for budget in budgets:
        print(f"\n    Budget={budget}:")
        budget_results = {}

        for strategy in strategies:
            noise_injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
            sigma_scores = {}

            for sigma in sigmas:
                fold_scores = []
                for train_idx, val_idx in _cv_split(groups, n_folds, sample_X, labels):
                    y_train = labels[train_idx].copy()
                    y_val = labels[val_idx]

                    if sigma > 0:
                        y_train = noise_injector.inject(y_train, sigma=sigma)

                    train_reps = {k: v[train_idx] for k, v in reps.items()}
                    val_reps = {k: v[val_idx] for k, v in reps.items()}

                    try:
                        X_train, fi = create_hybrid(
                            train_reps, y_train,
                            allocation_method='greedy',
                            budget=budget,
                            importance_method='treeshap',
                        )
                        X_val = apply_feature_selection(val_reps, fi)
                    except Exception:
                        continue

                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_train)
                    X_va = scaler.transform(X_val)

                    rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X_tr, y_train)
                    fold_scores.append(r2_score(y_val, rf.predict(X_va)))

                if fold_scores:
                    sigma_scores[sigma] = float(np.mean(fold_scores))

            # NDS
            s_arr = np.array(list(sigma_scores.keys()))
            r2_arr = np.array(list(sigma_scores.values()))
            if len(s_arr) >= 2 and np.std(s_arr) > 0:
                slope, _ = np.polyfit(s_arr, r2_arr, 1)
                nds = float(slope)
            else:
                nds = 0.0

            baseline_r2 = sigma_scores.get(0.0, sigma_scores.get(min(sigmas), 0))
            max_noise_r2 = sigma_scores.get(max(sigmas), float('nan'))
            retention = float(max_noise_r2 / baseline_r2 * 100) if baseline_r2 > 0 else 0.0

            budget_results[strategy] = {
                'scores_by_sigma': sigma_scores,
                'nds': nds,
                'baseline_r2': baseline_r2,
                'retention_pct': retention,
            }
            print(f"      {strategy}: NDS={nds:.4f}, base_R²={baseline_r2:.4f}, "
                  f"retention={retention:.1f}%")

        results['hybrids'][budget] = budget_results

    # --- Phase 3: Feature stability across noise levels ---
    print("\n  Phase 3: Feature stability across noise")
    budget_for_stability = budgets[len(budgets) // 2]  # middle budget
    print(f"    Using budget={budget_for_stability}")

    for strategy in strategies[:1]:  # just legacy for feature stability (expensive)
        noise_injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
        features_by_sigma = {}

        for sigma in sigmas:
            fold_features = []
            for train_idx, val_idx in _cv_split(groups, n_folds, sample_X, labels):
                y_train = labels[train_idx].copy()
                if sigma > 0:
                    y_train = noise_injector.inject(y_train, sigma=sigma)

                train_reps = {k: v[train_idx] for k, v in reps.items()}
                try:
                    _, fi = create_hybrid(
                        train_reps, y_train,
                        allocation_method='greedy',
                        budget=budget_for_stability,
                        importance_method='treeshap',
                    )
                    # Collect selected feature identifiers: (rep_name, feature_idx) tuples
                    selected = set()
                    for rep_name, indices in fi.selected_indices.items():
                        for idx in indices:
                            selected.add((rep_name, int(idx)))
                    fold_features.append(selected)
                except Exception:
                    continue

            features_by_sigma[sigma] = fold_features

        # Feature stability across folds (at sigma=0)
        if features_by_sigma.get(0.0) and len(features_by_sigma[0.0]) >= 2:
            fold_sets = features_by_sigma[0.0]
            jaccard_across_folds = []
            for i in range(len(fold_sets)):
                for j in range(i + 1, len(fold_sets)):
                    inter = len(fold_sets[i] & fold_sets[j])
                    union = len(fold_sets[i] | fold_sets[j])
                    jaccard_across_folds.append(inter / union if union > 0 else 0)
            results['feature_stability']['fold_jaccard_sigma0'] = float(np.mean(jaccard_across_folds))
            print(f"    Fold stability (sigma=0): Jaccard={np.mean(jaccard_across_folds):.3f}")

        # Feature stability across noise levels (compare sigma=0 features to each sigma)
        if features_by_sigma.get(0.0):
            ref_features = set.union(*features_by_sigma[0.0])  # union across folds at sigma=0
            noise_stability = {}
            for sigma in sigmas:
                if sigma == 0.0 or sigma not in features_by_sigma:
                    continue
                if not features_by_sigma[sigma]:
                    continue
                noisy_features = set.union(*features_by_sigma[sigma])
                inter = len(ref_features & noisy_features)
                union = len(ref_features | noisy_features)
                jaccard = inter / union if union > 0 else 0.0
                noise_stability[sigma] = float(jaccard)
                print(f"    sigma={sigma:.1f} vs sigma=0: Jaccard={jaccard:.3f} "
                      f"(shared={inter}/{len(ref_features)} ref features)")

            results['feature_stability']['noise_jaccard'] = noise_stability

            # NDS of feature stability (does feature overlap degrade linearly with noise?)
            if len(noise_stability) >= 2:
                s_arr = np.array(list(noise_stability.keys()))
                j_arr = np.array(list(noise_stability.values()))
                slope, _ = np.polyfit(s_arr, j_arr, 1)
                results['feature_stability']['feature_nds'] = float(slope)
                print(f"    Feature stability NDS: {slope:.4f} (negative = features change with noise)")

        # Per-rep allocation stability across noise
        if features_by_sigma.get(0.0):
            print("\n    Per-rep feature count stability:")
            for sigma in [0.0] + [s for s in sigmas if s > 0]:
                if sigma not in features_by_sigma or not features_by_sigma[sigma]:
                    continue
                # Average feature count per rep across folds
                rep_counts = {}
                for fold_feats in features_by_sigma[sigma]:
                    fold_reps = set(r for r, _ in fold_feats)
                    for rep_name in fold_reps:
                        rep_counts.setdefault(rep_name, [])
                        count = sum(1 for r, _ in fold_feats if r == rep_name)
                        rep_counts[rep_name].append(count)
                counts_str = ", ".join(
                    f"{r}={np.mean(c):.0f}" for r, c in sorted(rep_counts.items()))
                print(f"      sigma={sigma:.1f}: {counts_str}")

    # --- Summary table ---
    print("\n  NOISE ROBUSTNESS SUMMARY (NDS, higher = more robust):")
    print(f"  {'Representation':<20} " +
          " ".join(f"{s:<12}" for s in strategies))
    print("  " + "-" * (20 + 13 * len(strategies)))

    # Individual reps
    for rep_name in sorted(results['individual_reps'].keys()):
        row = f"  {rep_name:<20} "
        for s in strategies:
            nds = results['individual_reps'][rep_name].get(s, {}).get('nds', float('nan'))
            row += f"{nds:>+8.4f}    "
        print(row)

    # Hybrids
    for budget in sorted(b for b in results['hybrids'] if isinstance(b, (int, float))):
        row = f"  hybrid_b{budget:<11} "
        for s in strategies:
            nds = results['hybrids'][budget].get(s, {}).get('nds', float('nan'))
            row += f"{nds:>+8.4f}    "
        print(row)

    return results


def main():
    parser = argparse.ArgumentParser(description='Complexity theory experiments')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'],
                        help='quick (~30min, logd only) or full (~4-6h, all datasets)')
    parser.add_argument('--experiments', type=int, nargs='+', default=None,
                        help='Which experiments to run (1-10). Default: all.')
    parser.add_argument('--reps', type=str, default=None,
                        help='Representation preset (fast, fast+hf, all) or comma-separated list. '
                             'Default: fast for quick, fast+hf for full.')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Override dataset list (e.g., --datasets logd caco2)')
    parser.add_argument('--output', type=str, default='tests/results/complexity_theory')
    parser.add_argument('--delta', action='store_true',
                        help='Generate delta representations for antibody datasets '
                             '(ΔX = X_mutant - X_wildtype). Only applies to flab_* datasets.')
    parser.add_argument('--augment', action='store_true',
                        help='Include augmented features (developability, humanness, etc.) '
                             'for antibody datasets.')

    args = parser.parse_args()

    # Configuration based on mode
    if args.mode == 'quick':
        dataset_names = ['logd']
        rep_preset = args.reps or 'fast'
        n_folds = 3
        model_names = ['rf', 'knn', 'ridge']
        budgets = [25, 50, 100, 200]
        allocation_methods = ['greedy']
        importance_methods = ['treeshap']
        exp4_importance = ['treeshap', 'lightgbm_gain', 'deeplift', 'permutation']
        exp6_families = ['RF', 'Ridge', 'MLP']
        exp7_partitions = ['scaffold_frequency']
        exp8_conditions = ['optimal', 'overfit']
        exp9_subs = ['anova', 'bias_variance']
        exp10_sigmas = [0.0, 0.1, 0.3, 0.6]
        exp10_strategies = ['legacy', 'quantile']
        search_speed = 'fast'
    else:  # full
        dataset_names = ['logd', 'caco2', 'herg_ki', 'qm9']
        rep_preset = args.reps or 'fast+hf'
        n_folds = 5
        model_names = ['rf', 'knn', 'ridge', 'xgboost', 'mlp', 'lightgbm']
        budgets = [10, 25, 50, 100, 200, 500]
        allocation_methods = ['greedy', 'greedy_feature', 'performance_weighted']
        importance_methods = ['treeshap', 'lightgbm_gain', 'deeplift', 'permutation']
        exp4_importance = [
            'treeshap', 'xgboost_gain', 'lightgbm_gain', 'deeplift', 'permutation', 'kernelshap',
        ]
        exp6_families = None  # all families
        exp7_partitions = None  # all partitions
        exp8_conditions = None  # all conditions
        exp9_subs = None  # all sub-analyses
        exp10_sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        exp10_strategies = ['legacy', 'quantile', 'hetero', 'outlier', 'threshold', 'valprop']
        search_speed = 'medium'

    if args.datasets:
        dataset_names = args.datasets

    experiments = args.experiments if args.experiments else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    models = _get_model_factories(model_names)

    print(f"Mode: {args.mode}")
    print(f"Datasets: {dataset_names}")
    print(f"Reps: {rep_preset}")
    print(f"Models: {list(models.keys())}")
    print(f"Experiments: {experiments}")

    # Load all datasets
    datasets = {}
    for ds_name in dataset_names:
        print(f"\nLoading {ds_name}...")
        try:
            datasets[ds_name] = load_data(ds_name, rep_preset=rep_preset,
                                          delta=args.delta, augment=args.augment)
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
            'models': list(models.keys()),
            'experiments': experiments,
            'n_folds': n_folds,
            'budgets': budgets,
            'delta': args.delta,
            'augment': args.augment,
            'timestamp': datetime.now().isoformat(),
        },
        'datasets': {},
    }

    # Incremental save — write checkpoint after each experiment completes
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

    checkpoint_file = output_path / f"complexity_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def _save_checkpoint(label=''):
        with open(checkpoint_file, 'w') as f:
            json.dump(convert_numpy(all_results), f, indent=2)
        print(f"  [checkpoint saved{': ' + label if label else ''}]")

    for ds_name, data in datasets.items():
        data['dataset_name'] = ds_name
        print(f"\n{'#' * 70}")
        print(f"# DATASET: {ds_name.upper()} ({len(data['labels'])} samples)")
        print('#' * 70)

        ds_results = {'n_samples': len(data['labels']), 'n_scaffolds': data['n_scaffolds']}

        if 1 in experiments:
            ds_results['exp1_characterization'] = run_characterization(data, models, n_folds)
            all_results['datasets'][ds_name] = ds_results
            _save_checkpoint(f'{ds_name} exp1')

        if 2 in experiments:
            ds_results['exp2_complexity_grid'] = run_complexity_matching_grid(
                data, models, budgets, allocation_methods, importance_methods, n_folds)
            all_results['datasets'][ds_name] = ds_results
            _save_checkpoint(f'{ds_name} exp2')

        if 4 in experiments:
            ds_results['exp4_importance_metrics'] = run_importance_weight_metrics(
                data, exp4_importance, n_folds)
            all_results['datasets'][ds_name] = ds_results
            _save_checkpoint(f'{ds_name} exp4')

        if 5 in experiments:
            ds_results['exp5_learning_curves'] = run_learning_curves(
                data, models, n_folds, search_speed=search_speed)
            all_results['datasets'][ds_name] = ds_results
            _save_checkpoint(f'{ds_name} exp5')

        if 6 in experiments:
            # Exp 6 needs finer budget granularity to detect capacity-dimension interaction
            exp6_budgets = sorted(set([10] + budgets))
            ds_results['exp6_within_family'] = run_within_family_complexity(
                data, exp6_budgets, n_folds, families=exp6_families,
                search_speed=search_speed)
            all_results['datasets'][ds_name] = ds_results
            _save_checkpoint(f'{ds_name} exp6')

        if 7 in experiments:
            ds_results['exp7_data_complexity'] = run_data_complexity_analysis(
                data, models, budgets, n_folds, partitions=exp7_partitions,
                search_speed=search_speed)
            all_results['datasets'][ds_name] = ds_results
            _save_checkpoint(f'{ds_name} exp7')

        if 8 in experiments:
            ds_results['exp8_uncertainty'] = run_uncertainty_calibration(
                data, n_folds, conditions=exp8_conditions,
                exp2_results=ds_results.get('exp2_complexity_grid'),
                search_speed=search_speed)
            all_results['datasets'][ds_name] = ds_results
            _save_checkpoint(f'{ds_name} exp8')

        if 9 in experiments:
            ds_results['exp9_formal'] = run_formal_complexity_analysis(
                data, models, budgets, n_folds,
                sub_analyses=exp9_subs,
                exp2_results=ds_results.get('exp2_complexity_grid'),
                exp6_results=ds_results.get('exp6_within_family'),
                search_speed=search_speed,
            )
            all_results['datasets'][ds_name] = ds_results
            _save_checkpoint(f'{ds_name} exp9')

        if 10 in experiments:
            ds_results['exp10_noise'] = run_noise_robustness(
                data, models, budgets, n_folds,
                sigmas=exp10_sigmas, strategies=exp10_strategies,
                search_speed=search_speed)
            all_results['datasets'][ds_name] = ds_results
            _save_checkpoint(f'{ds_name} exp10')

        all_results['datasets'][ds_name] = ds_results

    # Experiment 3 is cross-dataset (needs all datasets loaded)
    if 3 in experiments and len(datasets) >= 1:
        all_results['exp3_cross_context'] = run_cross_context_analysis(datasets, n_folds)
        _save_checkpoint('exp3 cross-context')

    print(f"\n{'=' * 70}")
    print(f"All experiments complete. Final save.")
    print('=' * 70)


if __name__ == '__main__':
    main()
