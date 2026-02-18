"""Shared utilities for complexity and similarity experiment scripts.

Extracted from complexity_theory_experiments.py and similarity_experiments.py.
Provides mode configs, CV helpers, model factories, metrics, data loading, and IO.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import digamma
from scipy.stats import spearmanr, f as f_dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

# Add src and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from kirby.hybrid import create_hybrid, apply_feature_selection, compute_feature_importance
from kirby.find_best_hybrid import find_best_hybrid, SearchResult

warnings.filterwarnings('ignore')


# =============================================================================
# MODE CONFIGURATIONS
# =============================================================================

INTERMEDIARY = {
    'n_splits': 1,
    'models': ['rf', 'ridge', 'lightgbm'],
    'budgets': [10, 25, 50, 100, 200, 500],
    'allocations': ['greedy', 'performance_weighted'],
    'importance_methods': ['treeshap', 'lgbm_gain'],
    'search_speed': 'fast',
}

FULL = {
    'n_splits': 5,
    'models': ['rf', 'knn', 'ridge', 'mlp', 'xgboost', 'lightgbm'],
    'budgets': [10, 25, 50, 100, 200, 500],
    'allocations': ['greedy', 'performance_weighted'],
    'importance_methods': ['treeshap', 'lgbm_gain', 'permutation', 'shap'],
    'search_speed': 'medium',
}


# =============================================================================
# CV SPLITTING
# =============================================================================

def cv_split(groups, n_splits, X, labels, seed=42, mode='intermediary'):
    """Yield (train_idx, val_idx) using appropriate CV strategy.

    - mode='intermediary' with n_splits=1: single 80/20 GroupShuffleSplit
    - mode='full' or n_splits>1: GroupKFold (or KFold if not enough groups)
    """
    if n_splits == 1 and mode == 'intermediary':
        if groups is not None and len(np.unique(groups)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            yield from gss.split(X, labels, groups=groups)
        else:
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            yield from ss.split(X, labels)
    else:
        if groups is not None and len(np.unique(groups)) >= n_splits:
            yield from GroupKFold(n_splits=n_splits).split(X, labels, groups=groups)
        else:
            yield from KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, labels)


def _cv_split(groups, n_folds, X, labels, seed=42):
    """Legacy wrapper — uses GroupKFold or KFold. For backward compat."""
    if groups is not None and len(np.unique(groups)) >= n_folds:
        yield from GroupKFold(n_splits=n_folds).split(X, labels, groups=groups)
    else:
        yield from KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, labels)


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

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
    mode: str = 'full',
) -> Tuple[float, float, int]:
    """Proper per-fold CV: create_hybrid on train, apply_feature_selection on val.

    Returns (mean_r2, std_r2, n_features).
    """
    sample_X = next(iter(base_reps.values()))
    scores = []
    n_feats = []

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, seed, mode):
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
    mode: str = 'full',
) -> Tuple[float, float, float, float, int]:
    """Like evaluate_cv but also returns train R².

    Returns (val_r2_mean, val_r2_std, train_r2_mean, train_r2_std, n_features).
    """
    sample_X = next(iter(base_reps.values()))
    val_scores = []
    train_scores = []
    n_feats = []

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, seed, mode):
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


def evaluate_single_rep_cv(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_folds: int = 5,
    mode: str = 'full',
) -> Tuple[float, float]:
    """Simple CV for a single representation matrix. Returns (mean_r2, std_r2)."""
    scores = []
    for train_idx, val_idx in cv_split(groups, n_folds, X, labels, mode=mode):
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

def get_model_factories(model_names: List[str]) -> Dict[str, callable]:
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


def get_model_family_variants() -> Dict[str, List[Tuple[str, callable, float]]]:
    """Return model families with complexity gradients.

    Each family: list of (variant_name, factory_fn, capacity_index).
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


# =============================================================================
# METRICS
# =============================================================================

def compute_intrinsic_dimensionality(X: np.ndarray) -> Dict[str, Any]:
    """Compute intrinsic dimensionality using PCA and MLE methods."""
    results = {'nominal_dim': X.shape[1]}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n, d = X_scaled.shape

    if d > 2000:
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
        n_components = min(n, d)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        for threshold in [0.90, 0.95, 0.99]:
            key = f'pca_{int(threshold * 100)}'
            intrinsic_dim = np.searchsorted(cumvar, threshold) + 1
            results[key] = int(min(intrinsic_dim, d))

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
    distances = distances[:, 1:]

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
    """Compute dataset complexity metrics (Lorena et al. 2018 inspired)."""
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    n, d = X_concat.shape
    y = labels

    metrics = {'n_samples': n, 'n_features': d}

    erank = compute_effective_rank(X_concat)
    metrics['erank'] = erank
    metrics['n_over_d'] = n / d
    metrics['n_over_erank'] = n / erank

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
    metrics['C1'] = float(np.max(abs_corrs))
    metrics['C2'] = float(np.mean(abs_corrs))

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

    nn = NearestNeighbors(n_neighbors=2, n_jobs=-1)
    nn.fit(X_for_model)
    _, indices = nn.kneighbors(X_for_model)
    y_nn = y[indices[:, 1]]
    metrics['S3'] = float(np.mean(np.abs(y - y_nn)) / y_range) if y_range > 0 else 0.0

    metrics['label_std'] = float(np.std(y))
    counts, _ = np.histogram(y, bins=20)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    metrics['target_entropy'] = float(-np.sum(probs * np.log(probs)))

    X_for_mle = X_scaled
    if d > 2000:
        from sklearn.decomposition import TruncatedSVD
        X_for_mle = TruncatedSVD(n_components=min(200, n - 1), random_state=42).fit_transform(X_scaled)
    try:
        metrics['mle_id'] = float(estimate_intrinsic_dim_mle(X_for_mle, k=10))
    except Exception:
        metrics['mle_id'] = None

    return metrics


# =============================================================================
# DATA LOADING (domain-aware)
# =============================================================================

_ANTIBODY_PREFIXES = ('flab_', 'abcov_', 'sabdab_', 'abagym_', 'abbind')


def _is_antibody_dataset(name: str) -> bool:
    return any(name.lower().startswith(p) for p in _ANTIBODY_PREFIXES)


def load_data(dataset_name: str, rep_preset: str = 'fast',
              delta: bool = False, augment: bool = False) -> dict:
    """Load dataset, generate representations, assign groups.

    Domain-aware: detects antibody dataset names and dispatches accordingly.
    Returns dict with keys: labels, reps, groups, n_groups, task_type,
    augmentations, plus domain-specific keys.
    """
    if _is_antibody_dataset(dataset_name):
        from find_best_hybrid_antibody import (
            load_dataset as load_ab_dataset,
            get_representations as get_ab_representations,
            get_augmentations as get_ab_augmentations,
            get_mutation_augmentations as get_ab_mutation_augmentations,
            assign_identity_groups,
        )

        heavy_seqs, light_seqs, labels, task_type, metadata = load_ab_dataset(dataset_name)
        wt_heavy = metadata.get('wildtype_heavy')
        wt_light = metadata.get('wildtype_light')
        mutations = metadata.get('mutations')

        reps = get_ab_representations(
            heavy_seqs, light_seqs, rep_preset,
            wt_heavy=wt_heavy, wt_light=wt_light,
            mutations=mutations, delta=delta,
        )

        groups, n_groups = assign_identity_groups(heavy_seqs, min_groups=3)

        augmentations = None
        if mutations is not None:
            augmentations = get_ab_mutation_augmentations(
                mutations, wt_heavy, wt_light, heavy_seqs)
            if not augmentations:
                augmentations = None
        if augment:
            seq_augs = get_ab_augmentations(heavy_seqs, light_seqs)
            if seq_augs:
                if augmentations:
                    augmentations.update(seq_augs)
                else:
                    augmentations = seq_augs

        print(f"  {dataset_name}: {len(heavy_seqs)} antibodies, {n_groups} identity groups, "
              f"{len(reps)} reps ({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

        return {
            'heavy_seqs': heavy_seqs,
            'light_seqs': light_seqs,
            'labels': labels,
            'reps': reps,
            'groups': groups,
            'n_scaffolds': n_groups,
            'n_groups': n_groups,
            'task_type': task_type,
            'domain': 'antibody',
            'augmentations': augmentations,
        }
    else:
        from find_best_hybrid_molecular import (
            load_dataset as load_molecular_dataset,
            get_representations as get_molecular_representations,
            assign_scaffold_groups,
        )

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
# SIMILARITY-SPECIFIC HELPERS
# =============================================================================

def get_hybrid_features_and_weights(
    reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    search_speed: str = 'fast',
    weight_method: str = 'lightgbm_gain',
    budget: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, SearchResult]:
    """Run find_best_hybrid and return hybrid features + flat importance weights."""
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

    X_hybrid = apply_feature_selection(reps, _fitted_to_feature_info(fitted))
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
            parts.append(np.ones(len(fitted.selected_indices[rep_name])))
    return np.concatenate(parts)


# =============================================================================
# IO HELPERS
# =============================================================================

def convert_numpy(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
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


def save_experiment_results(results: dict, output_dir: str, experiment_id: str,
                            dataset: str = None):
    """Save results JSON to output_dir/experiment_id/{dataset}.json."""
    out = Path(output_dir) / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset}.json" if dataset else "results.json"
    filepath = out / filename
    with open(filepath, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"  [saved {filepath}]")
    return filepath


def load_experiment_results(output_dir: str, experiment_id: str,
                            dataset: str = None) -> Optional[dict]:
    """Load results JSON from output_dir/experiment_id/{dataset}.json."""
    out = Path(output_dir) / experiment_id
    filename = f"{dataset}.json" if dataset else "results.json"
    filepath = out / filename
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def create_experiment_parser(description: str, has_delta: bool = False,
                             has_augment: bool = False) -> argparse.ArgumentParser:
    """Create a standard argument parser for experiment scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--mode', choices=['intermediary', 'full'], default='intermediary',
                        help='intermediary (1 split, fast) or full (5-fold CV)')
    parser.add_argument('--dataset', type=str, default='logd',
                        help='Dataset name (logd, caco2, herg, qm9, flab_*, etc.)')
    parser.add_argument('--reps', type=str, default='fast',
                        help='Representation preset (fast, fast+hf, all)')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory for results JSON')
    if has_delta:
        parser.add_argument('--delta', action='store_true',
                            help='Generate delta representations for antibody datasets')
    if has_augment:
        parser.add_argument('--augment', action='store_true',
                            help='Include augmented features for antibody datasets')
    return parser


def get_config(args) -> dict:
    """Return the appropriate config dict based on args.mode."""
    return INTERMEDIARY if args.mode == 'intermediary' else FULL
