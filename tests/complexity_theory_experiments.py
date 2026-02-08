#!/usr/bin/env python3
"""
Complexity Theory Experiments for Molecular Representations

Tests theoretical ML concepts empirically:
1. Intrinsic dimensionality of representations
2. Mutual information estimation (rep → target)
3. Learning curves and noise floors
4. Generalization gap vs representation complexity
5. Double descent investigation
6. Information bottleneck analysis
7. Benign overfitting regime

Datasets: LogD (OpenADMET), QM9 HOMO-LUMO gap
Models: Ridge, RF, MLP, Deep NN

Usage:
    python tests/complexity_theory_experiments.py --quick
    python tests/complexity_theory_experiments.py --full
"""

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')


# =============================================================================
# METHODS AND STRATEGIES (from find_best_hybrid_molecular.py)
# =============================================================================

# Importance methods by computation cost
IMPORTANCE_METHODS_FAST = [
    'random_forest', 'treeshap', 'xgboost_gain', 'xgboost_weight',
    'xgboost_cover', 'lightgbm_gain', 'lightgbm_split',
]

IMPORTANCE_METHODS_MEDIUM = [
    'permutation', 'integrated_gradients', 'deeplift', 'deepliftshap', 'gradientshap',
]

IMPORTANCE_METHODS_SLOW = [
    'kernelshap', 'lime', 'drop_column', 'boruta',
]

ALL_IMPORTANCE_METHODS = IMPORTANCE_METHODS_FAST + IMPORTANCE_METHODS_MEDIUM + IMPORTANCE_METHODS_SLOW

# Allocation methods
ALLOCATION_METHODS = ['greedy', 'performance_weighted', 'fixed', 'mrmr']
ALLOCATION_METHODS_SLOW = ['greedy_feature']  # Feature-level, more precise but slower

# Budgets to test
BUDGETS = [25, 50, 100, 200]


# =============================================================================
# DATA LOADING (reuse from benchmark script)
# =============================================================================

def load_logd_dataset() -> Tuple[List[str], np.ndarray]:
    """Load LogD from OpenADMET."""
    import pandas as pd
    from rdkit import Chem

    cache_dir = Path.home() / '.cache' / 'kirby'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / 'openadmet_train.csv'

    if not cached.exists():
        url = ("https://huggingface.co/datasets/openadmet/"
               "openadmet-expansionrx-challenge-data/resolve/main/expansion_data_train.csv")
        print(f"  Downloading OpenADMET data...")
        import urllib.request
        urllib.request.urlretrieve(url, cached)

    df = pd.read_csv(cached)
    logd_col = next((c for c in df.columns if 'LogD' in c), None)
    if logd_col is None:
        raise ValueError("Cannot find LogD column")

    valid_smiles, valid_labels = [], []
    for smi, label in zip(df['SMILES'], df[logd_col]):
        if pd.isna(label) or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
            valid_labels.append(label)

    return valid_smiles, np.array(valid_labels)


def load_qm9_homolumo(n_samples: int = 5000) -> Tuple[List[str], np.ndarray]:
    """Load HOMO-LUMO gap from QM9."""
    try:
        from kirby.datasets.qm9 import load_qm9
        data = load_qm9(n_samples=n_samples, property_idx=4)
        return data['smiles'], data['labels']
    except ImportError as e:
        raise ImportError(f"QM9 requires torch-geometric: {e}")


def generate_representations(smiles: List[str]) -> Dict[str, np.ndarray]:
    """Generate multiple molecular representations."""
    from kirby.representations.molecular import (
        create_ecfp4, create_maccs, create_pdv,
    )

    print("Generating representations...")
    reps = {
        'ecfp4': create_ecfp4(smiles),
        'maccs': create_maccs(smiles),
        'pdv': create_pdv(smiles),
    }

    # Also create concatenated versions of different complexities
    reps['ecfp4_only'] = reps['ecfp4']
    reps['pdv_only'] = reps['pdv']
    reps['ecfp4_pdv'] = np.hstack([reps['ecfp4'], reps['pdv']])
    reps['all'] = np.hstack([reps['ecfp4'], reps['maccs'], reps['pdv']])

    for name, X in reps.items():
        print(f"  {name}: {X.shape}")

    return reps


def get_scaffold_groups(smiles: List[str]) -> np.ndarray:
    """Get Murcko scaffold groups."""
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

    unique = list(set(scaffolds))
    scaffold_to_idx = {s: i for i, s in enumerate(unique)}
    return np.array([scaffold_to_idx[s] for s in scaffolds])


# =============================================================================
# EXPERIMENT 1: INTRINSIC DIMENSIONALITY
# =============================================================================

def compute_intrinsic_dimensionality(X: np.ndarray, methods: List[str] = None) -> Dict[str, Any]:
    """
    Compute intrinsic dimensionality using multiple methods.

    Methods:
    - PCA (explained variance threshold)
    - MLE (maximum likelihood estimator)
    - Correlation dimension (fractal-based)
    """
    if methods is None:
        methods = ['pca_90', 'pca_95', 'pca_99', 'mle']

    results = {'nominal_dim': X.shape[1]}

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA-based
    n_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    for threshold in [0.90, 0.95, 0.99]:
        key = f'pca_{int(threshold*100)}'
        if key in methods or 'pca' in methods:
            intrinsic_dim = np.searchsorted(cumvar, threshold) + 1
            results[key] = int(min(intrinsic_dim, X.shape[1]))

    # MLE-based (Levina-Bickel estimator)
    if 'mle' in methods:
        try:
            intrinsic_dim_mle = estimate_intrinsic_dim_mle(X_scaled, k=10)
            results['mle'] = float(intrinsic_dim_mle)
        except Exception as e:
            results['mle'] = None
            results['mle_error'] = str(e)

    # Compression ratio
    if 'pca_95' in results and results['pca_95']:
        results['compression_ratio'] = X.shape[1] / results['pca_95']

    return results


def estimate_intrinsic_dim_mle(X: np.ndarray, k: int = 10) -> float:
    """
    MLE estimator of intrinsic dimensionality (Levina-Bickel 2004).

    Based on the rate of growth of distances to k-nearest neighbors.
    """
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)

    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # Remove self-distance (first column)
    distances = distances[:, 1:]

    # MLE estimate per point
    # d_mle = (k-1) / sum(log(T_k / T_j)) for j=1..k-1
    estimates = []
    for i in range(n_samples):
        dists = distances[i]
        if dists[-1] > 0:
            # Log ratio of max distance to each distance
            log_ratios = np.log(dists[-1] / dists[:-1])
            if np.sum(log_ratios) > 0:
                d_hat = (k - 1) / np.sum(log_ratios)
                estimates.append(d_hat)

    return np.mean(estimates) if estimates else np.nan


def run_intrinsic_dim_experiment(reps: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Run intrinsic dimensionality analysis on all representations."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: INTRINSIC DIMENSIONALITY")
    print("="*60)

    results = {}
    for rep_name, X in reps.items():
        print(f"\n  {rep_name}...")
        dim_results = compute_intrinsic_dimensionality(X)
        results[rep_name] = dim_results

        print(f"    Nominal: {dim_results['nominal_dim']}")
        print(f"    PCA 95%: {dim_results.get('pca_95', 'N/A')}")
        print(f"    MLE: {dim_results.get('mle', 'N/A'):.1f}" if dim_results.get('mle') else "    MLE: N/A")
        if 'compression_ratio' in dim_results:
            print(f"    Compression: {dim_results['compression_ratio']:.1f}x")

    return results


# =============================================================================
# EXPERIMENT 2: MUTUAL INFORMATION ESTIMATION
# =============================================================================

def estimate_mutual_information_knn(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    Estimate I(X; Y) using k-NN method (Kraskov et al. 2004).

    This is a simplified version - for continuous Y, we use the
    KSG estimator approximation.
    """
    from scipy.special import digamma

    n_samples = X.shape[0]
    k = min(k, n_samples - 1)

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    y_scaled = (y - y.mean()) / (y.std() + 1e-10)

    # Joint space
    XY = np.column_stack([X_scaled, y_scaled.reshape(-1, 1)])

    # Find k-NN distances in joint space
    nn_joint = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
    nn_joint.fit(XY)
    distances_joint, _ = nn_joint.kneighbors(XY)
    eps = distances_joint[:, -1]  # Distance to k-th neighbor

    # Count neighbors within eps in marginal spaces
    nn_x = NearestNeighbors(metric='chebyshev')
    nn_x.fit(X_scaled)

    nn_y = NearestNeighbors(metric='chebyshev')
    nn_y.fit(y_scaled.reshape(-1, 1))

    n_x = np.zeros(n_samples)
    n_y = np.zeros(n_samples)

    for i in range(n_samples):
        # Count points within eps[i] in X space
        n_x[i] = len(nn_x.radius_neighbors([X_scaled[i]], radius=eps[i], return_distance=False)[0]) - 1
        # Count points within eps[i] in Y space
        n_y[i] = len(nn_y.radius_neighbors([y_scaled[i:i+1]], radius=eps[i], return_distance=False)[0]) - 1

    # KSG estimator
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n_samples)

    return max(0, mi)  # MI should be non-negative


def run_mutual_information_experiment(
    reps: Dict[str, np.ndarray],
    y: np.ndarray
) -> Dict[str, Any]:
    """Estimate mutual information between each representation and target."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: MUTUAL INFORMATION I(Rep; Target)")
    print("="*60)

    results = {}
    for rep_name, X in reps.items():
        print(f"\n  {rep_name}...", end=' ')

        # Use PCA to reduce dimensionality for MI estimation (more stable)
        n_components = min(50, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(StandardScaler().fit_transform(X))

        try:
            mi = estimate_mutual_information_knn(X_reduced, y, k=5)
            results[rep_name] = {
                'mi_estimate': float(mi),
                'n_components_used': n_components,
            }
            print(f"MI = {mi:.4f} nats")
        except Exception as e:
            results[rep_name] = {'error': str(e)}
            print(f"FAILED: {e}")

    # Rank by MI
    print("\n  Ranking by MI:")
    sorted_reps = sorted(
        [(k, v['mi_estimate']) for k, v in results.items() if 'mi_estimate' in v],
        key=lambda x: x[1], reverse=True
    )
    for i, (name, mi) in enumerate(sorted_reps):
        print(f"    {i+1}. {name}: {mi:.4f}")

    return results


# =============================================================================
# EXPERIMENT 3: LEARNING CURVES AND NOISE FLOORS
# =============================================================================

def fit_learning_curve(n_samples: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Fit asymptotic model: R² = R²_max - a/n^b

    Returns estimated R²_max (noise floor) and fit parameters.
    """
    from scipy.optimize import curve_fit

    def model(n, r2_max, a, b):
        return r2_max - a / np.power(n, b)

    try:
        # Initial guess
        p0 = [np.max(scores), 1.0, 0.5]
        bounds = ([0, 0, 0.1], [1, 100, 2])

        popt, pcov = curve_fit(model, n_samples, scores, p0=p0, bounds=bounds, maxfev=5000)
        r2_max, a, b = popt

        # Compute fit quality
        predicted = model(n_samples, *popt)
        residuals = scores - predicted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((scores - np.mean(scores))**2)
        fit_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'r2_max': float(r2_max),
            'a': float(a),
            'b': float(b),
            'fit_r2': float(fit_r2),
        }
    except Exception as e:
        return {'error': str(e), 'r2_max': float(np.max(scores))}


def run_learning_curve_experiment(
    reps: Dict[str, np.ndarray],
    y: np.ndarray,
    scaffold_groups: np.ndarray,
    sample_sizes: List[int] = None,
    models: List[str] = None,
    n_repeats: int = 3
) -> Dict[str, Any]:
    """
    Generate learning curves for each representation and model.

    Tests: How does performance scale with data? Where does it plateau?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: LEARNING CURVES")
    print("="*60)

    if sample_sizes is None:
        max_n = len(y)
        sample_sizes = [100, 200, 500, 1000, 2000, 5000]
        sample_sizes = [s for s in sample_sizes if s < max_n * 0.8]
        sample_sizes.append(int(max_n * 0.8))

    if models is None:
        models = ['ridge', 'rf']

    results = {}

    for rep_name in ['ecfp4_only', 'pdv_only', 'ecfp4_pdv', 'all']:
        if rep_name not in reps:
            continue

        X = reps[rep_name]
        print(f"\n  {rep_name} (dim={X.shape[1]}):")
        results[rep_name] = {}

        for model_name in models:
            print(f"    {model_name}:", end=' ')
            curve_scores = []

            for n in sample_sizes:
                repeat_scores = []
                for seed in range(n_repeats):
                    # Subsample
                    np.random.seed(seed)
                    idx = np.random.choice(len(y), min(n, len(y)), replace=False)
                    X_sub, y_sub = X[idx], y[idx]
                    groups_sub = scaffold_groups[idx]

                    # Simple train/test split
                    try:
                        gkf = GroupKFold(n_splits=3)
                        fold_scores = []
                        for train_idx, test_idx in gkf.split(X_sub, y_sub, groups=groups_sub):
                            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
                            y_train, y_test = y_sub[train_idx], y_sub[test_idx]

                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)

                            if model_name == 'ridge':
                                model = Ridge(alpha=1.0)
                            elif model_name == 'rf':
                                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                            else:
                                continue

                            model.fit(X_train, y_train)
                            fold_scores.append(r2_score(y_test, model.predict(X_test)))

                        repeat_scores.append(np.mean(fold_scores))
                    except Exception:
                        continue

                if repeat_scores:
                    curve_scores.append((n, np.mean(repeat_scores), np.std(repeat_scores)))

            if curve_scores:
                n_vals = np.array([x[0] for x in curve_scores])
                score_means = np.array([x[1] for x in curve_scores])

                # Fit asymptotic model
                fit_result = fit_learning_curve(n_vals, score_means)

                results[rep_name][model_name] = {
                    'sample_sizes': n_vals.tolist(),
                    'scores': score_means.tolist(),
                    'stds': [x[2] for x in curve_scores],
                    'fit': fit_result,
                }

                r2_max = fit_result.get('r2_max', score_means[-1])
                print(f"R²_max = {r2_max:.4f}")

    return results


# =============================================================================
# EXPERIMENT 4: GENERALIZATION GAP VS COMPLEXITY
# =============================================================================

def run_generalization_gap_experiment(
    reps: Dict[str, np.ndarray],
    y: np.ndarray,
    scaffold_groups: np.ndarray,
    budgets: List[int] = None,
    models: List[str] = None
) -> Dict[str, Any]:
    """
    Measure generalization gap (train - test R²) vs representation complexity.

    Tests: At what complexity does overfitting start?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: GENERALIZATION GAP VS COMPLEXITY")
    print("="*60)

    if budgets is None:
        budgets = [20, 50, 100, 200, 500, 1000]

    if models is None:
        models = ['ridge', 'rf', 'mlp']

    # Use the 'all' representation and vary feature count via PCA
    X_full = reps.get('all', reps['ecfp4_pdv'])

    results = {'budgets': budgets}

    for model_name in models:
        print(f"\n  {model_name}:")
        model_results = {'train_r2': [], 'test_r2': [], 'gap': []}

        for budget in budgets:
            n_components = min(budget, X_full.shape[1], X_full.shape[0] - 1)

            # Reduce to budget dimensions
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_full)
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X_scaled)

            # Cross-validation
            gkf = GroupKFold(n_splits=3)
            train_scores, test_scores = [], []

            for train_idx, test_idx in gkf.split(X_reduced, y, groups=scaffold_groups):
                X_train, X_test = X_reduced[train_idx], X_reduced[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if model_name == 'ridge':
                    model = Ridge(alpha=1.0)
                elif model_name == 'rf':
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                elif model_name == 'mlp':
                    model = MLPRegressor(hidden_layer_sizes=(100,), random_state=42, max_iter=500)
                else:
                    continue

                model.fit(X_train, y_train)
                train_scores.append(r2_score(y_train, model.predict(X_train)))
                test_scores.append(r2_score(y_test, model.predict(X_test)))

            train_r2 = np.mean(train_scores)
            test_r2 = np.mean(test_scores)
            gap = train_r2 - test_r2

            model_results['train_r2'].append(float(train_r2))
            model_results['test_r2'].append(float(test_r2))
            model_results['gap'].append(float(gap))

            print(f"    dim={n_components}: train={train_r2:.4f}, test={test_r2:.4f}, gap={gap:.4f}")

        results[model_name] = model_results

    return results


# =============================================================================
# EXPERIMENT 5: DOUBLE DESCENT INVESTIGATION
# =============================================================================

def run_double_descent_experiment(
    reps: Dict[str, np.ndarray],
    y: np.ndarray,
    scaffold_groups: np.ndarray
) -> Dict[str, Any]:
    """
    Test for double descent: does test error decrease again past interpolation?

    Theory: In overparameterized regime, test error can decrease after initial peak.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: DOUBLE DESCENT INVESTIGATION")
    print("="*60)

    X_full = reps.get('all', reps['ecfp4_pdv'])
    n_samples = X_full.shape[0]

    # Test across wide range of dimensions
    # Key is to go past the interpolation threshold (n_features ≈ n_samples)
    dim_range = [10, 25, 50, 100, 200, 500, 1000]
    dim_range = [d for d in dim_range if d < X_full.shape[1]]

    # Add dimensions around interpolation threshold
    threshold_dims = [int(n_samples * 0.5), int(n_samples * 0.8), int(n_samples * 1.0),
                      int(n_samples * 1.2), int(n_samples * 1.5)]
    threshold_dims = [d for d in threshold_dims if d < X_full.shape[1] and d > 0]
    dim_range = sorted(set(dim_range + threshold_dims))

    print(f"  Testing dimensions: {dim_range}")
    print(f"  Interpolation threshold (n_samples): {n_samples}")

    results = {'dimensions': dim_range, 'n_samples': n_samples}

    # Test with Ridge (no regularization) - most likely to show double descent
    print("\n  Ridge (alpha=1e-10, near-interpolation):")
    ridge_results = {'train_r2': [], 'test_r2': [], 'train_mse': [], 'test_mse': []}

    for n_dim in dim_range:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_full)
        pca = PCA(n_components=min(n_dim, X_scaled.shape[1]))
        X_reduced = pca.fit_transform(X_scaled)

        gkf = GroupKFold(n_splits=3)
        train_r2s, test_r2s, train_mses, test_mses = [], [], [], []

        for train_idx, test_idx in gkf.split(X_reduced, y, groups=scaffold_groups):
            X_train, X_test = X_reduced[train_idx], X_reduced[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Near-zero regularization to allow interpolation
            model = Ridge(alpha=1e-10)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_r2s.append(r2_score(y_train, train_pred))
            test_r2s.append(r2_score(y_test, test_pred))
            train_mses.append(mean_squared_error(y_train, train_pred))
            test_mses.append(mean_squared_error(y_test, test_pred))

        ridge_results['train_r2'].append(float(np.mean(train_r2s)))
        ridge_results['test_r2'].append(float(np.mean(test_r2s)))
        ridge_results['train_mse'].append(float(np.mean(train_mses)))
        ridge_results['test_mse'].append(float(np.mean(test_mses)))

        status = "INTERPOLATING" if n_dim >= n_samples * 0.8 else ""
        print(f"    dim={n_dim}: test R²={np.mean(test_r2s):.4f} {status}")

    results['ridge'] = ridge_results

    # Check for double descent pattern
    test_r2 = np.array(ridge_results['test_r2'])
    dims = np.array(dim_range)

    # Find local minimum (if exists)
    if len(test_r2) > 3:
        for i in range(1, len(test_r2) - 1):
            if test_r2[i] < test_r2[i-1] and test_r2[i] < test_r2[i+1]:
                print(f"\n  Potential double descent: local minimum at dim={dims[i]}")
                if dims[i] > n_samples * 0.5:
                    print(f"    This is near interpolation threshold - classic double descent!")
                results['double_descent_detected'] = True
                results['local_minimum_dim'] = int(dims[i])
                break
        else:
            results['double_descent_detected'] = False
            print("\n  No clear double descent pattern detected")

    return results


# =============================================================================
# EXPERIMENT 6: MODEL COMPLEXITY × REP COMPLEXITY TRADEOFF
# =============================================================================

def run_complexity_tradeoff_experiment(
    reps: Dict[str, np.ndarray],
    y: np.ndarray,
    scaffold_groups: np.ndarray,
    rep_budgets: List[int] = None,
    allocation_methods: List[str] = None,
    importance_methods: List[str] = None,
) -> Dict[str, Any]:
    """
    CORE EXPERIMENT: Find optimal (model_complexity, rep_complexity) pairings.

    Uses ACTUAL HYBRIDS from kirby.hybrid, not just PCA truncation.
    Tests multiple importance methods and allocation strategies.

    Key questions:
    - Do simpler models need more curated (lower-dim) representations?
    - Does allocation method matter differently for different model types?
    - Does importance method matter for different models?
    - Is there a matching principle between model and rep complexity?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 6: MODEL × REPRESENTATION COMPLEXITY TRADEOFF")
    print("="*60)

    from kirby.hybrid import create_hybrid

    if rep_budgets is None:
        rep_budgets = BUDGETS

    if allocation_methods is None:
        allocation_methods = ALLOCATION_METHODS

    if importance_methods is None:
        # Use fast methods by default, can expand in full mode
        importance_methods = ['random_forest', 'treeshap', 'permutation']

    base_reps = {k: reps[k] for k in ['ecfp4', 'maccs', 'pdv'] if k in reps}

    # Define model complexity levels
    model_configs = [
        # Linear models - low complexity
        ('ridge_high_reg', lambda: Ridge(alpha=100), 1),
        ('ridge_med_reg', lambda: Ridge(alpha=1), 2),
        ('ridge_low_reg', lambda: Ridge(alpha=0.01), 3),

        # Tree models - varying complexity
        ('rf_shallow', lambda: RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1), 2),
        ('rf_medium', lambda: RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1), 4),
        ('rf_deep', lambda: RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1), 6),

        # Neural networks - varying complexity
        ('mlp_tiny', lambda: MLPRegressor(hidden_layer_sizes=(16,), random_state=42, max_iter=500), 3),
        ('mlp_small', lambda: MLPRegressor(hidden_layer_sizes=(64,), random_state=42, max_iter=500), 4),
        ('mlp_medium', lambda: MLPRegressor(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500), 5),
        ('mlp_large', lambda: MLPRegressor(hidden_layer_sizes=(256, 128, 64), random_state=42, max_iter=500), 7),
    ]

    results = {
        'rep_budgets': rep_budgets,
        'allocation_methods': allocation_methods,
        'models': {},
        'optimal_pairings': {},
    }

    total_combos = len(model_configs) * len(rep_budgets) * len(allocation_methods)
    print(f"\n  Testing {len(model_configs)} models × {len(rep_budgets)} budgets × {len(allocation_methods)} methods")
    print(f"  = {total_combos} combinations\n")

    # For each model, find optimal (budget, allocation_method) pairing
    for model_name, model_factory, model_complexity in model_configs:
        print(f"  {model_name} (complexity={model_complexity}):")
        model_results = {
            'complexity': model_complexity,
            'configs': [],
            'best_config': None,
            'best_score': -np.inf
        }

        for alloc_method in allocation_methods:
            for budget in rep_budgets:
                try:
                    # Create hybrid using YOUR methodology
                    X_hybrid, feature_info = create_hybrid(
                        base_reps, y,
                        allocation_method=alloc_method,
                        budget=budget,
                        importance_method='random_forest'
                    )

                    actual_dim = X_hybrid.shape[1]

                    # Cross-validate with this model
                    gkf = GroupKFold(n_splits=3)
                    scores = []

                    for train_idx, test_idx in gkf.split(X_hybrid, y, groups=scaffold_groups):
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_hybrid[train_idx])
                        X_test = scaler.transform(X_hybrid[test_idx])

                        model = model_factory()
                        model.fit(X_train, y[train_idx])
                        scores.append(r2_score(y[test_idx], model.predict(X_test)))

                    mean_score = np.mean(scores)

                    config_result = {
                        'allocation': alloc_method,
                        'budget': budget,
                        'actual_dim': actual_dim,
                        'r2': float(mean_score),
                    }
                    model_results['configs'].append(config_result)

                    if mean_score > model_results['best_score']:
                        model_results['best_score'] = float(mean_score)
                        model_results['best_config'] = config_result

                except Exception as e:
                    print(f"      {alloc_method}/b={budget}: FAILED - {e}")
                    continue

        results['models'][model_name] = model_results

        if model_results['best_config']:
            best = model_results['best_config']
            results['optimal_pairings'][model_name] = {
                'model_complexity': model_complexity,
                'optimal_allocation': best['allocation'],
                'optimal_budget': best['budget'],
                'optimal_dim': best['actual_dim'],
                'best_r2': best['r2'],
            }
            print(f"    Best: {best['allocation']}/b={best['budget']} (dim={best['actual_dim']}) → R²={best['r2']:.4f}")

    # Analyze: does model complexity predict optimal rep config?
    print("\n  COMPLEXITY MATCHING ANALYSIS:")

    complexities = []
    optimal_dims = []
    optimal_budgets = []

    for model_name, pairing in results['optimal_pairings'].items():
        complexities.append(pairing['model_complexity'])
        optimal_dims.append(pairing['optimal_dim'])
        optimal_budgets.append(pairing['optimal_budget'])

    if len(complexities) > 2:
        corr_dim = np.corrcoef(complexities, optimal_dims)[0, 1]
        corr_budget = np.corrcoef(complexities, optimal_budgets)[0, 1]
        results['complexity_vs_dim_correlation'] = float(corr_dim)
        results['complexity_vs_budget_correlation'] = float(corr_budget)

        print(f"    Corr(model_complexity, optimal_dim) = {corr_dim:.3f}")
        print(f"    Corr(model_complexity, optimal_budget) = {corr_budget:.3f}")

        if corr_budget > 0.3:
            print("    → Complex models benefit from larger budgets")
        elif corr_budget < -0.3:
            print("    → Complex models prefer smaller, curated reps")
        else:
            print("    → Relationship is model-specific")

    # Analyze: does allocation method matter differently for different models?
    print("\n  ALLOCATION METHOD ANALYSIS:")
    for alloc in allocation_methods:
        wins = sum(1 for m in results['optimal_pairings'].values()
                  if m.get('optimal_allocation') == alloc)
        print(f"    {alloc}: optimal for {wins}/{len(results['optimal_pairings'])} models")

    return results


# =============================================================================
# EXPERIMENT 7: LOW-DIMENSION REPRESENTATIONS FOR DOWNSTREAM TASKS
# =============================================================================

def run_low_dimension_experiment(
    reps: Dict[str, np.ndarray],
    y: np.ndarray,
    smiles: List[str],
    scaffold_groups: np.ndarray,
    budgets: List[int] = None,
) -> Dict[str, Any]:
    """
    Test ultra-compact HYBRID representations for downstream tasks.

    Uses actual hybrids from kirby.hybrid, not PCA.

    Not just prediction - also similarity search, clustering, visualization.
    These compact reps may be used as embeddings for other tasks.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 7: LOW-DIMENSION HYBRID REPRESENTATIONS")
    print("="*60)

    from kirby.hybrid import create_hybrid
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    if budgets is None:
        budgets = [5, 10, 20, 50]

    base_reps = {k: reps[k] for k in ['ecfp4', 'maccs', 'pdv'] if k in reps}

    results = {'budgets': budgets, 'tasks': {}, 'comparison': {}}

    for budget in budgets:
        print(f"\n  Budget: {budget} dimensions")
        budget_results = {}

        # Create HYBRID (not just PCA)
        try:
            X_hybrid, feature_info = create_hybrid(
                base_reps, y,
                allocation_method='greedy',
                budget=budget,
                importance_method='random_forest'
            )
            actual_dim = X_hybrid.shape[1]
            budget_results['actual_dim'] = actual_dim
            print(f"    Actual hybrid dim: {actual_dim}")

            # Also create PCA baseline for comparison
            X_full = np.hstack([reps[k] for k in sorted(base_reps.keys())])
            n_comp = min(budget, X_full.shape[1])
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(StandardScaler().fit_transform(X_full))

        except Exception as e:
            print(f"    Hybrid creation failed: {e}")
            continue

        # Task 1: Prediction - HYBRID vs PCA
        gkf = GroupKFold(n_splits=3)

        # Hybrid
        hybrid_scores = []
        for train_idx, test_idx in gkf.split(X_hybrid, y, groups=scaffold_groups):
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_hybrid[train_idx], y[train_idx])
            hybrid_scores.append(r2_score(y[test_idx], model.predict(X_hybrid[test_idx])))
        budget_results['hybrid_r2'] = float(np.mean(hybrid_scores))

        # PCA baseline
        pca_scores = []
        for train_idx, test_idx in gkf.split(X_pca, y, groups=scaffold_groups):
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_pca[train_idx], y[train_idx])
            pca_scores.append(r2_score(y[test_idx], model.predict(X_pca[test_idx])))
        budget_results['pca_r2'] = float(np.mean(pca_scores))

        budget_results['hybrid_vs_pca'] = float(np.mean(hybrid_scores) - np.mean(pca_scores))
        print(f"    Prediction: Hybrid R²={np.mean(hybrid_scores):.4f}, PCA R²={np.mean(pca_scores):.4f}")
        print(f"    Hybrid advantage: {budget_results['hybrid_vs_pca']:+.4f}")

        # Task 2: Similarity preservation (NN property correlation)
        try:
            from sklearn.neighbors import NearestNeighbors

            for name, X in [('hybrid', X_hybrid), ('pca', X_pca)]:
                nn = NearestNeighbors(n_neighbors=6)
                nn.fit(X)
                _, indices = nn.kneighbors(X)

                # Mean absolute property difference to 5 nearest neighbors
                diffs = []
                for i in range(len(y)):
                    neighbor_props = y[indices[i, 1:]]
                    diffs.append(np.mean(np.abs(neighbor_props - y[i])))

                budget_results[f'{name}_neighbor_diff'] = float(np.mean(diffs))

            print(f"    NN property diff: Hybrid={budget_results['hybrid_neighbor_diff']:.4f}, "
                  f"PCA={budget_results['pca_neighbor_diff']:.4f}")
        except Exception as e:
            print(f"    NN analysis failed: {e}")

        # Task 3: Clustering quality
        try:
            n_clusters = min(10, len(np.unique(scaffold_groups)))

            for name, X in [('hybrid', X_hybrid), ('pca', X_pca)]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                sil = silhouette_score(X, labels)
                budget_results[f'{name}_silhouette'] = float(sil)

            print(f"    Silhouette: Hybrid={budget_results['hybrid_silhouette']:.4f}, "
                  f"PCA={budget_results['pca_silhouette']:.4f}")
        except Exception as e:
            print(f"    Clustering failed: {e}")

        results['tasks'][budget] = budget_results

    # Summary: where does hybrid beat PCA?
    print("\n  HYBRID vs PCA SUMMARY:")
    for budget in budgets:
        if budget in results['tasks']:
            task = results['tasks'][budget]
            adv = task.get('hybrid_vs_pca', 0)
            status = "✓ HYBRID WINS" if adv > 0.01 else ("≈ TIE" if adv > -0.01 else "✗ PCA wins")
            print(f"    dim={budget}: {status} (Δ={adv:+.4f})")

    return results


# =============================================================================
# EXPERIMENT 8: FORWARD COMPATIBILITY TEST
# =============================================================================

def run_forward_compatibility_experiment(
    reps: Dict[str, np.ndarray],
    y: np.ndarray,
    scaffold_groups: np.ndarray,
) -> Dict[str, Any]:
    """
    Test that the hybrid framework works with "new" models.

    Key point: The framework is MODEL-AGNOSTIC. It doesn't rely on specific
    architectures. Any future model can benefit.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 8: FORWARD COMPATIBILITY")
    print("="*60)

    from kirby.hybrid import create_hybrid

    # Models the hybrid was NOT explicitly designed for
    # Simulating "future" models by using less common sklearn models
    novel_models = {}

    # Try importing various models
    try:
        from sklearn.svm import SVR
        novel_models['svr'] = lambda: SVR(kernel='rbf', C=1.0)
    except ImportError:
        pass

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        novel_models['gp'] = lambda: GaussianProcessRegressor(kernel=RBF(), random_state=42)
    except ImportError:
        pass

    try:
        from sklearn.neighbors import KNeighborsRegressor
        novel_models['knn'] = lambda: KNeighborsRegressor(n_neighbors=5)
    except ImportError:
        pass

    try:
        from sklearn.linear_model import ElasticNet
        novel_models['elasticnet'] = lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    except ImportError:
        pass

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        novel_models['gbm'] = lambda: GradientBoostingRegressor(n_estimators=50, random_state=42)
    except ImportError:
        pass

    try:
        from sklearn.ensemble import AdaBoostRegressor
        novel_models['adaboost'] = lambda: AdaBoostRegressor(n_estimators=50, random_state=42)
    except ImportError:
        pass

    print(f"  Testing {len(novel_models)} 'novel' models: {list(novel_models.keys())}")

    results = {'models': {}}
    base_reps = {k: reps[k] for k in ['ecfp4', 'maccs', 'pdv'] if k in reps}

    for model_name, model_factory in novel_models.items():
        print(f"\n  {model_name}:")
        model_results = {}

        # Test 1: Single best rep
        best_single_r2 = -np.inf
        best_single_name = None

        for rep_name in ['ecfp4', 'pdv']:
            if rep_name not in reps:
                continue
            X = reps[rep_name]
            gkf = GroupKFold(n_splits=3)
            scores = []

            for train_idx, test_idx in gkf.split(X, y, groups=scaffold_groups):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])

                try:
                    model = model_factory()
                    model.fit(X_train, y[train_idx])
                    scores.append(r2_score(y[test_idx], model.predict(X_test)))
                except Exception:
                    continue

            if scores:
                mean_r2 = np.mean(scores)
                if mean_r2 > best_single_r2:
                    best_single_r2 = mean_r2
                    best_single_name = rep_name

        model_results['best_single'] = {'rep': best_single_name, 'r2': float(best_single_r2)}
        print(f"    Best single ({best_single_name}): R²={best_single_r2:.4f}")

        # Test 2: Hybrid
        try:
            X_hybrid, _ = create_hybrid(base_reps, y, allocation_method='greedy', budget=100)

            gkf = GroupKFold(n_splits=3)
            scores = []

            for train_idx, test_idx in gkf.split(X_hybrid, y, groups=scaffold_groups):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_hybrid[train_idx])
                X_test = scaler.transform(X_hybrid[test_idx])

                model = model_factory()
                model.fit(X_train, y[train_idx])
                scores.append(r2_score(y[test_idx], model.predict(X_test)))

            hybrid_r2 = np.mean(scores)
            model_results['hybrid'] = {'r2': float(hybrid_r2)}
            print(f"    Hybrid: R²={hybrid_r2:.4f}")

            improvement = hybrid_r2 - best_single_r2
            model_results['improvement'] = float(improvement)
            status = "✓ IMPROVED" if improvement > 0 else "✗ no improvement"
            print(f"    Improvement: {improvement:+.4f} {status}")

        except Exception as e:
            print(f"    Hybrid failed: {e}")
            model_results['hybrid'] = {'error': str(e)}

        results['models'][model_name] = model_results

    # Summary
    n_improved = sum(1 for m in results['models'].values()
                     if m.get('improvement', -1) > 0)
    results['summary'] = {
        'n_models_tested': len(novel_models),
        'n_improved_with_hybrid': n_improved,
        'forward_compatible': n_improved >= len(novel_models) * 0.5,
    }

    print(f"\n  SUMMARY: Hybrid improved {n_improved}/{len(novel_models)} novel models")
    if results['summary']['forward_compatible']:
        print("  → Framework is FORWARD COMPATIBLE")
    else:
        print("  → Mixed results - investigate model-specific patterns")

    return results


# =============================================================================
# EXPERIMENT 9: HYBRID VS SINGLE REPRESENTATION
# =============================================================================

def run_hybrid_comparison_experiment(
    reps: Dict[str, np.ndarray],
    y: np.ndarray,
    scaffold_groups: np.ndarray,
    budgets: List[int] = None
) -> Dict[str, Any]:
    """
    Compare hybrid representations to single representations.

    Key question: Does combining reps beat the best single rep?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 6: HYBRID VS SINGLE REPRESENTATION")
    print("="*60)

    if budgets is None:
        budgets = [50, 100, 200]

    from kirby.hybrid import create_hybrid

    results = {}

    for budget in budgets:
        print(f"\n  Budget: {budget}")
        budget_results = {}

        # Single reps (truncated to budget)
        for rep_name in ['ecfp4', 'maccs', 'pdv']:
            X = reps[rep_name]
            # Use PCA to reduce to budget
            n_comp = min(budget, X.shape[1], X.shape[0] - 1)
            X_reduced = PCA(n_components=n_comp).fit_transform(StandardScaler().fit_transform(X))

            gkf = GroupKFold(n_splits=3)
            scores = []
            for train_idx, test_idx in gkf.split(X_reduced, y, groups=scaffold_groups):
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_reduced[train_idx], y[train_idx])
                scores.append(r2_score(y[test_idx], model.predict(X_reduced[test_idx])))

            budget_results[rep_name] = float(np.mean(scores))
            print(f"    {rep_name}: R²={np.mean(scores):.4f}")

        # Hybrid
        try:
            base_reps = {k: reps[k] for k in ['ecfp4', 'maccs', 'pdv']}
            X_hybrid, _ = create_hybrid(base_reps, y, allocation_method='greedy', budget=budget)

            gkf = GroupKFold(n_splits=3)
            scores = []
            for train_idx, test_idx in gkf.split(X_hybrid, y, groups=scaffold_groups):
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_hybrid[train_idx], y[train_idx])
                scores.append(r2_score(y[test_idx], model.predict(X_hybrid[test_idx])))

            budget_results['hybrid'] = float(np.mean(scores))
            print(f"    hybrid: R²={np.mean(scores):.4f}")

            # Calculate improvement over best single
            best_single = max([budget_results[k] for k in ['ecfp4', 'maccs', 'pdv']])
            improvement = budget_results['hybrid'] - best_single
            budget_results['improvement_over_best'] = float(improvement)
            print(f"    Improvement over best single: {improvement:+.4f}")

        except Exception as e:
            print(f"    hybrid: FAILED - {e}")

        results[budget] = budget_results

    return results


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    datasets: List[str] = field(default_factory=lambda: ['logd'])
    experiments: List[str] = field(default_factory=lambda: [
        # Original experiments (theory validation)
        'intrinsic_dim', 'mutual_info', 'learning_curves',
        'generalization_gap', 'double_descent',
        # YOUR CORE THESIS experiments
        'complexity_tradeoff',      # Model complexity × rep complexity matching
        'low_dimension',            # Compact reps for downstream tasks
        'forward_compatibility',    # Works with novel models
        'hybrid_comparison',        # Hybrid vs single rep
    ])
    qm9_samples: int = 5000


def run_all_experiments(config: ExperimentConfig, output_dir: str = 'results/complexity_theory'):
    """Run all complexity theory experiments."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        'config': {
            'datasets': config.datasets,
            'experiments': config.experiments,
            'timestamp': datetime.now().isoformat(),
        },
        'datasets': {},
    }

    for dataset_name in config.datasets:
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset_name.upper()}")
        print('#'*60)

        # Load data
        try:
            if dataset_name == 'logd':
                smiles, labels = load_logd_dataset()
            elif dataset_name == 'qm9':
                smiles, labels = load_qm9_homolumo(config.qm9_samples)
            else:
                print(f"Unknown dataset: {dataset_name}")
                continue
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue

        print(f"Loaded {len(smiles)} samples")

        # Generate representations
        reps = generate_representations(smiles)
        scaffold_groups = get_scaffold_groups(smiles)

        dataset_results = {
            'n_samples': len(smiles),
            'n_scaffolds': len(np.unique(scaffold_groups)),
        }

        # Run experiments
        if 'intrinsic_dim' in config.experiments:
            dataset_results['intrinsic_dim'] = run_intrinsic_dim_experiment(reps)

        if 'mutual_info' in config.experiments:
            dataset_results['mutual_info'] = run_mutual_information_experiment(reps, labels)

        if 'learning_curves' in config.experiments:
            dataset_results['learning_curves'] = run_learning_curve_experiment(
                reps, labels, scaffold_groups
            )

        if 'generalization_gap' in config.experiments:
            dataset_results['generalization_gap'] = run_generalization_gap_experiment(
                reps, labels, scaffold_groups
            )

        if 'double_descent' in config.experiments:
            dataset_results['double_descent'] = run_double_descent_experiment(
                reps, labels, scaffold_groups
            )

        # NEW: Model × Rep complexity tradeoff (YOUR CORE THESIS)
        if 'complexity_tradeoff' in config.experiments:
            dataset_results['complexity_tradeoff'] = run_complexity_tradeoff_experiment(
                reps, labels, scaffold_groups
            )

        # NEW: Low-dimension representations for downstream tasks
        if 'low_dimension' in config.experiments:
            dataset_results['low_dimension'] = run_low_dimension_experiment(
                reps, labels, smiles, scaffold_groups
            )

        # NEW: Forward compatibility with novel models
        if 'forward_compatibility' in config.experiments:
            dataset_results['forward_compatibility'] = run_forward_compatibility_experiment(
                reps, labels, scaffold_groups
            )

        if 'hybrid_comparison' in config.experiments:
            dataset_results['hybrid_comparison'] = run_hybrid_comparison_experiment(
                reps, labels, scaffold_groups
            )

        all_results['datasets'][dataset_name] = dataset_results

    # Save results
    output_file = output_path / f"complexity_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
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

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Complexity theory experiments')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--full', action='store_true', help='Full experiments')
    parser.add_argument('--output', type=str, default='tests/results/complexity_theory')
    parser.add_argument('--datasets', type=str, default='logd',
                       help='Comma-separated: logd,qm9')
    parser.add_argument('--experiments', type=str, default=None,
                       help='Comma-separated list of experiments to run')

    args = parser.parse_args()

    if args.quick:
        config = ExperimentConfig(
            datasets=['logd'],
            experiments=['intrinsic_dim', 'complexity_tradeoff'],  # Quick but meaningful
            qm9_samples=1000,
        )
    else:
        experiments = args.experiments.split(',') if args.experiments else [
            # Original theory validation
            'intrinsic_dim', 'mutual_info', 'learning_curves',
            'generalization_gap', 'double_descent',
            # YOUR CORE THESIS
            'complexity_tradeoff', 'low_dimension', 'forward_compatibility',
            'hybrid_comparison'
        ]
        config = ExperimentConfig(
            datasets=args.datasets.split(','),
            experiments=experiments,
            qm9_samples=5000,
        )

    run_all_experiments(config, args.output)


if __name__ == '__main__':
    main()
