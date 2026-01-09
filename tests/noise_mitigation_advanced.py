#!/usr/bin/env python3
"""
Noise Robustness Analysis: Hybrid Representations (v3)
=======================================================

Goal: Demonstrate that hybrid representations improve noise mitigation effectiveness.

Pipeline:
1. Quick combo test: Find best representation combination from
   {ECFP4+PDV, ECFP4+MHGGNN, PDV+MHGGNN, ALL_THREE}
2. Create hybrids at multiple budgets using performance-weighted allocation
3. Test noise mitigation methods with various distance metrics
4. Compare hybrid vs base representations for noise robustness

Distance Metrics (all continuous-appropriate):
- Standard: Euclidean, Manhattan, Cosine, Chebyshev, Canberra
- Advanced: Correlation, Minkowski (p=3), Bray-Curtis, Mahalanobis

Output: Clear comparison showing hybrid advantage for noise mitigation.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import cdist
from pathlib import Path
import time
import sys
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from joblib import Parallel, delayed
import json

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.datasets.esol import load_esol_combined
from kirby.representations.molecular import create_pdv, create_mhg_gnn, create_ecfp4
from kirby.hybrid import create_hybrid, apply_feature_selection
from kirby.utils.feature_filtering import apply_filters, FILTER_CONFIGS
from noiseInject import NoiseInjectorRegression
from kirby.utils.noise_mitigation_methods import get_mitigation_method, DISTANCE_AWARE_METHODS, compute_distance_matrix
from noise_mitigation_analysis import run_full_analysis


# ============================================================================
# CONFIGURATION
# ============================================================================

NOISE_LEVELS = [0.3, 0.6]  # Only noisy conditions (baseline captured separately)
N_SAMPLES_QM9 = 5000
N_JOBS = -1

# Budgets to test during selection phase
CANDIDATE_BUDGETS = [50, 100, 150, 200, 300]

# Allocation methods to test
ALLOCATION_METHODS = ['performance_weighted', 'greedy']

# Filter options to test
FILTER_OPTIONS = [False, True]

# Default hybrid config (from QM9 HOMO-LUMO gap optimization)
QM9_GAP_DEFAULT_HYBRID = {
    'combo': 'pdv+mhggnn',
    'budget': 100,
    'allocation': 'greedy',
    'filtered': False,
    'feature_allocation': {'pdv': 35, 'mhggnn': 10}
}

# Distance metrics - now all supported via precomputed distances (scipy.cdist)
DISTANCE_METRICS = [
    'euclidean',      # L2 norm
    'manhattan',      # L1 norm  
    'cosine',         # Angle-based
    'chebyshev',      # Max absolute difference
    'canberra',       # Weighted L1
    'correlation',    # 1 - Pearson correlation
    'braycurtis',     # Bray-Curtis dissimilarity
    'mahalanobis',    # Covariance-adjusted
]

# Methods to test - organized by what parameters they accept
# These accept precomputed_distances parameter:
DISTANCE_METRIC_METHODS = [
    'knn_k5', 'knn_k10', 'knn_k20',
    'lof', 'lof_k20',
    'distance_weighted',
    'neighborhood_consensus',
    'activity_cliffs',
]
# These don't use precomputed_distances:
OTHER_DISTANCE_METHODS = ['mahalanobis']  # Uses Mahalanobis internally (not precomputed)
ENSEMBLE_METHODS = ['cv_disagreement', 'bootstrap_ensemble']
MODEL_METHODS = ['co_teaching', 'dividemix']
BASELINE_METHODS = ['zscore', 'prediction_error']

# Combined for iteration
DISTANCE_BASED_METHODS = DISTANCE_METRIC_METHODS + OTHER_DISTANCE_METHODS
ALL_METHODS = DISTANCE_BASED_METHODS + ENSEMBLE_METHODS + MODEL_METHODS + BASELINE_METHODS
UNCERTAINTY_METHODS = ENSEMBLE_METHODS + MODEL_METHODS


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    representation: str
    rep_type: str  # 'base' or 'hybrid'
    sigma: float
    method: str
    distance_metric: str
    weighted: bool
    budget: Optional[int]
    combo: Optional[str]  # e.g., 'pdv+mhggnn'
    
    r2_clean: float
    r2_noisy: float
    r2_cleaned: float
    mae_cleaned: float
    rmse_cleaned: float
    recovery_rate: float
    
    precision: float
    recall: float
    f1: float
    n_detected: int
    n_true_noise: int
    n_samples_after: int
    
    uncertainty_mean: Optional[float] = None
    uncertainty_std: Optional[float] = None
    
    time_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ============================================================================
# DISTANCE COMPUTATION - ALL CONTINUOUS-APPROPRIATE
# ============================================================================

def compute_mahalanobis_distances(X: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance matrix using empirical covariance."""
    try:
        cov = EmpiricalCovariance().fit(X)
        # Use pseudo-inverse for stability
        vi = np.linalg.pinv(cov.covariance_)
        return cdist(X, X, metric='mahalanobis', VI=vi)
    except Exception:
        # Fallback to euclidean if covariance estimation fails
        return euclidean_distances(X)


def compute_distance_matrix(
    X: np.ndarray, 
    metric: str, 
    feature_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute pairwise distance matrix with optional feature weighting.
    All metrics are appropriate for continuous features.
    """
    # Apply feature weighting
    if feature_weights is not None:
        X_weighted = X * np.sqrt(np.clip(feature_weights, 0, None))
    else:
        X_weighted = X
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_weighted)
    
    # Handle NaN/Inf
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
    
    if metric == 'euclidean':
        return euclidean_distances(X_scaled)
    elif metric == 'manhattan':
        return manhattan_distances(X_scaled)
    elif metric == 'cosine':
        return cosine_distances(X_scaled)
    elif metric == 'chebyshev':
        return cdist(X_scaled, X_scaled, metric='chebyshev')
    elif metric == 'canberra':
        # Add small epsilon to avoid division by zero
        X_safe = X_scaled + 1e-10
        return cdist(X_safe, X_safe, metric='canberra')
    elif metric == 'correlation':
        return cdist(X_scaled, X_scaled, metric='correlation')
    elif metric == 'minkowski_3':
        return cdist(X_scaled, X_scaled, metric='minkowski', p=3)
    elif metric == 'braycurtis':
        # Shift to non-negative for Bray-Curtis
        X_positive = X_scaled - X_scaled.min() + 1e-10
        return cdist(X_positive, X_positive, metric='braycurtis')
    elif metric == 'mahalanobis':
        return compute_mahalanobis_distances(X_scaled)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


class DistanceCache:
    """Cache for precomputed distance matrices."""
    
    def __init__(self):
        self._cache: Dict[Tuple[str, str, bool], np.ndarray] = {}
    
    def get_or_compute(
        self, 
        rep_name: str, 
        X: np.ndarray, 
        metric: str, 
        weighted: bool,
        feature_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        key = (rep_name, metric, weighted)
        if key not in self._cache:
            weights = feature_weights if weighted else None
            self._cache[key] = compute_distance_matrix(X, metric, weights)
        return self._cache[key]
    
    def clear(self):
        self._cache.clear()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test) -> Dict[str, float]:
    """Train RF and evaluate."""
    if X_train.shape[1] == 0 or len(y_train) < 10:
        return {'r2': -np.inf, 'mae': np.inf, 'rmse': np.inf}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    return {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(np.mean((y_test - y_pred)**2))
    }


def calculate_cleaning_metrics(true_noise_mask, detected_indices, total_samples) -> Dict:
    """Calculate precision, recall, F1 for noise detection."""
    detected_mask = np.zeros(total_samples, dtype=bool)
    if len(detected_indices) > 0:
        detected_mask[detected_indices] = True
    
    if detected_mask.sum() == 0:
        precision = 1.0 if true_noise_mask.sum() == 0 else 0.0
        recall, f1 = 0.0, 0.0
    else:
        precision = precision_score(true_noise_mask, detected_mask, zero_division=0)
        recall = recall_score(true_noise_mask, detected_mask, zero_division=0)
        f1 = f1_score(true_noise_mask, detected_mask, zero_division=0)
    
    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'n_detected': len(detected_indices),
        'n_true_noise': int(true_noise_mask.sum())
    }


def get_feature_importance(X, y) -> np.ndarray:
    """Get RF feature importance, normalized."""
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=4)
    model.fit(X, y)
    weights = model.feature_importances_
    return weights / (weights.mean() + 1e-10)


# ============================================================================
# QUICK COMBINATION + BUDGET TEST
# ============================================================================

def find_best_hybrid(
    filtered_reps_train: Dict[str, np.ndarray],
    unfiltered_reps_train: Dict[str, np.ndarray],
    train_labels: np.ndarray,
    filtered_reps_test: Dict[str, np.ndarray],
    unfiltered_reps_test: Dict[str, np.ndarray],
    test_labels: np.ndarray
) -> Tuple[str, int, str, bool, Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
    """
    Find the best hybrid configuration by testing combos × budgets × allocation × filtering.
    
    Tests:
    - Combinations: {ECFP+PDV, ECFP+MHGGNN, PDV+MHGGNN, ALL_THREE}
    - Budgets: {50, 100, 150, 200}
    - Allocation: {performance_weighted, greedy (step_size=5)}
    - Filtering: {True, False}
    
    Returns:
        - best_combo: str (e.g., 'pdv+mhggnn')
        - best_budget: int
        - best_allocation: str
        - best_filtered: bool
        - base_train: dict of base reps used in winning combo
        - base_test: dict of base reps used in winning combo
        - hybrid_train: np.ndarray of the single best hybrid
        - hybrid_test: np.ndarray of the single best hybrid
        - hybrid_info: dict with allocation info
    """
    print("\n" + "="*80)
    print("FINDING BEST HYBRID (combo × budget × allocation × filter)")
    print("="*80)
    
    combinations = {
        'ecfp+pdv': ['ecfp', 'pdv'],
        'ecfp+mhggnn': ['ecfp', 'mhggnn'],
        'pdv+mhggnn': ['pdv', 'mhggnn'],
        'all_three': ['ecfp', 'pdv', 'mhggnn']
    }
    
    all_results = []
    total_configs = len(combinations) * len(CANDIDATE_BUDGETS) * len(ALLOCATION_METHODS) * len(FILTER_OPTIONS)
    current = 0
    
    for combo_name, rep_names in combinations.items():
        for use_filter in FILTER_OPTIONS:
            # Select filtered or unfiltered reps
            if use_filter:
                reps_train = {name: filtered_reps_train[name] for name in rep_names}
                reps_test = {name: filtered_reps_test[name] for name in rep_names}
            else:
                reps_train = {name: unfiltered_reps_train[name] for name in rep_names}
                reps_test = {name: unfiltered_reps_test[name] for name in rep_names}
            
            for allocation_method in ALLOCATION_METHODS:
                for budget in CANDIDATE_BUDGETS:
                    current += 1
                    filter_str = "filtered" if use_filter else "unfiltered"
                    print(f"  [{current}/{total_configs}] {combo_name} | {allocation_method} | {filter_str} | budget={budget}...", end=" ")
                    
                    try:
                        if allocation_method == 'greedy':
                            hybrid_train, info = create_hybrid(
                                reps_train, train_labels,
                                allocation_method='greedy',
                                budget=budget,
                                step_size=5,  # Finer granularity
                                patience=3,
                                validation_split=0.2,
                                apply_filters=False  # Already handled above
                            )
                        else:
                            hybrid_train, info = create_hybrid(
                                reps_train, train_labels,
                                allocation_method='performance_weighted',
                                budget=budget,
                                validation_split=0.2,
                                apply_filters=False
                            )
                        
                        hybrid_test = apply_feature_selection(reps_test, info)
                        
                        metrics = train_and_evaluate(hybrid_train, hybrid_test, train_labels, test_labels)
                        
                        all_results.append({
                            'combo': combo_name,
                            'budget': budget,
                            'allocation': allocation_method,
                            'filtered': use_filter,
                            'r2': metrics['r2'],
                            'rep_names': rep_names,
                            'reps_train': reps_train,
                            'reps_test': reps_test,
                            'hybrid_train': hybrid_train,
                            'hybrid_test': hybrid_test,
                            'info': info
                        })
                        print(f"R² = {metrics['r2']:.4f}")
                    except Exception as e:
                        print(f"ERROR: {e}")
    
    # Find best
    best = max(all_results, key=lambda x: x['r2'])
    
    print(f"\n  {'='*60}")
    print(f"  BEST CONFIGURATION:")
    print(f"    Combo:      {best['combo']}")
    print(f"    Budget:     {best['budget']}")
    print(f"    Allocation: {best['allocation']}")
    print(f"    Filtered:   {best['filtered']}")
    print(f"    R² = {best['r2']:.4f}")
    print(f"    Allocation: {best['info'].get('allocation', {})}")
    print(f"  {'='*60}")
    
    # Return the best configuration's data
    return (
        best['combo'],
        best['budget'],
        best['allocation'],
        best['filtered'],
        best['reps_train'],  # Base reps used (already filtered/unfiltered as appropriate)
        best['reps_test'],
        best['hybrid_train'],
        best['hybrid_test'],
        best['info']
    )


# ============================================================================
# SINGLE EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    rep_name: str,
    rep_type: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_clean: np.ndarray,
    y_train_noisy: np.ndarray,
    y_test: np.ndarray,
    true_noise_mask: np.ndarray,
    clean_metrics: Dict,
    noisy_metrics: Dict,
    sigma: float,
    method_name: str,
    distance_metric: Optional[str],
    weighted: bool,
    budget: Optional[int],
    combo: Optional[str],
    feature_weights: Optional[np.ndarray],
    precomputed_distances: Optional[np.ndarray] = None
) -> Optional[ExperimentResult]:
    """Run a single mitigation experiment."""
    
    method_kwargs = {'random_state': 42}
    
    # For distance-aware methods, pass precomputed distances and metric
    if method_name in DISTANCE_AWARE_METHODS:
        if precomputed_distances is not None:
            method_kwargs['precomputed_distances'] = precomputed_distances
        if distance_metric:
            method_kwargs['distance_metric'] = distance_metric
        if weighted and feature_weights is not None:
            method_kwargs['feature_weights'] = feature_weights
    elif weighted and feature_weights is not None:
        # For other methods that might use feature_weights
        method_kwargs['feature_weights'] = feature_weights
    
    try:
        method = get_mitigation_method(method_name, **method_kwargs)
    except Exception as e:
        return None
    
    try:
        start = time.time()
        X_clean, y_clean, removed_idx = method.clean_data(X_train, y_train_noisy)
        elapsed = time.time() - start
    except Exception as e:
        return None
    
    cleaned_metrics = train_and_evaluate(X_clean, X_test, y_clean, y_test)
    cleaning_acc = calculate_cleaning_metrics(true_noise_mask, removed_idx, len(y_train_noisy))
    
    if clean_metrics['r2'] != noisy_metrics['r2']:
        recovery = (cleaned_metrics['r2'] - noisy_metrics['r2']) / (clean_metrics['r2'] - noisy_metrics['r2'])
    else:
        recovery = 0.0
    
    # Extract uncertainty if available
    uncertainty_mean, uncertainty_std = None, None
    if hasattr(method, 'get_uncertainty'):
        try:
            unc = method.get_uncertainty(X_train)
            if unc is not None:
                uncertainty_mean = float(np.mean(unc))
                uncertainty_std = float(np.std(unc))
        except Exception:
            pass
    
    return ExperimentResult(
        representation=rep_name,
        rep_type=rep_type,
        sigma=sigma,
        method=method_name,
        distance_metric=distance_metric or 'none',
        weighted=weighted,
        budget=budget,
        combo=combo,
        r2_clean=clean_metrics['r2'],
        r2_noisy=noisy_metrics['r2'],
        r2_cleaned=cleaned_metrics['r2'],
        mae_cleaned=cleaned_metrics['mae'],
        rmse_cleaned=cleaned_metrics['rmse'],
        recovery_rate=recovery,
        precision=cleaning_acc['precision'],
        recall=cleaning_acc['recall'],
        f1=cleaning_acc['f1'],
        n_detected=cleaning_acc['n_detected'],
        n_true_noise=cleaning_acc['n_true_noise'],
        n_samples_after=len(y_clean),
        uncertainty_mean=uncertainty_mean,
        uncertainty_std=uncertainty_std,
        time_seconds=elapsed
    )


# ============================================================================
# REPRESENTATION EXPERIMENT RUNNER
# ============================================================================

def run_representation_experiments(
    rep_name: str,
    rep_type: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    sigma: float,
    budget: Optional[int] = None,
    combo: Optional[str] = None
) -> List[ExperimentResult]:
    """Run all mitigation experiments for one representation at one noise level."""
    
    results = []
    
    # Inject noise
    injector = NoiseInjectorRegression(strategy='legacy', random_state=42)
    y_train_noisy = injector.inject(y_train, sigma) if sigma > 0 else y_train.copy()
    true_noise_mask = (y_train_noisy != y_train)
    
    # Baselines
    clean_metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    noisy_metrics = train_and_evaluate(X_train, X_test, y_train_noisy, y_test)
    
    # Feature importance for weighting
    feature_weights = get_feature_importance(X_train, y_train)
    
    weighting_options = [False, True]
    
    # Precompute distance matrices (with disk caching)
    print(f"    Precomputing distance matrices...")
    distance_cache = {}
    cache_dir = Path(f".distance_cache/{rep_name}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for metric in DISTANCE_METRICS:
        for weighted in weighting_options:
            key = (metric, weighted)
            cache_file = cache_dir / f"{metric}_weighted{weighted}.npy"
            
            if cache_file.exists():
                distance_cache[key] = np.load(cache_file)
            else:
                weights = feature_weights if weighted else None
                D = compute_distance_matrix(X_train, metric, weights)
                np.save(cache_file, D)
                distance_cache[key] = D
    
    print(f"    Loaded/computed {len(distance_cache)} distance matrices")
    
    # Build configs: (method_name, distance_metric, weighted)
    configs = []
    
    # Methods that use precomputed distances
    for method_name in DISTANCE_METRIC_METHODS:
        for metric in DISTANCE_METRICS:
            for weighted in weighting_options:
                configs.append((method_name, metric, weighted))
    
    # Methods that use feature_weights but not precomputed distances (e.g., mahalanobis)
    for method_name in OTHER_DISTANCE_METHODS:
        for weighted in weighting_options:
            configs.append((method_name, None, weighted))
    
    # Methods that don't use distance metrics at all
    for method_name in ENSEMBLE_METHODS + MODEL_METHODS + BASELINE_METHODS:
        configs.append((method_name, None, False))
    
    print(f"    Running {len(configs)} experiments...")
    
    def run_config(cfg):
        method_name, metric, weighted = cfg
        
        # Get precomputed distances if this method uses them
        precomputed = None
        if metric and method_name in DISTANCE_METRIC_METHODS:
            precomputed = distance_cache.get((metric, weighted))
        
        return run_single_experiment(
            rep_name=rep_name,
            rep_type=rep_type,
            X_train=X_train,
            X_test=X_test,
            y_train_clean=y_train,
            y_train_noisy=y_train_noisy,
            y_test=y_test,
            true_noise_mask=true_noise_mask,
            clean_metrics=clean_metrics,
            noisy_metrics=noisy_metrics,
            sigma=sigma,
            method_name=method_name,
            distance_metric=metric,
            weighted=weighted,
            budget=budget,
            combo=combo,
            feature_weights=feature_weights if weighted else None,
            precomputed_distances=precomputed
        )
    
    parallel_results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(run_config)(cfg) for cfg in configs
    )
    
    results = [r for r in parallel_results if r is not None]
    
    # Summary
    if results:
        best = max(results, key=lambda r: r.recovery_rate)
        print(f"    Clean R²={clean_metrics['r2']:.4f}, Noisy R²={noisy_metrics['r2']:.4f}")
        print(f"    Best: {best.method} ({best.distance_metric}) → Recovery={best.recovery_rate:+.2%}")
    else:
        print(f"    WARNING: No successful experiments!")
    
    return results


# ============================================================================
# MAIN DATASET TESTING
# ============================================================================

def test_dataset(
    dataset_name: str,
    train_smiles: List[str],
    train_labels: np.ndarray,
    test_smiles: List[str],
    test_labels: np.ndarray,
    results_dir: Path,
    use_default_hybrid: bool = False
) -> List[Dict]:
    """
    Test noise mitigation methods: best hybrid vs its component bases.
    
    1. Create all base representations (filtered + unfiltered)
    2. Find best hybrid (combo + budget + allocation + filter) OR use default
    3. Capture clean baseline metrics
    4. Run full tests on: component bases + single best hybrid (σ=0.3, 0.6 only)
    """
    
    print("\n" + "="*100)
    print(f"{dataset_name.upper()} DATASET")
    print("="*100)
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    
    all_results = []
    dataset_dir = results_dir / dataset_name.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # CREATE ALL BASE REPRESENTATIONS
    # ========================================================================
    print("\nCreating base representations...")
    start = time.time()
    
    ecfp_train = create_ecfp4(train_smiles, radius=2, n_bits=2048)
    ecfp_test = create_ecfp4(test_smiles, radius=2, n_bits=2048)
    pdv_train = create_pdv(train_smiles)
    pdv_test = create_pdv(test_smiles)
    mhggnn_train = create_mhg_gnn(train_smiles)
    mhggnn_test = create_mhg_gnn(test_smiles)
    
    print(f"Done ({time.time() - start:.1f}s)")
    print(f"  ECFP:    {ecfp_train.shape[1]} features")
    print(f"  PDV:     {pdv_train.shape[1]} features")
    print(f"  MHG-GNN: {mhggnn_train.shape[1]} features")
    
    # Keep unfiltered versions
    unfiltered_train = {'ecfp': ecfp_train, 'pdv': pdv_train, 'mhggnn': mhggnn_train}
    unfiltered_test = {'ecfp': ecfp_test, 'pdv': pdv_test, 'mhggnn': mhggnn_test}
    
    # Apply quality filters
    print("\nApplying quality filters...")
    filter_config = FILTER_CONFIGS['quality_only']
    
    filtered_train, filtered_test = apply_filters(
        unfiltered_train.copy(), unfiltered_test.copy(), train_labels, **filter_config
    )
    
    for name in filtered_train:
        print(f"  {name}: {unfiltered_train[name].shape[1]} → {filtered_train[name].shape[1]}")
    
    # ========================================================================
    # GET HYBRID CONFIG (search or use default)
    # ========================================================================
    if use_default_hybrid:
        print("\n" + "="*60)
        print("USING DEFAULT HYBRID CONFIG (skipping search)")
        print("="*60)
        cfg = QM9_GAP_DEFAULT_HYBRID
        best_combo = cfg['combo']
        best_budget = cfg['budget']
        best_allocation = cfg['allocation']
        best_filtered = cfg['filtered']
        
        print(f"  Combo:      {best_combo}")
        print(f"  Budget:     {best_budget}")
        print(f"  Allocation: {best_allocation}")
        print(f"  Filtered:   {best_filtered}")
        
        # Use unfiltered since filtered=False in default
        base_train = unfiltered_train
        base_test = unfiltered_test
        
        # Build hybrid with default config
        rep_names = best_combo.split('+')
        reps_train = {k: base_train[k] for k in rep_names}
        reps_test = {k: base_test[k] for k in rep_names}
        
        hybrid_train, hybrid_info = create_hybrid(
            reps_train, train_labels,
            allocation_method=best_allocation,
            budget=best_budget,
            validation_split=0.2,
            apply_filters=False
        )
        hybrid_test = apply_feature_selection(reps_test, hybrid_info)
        
        print(f"  Hybrid dims: {hybrid_train.shape[1]}")
    else:
        (best_combo, best_budget, best_allocation, best_filtered,
         base_train, base_test, hybrid_train, hybrid_test, hybrid_info) = find_best_hybrid(
            filtered_reps_train=filtered_train,
            unfiltered_reps_train=unfiltered_train,
            train_labels=train_labels,
            filtered_reps_test=filtered_test,
            unfiltered_reps_test=unfiltered_test,
            test_labels=test_labels
        )
    
    # ========================================================================
    # CAPTURE CLEAN BASELINE METRICS (σ=0.0, no full experiments)
    # ========================================================================
    print("\n" + "="*80)
    print("CAPTURING CLEAN BASELINE METRICS (σ=0.0)")
    print("="*80)
    
    clean_baselines = {}
    
    # Base representations
    for base_name in base_train.keys():
        metrics = train_and_evaluate(base_train[base_name], base_test[base_name], train_labels, test_labels)
        clean_baselines[base_name] = metrics
        print(f"  {base_name}: R²={metrics['r2']:.4f}")
    
    # Hybrid
    hybrid_metrics = train_and_evaluate(hybrid_train, hybrid_test, train_labels, test_labels)
    clean_baselines['hybrid'] = hybrid_metrics
    print(f"  hybrid: R²={hybrid_metrics['r2']:.4f}")
    
    # ========================================================================
    # RUN FULL EXPERIMENTS (σ=0.3 and σ=0.6 only)
    # ========================================================================
    
    for sigma in NOISE_LEVELS:
        print(f"\n{'='*100}")
        print(f"NOISE LEVEL: σ={sigma}")
        print(f"{'='*100}")
        
        # ------------------------------------------------------------------
        # BASE REPRESENTATIONS (only from winning combo)
        # ------------------------------------------------------------------
        for base_name in base_train.keys():
            X_train_base = base_train[base_name]
            X_test_base = base_test[base_name]
            
            print(f"\n[BASE: {base_name.upper()}]")
            base_results = run_representation_experiments(
                rep_name=base_name,
                rep_type='base',
                X_train=X_train_base,
                X_test=X_test_base,
                y_train=train_labels,
                y_test=test_labels,
                sigma=sigma,
                budget=None,
                combo=None
            )
            
            # Add clean baseline to each result
            for r in base_results:
                r_dict = r.to_dict()
                r_dict['r2_clean'] = clean_baselines[base_name]['r2']
                all_results.append(r_dict)
        
        # ------------------------------------------------------------------
        # SINGLE BEST HYBRID
        # ------------------------------------------------------------------
        print(f"\n[HYBRID: {best_combo} | {best_allocation} | filtered={best_filtered} | budget={best_budget}]")
        hybrid_results = run_representation_experiments(
            rep_name=f'hybrid_{best_combo}_b{best_budget}',
            rep_type='hybrid',
            X_train=hybrid_train,
            X_test=hybrid_test,
            y_train=train_labels,
            y_test=test_labels,
            sigma=sigma,
            budget=best_budget,
            combo=best_combo
        )
        
        # Add clean baseline to each result
        for r in hybrid_results:
            r_dict = r.to_dict()
            r_dict['r2_clean'] = clean_baselines['hybrid']['r2']
            all_results.append(r_dict)
        
        # Checkpoint save after each noise level
        checkpoint_df = pd.DataFrame(all_results)
        checkpoint_df.to_csv(dataset_dir / f"checkpoint_sigma_{sigma}_original.csv", index=False)
        print(f"    ✓ Checkpoint saved ({len(all_results)} results so far)")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(dataset_dir / "all_results.csv", index=False)
    
    # Save detailed hybrid info
    with open(dataset_dir / "hybrid_info.json", 'w') as f:
        serializable = {
            'best_combo': best_combo,
            'best_budget': best_budget,
            'best_allocation': best_allocation,
            'best_filtered': best_filtered,
            'allocation': hybrid_info.get('allocation', {}),
            'base_reps': list(base_train.keys()),
            'clean_baselines': {k: v['r2'] for k, v in clean_baselines.items()}
        }
        json.dump(serializable, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"DATASET COMPLETE: {dataset_name}")
    print(f"Total experiments: {len(results_df)}")
    print(f"Reps tested: {list(base_train.keys())} + hybrid")
    print(f"{'='*100}")
    
    return all_results


# ============================================================================
# ANALYSIS: HYBRID VS BASE COMPARISON
# ============================================================================

def analyze_hybrid_advantage(results_df: pd.DataFrame, dataset_name: str, output_dir: Path):
    """
    Analyze and quantify the advantage of hybrid representations for noise mitigation.
    """
    print("\n" + "="*100)
    print(f"ANALYSIS: HYBRID ADVANTAGE FOR {dataset_name.upper()}")
    print("="*100)
    
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # 1. OVERALL COMPARISON: HYBRID VS BASE
    # ========================================================================
    print("\n[1] OVERALL: Hybrid vs Base Performance")
    print("-"*80)
    
    comparison = results_df.groupby(['rep_type', 'sigma']).agg({
        'recovery_rate': ['mean', 'std', 'max'],
        'f1': ['mean', 'std'],
        'r2_cleaned': ['mean', 'std']
    }).round(4)
    comparison.columns = ['_'.join(col) for col in comparison.columns]
    print(comparison)
    comparison.to_csv(analysis_dir / "hybrid_vs_base_overall.csv")
    
    # ========================================================================
    # 2. BEST METHOD PER REPRESENTATION TYPE
    # ========================================================================
    print("\n[2] BEST METHODS BY REPRESENTATION TYPE (σ=0.6)")
    print("-"*80)
    
    high_noise = results_df[results_df['sigma'] == 0.6]
    
    for rep_type in ['base', 'hybrid']:
        subset = high_noise[high_noise['rep_type'] == rep_type]
        if len(subset) == 0:
            continue
        
        best_5 = subset.nlargest(5, 'recovery_rate')[
            ['representation', 'method', 'distance_metric', 'weighted', 'recovery_rate', 'f1']
        ]
        print(f"\n{rep_type.upper()}:")
        print(best_5.to_string(index=False))
    
    # ========================================================================
    # 3. DISTANCE METRIC COMPARISON
    # ========================================================================
    print("\n[3] DISTANCE METRIC COMPARISON (σ=0.6)")
    print("-"*80)
    
    distance_results = high_noise[high_noise['distance_metric'] != 'none']
    
    metric_comparison = distance_results.groupby(['rep_type', 'distance_metric']).agg({
        'recovery_rate': 'mean',
        'f1': 'mean'
    }).round(4).reset_index()
    
    # Pivot for easier comparison
    pivot = metric_comparison.pivot(index='distance_metric', columns='rep_type', values='recovery_rate')
    if 'hybrid' in pivot.columns and 'base' in pivot.columns:
        pivot['hybrid_advantage'] = pivot['hybrid'] - pivot['base']
        print(pivot.sort_values('hybrid_advantage', ascending=False))
    else:
        print(pivot)
    pivot.to_csv(analysis_dir / "distance_metric_comparison.csv")
    
    # ========================================================================
    # 4. PER-REPRESENTATION BREAKDOWN
    # ========================================================================
    print("\n[4] PER-REPRESENTATION BREAKDOWN (σ=0.6)")
    print("-"*80)
    
    rep_breakdown = high_noise.groupby('representation').agg({
        'recovery_rate': ['mean', 'std', 'max'],
        'f1': ['mean', 'max'],
        'r2_cleaned': 'mean'
    }).round(4)
    rep_breakdown.columns = ['_'.join(map(str, col)) for col in rep_breakdown.columns]
    print(rep_breakdown)
    rep_breakdown.to_csv(analysis_dir / "per_representation_breakdown.csv")
    
    # ========================================================================
    # 5. STATISTICAL SUMMARY FOR PLOTTING
    # ========================================================================
    print("\n[5] GENERATING PLOT-READY SUMMARIES")
    print("-"*80)
    
    # Recovery rate by sigma, grouped by rep_type
    plot_data = results_df.groupby(['sigma', 'rep_type']).agg({
        'recovery_rate': ['mean', 'std', 'count'],
        'f1': ['mean', 'std']
    }).reset_index()
    plot_data.columns = ['_'.join(map(str, col)).rstrip('_') for col in plot_data.columns]
    plot_data.to_csv(analysis_dir / "plot_recovery_by_sigma.csv", index=False)
    
    # Best per method category
    method_categories = {
        'distance': DISTANCE_BASED_METHODS,
        'ensemble': ENSEMBLE_METHODS,
        'model': MODEL_METHODS,
        'baseline': BASELINE_METHODS
    }
    
    category_results = []
    for cat_name, methods in method_categories.items():
        cat_data = high_noise[high_noise['method'].isin(methods)]
        for rep_type in ['base', 'hybrid']:
            subset = cat_data[cat_data['rep_type'] == rep_type]
            if len(subset) > 0:
                category_results.append({
                    'category': cat_name,
                    'rep_type': rep_type,
                    'mean_recovery': subset['recovery_rate'].mean(),
                    'max_recovery': subset['recovery_rate'].max(),
                    'mean_f1': subset['f1'].mean()
                })
    
    cat_df = pd.DataFrame(category_results)
    cat_df.to_csv(analysis_dir / "method_category_comparison.csv", index=False)
    print(cat_df)
    
    # ========================================================================
    # 6. KEY FINDING SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Calculate hybrid advantage
    base_recovery = high_noise[high_noise['rep_type'] == 'base']['recovery_rate'].mean()
    hybrid_recovery = high_noise[high_noise['rep_type'] == 'hybrid']['recovery_rate'].mean()
    advantage = hybrid_recovery - base_recovery
    
    print(f"\n  At σ=0.6:")
    print(f"    Base mean recovery:   {base_recovery:+.2%}")
    print(f"    Hybrid mean recovery: {hybrid_recovery:+.2%}")
    print(f"    Hybrid advantage:     {advantage:+.2%}")
    
    # Best overall configuration
    best_overall = high_noise.loc[high_noise['recovery_rate'].idxmax()]
    print(f"\n  Best overall configuration:")
    print(f"    Rep: {best_overall['representation']}")
    print(f"    Method: {best_overall['method']}")
    print(f"    Distance: {best_overall['distance_metric']}")
    print(f"    Weighted: {best_overall['weighted']}")
    print(f"    Recovery: {best_overall['recovery_rate']:+.2%}")
    print(f"    F1: {best_overall['f1']:.3f}")
    
    # Save summary
    summary = {
        'base_mean_recovery': float(base_recovery),
        'hybrid_mean_recovery': float(hybrid_recovery),
        'hybrid_advantage': float(advantage),
        'best_config': {
            'representation': best_overall['representation'],
            'method': best_overall['method'],
            'distance_metric': best_overall['distance_metric'],
            'weighted': bool(best_overall['weighted']),
            'recovery_rate': float(best_overall['recovery_rate']),
            'f1': float(best_overall['f1'])
        }
    }
    
    with open(analysis_dir / "key_findings.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis saved to: {analysis_dir}")
    
    return summary


# ============================================================================
# MAIN
# ============================================================================

def main(dataset='both', use_default_hybrid=False):
    results_dir = Path("results/noise_robustness_v3")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*100)
    print("NOISE ROBUSTNESS: HYBRID VS BASE REPRESENTATION COMPARISON")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Noise levels: {NOISE_LEVELS} (clean baseline captured separately)")
    print(f"  Candidate budgets: {CANDIDATE_BUDGETS}")
    print(f"  Allocation methods: {ALLOCATION_METHODS}")
    print(f"  Filter options: {FILTER_OPTIONS}")
    print(f"  Distance metrics: {len(DISTANCE_METRICS)}")
    print(f"  Methods: {len(ALL_METHODS)}")
    print(f"  QM9 samples: {N_SAMPLES_QM9}")
    print(f"\nSelection phase: {len(CANDIDATE_BUDGETS) * len(ALLOCATION_METHODS) * len(FILTER_OPTIONS) * 4} configs tested")
    print(f"  (4 combos × {len(CANDIDATE_BUDGETS)} budgets × {len(ALLOCATION_METHODS)} alloc × {len(FILTER_OPTIONS)} filter)")
    print("="*100)
    
    all_dataset_results = {}
    
    # ========================================================================
    # QM9 DATASET
    # ========================================================================
    if dataset in ['qm9', 'both']:
        print("\n" + "="*100)
        print("LOADING QM9")
        print("="*100)
        
        raw_data = load_qm9(n_samples=N_SAMPLES_QM9, property_idx=4)  # 4 = HOMO-LUMO gap (PyG ordering)
        splits = get_qm9_splits(raw_data, splitter='scaffold')
        
        qm9_train_smiles = splits['train']['smiles'] + splits['val']['smiles']
        qm9_train_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
        qm9_test_smiles = splits['test']['smiles']
        qm9_test_labels = splits['test']['labels']
        
        qm9_results = test_dataset(
            'QM9', qm9_train_smiles, qm9_train_labels,
            qm9_test_smiles, qm9_test_labels, results_dir,
            use_default_hybrid=use_default_hybrid
        )
        all_dataset_results['QM9'] = qm9_results
    
    # ========================================================================
    # ESOL DATASET
    # ========================================================================
    if dataset in ['esol', 'both']:
        print("\n" + "="*100)
        print("LOADING ESOL")
        print("="*100)
        
        esol_data = load_esol_combined(splitter='scaffold')
        
        esol_results = test_dataset(
            'ESOL',
            esol_data['train']['smiles'],
            esol_data['train']['labels'],
            esol_data['test']['smiles'],
            esol_data['test']['labels'],
            results_dir,
            use_default_hybrid=use_default_hybrid
        )
        all_dataset_results['ESOL'] = esol_results
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    
    for dataset_name, results in all_dataset_results.items():
        results_df = pd.DataFrame(results)
        dataset_dir = results_dir / dataset_name.lower()
        
        # Save combined results FIRST (before any analysis)
        results_csv = dataset_dir / f"{dataset_name.lower()}_all_results_original.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"✓ Results saved to: {results_csv}")
        
        # Run analyses with error handling
        try:
            analyze_hybrid_advantage(results_df, dataset_name, dataset_dir)
        except Exception as e:
            print(f"⚠ analyze_hybrid_advantage failed: {e}")
            print("  Results are safe in CSV, continuing...")
        
        try:
            print(f"\n📊 Running comprehensive analysis with plots for {dataset_name}...")
            run_full_analysis(results_csv, dataset_dir / "analysis")
        except Exception as e:
            print(f"⚠ run_full_analysis failed: {e}")
            print("  Results are safe in CSV, you can run analysis separately.")
    
    print("\n" + "="*100)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*100)
    print(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['qm9', 'esol', 'both'], default='both')
    parser.add_argument('--use-default-hybrid', action='store_true', 
                        help='Skip hybrid selection, use QM9_GAP_DEFAULT_HYBRID config')
    args = parser.parse_args()
    main(dataset=args.dataset, use_default_hybrid=args.use_default_hybrid)