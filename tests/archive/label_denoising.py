#!/usr/bin/env python3
"""
Label Denoising Experiment
===========================

Question: Which molecular representation provides the best neighbor structure
for estimating and correcting label noise?

Approach:
1. Estimate noise magnitude per sample (continuous, not binary)
2. Use estimates to either:
   - Correct labels (smooth toward neighbors proportionally to noise score)
   - Weight samples (downweight high-noise samples during training)
3. Evaluate all methods with fixed ECFP4 + RF pipeline

Representations compared: ECFP4, PDV, MHG-GNN, Hybrid (multiple configurations)
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.datasets.esol import load_esol_combined
from kirby.representations.molecular import create_pdv, create_mhg_gnn, create_ecfp4
from kirby.hybrid import create_hybrid, apply_feature_selection
from noiseInject import NoiseInjectorRegression

# Import noise estimation module
from kirby.utils.noise_estimation import (
    get_noise_estimator,
    smooth_by_scores,
    weights_from_scores,
    ESTIMATOR_REGISTRY
)


# ============================================================================
# CONFIGURATION
# ============================================================================

NOISE_LEVELS = [0.3, 0.6]
N_SAMPLES_QM9 = 5000
N_JOBS = -1

# Base representations (used for hybrids and standalone)
BASE_REPS = ['ecfp4', 'pdv', 'mhggnn']

# Hybrid configurations to explore
HYBRID_CONFIGS = [
    # (name, rep_list, budget, allocation_method)
    ('hybrid_pdv_mhg_greedy100', ['pdv', 'mhggnn'], 100, 'greedy'),
    ('hybrid_pdv_mhg_greedy50', ['pdv', 'mhggnn'], 50, 'greedy'),
    ('hybrid_pdv_mhg_fixed50', ['pdv', 'mhggnn'], 50, 'fixed'),
    ('hybrid_all_greedy100', ['ecfp4', 'pdv', 'mhggnn'], 100, 'greedy'),
    ('hybrid_all_greedy150', ['ecfp4', 'pdv', 'mhggnn'], 150, 'greedy'),
    ('hybrid_ecfp_pdv_greedy100', ['ecfp4', 'pdv'], 100, 'greedy'),
    ('hybrid_ecfp_mhg_greedy100', ['ecfp4', 'mhggnn'], 100, 'greedy'),
]

# Noise estimation methods
# Group 1: Neighbor-based (use molecular representation)
NEIGHBOR_ESTIMATORS = ['neighbor_residual', 'local_variance_ratio', 'activity_cliff_aware']

# Group 2: Model-based SOTA (representation-agnostic, use ECFP4)
SOTA_ESTIMATORS = ['gaussian_process', 'heteroscedastic_nn', 'conformal', 
                   'deep_ensemble', 'bayesian_rf', 'quantile_regression', 'ngboost']

# Smoothing approaches
SMOOTHING_METHODS = ['proportional', 'soft_threshold', 'confidence']
WEIGHTING_METHODS = ['inverse', 'exponential']

# Strength values (alpha for smoothing, strength for weighting)
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7]

# Neighbor parameters (for neighbor-based methods)
K_NEIGHBORS = 10
DISTANCE_METRICS = ['cosine', 'euclidean', 'manhattan']


# ============================================================================
# RESULT STRUCTURE
# ============================================================================

@dataclass
class Result:
    dataset: str
    sigma: float
    smoothing_rep: str
    estimator: str
    distance_metric: str
    approach: str  # 'smooth' or 'weight'
    method: str    # specific smoothing/weighting method
    alpha: float
    
    r2_clean: float
    r2_noisy: float
    r2_denoised: float
    r2_recovery: float
    
    mae_clean: float
    mae_noisy: float
    mae_denoised: float
    mae_recovery: float
    
    # Noise score statistics
    noise_score_mean: float
    noise_score_std: float
    noise_score_max: float
    
    # What changed
    label_change_mean: float  # for smoothing
    weight_min: float         # for weighting
    weight_std: float
    
    time_seconds: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class HybridSearchResult:
    """Result from hybrid configuration search."""
    dataset: str
    hybrid_name: str
    reps_combined: str  # comma-separated
    budget: int
    allocation_method: str
    n_features: int
    allocation: str  # JSON-like string of allocation dict
    
    # Evaluation as neighbor structure (best across all estimators/methods)
    best_r2_recovery_sigma03: float
    best_config_sigma03: str
    best_r2_recovery_sigma06: float
    best_config_sigma06: str
    
    creation_time_seconds: float
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# EVALUATION
# ============================================================================

def train_evaluate_rf(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Train RF on ECFP, return metrics."""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=N_JOBS)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_test)
    return {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }


def run_experiment_with_scores(
    noise_scores: np.ndarray,
    neighbor_means: np.ndarray,
    X_ecfp_train: np.ndarray,
    X_ecfp_test: np.ndarray,
    y_noisy: np.ndarray,
    y_test: np.ndarray,
    approach: str,
    method: str,
    alpha: float,
    baselines: Dict,
    dataset: str,
    sigma: float,
    rep_name: str,
    est_name: str,
    dist_metric: str = 'cosine'
) -> Result:
    """Run experiment given pre-computed noise scores."""
    
    start = time.time()
    
    if approach == 'smooth':
        y_modified = smooth_by_scores(
            y_noisy, noise_scores, neighbor_means,
            alpha=alpha, method=method
        )
        label_diff = np.abs(y_modified - y_noisy)
        label_change_mean = float(label_diff.mean())
        weights = None
        weight_min = 1.0
        weight_std = 0.0
    else:  # weight
        y_modified = y_noisy
        weights = weights_from_scores(noise_scores, strength=alpha, method=method)
        label_change_mean = 0.0
        weight_min = float(weights.min())
        weight_std = float(weights.std())
    
    elapsed = time.time() - start
    
    metrics = train_evaluate_rf(X_ecfp_train, X_ecfp_test, y_modified, y_test, sample_weight=weights)
    
    r2_recovery = (metrics['r2'] - baselines['r2_noisy']) / (baselines['r2_clean'] - baselines['r2_noisy'] + 1e-10)
    mae_recovery = (baselines['mae_noisy'] - metrics['mae']) / (baselines['mae_noisy'] - baselines['mae_clean'] + 1e-10)
    
    return Result(
        dataset=dataset,
        sigma=sigma,
        smoothing_rep=rep_name,
        estimator=est_name,
        distance_metric=dist_metric,
        approach=approach,
        method=method,
        alpha=alpha,
        r2_clean=baselines['r2_clean'],
        r2_noisy=baselines['r2_noisy'],
        r2_denoised=metrics['r2'],
        r2_recovery=r2_recovery,
        mae_clean=baselines['mae_clean'],
        mae_noisy=baselines['mae_noisy'],
        mae_denoised=metrics['mae'],
        mae_recovery=mae_recovery,
        noise_score_mean=float(noise_scores.mean()),
        noise_score_std=float(noise_scores.std()),
        noise_score_max=float(noise_scores.max()),
        label_change_mean=label_change_mean,
        weight_min=weight_min,
        weight_std=weight_std,
        time_seconds=elapsed
    )


def print_result(result: Result):
    """Print a single result line."""
    rec = result.r2_recovery
    marker = "✓" if rec > 0 else "✗"
    print(f"    {result.estimator}/{result.method} α={result.alpha}: {rec:+.1%} {marker}")


# ============================================================================
# REPRESENTATION CREATION
# ============================================================================

def create_base_representations(
    train_smiles: List[str], 
    test_smiles: List[str]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create base representations (without hybrids)."""
    
    print("  Creating ECFP4...")
    ecfp_train = create_ecfp4(train_smiles, radius=2, n_bits=2048)
    ecfp_test = create_ecfp4(test_smiles, radius=2, n_bits=2048)
    
    print("  Creating PDV...")
    pdv_train = create_pdv(train_smiles)
    pdv_test = create_pdv(test_smiles)
    
    print("  Creating MHG-GNN...")
    mhggnn_train = create_mhg_gnn(train_smiles)
    mhggnn_test = create_mhg_gnn(test_smiles)
    
    return {
        'ecfp4': (ecfp_train, ecfp_test),
        'pdv': (pdv_train, pdv_test),
        'mhggnn': (mhggnn_train, mhggnn_test),
    }


def create_hybrid_representation(
    base_reps_train: Dict[str, np.ndarray],
    base_reps_test: Dict[str, np.ndarray],
    train_labels: np.ndarray,
    rep_names: List[str],
    budget: int,
    allocation_method: str
) -> Tuple[np.ndarray, np.ndarray, Dict, float]:
    """
    Create a single hybrid representation.
    
    Returns:
        (hybrid_train, hybrid_test, info, creation_time)
    
    Raises:
        ValueError: If hybrid creation fails
    """
    reps_train = {name: base_reps_train[name] for name in rep_names}
    reps_test = {name: base_reps_test[name] for name in rep_names}
    
    t0 = time.time()
    
    if allocation_method == 'fixed':
        n_per_rep = budget // len(rep_names)
        hybrid_train, hybrid_info = create_hybrid(
            reps_train, train_labels,
            allocation_method='fixed',
            n_per_rep=n_per_rep,
            validation_split=0.2
        )
    else:
        hybrid_train, hybrid_info = create_hybrid(
            reps_train, train_labels,
            allocation_method=allocation_method,
            budget=budget,
            step_size=10,
            patience=3,
            validation_split=0.2
        )
    
    hybrid_test = apply_feature_selection(reps_test, hybrid_info)
    creation_time = time.time() - t0
    
    return hybrid_train, hybrid_test, hybrid_info, creation_time


def create_all_hybrids(
    base_reps: Dict[str, Tuple[np.ndarray, np.ndarray]],
    train_labels: np.ndarray,
    configs: List[Tuple]
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict, float]]:
    """
    Create all hybrid configurations.
    
    Returns:
        Dict of {name: (train, test, info, creation_time)}
    """
    base_train = {name: reps[0] for name, reps in base_reps.items()}
    base_test = {name: reps[1] for name, reps in base_reps.items()}
    
    hybrids = {}
    
    for name, rep_list, budget, alloc_method in configs:
        print(f"  Creating {name} ({'+'.join(rep_list)}, budget={budget}, {alloc_method})...")
        
        try:
            hybrid_train, hybrid_test, info, creation_time = create_hybrid_representation(
                base_train, base_test, train_labels,
                rep_list, budget, alloc_method
            )
            hybrids[name] = (hybrid_train, hybrid_test, info, creation_time)
            
            alloc = info.get('allocation', {})
            total = sum(v for k, v in alloc.items() if isinstance(v, (int, float)))
            print(f"    → {hybrid_train.shape[1]} features, allocation: {alloc} ({creation_time:.1f}s)")
            
        except (ValueError, KeyError) as e:
            print(f"    → FAILED: {e}")
            continue
    
    return hybrids


# ============================================================================
# HYBRID SEARCH
# ============================================================================

def find_best_hybrids(
    hybrids: Dict[str, Tuple[np.ndarray, np.ndarray, Dict, float]],
    ecfp_train: np.ndarray,
    ecfp_test: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    dataset_name: str,
    results_dir: Path
) -> pd.DataFrame:
    """
    Find the best hybrid configurations for noise estimation.
    
    Args:
        hybrids: Dict of {name: (train, test, info, creation_time)} from create_all_hybrids
    
    Tests each hybrid as a neighbor structure and reports which works best.
    """
    print(f"\n{'='*60}")
    print("HYBRID CONFIGURATION SEARCH")
    print(f"{'='*60}")
    
    hybrid_results = []
    
    if not hybrids:
        print("  No hybrid representations to evaluate")
        return pd.DataFrame()
    
    # Test each noise level
    results_by_hybrid = {name: {'sigma_0.3': [], 'sigma_0.6': []} for name in hybrids}
    
    for sigma in NOISE_LEVELS:
        print(f"\n  σ = {sigma}")
        
        # Inject noise
        injector = NoiseInjectorRegression(strategy='legacy', random_state=42)
        y_noisy = injector.inject(train_labels, sigma)
        
        # Baselines
        clean_metrics = train_evaluate_rf(ecfp_train, ecfp_test, train_labels, test_labels)
        noisy_metrics = train_evaluate_rf(ecfp_train, ecfp_test, y_noisy, test_labels)
        
        baselines = {
            'r2_clean': clean_metrics['r2'],
            'r2_noisy': noisy_metrics['r2'],
            'mae_clean': clean_metrics['mae'],
            'mae_noisy': noisy_metrics['mae'],
        }
        
        # Test each hybrid
        for hybrid_name, (hybrid_train, hybrid_test, hybrid_info, creation_time) in hybrids.items():
            print(f"\n    [{hybrid_name}]")
            
            best_recovery = -np.inf
            best_config = None
            
            # Test with neighbor-based estimators
            for dist_metric in ['cosine']:  # Just cosine for speed
                for est_name in NEIGHBOR_ESTIMATORS:
                    estimator = get_noise_estimator(
                        est_name, k=K_NEIGHBORS, metric=dist_metric, random_state=42
                    )
                    
                    noise_scores = estimator.estimate(hybrid_train, y_noisy)
                    neighbor_means = getattr(estimator, 'neighbor_means_', None)
                    
                    if neighbor_means is None:
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric=dist_metric)
                        nn.fit(hybrid_train)
                        _, indices = nn.kneighbors(hybrid_train)
                        neighbor_means = y_noisy[indices[:, 1:]].mean(axis=1)
                    
                    # Quick test with best alpha values
                    for method in ['proportional', 'confidence']:
                        for alpha in [0.3, 0.5]:
                            result = run_experiment_with_scores(
                                noise_scores, neighbor_means,
                                ecfp_train, ecfp_test,
                                y_noisy, test_labels,
                                'smooth', method, alpha,
                                baselines, dataset_name, sigma,
                                hybrid_name, est_name, dist_metric
                            )
                            
                            if result.r2_recovery > best_recovery:
                                best_recovery = result.r2_recovery
                                best_config = f"{est_name}/{method}/α={alpha}"
            
            sigma_key = f"sigma_{sigma}"
            results_by_hybrid[hybrid_name][sigma_key] = (best_recovery, best_config)
            
            marker = "✓" if best_recovery > 0 else "✗"
            print(f"      Best: {best_recovery:+.1%} ({best_config}) {marker}")
    
    # Compile results
    for hybrid_name, (hybrid_train, hybrid_test, hybrid_info, creation_time) in hybrids.items():
        # Extract config from name
        config = next((c for c in HYBRID_CONFIGS if c[0] == hybrid_name), None)
        if config is None:
            continue
        
        _, rep_list, budget, alloc_method = config
        alloc = hybrid_info.get('allocation', {})
        
        r03, cfg03 = results_by_hybrid[hybrid_name].get('sigma_0.3', (-999, 'n/a'))
        r06, cfg06 = results_by_hybrid[hybrid_name].get('sigma_0.6', (-999, 'n/a'))
        
        hybrid_results.append(HybridSearchResult(
            dataset=dataset_name,
            hybrid_name=hybrid_name,
            reps_combined='+'.join(rep_list),
            budget=budget,
            allocation_method=alloc_method,
            n_features=hybrid_train.shape[1],
            allocation=str({k: v for k, v in alloc.items() if isinstance(v, (int, float))}),
            best_r2_recovery_sigma03=r03,
            best_config_sigma03=cfg03,
            best_r2_recovery_sigma06=r06,
            best_config_sigma06=cfg06,
            creation_time_seconds=creation_time
        ).to_dict())
    
    df = pd.DataFrame(hybrid_results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("HYBRID SEARCH SUMMARY")
    print(f"{'='*60}")
    
    if not df.empty:
        print("\nRanked by average R² recovery:")
        df['avg_recovery'] = (df['best_r2_recovery_sigma03'] + df['best_r2_recovery_sigma06']) / 2
        df_sorted = df.sort_values('avg_recovery', ascending=False)
        
        for _, row in df_sorted.iterrows():
            avg = row['avg_recovery']
            marker = "✓" if avg > 0 else "✗"
            print(f"  {row['hybrid_name']:30s}: avg={avg:+6.1%} "
                  f"(σ0.3:{row['best_r2_recovery_sigma03']:+.1%}, σ0.6:{row['best_r2_recovery_sigma06']:+.1%}) {marker}")
        
        # Save
        df.to_csv(results_dir / f"{dataset_name.lower()}_hybrid_search.csv", index=False)
        print(f"\n  Saved to {results_dir / f'{dataset_name.lower()}_hybrid_search.csv'}")
    
    return df


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_dataset(
    dataset_name: str, 
    train_smiles: List[str], 
    train_labels: np.ndarray,
    test_smiles: List[str], 
    test_labels: np.ndarray, 
    results_dir: Path
) -> List[Dict]:
    """Run full experiment for one dataset."""
    
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    
    dataset_dir = results_dir / dataset_name.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Create base representations
    print("\nCreating base representations...")
    t0 = time.time()
    base_reps = create_base_representations(train_smiles, test_smiles)
    print(f"  Done ({time.time() - t0:.1f}s)")
    
    for rep_name, (rep_train, _) in base_reps.items():
        print(f"  {rep_name}: {rep_train.shape}")
    
    # Create all hybrid configurations
    print("\nCreating hybrid representations...")
    hybrids = create_all_hybrids(base_reps, train_labels, HYBRID_CONFIGS)
    
    # Combine all representations
    representations = {}
    representations.update(base_reps)
    for name, (h_train, h_test, h_info, _) in hybrids.items():
        representations[name] = (h_train, h_test)
    
    # ECFP for evaluation
    ecfp_train, ecfp_test = base_reps['ecfp4']
    
    # Clean baseline
    print("\nComputing clean baseline (ECFP4/RF)...")
    clean_metrics = train_evaluate_rf(ecfp_train, ecfp_test, train_labels, test_labels)
    print(f"  Clean R² = {clean_metrics['r2']:.4f}, MAE = {clean_metrics['mae']:.4f}")
    
    # ============================================================
    # HYBRID SEARCH: Find best hybrid configurations
    # ============================================================
    hybrid_search_df = find_best_hybrids(
        hybrids, ecfp_train, ecfp_test,
        train_labels, test_labels,
        dataset_name, results_dir
    )
    
    # Determine best hybrid for full experiments
    if not hybrid_search_df.empty:
        hybrid_search_df['avg_recovery'] = (
            hybrid_search_df['best_r2_recovery_sigma03'] + 
            hybrid_search_df['best_r2_recovery_sigma06']
        ) / 2
        best_hybrid_name = hybrid_search_df.loc[hybrid_search_df['avg_recovery'].idxmax(), 'hybrid_name']
        print(f"\n  Best hybrid for full experiments: {best_hybrid_name}")
    else:
        best_hybrid_name = None
    
    # ============================================================
    # FULL EXPERIMENTS
    # ============================================================
    
    # Select representations for full comparison
    # Use base reps + best hybrid
    smoothing_reps = list(BASE_REPS)
    if best_hybrid_name and best_hybrid_name in representations:
        smoothing_reps.append(best_hybrid_name)
    
    for sigma in NOISE_LEVELS:
        print(f"\n{'='*60}")
        print(f"σ = {sigma}")
        print(f"{'='*60}")
        
        # Inject noise
        injector = NoiseInjectorRegression(strategy='legacy', random_state=42)
        y_noisy = injector.inject(train_labels, sigma)
        effective_noise = injector.get_effective_noise(train_labels, y_noisy, method='std_normalized')
        print(f"  Effective noise: {effective_noise:.3f} (σ-normalized)")
        
        # Noisy baseline
        noisy_metrics = train_evaluate_rf(ecfp_train, ecfp_test, y_noisy, test_labels)
        print(f"  Noisy R² = {noisy_metrics['r2']:.4f} (drop: {clean_metrics['r2'] - noisy_metrics['r2']:.4f})")
        
        baselines = {
            'r2_clean': clean_metrics['r2'],
            'r2_noisy': noisy_metrics['r2'],
            'mae_clean': clean_metrics['mae'],
            'mae_noisy': noisy_metrics['mae'],
        }
        
        # ============================================================
        # PART 1: Neighbor-based methods
        # ============================================================
        print(f"\n  --- NEIGHBOR-BASED METHODS ---")
        
        for rep_name in smoothing_reps:
            print(f"\n  [{rep_name.upper()}]")
            X_rep = representations[rep_name][0]
            
            for dist_metric in DISTANCE_METRICS:
                print(f"    Distance: {dist_metric}")
                
                for est_name in NEIGHBOR_ESTIMATORS:
                    estimator = get_noise_estimator(
                        est_name, k=K_NEIGHBORS, metric=dist_metric, random_state=42
                    )
                    
                    noise_scores = estimator.estimate(X_rep, y_noisy)
                    neighbor_means = getattr(estimator, 'neighbor_means_', None)
                    
                    if neighbor_means is None:
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric=dist_metric)
                        nn.fit(X_rep)
                        _, indices = nn.kneighbors(X_rep)
                        neighbor_means = y_noisy[indices[:, 1:]].mean(axis=1)
                    
                    # Smoothing
                    for smooth_method in SMOOTHING_METHODS:
                        for alpha in ALPHA_VALUES:
                            result = run_experiment_with_scores(
                                noise_scores, neighbor_means,
                                ecfp_train, ecfp_test,
                                y_noisy, test_labels,
                                'smooth', smooth_method, alpha,
                                baselines, dataset_name, sigma, 
                                rep_name, est_name, dist_metric
                            )
                            all_results.append(result.to_dict())
                            print_result(result)
                    
                    # Weighting
                    for weight_method in WEIGHTING_METHODS:
                        for strength in ALPHA_VALUES:
                            result = run_experiment_with_scores(
                                noise_scores, neighbor_means,
                                ecfp_train, ecfp_test,
                                y_noisy, test_labels,
                                'weight', weight_method, strength,
                                baselines, dataset_name, sigma,
                                rep_name, est_name, dist_metric
                            )
                            all_results.append(result.to_dict())
                            print_result(result)
        
        # ============================================================
        # PART 2: SOTA methods
        # ============================================================
        print(f"\n  --- SOTA METHODS (using ECFP4 features) ---")
        
        for est_name in SOTA_ESTIMATORS:
            print(f"\n  [{est_name.upper()}]")
            
            estimator = get_noise_estimator(est_name, random_state=42)
            
            try:
                noise_scores = estimator.estimate(ecfp_train, y_noisy)
            except Exception as e:
                print(f"    FAILED: {e}")
                continue
            
            predictions = getattr(estimator, 'predictions_', None)
            if predictions is None:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='cosine')
                nn.fit(ecfp_train)
                _, indices = nn.kneighbors(ecfp_train)
                neighbor_means = y_noisy[indices[:, 1:]].mean(axis=1)
            else:
                neighbor_means = predictions
            
            # Smoothing
            for smooth_method in SMOOTHING_METHODS:
                for alpha in ALPHA_VALUES:
                    result = run_experiment_with_scores(
                        noise_scores, neighbor_means,
                        ecfp_train, ecfp_test,
                        y_noisy, test_labels,
                        'smooth', smooth_method, alpha,
                        baselines, dataset_name, sigma,
                        'ecfp4', est_name, 'n/a'
                    )
                    all_results.append(result.to_dict())
                    print_result(result)
            
            # Weighting
            for weight_method in WEIGHTING_METHODS:
                for strength in ALPHA_VALUES:
                    result = run_experiment_with_scores(
                        noise_scores, neighbor_means,
                        ecfp_train, ecfp_test,
                        y_noisy, test_labels,
                        'weight', weight_method, strength,
                        baselines, dataset_name, sigma,
                        'ecfp4', est_name, 'n/a'
                    )
                    all_results.append(result.to_dict())
                    print_result(result)
        
        # Checkpoint
        checkpoint_df = pd.DataFrame(all_results)
        checkpoint_df.to_csv(dataset_dir / f"checkpoint_sigma{sigma}.csv", index=False)
        print(f"\n  ✓ Checkpoint saved ({len(all_results)} results)")
    
    # Final save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(dataset_dir / "all_results.csv", index=False)
    print(f"\n✓ Saved {len(results_df)} results")
    
    # Summary
    print_summary(results_df)
    
    return all_results


def print_summary(df: pd.DataFrame):
    """Print summary of best results."""
    print(f"\n{'='*60}")
    print("SUMMARY: Best R² recovery by representation × distance metric")
    print(f"{'='*60}")
    
    for sigma in df['sigma'].unique():
        print(f"\nσ = {sigma}:")
        df_sigma = df[df['sigma'] == sigma]
        
        # Best per representation × distance metric (neighbor-based)
        df_neighbor = df_sigma[df_sigma['distance_metric'] != 'n/a']
        
        reps_in_data = df_neighbor['smoothing_rep'].unique()
        for rep in reps_in_data:
            for dm in DISTANCE_METRICS:
                df_sub = df_neighbor[(df_neighbor['smoothing_rep'] == rep) & 
                                     (df_neighbor['distance_metric'] == dm)]
                if df_sub.empty:
                    continue
                best = df_sub.loc[df_sub['r2_recovery'].idxmax()]
                rec = best['r2_recovery']
                marker = "✓" if rec > 0 else "✗"
                print(f"  {rep:30s}/{dm:10s}: {rec:+6.1%} ({best['estimator']}/{best['method']}, α={best['alpha']}) {marker}")
        
        # Best SOTA methods
        df_sota = df_sigma[df_sigma['distance_metric'] == 'n/a']
        if not df_sota.empty:
            print(f"\n  SOTA methods:")
            for est in SOTA_ESTIMATORS:
                df_est = df_sota[df_sota['estimator'] == est]
                if df_est.empty:
                    continue
                best = df_est.loc[df_est['r2_recovery'].idxmax()]
                rec = best['r2_recovery']
                marker = "✓" if rec > 0 else "✗"
                print(f"    {est:20s}: {rec:+6.1%} ({best['method']}, α={best['alpha']}) {marker}")
        
        # Overall best
        best_overall = df_sigma.loc[df_sigma['r2_recovery'].idxmax()]
        dm_str = f"/{best_overall['distance_metric']}" if best_overall['distance_metric'] != 'n/a' else ""
        print(f"\n  {'BEST':30s}: {best_overall['r2_recovery']:+6.1%} "
              f"({best_overall['smoothing_rep']}{dm_str}/{best_overall['estimator']}/{best_overall['method']})")


def main(dataset: str = 'both'):
    """Main entry point."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results_denoising_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LABEL DENOISING EXPERIMENT")
    print("="*80)
    print(f"Base representations: {', '.join(BASE_REPS)}")
    print(f"Hybrid configurations: {len(HYBRID_CONFIGS)}")
    for name, reps, budget, method in HYBRID_CONFIGS:
        print(f"  - {name}: {'+'.join(reps)}, budget={budget}, {method}")
    print(f"Distance metrics: {', '.join(DISTANCE_METRICS)}")
    print(f"Neighbor-based estimators: {', '.join(NEIGHBOR_ESTIMATORS)}")
    print(f"SOTA estimators: {', '.join(SOTA_ESTIMATORS)}")
    print(f"Smoothing methods: {', '.join(SMOOTHING_METHODS)}")
    print(f"Weighting methods: {', '.join(WEIGHTING_METHODS)}")
    print(f"Alpha values: {ALPHA_VALUES}")
    print(f"k={K_NEIGHBORS}")
    print(f"Evaluation: ECFP4 + RF")
    print(f"Results: {results_dir}")
    print("="*80)
    
    if dataset in ['qm9', 'both']:
        raw_data = load_qm9(n_samples=N_SAMPLES_QM9, property_idx=4)
        splits = get_qm9_splits(raw_data, splitter='scaffold')
        
        train_smiles = splits['train']['smiles'] + splits['val']['smiles']
        train_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
        
        run_dataset(
            'QM9', train_smiles, train_labels,
            splits['test']['smiles'], splits['test']['labels'],
            results_dir
        )
    
    if dataset in ['esol', 'both']:
        esol_data = load_esol_combined(splitter='scaffold')
        
        run_dataset(
            'ESOL',
            esol_data['train']['smiles'], esol_data['train']['labels'],
            esol_data['test']['smiles'], esol_data['test']['labels'],
            results_dir
        )
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Results: {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label denoising experiment')
    parser.add_argument('--dataset', choices=['qm9', 'esol', 'both'], default='both')
    args = parser.parse_args()
    main(dataset=args.dataset)