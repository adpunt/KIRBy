#!/usr/bin/env python3
"""
Noise Mitigation Experiment
============================

Design:
- Hybrid representation (pdv+mhggnn) for noise DETECTION
- ECFP4 + Random Forest for EVALUATION

This cleanly separates the hypothesis: do hybrid representations 
provide better distance estimates for identifying noisy samples?
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.datasets.esol import load_esol_combined
from kirby.representations.molecular import create_pdv, create_mhg_gnn, create_ecfp4
from kirby.hybrid import create_hybrid, apply_feature_selection
from noiseInject import NoiseInjectorRegression
from noise_mitigation_methods import get_mitigation_method, DISTANCE_AWARE_METHODS, compute_distance_matrix


# ============================================================================
# CONFIGURATION
# ============================================================================

NOISE_LEVELS = [0.3, 0.6]
N_SAMPLES_QM9 = 5000
N_JOBS = -1

# Default hybrid config (from QM9 HOMO-LUMO gap optimization)
DEFAULT_HYBRID = {
    'combo': 'pdv+mhggnn',
    'budget': 100,
    'allocation': 'greedy',
    'filtered': False,
}

# Distance metrics to test
DISTANCE_METRICS = ['euclidean', 'manhattan', 'cosine', 'chebyshev']

# Methods that use precomputed distances
DISTANCE_METHODS = [
    'knn_k5', 'knn_k10', 'knn_k20',
    'lof', 'lof_k20',
    'distance_weighted',
    'neighborhood_consensus',
    'activity_cliffs',
]

# Methods that don't use distances
OTHER_METHODS = ['mahalanobis', 'cv_disagreement', 'bootstrap_ensemble', 
                 'co_teaching', 'dividemix', 'zscore', 'prediction_error']


# ============================================================================
# RESULT STRUCTURE
# ============================================================================

@dataclass
class Result:
    dataset: str
    sigma: float
    detection_rep: str  # 'hybrid', 'ecfp', 'pdv', 'mhggnn'
    method: str
    distance_metric: str
    weighted: bool
    
    r2_clean: float
    r2_noisy: float
    r2_cleaned: float
    recovery_rate: float
    
    precision: float
    recall: float
    f1: float
    n_removed: int
    n_true_noise: int
    
    time_seconds: float
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def train_evaluate_ecfp(X_ecfp_train, X_ecfp_test, y_train, y_test):
    """Train RF on ECFP, return metrics."""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_ecfp_train, y_train)
    y_pred = model.predict(X_ecfp_test)
    return {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }


def get_feature_importance(X, y):
    """Get RF feature importances for weighting."""
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model.feature_importances_


def cleaning_metrics(true_noise_mask, removed_idx, n_total):
    """Calculate precision, recall, F1 for noise detection."""
    n_true_noise = true_noise_mask.sum()
    n_removed = len(removed_idx)
    
    if n_removed == 0:
        return 0.0, 0.0, 0.0
    
    true_positives = true_noise_mask[removed_idx].sum()
    precision = true_positives / n_removed if n_removed > 0 else 0
    recall = true_positives / n_true_noise if n_true_noise > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def run_single_experiment(
    X_detect: np.ndarray,
    X_ecfp_train: np.ndarray,
    X_ecfp_test: np.ndarray,
    y_train_clean: np.ndarray,
    y_train_noisy: np.ndarray,
    y_test: np.ndarray,
    true_noise_mask: np.ndarray,
    precomputed_distances: Optional[np.ndarray],
    feature_weights: Optional[np.ndarray],
    method_name: str,
    distance_metric: Optional[str],
    weighted: bool,
    r2_clean: float,
    r2_noisy: float,
    dataset: str,
    sigma: float,
    detection_rep: str
) -> Optional[Result]:
    """Run one mitigation method, evaluate with ECFP/RF."""
    
    method_kwargs = {'random_state': 42}
    
    if method_name in DISTANCE_AWARE_METHODS:
        if precomputed_distances is not None:
            method_kwargs['precomputed_distances'] = precomputed_distances
        if distance_metric:
            method_kwargs['distance_metric'] = distance_metric
        if weighted and feature_weights is not None:
            method_kwargs['feature_weights'] = feature_weights
    
    try:
        method = get_mitigation_method(method_name, **method_kwargs)
        start = time.time()
        noisy_indices = method.identify_noise(X_detect, y_train_noisy)
        elapsed = time.time() - start
    except Exception as e:
        return None
    
    # Remove detected noise from ECFP training data
    clean_mask = np.ones(len(y_train_noisy), dtype=bool)
    clean_mask[noisy_indices] = False
    
    X_ecfp_cleaned = X_ecfp_train[clean_mask]
    y_cleaned = y_train_noisy[clean_mask]
    
    if len(y_cleaned) < 10:
        return None
    
    # Evaluate with ECFP/RF
    metrics = train_evaluate_ecfp(X_ecfp_cleaned, X_ecfp_test, y_cleaned, y_test)
    
    # Cleaning accuracy
    precision, recall, f1 = cleaning_metrics(true_noise_mask, noisy_indices, len(y_train_noisy))
    
    # Recovery rate
    if r2_clean != r2_noisy:
        recovery = (metrics['r2'] - r2_noisy) / (r2_clean - r2_noisy)
    else:
        recovery = 0.0
    
    return Result(
        dataset=dataset,
        sigma=sigma,
        detection_rep=detection_rep,
        method=method_name,
        distance_metric=distance_metric or 'none',
        weighted=weighted,
        r2_clean=r2_clean,
        r2_noisy=r2_noisy,
        r2_cleaned=metrics['r2'],
        recovery_rate=recovery,
        precision=precision,
        recall=recall,
        f1=f1,
        n_removed=len(noisy_indices),
        n_true_noise=int(true_noise_mask.sum()),
        time_seconds=elapsed
    )


def run_all_methods(
    X_detect: np.ndarray,
    X_ecfp_train: np.ndarray,
    X_ecfp_test: np.ndarray,
    y_train_clean: np.ndarray,
    y_train_noisy: np.ndarray,
    y_test: np.ndarray,
    true_noise_mask: np.ndarray,
    r2_clean: float,
    r2_noisy: float,
    dataset: str,
    sigma: float,
    detection_rep: str,
    cache_dir: Path
) -> List[Result]:
    """Run all methods for one detection representation."""
    
    results = []
    feature_weights = get_feature_importance(X_detect, y_train_clean)
    
    # Precompute/load distance matrices
    print(f"      Loading/computing distance matrices...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    distance_cache = {}
    
    for metric in DISTANCE_METRICS:
        for weighted in [False, True]:
            key = (metric, weighted)
            cache_file = cache_dir / f"{metric}_w{weighted}.npy"
            
            if cache_file.exists():
                distance_cache[key] = np.load(cache_file)
            else:
                weights = feature_weights if weighted else None
                D = compute_distance_matrix(X_detect, metric, weights)
                np.save(cache_file, D)
                distance_cache[key] = D
    
    # Build experiment configs
    configs = []
    
    # Distance-aware methods
    for method in DISTANCE_METHODS:
        for metric in DISTANCE_METRICS:
            for weighted in [False, True]:
                configs.append((method, metric, weighted))
    
    # Other methods (no distance metric)
    for method in OTHER_METHODS:
        configs.append((method, None, False))
    
    print(f"      Running {len(configs)} experiments...")
    
    def run_one(cfg):
        method, metric, weighted = cfg
        precomputed = distance_cache.get((metric, weighted)) if metric else None
        weights = feature_weights if weighted else None
        
        return run_single_experiment(
            X_detect=X_detect,
            X_ecfp_train=X_ecfp_train,
            X_ecfp_test=X_ecfp_test,
            y_train_clean=y_train_clean,
            y_train_noisy=y_train_noisy,
            y_test=y_test,
            true_noise_mask=true_noise_mask,
            precomputed_distances=precomputed,
            feature_weights=weights,
            method_name=method,
            distance_metric=metric,
            weighted=weighted,
            r2_clean=r2_clean,
            r2_noisy=r2_noisy,
            dataset=dataset,
            sigma=sigma,
            detection_rep=detection_rep
        )
    
    parallel_results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(run_one)(cfg) for cfg in configs
    )
    
    results = [r for r in parallel_results if r is not None]
    
    if results:
        best = max(results, key=lambda r: r.recovery_rate)
        print(f"      Best: {best.method}/{best.distance_metric} → {best.recovery_rate:+.1%}")
    
    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_dataset(dataset_name: str, train_smiles: List[str], train_labels: np.ndarray,
                test_smiles: List[str], test_labels: np.ndarray, 
                results_dir: Path) -> List[Dict]:
    """Run full experiment for one dataset."""
    
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    
    dataset_dir = results_dir / dataset_name.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Create representations
    print("\nCreating representations...")
    t0 = time.time()
    
    ecfp_train = create_ecfp4(train_smiles, radius=2, n_bits=2048)
    ecfp_test = create_ecfp4(test_smiles, radius=2, n_bits=2048)
    pdv_train = create_pdv(train_smiles)
    pdv_test = create_pdv(test_smiles)
    mhggnn_train = create_mhg_gnn(train_smiles)
    mhggnn_test = create_mhg_gnn(test_smiles)
    
    print(f"  ECFP:    {ecfp_train.shape}")
    print(f"  PDV:     {pdv_train.shape}")
    print(f"  MHG-GNN: {mhggnn_train.shape}")
    
    # Create hybrid
    print("\nCreating hybrid (pdv+mhggnn)...")
    reps_train = {'pdv': pdv_train, 'mhggnn': mhggnn_train}
    reps_test = {'pdv': pdv_test, 'mhggnn': mhggnn_test}
    
    hybrid_train, hybrid_info = create_hybrid(
        reps_train, train_labels,
        allocation_method=DEFAULT_HYBRID['allocation'],
        budget=DEFAULT_HYBRID['budget'],
        validation_split=0.2,
        apply_filters=False
    )
    hybrid_test = apply_feature_selection(reps_test, hybrid_info)
    print(f"  Hybrid:  {hybrid_train.shape}")
    print(f"  Done ({time.time() - t0:.1f}s)")
    
    # Clean baseline (ECFP/RF on clean data)
    print("\nComputing clean baseline...")
    clean_baseline = train_evaluate_ecfp(ecfp_train, ecfp_test, train_labels, test_labels)
    r2_clean = clean_baseline['r2']
    print(f"  Clean R² = {r2_clean:.4f}")
    
    # Run experiments for each noise level
    for sigma in NOISE_LEVELS:
        print(f"\n{'='*60}")
        print(f"σ = {sigma}")
        print(f"{'='*60}")
        
        # Inject noise
        injector = NoiseInjectorRegression(strategy='legacy', random_state=42)
        y_noisy = injector.inject(train_labels, sigma)
        true_noise_mask = (y_noisy != train_labels)
        print(f"  Injected {true_noise_mask.sum()} noisy labels ({true_noise_mask.mean():.1%})")
        
        # Noisy baseline
        noisy_baseline = train_evaluate_ecfp(ecfp_train, ecfp_test, y_noisy, test_labels)
        r2_noisy = noisy_baseline['r2']
        print(f"  Noisy R² = {r2_noisy:.4f} (drop: {r2_clean - r2_noisy:.4f})")
        
        # Use HYBRID for noise detection
        print(f"\n  [HYBRID] for noise detection → [ECFP4/RF] for evaluation")
        
        cache_dir = dataset_dir / f".cache/hybrid_sigma{sigma}"
        
        rep_results = run_all_methods(
            X_detect=hybrid_train,
            X_ecfp_train=ecfp_train,
            X_ecfp_test=ecfp_test,
            y_train_clean=train_labels,
            y_train_noisy=y_noisy,
            y_test=test_labels,
            true_noise_mask=true_noise_mask,
            r2_clean=r2_clean,
            r2_noisy=r2_noisy,
            dataset=dataset_name,
            sigma=sigma,
            detection_rep='hybrid',
            cache_dir=cache_dir
        )
        
        all_results.extend([r.to_dict() for r in rep_results])
        
        # Checkpoint save
        checkpoint_df = pd.DataFrame(all_results)
        checkpoint_df.to_csv(dataset_dir / f"checkpoint_sigma{sigma}.csv", index=False)
        print(f"      ✓ Checkpoint saved ({len(all_results)} total results)")
    
    # Final save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(dataset_dir / "all_results.csv", index=False)
    print(f"\n✓ Saved {len(results_df)} results to {dataset_dir / 'all_results.csv'}")
    
    return all_results


def main(dataset='both'):
    """Main entry point."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results_noise_mitigation_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("NOISE MITIGATION EXPERIMENT")
    print("="*80)
    print(f"Detection: hybrid (pdv+mhggnn)")
    print(f"Evaluation: ECFP4 + Random Forest")
    print(f"Results: {results_dir}")
    print("="*80)
    
    all_results = {}
    
    if dataset in ['qm9', 'both']:
        raw_data = load_qm9(n_samples=N_SAMPLES_QM9, property_idx=4)
        splits = get_qm9_splits(raw_data, splitter='scaffold')
        
        train_smiles = splits['train']['smiles'] + splits['val']['smiles']
        train_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
        
        all_results['QM9'] = run_dataset(
            'QM9', train_smiles, train_labels,
            splits['test']['smiles'], splits['test']['labels'],
            results_dir
        )
    
    if dataset in ['esol', 'both']:
        esol_data = load_esol_combined(splitter='scaffold')
        
        all_results['ESOL'] = run_dataset(
            'ESOL',
            esol_data['train']['smiles'], esol_data['train']['labels'],
            esol_data['test']['smiles'], esol_data['test']['labels'],
            results_dir
        )
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['qm9', 'esol', 'both'], default='both')
    args = parser.parse_args()
    main(dataset=args.dataset)