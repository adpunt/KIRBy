#!/usr/bin/env python3
"""
Noise Robustness Analysis: Hybrid Representations
==================================================

Test noise mitigation techniques across:
- Representations: PDV, MHG-GNN, Hybrid (greedy)
- Noise levels: σ ∈ {0.0, 0.3, 0.6}
- Methods: Distance-based, Ensemble, Model-based, Baselines
- Distance metrics: Euclidean, Manhattan, Cosine, Mahalanobis
- Feature weighting: None vs importance scores

Metrics:
- Performance recovery: How well does cleaning restore clean performance?
- Cleaning accuracy: Precision/recall of noise detection
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.datasets.esol import load_esol_combined
from kirby.representations.molecular import create_pdv, create_mhg_gnn
from kirby.hybrid import create_hybrid, apply_feature_selection
from kirby.utils.feature_filtering import apply_filters, FILTER_CONFIGS
from noiseInject import NoiseInjectorRegression

# Import noise mitigation methods
from kirby.noise_mitigation_methods import get_mitigation_method


# ============================================================================
# CONFIGURATION
# ============================================================================

NOISE_LEVELS = [0.0, 0.3, 0.6]  # Clean, moderate, high
SEEDS = 5

# Methods to test
DISTANCE_BASED_METHODS = ['knn_k5', 'knn_k10', 'activity_cliffs', 'mahalanobis']
ENSEMBLE_METHODS = ['cv_disagreement', 'bootstrap_ensemble']
MODEL_METHODS = ['co_teaching', 'dividemix']
BASELINE_METHODS = ['zscore', 'prediction_error']

ALL_METHODS = DISTANCE_BASED_METHODS + ENSEMBLE_METHODS + MODEL_METHODS + BASELINE_METHODS

# Distance metrics (for distance-based methods)
DISTANCE_METRICS = ['euclidean', 'manhattan', 'cosine']

# Feature weighting options
FEATURE_WEIGHTING = [None, 'importance']  # None = unweighted, 'importance' = use scores


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train RF and evaluate performance"""
    if X_train.shape[1] == 0 or len(y_train) < 10:
        return {'r2': -np.inf, 'mae': np.inf, 'rmse': np.inf}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    return {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(np.mean((y_test - y_pred)**2))
    }


def calculate_cleaning_metrics(true_noise_mask, detected_noise_indices, total_samples):
    """
    Calculate precision, recall, F1 for noise detection.
    
    Parameters:
    -----------
    true_noise_mask : array of bool
        True noise labels (True = actually corrupted)
    detected_noise_indices : array of int
        Indices flagged as noisy by method
    total_samples : int
        Total number of samples
    
    Returns:
    --------
    metrics : dict
    """
    # Convert detected indices to mask
    detected_mask = np.zeros(total_samples, dtype=bool)
    detected_mask[detected_noise_indices] = True
    
    # Calculate metrics
    if detected_mask.sum() == 0:
        # No samples flagged
        precision = 1.0 if true_noise_mask.sum() == 0 else 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = precision_score(true_noise_mask, detected_mask, zero_division=0)
        recall = recall_score(true_noise_mask, detected_mask, zero_division=0)
        f1 = f1_score(true_noise_mask, detected_mask, zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_detected': len(detected_noise_indices),
        'n_true_noise': true_noise_mask.sum()
    }


def get_feature_importance_weights(X, y, method='rf'):
    """
    Get feature importance scores for weighting.
    
    Parameters:
    -----------
    X : array
        Feature matrix
    y : array
        Labels
    method : str
        'rf' for random forest importance
    
    Returns:
    --------
    weights : array
        Feature importance scores (normalized)
    """
    if method == 'rf':
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
        weights = model.feature_importances_
        # Normalize to sum to number of features (so mean weight = 1)
        weights = weights / weights.mean()
        return weights
    else:
        raise ValueError(f"Unknown importance method: {method}")


# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_noise_mitigation_experiment(
    X_train, X_test, y_train, y_test,
    rep_name, sigma, seed,
    feature_importance_scores=None
):
    """
    Run full noise mitigation experiment for one representation at one noise level.
    
    Parameters:
    -----------
    X_train, X_test : arrays
        Feature matrices
    y_train, y_test : arrays
        Labels (y_train is CLEAN ground truth)
    rep_name : str
        Representation name
    sigma : float
        Noise level
    seed : int
        Random seed
    feature_importance_scores : array or None
        Pre-computed feature importance for weighting
    
    Returns:
    --------
    results : list of dicts
    """
    
    results = []
    
    # Add noise to training labels
    injector = NoiseInjectorRegression(strategy='legacy', random_state=seed)
    y_train_noisy = injector.inject(y_train, sigma) if sigma > 0 else y_train.copy()
    
    # Track which samples were corrupted (ground truth)
    true_noise_mask = (y_train_noisy != y_train)
    
    # Baseline: clean data (upper bound)
    clean_metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Baseline: noisy data (no mitigation)
    noisy_metrics = train_and_evaluate(X_train, X_test, y_train_noisy, y_test)
    
    print(f"\n  Baselines: Clean R²={clean_metrics['r2']:.4f}, Noisy R²={noisy_metrics['r2']:.4f}")
    
    # Test each mitigation method
    for method_name in ALL_METHODS:
        
        # Determine if method uses distance metrics
        if method_name in DISTANCE_BASED_METHODS:
            distance_metrics = DISTANCE_METRICS
            # Mahalanobis doesn't need separate metric testing
            if method_name == 'mahalanobis':
                distance_metrics = [None]
        else:
            distance_metrics = [None]
        
        # Test distance metrics
        for distance_metric in distance_metrics:
            
            # Test feature weighting
            for weighting in FEATURE_WEIGHTING:
                
                # Skip weighting if not distance-based
                if method_name not in DISTANCE_BASED_METHODS and weighting is not None:
                    continue
                
                # Get feature weights
                if weighting == 'importance' and feature_importance_scores is not None:
                    feature_weights = feature_importance_scores
                else:
                    feature_weights = None
                
                # Build method kwargs
                method_kwargs = {'random_state': seed}
                
                if distance_metric is not None:
                    method_kwargs['distance_metric'] = distance_metric
                
                if feature_weights is not None:
                    method_kwargs['feature_weights'] = feature_weights
                
                # Get method
                try:
                    method = get_mitigation_method(method_name, **method_kwargs)
                except Exception as e:
                    print(f"  ERROR: {method_name} with {distance_metric}, {weighting}: {e}")
                    continue
                
                # Apply mitigation
                try:
                    start = time.time()
                    X_train_clean, y_train_clean, removed_indices = method.clean_data(
                        X_train, y_train_noisy
                    )
                    elapsed = time.time() - start
                except Exception as e:
                    print(f"  ERROR: {method_name} mitigation failed: {e}")
                    continue
                
                # Evaluate cleaned data
                cleaned_metrics = train_and_evaluate(X_train_clean, X_test, y_train_clean, y_test)
                
                # Calculate cleaning accuracy
                cleaning_acc = calculate_cleaning_metrics(
                    true_noise_mask, removed_indices, len(y_train)
                )
                
                # Calculate performance recovery
                if clean_metrics['r2'] != noisy_metrics['r2']:
                    recovery_rate = (cleaned_metrics['r2'] - noisy_metrics['r2']) / \
                                  (clean_metrics['r2'] - noisy_metrics['r2'])
                else:
                    recovery_rate = 0.0
                
                # Store result
                result = {
                    'representation': rep_name,
                    'sigma': sigma,
                    'seed': seed,
                    'method': method_name,
                    'distance_metric': distance_metric if distance_metric else 'none',
                    'weighted': weighting is not None,
                    
                    # Performance metrics
                    'r2_clean': clean_metrics['r2'],
                    'r2_noisy': noisy_metrics['r2'],
                    'r2_cleaned': cleaned_metrics['r2'],
                    'mae_cleaned': cleaned_metrics['mae'],
                    'rmse_cleaned': cleaned_metrics['rmse'],
                    'recovery_rate': recovery_rate,
                    
                    # Cleaning accuracy
                    'precision': cleaning_acc['precision'],
                    'recall': cleaning_acc['recall'],
                    'f1': cleaning_acc['f1'],
                    'n_detected': cleaning_acc['n_detected'],
                    'n_true_noise': cleaning_acc['n_true_noise'],
                    'n_samples_after': len(y_train_clean),
                    
                    # Other
                    'time_seconds': elapsed
                }
                
                results.append(result)
                
                # Print progress
                print(f"    {method_name:20s} {str(distance_metric):12s} weighted={weighting is not None} "
                      f"→ R²={cleaned_metrics['r2']:.4f} (+{recovery_rate:+.2%}), "
                      f"F1={cleaning_acc['f1']:.3f}")
    
    return results


# ============================================================================
# DATASET TESTING
# ============================================================================

def test_dataset(dataset_name, train_smiles, train_labels, test_smiles, test_labels, 
                results_dir):
    """
    Test all methods on a single dataset.
    """
    
    print("\n" + "="*100)
    print(f"{dataset_name.upper()} DATASET")
    print("="*100)
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    
    all_results = []
    
    # Create output directory
    dataset_dir = Path(results_dir) / dataset_name.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in range(SEEDS):
        print(f"\n{'='*100}")
        print(f"SEED {seed}")
        print(f"{'='*100}")
        
        # Create representations
        print("\nCreating representations...")
        start = time.time()
        
        pdv_train = create_pdv(train_smiles)
        pdv_test = create_pdv(test_smiles)
        mhggnn_train = create_mhg_gnn(train_smiles)
        mhggnn_test = create_mhg_gnn(test_smiles)
        
        print(f"Done ({time.time() - start:.1f}s)")
        print(f"  PDV:    {pdv_train.shape[1]} features")
        print(f"  mhggnn: {mhggnn_train.shape[1]} features")
        
        # Apply quality filters
        print("\nApplying quality filters...")
        filter_config = FILTER_CONFIGS['quality_only']
        
        rep_dict_train = {'pdv': pdv_train, 'mhggnn': mhggnn_train}
        rep_dict_test = {'pdv': pdv_test, 'mhggnn': mhggnn_test}
        
        filtered_train, filtered_test = apply_filters(
            rep_dict_train, rep_dict_test, train_labels, **filter_config
        )
        
        print(f"  PDV:    {pdv_train.shape[1]} → {filtered_train['pdv'].shape[1]}")
        print(f"  mhggnn: {mhggnn_train.shape[1]} → {filtered_train['mhggnn'].shape[1]}")
        
        pdv_train_filt = filtered_train['pdv']
        mhggnn_train_filt = filtered_train['mhggnn']
        pdv_test_filt = filtered_test['pdv']
        mhggnn_test_filt = filtered_test['mhggnn']
        
        # Create hybrid with greedy allocation
        print("\nCreating hybrid (greedy, budget=100)...")
        rep_dict_train_filt = {'pdv': pdv_train_filt, 'mhggnn': mhggnn_train_filt}
        rep_dict_test_filt = {'pdv': pdv_test_filt, 'mhggnn': mhggnn_test_filt}
        
        hybrid_train, feature_info = create_hybrid(
            rep_dict_train_filt,
            train_labels,
            allocation_method='greedy',
            budget=100,
            step_size=10,
            patience=3,
            validation_split=0.2,
            apply_filters=False
        )
        
        hybrid_test = apply_feature_selection(rep_dict_test_filt, feature_info)
        
        print(f"  Hybrid: {hybrid_train.shape[1]} features")
        print(f"  Allocation: {feature_info['allocation']}")
        
        # Get feature importance scores for each representation
        print("\nComputing feature importance scores...")
        pdv_importance = get_feature_importance_weights(pdv_train_filt, train_labels)
        mhggnn_importance = get_feature_importance_weights(mhggnn_train_filt, train_labels)
        hybrid_importance = get_feature_importance_weights(hybrid_train, train_labels)
        
        # Test at each noise level
        for sigma in NOISE_LEVELS:
            print(f"\n{'-'*100}")
            print(f"NOISE LEVEL: σ={sigma}")
            print(f"{'-'*100}")
            
            # Test PDV
            print("\n[PDV Baseline]")
            pdv_results = run_noise_mitigation_experiment(
                pdv_train_filt, pdv_test_filt, train_labels, test_labels,
                'pdv', sigma, seed, pdv_importance
            )
            all_results.extend(pdv_results)
            
            # Test MHG-GNN
            print("\n[MHG-GNN Baseline]")
            mhggnn_results = run_noise_mitigation_experiment(
                mhggnn_train_filt, mhggnn_test_filt, train_labels, test_labels,
                'mhggnn', sigma, seed, mhggnn_importance
            )
            all_results.extend(mhggnn_results)
            
            # Test Hybrid
            print("\n[Hybrid (Greedy)]")
            hybrid_results = run_noise_mitigation_experiment(
                hybrid_train, hybrid_test, train_labels, test_labels,
                'hybrid', sigma, seed, hybrid_importance
            )
            all_results.extend(hybrid_results)
        
        # Save seed results
        seed_df = pd.DataFrame(all_results)
        seed_file = dataset_dir / f"seed_{seed}_results.csv"
        seed_df.to_csv(seed_file, index=False)
        print(f"\n✓ Saved {len(seed_df)} results for seed {seed}")
    
    # Aggregate all seeds
    all_df = pd.DataFrame(all_results)
    all_file = dataset_dir / "all_results.csv"
    all_df.to_csv(all_file, index=False)
    
    print(f"\n{'='*100}")
    print(f"DATASET COMPLETE: {dataset_name}")
    print(f"Total experiments: {len(all_df)}")
    print(f"Results saved to: {dataset_dir}")
    print(f"{'='*100}")
    
    return all_results


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_results(results_df, dataset_name):
    """
    Analyze and summarize results.
    """
    
    print("\n" + "="*100)
    print(f"ANALYSIS: {dataset_name.upper()}")
    print("="*100)
    
    # Average across seeds
    grouped = results_df.groupby([
        'representation', 'sigma', 'method', 'distance_metric', 'weighted'
    ]).agg({
        'r2_clean': 'mean',
        'r2_noisy': 'mean',
        'r2_cleaned': 'mean',
        'recovery_rate': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean'
    }).reset_index()
    
    # Best methods per representation at σ=0.6
    print("\n" + "-"*100)
    print("TOP 5 METHODS BY RECOVERY RATE (σ=0.6)")
    print("-"*100)
    
    high_noise = grouped[grouped['sigma'] == 0.6]
    
    for rep in ['pdv', 'mhggnn', 'hybrid']:
        rep_data = high_noise[high_noise['representation'] == rep]
        top5 = rep_data.nlargest(5, 'recovery_rate')
        
        print(f"\n{rep.upper()}:")
        for _, row in top5.iterrows():
            print(f"  {row['method']:20s} {row['distance_metric']:12s} "
                  f"weighted={row['weighted']:5} → "
                  f"Recovery={row['recovery_rate']:+.2%}, F1={row['f1']:.3f}")
    
    # Compare representations
    print("\n" + "-"*100)
    print("REPRESENTATION COMPARISON (Average across all methods)")
    print("-"*100)
    
    rep_summary = grouped.groupby(['representation', 'sigma']).agg({
        'recovery_rate': 'mean',
        'f1': 'mean'
    }).reset_index()
    
    print(f"\n{'Rep':10s} {'σ':5s} {'Avg Recovery':15s} {'Avg F1':10s}")
    print("-"*50)
    for _, row in rep_summary.iterrows():
        print(f"{row['representation']:10s} {row['sigma']:5.1f} "
              f"{row['recovery_rate']:+14.2%} {row['f1']:10.3f}")
    
    # Distance metric comparison (for distance-based methods)
    print("\n" + "-"*100)
    print("DISTANCE METRIC COMPARISON (Distance-based methods, σ=0.6)")
    print("-"*100)
    
    distance_methods = high_noise[
        high_noise['method'].isin(DISTANCE_BASED_METHODS) &
        (high_noise['distance_metric'] != 'none')
    ]
    
    metric_summary = distance_methods.groupby(['representation', 'distance_metric']).agg({
        'recovery_rate': 'mean',
        'f1': 'mean'
    }).reset_index()
    
    for rep in ['pdv', 'mhggnn', 'hybrid']:
        rep_data = metric_summary[metric_summary['representation'] == rep]
        print(f"\n{rep.upper()}:")
        for _, row in rep_data.iterrows():
            print(f"  {row['distance_metric']:12s}: Recovery={row['recovery_rate']:+.2%}, F1={row['f1']:.3f}")
    
    # Feature weighting impact
    print("\n" + "-"*100)
    print("FEATURE WEIGHTING IMPACT (Distance-based methods, σ=0.6)")
    print("-"*100)
    
    weighting_comparison = distance_methods.groupby(['representation', 'weighted']).agg({
        'recovery_rate': 'mean',
        'f1': 'mean'
    }).reset_index()
    
    for rep in ['pdv', 'mhggnn', 'hybrid']:
        rep_data = weighting_comparison[weighting_comparison['representation'] == rep]
        print(f"\n{rep.upper()}:")
        for _, row in rep_data.iterrows():
            weighted_str = "Weighted" if row['weighted'] else "Unweighted"
            print(f"  {weighted_str:12s}: Recovery={row['recovery_rate']:+.2%}, F1={row['f1']:.3f}")
    
    return grouped


# ============================================================================
# MAIN
# ============================================================================

def main():
    
    results_dir = Path("results/noise_robustness")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*100)
    print("NOISE ROBUSTNESS ANALYSIS: HYBRID REPRESENTATIONS")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Noise levels: {NOISE_LEVELS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Methods: {len(ALL_METHODS)}")
    print(f"  Distance metrics: {DISTANCE_METRICS}")
    print(f"  Feature weighting: {FEATURE_WEIGHTING}")
    print("="*100)
    
    all_dataset_results = {}
    
    # ========================================================================
    # QM9 DATASET
    # ========================================================================
    print("\n" + "="*100)
    print("LOADING QM9 DATASET")
    print("="*100)
    
    raw_data = load_qm9(n_samples=10000, property_idx=4)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    qm9_train_smiles = splits['train']['smiles'] + splits['val']['smiles']
    qm9_train_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
    qm9_test_smiles = splits['test']['smiles']
    qm9_test_labels = splits['test']['labels']
    
    qm9_results = test_dataset(
        'QM9', qm9_train_smiles, qm9_train_labels,
        qm9_test_smiles, qm9_test_labels, results_dir
    )
    all_dataset_results['QM9'] = qm9_results
    
    # ========================================================================
    # ESOL DATASET
    # ========================================================================
    print("\n" + "="*100)
    print("LOADING ESOL DATASET")
    print("="*100)
    
    esol_data = load_esol_combined(splitter='scaffold')
    
    esol_train_smiles = esol_data['train']['smiles']
    esol_train_labels = esol_data['train']['labels']
    esol_test_smiles = esol_data['test']['smiles']
    esol_test_labels = esol_data['test']['labels']
    
    esol_results = test_dataset(
        'ESOL', esol_train_smiles, esol_train_labels,
        esol_test_smiles, esol_test_labels, results_dir
    )
    all_dataset_results['ESOL'] = esol_results
    
    # ========================================================================
    # CROSS-DATASET ANALYSIS
    # ========================================================================
    
    for dataset_name, results in all_dataset_results.items():
        results_df = pd.DataFrame(results)
        analyze_results(results_df, dataset_name)
    
    print("\n" + "="*100)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*100)
    print(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    main()