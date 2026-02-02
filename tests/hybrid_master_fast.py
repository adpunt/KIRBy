#!/usr/bin/env python3
"""
Hybrid Master Fast Test - Extended

Tests new KIRBy hybrid features:
1. MACCS as additional base representation
2. Augmentations with greedy ablation (graph_topology, spectral, subgraph_counts, graph_distances)
3. Comparison of augmentation strategies ('none', 'all', 'greedy_ablation')

Datasets: ESOL (water solubility), Lipophilicity, FreeSolv (optional)
Models: RandomForest, XGBoost, (optionally MLP)

Usage:
    python hybrid_master_fast.py                    # Run all tests
    python hybrid_master_fast.py --quick            # Quick smoke test
    python hybrid_master_fast.py --augmentations    # Focus on augmentation testing
    python hybrid_master_fast.py --threshold-sweep  # Test augmentation threshold values
"""

import numpy as np
import time
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Try importing XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available, skipping XGBoost tests")

# Try importing MoleculeNet datasets
try:
    from deepchem.molnet import load_delaney, load_lipo, load_freesolv
    HAS_DEEPCHEM = True
except ImportError:
    HAS_DEEPCHEM = False
    print("Warning: DeepChem not available, using synthetic data")


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class Dataset:
    """Container for dataset with SMILES and labels"""
    name: str
    smiles: List[str]
    labels: np.ndarray
    task_type: str = 'regression'


def load_esol() -> Dataset:
    """Load ESOL (water solubility) dataset"""
    if HAS_DEEPCHEM:
        tasks, datasets, _ = load_delaney(featurizer='Raw', splitter=None)
        train = datasets[0]
        smiles = [x[0] for x in train.ids]
        labels = train.y.flatten()
        return Dataset('ESOL', smiles, labels)
    else:
        return _generate_synthetic_data('ESOL_synthetic', n_samples=500)


def load_lipophilicity() -> Dataset:
    """Load Lipophilicity dataset"""
    if HAS_DEEPCHEM:
        tasks, datasets, _ = load_lipo(featurizer='Raw', splitter=None)
        train = datasets[0]
        smiles = [x[0] for x in train.ids]
        labels = train.y.flatten()
        return Dataset('Lipophilicity', smiles, labels)
    else:
        return _generate_synthetic_data('Lipo_synthetic', n_samples=500)


def load_freesolv_data() -> Dataset:
    """Load FreeSolv (hydration free energy) dataset"""
    if HAS_DEEPCHEM:
        tasks, datasets, _ = load_freesolv(featurizer='Raw', splitter=None)
        train = datasets[0]
        smiles = [x[0] for x in train.ids]
        labels = train.y.flatten()
        return Dataset('FreeSolv', smiles, labels)
    else:
        return _generate_synthetic_data('FreeSolv_synthetic', n_samples=300)


def _generate_synthetic_data(name: str, n_samples: int) -> Dataset:
    """Generate synthetic SMILES-like data for testing when DeepChem unavailable"""
    # Simple alkane-like SMILES for testing
    np.random.seed(42)
    smiles = []
    for i in range(n_samples):
        chain_length = np.random.randint(2, 10)
        smiles.append('C' * chain_length)
    labels = np.random.randn(n_samples)
    return Dataset(name, smiles, labels)


# =============================================================================
# REPRESENTATION GENERATION
# =============================================================================

def generate_base_representations(smiles: List[str], include_maccs: bool = True) -> Dict[str, np.ndarray]:
    """
    Generate base representations for molecules.
    
    Args:
        smiles: List of SMILES strings
        include_maccs: Whether to include MACCS keys (new feature to test)
        
    Returns:
        Dict of {rep_name: features}
    """
    from kirby.representations.molecular import create_ecfp4, create_pdv, create_maccs
    
    print(f"  Generating representations for {len(smiles)} molecules...")
    
    reps = {}
    
    # ECFP4 - standard fingerprint
    t0 = time.time()
    reps['ecfp4'] = create_ecfp4(smiles)
    print(f"    ECFP4: {reps['ecfp4'].shape} in {time.time()-t0:.2f}s")
    
    # PDV - physical descriptors
    t0 = time.time()
    reps['pdv'] = create_pdv(smiles)
    print(f"    PDV: {reps['pdv'].shape} in {time.time()-t0:.2f}s")
    
    # MACCS - new representation to test
    if include_maccs:
        t0 = time.time()
        reps['maccs'] = create_maccs(smiles)
        print(f"    MACCS: {reps['maccs'].shape} in {time.time()-t0:.2f}s")
    
    return reps


def generate_augmentations(smiles: List[str]) -> Dict[str, np.ndarray]:
    """
    Generate augmentation features for molecules.
    
    Args:
        smiles: List of SMILES strings
        
    Returns:
        Dict of {aug_name: features}
    """
    from kirby.representations.molecular import (
        compute_graph_topology,
        compute_spectral_features,
        compute_subgraph_counts,
        compute_graph_distances
    )
    
    print(f"  Generating augmentations for {len(smiles)} molecules...")
    
    augs = {}
    
    t0 = time.time()
    augs['graph_topology'] = compute_graph_topology(smiles)
    print(f"    graph_topology: {augs['graph_topology'].shape} in {time.time()-t0:.2f}s")
    
    t0 = time.time()
    augs['spectral'] = compute_spectral_features(smiles, k=10)
    print(f"    spectral: {augs['spectral'].shape} in {time.time()-t0:.2f}s")
    
    t0 = time.time()
    augs['subgraph_counts'] = compute_subgraph_counts(smiles)
    print(f"    subgraph_counts: {augs['subgraph_counts'].shape} in {time.time()-t0:.2f}s")
    
    t0 = time.time()
    augs['graph_distances'] = compute_graph_distances(smiles)
    print(f"    graph_distances: {augs['graph_distances'].shape} in {time.time()-t0:.2f}s")
    
    return augs


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
    model_type: str = 'rf'
) -> Dict[str, float]:
    """
    Train and evaluate a model.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Labels
        model_type: 'rf' for RandomForest, 'xgb' for XGBoost
        
    Returns:
        Dict with 'r2', 'rmse', 'train_time'
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    t0 = time.time()
    
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'xgb':
        if not HAS_XGBOOST:
            return {'r2': np.nan, 'rmse': np.nan, 'train_time': 0}
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - t0
    
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {'r2': r2, 'rmse': rmse, 'train_time': train_time}


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_maccs_contribution(dataset: Dataset, test_size: float = 0.2) -> Dict:
    """
    Test whether MACCS improves hybrid performance.
    
    Compares:
    - Baseline: ECFP4 + PDV
    - With MACCS: ECFP4 + PDV + MACCS
    """
    from kirby.hybrid import create_hybrid, apply_feature_selection
    
    print(f"\n{'='*60}")
    print(f"Testing MACCS contribution on {dataset.name}")
    print(f"{'='*60}")
    
    # Split data
    train_smiles, test_smiles, train_labels, test_labels = train_test_split(
        dataset.smiles, dataset.labels, test_size=test_size, random_state=42
    )
    
    results = {}
    
    # Generate representations
    print("\n[Baseline: ECFP4 + PDV]")
    base_train = generate_base_representations(train_smiles, include_maccs=False)
    base_test = generate_base_representations(test_smiles, include_maccs=False)
    
    # Create hybrid without MACCS
    hybrid_train, feature_info = create_hybrid(
        base_train, train_labels,
        allocation_method='greedy',
        budget=100
    )
    hybrid_test = apply_feature_selection(base_test, feature_info)
    
    for model_type in ['rf', 'xgb']:
        if model_type == 'xgb' and not HAS_XGBOOST:
            continue
        metrics = evaluate_model(hybrid_train, hybrid_test, train_labels, test_labels, model_type)
        results[f'baseline_{model_type}'] = metrics
        print(f"  {model_type.upper()}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # Generate representations with MACCS
    print("\n[With MACCS: ECFP4 + PDV + MACCS]")
    maccs_train = generate_base_representations(train_smiles, include_maccs=True)
    maccs_test = generate_base_representations(test_smiles, include_maccs=True)
    
    # Create hybrid with MACCS
    hybrid_train, feature_info = create_hybrid(
        maccs_train, train_labels,
        allocation_method='greedy',
        budget=100
    )
    hybrid_test = apply_feature_selection(maccs_test, feature_info)
    
    for model_type in ['rf', 'xgb']:
        if model_type == 'xgb' and not HAS_XGBOOST:
            continue
        metrics = evaluate_model(hybrid_train, hybrid_test, train_labels, test_labels, model_type)
        results[f'with_maccs_{model_type}'] = metrics
        print(f"  {model_type.upper()}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    # Summary
    print("\n[MACCS Impact Summary]")
    for model_type in ['rf', 'xgb']:
        if model_type == 'xgb' and not HAS_XGBOOST:
            continue
        baseline_r2 = results[f'baseline_{model_type}']['r2']
        maccs_r2 = results[f'with_maccs_{model_type}']['r2']
        improvement = maccs_r2 - baseline_r2
        status = "✓ IMPROVED" if improvement > 0.005 else ("≈ NEUTRAL" if improvement > -0.005 else "✗ WORSE")
        print(f"  {model_type.upper()}: {improvement:+.4f} R² ({status})")
    
    return results


def test_augmentation_strategies(
    dataset: Dataset, 
    test_size: float = 0.2,
    threshold: Optional[float] = None
) -> Dict:
    """
    Test different augmentation strategies.
    
    Compares:
    - No augmentation
    - All augmentations
    - Greedy ablation (keep only helpful ones)
    """
    from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection
    
    print(f"\n{'='*60}")
    print(f"Testing Augmentation Strategies on {dataset.name}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}")
    
    # Split data
    train_smiles, test_smiles, train_labels, test_labels = train_test_split(
        dataset.smiles, dataset.labels, test_size=test_size, random_state=42
    )
    
    # Generate base representations (with MACCS)
    print("\n[Generating base representations]")
    base_train = generate_base_representations(train_smiles, include_maccs=True)
    base_test = generate_base_representations(test_smiles, include_maccs=True)
    
    # Generate augmentations
    print("\n[Generating augmentations]")
    aug_train = generate_augmentations(train_smiles)
    aug_test = generate_augmentations(test_smiles)
    
    results = {}
    
    # Strategy 1: No augmentation
    print("\n[Strategy: none]")
    hybrid_train, feature_info = create_hybrid(
        base_train, train_labels,
        allocation_method='greedy',
        budget=100,
        augmentations=aug_train,
        augmentation_strategy='none'
    )
    hybrid_test = apply_feature_selection(base_test, feature_info)
    
    for model_type in ['rf', 'xgb']:
        if model_type == 'xgb' and not HAS_XGBOOST:
            continue
        metrics = evaluate_model(hybrid_train, hybrid_test, train_labels, test_labels, model_type)
        results[f'none_{model_type}'] = metrics
        print(f"  {model_type.upper()}: R²={metrics['r2']:.4f} (dims={hybrid_train.shape[1]})")
    
    # Strategy 2: All augmentations
    print("\n[Strategy: all]")
    hybrid_train, feature_info = create_hybrid(
        base_train, train_labels,
        allocation_method='greedy',
        budget=100,
        augmentations=aug_train,
        augmentation_strategy='all',
        augmentation_budget=20
    )
    hybrid_test = apply_feature_selection(base_test, feature_info)
    if 'augmentation_info' in feature_info:
        aug_features = apply_augmentation_selection(aug_test, feature_info['augmentation_info'])
        if aug_features is not None:
            hybrid_test = np.hstack([hybrid_test, aug_features])
    
    for model_type in ['rf', 'xgb']:
        if model_type == 'xgb' and not HAS_XGBOOST:
            continue
        metrics = evaluate_model(hybrid_train, hybrid_test, train_labels, test_labels, model_type)
        results[f'all_{model_type}'] = metrics
        print(f"  {model_type.upper()}: R²={metrics['r2']:.4f} (dims={hybrid_train.shape[1]})")
    
    # Strategy 3: Greedy ablation
    print("\n[Strategy: greedy_ablation]")
    hybrid_train, feature_info = create_hybrid(
        base_train, train_labels,
        allocation_method='greedy',
        budget=100,
        augmentations=aug_train,
        augmentation_strategy='greedy_ablation',
        augmentation_threshold=threshold,
        augmentation_budget=20
    )
    hybrid_test = apply_feature_selection(base_test, feature_info)
    if 'augmentation_info' in feature_info and feature_info['augmentation_info']:
        aug_features = apply_augmentation_selection(aug_test, feature_info['augmentation_info'])
        if aug_features is not None:
            hybrid_test = np.hstack([hybrid_test, aug_features])
    
    kept = feature_info.get('kept_augmentations', [])
    print(f"  Kept augmentations: {kept if kept else 'none'}")
    
    if 'augmentation_scores' in feature_info:
        print("  Augmentation scores:")
        for aug_name, scores in feature_info['augmentation_scores'].items():
            status = "✓ kept" if aug_name in kept else "✗ dropped"
            print(f"    {aug_name}: improvement={scores['improvement']:+.4f} ({status})")
    
    for model_type in ['rf', 'xgb']:
        if model_type == 'xgb' and not HAS_XGBOOST:
            continue
        metrics = evaluate_model(hybrid_train, hybrid_test, train_labels, test_labels, model_type)
        results[f'greedy_{model_type}'] = metrics
        print(f"  {model_type.upper()}: R²={metrics['r2']:.4f} (dims={hybrid_train.shape[1]})")
    
    # Summary
    print("\n[Augmentation Strategy Summary]")
    for model_type in ['rf', 'xgb']:
        if model_type == 'xgb' and not HAS_XGBOOST:
            continue
        none_r2 = results[f'none_{model_type}']['r2']
        all_r2 = results[f'all_{model_type}']['r2']
        greedy_r2 = results[f'greedy_{model_type}']['r2']
        
        best = max([('none', none_r2), ('all', all_r2), ('greedy', greedy_r2)], key=lambda x: x[1])
        
        print(f"  {model_type.upper()}:")
        print(f"    none:   {none_r2:.4f}")
        print(f"    all:    {all_r2:.4f} ({all_r2 - none_r2:+.4f} vs none)")
        print(f"    greedy: {greedy_r2:.4f} ({greedy_r2 - none_r2:+.4f} vs none)")
        print(f"    BEST:   {best[0]} ({best[1]:.4f})")
    
    results['kept_augmentations'] = kept
    results['feature_info'] = feature_info
    
    return results


def test_threshold_sweep(dataset: Dataset, test_size: float = 0.2) -> Dict:
    """
    Test different augmentation threshold values.
    
    Sweeps through threshold values: [None (any improvement), 0.001, 0.005, 0.01, 0.02, 0.05]
    """
    print(f"\n{'='*60}")
    print(f"Threshold Sweep on {dataset.name}")
    print(f"{'='*60}")
    
    thresholds = [None, 0.001, 0.005, 0.01, 0.02, 0.05]
    
    results = {}
    
    for threshold in thresholds:
        threshold_name = str(threshold) if threshold is not None else 'any'
        print(f"\n--- Threshold: {threshold_name} ---")
        
        result = test_augmentation_strategies(dataset, test_size, threshold)
        results[threshold_name] = result
    
    # Summary table
    print(f"\n{'='*60}")
    print("THRESHOLD SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Threshold':<12} {'RF R²':<10} {'Kept Augs'}")
    print("-" * 40)
    
    for threshold in thresholds:
        threshold_name = str(threshold) if threshold is not None else 'any'
        r2 = results[threshold_name]['greedy_rf']['r2']
        kept = results[threshold_name]['kept_augmentations']
        kept_str = ', '.join(kept) if kept else 'none'
        print(f"{threshold_name:<12} {r2:.4f}     {kept_str}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def run_quick_test():
    """Quick smoke test to verify everything works"""
    print("\n" + "="*60)
    print("QUICK SMOKE TEST")
    print("="*60)
    
    # Use smallest dataset
    dataset = load_freesolv_data()
    
    # Limit to 200 samples for speed
    if len(dataset.smiles) > 200:
        indices = np.random.RandomState(42).choice(len(dataset.smiles), 200, replace=False)
        dataset = Dataset(
            dataset.name + '_subset',
            [dataset.smiles[i] for i in indices],
            dataset.labels[indices]
        )
    
    print(f"\nDataset: {dataset.name} ({len(dataset.smiles)} samples)")
    
    # Test MACCS
    test_maccs_contribution(dataset)
    
    # Test augmentation (one strategy only)
    test_augmentation_strategies(dataset, threshold=None)
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETE")
    print("="*60)


def run_full_test():
    """Run full test suite"""
    print("\n" + "="*60)
    print("FULL TEST SUITE")
    print("="*60)
    
    datasets = [load_esol(), load_lipophilicity()]
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset.name} ({len(dataset.smiles)} samples)")
        print(f"{'#'*60}")
        
        # Test MACCS contribution
        maccs_results = test_maccs_contribution(dataset)
        all_results[f'{dataset.name}_maccs'] = maccs_results
        
        # Test augmentation strategies
        aug_results = test_augmentation_strategies(dataset, threshold=None)
        all_results[f'{dataset.name}_augmentations'] = aug_results
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print("\n[MACCS Impact Across Datasets]")
    for dataset in datasets:
        key = f'{dataset.name}_maccs'
        if key in all_results:
            baseline = all_results[key].get('baseline_rf', {}).get('r2', np.nan)
            with_maccs = all_results[key].get('with_maccs_rf', {}).get('r2', np.nan)
            improvement = with_maccs - baseline
            print(f"  {dataset.name}: {improvement:+.4f} R²")
    
    print("\n[Best Augmentation Strategy Across Datasets]")
    for dataset in datasets:
        key = f'{dataset.name}_augmentations'
        if key in all_results:
            none_r2 = all_results[key].get('none_rf', {}).get('r2', np.nan)
            greedy_r2 = all_results[key].get('greedy_rf', {}).get('r2', np.nan)
            kept = all_results[key].get('kept_augmentations', [])
            print(f"  {dataset.name}: greedy improvement={greedy_r2 - none_r2:+.4f}, kept={kept}")
    
    return all_results


def run_augmentation_focus():
    """Focus specifically on augmentation testing"""
    print("\n" + "="*60)
    print("AUGMENTATION FOCUS TEST")
    print("="*60)
    
    dataset = load_esol()
    
    # Test multiple thresholds
    results = test_threshold_sweep(dataset)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Hybrid Master Fast Test')
    parser.add_argument('--quick', action='store_true', help='Quick smoke test')
    parser.add_argument('--augmentations', action='store_true', help='Focus on augmentation testing')
    parser.add_argument('--threshold-sweep', action='store_true', help='Sweep augmentation thresholds')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    elif args.augmentations or args.threshold_sweep:
        run_augmentation_focus()
    else:
        run_full_test()


if __name__ == '__main__':
    main()