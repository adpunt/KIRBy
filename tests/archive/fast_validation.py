import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
Fast Validation Script: NoiseInject Pipeline Test
==================================================

Minimal test to verify the entire pipeline works before running full experiments.

Configuration:
- Dataset: QM9 (n=1000 samples only)
- Representation: ECFP4 only
- Model: Random Forest only
- Strategy: legacy only
- Noise levels: σ ∈ {0.0, 0.3, 0.6} (3 levels)
- No repetitions

Expected runtime: ~2-5 minutes

Usage:
    python fast_validation.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import sys
import time

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.representations.molecular import create_ecfp4

# NoiseInject imports
from noiseInject import (
    NoiseInjectorRegression,
    calculate_noise_metrics
)


def main():
    start_time = time.time()
    
    print("="*80)
    print("FAST VALIDATION - NoiseInject Pipeline Test")
    print("="*80)
    
    # Configuration
    n_samples = 1000
    strategy = 'legacy'
    sigma_levels = [0.0, 0.3, 0.6]  # Just 3 levels for speed
    results_dir = Path(__file__).parent.parent / 'results' / 'validation'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print(f"\n[1/5] Loading QM9 (n={n_samples})...")
    raw_data = load_qm9(n_samples=n_samples, property_idx=4)
    
    print("[2/5] Creating scaffold splits...")
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    train_smiles = splits['train']['smiles']
    train_labels = np.array(splits['train']['labels'])
    test_smiles = splits['test']['smiles']
    test_labels = np.array(splits['test']['labels'])
    
    print(f"      Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    
    # =========================================================================
    # CREATE REPRESENTATION
    # =========================================================================
    print("[3/5] Creating ECFP4 fingerprints...")
    ecfp4_train = create_ecfp4(train_smiles, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    print(f"      Shape: train={ecfp4_train.shape}, test={ecfp4_test.shape}")
    
    # =========================================================================
    # RUN NOISE EXPERIMENT
    # =========================================================================
    print(f"[4/5] Running noise experiment (strategy={strategy})...")
    
    # Run experiment
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions = {}
    
    for sigma in sigma_levels:
        print(f"      σ={sigma:.1f}...", end='')
        
        # Inject noise
        if sigma == 0.0:
            y_noisy = train_labels
        else:
            y_noisy = injector.inject(train_labels, sigma)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(ecfp4_train, y_noisy)
        
        # Predict
        predictions[sigma] = model.predict(ecfp4_test)
        print(" done")
    
    # =========================================================================
    # CALCULATE METRICS
    # =========================================================================
    print("[5/5] Calculating metrics...")
    
    per_sigma_df, summary_df = calculate_noise_metrics(
        test_labels, 
        predictions,
        metrics=['r2', 'rmse', 'mae']
    )
    
    # Add metadata
    per_sigma_df['model'] = 'RF'
    per_sigma_df['rep'] = 'ECFP4'
    per_sigma_df['strategy'] = strategy
    
    # Save results
    per_sigma_df.to_csv(results_dir / 'per_sigma.csv', index=False)
    summary_df.to_csv(results_dir / 'summary.csv', index=False)
    
    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\nPer-sigma performance:")
    print(per_sigma_df[['sigma', 'r2', 'rmse', 'mae']].to_string(index=False))
    
    print("\nSummary statistics:")
    summary = summary_df.iloc[0]
    print(f"  Baseline R²:     {summary['baseline_r2']:.4f}")
    print(f"  NSI (R²):        {summary['nsi_r2']:.4f}")
    print(f"  Retention:       {summary['retention_pct_r2']:.2f}%")
    print(f"  P-value (NSI):   {summary['nsi_r2_pval']:.4f}")
    
    # Calculate manual NSI for verification (using σ_max = 0.6)
    baseline = per_sigma_df[per_sigma_df['sigma'] == 0.0]['r2'].values[0]
    high_noise = per_sigma_df[per_sigma_df['sigma'] == 0.6]['r2'].values[0]
    manual_nsi = (baseline - high_noise) / 0.6
    manual_retention = (high_noise / baseline) * 100 if baseline > 0 else 0
    
    print("\nManual verification (σ=0 to σ=0.6):")
    print(f"  NSI:             {manual_nsi:.4f}")
    print(f"  Retention:       {manual_retention:.2f}%")
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Total runtime: {elapsed:.1f} seconds")
    print(f"Results saved to: {results_dir}/")
    
    # Check if results make sense
    print("\n" + "="*80)
    print("SANITY CHECKS")
    print("="*80)
    
    checks_passed = True
    
    # Check 1: Baseline R² should be reasonable (> 0.5 for QM9)
    if baseline > 0.5:
        print("✓ Baseline R² > 0.5")
    else:
        print("✗ Baseline R² too low")
        checks_passed = False
    
    # Check 2: Performance should degrade with noise
    if high_noise < baseline:
        print("✓ Performance degrades with noise")
    else:
        print("✗ Performance does not degrade")
        checks_passed = False
    
    # Check 3: NSI should be positive
    if manual_nsi > 0:
        print("✓ NSI > 0 (performance degrades)")
    else:
        print("✗ NSI ≤ 0")
        checks_passed = False
    
    # Check 4: Retention should be reasonable (40-95%)
    if 40 <= manual_retention <= 95:
        print(f"✓ Retention in reasonable range ({manual_retention:.1f}%)")
    else:
        print(f"✗ Retention outside expected range ({manual_retention:.1f}%)")
        checks_passed = False
    
    print("\n" + "="*80)
    if checks_passed:
        print("ALL CHECKS PASSED - Pipeline is working correctly!")
        print("You can now run the full experiments.")
    else:
        print("SOME CHECKS FAILED - Please investigate before running full experiments.")
    print("="*80)


if __name__ == '__main__':
    main()
