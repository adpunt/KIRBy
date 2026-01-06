#!/usr/bin/env python3
"""
Test Feature Allocation Strategies

Focused test: QM9 only, ecfp4+pdv only, budget=400
Tests 3 allocation strategies, shows clear winner.

Once winner is identified:
1. Put winner in kirby/utils/
2. Update hybrid_master_fast.py to use it
3. Delete this script and the losers
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time

import sys
sys.path.insert(0, '/mnt/user-data/outputs/src')

from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.representations.molecular import create_ecfp4, create_pdv


def create_hybrid_with_allocation(rep_dict, labels, n_per_rep_dict, selection_method='tree_importance'):
    """
    Create hybrid with PER-REP feature allocation.
    
    Args:
        rep_dict: {rep_name: features}
        labels: Target labels
        n_per_rep_dict: {rep_name: n_features} - DIFFERENT per rep
        selection_method: 'tree_importance'
        
    Returns:
        hybrid_features, feature_info
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    is_classification = len(np.unique(labels)) < 10
    selected_features = []
    feature_info_new = {}
    
    for name, X in rep_dict.items():
        # Clean data
        X_clipped = np.clip(X, -1e10, 1e10)
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Get n_per_rep for THIS rep
        n_select = n_per_rep_dict[name]
        
        # If -1, take all
        if n_select == -1:
            selected_features.append(X_clipped)
            feature_info_new[name] = {
                'selected_indices': np.arange(X_clipped.shape[1]),
                'n_features': X_clipped.shape[1]
            }
            continue
        
        # Feature selection by importance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clipped)
        
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_scaled, labels)
        importances = model.feature_importances_
        
        if np.any(np.isnan(importances)) or np.any(np.isinf(importances)):
            importances = np.nan_to_num(importances, nan=0.0)
        
        n_select = min(n_select, X_clipped.shape[1])
        top_idx = np.argsort(importances)[-n_select:][::-1]
        
        X_selected = X_clipped[:, top_idx]
        selected_features.append(X_selected)
        
        feature_info_new[name] = {
            'selected_indices': top_idx,
            'importance_scores': importances[top_idx],
            'n_features': n_select
        }
    
    return np.hstack(selected_features).astype(np.float32), feature_info_new


def apply_allocation(rep_dict_test, feature_info):
    """Apply saved feature selection to test set"""
    selected_parts = []
    for rep_name, X in rep_dict_test.items():
        X_clipped = np.clip(X, -1e10, 1e10)
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
        
        indices = feature_info[rep_name]['selected_indices']
        X_selected = X_clipped[:, indices]
        selected_parts.append(X_selected)
    
    return np.hstack(selected_parts).astype(np.float32)


def test_single(X_train, X_test, y_train, y_test):
    """Test single representation with RF"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    return {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred)
    }


def main():
    print("="*100)
    print("FEATURE ALLOCATION STRATEGY TEST")
    print("="*100)
    print("\nDataset: QM9 HOMO-LUMO Gap")
    print("Reps: ecfp4 + pdv")
    print("Budget: 400 features")
    print("Strategies: equal, performance_proportional, rank_based")
    print("="*100)
    
    # Load QM9
    print("\nLoading QM9 (5000 samples)...")
    raw_data = load_qm9(n_samples=5000, property_idx=4)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    train_smiles = splits['train']['smiles'] + splits['val']['smiles']
    train_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
    test_smiles = splits['test']['smiles']
    test_labels = splits['test']['labels']
    
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    
    # Create representations
    print("\nCreating representations...")
    start = time.time()
    ecfp4_train = create_ecfp4(train_smiles, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    pdv_train = create_pdv(train_smiles)
    pdv_test = create_pdv(test_smiles)
    print(f"Done ({time.time() - start:.2f}s)")
    print(f"  ecfp4: {ecfp4_train.shape[1]} features")
    print(f"  pdv:   {pdv_train.shape[1]} features")
    
    # Test baselines
    print(f"\n{'BASELINE PERFORMANCES':=^100}")
    
    print("\necfp4:")
    ecfp4_metrics = test_single(ecfp4_train, ecfp4_test, train_labels, test_labels)
    print(f"  R²={ecfp4_metrics['r2']:.4f}, MAE={ecfp4_metrics['mae']:.4f}")
    
    print("\npdv:")
    pdv_metrics = test_single(pdv_train, pdv_test, train_labels, test_labels)
    print(f"  R²={pdv_metrics['r2']:.4f}, MAE={pdv_metrics['mae']:.4f}")
    
    rep_performances = {
        'ecfp4': ecfp4_metrics['r2'],
        'pdv': pdv_metrics['r2']
    }
    
    # Test allocations
    print(f"\n{'ALLOCATION STRATEGIES (Budget=400)':=^100}")
    
    total_budget = 400
    results = []
    
    # Strategy 1: Equal
    print("\n[1/3] EQUAL ALLOCATION")
    n_per_rep_equal = {'ecfp4': 200, 'pdv': 200}
    print(f"  Allocation: ecfp4={n_per_rep_equal['ecfp4']}, pdv={n_per_rep_equal['pdv']}")
    
    rep_dict_train = {'ecfp4': ecfp4_train, 'pdv': pdv_train}
    rep_dict_test = {'ecfp4': ecfp4_test, 'pdv': pdv_test}
    
    X_train, feat_info = create_hybrid_with_allocation(rep_dict_train, train_labels, n_per_rep_equal)
    X_test = apply_allocation(rep_dict_test, feat_info)
    
    metrics = test_single(X_train, X_test, train_labels, test_labels)
    results.append(('equal', n_per_rep_equal, metrics))
    print(f"  Result: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
    
    # Strategy 2: Performance-Proportional
    print("\n[2/3] PERFORMANCE-PROPORTIONAL ALLOCATION")
    total_perf = rep_performances['ecfp4'] + rep_performances['pdv']
    n_ecfp4 = int((rep_performances['ecfp4'] / total_perf) * total_budget)
    n_pdv = total_budget - n_ecfp4
    n_per_rep_prop = {'ecfp4': n_ecfp4, 'pdv': n_pdv}
    print(f"  Allocation: ecfp4={n_per_rep_prop['ecfp4']}, pdv={n_per_rep_prop['pdv']}")
    
    X_train, feat_info = create_hybrid_with_allocation(rep_dict_train, train_labels, n_per_rep_prop)
    X_test = apply_allocation(rep_dict_test, feat_info)
    
    metrics = test_single(X_train, X_test, train_labels, test_labels)
    results.append(('performance_proportional', n_per_rep_prop, metrics))
    print(f"  Result: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
    
    # Strategy 3: Rank-Based (70/30)
    print("\n[3/3] RANK-BASED ALLOCATION (70/30)")
    # Determine which is better
    if rep_performances['ecfp4'] > rep_performances['pdv']:
        n_per_rep_rank = {'ecfp4': 280, 'pdv': 120}
    else:
        n_per_rep_rank = {'ecfp4': 120, 'pdv': 280}
    print(f"  Allocation: ecfp4={n_per_rep_rank['ecfp4']}, pdv={n_per_rep_rank['pdv']}")
    
    X_train, feat_info = create_hybrid_with_allocation(rep_dict_train, train_labels, n_per_rep_rank)
    X_test = apply_allocation(rep_dict_test, feat_info)
    
    metrics = test_single(X_train, X_test, train_labels, test_labels)
    results.append(('rank_based', n_per_rep_rank, metrics))
    print(f"  Result: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
    
    # Summary
    print(f"\n{'SUMMARY':=^100}")
    
    print(f"\n{'Strategy':<30} {'Allocation':^30} {'R²':>10} {'MAE':>10}")
    print("-"*100)
    
    for strategy, allocation, metrics in results:
        alloc_str = f"ecfp4={allocation['ecfp4']}, pdv={allocation['pdv']}"
        print(f"{strategy:<30} {alloc_str:^30} {metrics['r2']:>10.4f} {metrics['mae']:>10.4f}")
    
    # Find winner
    best_idx = np.argmax([r[2]['r2'] for r in results])
    winner_strategy, winner_alloc, winner_metrics = results[best_idx]
    
    print("\n" + "="*100)
    print(f"WINNER: {winner_strategy.upper()}")
    print(f"  Allocation: {winner_alloc}")
    print(f"  R²={winner_metrics['r2']:.4f}, MAE={winner_metrics['mae']:.4f}")
    
    # Compare to best baseline
    best_baseline_r2 = max(rep_performances.values())
    improvement = ((winner_metrics['r2'] - best_baseline_r2) / best_baseline_r2) * 100
    
    print(f"\nImprovement over best baseline: {improvement:+.2f}%")
    print("="*100)
    
    print(f"\nNext steps:")
    print(f"1. If winner is good, implement in kirby/utils/")
    print(f"2. Update hybrid_master_fast.py to use winner")
    print(f"3. Delete this test script and losing strategies")


if __name__ == '__main__':
    main()