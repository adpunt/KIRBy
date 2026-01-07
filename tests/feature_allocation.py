#!/usr/bin/env python3
"""
Final Feature Allocation Test

Comprehensive test to determine optimal configuration:
- Filters: none vs quality_only
- Patience: None vs 3
- Budget: 50, 100, 150
- Reps: 2 (ECFP4+PDV) and 3 (ECFP4+PDV+mhggnn)

Total: 24 tests (2 reps × 2 filters × 2 patience × 3 budgets)
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
import os

# Add parent src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kirby.datasets.esol import load_esol_combined
from kirby.representations.molecular import create_ecfp4, create_pdv, create_mhg_gnn
from kirby.utils.feature_filtering import apply_filters, FILTER_CONFIGS


def create_hybrid_with_allocation(rep_dict, labels, n_per_rep_dict):
    """Create hybrid with per-rep feature allocation via RF importance"""
    selected_features = []
    feature_info_new = {}
    
    for name, X in rep_dict.items():
        X_clipped = np.clip(X, -1e10, 1e10)
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
        
        n_select = n_per_rep_dict[name]
        
        if n_select == 0:
            continue
        
        if n_select == -1 or n_select >= X_clipped.shape[1]:
            selected_features.append(X_clipped)
            feature_info_new[name] = {
                'selected_indices': np.arange(X_clipped.shape[1]),
                'n_features': X_clipped.shape[1]
            }
            continue
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clipped)
        
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_scaled, labels)
        importances = model.feature_importances_
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
    
    if len(selected_features) == 0:
        return np.zeros((len(labels), 0)), feature_info_new
    
    return np.hstack(selected_features).astype(np.float32), feature_info_new


def apply_allocation(rep_dict_test, feature_info):
    """Apply saved feature selection to test set"""
    selected_parts = []
    for rep_name, X in rep_dict_test.items():
        if rep_name not in feature_info:
            continue
        X_clipped = np.clip(X, -1e10, 1e10)
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
        
        indices = feature_info[rep_name]['selected_indices']
        X_selected = X_clipped[:, indices]
        selected_parts.append(X_selected)
    
    if len(selected_parts) == 0:
        return np.zeros((X.shape[0], 0))
    
    return np.hstack(selected_parts).astype(np.float32)


def test_single(X_train, X_test, y_train, y_test):
    """Test with RF"""
    if X_train.shape[1] == 0:
        return {'r2': -np.inf, 'mae': np.inf}
    
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


def allocate_greedy_forward(rep_dict_train, rep_dict_val, train_labels, val_labels, 
                           total_budget=100, step_size=10, patience=None):
    """Greedy forward selection with optional plateau detection"""
    allocation = {rep: 0 for rep in rep_dict_train.keys()}
    remaining_budget = total_budget
    
    best_val_r2 = -np.inf
    steps_without_improvement = 0
    
    iteration = 0
    while remaining_budget >= step_size:
        iteration += 1
        best_gain = -np.inf
        best_rep = None
        
        for rep_name in rep_dict_train.keys():
            trial_allocation = allocation.copy()
            trial_allocation[rep_name] += step_size
            
            max_features = rep_dict_train[rep_name].shape[1]
            if trial_allocation[rep_name] > max_features:
                continue
            
            X_train_trial, feat_info = create_hybrid_with_allocation(
                rep_dict_train, train_labels, trial_allocation
            )
            X_val_trial = apply_allocation(rep_dict_val, feat_info)
            
            metrics = test_single(X_train_trial, X_val_trial, train_labels, val_labels)
            
            if metrics['r2'] > best_gain:
                best_gain = metrics['r2']
                best_rep = rep_name
        
        if best_rep is None:
            break
        
        if patience is not None:
            if best_gain <= best_val_r2:
                steps_without_improvement += 1
                if steps_without_improvement >= patience:
                    break
            else:
                steps_without_improvement = 0
                best_val_r2 = best_gain
        else:
            best_val_r2 = max(best_val_r2, best_gain)
        
        allocation[best_rep] += step_size
        remaining_budget -= step_size
    
    X_train_final, feat_info = create_hybrid_with_allocation(
        rep_dict_train, train_labels, allocation
    )
    
    return allocation, feat_info


def test_configuration(rep_dict_train_raw, rep_dict_val_raw, rep_dict_test_raw,
                      train_labels, val_labels, test_labels,
                      filter_config, patience, budget):
    """Test one complete configuration"""
    
    # Apply filters if needed
    if filter_config is not None:
        rep_dict_full = {}
        for rep_name in rep_dict_train_raw.keys():
            rep_dict_full[rep_name] = np.vstack([rep_dict_train_raw[rep_name], rep_dict_val_raw[rep_name]])
        
        labels_full = np.concatenate([train_labels, val_labels])
        
        filtered_full, filtered_test = apply_filters(
            rep_dict_full, rep_dict_test_raw, labels_full, **filter_config
        )
        
        n_train = len(train_labels)
        rep_dict_train = {name: X[:n_train] for name, X in filtered_full.items()}
        rep_dict_val = {name: X[n_train:] for name, X in filtered_full.items()}
        rep_dict_test = filtered_test
    else:
        rep_dict_train = rep_dict_train_raw
        rep_dict_val = rep_dict_val_raw
        rep_dict_test = rep_dict_test_raw
    
    # Run greedy
    allocation, feat_info = allocate_greedy_forward(
        rep_dict_train, rep_dict_val, train_labels, val_labels,
        budget, step_size=10, patience=patience
    )
    
    # Test on full train+val combined
    rep_dict_full_final = {}
    for rep_name in rep_dict_train.keys():
        rep_dict_full_final[rep_name] = np.vstack([rep_dict_train[rep_name], rep_dict_val[rep_name]])
    labels_full = np.concatenate([train_labels, val_labels])
    
    X_train, feat_info_final = create_hybrid_with_allocation(
        rep_dict_full_final, labels_full, allocation
    )
    X_test = apply_allocation(rep_dict_test, feat_info_final)
    
    metrics = test_single(X_train, X_test, labels_full, test_labels)
    
    return {
        'allocation': allocation,
        'metrics': metrics,
        'n_features': sum(allocation.values())
    }


def main():
    print("="*100)
    print("FINAL FEATURE ALLOCATION TEST")
    print("="*100)
    print("\nTest Matrix:")
    print("  Reps: 2 (ECFP4+PDV), 3 (ECFP4+PDV+mhggnn)")
    print("  Filters: none, quality_only")
    print("  Patience: None, 3")
    print("  Budgets: 50, 100, 150")
    print("  Total: 24 tests")
    print("="*100)
    
    # Load data
    print("\nLoading ESOL...")
    data = load_esol_combined(splitter='scaffold')
    
    train_smiles = data['train']['smiles']
    train_labels = data['train']['labels']
    test_smiles = data['test']['smiles']
    test_labels = data['test']['labels']
    
    # Split train into train/val
    n_val = len(train_smiles) // 5
    indices = np.random.RandomState(42).permutation(len(train_smiles))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_smiles_split = [train_smiles[i] for i in train_indices]
    train_labels_split = train_labels[train_indices]
    val_smiles_split = [train_smiles[i] for i in val_indices]
    val_labels_split = train_labels[val_indices]
    
    print(f"Split: Train={len(train_smiles_split)}, Val={len(val_smiles_split)}, Test={len(test_smiles)}")
    
    # Create representations
    print("\nCreating representations...")
    start = time.time()
    
    ecfp4_train = create_ecfp4(train_smiles_split, n_bits=2048)
    ecfp4_val = create_ecfp4(val_smiles_split, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    
    pdv_train = create_pdv(train_smiles_split)
    pdv_val = create_pdv(val_smiles_split)
    pdv_test = create_pdv(test_smiles)
    
    mhggnn_train = create_mhg_gnn(train_smiles_split)
    mhggnn_val = create_mhg_gnn(val_smiles_split)
    mhggnn_test = create_mhg_gnn(test_smiles)
    
    print(f"Done ({time.time() - start:.1f}s)")
    print(f"  ECFP4:  {ecfp4_train.shape[1]} features")
    print(f"  PDV:    {pdv_train.shape[1]} features")
    print(f"  mhggnn: {mhggnn_train.shape[1]} features")
    
    # BASELINES: Test each rep individually
    print(f"\n{'='*100}")
    print("BASELINE PERFORMANCES (individual representations)")
    print(f"{'='*100}")
    
    baseline_scores = {}
    
    # Combine train+val for baseline testing (same as final hybrid testing)
    ecfp4_full = np.vstack([ecfp4_train, ecfp4_val])
    pdv_full = np.vstack([pdv_train, pdv_val])
    mhggnn_full = np.vstack([mhggnn_train, mhggnn_val])
    labels_full = np.concatenate([train_labels_split, val_labels_split])
    
    for rep_name, X_train_full, X_test_full in [
        ('ecfp4', ecfp4_full, ecfp4_test),
        ('pdv', pdv_full, pdv_test),
        ('mhggnn', mhggnn_full, mhggnn_test)
    ]:
        print(f"\n{rep_name} ({X_train_full.shape[1]} features):")
        
        metrics = test_single(X_train_full, X_test_full, labels_full, test_labels)
        
        baseline_scores[rep_name] = metrics
        print(f"  R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
    
    best_baseline_rep = max(baseline_scores.items(), key=lambda x: x[1]['r2'])[0]
    best_baseline_r2 = baseline_scores[best_baseline_rep]['r2']
    
    print(f"\n{'='*100}")
    print(f"Best baseline: {best_baseline_rep} (R²={best_baseline_r2:.4f})")
    print(f"{'='*100}")
    
    # Test configurations
    filters_to_test = {
        'none': None,
        'quality_only': FILTER_CONFIGS['quality_only']
    }
    
    patience_values = [None, 3]
    budgets = [50, 100, 150]
    
    results = {}
    
    # 2 REPS
    print(f"\n{'='*100}")
    print("TESTING: 2 REPS (ECFP4 + PDV)")
    print(f"{'='*100}")
    
    rep_dict_train_2 = {'ecfp4': ecfp4_train, 'pdv': pdv_train}
    rep_dict_val_2 = {'ecfp4': ecfp4_val, 'pdv': pdv_val}
    rep_dict_test_2 = {'ecfp4': ecfp4_test, 'pdv': pdv_test}
    
    results['2rep'] = {}
    
    for filter_name, filter_config in filters_to_test.items():
        for patience in patience_values:
            for budget in budgets:
                patience_str = "no_patience" if patience is None else f"patience_{patience}"
                config_name = f"{filter_name}_{patience_str}_budget_{budget}"
                
                print(f"\n  [{config_name}]")
                
                result = test_configuration(
                    rep_dict_train_2, rep_dict_val_2, rep_dict_test_2,
                    train_labels_split, val_labels_split, test_labels,
                    filter_config, patience, budget
                )
                
                results['2rep'][config_name] = result
                
                print(f"    R²={result['metrics']['r2']:.4f}, "
                      f"features={result['n_features']}/{budget}, "
                      f"allocation={result['allocation']}")
    
    # 3 REPS
    print(f"\n{'='*100}")
    print("TESTING: 3 REPS (ECFP4 + PDV + mhggnn)")
    print(f"{'='*100}")
    
    rep_dict_train_3 = {'ecfp4': ecfp4_train, 'pdv': pdv_train, 'mhggnn': mhggnn_train}
    rep_dict_val_3 = {'ecfp4': ecfp4_val, 'pdv': pdv_val, 'mhggnn': mhggnn_val}
    rep_dict_test_3 = {'ecfp4': ecfp4_test, 'pdv': pdv_test, 'mhggnn': mhggnn_test}
    
    results['3rep'] = {}
    
    for filter_name, filter_config in filters_to_test.items():
        for patience in patience_values:
            for budget in budgets:
                patience_str = "no_patience" if patience is None else f"patience_{patience}"
                config_name = f"{filter_name}_{patience_str}_budget_{budget}"
                
                print(f"\n  [{config_name}]")
                
                result = test_configuration(
                    rep_dict_train_3, rep_dict_val_3, rep_dict_test_3,
                    train_labels_split, val_labels_split, test_labels,
                    filter_config, patience, budget
                )
                
                results['3rep'][config_name] = result
                
                print(f"    R²={result['metrics']['r2']:.4f}, "
                      f"features={result['n_features']}/{budget}, "
                      f"allocation={result['allocation']}")
    
    # SUMMARY
    print(f"\n\n{'='*100}")
    print("SUMMARY - BEST CONFIGURATIONS")
    print(f"{'='*100}")
    
    # Find best for 2-rep
    best_2rep = max(results['2rep'].items(), key=lambda x: x[1]['metrics']['r2'])
    print(f"\nBEST 2-REP: {best_2rep[0]}")
    print(f"  R²={best_2rep[1]['metrics']['r2']:.4f}")
    print(f"  Allocation: {best_2rep[1]['allocation']}")
    print(f"  Features used: {best_2rep[1]['n_features']}")
    improvement_2rep = ((best_2rep[1]['metrics']['r2'] - best_baseline_r2) / best_baseline_r2) * 100
    print(f"  vs best baseline ({best_baseline_rep}): {improvement_2rep:+.2f}%")
    
    # Find best for 3-rep
    best_3rep = max(results['3rep'].items(), key=lambda x: x[1]['metrics']['r2'])
    print(f"\nBEST 3-REP: {best_3rep[0]}")
    print(f"  R²={best_3rep[1]['metrics']['r2']:.4f}")
    print(f"  Allocation: {best_3rep[1]['allocation']}")
    print(f"  Features used: {best_3rep[1]['n_features']}")
    improvement_3rep = ((best_3rep[1]['metrics']['r2'] - best_baseline_r2) / best_baseline_r2) * 100
    print(f"  vs best baseline ({best_baseline_rep}): {improvement_3rep:+.2f}%")
    
    # Detailed tables
    print(f"\n\n{'='*100}")
    print("DETAILED RESULTS")
    print(f"{'='*100}")
    
    for rep_type in ['2rep', '3rep']:
        print(f"\n{rep_type.upper()}:")
        print(f"  {'Config':<50} {'R²':>8} {'Features':>10} {'Allocation'}")
        print(f"  {'-'*100}")
        
        # Sort by R²
        sorted_results = sorted(results[rep_type].items(), 
                               key=lambda x: x[1]['metrics']['r2'], 
                               reverse=True)
        
        for config_name, result in sorted_results:
            r2 = result['metrics']['r2']
            n_feat = result['n_features']
            alloc = result['allocation']
            print(f"  {config_name:<50} {r2:>8.4f} {n_feat:>10} {alloc}")
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()