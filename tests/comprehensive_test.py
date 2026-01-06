#!/usr/bin/env python3
"""
KIRBy Hybrid Master - COMPREHENSIVE EVALUATION VERSION (~10-15 min)

Complete evaluation across all datasets and configurations.

Tests:
1. QM9 HOMO-LUMO gap (scaffold split) - PRIMARY
2. ESOL solubility (regression)
3. hERG cardiotoxicity (classification)

All with:
- All baseline reps (ECFP4, PDV, mol2vec, mhggnn)
- Multiple models (RF, XGBoost, MLP)
- Full n_per_rep sweep [25, 50, 100, 200, 500, 1000, -1]

Use this for: Final evaluation before thesis, comprehensive comparisons
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import time
import json
from datetime import datetime

import sys
sys.path.insert(0, '/mnt/user-data/outputs/src')

from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.datasets.esol import load_esol_combined
from kirby.datasets.herg import load_herg
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_mol2vec,
    create_mhg_gnn
)


def create_hybrid_wrapper(rep_dict, labels, n_per_rep, selection_method='tree_importance',
                         feature_info=None):
    """Wrapper for feature selection with train/test alignment"""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Keep everything case
    if n_per_rep == -1:
        all_features = []
        for name, X in rep_dict.items():
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
            all_features.append(X_clipped)
        result = np.hstack(all_features)
        return result, feature_info
    
    # Clean the data
    clean_reps = {}
    for name, X in rep_dict.items():
        X_clipped = np.clip(X, -1e10, 1e10)
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
        clean_reps[name] = X_clipped
    
    # If feature_info provided, apply existing selection (for test set)
    if feature_info is not None:
        selected_parts = []
        for rep_name in clean_reps.keys():
            X = clean_reps[rep_name]
            indices = feature_info[rep_name]['selected_indices']
            X_selected = X[:, indices]
            selected_parts.append(X_selected)
        return np.hstack(selected_parts).astype(np.float32), feature_info
    
    # Compute new selection (for train set)
    is_classification = len(np.unique(labels)) < 10
    selected_features = []
    feature_info_new = {}
    
    for name, X in clean_reps.items():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        
        model.fit(X_scaled, labels)
        importances = model.feature_importances_
        
        if np.any(np.isnan(importances)) or np.any(np.isinf(importances)):
            importances = np.nan_to_num(importances, nan=0.0)
        
        n_select = min(n_per_rep, X.shape[1])
        top_idx = np.argsort(importances)[-n_select:][::-1]
        
        X_selected = X[:, top_idx]
        selected_features.append(X_selected)
        
        feature_info_new[name] = {
            'selected_indices': top_idx,
            'importance_scores': importances[top_idx],
            'n_features': n_select
        }
    
    return np.hstack(selected_features).astype(np.float32), feature_info_new


def evaluate_regression(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }


def evaluate_classification(y_true, y_pred, y_prob):
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred, average='binary')
    }


def test_single(X_train, X_test, y_train, y_test, model, is_classification, model_name=''):
    """Test single representation with appropriate scaler"""
    
    # Check for NaN/inf - crash if found
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError(f"NaN/inf in X_train for {model_name}")
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        raise ValueError(f"NaN/inf in X_test for {model_name}")
    
    # Use RobustScaler for MLP, StandardScaler for others
    if 'MLP' in model_name:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Check predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError(f"NaN/inf in predictions for {model_name}")
    
    if is_classification:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        return evaluate_classification(y_test, y_pred, y_prob)
    else:
        return evaluate_regression(y_test, y_pred)


def run_comprehensive_evaluation(n_per_rep_values=[25, 50, 100, 200, 500, 1000, -1]):
    """Comprehensive evaluation across all datasets"""
    
    print("="*100)
    print("KIRBY COMPREHENSIVE EVALUATION")
    print("="*100)
    print("\nDatasets:")
    print("  1. QM9 HOMO-LUMO gap (scaffold split) - PRIMARY")
    print("  2. ESOL solubility (scaffold split)")
    print("  3. hERG cardiotoxicity (FLuID)")
    print("\nModels: RF, XGBoost, MLP")
    print(f"n_per_rep values: {n_per_rep_values}")
    print("="*100)
    
    results = []
    overall_start = time.time()
    
    # ========================================================================
    # QM9 HOMO-LUMO GAP - PRIMARY TEST
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET 1: QM9 HOMO-LUMO Gap (Scaffold Split)")
    print(f"{'='*80}")
    
    raw_data = load_qm9(n_samples=2000, property_idx=4)  # HOMO-LUMO gap
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    train_smiles_full = splits['train']['smiles'] + splits['val']['smiles']
    train_labels_full = np.concatenate([splits['train']['labels'], splits['val']['labels']])
    test_smiles = splits['test']['smiles']
    test_labels = splits['test']['labels']
    
    print(f"Target: {raw_data['property_name']} (eV)")
    print(f"Train: {len(train_smiles_full)}, Test: {len(test_smiles)}")
    
    # Subsample for efficiency
    n_baseline = 1000
    print(f"\nSubsampling to n={n_baseline}...")
    indices = np.random.choice(len(train_labels_full), 
                              min(n_baseline, len(train_labels_full)), 
                              replace=False)
    train_smiles = [train_smiles_full[i] for i in indices]
    train_labels = train_labels_full[indices]
    
    # Create representations
    print("Creating representations...")
    start = time.time()
    reps_train = {
        'ecfp4': create_ecfp4(train_smiles, n_bits=2048),
        'pdv': create_pdv(train_smiles),
        'mol2vec': create_mol2vec(train_smiles),
        'mhggnn': create_mhg_gnn(train_smiles)
    }
    reps_test = {
        'ecfp4': create_ecfp4(test_smiles, n_bits=2048),
        'pdv': create_pdv(test_smiles),
        'mol2vec': create_mol2vec(test_smiles),
        'mhggnn': create_mhg_gnn(test_smiles)
    }
    print(f"Representations created ({time.time() - start:.2f}s)")
    
    # STEP 1: Baseline with all models
    print(f"\n{'STEP 1: BASELINE (all reps with RF, XGBoost, MLP)':=^80}")
    
    models_to_test = [
        ('RF', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('XGBoost', xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)),
        ('MLP', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
    ]
    
    baseline_results = []
    for rep_name in ['ecfp4', 'pdv', 'mol2vec', 'mhggnn']:
        print(f"\n  {rep_name} ({reps_train[rep_name].shape[1]} features):")
        
        X_train = reps_train[rep_name]
        X_test = reps_test[rep_name]
        
        for model_name, model in models_to_test:
            start = time.time()
            metrics = test_single(X_train, X_test, train_labels, test_labels, 
                                 model, is_classification=False, model_name=model_name)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'qm9_homo_lumo_gap',
                'representation': rep_name,
                'n_samples': n_baseline,
                'n_features_input': X_train.shape[1],
                'n_features_selected': X_train.shape[1],
                'feature_reduction': 0.0,
                'model': model_name,
                'is_hybrid': False,
                'time': elapsed,
                **metrics
            }
            results.append(result)
            baseline_results.append((rep_name, model_name, metrics['r2']))
            
            print(f"    [{model_name:8s}] R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}  ({elapsed:.2f}s)")
    
    # Get top 3 performers
    baseline_results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  → Top 3 rep+model combos:")
    for i in range(min(3, len(baseline_results))):
        rep, model, r2 = baseline_results[i]
        print(f"      {i+1}. {rep:15s} + {model:8s}  R²={r2:.4f}")
    
    # Get top reps
    seen_reps = set()
    top_reps = []
    for rep, model, r2 in baseline_results:
        if rep not in seen_reps:
            top_reps.append(rep)
            seen_reps.add(rep)
        if len(top_reps) >= 3:
            break
    
    # STEP 2: Strategic hybrids
    print(f"\n{'STEP 2: STRATEGIC HYBRIDS (top 2 combo with varying n_per_rep)':=^80}")
    
    best_combo = top_reps[:2]
    print(f"\n  Testing {'+'.join(best_combo)} with n_per_rep = {n_per_rep_values}")
    
    for n_per_rep in n_per_rep_values:
        print(f"\n  n_per_rep = {n_per_rep if n_per_rep != -1 else 'ALL'}:")
        
        hybrid_dict = {name: reps_train[name] for name in best_combo}
        X_train_hybrid, feature_info = create_hybrid_wrapper(
            hybrid_dict, train_labels, n_per_rep, 'tree_importance'
        )
        
        hybrid_dict_test = {name: reps_test[name] for name in best_combo}
        X_test_hybrid, _ = create_hybrid_wrapper(
            hybrid_dict_test, None, n_per_rep, 'tree_importance', feature_info
        )
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        print(f"    Features: {n_input} → {n_selected} ({reduction:.1f}% reduction)")
        
        # Test with all models
        for model_name, model_class in [
            ('RF', RandomForestRegressor),
            ('XGBoost', xgb.XGBRegressor),
            ('MLP', MLPRegressor)
        ]:
            start = time.time()
            
            if model_name == 'XGBoost':
                model = model_class(n_estimators=100, random_state=42, n_jobs=1)
            elif model_name == 'MLP':
                model = model_class(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
            else:
                model = model_class(n_estimators=100, random_state=42, n_jobs=-1)
            
            metrics = test_single(X_train_hybrid, X_test_hybrid, train_labels, test_labels,
                                 model, is_classification=False, model_name=model_name)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'qm9_homo_lumo_gap',
                'representation': '+'.join(best_combo),
                'n_samples': n_baseline,
                'n_features_input': n_input,
                'n_features_selected': n_selected,
                'feature_reduction': reduction,
                'n_per_rep': n_per_rep,
                'model': model_name,
                'is_hybrid': True,
                'time': elapsed,
                **metrics
            }
            results.append(result)
            
            print(f"    [{model_name:8s}] R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}  ({elapsed:.2f}s)")
    
    # ========================================================================
    # ESOL - Regression
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET 2: ESOL Solubility (Scaffold Split)")
    print(f"{'='*80}")
    
    data = load_esol_combined(splitter='scaffold')
    train_smiles_full = data['train']['smiles']
    train_labels_full = data['train']['labels']
    test_smiles = data['test']['smiles']
    test_labels = data['test']['labels']
    
    # Subsample
    n_baseline = 1000
    print(f"\nSubsampling to n={n_baseline}...")
    indices = np.random.choice(len(train_labels_full), 
                              min(n_baseline, len(train_labels_full)), 
                              replace=False)
    train_smiles = [train_smiles_full[i] for i in indices]
    train_labels = train_labels_full[indices]
    
    # Create representations
    print("Creating representations...")
    start = time.time()
    reps_train = {
        'ecfp4': create_ecfp4(train_smiles, n_bits=2048),
        'pdv': create_pdv(train_smiles),
        'mol2vec': create_mol2vec(train_smiles),
        'mhggnn': create_mhg_gnn(train_smiles)
    }
    reps_test = {
        'ecfp4': create_ecfp4(test_smiles, n_bits=2048),
        'pdv': create_pdv(test_smiles),
        'mol2vec': create_mol2vec(test_smiles),
        'mhggnn': create_mhg_gnn(test_smiles)
    }
    print(f"Representations created ({time.time() - start:.2f}s)")
    
    # STEP 1: Baseline
    print(f"\n{'STEP 1: BASELINE (all reps with RF, XGBoost, MLP)':=^80}")
    
    models_to_test = [
        ('RF', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('XGBoost', xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)),
        ('MLP', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
    ]
    
    baseline_results = []
    for rep_name in ['ecfp4', 'pdv', 'mol2vec', 'mhggnn']:
        print(f"\n  {rep_name} ({reps_train[rep_name].shape[1]} features):")
        
        X_train = reps_train[rep_name]
        X_test = reps_test[rep_name]
        
        for model_name, model in models_to_test:
            start = time.time()
            metrics = test_single(X_train, X_test, train_labels, test_labels,
                                 model, is_classification=False, model_name=model_name)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'esol',
                'representation': rep_name,
                'n_samples': n_baseline,
                'n_features_input': X_train.shape[1],
                'n_features_selected': X_train.shape[1],
                'feature_reduction': 0.0,
                'model': model_name,
                'is_hybrid': False,
                'time': elapsed,
                **metrics
            }
            results.append(result)
            baseline_results.append((rep_name, model_name, metrics['r2']))
            
            print(f"    [{model_name:8s}] R²={metrics['r2']:.4f}  ({elapsed:.2f}s)")
    
    # Get top performers
    baseline_results.sort(key=lambda x: x[2], reverse=True)
    seen_reps = set()
    top_reps = []
    for rep, model, r2 in baseline_results:
        if rep not in seen_reps:
            top_reps.append(rep)
            seen_reps.add(rep)
        if len(top_reps) >= 3:
            break
    
    # STEP 2: Hybrids
    print(f"\n{'STEP 2: STRATEGIC HYBRIDS (top 2 combo with varying n_per_rep)':=^80}")
    
    best_combo = top_reps[:2]
    print(f"\n  Testing {'+'.join(best_combo)} with n_per_rep = {n_per_rep_values}")
    
    for n_per_rep in n_per_rep_values:
        print(f"\n  n_per_rep = {n_per_rep if n_per_rep != -1 else 'ALL'}:")
        
        hybrid_dict = {name: reps_train[name] for name in best_combo}
        X_train_hybrid, feature_info = create_hybrid_wrapper(
            hybrid_dict, train_labels, n_per_rep, 'tree_importance'
        )
        
        hybrid_dict_test = {name: reps_test[name] for name in best_combo}
        X_test_hybrid, _ = create_hybrid_wrapper(
            hybrid_dict_test, None, n_per_rep, 'tree_importance', feature_info
        )
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        print(f"    Features: {n_input} → {n_selected} ({reduction:.1f}% reduction)")
        
        for model_name, model_class in [
            ('RF', RandomForestRegressor),
            ('XGBoost', xgb.XGBRegressor),
            ('MLP', MLPRegressor)
        ]:
            start = time.time()
            
            if model_name == 'XGBoost':
                model = model_class(n_estimators=100, random_state=42, n_jobs=1)
            elif model_name == 'MLP':
                model = model_class(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
            else:
                model = model_class(n_estimators=100, random_state=42, n_jobs=-1)
            
            metrics = test_single(X_train_hybrid, X_test_hybrid, train_labels, test_labels,
                                 model, is_classification=False, model_name=model_name)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'esol',
                'representation': '+'.join(best_combo),
                'n_samples': n_baseline,
                'n_features_input': n_input,
                'n_features_selected': n_selected,
                'feature_reduction': reduction,
                'n_per_rep': n_per_rep,
                'model': model_name,
                'is_hybrid': True,
                'time': elapsed,
                **metrics
            }
            results.append(result)
            
            print(f"    [{model_name:8s}] R²={metrics['r2']:.4f}  ({elapsed:.2f}s)")
    
    # ========================================================================
    # hERG - Classification
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET 3: hERG FLuID Cardiotoxicity (Classification)")
    print(f"{'='*80}")
    
    train_data = load_herg(source='fluid', use_test=False)
    test_data = load_herg(source='fluid', use_test=True)
    train_smiles_full = train_data['smiles']
    train_labels_full = train_data['labels']
    test_smiles = test_data['smiles']
    test_labels = test_data['labels']
    
    # Subsample
    n_baseline = 1000
    print(f"\nSubsampling to n={n_baseline}...")
    indices = np.random.choice(len(train_labels_full),
                              min(n_baseline, len(train_labels_full)),
                              replace=False)
    train_smiles = [train_smiles_full[i] for i in indices]
    train_labels = train_labels_full[indices]
    
    # Create representations
    print("Creating representations...")
    start = time.time()
    reps_train = {
        'ecfp4': create_ecfp4(train_smiles, n_bits=2048),
        'pdv': create_pdv(train_smiles),
        'mol2vec': create_mol2vec(train_smiles),
        'mhggnn': create_mhg_gnn(train_smiles)
    }
    reps_test = {
        'ecfp4': create_ecfp4(test_smiles, n_bits=2048),
        'pdv': create_pdv(test_smiles),
        'mol2vec': create_mol2vec(test_smiles),
        'mhggnn': create_mhg_gnn(test_smiles)
    }
    print(f"Representations created ({time.time() - start:.2f}s)")
    
    # STEP 1: Baseline
    print(f"\n{'STEP 1: BASELINE (all reps with RF, XGBoost, MLP)':=^80}")
    
    models_to_test = [
        ('RF', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('XGBoost', xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1)),
        ('MLP', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
    ]
    
    baseline_results = []
    for rep_name in ['ecfp4', 'pdv', 'mol2vec', 'mhggnn']:
        print(f"\n  {rep_name} ({reps_train[rep_name].shape[1]} features):")
        
        X_train = reps_train[rep_name]
        X_test = reps_test[rep_name]
        
        for model_name, model in models_to_test:
            start = time.time()
            metrics = test_single(X_train, X_test, train_labels, test_labels,
                                 model, is_classification=True, model_name=model_name)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'herg_fluid',
                'representation': rep_name,
                'n_samples': n_baseline,
                'n_features_input': X_train.shape[1],
                'n_features_selected': X_train.shape[1],
                'feature_reduction': 0.0,
                'model': model_name,
                'is_hybrid': False,
                'time': elapsed,
                **metrics
            }
            results.append(result)
            baseline_results.append((rep_name, model_name, metrics['auc']))
            
            print(f"    [{model_name:8s}] AUC={metrics['auc']:.4f}  ({elapsed:.2f}s)")
    
    # Get top reps
    baseline_results.sort(key=lambda x: x[2], reverse=True)
    seen_reps = set()
    top_reps = []
    for rep, model, auc in baseline_results:
        if rep not in seen_reps:
            top_reps.append(rep)
            seen_reps.add(rep)
        if len(top_reps) >= 3:
            break
    
    # STEP 2: Hybrids
    print(f"\n{'STEP 2: STRATEGIC HYBRIDS (top 2 combo with varying n_per_rep)':=^80}")
    
    best_combo = top_reps[:2]
    print(f"\n  Testing {'+'.join(best_combo)} with n_per_rep = {n_per_rep_values}")
    
    for n_per_rep in n_per_rep_values:
        print(f"\n  n_per_rep = {n_per_rep if n_per_rep != -1 else 'ALL'}:")
        
        hybrid_dict = {name: reps_train[name] for name in best_combo}
        X_train_hybrid, feature_info = create_hybrid_wrapper(
            hybrid_dict, train_labels, n_per_rep, 'tree_importance'
        )
        
        hybrid_dict_test = {name: reps_test[name] for name in best_combo}
        X_test_hybrid, _ = create_hybrid_wrapper(
            hybrid_dict_test, None, n_per_rep, 'tree_importance', feature_info
        )
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        print(f"    Features: {n_input} → {n_selected} ({reduction:.1f}% reduction)")
        
        for model_name, model_class in [
            ('RF', RandomForestClassifier),
            ('XGBoost', xgb.XGBClassifier),
            ('MLP', MLPClassifier)
        ]:
            start = time.time()
            
            if model_name == 'XGBoost':
                model = model_class(n_estimators=100, random_state=42, n_jobs=1)
            elif model_name == 'MLP':
                model = model_class(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
            else:
                model = model_class(n_estimators=100, random_state=42, n_jobs=-1)
            
            metrics = test_single(X_train_hybrid, X_test_hybrid, train_labels, test_labels,
                                 model, is_classification=True, model_name=model_name)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'herg_fluid',
                'representation': '+'.join(best_combo),
                'n_samples': n_baseline,
                'n_features_input': n_input,
                'n_features_selected': n_selected,
                'feature_reduction': reduction,
                'n_per_rep': n_per_rep,
                'model': model_name,
                'is_hybrid': True,
                'time': elapsed,
                **metrics
            }
            results.append(result)
            
            print(f"    [{model_name:8s}] AUC={metrics['auc']:.4f}  ({elapsed:.2f}s)")
    
    # ========================================================================
    # COMPREHENSIVE SUMMARY
    # ========================================================================
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    df = pd.DataFrame(results)
    
    print(f"\nRan {len(results)} experiments in {total_time:.2f}s ({total_time/60:.2f}min)")
    
    print("\n1. BEST BASELINE (rep+model) PER DATASET:")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_baseline = df_ds[~df_ds['is_hybrid']]
        
        if len(df_baseline) == 0:
            continue
        
        metric = 'auc' if 'herg' in dataset.lower() else 'r2'
        
        if df_baseline[metric].isna().all():
            continue
        
        best = df_baseline.loc[df_baseline[metric].idxmax()]
        print(f"\n{dataset.upper()}:")
        print(f"  Best: {best['representation']:15s} + {best['model']:8s}  {metric.upper()}={best[metric]:.4f}")
    
    print("\n2. BEST HYBRID (rep+model+n_per_rep) PER DATASET:")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_hybrid = df_ds[df_ds['is_hybrid']]
        
        if len(df_hybrid) == 0:
            continue
        
        metric = 'auc' if 'herg' in dataset.lower() else 'r2'
        
        if df_hybrid[metric].isna().all():
            continue
        
        best = df_hybrid.loc[df_hybrid[metric].idxmax()]
        
        print(f"\n{dataset.upper()}:")
        print(f"  Best: {best['representation']:30s} + {best['model']:8s}  n_per_rep={best['n_per_rep']:.0f}")
        print(f"        {metric.upper()}={best[metric]:.4f}")
        print(f"        Features: {best['n_features_input']:.0f} → {best['n_features_selected']:.0f} ({best['feature_reduction']:.1f}%)")
        
        # Compare to best baseline
        df_baseline = df_ds[~df_ds['is_hybrid']]
        if len(df_baseline) > 0 and not df_baseline[metric].isna().all():
            best_baseline = df_baseline[metric].max()
            improvement = ((best[metric] - best_baseline) / best_baseline) * 100
            
            if improvement > 0:
                print(f"        ✓ Beats best baseline by +{improvement:.2f}%")
            else:
                print(f"        Baseline better by {-improvement:.2f}%")
    
    print("\n3. n_per_rep SENSITIVITY (for best hybrid per dataset):")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_hybrid = df_ds[df_ds['is_hybrid']]
        
        if len(df_hybrid) == 0:
            continue
        
        metric = 'auc' if 'herg' in dataset.lower() else 'r2'
        
        if df_hybrid[metric].isna().all():
            continue
        
        print(f"\n{dataset.upper()}:")
        
        # Get best combo+model
        best = df_hybrid.loc[df_hybrid[metric].idxmax()]
        best_combo = best['representation']
        best_model = best['model']
        
        df_sweep = df_hybrid[(df_hybrid['representation'] == best_combo) &
                            (df_hybrid['model'] == best_model)]
        
        for _, row in df_sweep.iterrows():
            n_display = "ALL" if row['n_per_rep'] == -1 else f"{row['n_per_rep']:3.0f}"
            print(f"  n_per_rep={n_display:>4s}: {metric.upper()}={row[metric]:.4f}  ({row['n_features_selected']:.0f} feats)")
    
    # Save
    output = {
        'tier': 'comprehensive',
        'start_time': datetime.now().isoformat(),
        'total_time': total_time,
        'n_experiments': len(results),
        'philosophy': 'comprehensive: QM9 + ESOL + hERG, all models, full n_per_rep sweep',
        'results': results
    }
    
    with open('hybrid_master_comprehensive_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    df.to_csv('hybrid_master_comprehensive_results.csv', index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved: hybrid_master_comprehensive_results.*")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_comprehensive_evaluation()