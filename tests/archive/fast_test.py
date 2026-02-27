#!/usr/bin/env python3
"""
KIRBy Hybrid Master - FAST ITERATION VERSION

Quick validation script for testing new features rapidly.

Datasets:
1. QM9 HOMO-LUMO gap (5000 samples, scaffold split)
2. ESOL solubility (scaffold split)
3. hERG FLuID cardiotoxicity

Models: RF, XGBoost, MLP
Representations: ECFP4, PDV, mhggnn (optional)
n_per_rep: [25, 50, 100, 200, 500, 1000, -1]

Usage:
  python hybrid_master_fast.py                 # Include mhggnn
  python hybrid_master_fast.py --skip-mhggnn   # Skip mhggnn for speed
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time
import json
from datetime import datetime

import sys
sys.path.insert(0, '/mnt/user-data/outputs/src')

from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.datasets.esol import load_esol_combined
from kirby.datasets.herg import load_herg_fluid
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_mhg_gnn
)


def create_hybrid_wrapper(rep_dict, labels, n_per_rep, selection_method='tree_importance', 
                         feature_info=None):
    """Simple hybrid wrapper using RF importance"""
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
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        
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


def test_single(X_train, X_test, y_train, y_test, model, is_classification):
    """Test single representation"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    if is_classification:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        return evaluate_classification(y_test, y_pred, y_prob)
    else:
        return evaluate_regression(y_test, y_pred)


def run_fast_iteration(skip_mhggnn=False):
    """Fast iteration testing"""
    
    print("="*100)
    print("KIRBY FAST ITERATION TEST")
    print("="*100)
    print("\nDatasets:")
    print("  1. QM9 HOMO-LUMO gap (5000 samples, scaffold split)")
    print("  2. ESOL solubility (scaffold split)")
    print("  3. hERG FLuID cardiotoxicity")
    print("\nModels: RF, XGBoost, MLP")
    reps_list = "ECFP4, PDV" + ("" if skip_mhggnn else ", mhggnn")
    print(f"Representations: {reps_list}")
    print("n_per_rep: [25, 50, 100, 200, 500, 1000, -1]")
    print("="*100)
    
    results = []
    overall_start = time.time()
    n_per_rep_values = [25, 50, 100, 200, 500, 1000, -1]
    
    # ========================================================================
    # QM9 HOMO-LUMO GAP
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET 1: QM9 HOMO-LUMO Gap (Scaffold Split)")
    print(f"{'='*80}")
    
    raw_data = load_qm9(n_samples=5000, property_idx=4)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    train_smiles = splits['train']['smiles'] + splits['val']['smiles']
    train_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
    test_smiles = splits['test']['smiles']
    test_labels = splits['test']['labels']
    
    print(f"Target: {raw_data['property_name']} (eV)")
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    
    # Create representations
    print("\nCreating representations...")
    start = time.time()
    reps_train = {
        'ecfp4': create_ecfp4(train_smiles, n_bits=2048),
        'pdv': create_pdv(train_smiles)
    }
    reps_test = {
        'ecfp4': create_ecfp4(test_smiles, n_bits=2048),
        'pdv': create_pdv(test_smiles)
    }
    
    if not skip_mhggnn:
        print("  Adding mhggnn...")
        reps_train['mhggnn'] = create_mhg_gnn(train_smiles)
        reps_test['mhggnn'] = create_mhg_gnn(test_smiles)
    
    print(f"Representations created ({time.time() - start:.2f}s)")
    
    # STEP 1: Baseline with ALL models
    print(f"\n{'STEP 1: BASELINE (all reps with RF, XGBoost, MLP)':=^80}")
    
    models_to_test = [
        ('RF', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('XGBoost', xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)),
        ('MLP', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
    ]
    
    baseline_results = []
    
    for rep_name in reps_train.keys():
        print(f"\n  {rep_name} ({reps_train[rep_name].shape[1]} features):")
        
        for model_name, model in models_to_test:
            start = time.time()
            metrics = test_single(reps_train[rep_name], reps_test[rep_name],
                                 train_labels, test_labels, model, is_classification=False)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'qm9_homo_lumo_gap',
                'representation': rep_name,
                'n_features': reps_train[rep_name].shape[1],
                'model': model_name,
                'is_hybrid': False,
                'time': elapsed,
                **metrics
            }
            results.append(result)
            baseline_results.append((rep_name, model_name, metrics['r2']))
            
            print(f"    [{model_name:8s}] R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}  ({elapsed:.2f}s)")
    
    # Find top 2 reps (by best R² across all models)
    baseline_results.sort(key=lambda x: x[2], reverse=True)
    seen_reps = set()
    top_reps = []
    for rep, model, r2 in baseline_results:
        if rep not in seen_reps:
            top_reps.append(rep)
            seen_reps.add(rep)
        if len(top_reps) >= 2:
            break
    
    print(f"\n  → Top 2 reps: {', '.join(top_reps)}")
    
    # STEP 2: Hybrids with ALL n_per_rep values and ALL models
    print(f"\n{'STEP 2: HYBRIDS (top 2 combo with all n_per_rep and all models)':=^80}")
    
    for n_per_rep in n_per_rep_values:
        print(f"\n  n_per_rep={n_per_rep if n_per_rep != -1 else 'ALL'}:")
        
        hybrid_dict = {name: reps_train[name] for name in top_reps}
        X_train_hybrid, feature_info = create_hybrid_wrapper(
            hybrid_dict, train_labels, n_per_rep, 'tree_importance'
        )
        
        hybrid_dict_test = {name: reps_test[name] for name in top_reps}
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
                                 model, is_classification=False)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'qm9_homo_lumo_gap',
                'representation': '+'.join(top_reps),
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
    # ESOL - Solubility Regression
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET 2: ESOL (Solubility Regression, Scaffold Split)")
    print(f"{'='*80}")
    
    data = load_esol_combined(splitter='scaffold')
    train_smiles = data['train']['smiles']
    train_labels = data['train']['labels']
    test_smiles = data['test']['smiles']
    test_labels = data['test']['labels']
    
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    
    # Create representations
    print("\nCreating representations...")
    start = time.time()
    reps_train = {
        'ecfp4': create_ecfp4(train_smiles, n_bits=2048),
        'pdv': create_pdv(train_smiles)
    }
    reps_test = {
        'ecfp4': create_ecfp4(test_smiles, n_bits=2048),
        'pdv': create_pdv(test_smiles)
    }
    
    if not skip_mhggnn:
        print("  Adding mhggnn...")
        reps_train['mhggnn'] = create_mhg_gnn(train_smiles)
        reps_test['mhggnn'] = create_mhg_gnn(test_smiles)
    
    print(f"Representations created ({time.time() - start:.2f}s)")
    
    # STEP 1: Baseline
    print(f"\n{'STEP 1: BASELINE (all reps with RF, XGBoost, MLP)':=^80}")
    
    baseline_results = []
    
    for rep_name in reps_train.keys():
        print(f"\n  {rep_name} ({reps_train[rep_name].shape[1]} features):")
        
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
            
            metrics = test_single(reps_train[rep_name], reps_test[rep_name],
                                 train_labels, test_labels, model, is_classification=False)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'esol',
                'representation': rep_name,
                'n_features': reps_train[rep_name].shape[1],
                'model': model_name,
                'is_hybrid': False,
                'time': elapsed,
                **metrics
            }
            results.append(result)
            baseline_results.append((rep_name, model_name, metrics['rmse']))
            
            print(f"    [{model_name:8s}] RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}  ({elapsed:.2f}s)")
    
    # Find top 2 reps (by best RMSE)
    baseline_results.sort(key=lambda x: x[2])
    seen_reps = set()
    top_reps = []
    for rep, model, rmse in baseline_results:
        if rep not in seen_reps:
            top_reps.append(rep)
            seen_reps.add(rep)
        if len(top_reps) >= 2:
            break
    
    print(f"\n  → Top 2 reps: {', '.join(top_reps)}")
    
    # STEP 2: Hybrids
    print(f"\n{'STEP 2: HYBRIDS (top 2 combo with all n_per_rep and all models)':=^80}")
    
    for n_per_rep in n_per_rep_values:
        print(f"\n  n_per_rep={n_per_rep if n_per_rep != -1 else 'ALL'}:")
        
        hybrid_dict = {name: reps_train[name] for name in top_reps}
        X_train_hybrid, feature_info = create_hybrid_wrapper(
            hybrid_dict, train_labels, n_per_rep, 'tree_importance'
        )
        
        hybrid_dict_test = {name: reps_test[name] for name in top_reps}
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
                                 model, is_classification=False)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'esol',
                'representation': '+'.join(top_reps),
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
            
            print(f"    [{model_name:8s}] RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}  ({elapsed:.2f}s)")
    
    # ========================================================================
    # hERG - Classification
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET 3: hERG FLuID (Cardiotoxicity Classification)")
    print(f"{'='*80}")
    
    train_data = load_herg_fluid(use_test=False)
    test_data = load_herg_fluid(use_test=True)
    
    train_smiles = train_data['smiles']
    train_labels = train_data['labels']
    test_smiles = test_data['smiles']
    test_labels = test_data['labels']
    
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    
    # Create representations
    print("\nCreating representations...")
    start = time.time()
    reps_train = {
        'ecfp4': create_ecfp4(train_smiles, n_bits=2048),
        'pdv': create_pdv(train_smiles)
    }
    reps_test = {
        'ecfp4': create_ecfp4(test_smiles, n_bits=2048),
        'pdv': create_pdv(test_smiles)
    }
    
    if not skip_mhggnn:
        print("  Adding mhggnn...")
        reps_train['mhggnn'] = create_mhg_gnn(train_smiles)
        reps_test['mhggnn'] = create_mhg_gnn(test_smiles)
    
    print(f"Representations created ({time.time() - start:.2f}s)")
    
    # STEP 1: Baseline
    print(f"\n{'STEP 1: BASELINE (all reps with RF, XGBoost, MLP)':=^80}")
    
    baseline_results = []
    
    for rep_name in reps_train.keys():
        print(f"\n  {rep_name} ({reps_train[rep_name].shape[1]} features):")
        
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
            
            metrics = test_single(reps_train[rep_name], reps_test[rep_name],
                                 train_labels, test_labels, model, is_classification=True)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'herg_fluid',
                'representation': rep_name,
                'n_features': reps_train[rep_name].shape[1],
                'model': model_name,
                'is_hybrid': False,
                'time': elapsed,
                **metrics
            }
            results.append(result)
            baseline_results.append((rep_name, model_name, metrics['auc']))
            
            print(f"    [{model_name:8s}] AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}  ({elapsed:.2f}s)")
    
    # Find top 2 reps
    baseline_results.sort(key=lambda x: x[2], reverse=True)
    seen_reps = set()
    top_reps = []
    for rep, model, auc in baseline_results:
        if rep not in seen_reps:
            top_reps.append(rep)
            seen_reps.add(rep)
        if len(top_reps) >= 2:
            break
    
    print(f"\n  → Top 2 reps: {', '.join(top_reps)}")
    
    # STEP 2: Hybrids
    print(f"\n{'STEP 2: HYBRIDS (top 2 combo with all n_per_rep and all models)':=^80}")
    
    for n_per_rep in n_per_rep_values:
        print(f"\n  n_per_rep={n_per_rep if n_per_rep != -1 else 'ALL'}:")
        
        hybrid_dict = {name: reps_train[name] for name in top_reps}
        X_train_hybrid, feature_info = create_hybrid_wrapper(
            hybrid_dict, train_labels, n_per_rep, 'tree_importance'
        )
        
        hybrid_dict_test = {name: reps_test[name] for name in top_reps}
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
                                 model, is_classification=True)
            elapsed = time.time() - start
            
            result = {
                'dataset': 'herg_fluid',
                'representation': '+'.join(top_reps),
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
            
            print(f"    [{model_name:8s}] AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}  ({elapsed:.2f}s)")
            print(f"    [{model_name:8s}] AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}  ({elapsed:.2f}s)")
    
    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.time() - overall_start
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'='*100}")
    
    df = pd.DataFrame(results)
    print(f"\nRan {len(results)} experiments in {total_time:.2f}s ({total_time/60:.2f}min)")
    
    # Per-dataset detailed analysis
    for dataset_name, task_type, metric, maximize in [
        ('qm9_homo_lumo_gap', 'Regression', 'r2', True),
        ('esol', 'Regression', 'rmse', False),
        ('herg_fluid', 'Classification', 'auc', True)
    ]:
        print(f"\n{'='*100}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*100}")
        
        df_subset = df[df['dataset'] == dataset_name]
        
        # BEST BASELINE
        df_baseline = df_subset[~df_subset['is_hybrid']]
        if len(df_baseline) > 0:
            if maximize:
                best_baseline = df_baseline.loc[df_baseline[metric].idxmax()]
            else:
                best_baseline = df_baseline.loc[df_baseline[metric].idxmin()]
            
            print(f"\nBEST BASELINE:")
            print(f"  Representation: {best_baseline['representation']}")
            print(f"  Model:          {best_baseline['model']}")
            print(f"  Features:       {best_baseline['n_features']:.0f}")
            if metric == 'r2':
                print(f"  R²:             {best_baseline[metric]:.4f}")
                print(f"  MAE:            {best_baseline['mae']:.4f}")
            elif metric == 'rmse':
                print(f"  RMSE:           {best_baseline[metric]:.4f}")
                print(f"  R²:             {best_baseline['r2']:.4f}")
            else:  # auc
                print(f"  AUC:            {best_baseline[metric]:.4f}")
                print(f"  F1:             {best_baseline['f1']:.4f}")
        
        # BEST HYBRID
        df_hybrid = df_subset[df_subset['is_hybrid']]
        if len(df_hybrid) > 0:
            if maximize:
                best_hybrid = df_hybrid.loc[df_hybrid[metric].idxmax()]
            else:
                best_hybrid = df_hybrid.loc[df_hybrid[metric].idxmin()]
            
            print(f"\nBEST HYBRID:")
            print(f"  Representation: {best_hybrid['representation']}")
            print(f"  Model:          {best_hybrid['model']}")
            print(f"  n_per_rep:      {best_hybrid['n_per_rep']:.0f}")
            print(f"  Features:       {best_hybrid['n_features_selected']:.0f}/{best_hybrid['n_features_input']:.0f} "
                  f"({best_hybrid['feature_reduction']:.1f}% reduction)")
            if metric == 'r2':
                print(f"  R²:             {best_hybrid[metric]:.4f}")
                print(f"  MAE:            {best_hybrid['mae']:.4f}")
            elif metric == 'rmse':
                print(f"  RMSE:           {best_hybrid[metric]:.4f}")
                print(f"  R²:             {best_hybrid['r2']:.4f}")
            else:  # auc
                print(f"  AUC:            {best_hybrid[metric]:.4f}")
                print(f"  F1:             {best_hybrid['f1']:.4f}")
            
            # COMPARISON
            print(f"\nHYBRID vs BASELINE:")
            if maximize:
                improvement = ((best_hybrid[metric] - best_baseline[metric]) / best_baseline[metric]) * 100
                print(f"  Performance:    {best_hybrid[metric]:.4f} vs {best_baseline[metric]:.4f} "
                      f"({'+' if improvement > 0 else ''}{improvement:.2f}%)")
            else:  # rmse - lower is better
                improvement = ((best_baseline[metric] - best_hybrid[metric]) / best_baseline[metric]) * 100
                print(f"  Performance:    {best_hybrid[metric]:.4f} vs {best_baseline[metric]:.4f} "
                      f"({'+' if improvement > 0 else ''}{improvement:.2f}% better)")
            
            feature_efficiency = best_baseline['n_features'] / best_hybrid['n_features_selected']
            print(f"  Feature Count:  {best_hybrid['n_features_selected']:.0f} vs {best_baseline['n_features']:.0f} "
                  f"({feature_efficiency:.1f}x fewer)")
            
            if improvement > 0:
                print(f"  ✓ WINNER: Hybrid achieves better performance with {feature_efficiency:.1f}x fewer features")
            else:
                print(f"  ✗ Baseline wins on performance (but hybrid uses {feature_efficiency:.1f}x fewer features)")
            
            # CONSTITUENT BASE REPS (for noise mitigation comparison)
            print(f"\nCONSTITUENT BASE REPRESENTATIONS:")
            base_reps = best_hybrid['representation'].split('+')
            for base_rep in base_reps:
                base_perf = df_baseline[df_baseline['representation'] == base_rep]
                if len(base_perf) > 0:
                    # Get best performance for this rep across all models
                    if maximize:
                        best_base = base_perf.loc[base_perf[metric].idxmax()]
                    else:
                        best_base = base_perf.loc[base_perf[metric].idxmin()]
                    
                    if metric == 'r2':
                        print(f"  - {base_rep:15s} (best: {best_base['model']:8s} R²={best_base[metric]:.4f})")
                    elif metric == 'rmse':
                        print(f"  - {base_rep:15s} (best: {best_base['model']:8s} RMSE={best_base[metric]:.4f})")
                    else:
                        print(f"  - {base_rep:15s} (best: {best_base['model']:8s} AUC={best_base[metric]:.4f})")
    
    # OVERALL BEST
    print(f"\n{'='*100}")
    print("OVERALL FINDINGS")
    print(f"{'='*100}")
    
    print("\nBest hybrid per dataset:")
    for dataset_name, task_type, metric, maximize in [
        ('qm9_homo_lumo_gap', 'Regression', 'r2', True),
        ('esol', 'Regression', 'rmse', False),
        ('herg_fluid', 'Classification', 'auc', True)
    ]:
        df_subset = df[df['dataset'] == dataset_name]
        df_hybrid = df_subset[df_subset['is_hybrid']]
        
        if len(df_hybrid) > 0:
            if maximize:
                best = df_hybrid.loc[df_hybrid[metric].idxmax()]
            else:
                best = df_hybrid.loc[df_hybrid[metric].idxmin()]
            
            print(f"\n  {dataset_name}:")
            print(f"    {best['representation']} + {best['model']}, n_per_rep={best['n_per_rep']:.0f}")
            if metric == 'r2':
                print(f"    R²={best[metric]:.4f}, {best['n_features_selected']:.0f} features")
            elif metric == 'rmse':
                print(f"    RMSE={best[metric]:.4f}, {best['n_features_selected']:.0f} features")
            else:
                print(f"    AUC={best[metric]:.4f}, {best['n_features_selected']:.0f} features")
    
    # Save
    output = {
        'tier': 'fast_iteration',
        'start_time': datetime.now().isoformat(),
        'total_time': total_time,
        'n_experiments': len(results),
        'skip_mhggnn': skip_mhggnn,
        'datasets': ['qm9_homo_lumo_gap', 'esol', 'herg_fluid'],
        'models': ['RF', 'XGBoost', 'MLP'],
        'n_per_rep_values': n_per_rep_values,
        'results': results
    }
    
    with open('hybrid_master_fast_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    df.to_csv('hybrid_master_fast_results.csv', index=False)
    
    print(f"\n{'='*100}")
    print(f"Results saved: hybrid_master_fast_results.*")
    print(f"{'='*100}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast iteration testing for KIRBy')
    parser.add_argument('--skip-mhggnn', action='store_true',
                       help='Skip mhggnn representation for faster runtime')
    args = parser.parse_args()
    
    run_fast_iteration(skip_mhggnn=args.skip_mhggnn)