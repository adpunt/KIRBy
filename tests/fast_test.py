#!/usr/bin/env python3
"""
KIRBy Hybrid Master - TIER 1 SUPERFAST (~60-90s)

STRUCTURED INTELLIGENT SAMPLING

Defaults to YOUR create_hybrid with Random Forest feature importance.
Alternatives available: SHAP, fasttreeshap, permutation, MI, interpretML

Strategy:
1. Subsample to n=1000 FIRST (efficiency)
2. Baseline: All reps with RF, XGBoost, MLP
3. Strategic hybrids: Feature selection with varying n_per_rep
4. Find optimal model+n_per_rep combination

~20-30 experiments in 60-90 seconds
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
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time
import json
from datetime import datetime
from typing import Dict, Callable

import sys
sys.path.insert(0, '/mnt/user-data/outputs/src')

from kirby.datasets.esol import load_esol_combined
from kirby.datasets.herg import load_herg
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_mol2vec,
    create_mhg_gnn
)


def create_hybrid_wrapper(rep_dict: Dict[str, np.ndarray],
                         labels: np.ndarray,
                         n_per_rep: int = 50,
                         selection_method: str = 'tree_importance',
                         feature_info: Dict = None) -> tuple:
    """
    Wrapper around enhanced create_hybrid with all methods.
    
    Args:
        feature_info: If provided, apply existing feature selection (for test set).
                     If None, compute new feature selection (for train set).
    
    Returns:
        tuple: (hybrid_features, feature_info) or just hybrid_features if applying existing selection
    
    Special case: n_per_rep = -1 means "keep ALL features" (naive concatenation)
    """
    # Special case: keep everything (no selection at all)
    if n_per_rep == -1:
        all_features = []
        for name, X in rep_dict.items():
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
            all_features.append(X_clipped)
        result = np.hstack(all_features)
        return (result, None) if feature_info is None else result
    
    # Clean the data first
    clean_reps = {}
    for name, X in rep_dict.items():
        X_clipped = np.clip(X, -1e10, 1e10)
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
        clean_reps[name] = X_clipped
    
    # If feature_info provided, apply existing selection (for test set)
    if feature_info is not None:
        from hybrid_enhanced import apply_feature_selection
        return apply_feature_selection(clean_reps, feature_info)
    
    # Otherwise, compute new selection (for train set)
    try:
        from hybrid_enhanced import create_hybrid
        
        hybrid_features, feature_info_new = create_hybrid(
            base_reps=clean_reps,
            labels=labels,
            n_per_rep=n_per_rep,
            importance_method=selection_method
        )
        return hybrid_features, feature_info_new
        
    except ImportError:
        raise ImportError("hybrid_enhanced.py not found. Make sure it's in the same directory.")


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
    """Test single representation with appropriate scaler for model type"""
    from sklearn.preprocessing import RobustScaler
    
    # Use RobustScaler for MLP (handles outliers better), StandardScaler for others
    if 'MLP' in model_name:
        scaler = RobustScaler()
    else:
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


def run_structured_sampling(feature_selector, n_per_rep_values=[25, 50, 100, 200, 500, 1000, -1]):
    """
    Structured intelligent sampling - tests ALL available feature selection methods.
    
    Automatically detects which methods are available and tests them all.
    """
    
    # Detect available methods
    methods_to_test = [('tree_importance', 'RF feature importance')]
    
    try:
        import shap
        methods_to_test.append(('shap', 'SHAP TreeExplainer'))
    except ImportError:
        pass
    
    try:
        import fasttreeshap
        methods_to_test.append(('fasttreeshap', 'FastTreeSHAP'))
    except ImportError:
        pass
    
    methods_to_test.append(('permutation', 'Permutation importance'))
    methods_to_test.append(('mutual_info', 'Mutual Information'))
    
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
        methods_to_test.append(('interpretml', 'InterpretML EBM'))
    except ImportError:
        pass
    
    print("="*100)
    print("TIER 1: STRUCTURED INTELLIGENT SAMPLING")
    print("="*100)
    print("\nTesting ALL available feature selection methods:")
    for i, (method, desc) in enumerate(methods_to_test, 1):
        print(f"  {i}. {method:20s} - {desc}")
    print(f"\nStrategy:")
    print("  1. Subsample to n=1000 FIRST (efficiency)")
    print("  2. Baseline: All reps with RF, XGBoost, MLP")
    print("  3. Strategic hybrids: Multiple feature selection methods with varying n_per_rep")
    print("  4. Compare methods and find optimal model + n_per_rep combo")
    print("="*100)
    
    all_results = []
    overall_start = time.time()
    
    # ========================================================================
    # ESOL - Regression
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET: ESOL (Solubility Regression)")
    print(f"{'='*80}")
    
    data = load_esol_combined(splitter='scaffold')
    train_smiles_full = data['train']['smiles']
    train_labels_full = data['train']['labels']
    test_smiles = data['test']['smiles']
    test_labels = data['test']['labels']
    
    # SUBSAMPLE FIRST!
    n_baseline = 1000
    print(f"\nSubsampling to n={n_baseline} before encoding...")
    indices = np.random.choice(len(train_labels_full), min(n_baseline, len(train_labels_full)), replace=False)
    train_smiles = [train_smiles_full[i] for i in indices]
    train_labels = train_labels_full[indices]
    
    # Create representations on subsampled data
    print("Creating representations (on subsampled data only)...")
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
    for name, X in reps_train.items():
        print(f"  {name}: {X.shape}")
    
    # STEP 1: Baseline - all reps with MULTIPLE MODELS
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
            
            metrics = test_single(X_train, X_test, train_labels, test_labels, model, is_classification=False, model_name=model_name)
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
    
    # Find top performers
    baseline_results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  → Top 3 rep+model combos:")
    for i in range(min(3, len(baseline_results))):
        rep, model, r2 = baseline_results[i]
        print(f"      {i+1}. {rep:15s} + {model:8s}  R²={r2:.4f}")
    
    # Get top representations (unique)
    seen_reps = set()
    top_reps = []
    for rep, model, r2 in baseline_results:
        if rep not in seen_reps:
            top_reps.append(rep)
            seen_reps.add(rep)
        if len(top_reps) >= 3:
            break
    
    # STEP 2: Strategic hybrids with MULTIPLE n_per_rep values
    print(f"\n{'STEP 2: STRATEGIC HYBRIDS (top 2 combo with varying n_per_rep)':=^80}")
    
    best_combo = top_reps[:2]
    print(f"\n  Testing {'+'.join(best_combo)} with n_per_rep = {n_per_rep_values}")
    
    for n_per_rep in n_per_rep_values:
        print(f"\n  n_per_rep = {n_per_rep}:")
        
        # Create hybrid once
        hybrid_dict = {name: reps_train[name] for name in best_combo}
        X_train_hybrid, feature_info = feature_selector(hybrid_dict, train_labels, n_per_rep, selection_method)
        
        hybrid_dict_test = {name: reps_test[name] for name in best_combo}
        X_test_hybrid = feature_selector(hybrid_dict_test, None, n_per_rep, selection_method, feature_info=feature_info)
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        print(f"    Features: {n_input} → {n_selected} ({reduction:.1f}% reduction)")
        
        # Test with multiple models
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
            
            metrics = test_single(X_train_hybrid, X_test_hybrid, train_labels, test_labels, model, is_classification=False, model_name=model_name)
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
    print("DATASET: hERG FLuID (Cardiotoxicity Classification)")
    print(f"{'='*80}")
    
    train_data = load_herg(source='fluid', use_test=False)
    test_data = load_herg(source='fluid', use_test=True)
    train_smiles_full = train_data['smiles']
    train_labels_full = train_data['labels']
    test_smiles = test_data['smiles']
    test_labels = test_data['labels']
    
    # SUBSAMPLE FIRST!
    n_baseline = 1000
    print(f"\nSubsampling to n={n_baseline} before encoding...")
    indices = np.random.choice(len(train_labels_full), min(n_baseline, len(train_labels_full)), replace=False)
    train_smiles = [train_smiles_full[i] for i in indices]
    train_labels = train_labels_full[indices]
    
    # Create representations on subsampled data
    print("Creating representations (on subsampled data only)...")
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
    
    # STEP 1: Baseline with multiple models
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
            
            metrics = test_single(X_train, X_test, train_labels, test_labels, model, is_classification=True, model_name=model_name)
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
    
    # Find top performers
    baseline_results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  → Top 3 rep+model combos:")
    for i in range(min(3, len(baseline_results))):
        rep, model, auc = baseline_results[i]
        print(f"      {i+1}. {rep:15s} + {model:8s}  AUC={auc:.4f}")
    
    # Get top reps
    seen_reps = set()
    top_reps = []
    for rep, model, auc in baseline_results:
        if rep not in seen_reps:
            top_reps.append(rep)
            seen_reps.add(rep)
        if len(top_reps) >= 3:
            break
    
    # STEP 2: Strategic hybrids with multiple n_per_rep
    print(f"\n{'STEP 2: STRATEGIC HYBRIDS (top 2 combo with varying n_per_rep)':=^80}")
    
    best_combo = top_reps[:2]
    print(f"\n  Testing {'+'.join(best_combo)} with n_per_rep = {n_per_rep_values}")
    
    for n_per_rep in n_per_rep_values:
        print(f"\n  n_per_rep = {n_per_rep}:")
        
        # Create hybrid once
        hybrid_dict = {name: reps_train[name] for name in best_combo}
        X_train_hybrid, feature_info = feature_selector(hybrid_dict, train_labels, n_per_rep, selection_method)
        
        hybrid_dict_test = {name: reps_test[name] for name in best_combo}
        X_test_hybrid = feature_selector(hybrid_dict_test, None, n_per_rep, selection_method, feature_info=feature_info)
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        print(f"    Features: {n_input} → {n_selected} ({reduction:.1f}% reduction)")
        
        # Test with multiple models
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
            
            metrics = test_single(X_train_hybrid, X_test_hybrid, train_labels, test_labels, model, is_classification=True, model_name=model_name)
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
    # Analysis
    # ========================================================================
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    df = pd.DataFrame(results)
    
    print(f"\nRan {len(results)} experiments in {total_time:.2f}s ({total_time/60:.2f}min)")
    
    print("\n1. BEST BASELINE (rep+model) PER DATASET:")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_baseline = df_ds[~df_ds['is_hybrid']]
        metric = 'auc' if 'auc' in df_baseline.columns else 'r2'
        
        best = df_baseline.loc[df_baseline[metric].idxmax()]
        print(f"\n{dataset.upper()}:")
        print(f"  Best: {best['representation']:15s} + {best['model']:8s}  {metric.upper()}={best[metric]:.4f}")
    
    print("\n2. BEST HYBRID (rep+model+n_per_rep) PER DATASET:")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_hybrid = df_ds[df_ds['is_hybrid']]
        
        if len(df_hybrid) > 0:
            metric = 'auc' if 'auc' in df_hybrid.columns else 'r2'
            best = df_hybrid.loc[df_hybrid[metric].idxmax()]
            
            print(f"\n{dataset.upper()}:")
            print(f"  Best: {best['representation']:30s} + {best['model']:8s}  n_per_rep={best['n_per_rep']:.0f}")
            print(f"        {metric.upper()}={best[metric]:.4f}")
            print(f"        Features: {best['n_features_input']:.0f} → {best['n_features_selected']:.0f} ({best['feature_reduction']:.1f}%)")
            
            # Compare to best baseline
            df_baseline = df_ds[~df_ds['is_hybrid']]
            best_baseline = df_baseline[metric].max()
            improvement = ((best[metric] - best_baseline) / best_baseline) * 100
            
            if improvement > 0:
                print(f"        ✓ Beats best baseline by +{improvement:.2f}%")
            else:
                print(f"        Baseline better by {-improvement:.2f}%")
    
    print("\n3. n_per_rep SENSITIVITY (for best hybrid):")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_hybrid = df_ds[df_ds['is_hybrid']]
        
        if len(df_hybrid) > 0:
            metric = 'auc' if 'auc' in df_hybrid.columns else 'r2'
            
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
    
    print("\n4. MODEL SENSITIVITY (for best hybrid+n_per_rep):")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_hybrid = df_ds[df_ds['is_hybrid']]
        
        if len(df_hybrid) > 0:
            metric = 'auc' if 'auc' in df_hybrid.columns else 'r2'
            
            print(f"\n{dataset.upper()}:")
            
            # Get best combo+n_per_rep
            best = df_hybrid.loc[df_hybrid[metric].idxmax()]
            best_combo = best['representation']
            best_n = best['n_per_rep']
            
            df_models = df_hybrid[(df_hybrid['representation'] == best_combo) & 
                                 (df_hybrid['n_per_rep'] == best_n)]
            
            for _, row in df_models.iterrows():
                print(f"  {row['model']:8s}: {metric.upper()}={row[metric]:.4f}")
    
    # Save
    output = {
        'tier': 'superfast_structured_multimodel',
        'start_time': datetime.now().isoformat(),
        'total_time': total_time,
        'n_experiments': len(results),
        'philosophy': 'subsample first, test all reps with multiple models, vary n_per_rep',
        'results': results
    }
    
    with open('hybrid_master_superfast_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    df.to_csv('hybrid_master_superfast_results.csv', index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved: hybrid_master_superfast_results.*")
    print(f"{'='*80}")


if __name__ == '__main__':
    # Choose feature selection method:
    # 'tree_importance' - YOUR create_hybrid with RF importance (DEFAULT, BASELINE)
    # 'shap' - SHAP TreeExplainer (requires: pip install shap)
    # 'fasttreeshap' - Fast approximation (requires: pip install fasttreeshap)
    # 'permutation' - Permutation importance (slow, no extra deps)
    # 'mutual_info' - Mutual information (weak, no extra deps)
    # 'interpretml' - InterpretML EBM (requires: pip install interpret)
    
    selection_method = 'tree_importance'  # Change this to test alternatives
    
    run_structured_sampling(
        create_hybrid_wrapper, 
        n_per_rep_values=[25, 50, 100, 200, 500, 1000, -1],
        selection_method=selection_method
    )
    """
    Structured intelligent sampling:
    1. Baseline all reps
    2. Strategic hybrids
    3. Model variations
    """
    
    print("="*100)
    print("TIER 1: STRUCTURED INTELLIGENT SAMPLING")
    print("="*100)
    print("\nStrategy:")
    print("  1. Baseline: All reps @ n=1000 with RF")
    print("  2. Strategic hybrids from top performers")
    print("  3. Model variations on best")
    print("="*100)
    
    results = []
    overall_start = time.time()
    
    # ========================================================================
    # ESOL - Regression
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET: ESOL (Solubility Regression)")
    print(f"{'='*80}")
    
    data = load_esol_combined(splitter='scaffold')
    train_smiles = data['train']['smiles']
    train_labels = data['train']['labels']
    test_smiles = data['test']['smiles']
    test_labels = data['test']['labels']
    
    # Create representations
    print("\nCreating representations...")
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
    for name, X in reps_train.items():
        print(f"  {name}: {X.shape}")
    
    # STEP 1: Baseline - all reps at n=1000 with RF
    print(f"\n{'STEP 1: BASELINE (all reps @ n=1000 with RF)':=^80}")
    
    n_baseline = 1000
    indices = np.random.choice(len(train_labels), min(n_baseline, len(train_labels)), replace=False)
    
    baseline_results = []
    for rep_name in ['ecfp4', 'pdv', 'mol2vec', 'mhggnn']:
        start = time.time()
        
        X_train = reps_train[rep_name][indices]
        y_train = train_labels[indices]
        X_test = reps_test[rep_name]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        metrics = test_single(X_train, X_test, y_train, test_labels, model, is_classification=False, model_name=model_name)
        elapsed = time.time() - start
        
        result = {
            'dataset': 'esol',
            'representation': rep_name,
            'n_samples': n_baseline,
            'n_features_input': X_train.shape[1],
            'n_features_selected': X_train.shape[1],
            'feature_reduction': 0.0,
            'model': 'RandomForest',
            'is_hybrid': False,
            'time': elapsed,
            **metrics
        }
        results.append(result)
        baseline_results.append((rep_name, metrics['r2']))
        
        print(f"  {rep_name:15s} {X_train.shape[1]:5d} feats  R²={metrics['r2']:.4f}  ({elapsed:.2f}s)")
    
    # Sort by performance
    baseline_results.sort(key=lambda x: x[1], reverse=True)
    top_reps = [r[0] for r in baseline_results[:3]]
    
    print(f"\n  Top performers: {', '.join(top_reps)}")
    
    # STEP 2: Strategic hybrids from top performers
    print(f"\n{'STEP 2: STRATEGIC HYBRIDS (from top performers @ n=1000)':=^80}")
    
    # Top 2 combo
    hybrid_combos = [
        (top_reps[:2], RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), "Top 2 combo with RF"),
        (top_reps[:3], RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), "Top 3 combo with RF"),
    ]
    
    for combo, model, description in hybrid_combos:
        start_hybrid = time.time()
        
        hybrid_dict = {name: reps_train[name][indices] for name in combo}
        X_train_hybrid, feature_info = feature_selector(hybrid_dict, train_labels[indices], n_per_rep, selection_method)
        
        hybrid_dict_test = {name: reps_test[name] for name in combo}
        X_test_hybrid = feature_selector(hybrid_dict_test, None, n_per_rep, selection_method, feature_info=feature_info)
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        metrics = test_single(X_train_hybrid, X_test_hybrid, train_labels[indices],
                             test_labels, model, is_classification=False, model_name=model_name)
        elapsed = time.time() - start_hybrid
        
        result = {
            'dataset': 'esol',
            'representation': '+'.join(combo),
            'n_samples': n_baseline,
            'n_features_input': n_input,
            'n_features_selected': n_selected,
            'feature_reduction': reduction,
            'model': 'RandomForest',
            'is_hybrid': True,
            'time': elapsed,
            **metrics
        }
        results.append(result)
        
        print(f"  {'+'.join(combo):30s} {n_input:5d}→{n_selected:4d} ({reduction:4.1f}%)  R²={metrics['r2']:.4f}  ({elapsed:.2f}s)")
        print(f"    → {description}")
    
    # STEP 3: Model variations on best hybrid
    print(f"\n{'STEP 3: MODEL VARIATIONS (best hybrid with different models)':=^80}")
    
    best_combo = top_reps[:2]  # Top 2 performers
    
    models_to_test = [
        (xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=1), "XGBoost"),
    ]
    
    for model, model_name in models_to_test:
        start_hybrid = time.time()
        
        hybrid_dict = {name: reps_train[name][indices] for name in best_combo}
        X_train_hybrid, feature_info = feature_selector(hybrid_dict, train_labels[indices], n_per_rep, selection_method)
        
        hybrid_dict_test = {name: reps_test[name] for name in best_combo}
        X_test_hybrid = feature_selector(hybrid_dict_test, None, n_per_rep, selection_method, feature_info=feature_info)
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        metrics = test_single(X_train_hybrid, X_test_hybrid, train_labels[indices],
                             test_labels, model, is_classification=False, model_name=model_name)
        elapsed = time.time() - start_hybrid
        
        result = {
            'dataset': 'esol',
            'representation': '+'.join(best_combo),
            'n_samples': n_baseline,
            'n_features_input': n_input,
            'n_features_selected': n_selected,
            'feature_reduction': reduction,
            'model': model_name,
            'is_hybrid': True,
            'time': elapsed,
            **metrics
        }
        results.append(result)
        
        print(f"  {'+'.join(best_combo):30s} with {model_name:12s}  R²={metrics['r2']:.4f}  ({elapsed:.2f}s)")
    
    # ========================================================================
    # hERG - Classification
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET: hERG FLuID (Cardiotoxicity Classification)")
    print(f"{'='*80}")
    
    train_data = load_herg(source='fluid', use_test=False)
    test_data = load_herg(source='fluid', use_test=True)
    train_smiles = train_data['smiles']
    train_labels = train_data['labels']
    test_smiles = test_data['smiles']
    test_labels = test_data['labels']
    
    # Create representations
    print("\nCreating representations...")
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
    print(f"\n{'STEP 1: BASELINE (all reps @ n=1000 with RF)':=^80}")
    
    n_baseline = 1000
    indices = np.random.choice(len(train_labels), min(n_baseline, len(train_labels)), replace=False)
    
    baseline_results = []
    for rep_name in ['ecfp4', 'pdv', 'mol2vec', 'mhggnn']:
        start = time.time()
        
        X_train = reps_train[rep_name][indices]
        y_train = train_labels[indices]
        X_test = reps_test[rep_name]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        metrics = test_single(X_train, X_test, y_train, test_labels, model, is_classification=True, model_name=model_name)
        elapsed = time.time() - start
        
        result = {
            'dataset': 'herg_fluid',
            'representation': rep_name,
            'n_samples': n_baseline,
            'n_features_input': X_train.shape[1],
            'n_features_selected': X_train.shape[1],
            'feature_reduction': 0.0,
            'model': 'RandomForest',
            'is_hybrid': False,
            'time': elapsed,
            **metrics
        }
        results.append(result)
        baseline_results.append((rep_name, metrics['auc']))
        
        print(f"  {rep_name:15s} {X_train.shape[1]:5d} feats  AUC={metrics['auc']:.4f}  ({elapsed:.2f}s)")
    
    baseline_results.sort(key=lambda x: x[1], reverse=True)
    top_reps = [r[0] for r in baseline_results[:3]]
    
    print(f"\n  Top performers: {', '.join(top_reps)}")
    
    # STEP 2: Strategic hybrids
    print(f"\n{'STEP 2: STRATEGIC HYBRIDS (from top performers @ n=1000)':=^80}")
    
    hybrid_combos = [
        (top_reps[:2], RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), "Top 2 combo with RF"),
        (top_reps[:3], RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), "Top 3 combo with RF"),
    ]
    
    for combo, model, description in hybrid_combos:
        start_hybrid = time.time()
        
        hybrid_dict = {name: reps_train[name][indices] for name in combo}
        X_train_hybrid, feature_info = feature_selector(hybrid_dict, train_labels[indices], n_per_rep, selection_method)
        
        hybrid_dict_test = {name: reps_test[name] for name in combo}
        X_test_hybrid = feature_selector(hybrid_dict_test, None, n_per_rep, selection_method, feature_info=feature_info)
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        metrics = test_single(X_train_hybrid, X_test_hybrid, train_labels[indices],
                             test_labels, model, is_classification=True, model_name=model_name)
        elapsed = time.time() - start_hybrid
        
        result = {
            'dataset': 'herg_fluid',
            'representation': '+'.join(combo),
            'n_samples': n_baseline,
            'n_features_input': n_input,
            'n_features_selected': n_selected,
            'feature_reduction': reduction,
            'model': 'RandomForest',
            'is_hybrid': True,
            'time': elapsed,
            **metrics
        }
        results.append(result)
        
        print(f"  {'+'.join(combo):30s} {n_input:5d}→{n_selected:4d} ({reduction:4.1f}%)  AUC={metrics['auc']:.4f}  ({elapsed:.2f}s)")
        print(f"    → {description}")
    
    # STEP 3: Model variations
    print(f"\n{'STEP 3: MODEL VARIATIONS (best hybrid with different models)':=^80}")
    
    best_combo = top_reps[:2]
    
    models_to_test = [
        (xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1), "XGBoost"),
    ]
    
    for model, model_name in models_to_test:
        start_hybrid = time.time()
        
        hybrid_dict = {name: reps_train[name][indices] for name in best_combo}
        X_train_hybrid, feature_info = feature_selector(hybrid_dict, train_labels[indices], n_per_rep, selection_method)
        
        hybrid_dict_test = {name: reps_test[name] for name in best_combo}
        X_test_hybrid = feature_selector(hybrid_dict_test, None, n_per_rep, selection_method, feature_info=feature_info)
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        metrics = test_single(X_train_hybrid, X_test_hybrid, train_labels[indices],
                             test_labels, model, is_classification=True, model_name=model_name)
        elapsed = time.time() - start_hybrid
        
        result = {
            'dataset': 'herg_fluid',
            'representation': '+'.join(best_combo),
            'n_samples': n_baseline,
            'n_features_input': n_input,
            'n_features_selected': n_selected,
            'feature_reduction': reduction,
            'model': model_name,
            'is_hybrid': True,
            'time': elapsed,
            **metrics
        }
        results.append(result)
        
        print(f"  {'+'.join(best_combo):30s} with {model_name:12s}  AUC={metrics['auc']:.4f}  ({elapsed:.2f}s)")
    
    # ========================================================================
    # Analysis
    # ========================================================================
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    df = pd.DataFrame(results)
    
    print(f"\nRan {len(results)} structured experiments in {total_time:.2f}s ({total_time/60:.2f}min)")
    
    print("\n1. BEST PERFORMERS PER DATASET:")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        metric = 'auc' if 'auc' in df_ds.columns else 'r2'
        
        print(f"\n{dataset.upper()}:")
        
        # Best baseline
        best_baseline = df_ds[~df_ds['is_hybrid']].loc[df_ds[~df_ds['is_hybrid']][metric].idxmax()]
        print(f"  Best baseline: {best_baseline['representation']:15s} {metric.upper()}={best_baseline[metric]:.4f}")
        
        # Best hybrid
        if len(df_ds[df_ds['is_hybrid']]) > 0:
            best_hybrid = df_ds[df_ds['is_hybrid']].loc[df_ds[df_ds['is_hybrid']][metric].idxmax()]
            print(f"  Best hybrid:   {best_hybrid['representation']:30s} {metric.upper()}={best_hybrid[metric]:.4f}")
            print(f"                 Features: {best_hybrid['n_features_input']:.0f} → {best_hybrid['n_features_selected']:.0f} ({best_hybrid['feature_reduction']:.1f}% reduction)")
            
            improvement = ((best_hybrid[metric] - best_baseline[metric]) / best_baseline[metric]) * 100
            if improvement > 0:
                print(f"                 ✓ Hybrid wins by +{improvement:.2f}%")
            else:
                print(f"                 Baseline better by {-improvement:.2f}%")
    
    print("\n2. KEY INSIGHTS:")
    print(f"  Average feature reduction: {df[df['is_hybrid']]['feature_reduction'].mean():.1f}%")
    
    # Save
    output = {
        'tier': 'superfast_structured',
        'start_time': datetime.now().isoformat(),
        'total_time': total_time,
        'n_experiments': len(results),
        'feature_selection_method': selection_method,
        'philosophy': f'structured: baseline all reps, strategic hybrids with {selection_method} feature selection',
        'results': results
    }
    
    with open('hybrid_master_superfast_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    df.to_csv('hybrid_master_superfast_results.csv', index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved: hybrid_master_superfast_results.*")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_structured_sampling(create_hybrid_wrapper, n_per_rep=50)