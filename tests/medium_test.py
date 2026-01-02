#!/usr/bin/env python3
"""
KIRBy Hybrid Master - TIER 2 MEDIUM (~5-10 min)

STRUCTURED INTELLIGENT SAMPLING - Expanded

Strategy:
1. Baseline: Test ALL base reps at n=1000 with RF (4 datasets)
2. Identify top performers per dataset
3. Create strategic hybrids from top performers
4. Vary models and sample sizes on best combinations

~60-80 experiments in 5-10 minutes
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
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False
import time
import json
from datetime import datetime
from typing import Dict, Callable

import sys
sys.path.insert(0, '/mnt/user-data/outputs/src')

from kirby.datasets.esol import load_esol_combined
from kirby.datasets.herg import load_herg, get_herg_splits
from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_mol2vec,
    create_mhg_gnn,
    create_graph_kernel
)


def default_feature_selector(rep_dict: Dict[str, np.ndarray],
                             labels: np.ndarray,
                             n_per_rep: int = 50) -> np.ndarray:
    """
    Default KIRBy feature selection (mutual information).
    
    YOUR NEW METHOD GOES HERE!
    
    This includes numerical stability fixes to prevent overflow.
    """
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    
    is_classification = len(np.unique(labels)) < 10
    mi_func = mutual_info_classif if is_classification else mutual_info_regression
    
    selected_features = []
    
    for name, X in rep_dict.items():
        # FIX: Clip extreme values to prevent overflow
        X_clipped = np.clip(X, -1e10, 1e10)
        
        # FIX: Replace inf/nan with reasonable values
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Compute MI with safe values
        mi_scores = mi_func(X_clipped, labels, random_state=42)
        
        # FIX: Handle case where MI calculation failed
        if np.any(np.isnan(mi_scores)) or np.any(np.isinf(mi_scores)):
            mi_scores = np.nan_to_num(mi_scores, nan=0.0)
        
        n_select = min(n_per_rep, len(mi_scores))
        top_idx = np.argsort(mi_scores)[-n_select:]
        
        # Use clipped version for output
        selected_features.append(X_clipped[:, top_idx])
    
    return np.hstack(selected_features)


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


def test_dataset_structured(dataset_name, train_smiles, train_labels, test_smiles, test_labels,
                            is_classification, feature_selector, n_per_rep, include_graph=False):
    """
    Structured testing:
    1. Baseline all reps @ n=1000 with RF
    2. Strategic hybrids from top performers
    3. Model/size variations
    """
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Create base representations
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
    
    # Add Graph Kernel at n=1000 (fast enough)
    if include_graph:
        print("  Adding Graph Kernel...")
        graph_train, vocab = create_graph_kernel(
            train_smiles[:1000], kernel='weisfeiler_lehman', n_features=100, return_vocabulary=True
        )
        graph_test = create_graph_kernel(
            test_smiles, kernel='weisfeiler_lehman', n_features=100, reference_vocabulary=vocab
        )
        reps_train['graph_kernel'] = graph_train
        reps_test['graph_kernel'] = graph_test
    
    print(f"Representations created ({time.time() - start:.2f}s)")
    for name, X in reps_train.items():
        print(f"  {name}: {X.shape}")
    
    # STEP 1: Baseline - all reps at n=1000 with RF
    print(f"\n{'STEP 1: BASELINE (all reps @ n=1000 with RF)':=^80}")
    
    n_baseline = 1000
    indices = np.random.choice(len(train_labels), min(n_baseline, len(train_labels)), replace=False)
    
    baseline_results = []
    rep_names = list(reps_train.keys())
    
    for rep_name in rep_names:
        start = time.time()
        
        # Handle graph kernel special case
        if rep_name == 'graph_kernel':
            X_train = reps_train[rep_name]  # Already subsampled to 1000
            y_train = train_labels[:n_baseline]
        else:
            X_train = reps_train[rep_name][indices]
            y_train = train_labels[indices]
        
        X_test = reps_test[rep_name]
        
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        metrics = test_single(X_train, X_test, y_train, test_labels, model, is_classification)
        elapsed = time.time() - start
        
        metric_val = metrics['auc'] if is_classification else metrics['r2']
        metric_name = 'AUC' if is_classification else 'R²'
        
        result = {
            'dataset': dataset_name,
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
        baseline_results.append((rep_name, metric_val))
        
        print(f"  {rep_name:15s} {X_train.shape[1]:5d} feats  {metric_name}={metric_val:.4f}  ({elapsed:.2f}s)")
    
    # Sort by performance
    baseline_results.sort(key=lambda x: x[1], reverse=True)
    top_reps = [r[0] for r in baseline_results[:3]]
    
    print(f"\n  → Top performers: {', '.join(top_reps)}")
    
    # STEP 2: Strategic hybrids from top performers
    print(f"\n{'STEP 2: STRATEGIC HYBRIDS (from top performers @ n=1000)':=^80}")
    
    hybrid_combos = [
        (top_reps[:2], n_baseline, RandomForestClassifier if is_classification else RandomForestRegressor, "Top 2 with RF"),
        (top_reps[:3], n_baseline, RandomForestClassifier if is_classification else RandomForestRegressor, "Top 3 with RF"),
        (['ecfp4', 'pdv'], n_baseline, RandomForestClassifier if is_classification else RandomForestRegressor, "Classic (ECFP4+PDV) with RF"),
    ]
    
    for combo, n_samples, model_class, description in hybrid_combos:
        # Skip if we don't have these reps
        if not all(c in reps_train for c in combo):
            continue
        
        start_hybrid = time.time()
        
        # Get indices for this sample size
        if n_samples != n_baseline:
            sample_indices = np.random.choice(len(train_labels), min(n_samples, len(train_labels)), replace=False)
        else:
            sample_indices = indices
        
        hybrid_dict = {name: reps_train[name][sample_indices] if name != 'graph_kernel' else reps_train[name]
                       for name in combo}
        y_train = train_labels[sample_indices] if 'graph_kernel' not in combo else train_labels[:n_samples]
        
        X_train_hybrid = feature_selector(hybrid_dict, y_train, n_per_rep)
        
        hybrid_dict_test = {name: reps_test[name] for name in combo}
        X_test_hybrid = feature_selector(hybrid_dict_test, test_labels, n_per_rep)
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        model = model_class(n_estimators=100, random_state=42, n_jobs=-1)
        metrics = test_single(X_train_hybrid, X_test_hybrid, y_train, test_labels, model, is_classification)
        elapsed = time.time() - start_hybrid
        
        metric_val = metrics['auc'] if is_classification else metrics['r2']
        metric_name = 'AUC' if is_classification else 'R²'
        
        result = {
            'dataset': dataset_name,
            'representation': '+'.join(combo),
            'n_samples': n_samples,
            'n_features_input': n_input,
            'n_features_selected': n_selected,
            'feature_reduction': reduction,
            'model': 'RandomForest',
            'is_hybrid': True,
            'time': elapsed,
            **metrics
        }
        results.append(result)
        
        print(f"  {'+'.join(combo):35s} {n_input:5d}→{n_selected:4d} ({reduction:4.1f}%)  {metric_name}={metric_val:.4f}  ({elapsed:.2f}s)")
        print(f"    → {description}")
    
    # STEP 3: Model variations on best hybrid
    print(f"\n{'STEP 3: MODEL & SIZE VARIATIONS':=^80}")
    
    best_combo = top_reps[:2]
    
    # Model variations at n=1000
    test_configs = [
        (best_combo, 1000, xgb.XGBClassifier if is_classification else xgb.XGBRegressor, "XGBoost @ n=1000"),
    ]
    
    if HAS_LIGHTGBM:
        test_configs.append(
            (best_combo, 1000, lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor, "LightGBM @ n=1000")
        )
    
    # Size variations with RF
    test_configs.extend([
        (best_combo, 500, RandomForestClassifier if is_classification else RandomForestRegressor, "RF @ n=500"),
        (best_combo, 2000, RandomForestClassifier if is_classification else RandomForestRegressor, "RF @ n=2000"),
    ])
    
    for combo, n_samples, model_class, description in test_configs:
        if n_samples > len(train_labels):
            continue
        
        start_hybrid = time.time()
        
        sample_indices = np.random.choice(len(train_labels), min(n_samples, len(train_labels)), replace=False)
        
        hybrid_dict = {name: reps_train[name][sample_indices] for name in combo}
        y_train = train_labels[sample_indices]
        
        X_train_hybrid = feature_selector(hybrid_dict, y_train, n_per_rep)
        
        hybrid_dict_test = {name: reps_test[name] for name in combo}
        X_test_hybrid = feature_selector(hybrid_dict_test, test_labels, n_per_rep)
        
        n_input = sum(hybrid_dict[k].shape[1] for k in hybrid_dict)
        n_selected = X_train_hybrid.shape[1]
        reduction = (1 - n_selected / n_input) * 100
        
        if 'XGB' in description:
            model = model_class(n_estimators=100, random_state=42, n_jobs=1)
        elif 'LightGBM' in description:
            model = model_class(n_estimators=100, random_state=42, n_jobs=1, verbose=-1)
        else:
            model = model_class(n_estimators=100, random_state=42, n_jobs=-1)
        
        metrics = test_single(X_train_hybrid, X_test_hybrid, y_train, test_labels, model, is_classification)
        elapsed = time.time() - start_hybrid
        
        metric_val = metrics['auc'] if is_classification else metrics['r2']
        metric_name = 'AUC' if is_classification else 'R²'
        
        model_name = description.split('@')[0].strip()
        
        result = {
            'dataset': dataset_name,
            'representation': '+'.join(combo),
            'n_samples': n_samples,
            'n_features_input': n_input,
            'n_features_selected': n_selected,
            'feature_reduction': reduction,
            'model': model_name,
            'is_hybrid': True,
            'time': elapsed,
            **metrics
        }
        results.append(result)
        
        print(f"  {'+'.join(combo):35s} {description:20s}  {metric_name}={metric_val:.4f}  ({elapsed:.2f}s)")
    
    return results


def run_structured_exploration(feature_selector, n_per_rep=50):
    """
    Structured exploration across 4 datasets
    """
    
    print("="*80)
    print("TIER 2: STRUCTURED INTELLIGENT SAMPLING")
    print("="*80)
    print("\nStrategy:")
    print("  1. Baseline: All reps @ n=1000 with RF (4 datasets)")
    print("  2. Strategic hybrids from top performers")
    print("  3. Model & size variations on best")
    print("="*80)
    
    all_results = []
    overall_start = time.time()
    
    # ========================================================================
    # ESOL
    # ========================================================================
    data = load_esol_combined(splitter='scaffold')
    results = test_dataset_structured(
        'esol',
        data['train']['smiles'], data['train']['labels'],
        data['test']['smiles'], data['test']['labels'],
        is_classification=False,
        feature_selector=feature_selector,
        n_per_rep=n_per_rep,
        include_graph=True
    )
    all_results.extend(results)
    
    # ========================================================================
    # hERG FLuID
    # ========================================================================
    train_data = load_herg(source='fluid', use_test=False)
    test_data = load_herg(source='fluid', use_test=True)
    results = test_dataset_structured(
        'herg_fluid',
        train_data['smiles'], train_data['labels'],
        test_data['smiles'], test_data['labels'],
        is_classification=True,
        feature_selector=feature_selector,
        n_per_rep=n_per_rep,
        include_graph=False
    )
    all_results.extend(results)
    
    # ========================================================================
    # hERG ChEMBL
    # ========================================================================
    data = load_herg(source='chembl')
    splits = get_herg_splits(data, splitter='scaffold')
    train_smiles = splits['train']['smiles'] + splits['val']['smiles']
    train_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
    results = test_dataset_structured(
        'herg_chembl',
        train_smiles, train_labels,
        splits['test']['smiles'], splits['test']['labels'],
        is_classification=True,
        feature_selector=feature_selector,
        n_per_rep=n_per_rep,
        include_graph=False
    )
    all_results.extend(results)
    
    # ========================================================================
    # QM9
    # ========================================================================
    raw_data = load_qm9(n_samples=5000, property_idx=4)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    train_smiles = splits['train']['smiles'] + splits['val']['smiles']
    train_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
    results = test_dataset_structured(
        'qm9',
        train_smiles, train_labels,
        splits['test']['smiles'], splits['test']['labels'],
        is_classification=False,
        feature_selector=feature_selector,
        n_per_rep=n_per_rep,
        include_graph=False
    )
    all_results.extend(results)
    
    # ========================================================================
    # Analysis
    # ========================================================================
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'='*80}")
    
    df = pd.DataFrame(all_results)
    
    print(f"\nRan {len(all_results)} structured experiments in {total_time:.2f}s ({total_time/60:.2f}min)")
    
    print("\n1. BEST PERFORMERS PER DATASET:")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        metric = 'auc' if 'auc' in df_ds.columns else 'r2'
        
        print(f"\n{dataset.upper()}:")
        
        best_baseline = df_ds[~df_ds['is_hybrid']].loc[df_ds[~df_ds['is_hybrid']][metric].idxmax()]
        print(f"  Best baseline: {best_baseline['representation']:15s} {metric.upper()}={best_baseline[metric]:.4f}")
        
        if len(df_ds[df_ds['is_hybrid']]) > 0:
            best_hybrid = df_ds[df_ds['is_hybrid']].loc[df_ds[df_ds['is_hybrid']][metric].idxmax()]
            print(f"  Best hybrid:   {best_hybrid['representation']:35s} {metric.upper()}={best_hybrid[metric]:.4f}")
            print(f"                 Model: {best_hybrid['model']}, n={best_hybrid['n_samples']:.0f}")
            print(f"                 Features: {best_hybrid['n_features_input']:.0f} → {best_hybrid['n_features_selected']:.0f} ({best_hybrid['feature_reduction']:.1f}%)")
            
            improvement = ((best_hybrid[metric] - best_baseline[metric]) / best_baseline[metric]) * 100
            if improvement > 0:
                print(f"                 ✓ Hybrid wins by +{improvement:.2f}%")
            else:
                print(f"                 Baseline better by {-improvement:.2f}%")
    
    print("\n2. SAMPLE SIZE EFFECTS (best hybrid):")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_hyb = df_ds[df_ds['is_hybrid']]
        
        if len(df_hyb) > 0:
            metric = 'auc' if 'auc' in df_hyb.columns else 'r2'
            
            print(f"\n{dataset.upper()}:")
            for n in sorted(df_hyb['n_samples'].unique()):
                df_n = df_hyb[df_hyb['n_samples'] == n]
                if len(df_n) > 0:
                    best = df_n.loc[df_n[metric].idxmax()]
                    print(f"  n={n:4.0f}: {metric.upper()}={best[metric]:.4f}  ({best['representation']})")
    
    print("\n3. MODEL COMPARISON (on hybrids):")
    for model in sorted(df[df['is_hybrid']]['model'].unique()):
        df_m = df[df['model'] == model]
        metric = 'auc' if 'auc' in df_m.columns else 'r2'
        print(f"  {model:15s}: avg {metric.upper()}={df_m[metric].mean():.4f}")
    
    # Save
    output = {
        'tier': 'medium_structured',
        'start_time': datetime.now().isoformat(),
        'total_time': total_time,
        'n_experiments': len(all_results),
        'philosophy': 'structured: baseline all reps @ n=1000, strategic hybrids, model/size variations',
        'results': all_results
    }
    
    with open('hybrid_master_medium_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    df.to_csv('hybrid_master_medium_results.csv', index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved: hybrid_master_medium_results.*")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_structured_exploration(default_feature_selector, n_per_rep=50)