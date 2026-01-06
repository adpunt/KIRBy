#!/usr/bin/env python3
"""
KIRBy Hybrid Master - COMPREHENSIVE EVALUATION (YOUR ORIGINAL + QM9 + EXTENDED)

THIS PRESERVES ALL YOUR ORIGINAL CODE AND ADDS:
1. QM9 HOMO-LUMO gap (scaffold split) - NEWLY ADDED PRIMARY TEST
2. ESOL solubility (scaffold split) - YOUR ORIGINAL
3. hERG cardiotoxicity (ChEMBL + FLuID) - EXTENDED WITH BOTH SOURCES
4. Additional representations (Graph Kernel, GAUCHE GP when --server flag)
5. Server flag for GPU-heavy methods (MoLFormer, ChemBERTa)

Plus all your original features:
- Sanity check (PDV+mhggnn alignment verification)
- All baseline reps with RF, XGBoost, MLP
- Full n_per_rep sweep [25, 50, 100, 200, 500, 1000, -1]
- Comprehensive analysis and comparisons

Runtime:
- Without --server: ~15-20 minutes
- With --server: ~45-60 minutes (includes fine-tuning)

Usage:
    python hybrid_master_comprehensive.py                    # Fast, no fine-tuning
    python hybrid_master_comprehensive.py --server          # Full evaluation with fine-tuning
    python hybrid_master_comprehensive.py --server --herg-sources fluid  # Only FLuID hERG
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
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import time
import json
from datetime import datetime
from typing import Dict, Callable

import sys
sys.path.insert(0, '/mnt/user-data/outputs/src')

from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.datasets.esol import load_esol_combined
from kirby.datasets.herg import load_herg, load_herg_fluid
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_mol2vec,
    create_graph_kernel,
    create_mhg_gnn,
    finetune_molformer,
    finetune_chemberta,
    finetune_gnn,
    encode_from_model,
    train_gauche_gp,
    predict_gauche_gp
)


def create_hybrid_wrapper(rep_dict: Dict[str, np.ndarray],
                         labels: np.ndarray,
                         n_per_rep: int = 50,
                         selection_method: str = 'tree_importance',
                         feature_info: Dict = None):
    """
    Wrapper for feature selection with train/test alignment.
    
    ALWAYS returns: (hybrid_features, feature_info)
    For test set calls, feature_info is ignored in output.
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Special case: keep everything (no selection at all)
    if n_per_rep == -1:
        all_features = []
        for name, X in rep_dict.items():
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
            all_features.append(X_clipped)
        result = np.hstack(all_features)
        return result, feature_info  # Return same feature_info that was passed
    
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
    try:
        from kirby.representations.hybrid import create_hybrid
        return create_hybrid(
            base_reps=clean_reps,
            labels=labels,
            n_per_rep=n_per_rep,
            importance_method='random_forest'
        )
    except ImportError:
        pass
    
    # Inline RF implementation
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
    """Test single representation with appropriate scaler for model type"""
    from sklearn.preprocessing import RobustScaler
    
    # Check for NaN/inf in inputs - PRINT AND CRASH
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print(f"  ERROR: NaN/inf in X_train for {model_name}")
        print(f"    NaN count: {np.sum(np.isnan(X_train))}, Inf count: {np.sum(np.isinf(X_train))}")
        raise ValueError(f"NaN/inf in X_train for {model_name}")
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        print(f"  ERROR: NaN/inf in X_test for {model_name}")
        raise ValueError(f"NaN/inf in X_test for {model_name}")
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        print(f"  ERROR: NaN/inf in y_train for {model_name}")
        raise ValueError(f"NaN/inf in y_train for {model_name}")
    if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
        print(f"  ERROR: NaN/inf in y_test for {model_name}")
        raise ValueError(f"NaN/inf in y_test for {model_name}")
    
    # Use RobustScaler for MLP (handles outliers better), StandardScaler for others
    if 'MLP' in model_name:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Check predictions - CRASH if bad
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print(f"  ERROR: NaN/inf in predictions for {model_name}")
        print(f"    NaN count: {np.sum(np.isnan(y_pred))}, Inf count: {np.sum(np.isinf(y_pred))}")
        raise ValueError(f"NaN/inf in predictions for {model_name}")
    
    if is_classification:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        return evaluate_classification(y_test, y_pred, y_prob)
    else:
        return evaluate_regression(y_test, y_pred)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Comprehensive KIRBy evaluation across QM9, ESOL, and hERG datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast evaluation (no fine-tuning, ~15-20 min)
  python hybrid_master_comprehensive.py
  
  # Full evaluation with fine-tuning (~45-60 min, requires GPU)
  python hybrid_master_comprehensive.py --server
  
  # Test only FLuID hERG
  python hybrid_master_comprehensive.py --herg-sources fluid
  
  # Test both ChEMBL and FLuID hERG with fine-tuning
  python hybrid_master_comprehensive.py --server --herg-sources chembl fluid
"""
    )
    
    parser.add_argument('--server', action='store_true',
                       help='Enable GPU-heavy fine-tuning (MoLFormer, ChemBERTa). Requires CUDA.')
    parser.add_argument('--herg-sources', nargs='+', 
                       choices=['chembl', 'fluid'],
                       default=['fluid'],
                       help='hERG data sources to test (default: fluid)')
    parser.add_argument('--n-per-rep', nargs='+', type=int,
                       default=[25, 50, 100, 200, 500, 1000, -1],
                       help='n_per_rep values to test (default: 25 50 100 200 500 1000 -1)')
    parser.add_argument('--selection-method', choices=['tree_importance'],
                       default='tree_importance',
                       help='Feature selection method (default: tree_importance)')
    
    return parser.parse_args()


def test_single(X_train, X_test, y_train, y_test, model, is_classification, model_name=''):
    """Test single representation with appropriate scaler for model type"""
    from sklearn.preprocessing import RobustScaler
    
    # Check for NaN/inf in inputs - PRINT AND CRASH
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print(f"  ERROR: NaN/inf in X_train for {model_name}")
        print(f"    NaN count: {np.sum(np.isnan(X_train))}, Inf count: {np.sum(np.isinf(X_train))}")
        raise ValueError(f"NaN/inf in X_train for {model_name}")
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        print(f"  ERROR: NaN/inf in X_test for {model_name}")
        raise ValueError(f"NaN/inf in X_test for {model_name}")
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        print(f"  ERROR: NaN/inf in y_train for {model_name}")
        raise ValueError(f"NaN/inf in y_train for {model_name}")
    if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
        print(f"  ERROR: NaN/inf in y_test for {model_name}")
        raise ValueError(f"NaN/inf in y_test for {model_name}")
    
    # Use RobustScaler for MLP (handles outliers better), StandardScaler for others
    if 'MLP' in model_name:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Check predictions - CRASH if bad
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print(f"  ERROR: NaN/inf in predictions for {model_name}")
        print(f"    NaN count: {np.sum(np.isnan(y_pred))}, Inf count: {np.sum(np.isinf(y_pred))}")
        raise ValueError(f"NaN/inf in predictions for {model_name}")
    
    if is_classification:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        return evaluate_classification(y_test, y_pred, y_prob)
    else:
        return evaluate_regression(y_test, y_pred)


def run_structured_sampling(feature_selector, args):
    """
    YOUR ORIGINAL COMPREHENSIVE TEST + QM9 + EXTENDED CAPABILITIES
    
    Args:
        feature_selector: Feature selection function
        args: Command-line arguments (from parse_args)
    """
    
    n_per_rep_values = args.n_per_rep
    selection_method = args.selection_method
    use_server = args.server
    herg_sources = args.herg_sources
    
    print("="*100)
    print("COMPREHENSIVE EVALUATION: ALL DATASETS + ALL MODELS + EXTENDED")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Feature selection: {selection_method}")
    print(f"  Server mode (GPU fine-tuning): {use_server}")
    print(f"  hERG sources: {', '.join(herg_sources)}")
    print(f"\nStrategy:")
    print("  0. SANITY CHECK: Verify train/test alignment")
    print("  1. QM9 HOMO-LUMO GAP (PRIMARY)")
    print("  2. ESOL: Solubility regression")
    print(f"  3. hERG: Cardiotoxicity classification ({', '.join(herg_sources)})")
    print("  4. All with: RF, XGBoost, MLP")
    print(f"  5. Full n_per_rep sweep: {n_per_rep_values}")
    if use_server:
        print("  6. EXTENDED: MoLFormer, ChemBERTa, GAUCHE GP fine-tuning")
    print("="*100)
    
    results = []
    overall_start = time.time()
    
    # ========================================================================
    # SANITY CHECK: PDV+mhggnn with n_per_rep=1000 vs -1 (YOUR ORIGINAL)
    # ========================================================================
    print(f"\n{'='*100}")
    print("SANITY CHECK: Verify train/test feature alignment fix (YOUR ORIGINAL)")
    print(f"{'='*100}")
    print("\nTesting PDV+mhggnn hybrid:")
    print("  n_per_rep=1000: Should give ~1200 features (200+1000)")
    print("  n_per_rep=-1:   Should give 1224 features (200+1024)")
    print("  Performance should be SIMILAR (not 0.78 vs 0.89)")
    print(f"{'='*100}")
    
    # Use hERG FLuID for this test (has pre-split train/test)
    train_data = load_herg_fluid(use_test=False)  # Training set
    test_data = load_herg_fluid(use_test=True)    # Test set
    
    train_smiles_full = train_data['smiles']
    train_labels_full = train_data['labels']
    test_smiles = test_data['smiles']
    test_labels = test_data['labels']
    
    # Subsample
    n_sanity = 1000
    indices = np.random.choice(len(train_labels_full), min(n_sanity, len(train_labels_full)), replace=False)
    train_smiles = [train_smiles_full[i] for i in indices]
    train_labels = train_labels_full[indices]
    
    # Create ONLY PDV and mhggnn
    print("\nCreating representations (PDV and mhggnn only)...")
    start = time.time()
    reps_train = {
        'pdv': create_pdv(train_smiles),
        'mhggnn': create_mhg_gnn(train_smiles)
    }
    reps_test = {
        'pdv': create_pdv(test_smiles),
        'mhggnn': create_mhg_gnn(test_smiles)
    }
    print(f"Representations created ({time.time() - start:.2f}s)")
    print(f"  PDV shape: {reps_train['pdv'].shape}")
    print(f"  mhggnn shape: {reps_train['mhggnn'].shape}")
    
    # Test with RF only (quick)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    for n_per_rep in [1000, -1]:
        print(f"\n  Testing n_per_rep = {n_per_rep}:")
        
        hybrid_dict = {'pdv': reps_train['pdv'], 'mhggnn': reps_train['mhggnn']}
        hybrid_dict_test = {'pdv': reps_test['pdv'], 'mhggnn': reps_test['mhggnn']}
        
        # Get hybrid features with proper train/test alignment
        X_train_hybrid, feature_info = feature_selector(hybrid_dict, train_labels, n_per_rep, selection_method)
        X_test_hybrid, _ = feature_selector(hybrid_dict_test, None, n_per_rep, selection_method, feature_info=feature_info)
        
        total_features = reps_train['pdv'].shape[1] + reps_train['mhggnn'].shape[1]
        reduction = 100 * (1 - X_train_hybrid.shape[1] / total_features)
        
        print(f"    Features: {total_features} → {X_train_hybrid.shape[1]} ({reduction:.1f}% reduction)")
        
        # Verify train and test have same shape
        assert X_train_hybrid.shape[1] == X_test_hybrid.shape[1], \
            f"MISMATCH! Train has {X_train_hybrid.shape[1]} features, test has {X_test_hybrid.shape[1]}"
        
        start = time.time()
        metrics = test_single(X_train_hybrid, X_test_hybrid, train_labels, test_labels, 
                             model, is_classification=True, model_name='RF')
        elapsed = time.time() - start
        
        print(f"    [RF] AUC={metrics['auc']:.4f}  ({elapsed:.2f}s)")
    
    print(f"\n{'='*100}")
    print("SANITY CHECK COMPLETE")
    print("If AUC values are similar (~0.02 difference), the fix is working!")
    print("If AUC values differ by >0.05, there's still a bug.")
    print(f"{'='*100}\n")
    
    # ========================================================================
    # QM9 HOMO-LUMO GAP - NEWLY ADDED PRIMARY TEST
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET 0: QM9 HOMO-LUMO Gap (Scaffold Split) - NEWLY ADDED PRIMARY TEST")
    print(f"{'='*80}")
    
    # Load QM9 with HOMO-LUMO gap (property_idx=4)
    raw_data = load_qm9(n_samples=2000, property_idx=4)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    # Combine train+val for final testing
    train_smiles_full = splits['train']['smiles'] + splits['val']['smiles']
    train_labels_full = np.concatenate([splits['train']['labels'], splits['val']['labels']])
    test_smiles = splits['test']['smiles']
    test_labels = splits['test']['labels']
    
    print(f"Target: {raw_data['property_name']} (eV)")
    print(f"Train: {len(train_smiles_full)}, Test: {len(test_smiles)}")
    
    # SUBSAMPLE FIRST!
    n_baseline = 1000
    print(f"\nSubsampling to n={n_baseline} before encoding...")
    indices = np.random.choice(len(train_labels_full), min(n_baseline, len(train_labels_full)), replace=False)
    train_smiles = [train_smiles_full[i] for i in indices]
    train_labels = train_labels_full[indices]
    
    # Create validation split for fine-tuning
    n_val = len(train_smiles) // 5
    val_smiles = train_smiles[:n_val]
    val_labels = train_labels[:n_val]
    train_smiles_fit = train_smiles[n_val:]
    train_labels_fit = train_labels[n_val:]
    
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
    
    # Add graph kernel
    print("  Adding Graph Kernel...")
    graphkernel_train, vocab = create_graph_kernel(
        train_smiles,
        kernel='weisfeiler_lehman',
        n_iter=5,
        return_vocabulary=True
    )
    graphkernel_test = create_graph_kernel(
        test_smiles,
        kernel='weisfeiler_lehman',
        n_iter=5,
        reference_vocabulary=vocab
    )
    reps_train['graphkernel'] = graphkernel_train
    reps_test['graphkernel'] = graphkernel_test
    
    # Add fine-tuned representations if server mode
    if use_server:
        print("\n  SERVER MODE: Adding fine-tuned representations...")
        
        # GNN
        print("    Fine-tuning GNN...")
        gnn_model = finetune_gnn(
            train_smiles_fit, train_labels_fit,
            val_smiles, val_labels,
            gnn_type='gcn', hidden_dim=128, epochs=50  # Match test_qm9.py (TODO: 150 for production)
        )
        reps_train['gnn'] = encode_from_model(gnn_model, train_smiles)
        reps_test['gnn'] = encode_from_model(gnn_model, test_smiles)
        
        # MoLFormer (SMILES only - no SELFIES for QM9)
        print("    Fine-tuning MoLFormer (SMILES only)...")
        molformer_model = finetune_molformer(
            train_smiles_fit, train_labels_fit,
            val_smiles, val_labels,
            epochs=3, batch_size=32, use_selfies=False  # TODO: 10 epochs for production
        )
        reps_train['molformer'] = encode_from_model(molformer_model, train_smiles)
        reps_test['molformer'] = encode_from_model(molformer_model, test_smiles)
        
        # ChemBERTa (SMILES only)
        print("    Fine-tuning ChemBERTa...")
        chemberta_model = finetune_chemberta(
            train_smiles_fit, train_labels_fit,
            val_smiles, val_labels,
            epochs=3, batch_size=32, use_selfies=False  # TODO: 10 epochs for production
        )
        reps_train['chemberta'] = encode_from_model(chemberta_model, train_smiles)
        reps_test['chemberta'] = encode_from_model(chemberta_model, test_smiles)
        
        # GAUCHE GP - test standalone (like test_qm9.py)
        print("    Training GAUCHE GP (Weisfeiler-Lehman kernel)...")
        gp_dict = train_gauche_gp(
            train_smiles_fit, train_labels_fit,
            kernel='weisfeiler_lehman',
            num_epochs=50
        )
        gp_predictions = predict_gauche_gp(gp_dict, test_smiles)
        
        # Test GAUCHE GP immediately
        from sklearn.metrics import mean_absolute_error
        gp_mae = mean_absolute_error(test_labels, gp_predictions['predictions'])
        gp_r2 = r2_score(test_labels, gp_predictions['predictions'])
        print(f"    [GAUCHE GP] MAE={gp_mae:.4f}, R²={gp_r2:.4f}")
        
        results.append({
            'dataset': 'qm9_homo_lumo_gap',
            'representation': 'gauche_gp',
            'n_samples': n_baseline,
            'n_features_input': 0,  # GP doesn't use features
            'n_features_selected': 0,
            'feature_reduction': 0.0,
            'model': 'GP',
            'is_hybrid': False,
            'time': 0,
            'mae': gp_mae,
            'rmse': np.sqrt(mean_squared_error(test_labels, gp_predictions['predictions'])),
            'r2': gp_r2
        })
    
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
    rep_names = list(reps_train.keys())
    
    for rep_name in rep_names:
        print(f"\n  {rep_name} ({reps_train[rep_name].shape[1]} features):")
        
        X_train = reps_train[rep_name]
        X_test = reps_test[rep_name]
        
        for model_name, model in models_to_test:
            start = time.time()
            
            metrics = test_single(X_train, X_test, train_labels, test_labels, model, is_classification=False, model_name=model_name)
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
        print(f"\n  n_per_rep = {n_per_rep if n_per_rep != -1 else 'ALL'}:")
        
        # Create hybrid once
        hybrid_dict = {name: reps_train[name] for name in best_combo}
        X_train_hybrid, feature_info = feature_selector(hybrid_dict, train_labels, n_per_rep, selection_method)
        
        hybrid_dict_test = {name: reps_test[name] for name in best_combo}
        X_test_hybrid, _ = feature_selector(hybrid_dict_test, None, n_per_rep, selection_method, feature_info=feature_info)
        
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
    
    # STEP 3: Test specific hybrid combinations (from test_qm9.py pattern)
    print(f"\n{'STEP 3: SPECIFIC HYBRID PATTERNS (test_qm9.py style)':=^80}")
    
    # Hybrid 1: ECFP4 + PDV (classic baseline - always available)
    print("\n  [Hybrid 1] ECFP4 + PDV...")
    hybrid_dict = {'ecfp4': reps_train['ecfp4'], 'pdv': reps_train['pdv']}
    X_train_h1, feature_info = feature_selector(hybrid_dict, train_labels, -1, selection_method)
    
    hybrid_dict_test = {'ecfp4': reps_test['ecfp4'], 'pdv': reps_test['pdv']}
    X_test_h1, _ = feature_selector(hybrid_dict_test, None, -1, selection_method, feature_info)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    metrics = test_single(X_train_h1, X_test_h1, train_labels, test_labels, rf, is_classification=False, model_name='RF')
    print(f"    R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
    
    results.append({
        'dataset': 'qm9_homo_lumo_gap',
        'representation': 'ecfp4+pdv',
        'n_samples': n_baseline,
        'n_features_input': reps_train['ecfp4'].shape[1] + reps_train['pdv'].shape[1],
        'n_features_selected': X_train_h1.shape[1],
        'feature_reduction': (1 - X_train_h1.shape[1] / (reps_train['ecfp4'].shape[1] + reps_train['pdv'].shape[1])) * 100,
        'n_per_rep': -1,
        'model': 'RF',
        'is_hybrid': True,
        'time': 0,
        **metrics
    })
    
    # Hybrid 2: ECFP4 + Graph Kernel (always available)
    print("\n  [Hybrid 2] ECFP4 + Graph Kernel...")
    hybrid_dict = {'ecfp4': reps_train['ecfp4'], 'graphkernel': reps_train['graphkernel']}
    X_train_h2, feature_info = feature_selector(hybrid_dict, train_labels, -1, selection_method)
    
    hybrid_dict_test = {'ecfp4': reps_test['ecfp4'], 'graphkernel': reps_test['graphkernel']}
    X_test_h2, _ = feature_selector(hybrid_dict_test, None, -1, selection_method, feature_info)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    metrics = test_single(X_train_h2, X_test_h2, train_labels, test_labels, rf, is_classification=False, model_name='RF')
    print(f"    R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
    
    results.append({
        'dataset': 'qm9_homo_lumo_gap',
        'representation': 'ecfp4+graphkernel',
        'n_samples': n_baseline,
        'n_features_input': reps_train['ecfp4'].shape[1] + reps_train['graphkernel'].shape[1],
        'n_features_selected': X_train_h2.shape[1],
        'feature_reduction': (1 - X_train_h2.shape[1] / (reps_train['ecfp4'].shape[1] + reps_train['graphkernel'].shape[1])) * 100,
        'n_per_rep': -1,
        'model': 'RF',
        'is_hybrid': True,
        'time': 0,
        **metrics
    })
    
    # Extended hybrids only in server mode
    if use_server:
        # Hybrid 3: All Neural (MoLFormer + ChemBERTa + GNN)
        print("\n  [Hybrid 3] All Neural (MoLFormer + ChemBERTa + GNN)...")
        hybrid_dict = {
            'molformer': reps_train['molformer'],
            'chemberta': reps_train['chemberta'],
            'gnn': reps_train['gnn']
        }
        X_train_h3, feature_info = feature_selector(hybrid_dict, train_labels, -1, selection_method)
        
        hybrid_dict_test = {
            'molformer': reps_test['molformer'],
            'chemberta': reps_test['chemberta'],
            'gnn': reps_test['gnn']
        }
        X_test_h3, _ = feature_selector(hybrid_dict_test, None, -1, selection_method, feature_info)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        metrics = test_single(X_train_h3, X_test_h3, train_labels, test_labels, rf, is_classification=False, model_name='RF')
        print(f"    R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
        
        results.append({
            'dataset': 'qm9_homo_lumo_gap',
            'representation': 'all_neural',
            'n_samples': n_baseline,
            'n_features_input': sum(hybrid_dict[k].shape[1] for k in hybrid_dict),
            'n_features_selected': X_train_h3.shape[1],
            'feature_reduction': (1 - X_train_h3.shape[1] / sum(hybrid_dict[k].shape[1] for k in hybrid_dict)) * 100,
            'n_per_rep': -1,
            'model': 'RF',
            'is_hybrid': True,
            'time': 0,
            **metrics
        })
        
        # Hybrid 4: Everything
        print("\n  [Hybrid 4] Everything...")
        hybrid_dict = {
            'ecfp4': reps_train['ecfp4'],
            'pdv': reps_train['pdv'],
            'mol2vec': reps_train['mol2vec'],
            'graphkernel': reps_train['graphkernel'],
            'mhggnn': reps_train['mhggnn'],
            'molformer': reps_train['molformer'],
            'chemberta': reps_train['chemberta'],
            'gnn': reps_train['gnn']
        }
        X_train_h4, feature_info = feature_selector(hybrid_dict, train_labels, -1, selection_method)
        
        hybrid_dict_test = {
            'ecfp4': reps_test['ecfp4'],
            'pdv': reps_test['pdv'],
            'mol2vec': reps_test['mol2vec'],
            'graphkernel': reps_test['graphkernel'],
            'mhggnn': reps_test['mhggnn'],
            'molformer': reps_test['molformer'],
            'chemberta': reps_test['chemberta'],
            'gnn': reps_test['gnn']
        }
        X_test_h4, _ = feature_selector(hybrid_dict_test, None, -1, selection_method, feature_info)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        metrics = test_single(X_train_h4, X_test_h4, train_labels, test_labels, rf, is_classification=False, model_name='RF')
        print(f"    R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
        
        results.append({
            'dataset': 'qm9_homo_lumo_gap',
            'representation': 'everything',
            'n_samples': n_baseline,
            'n_features_input': sum(hybrid_dict[k].shape[1] for k in hybrid_dict),
            'n_features_selected': X_train_h4.shape[1],
            'feature_reduction': (1 - X_train_h4.shape[1] / sum(hybrid_dict[k].shape[1] for k in hybrid_dict)) * 100,
            'n_per_rep': -1,
            'model': 'RF',
            'is_hybrid': True,
            'time': 0,
            **metrics
        })
    else:
        # Without server mode, test "Everything" with available reps only
        print("\n  [Hybrid 3] Everything (available reps only, no fine-tuning)...")
        hybrid_dict = {
            'ecfp4': reps_train['ecfp4'],
            'pdv': reps_train['pdv'],
            'mol2vec': reps_train['mol2vec'],
            'graphkernel': reps_train['graphkernel'],
            'mhggnn': reps_train['mhggnn']
        }
        X_train_h3, feature_info = feature_selector(hybrid_dict, train_labels, -1, selection_method)
        
        hybrid_dict_test = {
            'ecfp4': reps_test['ecfp4'],
            'pdv': reps_test['pdv'],
            'mol2vec': reps_test['mol2vec'],
            'graphkernel': reps_test['graphkernel'],
            'mhggnn': reps_test['mhggnn']
        }
        X_test_h3, _ = feature_selector(hybrid_dict_test, None, -1, selection_method, feature_info)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        metrics = test_single(X_train_h3, X_test_h3, train_labels, test_labels, rf, is_classification=False, model_name='RF')
        print(f"    R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
        
        results.append({
            'dataset': 'qm9_homo_lumo_gap',
            'representation': 'everything_noft',
            'n_samples': n_baseline,
            'n_features_input': sum(hybrid_dict[k].shape[1] for k in hybrid_dict),
            'n_features_selected': X_train_h3.shape[1],
            'feature_reduction': (1 - X_train_h3.shape[1] / sum(hybrid_dict[k].shape[1] for k in hybrid_dict)) * 100,
            'n_per_rep': -1,
            'model': 'RF',
            'is_hybrid': True,
            'time': 0,
            **metrics
        })
    
    # ========================================================================
    # ESOL - Regression (YOUR ORIGINAL)
    # ========================================================================
    print(f"\n{'='*80}")
    print("DATASET 1: ESOL (Solubility Regression) - YOUR ORIGINAL")
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
    
    # Create validation split for fine-tuning
    n_val = len(train_smiles) // 5
    val_smiles = train_smiles[:n_val]
    val_labels = train_labels[:n_val]
    train_smiles_fit = train_smiles[n_val:]
    train_labels_fit = train_labels[n_val:]
    
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
    
    # Add graph kernel
    print("  Adding Graph Kernel...")
    graphkernel_train, vocab = create_graph_kernel(
        train_smiles,
        kernel='weisfeiler_lehman',
        n_iter=5,
        return_vocabulary=True
    )
    graphkernel_test = create_graph_kernel(
        test_smiles,
        kernel='weisfeiler_lehman',
        n_iter=5,
        reference_vocabulary=vocab
    )
    reps_train['graphkernel'] = graphkernel_train
    reps_test['graphkernel'] = graphkernel_test
    
    # Add fine-tuned representations if server mode
    if use_server:
        print("\n  SERVER MODE: Adding fine-tuned representations...")
        
        # GNN
        print("    Fine-tuning GNN...")
        gnn_model = finetune_gnn(
            train_smiles_fit, train_labels_fit,
            val_smiles, val_labels,
            gnn_type='gcn', hidden_dim=128, epochs=150  # Match test_esol.py
        )
        reps_train['gnn'] = encode_from_model(gnn_model, train_smiles)
        reps_test['gnn'] = encode_from_model(gnn_model, test_smiles)
        
        # MoLFormer (SMILES)
        print("    Fine-tuning MoLFormer (SMILES)...")
        molformer_model = finetune_molformer(
            train_smiles_fit, train_labels_fit,
            val_smiles, val_labels,
            epochs=10, batch_size=32, use_selfies=False
        )
        reps_train['molformer'] = encode_from_model(molformer_model, train_smiles)
        reps_test['molformer'] = encode_from_model(molformer_model, test_smiles)
        
        # MoLFormer (SELFIES) - ESOL supports SELFIES
        print("    Fine-tuning MoLFormer (SELFIES)...")
        molformer_selfies_model = finetune_molformer(
            train_smiles_fit, train_labels_fit,
            val_smiles, val_labels,
            epochs=10, batch_size=32, use_selfies=True
        )
        reps_train['molformer_selfies'] = encode_from_model(molformer_selfies_model, train_smiles)
        reps_test['molformer_selfies'] = encode_from_model(molformer_selfies_model, test_smiles)
        
        # ChemBERTa
        print("    Fine-tuning ChemBERTa...")
        chemberta_model = finetune_chemberta(
            train_smiles_fit, train_labels_fit,
            val_smiles, val_labels,
            epochs=1, batch_size=32, use_selfies=False  # Note: 1 epoch in test_esol.py
        )
        reps_train['chemberta'] = encode_from_model(chemberta_model, train_smiles)
        reps_test['chemberta'] = encode_from_model(chemberta_model, test_smiles)
        
        # GAUCHE GP - test standalone
        print("    Training GAUCHE GP (Weisfeiler-Lehman kernel)...")
        gp_dict = train_gauche_gp(
            train_smiles_fit, train_labels_fit,
            kernel='weisfeiler_lehman',
            num_epochs=50
        )
        gp_predictions = predict_gauche_gp(gp_dict, test_smiles)
        
        # Test GAUCHE GP immediately
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        gp_rmse = np.sqrt(mean_squared_error(test_labels, gp_predictions['predictions']))
        gp_r2 = r2_score(test_labels, gp_predictions['predictions'])
        print(f"    [GAUCHE GP] RMSE={gp_rmse:.4f}, R²={gp_r2:.4f}")
        
        results.append({
            'dataset': 'esol',
            'representation': 'gauche_gp',
            'n_samples': n_baseline,
            'n_features_input': 0,
            'n_features_selected': 0,
            'feature_reduction': 0.0,
            'model': 'GP',
            'is_hybrid': False,
            'time': 0,
            'mae': mean_absolute_error(test_labels, gp_predictions['predictions']),
            'rmse': gp_rmse,
            'r2': gp_r2
        })
    
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
    rep_names = list(reps_train.keys())
    
    for rep_name in rep_names:
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
        print(f"\n  n_per_rep = {n_per_rep if n_per_rep != -1 else 'ALL'}:")
        
        # Create hybrid once
        hybrid_dict = {name: reps_train[name] for name in best_combo}
        X_train_hybrid, feature_info = feature_selector(hybrid_dict, train_labels, n_per_rep, selection_method)
        
        hybrid_dict_test = {name: reps_test[name] for name in best_combo}
        X_test_hybrid, _ = feature_selector(hybrid_dict_test, None, n_per_rep, selection_method, feature_info=feature_info)
        
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
    # hERG - Classification (EXTENDED: Multiple Sources)
    # ========================================================================
    for herg_source in herg_sources:
        print(f"\n{'='*80}")
        print(f"DATASET 2-{herg_source.upper()}: hERG {herg_source.upper()} (Cardiotoxicity Classification)")
        print(f"{'='*80}")
        
        if herg_source == 'fluid':
            train_data = load_herg_fluid(use_test=False)
            test_data = load_herg_fluid(use_test=True)
        else:  # chembl
            from kirby.datasets.herg import get_herg_splits
            data = load_herg(source='chembl')
            splits = get_herg_splits(data, splitter='scaffold')
            train_data = {
                'smiles': splits['train']['smiles'] + splits['val']['smiles'],
                'labels': np.concatenate([splits['train']['labels'], splits['val']['labels']])
            }
            test_data = {
                'smiles': splits['test']['smiles'],
                'labels': splits['test']['labels']
            }
        
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
        
        # Create val split for fine-tuning
        n_val = len(train_smiles) // 5
        val_smiles = train_smiles[:n_val]
        val_labels = train_labels[:n_val]
        train_smiles_fit = train_smiles[n_val:]
        train_labels_fit = train_labels[n_val:]
        
        print(f"Class balance: {100*train_labels.mean():.1f}% blockers in train, "
              f"{100*test_labels.mean():.1f}% in test")
        
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
        
        # Add graph kernel
        print("  Adding Graph Kernel...")
        graphkernel_train, vocab = create_graph_kernel(
            train_smiles,
            kernel='weisfeiler_lehman',
            n_iter=5,
            return_vocabulary=True
        )
        graphkernel_test = create_graph_kernel(
            test_smiles,
            kernel='weisfeiler_lehman',
            n_iter=5,
            reference_vocabulary=vocab
        )
        reps_train['graphkernel'] = graphkernel_train
        reps_test['graphkernel'] = graphkernel_test
        
        print(f"Representations created ({time.time() - start:.2f}s)")
        
        # Add fine-tuned representations if server mode
        if use_server:
            print("\n  SERVER MODE: Adding fine-tuned representations...")
            
            # GNN
            print("    Fine-tuning GNN...")
            gnn_model = finetune_gnn(
                train_smiles_fit, train_labels_fit,
                val_smiles, val_labels,
                gnn_type='gcn', hidden_dim=128, epochs=50
            )
            reps_train['gnn'] = encode_from_model(gnn_model, train_smiles)
            reps_test['gnn'] = encode_from_model(gnn_model, test_smiles)
            
            # MoLFormer
            print("    Fine-tuning MoLFormer...")
            molformer_model = finetune_molformer(
                train_smiles_fit, train_labels_fit,
                val_smiles, val_labels,
                epochs=10, batch_size=32, use_selfies=False
            )
            reps_train['molformer'] = encode_from_model(molformer_model, train_smiles)
            reps_test['molformer'] = encode_from_model(molformer_model, test_smiles)
            
            # ChemBERTa
            print("    Fine-tuning ChemBERTa...")
            chemberta_model = finetune_chemberta(
                train_smiles_fit, train_labels_fit,
                val_smiles, val_labels,
                epochs=10, batch_size=32, use_selfies=False
            )
            reps_train['chemberta'] = encode_from_model(chemberta_model, train_smiles)
            reps_test['chemberta'] = encode_from_model(chemberta_model, test_smiles)
        
        # STEP 1: Baseline with multiple models
        print(f"\n{'STEP 1: BASELINE (all reps with RF, XGBoost, MLP)':=^80}")
        
        models_to_test = [
            ('RF', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('XGBoost', xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1)),
            ('MLP', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
        ]
        
        baseline_results = []
        rep_names = list(reps_train.keys())
        
        for rep_name in rep_names:
            print(f"\n  {rep_name} ({reps_train[rep_name].shape[1]} features):")
            
            X_train = reps_train[rep_name]
            X_test = reps_test[rep_name]
            
            for model_name, model in models_to_test:
                start = time.time()
                
                metrics = test_single(X_train, X_test, train_labels, test_labels, model, is_classification=True, model_name=model_name)
                elapsed = time.time() - start
                
                result = {
                    'dataset': f'herg_{herg_source}',
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
            print(f"\n  n_per_rep = {n_per_rep if n_per_rep != -1 else 'ALL'}:")
            
            # Create hybrid once
            hybrid_dict = {name: reps_train[name] for name in best_combo}
            X_train_hybrid, feature_info = feature_selector(hybrid_dict, train_labels, n_per_rep, selection_method)
            
            hybrid_dict_test = {name: reps_test[name] for name in best_combo}
            X_test_hybrid, _ = feature_selector(hybrid_dict_test, None, n_per_rep, selection_method, feature_info=feature_info)
            
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
                    'dataset': f'herg_{herg_source}',
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
    # Analysis (YOUR ORIGINAL)
    # ========================================================================
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'='*80}")
    
    df = pd.DataFrame(results)
    
    print(f"\nRan {len(results)} experiments in {total_time:.2f}s ({total_time/60:.2f}min)")
    
    print("\n1. BEST BASELINE (rep+model) PER DATASET:")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_baseline = df_ds[~df_ds['is_hybrid']]
        
        if len(df_baseline) == 0:
            print(f"\n{dataset.upper()}: No baseline results")
            continue
            
        # Select metric based on dataset type
        metric = 'auc' if 'herg' in dataset.lower() else 'r2'
        
        if df_baseline[metric].isna().all():
            print(f"\n{dataset.upper()}: All baseline results are NaN")
            continue
        
        best = df_baseline.loc[df_baseline[metric].idxmax()]
        print(f"\n{dataset.upper()}:")
        print(f"  Best: {best['representation']:15s} + {best['model']:8s}  {metric.upper()}={best[metric]:.4f}")
    
    print("\n2. BEST HYBRID (rep+model+n_per_rep) PER DATASET:")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_hybrid = df_ds[df_ds['is_hybrid']]
        
        if len(df_hybrid) == 0:
            print(f"\n{dataset.upper()}: No hybrid results")
            continue
            
        metric = 'auc' if 'herg' in dataset.lower() else 'r2'
        
        if df_hybrid[metric].isna().all():
            print(f"\n{dataset.upper()}: All hybrid results are NaN")
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
    
    print("\n3. n_per_rep SENSITIVITY (for best hybrid):")
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
    
    print("\n4. MODEL SENSITIVITY (for best hybrid+n_per_rep):")
    for dataset in df['dataset'].unique():
        df_ds = df[df['dataset'] == dataset]
        df_hybrid = df_ds[df_ds['is_hybrid']]
        
        if len(df_hybrid) == 0:
            continue
            
        metric = 'auc' if 'herg' in dataset.lower() else 'r2'
        
        if df_hybrid[metric].isna().all():
            continue
        
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
        'tier': 'comprehensive_with_qm9',
        'start_time': datetime.now().isoformat(),
        'total_time': total_time,
        'n_experiments': len(results),
        'philosophy': 'comprehensive: YOUR ORIGINAL + QM9 HOMO-LUMO gap added',
        'results': results
    }
    
    with open('hybrid_master_comprehensive_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    df.to_csv('hybrid_master_comprehensive_results.csv', index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved: hybrid_master_comprehensive_results.*")
    print(f"{'='*80}")


if __name__ == '__main__':
    args = parse_args()
    
    # Print configuration summary
    print("\n" + "="*100)
    print("KIRBY COMPREHENSIVE EVALUATION - CONFIGURATION")
    print("="*100)
    print(f"Server mode (GPU fine-tuning): {args.server}")
    print(f"hERG sources: {', '.join(args.herg_sources)}")
    print(f"n_per_rep values: {args.n_per_rep}")
    print(f"Feature selection: {args.selection_method}")
    
    if not args.server:
        print("\nℹ️  Running in FAST mode (no fine-tuning)")
        print("   For full evaluation with MoLFormer/ChemBERTa, use --server flag")
    else:
        print("\n⚡ Running in SERVER mode (includes GPU fine-tuning)")
        print("   This will take 45-60 minutes")
    
    print("="*100)
    
    run_structured_sampling(
        create_hybrid_wrapper,
        args
    )