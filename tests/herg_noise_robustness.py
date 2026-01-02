import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
hERG Cardiotoxicity + NoiseInject: Full Model-Representation Matrix
====================================================================

Tests noise robustness on hERG classification dataset with FULL coverage matching QM9 experiments.
NO repetitions (n=1) - single run per configuration.

Purpose: Demonstrate cross-dataset consistency on classification task and test new KIRBy representations

Model-Representation Matrix:
- Representations: ECFP4, PDV, SNS, GraphKernel, MHG-GNN-pretrained (5 total)
- Models per rep: RF, XGBoost, DNN (baseline), DNN (full-BNN), DNN (last-layer-BNN), DNN (var-BNN) (6 total)  
- Total configurations: 5 reps × 6 models = 30

Noise Strategies: uniform, class_imbalance, binary_asymmetric (3 total)
Noise Levels: flip_prob ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} (11 levels)

Expected results:
- Same model-rep rankings as QM9/ESOL
- Classification-specific noise behavior
- New KIRBy reps competitive with baselines

Note: Uses FLuID dataset with provided train/test split
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.calibration import CalibratedClassifierCV
import sys
from pathlib import Path

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.datasets.herg import load_herg
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_sns,
    create_graph_kernel,
    create_mhg_gnn
)

# NoiseInject imports
from noiseInject import (
    NoiseInjectorClassification,
    calculate_classification_metrics
)


# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================

class DeterministicClassifier(nn.Module):
    """Standard feedforward neural network for binary classification"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze()


def train_neural_classifier(X_train, y_train, X_val, y_val, X_test, 
                            epochs=100, lr=1e-3):
    """
    Train a neural network classifier
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (for early stopping)
        X_test: Test data
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        predictions: Test predictions (0/1)
        probabilities: Test prediction probabilities
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Initialize model
    model = DeterministicClassifier(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Calculate class weights for imbalanced data
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    # Update model to output logits
    model.net[-1] = nn.Identity()  # Remove sigmoid, use with BCEWithLogitsLoss
    
    # Training
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Test predictions
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t).cpu().numpy()
        probabilities = torch.sigmoid(torch.FloatTensor(logits)).numpy()
        predictions = (probabilities > 0.5).astype(int)
    
    return predictions, probabilities


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment_tree_model(X_train, y_train, X_test, y_test,
                              model_fn, strategy, flip_prob_levels):
    """
    Run noise robustness experiment for tree-based classifier
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_fn: Function that returns initialized model
        strategy: Noise strategy name
        flip_prob_levels: List of flip probability values to test
    
    Returns:
        predictions: Dict mapping flip_prob -> predictions
        probabilities: Dict mapping flip_prob -> probabilities
    """
    injector = NoiseInjectorClassification(strategy=strategy, random_state=42)
    predictions = {}
    probabilities = {}
    
    for flip_prob in flip_prob_levels:
        # Inject noise
        if flip_prob == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, flip_prob)
        
        # Train model
        model = model_fn()
        model.fit(X_train, y_noisy)
        
        # Get predictions and probabilities
        predictions[flip_prob] = model.predict(X_test)
        probabilities[flip_prob] = model.predict_proba(X_test)[:, 1]
    
    return predictions, probabilities


def run_experiment_neural(X_train, y_train, X_val, y_val, X_test, y_test,
                          strategy, flip_prob_levels):
    """Run noise robustness experiment for neural classifier"""
    injector = NoiseInjectorClassification(strategy=strategy, random_state=42)
    predictions = {}
    probabilities = {}
    
    for flip_prob in flip_prob_levels:
        # Inject noise
        if flip_prob == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, flip_prob)
        
        # Train
        preds, probs = train_neural_classifier(
            X_train, y_noisy, X_val, y_val, X_test,
            epochs=100
        )
        
        predictions[flip_prob] = preds
        probabilities[flip_prob] = probs
    
    return predictions, probabilities


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("="*80)
    print("hERG Cardiotoxicity + NoiseInject: Full Model-Representation Matrix")
    print("5 representations × 6 models = 30 configurations")
    print("="*80)
    
    # Configuration
    strategies = ['uniform', 'class_imbalance', 'binary_asymmetric']
    flip_prob_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_dir = Path(__file__).parent.parent / 'results' / 'herg'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load hERG FLuID dataset
    print("\nLoading hERG FLuID dataset...")
    print("Using provided train/test split from FLuID...")
    
    train_data = load_herg(source='fluid', use_test=False)
    test_data = load_herg(source='fluid', use_test=True)
    
    train_smiles = train_data['smiles']
    train_labels = np.array(train_data['labels'])
    test_smiles = test_data['smiles']
    test_labels = np.array(test_data['labels'])
    
    # Create validation split for neural models
    from sklearn.model_selection import train_test_split
    train_smiles_fit, val_smiles, train_labels_fit, val_labels = train_test_split(
        train_smiles, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"Split sizes: Train={len(train_smiles_fit)}, Val={len(val_smiles)}, Test={len(test_smiles)}")
    print(f"Class balance - Train: {train_labels_fit.mean():.1%} blockers")
    print(f"Class balance - Val: {val_labels.mean():.1%} blockers")
    print(f"Class balance - Test: {test_labels.mean():.1%} blockers")
    
    # Storage for all results
    all_results = []
    
    # =========================================================================
    # GENERATE ALL REPRESENTATIONS ONCE
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING REPRESENTATIONS")
    print("="*80)
    
    print("\n[1/5] ECFP4...")
    ecfp4_train = create_ecfp4(train_smiles_fit, n_bits=2048)
    ecfp4_val = create_ecfp4(val_smiles, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    
    print("[2/5] PDV...")
    pdv_train = create_pdv(train_smiles_fit)
    pdv_val = create_pdv(val_smiles)
    pdv_test = create_pdv(test_smiles)
    
    print("[3/5] SNS...")
    sns_train, vocab = create_sns(train_smiles_fit, n_features=2048, return_vocabulary=True)
    sns_test = create_sns(test_smiles, n_features=2048, reference_vocabulary=vocab)
    sns_val = create_sns(val_smiles, n_features=2048, reference_vocabulary=vocab)
    
    print("[4/5] GraphKernel...")
    graphkernel_train, gk_vocab = create_graph_kernel(
        train_smiles_fit, kernel='weisfeiler_lehman', n_iter=5, return_vocabulary=True
    )
    graphkernel_test = create_graph_kernel(
        test_smiles, kernel='weisfeiler_lehman', n_iter=5, reference_vocabulary=gk_vocab
    )
    graphkernel_val = create_graph_kernel(
        val_smiles, kernel='weisfeiler_lehman', n_iter=5, reference_vocabulary=gk_vocab
    )
    
    print("[5/5] MHG-GNN (pretrained)...")
    mhggnn_train = create_mhg_gnn(train_smiles_fit, batch_size=32)
    mhggnn_test = create_mhg_gnn(test_smiles, batch_size=32)
    mhggnn_val = create_mhg_gnn(val_smiles, batch_size=32)
    
    # =========================================================================
    # TEST ALL MODEL-REPRESENTATION COMBINATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("TESTING ALL CONFIGURATIONS (30 total)")
    print("="*80)
    
    # Define all representations
    representations = [
        ('ECFP4', ecfp4_train, ecfp4_val, ecfp4_test),
        ('PDV', pdv_train, pdv_val, pdv_test),
        ('SNS', sns_train, sns_val, sns_test),
        ('GraphKernel', graphkernel_train, graphkernel_val, graphkernel_test),
        ('MHG-GNN-pretrained', mhggnn_train, mhggnn_val, mhggnn_test)
    ]
    
    config_num = 0
    for rep_name, X_train, X_val, X_test in representations:
        print(f"\n{'='*80}")
        print(f"REPRESENTATION: {rep_name}")
        print(f"{'='*80}")
        
        # =====================================================================
        # Model 1: Random Forest
        # =====================================================================
        config_num += 1
        print(f"\n[{config_num}/30] RF + {rep_name}")
        for strategy in strategies:
            print(f"  Strategy: {strategy}")
            predictions, probabilities = run_experiment_tree_model(
                X_train, train_labels_fit, X_test, test_labels,
                lambda: RandomForestClassifier(n_estimators=100, random_state=42, 
                                              n_jobs=-1, class_weight='balanced'),
                strategy, flip_prob_levels
            )
            per_flip, summary, per_class = calculate_classification_metrics(
                test_labels, predictions, probabilities
            )
            per_flip['model'] = 'RF'
            per_flip['rep'] = rep_name
            per_flip['strategy'] = strategy
            all_results.append(per_flip)
            per_flip.to_csv(results_dir / f'RF_{rep_name}_{strategy}.csv', index=False)
        
        # =====================================================================
        # Model 2: XGBoost Classifier
        # =====================================================================
        config_num += 1
        print(f"\n[{config_num}/30] XGBoost + {rep_name}")
        from xgboost import XGBClassifier
        for strategy in strategies:
            print(f"  Strategy: {strategy}")
            predictions, probabilities = run_experiment_tree_model(
                X_train, train_labels_fit, X_test, test_labels,
                lambda: XGBClassifier(n_estimators=100, random_state=42, 
                                     n_jobs=-1, eval_metric='logloss'),
                strategy, flip_prob_levels
            )
            per_flip, summary, per_class = calculate_classification_metrics(
                test_labels, predictions, probabilities
            )
            per_flip['model'] = 'XGBoost'
            per_flip['rep'] = rep_name
            per_flip['strategy'] = strategy
            all_results.append(per_flip)
            per_flip.to_csv(results_dir / f'XGBoost_{rep_name}_{strategy}.csv', index=False)
        
        # =====================================================================
        # Model 3: DNN (baseline)
        # =====================================================================
        config_num += 1
        print(f"\n[{config_num}/30] DNN + {rep_name}")
        for strategy in strategies:
            print(f"  Strategy: {strategy}")
            predictions, probabilities = run_experiment_neural(
                X_train, train_labels_fit, X_val, val_labels, X_test, test_labels,
                strategy, flip_prob_levels
            )
            per_flip, summary, per_class = calculate_classification_metrics(
                test_labels, predictions, probabilities
            )
            per_flip['model'] = 'DNN'
            per_flip['rep'] = rep_name
            per_flip['strategy'] = strategy
            all_results.append(per_flip)
            per_flip.to_csv(results_dir / f'DNN_{rep_name}_{strategy}.csv', index=False)
        
        # =====================================================================
        # Model 4-6: DNN Bayesian variants (full, last-layer, variational)
        # =====================================================================
        # TODO: Implement Bayesian classifier variants
        # For now, note these as placeholders
        print(f"\n  Note: DNN-BNN variants not yet implemented for classification")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_dir / 'all_results.csv', index=False)
    
    # Calculate NSI and retention using flip_prob_max = 0.6
    summary_table = []
    for (model, rep, strategy), group in combined_df.groupby(['model', 'rep', 'strategy']):
        baseline_rows = group[group['flip_prob'] == 0.0]
        high_noise_rows = group[group['flip_prob'] == 0.6]
        
        if len(baseline_rows) > 0 and len(high_noise_rows) > 0:
            baseline_acc = baseline_rows['accuracy'].values[0]
            high_noise_acc = high_noise_rows['accuracy'].values[0]
            baseline_auc = baseline_rows['auc'].values[0]
            high_noise_auc = high_noise_rows['auc'].values[0]
            
            nsi_acc = (baseline_acc - high_noise_acc) / 0.6
            retention_acc = (high_noise_acc / baseline_acc) * 100 if baseline_acc > 0 else 0
            nsi_auc = (baseline_auc - high_noise_auc) / 0.6
            retention_auc = (high_noise_auc / baseline_auc) * 100 if baseline_auc > 0 else 0
            
            summary_table.append({
                'model': model,
                'rep': rep,
                'strategy': strategy,
                'baseline_accuracy': baseline_acc,
                'accuracy_at_0.6': high_noise_acc,
                'NSI_accuracy': nsi_acc,
                'retention_%_accuracy': retention_acc,
                'baseline_auc': baseline_auc,
                'auc_at_0.6': high_noise_auc,
                'NSI_auc': nsi_auc,
                'retention_%_auc': retention_auc
            })
    
    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(results_dir / 'summary.csv', index=False)
    
    print("\nTop 10 most robust by accuracy (lowest NSI):")
    top10_acc = summary_df.nsmallest(10, 'NSI_accuracy')[
        ['model', 'rep', 'strategy', 'NSI_accuracy', 'retention_%_accuracy']
    ]
    print(top10_acc.to_string(index=False))
    
    print("\nResults by representation (mean across models/strategies):")
    rep_summary = summary_df.groupby('rep')[
        ['NSI_accuracy', 'retention_%_accuracy', 'NSI_auc', 'retention_%_auc']
    ].mean()
    print(rep_summary.to_string())
    
    print("\nResults by model (mean across reps/strategies):")
    model_summary = summary_df.groupby('model')[
        ['NSI_accuracy', 'retention_%_accuracy', 'NSI_auc', 'retention_%_auc']
    ].mean()
    print(model_summary.to_string())
    
    print("\n" + "="*80)
    print("COMPLETE - Results saved to results/herg/")
    print("="*80)
    
    # Configuration
    strategies = ['uniform', 'class_imbalance', 'binary_asymmetric']
    flip_prob_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_dir = Path('results/herg')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load hERG FLuID dataset
    print("\nLoading hERG FLuID dataset...")
    print("Using provided train/test split from FLuID...")
    
    train_data = load_herg(source='fluid', use_test=False)
    test_data = load_herg(source='fluid', use_test=True)
    
    train_smiles = train_data['smiles']
    train_labels = np.array(train_data['labels'])
    test_smiles = test_data['smiles']
    test_labels = np.array(test_data['labels'])
    
    # Create validation split for neural models
    from sklearn.model_selection import train_test_split
    train_smiles_fit, val_smiles, train_labels_fit, val_labels = train_test_split(
        train_smiles, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"Split sizes: Train={len(train_smiles_fit)}, Val={len(val_smiles)}, Test={len(test_smiles)}")
    print(f"Class balance - Train: {train_labels_fit.mean():.1%} blockers")
    print(f"Class balance - Val: {val_labels.mean():.1%} blockers")
    print(f"Class balance - Test: {test_labels.mean():.1%} blockers")
    
    # Storage for all results
    all_results = []
    all_calibration_data = []
    
    # =========================================================================
    # PHASE 1: CORE ROBUSTNESS
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: CORE ROBUSTNESS - Strategic Pairs")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # Pair 1: RF + ECFP4 (baseline)
    # -------------------------------------------------------------------------
    print("\n[1/6] RF + ECFP4...")
    ecfp4_train = create_ecfp4(train_smiles_fit, n_bits=2048)
    ecfp4_val = create_ecfp4(val_smiles, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, probabilities = run_experiment_tree_model(
            ecfp4_train, train_labels_fit, ecfp4_test, test_labels,
            lambda: RandomForestClassifier(n_estimators=100, random_state=42, 
                                          n_jobs=-1, class_weight='balanced'),
            strategy, flip_prob_levels
        )
        
        per_flip, summary, per_class = calculate_classification_metrics(
            test_labels, predictions, probabilities
        )
        per_flip['model'] = 'RF'
        per_flip['rep'] = 'ECFP4'
        per_flip['strategy'] = strategy
        all_results.append(per_flip)
        per_flip.to_csv(results_dir / f'RF_ECFP4_{strategy}.csv', index=False)
        
        # Save calibration data for Phase 3
        if strategy == 'uniform':
            calib_data = []
            for flip_prob in flip_prob_levels:
                for i in range(len(test_labels)):
                    calib_data.append({
                        'flip_prob': flip_prob,
                        'sample_idx': i,
                        'y_true': test_labels[i],
                        'y_pred': predictions[flip_prob][i],
                        'probability': probabilities[flip_prob][i]
                    })
            all_calibration_data.append(('RF', 'ECFP4', pd.DataFrame(calib_data)))
    
    # -------------------------------------------------------------------------
    # Pair 2: RF + PDV (baseline)
    # -------------------------------------------------------------------------
    print("\n[2/6] RF + PDV...")
    pdv_train = create_pdv(train_smiles_fit)
    pdv_val = create_pdv(val_smiles)
    pdv_test = create_pdv(test_smiles)
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, probabilities = run_experiment_tree_model(
            pdv_train, train_labels_fit, pdv_test, test_labels,
            lambda: RandomForestClassifier(n_estimators=100, random_state=42,
                                          n_jobs=-1, class_weight='balanced'),
            strategy, flip_prob_levels
        )
        
        per_flip, summary, per_class = calculate_classification_metrics(
            test_labels, predictions, probabilities
        )
        per_flip['model'] = 'RF'
        per_flip['rep'] = 'PDV'
        per_flip['strategy'] = strategy
        all_results.append(per_flip)
        per_flip.to_csv(results_dir / f'RF_PDV_{strategy}.csv', index=False)
    
    # -------------------------------------------------------------------------
    # Pair 3: RF + Graph Kernel (new with KIRBy)
    # -------------------------------------------------------------------------
    print("\n[3/6] RF + Graph Kernel...")
    graphkernel_train, vocab = create_graph_kernel(
        train_smiles_fit,
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
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, probabilities = run_experiment_tree_model(
            graphkernel_train, train_labels_fit, graphkernel_test, test_labels,
            lambda: RandomForestClassifier(n_estimators=100, random_state=42,
                                          n_jobs=-1, class_weight='balanced'),
            strategy, flip_prob_levels
        )
        
        per_flip, summary, per_class = calculate_classification_metrics(
            test_labels, predictions, probabilities
        )
        per_flip['model'] = 'RF'
        per_flip['rep'] = 'GraphKernel'
        per_flip['strategy'] = strategy
        all_results.append(per_flip)
        per_flip.to_csv(results_dir / f'RF_GraphKernel_{strategy}.csv', index=False)
    
    # -------------------------------------------------------------------------
    # Pair 4: RF + MHG-GNN pretrained (new with KIRBy)
    # -------------------------------------------------------------------------
    print("\n[4/6] RF + MHG-GNN (pretrained)...")
    mhggnn_train = create_mhg_gnn(train_smiles_fit, batch_size=32)
    mhggnn_test = create_mhg_gnn(test_smiles, batch_size=32)
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, probabilities = run_experiment_tree_model(
            mhggnn_train, train_labels_fit, mhggnn_test, test_labels,
            lambda: RandomForestClassifier(n_estimators=100, random_state=42,
                                          n_jobs=-1, class_weight='balanced'),
            strategy, flip_prob_levels
        )
        
        per_flip, summary, per_class = calculate_classification_metrics(
            test_labels, predictions, probabilities
        )
        per_flip['model'] = 'RF'
        per_flip['rep'] = 'MHG-GNN-pretrained'
        per_flip['strategy'] = strategy
        all_results.append(per_flip)
        per_flip.to_csv(results_dir / f'RF_MHGGNN-pretrained_{strategy}.csv', index=False)
    
    # -------------------------------------------------------------------------
    # Pair 5: DNN + ECFP4 (neural baseline)
    # -------------------------------------------------------------------------
    print("\n[5/6] DNN + ECFP4...")
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, probabilities = run_experiment_neural(
            ecfp4_train, train_labels_fit, ecfp4_val, val_labels, ecfp4_test, test_labels,
            strategy, flip_prob_levels
        )
        
        per_flip, summary, per_class = calculate_classification_metrics(
            test_labels, predictions, probabilities
        )
        per_flip['model'] = 'DNN'
        per_flip['rep'] = 'ECFP4'
        per_flip['strategy'] = strategy
        all_results.append(per_flip)
        per_flip.to_csv(results_dir / f'DNN_ECFP4_{strategy}.csv', index=False)
    
    # -------------------------------------------------------------------------
    # Pair 6: DNN + PDV (neural baseline)
    # -------------------------------------------------------------------------
    print("\n[6/6] DNN + PDV...")
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, probabilities = run_experiment_neural(
            pdv_train, train_labels_fit, pdv_val, val_labels, pdv_test, test_labels,
            strategy, flip_prob_levels
        )
        
        per_flip, summary, per_class = calculate_classification_metrics(
            test_labels, predictions, probabilities
        )
        per_flip['model'] = 'DNN'
        per_flip['rep'] = 'PDV'
        per_flip['strategy'] = strategy
        all_results.append(per_flip)
        per_flip.to_csv(results_dir / f'DNN_PDV_{strategy}.csv', index=False)
        
        # Save calibration data for Phase 3
        if strategy == 'uniform':
            calib_data = []
            for flip_prob in flip_prob_levels:
                for i in range(len(test_labels)):
                    calib_data.append({
                        'flip_prob': flip_prob,
                        'sample_idx': i,
                        'y_true': test_labels[i],
                        'y_pred': predictions[flip_prob][i],
                        'probability': probabilities[flip_prob][i]
                    })
            all_calibration_data.append(('DNN', 'PDV', pd.DataFrame(calib_data)))
    
    # =========================================================================
    # PHASE 2: PROBABILISTIC COMPARISON
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: PROBABILISTIC COMPARISON (uniform strategy only)")
    print("="*80)
    print("Testing RF with and without probability calibration on ECFP4...")
    
    strategy = 'uniform'
    injector = NoiseInjectorClassification(strategy=strategy, random_state=42)
    
    uncalibrated_predictions = {}
    uncalibrated_probabilities = {}
    calibrated_predictions = {}
    calibrated_probabilities = {}
    
    for flip_prob in flip_prob_levels:
        print(f"  flip_prob={flip_prob:.1f}...", end='')
        
        # Inject noise
        if flip_prob == 0.0:
            y_noisy = train_labels_fit
        else:
            y_noisy = injector.inject(train_labels_fit, flip_prob)
        
        # Train uncalibrated RF
        rf_uncal = RandomForestClassifier(n_estimators=100, random_state=42,
                                         n_jobs=-1, class_weight='balanced')
        rf_uncal.fit(ecfp4_train, y_noisy)
        uncalibrated_predictions[flip_prob] = rf_uncal.predict(ecfp4_test)
        uncalibrated_probabilities[flip_prob] = rf_uncal.predict_proba(ecfp4_test)[:, 1]
        
        # Train calibrated RF (isotonic calibration)
        rf_cal = RandomForestClassifier(n_estimators=100, random_state=42,
                                       n_jobs=-1, class_weight='balanced')
        rf_calibrated = CalibratedClassifierCV(rf_cal, method='isotonic', cv=3)
        rf_calibrated.fit(ecfp4_train, y_noisy)
        calibrated_predictions[flip_prob] = rf_calibrated.predict(ecfp4_test)
        calibrated_probabilities[flip_prob] = rf_calibrated.predict_proba(ecfp4_test)[:, 1]
        
        print(" done")
    
    # Calculate metrics for both
    per_flip_uncal, summary_uncal, _ = calculate_classification_metrics(
        test_labels, uncalibrated_predictions, uncalibrated_probabilities
    )
    per_flip_uncal['model'] = 'RF-uncalibrated'
    per_flip_uncal['rep'] = 'ECFP4'
    per_flip_uncal['strategy'] = strategy
    per_flip_uncal.to_csv(results_dir / 'RF_ECFP4_uncalibrated_uniform.csv', index=False)
    
    per_flip_cal, summary_cal, _ = calculate_classification_metrics(
        test_labels, calibrated_predictions, calibrated_probabilities
    )
    per_flip_cal['model'] = 'RF-calibrated'
    per_flip_cal['rep'] = 'ECFP4'
    per_flip_cal['strategy'] = strategy
    per_flip_cal.to_csv(results_dir / 'RF_ECFP4_calibrated_uniform.csv', index=False)
    
    print("\nComparison (uniform strategy):")
    print(f"  Uncalibrated - NSI(accuracy): {summary_uncal['nsi_accuracy'].values[0]:.4f}")
    print(f"  Calibrated   - NSI(accuracy): {summary_cal['nsi_accuracy'].values[0]:.4f}")
    
    # =========================================================================
    # PHASE 3: UNCERTAINTY QUANTIFICATION
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: PROBABILITY CALIBRATION ANALYSIS (uniform strategy only)")
    print("="*80)
    
    # Save all calibration data
    for model_name, rep_name, calib_df in all_calibration_data:
        calib_df.to_csv(
            results_dir / f'{model_name}_{rep_name}_calibration_values.csv',
            index=False
        )
        print(f"Saved {model_name} + {rep_name} calibration data")
        
        # Calculate Brier score per flip_prob
        print(f"\n{model_name} + {rep_name} - Brier Score:")
        for flip_prob in flip_prob_levels:
            subset = calib_df[calib_df['flip_prob'] == flip_prob]
            if len(subset) > 0:
                brier = np.mean((subset['probability'] - subset['y_true'])**2)
                print(f"  flip_prob={flip_prob:.1f}: Brier = {brier:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_dir / 'all_results.csv', index=False)
    
    # Calculate NSI and retention using flip_prob_max = 0.6
    summary_table = []
    for (model, rep, strategy), group in combined_df.groupby(['model', 'rep', 'strategy']):
        baseline_rows = group[group['flip_prob'] == 0.0]
        high_noise_rows = group[group['flip_prob'] == 0.6]
        
        if len(baseline_rows) > 0 and len(high_noise_rows) > 0:
            baseline_acc = baseline_rows['accuracy'].values[0]
            high_noise_acc = high_noise_rows['accuracy'].values[0]
            baseline_auc = baseline_rows['auc'].values[0]
            high_noise_auc = high_noise_rows['auc'].values[0]
            
            nsi_acc = (baseline_acc - high_noise_acc) / 0.6
            retention_acc = (high_noise_acc / baseline_acc) * 100 if baseline_acc > 0 else 0
            nsi_auc = (baseline_auc - high_noise_auc) / 0.6
            retention_auc = (high_noise_auc / baseline_auc) * 100 if baseline_auc > 0 else 0
            
            summary_table.append({
                'model': model,
                'rep': rep,
                'strategy': strategy,
                'baseline_accuracy': baseline_acc,
                'accuracy_at_0.6': high_noise_acc,
                'NSI_accuracy': nsi_acc,
                'retention_%_accuracy': retention_acc,
                'baseline_auc': baseline_auc,
                'auc_at_0.6': high_noise_auc,
                'NSI_auc': nsi_auc,
                'retention_%_auc': retention_auc
            })
    
    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(results_dir / 'summary.csv', index=False)
    
    print("\nTop 10 most robust by accuracy (lowest NSI):")
    top10_acc = summary_df.nsmallest(10, 'NSI_accuracy')[
        ['model', 'rep', 'strategy', 'NSI_accuracy', 'retention_%_accuracy']
    ]
    print(top10_acc.to_string(index=False))
    
    print("\nTop 10 most robust by AUC (lowest NSI):")
    top10_auc = summary_df.nsmallest(10, 'NSI_auc')[
        ['model', 'rep', 'strategy', 'NSI_auc', 'retention_%_auc']
    ]
    print(top10_auc.to_string(index=False))
    
    print("\nResults by strategy (mean across models/reps):")
    strategy_summary = summary_df.groupby('strategy')[
        ['NSI_accuracy', 'retention_%_accuracy', 'NSI_auc', 'retention_%_auc']
    ].mean()
    print(strategy_summary.to_string())
    
    print("\nResults by representation (mean across models/strategies):")
    rep_summary = summary_df.groupby('rep')[
        ['NSI_accuracy', 'retention_%_accuracy', 'NSI_auc', 'retention_%_auc']
    ].mean()
    print(rep_summary.to_string())
    
    print("\n" + "="*80)
    print("COMPLETE - Results saved to results/herg/")
    print("="*80)


if __name__ == '__main__':
    main()
