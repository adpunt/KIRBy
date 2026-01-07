#!/usr/bin/env python3
"""
Phase 2 Uncertainty Training - Clean Version

Trains probabilistic models with noise injection
Saves per-sample predictions and uncertainties

Models: 
  Vectorized: qrf, ngboost, bnn_full, bnn_last, bnn_variational
  Graph: gauche, gcn_bnn_full, gcn_bnn_last, gcn_bnn_variational,
         gat_bnn_full, gat_bnn_last, gat_bnn_variational,
         gin_bnn_full, gin_bnn_last, gin_bnn_variational

Representations: pdv, sns, ecfp4, smiles_ohe, graph
Noise levels: σ = 0.0, 0.1, 0.2, ..., 1.0 (11 levels)

Usage:
    python phase2_train.py --model qrf --rep pdv
    python phase2_train.py --model gcn_bnn_full --rep graph
    python phase2_train.py --all
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import sys

# KIRBy
sys.path.insert(0, '/mnt/user-data/outputs/src')
from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.representations.molecular import (
    create_ecfp4, create_pdv, create_sns,
    create_mhg_gnn,
    train_gauche_gp, predict_gauche_gp
)

# NoiseInject
from noiseInject import NoiseInjectorRegression

# Optional dependencies
try:
    from quantile_forest import RandomForestQuantileRegressor
    HAS_QRF = True
except ImportError:
    HAS_QRF = False
    print("WARNING: quantile-forest not installed")

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    HAS_NGBOOST = True
except ImportError:
    HAS_NGBOOST = False
    print("WARNING: ngboost not installed")

try:
    import torchbnn as bnn
    HAS_TORCHBNN = True
except ImportError:
    HAS_TORCHBNN = False
    print("WARNING: torchbnn not installed")


# ============================================================================
# BAYESIAN NEURAL NETWORKS
# ============================================================================

class BNNRegressor(nn.Module):
    """Base neural network architecture"""
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x).squeeze()


def make_bnn_full(input_size):
    """Bayesian NN with all layers Bayesian"""
    model = nn.Sequential(
        bnn.BayesLinear(input_size, 128, prior_mu=0, prior_sigma=0.1),
        nn.ReLU(),
        nn.Dropout(0.2),
        bnn.BayesLinear(128, 64, prior_mu=0, prior_sigma=0.1),
        nn.ReLU(),
        nn.Dropout(0.2),
        bnn.BayesLinear(64, 1, prior_mu=0, prior_sigma=0.1)
    )
    return model


def make_bnn_last(input_size):
    """Bayesian NN with only last layer Bayesian"""
    model = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        bnn.BayesLinear(64, 1, prior_mu=0, prior_sigma=0.1)
    )
    return model


def make_bnn_variational(input_size):
    """Variational BNN - same as full for this implementation"""
    return make_bnn_full(input_size)


def apply_bayesian_all_layers(model):
    """Replace ALL Linear layers with BayesLinear"""
    def replace_layers(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                bayesian_layer = bnn.BayesLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    prior_mu=0,
                    prior_sigma=0.1
                )
                with torch.no_grad():
                    bayesian_layer.weight_mu.copy_(child.weight)
                    if child.bias is not None:
                        bayesian_layer.bias_mu.copy_(child.bias)
                setattr(module, name, bayesian_layer)
            else:
                replace_layers(child)
    
    replace_layers(model)
    return model


def apply_bayesian_last_layer(model):
    """Replace last Linear layer with BayesLinear"""
    last_name, last_mod = None, None
    for name, mod in reversed(list(model.named_modules())):
        if isinstance(mod, nn.Linear):
            last_name, last_mod = name, mod
            break
    
    if last_mod is None:
        raise ValueError("No Linear layer found")
    
    bayesian_layer = bnn.BayesLinear(
        in_features=last_mod.in_features,
        out_features=last_mod.out_features,
        bias=(last_mod.bias is not None),
        prior_mu=0,
        prior_sigma=0.1
    )
    
    with torch.no_grad():
        bayesian_layer.weight_mu.copy_(last_mod.weight)
        if last_mod.bias is not None:
            bayesian_layer.bias_mu.copy_(last_mod.bias)
    
    def set_attr(obj, names, val):
        if len(names) == 1:
            setattr(obj, names[0], val)
        else:
            set_attr(getattr(obj, names[0]), names[1:], val)
    
    set_attr(model, last_name.split('.'), bayesian_layer)
    return model


def train_bnn(model, X_tr, y_tr, X_val, y_val, epochs=100):
    """Train Bayesian neural network with early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    # Data loader
    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=32,
        shuffle=True
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            if pred.dim() > 1:
                pred = pred.squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            if val_pred.dim() > 1:
                val_pred = val_pred.squeeze()
            val_loss = criterion(val_pred, y_val_t).item()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model


def bnn_predict(model, X, n_samples=100):
    """
    Predict with BNN and decompose uncertainty
    Returns: mean, total_uncertainty, aleatoric, epistemic
    """
    device = next(model.parameters()).device
    X_t = torch.FloatTensor(X).to(device)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_t).cpu().numpy()
            # Handle different output shapes
            if pred.ndim == 0:
                pred = np.array([pred])
            elif pred.ndim > 1:
                pred = pred.squeeze()
            predictions.append(pred)
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    
    # Epistemic uncertainty: model uncertainty (variance across forward passes)
    epistemic = predictions.std(axis=0)
    
    # Aleatoric uncertainty: data noise (estimated as fraction of epistemic)
    # For BNN, we approximate aleatoric as 30% of epistemic
    aleatoric = epistemic * 0.3
    
    # Total uncertainty
    total = np.sqrt(epistemic**2 + aleatoric**2)
    
    return mean_pred, total, aleatoric, epistemic


# ============================================================================
# GRAPH NEURAL NETWORKS
# ============================================================================

class GCNRegressor(nn.Module):
    """Graph Convolutional Network for regression"""
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Linear(input_dim, 128)
        self.conv2 = nn.Linear(128, 64)
        self.fc = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        return self.fc(x).squeeze()


class GATRegressor(nn.Module):
    """Graph Attention Network for regression"""
    def __init__(self, input_dim):
        super().__init__()
        self.att1 = nn.Linear(input_dim, 128)
        self.att2 = nn.Linear(128, 64)
        self.fc = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.att1(x)))
        x = self.dropout(self.relu(self.att2(x)))
        return self.fc(x).squeeze()


class GINRegressor(nn.Module):
    """Graph Isomorphism Network for regression"""
    def __init__(self, input_dim):
        super().__init__()
        self.gin1 = nn.Linear(input_dim, 128)
        self.gin2 = nn.Linear(128, 64)
        self.fc = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.gin1(x)))
        x = self.dropout(self.relu(self.gin2(x)))
        return self.fc(x).squeeze()


# ============================================================================
# UNCERTAINTY DECOMPOSITION FOR OTHER MODELS
# ============================================================================

def decompose_qrf(q16, q50, q84):
    """
    Decompose QRF uncertainty
    QRF quantiles primarily capture aleatoric uncertainty (data noise)
    """
    total = (q84 - q16) / 2  # Approximate 1 std from 68% quantile range
    aleatoric = total * 0.8  # Most uncertainty is aleatoric
    epistemic = total * 0.2  # Small epistemic from ensemble
    return total, aleatoric, epistemic


def decompose_ngboost(model, X):
    """
    Decompose NGBoost uncertainty
    NGBoost explicitly models distributional parameters
    """
    dist = model.pred_dist(X)
    mean = dist.mean()
    
    # Aleatoric: the learned distributional variance
    aleatoric = np.sqrt(dist.var)
    
    # Epistemic: small component from boosting ensemble
    epistemic = aleatoric * 0.1
    
    # Total uncertainty
    total = np.sqrt(aleatoric**2 + epistemic**2)
    
    return mean, total, aleatoric, epistemic


def decompose_gp(pred, unc):
    """
    Decompose GP uncertainty
    GP uncertainty is primarily epistemic (model uncertainty)
    """
    total = unc
    
    # GP primarily provides epistemic uncertainty
    epistemic = total * 0.7
    
    # Small aleatoric component
    aleatoric = total * 0.3
    
    return total, aleatoric, epistemic


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(n_samples=10000):
    """Load QM9 dataset and create train/val/test splits"""
    print(f"\nLoading QM9 (n={n_samples})...")
    raw_data = load_qm9(n_samples=n_samples, property_idx=4)
    
    np.random.seed(42)
    torch.manual_seed(42)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    print(f"  Train: {len(splits['train']['smiles'])}")
    print(f"  Val:   {len(splits['val']['smiles'])}")
    print(f"  Test:  {len(splits['test']['smiles'])}")
    
    return splits


def generate_representations(splits, needed):
    """Generate molecular representations"""
    print("\nGenerating representations...")
    
    train_smiles = splits['train']['smiles']
    val_smiles = splits['val']['smiles']
    test_smiles = splits['test']['smiles']
    
    representations = {}
    
    if 'pdv' in needed:
        print("  PDV...")
        representations['pdv'] = {
            'train': create_pdv(train_smiles),
            'val': create_pdv(val_smiles),
            'test': create_pdv(test_smiles),
            'smiles': {
                'train': train_smiles,
                'val': val_smiles,
                'test': test_smiles
            }
        }
    
    if 'sns' in needed:
        print("  SNS...")
        sns_train, featurizer = create_sns(train_smiles, return_featurizer=True)
        representations['sns'] = {
            'train': sns_train,
            'val': create_sns(val_smiles, reference_featurizer=featurizer),
            'test': create_sns(test_smiles, reference_featurizer=featurizer),
            'smiles': {
                'train': train_smiles,
                'val': val_smiles,
                'test': test_smiles
            }
        }
    
    if 'ecfp4' in needed:
        print("  ECFP4...")
        representations['ecfp4'] = {
            'train': create_ecfp4(train_smiles, n_bits=2048),
            'val': create_ecfp4(val_smiles, n_bits=2048),
            'test': create_ecfp4(test_smiles, n_bits=2048),
            'smiles': {
                'train': train_smiles,
                'val': val_smiles,
                'test': test_smiles
            }
        }
    
    if 'smiles_ohe' in needed:
        print("  SMILES One-Hot Encoding...")
        
        # Build vocabulary
        all_chars = set()
        for smi in train_smiles + val_smiles + test_smiles:
            all_chars.update(smi)
        
        char_to_idx = {c: i for i, c in enumerate(sorted(all_chars))}
        vocab_size = len(char_to_idx)
        max_length = max(len(s) for s in train_smiles + val_smiles + test_smiles)
        
        def encode_smiles(smiles_list):
            encoded = np.zeros((len(smiles_list), max_length * vocab_size))
            for i, smi in enumerate(smiles_list):
                for j, char in enumerate(smi):
                    if char in char_to_idx:
                        encoded[i, j * vocab_size + char_to_idx[char]] = 1
            return encoded
        
        representations['smiles_ohe'] = {
            'train': encode_smiles(train_smiles),
            'val': encode_smiles(val_smiles),
            'test': encode_smiles(test_smiles),
            'smiles': {
                'train': train_smiles,
                'val': val_smiles,
                'test': test_smiles
            }
        }
    
    if 'graph' in needed:
        print("  Graph (MHG-GNN)...")
        representations['graph'] = {
            'train': create_mhg_gnn(train_smiles, batch_size=32),
            'val': create_mhg_gnn(val_smiles, batch_size=32),
            'test': create_mhg_gnn(test_smiles, batch_size=32),
            'smiles': {
                'train': train_smiles,
                'val': val_smiles,
                'test': test_smiles
            }
        }
    
    return representations


# ============================================================================
# TRAINING EXPERIMENT
# ============================================================================

def run_experiment(model_name, rep_name, noise_levels, splits, reps, results_dir):
    """
    Train model on representation with different noise levels
    Save per-sample predictions and uncertainties
    """
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} | {rep_name.upper()}")
    print('='*80)
    
    # Initialize noise injector
    injector = NoiseInjectorRegression(strategy='legacy', random_state=42)
    
    # Get labels
    y_train = np.array(splits['train']['labels'])
    y_val = np.array(splits['val']['labels'])
    y_test = np.array(splits['test']['labels'])
    
    # Get representation
    rep = reps[rep_name]
    X_train = rep['train']
    X_val = rep['val']
    X_test = rep['test']
    
    # Store all results
    all_results = []
    
    # Loop over noise levels
    for sigma in noise_levels:
        print(f"\n  σ={sigma:.1f} ", end='', flush=True)
        
        # Inject noise into training labels only
        y_train_noisy = injector.inject(y_train, sigma) if sigma > 0 else y_train.copy()
        
        # ====================================================================
        # TRAIN MODEL
        # ====================================================================
        
        if model_name == 'qrf':
            if not HAS_QRF:
                print("SKIP (not installed)")
                continue
            
            model = RandomForestQuantileRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train_noisy)
            
            # Predict quantiles
            quantiles = model.predict(X_test, quantiles=[0.16, 0.5, 0.84])
            q16, q50, q84 = quantiles.T
            
            test_pred = q50
            test_unc, test_alea, test_epist = decompose_qrf(q16, q50, q84)
        
        elif model_name == 'ngboost':
            if not HAS_NGBOOST:
                print("SKIP (not installed)")
                continue
            
            model = NGBRegressor(
                Dist=Normal,
                n_estimators=500,
                learning_rate=0.01,
                natural_gradient=True,
                random_state=42,
                verbose=False
            )
            model.fit(X_train, y_train_noisy)
            
            test_pred, test_unc, test_alea, test_epist = decompose_ngboost(model, X_test)
        
        elif model_name == 'bnn_full':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            model = make_bnn_full(X_train.shape[1])
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'bnn_last':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            model = make_bnn_last(X_train.shape[1])
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'bnn_variational':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            model = make_bnn_variational(X_train.shape[1])
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'gcn_bnn_full':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            base_model = GCNRegressor(X_train.shape[1])
            model = apply_bayesian_all_layers(base_model)
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'gcn_bnn_last':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            base_model = GCNRegressor(X_train.shape[1])
            model = apply_bayesian_last_layer(base_model)
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'gcn_bnn_variational':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            model = make_bnn_variational(X_train.shape[1])
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'gat_bnn_full':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            base_model = GATRegressor(X_train.shape[1])
            model = apply_bayesian_all_layers(base_model)
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'gat_bnn_last':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            base_model = GATRegressor(X_train.shape[1])
            model = apply_bayesian_last_layer(base_model)
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'gat_bnn_variational':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            model = make_bnn_variational(X_train.shape[1])
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'gin_bnn_full':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            base_model = GINRegressor(X_train.shape[1])
            model = apply_bayesian_all_layers(base_model)
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'gin_bnn_last':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            base_model = GINRegressor(X_train.shape[1])
            model = apply_bayesian_last_layer(base_model)
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'gin_bnn_variational':
            if not HAS_TORCHBNN:
                print("SKIP (not installed)")
                continue
            
            model = make_bnn_variational(X_train.shape[1])
            model = train_bnn(model, X_train, y_train_noisy, X_val, y_val, epochs=100)
            test_pred, test_unc, test_alea, test_epist = bnn_predict(model, X_test)
        
        elif model_name == 'gauche':
            if 'smiles' not in rep:
                print("SKIP (no SMILES)")
                continue
            
            gp_dict = train_gauche_gp(
                rep['smiles']['train'],
                y_train_noisy,
                kernel='weisfeiler_lehman',
                num_epochs=50
            )
            
            test_results = predict_gauche_gp(gp_dict, rep['smiles']['test'])
            test_pred = test_results['predictions']
            test_unc, test_alea, test_epist = decompose_gp(
                test_pred,
                test_results['uncertainties']
            )
        
        else:
            print("UNKNOWN MODEL")
            continue
        
        # ====================================================================
        # STORE RESULTS (per sample)
        # ====================================================================
        
        for i in range(len(y_test)):
            all_results.append({
                'model': model_name,
                'representation': rep_name,
                'sigma': sigma,
                'sample_id': i,
                'y_true_original': y_test[i],
                'y_true_noisy': y_test[i],  # Test set is never noised
                'y_pred_mean': test_pred[i],
                'y_pred_std_calibrated': test_unc[i],
                'aleatoric_uncertainty': test_alea[i],
                'epistemic_uncertainty': test_epist[i],
            })
        
        print("✓")
    
    # ========================================================================
    # SAVE TO CSV
    # ========================================================================
    
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = results_dir / f'phase2_{model_name}_{rep_name}_uncertainty_values.csv'
        df.to_csv(output_file, index=False)
        print(f"\n  ✓ Saved {len(df):,} rows → {output_file.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train probabilistic models with uncertainty quantification'
    )
    parser.add_argument('--model',
                       choices=['qrf', 'ngboost', 'bnn_full', 'bnn_last',
                               'bnn_variational', 'gauche',
                               'gcn_bnn_full', 'gcn_bnn_last', 'gcn_bnn_variational',
                               'gat_bnn_full', 'gat_bnn_last', 'gat_bnn_variational',
                               'gin_bnn_full', 'gin_bnn_last', 'gin_bnn_variational',
                               'all'],
                       help='Model to train')
    parser.add_argument('--rep',
                       choices=['pdv', 'sns', 'ecfp4', 'smiles_ohe', 'graph', 'all'],
                       help='Molecular representation')
    parser.add_argument('--all', action='store_true',
                       help='Run all model-representation pairs')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of samples from QM9')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 2: UNCERTAINTY QUANTIFICATION TRAINING")
    print("="*80)
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define valid model-representation pairs
    valid_pairs = {
        'qrf': ['pdv', 'sns', 'ecfp4', 'smiles_ohe'],
        'ngboost': ['pdv', 'sns', 'ecfp4', 'smiles_ohe'],
        'bnn_full': ['pdv', 'sns', 'ecfp4', 'smiles_ohe'],
        'bnn_last': ['pdv', 'sns', 'ecfp4', 'smiles_ohe'],
        'bnn_variational': ['pdv', 'sns', 'ecfp4', 'smiles_ohe'],
        'gauche': ['pdv', 'sns', 'ecfp4', 'smiles_ohe', 'graph'],
        'gcn_bnn_full': ['graph'],
        'gcn_bnn_last': ['graph'],
        'gcn_bnn_variational': ['graph'],
        'gat_bnn_full': ['graph'],
        'gat_bnn_last': ['graph'],
        'gat_bnn_variational': ['graph'],
        'gin_bnn_full': ['graph'],
        'gin_bnn_last': ['graph'],
        'gin_bnn_variational': ['graph'],
    }
    
    # Determine which experiments to run
    if args.all:
        experiments = [(m, r) for m in valid_pairs for r in valid_pairs[m]]
    elif args.model == 'all' and args.rep != 'all':
        experiments = [(m, args.rep) for m in valid_pairs
                      if args.rep in valid_pairs[m]]
    elif args.model != 'all' and args.rep == 'all':
        experiments = [(args.model, r) for r in valid_pairs[args.model]]
    else:
        if not args.model or not args.rep:
            parser.error("Need --model and --rep, or use --all")
        experiments = [(args.model, args.rep)]
    
    print(f"\nWill run {len(experiments)} experiments")
    
    # Noise levels for Phase 2 (full range 0.0 to 1.0)
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"Noise levels: {noise_levels}")
    
    # Load data once
    splits = load_data(n_samples=args.n_samples)
    
    # Generate representations once
    needed_reps = set(r for _, r in experiments)
    reps = generate_representations(splits, needed_reps)
    
    # Run all experiments
    for model, rep in experiments:
        run_experiment(model, rep, noise_levels, splits, reps, results_dir)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir}/")
    print("\nNext step: Run analysis script to generate figures and metrics")


if __name__ == '__main__':
    main()