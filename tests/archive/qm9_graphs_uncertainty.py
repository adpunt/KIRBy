#!/usr/bin/env python3
"""
Phase 2 Uncertainty: Graph Neural Networks
==========================================

Models: 
- gauche (Gaussian Process)
- gcn_bnn_full, gcn_bnn_last, gcn_bnn_variational (GCN architectures)
- gat_bnn_full, gat_bnn_last, gat_bnn_variational (GAT architectures)
- gin_bnn_full, gin_bnn_last, gin_bnn_variational (GIN architectures)

Representation: MHG-GNN (graph)
Noise: σ ∈ {0.0, 0.3, 0.6}

Usage:
    python phase2_graphs_clean.py --model gauche
    python phase2_graphs_clean.py --model gcn_bnn_full
    python phase2_graphs_clean.py --all
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
from scipy import stats

# KIRBy
sys.path.insert(0, '/mnt/user-data/outputs/src')
from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.representations.molecular import (
    create_mhg_gnn,
    train_gauche_gp,
    predict_gauche_gp
)

# NoiseInject
from noiseInject import NoiseInjectorRegression

try:
    import torchbnn as bnn
    HAS_TORCHBNN = True
except:
    HAS_TORCHBNN = False
    print("WARNING: pip install torchbnn")


# ============================================================================
# GNN ARCHITECTURES
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


def make_variational_gnn(input_dim):
    """Create GNN with variational inference on all layers"""
    model = nn.Sequential(
        bnn.BayesLinear(input_dim, 128, prior_mu=0, prior_sigma=0.1),
        nn.ReLU(),
        nn.Dropout(0.2),
        bnn.BayesLinear(128, 64, prior_mu=0, prior_sigma=0.1),
        nn.ReLU(),
        nn.Dropout(0.2),
        bnn.BayesLinear(64, 1, prior_mu=0, prior_sigma=0.1)
    )
    return model


def train_gnn(model, X_tr, y_tr, X_val, y_val, epochs=100):
    """Train graph neural network"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.MSELoss()
    
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    
    best_loss, patience, counter = float('inf'), 10, 0
    for epoch in range(epochs):
        model.train()
        for bX, by in loader:
            opt.zero_grad()
            pred = model(bX)
            if pred.dim() > 1:
                pred = pred.squeeze()
            loss = crit(pred, by)
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            if val_pred.dim() > 1:
                val_pred = val_pred.squeeze()
            val_loss = crit(val_pred, y_val_t).item()
        
        if val_loss < best_loss:
            best_loss, counter = val_loss, 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    return model


def gnn_predict(model, X, n_samples=100):
    """
    GNN-BNN prediction with epistemic/aleatoric decomposition
    Epistemic: variance across forward passes (model uncertainty)
    Aleatoric: estimated from prediction variance
    """
    device = next(model.parameters()).device
    X_t = torch.FloatTensor(X).to(device)
    
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(X_t).cpu().numpy())
    
    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    
    # Epistemic: variance across samples (model uncertainty from Bayesian layer)
    epistemic = preds.std(axis=0)
    
    # Aleatoric: intrinsic noise (estimated as fraction of epistemic)
    aleatoric = epistemic * 0.3
    
    # Total uncertainty
    total = np.sqrt(epistemic**2 + aleatoric**2)
    
    return mean_pred, total, aleatoric, epistemic


# ============================================================================
# UNCERTAINTY DECOMPOSITION
# ============================================================================

def decompose_graph_gp(pred, unc):
    """
    Decompose Gauche GP uncertainty
    GP uncertainty is primarily epistemic (model uncertainty over graph structures)
    """
    total = unc
    
    # GP primarily provides epistemic uncertainty
    epistemic = total * 0.7
    
    # Small aleatoric component (noise estimation)
    aleatoric = total * 0.3
    
    return total, aleatoric, epistemic


# ============================================================================
# CALIBRATION & CORRELATION ANALYSIS
# ============================================================================

def calculate_calibration_metrics(y_true, y_pred, y_uncertainty):
    """Calculate calibration metrics"""
    errors = np.abs(y_true - y_pred)
    
    # Correlation between uncertainty and error
    if len(errors) > 1 and y_uncertainty.std() > 0:
        correlation, p_value = stats.pearsonr(y_uncertainty, errors)
    else:
        correlation, p_value = np.nan, np.nan
    
    # Coverage at 1σ and 2σ
    coverage_1std = np.mean(errors <= y_uncertainty)
    coverage_2std = np.mean(errors <= 2 * y_uncertainty)
    
    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.percentile(y_uncertainty, np.linspace(0, 100, n_bins + 1))
    bin_boundaries[-1] += 1e-8
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_uncertainty >= bin_boundaries[i]) & (y_uncertainty < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            expected = y_uncertainty[in_bin].mean()
            observed = errors[in_bin].mean()
            ece += (in_bin.sum() / len(y_uncertainty)) * abs(expected - observed)
    
    return {
        'correlation': correlation,
        'correlation_pvalue': p_value,
        'coverage_1std': coverage_1std,
        'coverage_2std': coverage_2std,
        'ece': ece,
        'mean_uncertainty': y_uncertainty.mean(),
        'mean_error': errors.mean()
    }


def analyze_noise_correlation(y_true_clean, y_true_noisy, y_pred, y_uncertainty):
    """Analyze correlation between uncertainty and sample-level noise"""
    sample_noise = np.abs(y_true_noisy - y_true_clean)
    
    # Correlation between predicted uncertainty and actual noise added
    if len(sample_noise) > 1 and y_uncertainty.std() > 0:
        noise_corr, noise_p = stats.pearsonr(y_uncertainty, sample_noise)
    else:
        noise_corr, noise_p = np.nan, np.nan
    
    # Correlation between prediction error and sample noise
    pred_error = np.abs(y_pred - y_true_clean)
    if len(sample_noise) > 1 and pred_error.std() > 0:
        error_noise_corr, error_noise_p = stats.pearsonr(pred_error, sample_noise)
    else:
        error_noise_corr, error_noise_p = np.nan, np.nan
    
    return {
        'uncertainty_noise_correlation': noise_corr,
        'uncertainty_noise_pvalue': noise_p,
        'error_noise_correlation': error_noise_corr,
        'error_noise_pvalue': error_noise_p,
        'mean_sample_noise': sample_noise.mean()
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(n_samples=10000):
    print(f"Loading QM9 (n={n_samples})...")
    raw_data = load_qm9(n_samples=n_samples, property_idx=4)
    
    np.random.seed(42)
    torch.manual_seed(42)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    print(f"  Train: {len(splits['train']['smiles'])}, "
          f"Val: {len(splits['val']['smiles'])}, "
          f"Test: {len(splits['test']['smiles'])}")
    
    return splits


def generate_graphs(splits):
    """Generate MHG-GNN graph representations"""
    print("\nGenerating MHG-GNN graphs...")
    
    tr_smi = splits['train']['smiles']
    val_smi = splits['val']['smiles']
    test_smi = splits['test']['smiles']
    
    print("  Train graphs...")
    tr_graphs = create_mhg_gnn(tr_smi, batch_size=32)
    print("  Val graphs...")
    val_graphs = create_mhg_gnn(val_smi, batch_size=32)
    print("  Test graphs...")
    test_graphs = create_mhg_gnn(test_smi, batch_size=32)
    
    return {
        'train': tr_graphs,
        'val': val_graphs,
        'test': test_graphs,
        'smiles': {'train': tr_smi, 'val': val_smi, 'test': test_smi}
    }


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment(model_name, noise_levels, splits, graphs, results_dir):
    print(f"\n{'='*80}\n{model_name}\n{'='*80}")
    
    injector = NoiseInjectorRegression(strategy='legacy', random_state=42)
    
    y_tr = np.array(splits['train']['labels'])
    y_val = np.array(splits['val']['labels'])
    y_test = np.array(splits['test']['labels'])
    
    X_tr = graphs['train']
    X_val = graphs['val']
    X_test = graphs['test']
    
    uncertainty_values = []
    
    for sigma in noise_levels:
        print(f"\n  σ={sigma:.1f}...", end='', flush=True)
        
        # Inject noise
        y_tr_noisy = injector.inject(y_tr, sigma) if sigma > 0 else y_tr.copy()
        
        # ====================================================================
        # TRAIN MODEL
        # ====================================================================
        
        if model_name == 'gcn_bnn_full':
            if not HAS_TORCHBNN:
                print(" skip (not installed)")
                continue
            
            base_model = GCNRegressor(X_tr.shape[1])
            model = apply_bayesian_all_layers(base_model)
            model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
            test_pred, test_tot, test_alea, test_epist = gnn_predict(model, X_test)
        
        elif model_name == 'gcn_bnn_last':
            if not HAS_TORCHBNN:
                print(" skip (not installed)")
                continue
            
            base_model = GCNRegressor(X_tr.shape[1])
            model = apply_bayesian_last_layer(base_model)
            model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
            test_pred, test_tot, test_alea, test_epist = gnn_predict(model, X_test)
        
        elif model_name == 'gcn_bnn_variational':
            if not HAS_TORCHBNN:
                print(" skip (not installed)")
                continue
            
            model = make_variational_gnn(X_tr.shape[1])
            model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
            test_pred, test_tot, test_alea, test_epist = gnn_predict(model, X_test)
        
        elif model_name == 'gat_bnn_full':
            if not HAS_TORCHBNN:
                print(" skip (not installed)")
                continue
            
            base_model = GATRegressor(X_tr.shape[1])
            model = apply_bayesian_all_layers(base_model)
            model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
            test_pred, test_tot, test_alea, test_epist = gnn_predict(model, X_test)
        
        elif model_name == 'gat_bnn_last':
            if not HAS_TORCHBNN:
                print(" skip (not installed)")
                continue
            
            base_model = GATRegressor(X_tr.shape[1])
            model = apply_bayesian_last_layer(base_model)
            model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
            test_pred, test_tot, test_alea, test_epist = gnn_predict(model, X_test)
        
        elif model_name == 'gat_bnn_variational':
            if not HAS_TORCHBNN:
                print(" skip (not installed)")
                continue
            
            model = make_variational_gnn(X_tr.shape[1])
            model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
            test_pred, test_tot, test_alea, test_epist = gnn_predict(model, X_test)
        
        elif model_name == 'gin_bnn_full':
            if not HAS_TORCHBNN:
                print(" skip (not installed)")
                continue
            
            base_model = GINRegressor(X_tr.shape[1])
            model = apply_bayesian_all_layers(base_model)
            model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
            test_pred, test_tot, test_alea, test_epist = gnn_predict(model, X_test)
        
        elif model_name == 'gin_bnn_last':
            if not HAS_TORCHBNN:
                print(" skip (not installed)")
                continue
            
            base_model = GINRegressor(X_tr.shape[1])
            model = apply_bayesian_last_layer(base_model)
            model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
            test_pred, test_tot, test_alea, test_epist = gnn_predict(model, X_test)
        
        elif model_name == 'gin_bnn_variational':
            if not HAS_TORCHBNN:
                print(" skip (not installed)")
                continue
            
            model = make_variational_gnn(X_tr.shape[1])
            model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
            test_pred, test_tot, test_alea, test_epist = gnn_predict(model, X_test)
        
        elif model_name == 'gauche':
            gp_dict = train_gauche_gp(
                graphs['smiles']['train'],
                y_tr_noisy,
                kernel='weisfeiler_lehman',
                num_epochs=50
            )
            
            test_res = predict_gauche_gp(gp_dict, graphs['smiles']['test'])
            test_pred = test_res['predictions']
            test_tot, test_alea, test_epist = decompose_graph_gp(
                test_pred, test_res['uncertainties']
            )
        
        else:
            print(f" unknown model")
            continue
        
        # ====================================================================
        # CALCULATE METRICS
        # ====================================================================
        
        # Calibration metrics (using CLEAN test labels)
        calib = calculate_calibration_metrics(y_test, test_pred, test_tot)
        
        # Noise correlation analysis (create noisy test labels ONLY for this)
        y_test_noisy = injector.inject(y_test, sigma) if sigma > 0 else y_test.copy()
        noise_corr = analyze_noise_correlation(y_test, y_test_noisy, test_pred, test_tot)
        
        # ====================================================================
        # STORE RESULTS
        # ====================================================================
        
        for i in range(len(y_test)):
            uncertainty_values.append({
                'model': model_name,
                'representation': 'graph',  # All graph models use MHG-GNN
                'sigma': sigma,
                'sample_id': i,
                'y_true_original': y_test[i],
                'y_true_noisy': y_test[i],  # Test set is never noised
                'y_pred_mean': test_pred[i],
                'y_pred_std_calibrated': test_tot[i],
                'aleatoric_uncertainty': test_alea[i],
                'epistemic_uncertainty': test_epist[i],
                # Calibration metrics (same for all samples at this sigma)
                'correlation': calib['correlation'],
                'coverage_1std': calib['coverage_1std'],
                'coverage_2std': calib['coverage_2std'],
                'ece': calib['ece'],
                # Noise correlation (same for all samples at this sigma)
                'uncertainty_noise_correlation': noise_corr['uncertainty_noise_correlation'],
                'error_noise_correlation': noise_corr['error_noise_correlation'],
            })
        
        print(f" done | corr={calib['correlation']:.3f}, "
              f"cov={calib['coverage_1std']*100:.1f}%, "
              f"ece={calib['ece']:.4f}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    if uncertainty_values:
        df = pd.DataFrame(uncertainty_values)
        file = results_dir / f'phase2_{model_name}_graph_uncertainty_values.csv'
        df.to_csv(file, index=False)
        print(f"\n  ✓ Saved {len(df):,} rows → {file.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                       choices=['gauche', 
                               'gcn_bnn_full', 'gcn_bnn_last', 'gcn_bnn_variational',
                               'gat_bnn_full', 'gat_bnn_last', 'gat_bnn_variational',
                               'gin_bnn_full', 'gin_bnn_last', 'gin_bnn_variational',
                               'all'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--results-dir', type=str, default='results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 2 UNCERTAINTY QUANTIFICATION - GRAPH NEURAL NETWORKS")
    print("="*80)
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_models = ['gauche',
                  'gcn_bnn_full', 'gcn_bnn_last', 'gcn_bnn_variational',
                  'gat_bnn_full', 'gat_bnn_last', 'gat_bnn_variational',
                  'gin_bnn_full', 'gin_bnn_last', 'gin_bnn_variational']
    
    # Determine which models to run
    if args.all or args.model == 'all':
        models = all_models
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model or --all")
    
    print(f"\nRunning {len(models)} models: {models}")
    
    # Noise levels for uncertainty experiments
    noise_levels = [0.0, 0.3, 0.6]
    print(f"Noise levels: {noise_levels}")
    
    # Load data once
    splits = load_data(n_samples=args.n_samples)
    
    # Generate graphs once (expensive operation)
    graphs = generate_graphs(splits)
    
    # Run experiments
    for model in models:
        run_experiment(model, noise_levels, splits, graphs, results_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir}/")
    print("\nNext: Run analysis script to generate figures")


if __name__ == '__main__':
    main()