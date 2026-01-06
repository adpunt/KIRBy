#!/usr/bin/env python3
"""
Phase 1 Graphs: Deterministic vs Bayesian GNNs
==============================================

Models: GCN, GAT, GIN, MPNN and their Bayesian counterparts
Representation: MHG-GNN
Noise: σ ∈ {0.0, 0.1, 0.2, ..., 1.0}
Replicates: 10 seeds

Usage:
    python phase1_graphs_robustness.py
    python phase1_graphs_robustness.py --n-samples 5000
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

try:
    import torchbnn as bnn
    HAS_TORCHBNN = True
except ImportError:
    HAS_TORCHBNN = False
    print("WARNING: pip install torchbnn")

from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.representations.molecular import create_mhg_gnn
from noiseInject import NoiseInjectorRegression


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class GCNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()


class GATRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()


class GINRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()


class MPNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()


# ============================================================================
# BAYESIAN TRANSFORMATIONS
# ============================================================================

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


# ============================================================================
# TRAINING
# ============================================================================

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


def predict_gnn(model, X_test):
    """Standard prediction for deterministic models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        preds = model(X_test_t)
        if preds.dim() > 1:
            preds = preds.squeeze()
    
    return preds.cpu().numpy()


def predict_bnn(model, X_test, n_samples=30):
    """Bayesian prediction with multiple forward passes"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(X_test_t)
            if pred.dim() > 1:
                pred = pred.squeeze()
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    
    return mean_pred


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(n_samples=10000, seed=42):
    """Load QM9 data"""
    print(f"\nLoading QM9 (n={n_samples}, seed={seed})...")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    raw_data = load_qm9(n_samples=n_samples, property_idx=4)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    print(f"  Train: {len(splits['train']['smiles'])}, "
          f"Val: {len(splits['val']['smiles'])}, "
          f"Test: {len(splits['test']['smiles'])}")
    
    return splits


def generate_graphs(splits):
    """Generate MHG-GNN graph representations"""
    print("Generating MHG-GNN graphs...")
    
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
    }


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_single_experiment(model_name, graphs, splits, sigma, seed):
    """Run single model at single noise level"""
    
    injector = NoiseInjectorRegression(strategy='legacy', random_state=seed)
    
    y_tr = np.array(splits['train']['labels'])
    y_val = np.array(splits['val']['labels'])
    y_test = np.array(splits['test']['labels'])
    
    X_tr = graphs['train']
    X_val = graphs['val']
    X_test = graphs['test']
    
    # Inject noise into training labels
    y_tr_noisy = injector.inject(y_tr, sigma) if sigma > 0 else y_tr.copy()
    
    # Determine model architecture
    if 'gcn' in model_name.lower():
        base_arch = GCNRegressor
    elif 'gat' in model_name.lower():
        base_arch = GATRegressor
    elif 'gin' in model_name.lower():
        base_arch = GINRegressor
    elif 'mpnn' in model_name.lower():
        base_arch = MPNNRegressor
    else:
        raise ValueError(f"Unknown architecture in {model_name}")
    
    # Create and transform model based on variant
    if model_name.upper() in ['GCN', 'GAT', 'GIN', 'MPNN']:
        # Deterministic
        model = base_arch(X_tr.shape[1])
        model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
        y_pred = predict_gnn(model, X_test)
        model_type = 'deterministic'
        
    elif 'bnn_full' in model_name.lower():
        if not HAS_TORCHBNN:
            return None
        base_model = base_arch(X_tr.shape[1])
        model = apply_bayesian_all_layers(base_model)
        model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
        y_pred = predict_bnn(model, X_test)
        model_type = 'probabilistic'
        
    elif 'bnn_last' in model_name.lower():
        if not HAS_TORCHBNN:
            return None
        base_model = base_arch(X_tr.shape[1])
        model = apply_bayesian_last_layer(base_model)
        model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
        y_pred = predict_bnn(model, X_test)
        model_type = 'probabilistic'
        
    elif 'bnn_variational' in model_name.lower():
        if not HAS_TORCHBNN:
            return None
        model = make_variational_gnn(X_tr.shape[1])
        model = train_gnn(model, X_tr, y_tr_noisy, X_val, y_val)
        y_pred = predict_bnn(model, X_test)
        model_type = 'probabilistic'
        
    else:
        raise ValueError(f"Unknown model variant: {model_name}")
    
    # Calculate metrics
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mae = np.mean(np.abs(y_test - y_pred))
    
    return {
        'model': model_name,
        'sigma': sigma,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'model_type': model_type
    }


def run_all_experiments(model_name, n_samples=10000, n_seeds=10, results_dir='results'):
    """Run experiments for a single model across seeds"""
    
    print("="*80)
    print(f"PHASE 1 GRAPH ROBUSTNESS: {model_name.upper()}")
    print("="*80)
    
    results_dir = Path(results_dir) / "phase1_graphs_updated"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Noise levels
    noise_levels = np.arange(0.0, 1.1, 0.1)
    
    print(f"\nModel: {model_name}")
    print(f"Noise levels: {len(noise_levels)}")
    print(f"Seeds: {n_seeds}")
    print(f"Total experiments: {len(noise_levels) * n_seeds}")
    
    # Run experiments by seed
    for seed in range(n_seeds):
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}")
        
        # Load data for this seed
        splits = load_data(n_samples=n_samples, seed=seed)
        graphs = generate_graphs(splits)
        
        seed_results = []
        
        # Test model at all noise levels
        with tqdm(total=len(noise_levels), desc=f"Seed {seed}") as pbar:
            for sigma in noise_levels:
                result = run_single_experiment(
                    model_name, graphs, splits, sigma, seed
                )
                
                if result is not None:
                    result['seed'] = seed
                    seed_results.append(result)
                
                pbar.update(1)
        
        # Save seed results
        seed_dir = results_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        
        # Check if file exists and append
        output_file = seed_dir / "all_results.csv"
        seed_df = pd.DataFrame(seed_results)
        
        if output_file.exists():
            existing_df = pd.read_csv(output_file)
            # Remove existing entries for this model
            existing_df = existing_df[existing_df['model'] != model_name]
            # Append new results
            combined_df = pd.concat([existing_df, seed_df], ignore_index=True)
            combined_df.to_csv(output_file, index=False)
        else:
            seed_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved {len(seed_df)} results for seed {seed}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       help='Model to run (e.g., GCN, gcn_bnn_full, GAT, etc.)')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of QM9 samples')
    parser.add_argument('--n-seeds', type=int, default=10,
                       help='Number of random seeds')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Validate model name
    valid_models = ['GCN', 'GAT', 'GIN', 'MPNN']
    valid_models += [f'{arch}_bnn_full' for arch in ['gcn', 'gat', 'gin', 'mpnn']]
    valid_models += [f'{arch}_bnn_last' for arch in ['gcn', 'gat', 'gin', 'mpnn']]
    valid_models += [f'{arch}_bnn_variational' for arch in ['gcn', 'gat', 'gin', 'mpnn']]
    
    if args.model not in valid_models:
        print(f"ERROR: Invalid model '{args.model}'")
        print(f"Valid models: {valid_models}")
        return
    
    run_all_experiments(
        model_name=args.model,
        n_samples=args.n_samples,
        n_seeds=args.n_seeds,
        results_dir=args.results_dir
    )


if __name__ == '__main__':
    main()