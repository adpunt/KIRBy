import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
QM9 Graph Models + NoiseInject: Comprehensive Testing
======================================================

Comprehensive noise robustness testing of graph neural networks on QM9.
Designed to be run 5 times with different random seeds.

Usage:
    python qm9_graphs_noise_robustness.py --random-seed 42
    python qm9_graphs_noise_robustness.py --random-seed 43
    ... (run 5 times total)

Graph Models:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GIN (Graph Isomorphism Network)
- MPNN (Message Passing Neural Network)
- Graph-GP (Gauche with Weisfeiler-Lehman kernel)

Limited testing of expensive methods:
- MHG-GNN-finetuned + RF (1 run only)

Phases:
1. Core Robustness - Legacy strategy only, all graph models (GCN, GAT, GIN, MPNN, Graph-GP)
2. MHG-GNN Finetuned - Limited (RF only, legacy only)
3. Uncertainty Quantification - Graph-GP only (legacy strategy)

Noise Strategy: Legacy only for Phase 1
Noise Levels: σ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from sklearn.ensemble import RandomForestRegressor

# Optional ML model imports
try:
    from quantile_forest import RandomForestQuantileRegressor
    HAS_QRF = True
except ImportError:
    HAS_QRF = False
    print("WARNING: quantile_forest not installed")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed")

try:
    from ngboost import NGBRegressor
    HAS_NGBOOST = True
except ImportError:
    HAS_NGBOOST = False
    print("WARNING: ngboost not installed")

# Bayesian NN support (torchhk)
try:
    import torchbnn as bnn
    from torchhk.transform import transform_model, transform_layer
    HAS_BNN = True
except ImportError:
    HAS_BNN = False
    print("WARNING: torchbnn/torchhk not installed - BNN variants disabled")

import sys
from pathlib import Path
from rdkit import Chem

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.representations.molecular import (
    create_mhg_gnn,
    finetune_gnn,
    encode_from_model,
    train_gauche_gp,
    predict_gauche_gp
)

# NoiseInject imports
from noiseInject import (
    NoiseInjectorRegression,
    calculate_noise_metrics
)


# =============================================================================
# SMILES TO GRAPH CONVERSION
# =============================================================================

def smiles_to_graph(smiles, y_value=None):
    """Convert SMILES to PyTorch Geometric Data object"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    
    # Node features: [atomic_num, degree, formal_charge, is_aromatic, hybridization, num_h]
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.GetHybridization()),
            atom.GetTotalNumHs()
        ])
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge indices and features
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Bond features: [bond_type, is_conjugated, is_in_ring]
        bond_features = [
            float(bond.GetBondTypeAsDouble()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]
        
        # Add both directions (undirected graph)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_features)
        edge_attr.append(bond_features)
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if y_value is not None:
        data.y = torch.tensor([y_value], dtype=torch.float)
    
    return data


# =============================================================================
# GRAPH NEURAL NETWORK MODELS
# =============================================================================

class GCNRegressor(nn.Module):
    """Graph Convolutional Network for regression"""
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()


class GATRegressor(nn.Module):
    """Graph Attention Network for regression"""
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(num_node_features, hidden_dim, heads=heads, concat=True))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
        
        # Last conv layer
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False))
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()


class GINRegressor(nn.Module):
    """Graph Isomorphism Network for regression"""
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(num_node_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            self.convs.append(GINConv(mlp, train_eps=True))
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()


class MPNNRegressor(nn.Module):
    """Message Passing Neural Network for regression"""
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Initial projection to hidden_dim
        self.initial_projection = nn.Linear(num_node_features, hidden_dim)
        
        # Message functions - all work with hidden_dim after initial projection
        self.msg_layers = nn.ModuleList()
        for i in range(num_layers):
            self.msg_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        
        # Update functions - all work with hidden_dim
        self.update_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.update_layers.append(nn.GRUCell(hidden_dim, hidden_dim))
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Project initial features to hidden_dim once at the start
        x = self.initial_projection(x)
        
        for layer in range(self.num_layers):
            # Message passing
            row, col = edge_index
            messages = torch.cat([x[row], x[col]], dim=1)
            messages = self.msg_layers[layer](messages)
            messages = F.relu(messages)
            
            # Aggregate messages
            aggregated = torch.zeros(x.size(0), messages.size(1), device=x.device)
            aggregated.index_add_(0, row, messages)
            
            # Update node features
            x = self.update_layers[layer](aggregated, x)
            
            if layer < self.num_layers - 1:
                x = F.dropout(x, p=0.2, training=self.training)
        
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()


class DNNRegressor(nn.Module):
    """Simple DNN for tabular/fixed-length features"""
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()


# =============================================================================
# BAYESIAN TRANSFORMATIONS
# =============================================================================

if HAS_BNN:
    def apply_bayesian_transformation(model):
        """Convert all Linear layers to Bayesian Linear layers"""
        transform_model(
            model, 
            nn.Linear, 
            bnn.BayesLinear, 
            args={
                "prior_mu": 0, 
                "prior_sigma": 0.1, 
                "in_features": ".in_features",
                "out_features": ".out_features", 
                "bias": ".bias"
            }, 
            attrs={"weight_mu": ".weight"}
        )
        return model
    
    def apply_bayesian_transformation_last_layer(model):
        """Replace only the final Linear layer with Bayesian Linear"""
        last_linear_name = None
        last_linear_module = None
        
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                last_linear_name = name
                last_linear_module = module
                break
        
        if last_linear_module is None:
            raise ValueError("No nn.Linear layer found to replace.")
        
        bayesian_layer = transform_layer(
            last_linear_module,
            nn.Linear,
            bnn.BayesLinear,
            args={
                "prior_mu": 0,
                "prior_sigma": 0.1,
                "in_features": ".in_features",
                "out_features": ".out_features",
                "bias": ".bias"
            },
            attrs={"weight_mu": ".weight"}
        )
        
        def set_nested_attr(obj, attr_path, value):
            attrs = attr_path.split(".")
            for a in attrs[:-1]:
                obj = getattr(obj, a)
            setattr(obj, attrs[-1], value)
        
        set_nested_attr(model, last_linear_name, bayesian_layer)
        return model
    
    def apply_bayesian_transformation_last_layer_variational(model):
        """Variational Bayesian Last Layer - same as last_layer"""
        return apply_bayesian_transformation_last_layer(model)


# =============================================================================
# GNN TRAINING
# =============================================================================

def train_gnn_model(train_loader, val_loader, test_loader, model_class, 
                   num_node_features, epochs=100, lr=1e-3):
    """
    Train a graph neural network
    
    Args:
        train_loader, val_loader, test_loader: PyG DataLoaders
        model_class: GNN model class
        num_node_features: Number of node features
        epochs: Training epochs
        lr: Learning rate
    
    Returns:
        predictions: Test set predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model_class(num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
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
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            predictions.extend(out.cpu().numpy().tolist())
    
    return np.array(predictions)


def train_dnn_tabular(X_train, y_train, X_val, y_val, X_test, epochs=100, lr=1e-3):
    """Train standard DNN on tabular features"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    model = DNNRegressor(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()
    
    return predictions


def train_bnn_tabular(X_train, y_train, X_val, y_val, X_test, transformation_type, epochs=100, lr=1e-3):
    """Train Bayesian NN on tabular features with multiple forward passes for uncertainty"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Create standard model then transform
    model = DNNRegressor(X_train.shape[1]).to(device)
    
    # Apply Bayesian transformation
    if transformation_type == 'full':
        model = apply_bayesian_transformation(model)
    elif transformation_type == 'last_layer':
        model = apply_bayesian_transformation_last_layer(model)
    elif transformation_type == 'variational':
        model = apply_bayesian_transformation_last_layer_variational(model)
    else:
        raise ValueError(f"Unknown transformation type: {transformation_type}")
    
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.load_state_dict(best_state)
    
    # Multiple forward passes for uncertainty
    model.eval()
    predictions_list = []
    num_samples = 100
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(X_test_t).cpu().numpy()
            predictions_list.append(pred)
    
    # Return mean prediction
    predictions = np.mean(predictions_list, axis=0)
    return predictions



# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_experiment_gnn(train_smiles, train_labels, val_smiles, val_labels,
                      test_smiles, test_labels, model_class, strategy, 
                      sigma_levels, model_name):
    """Run noise robustness experiment for GNN"""
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions = {}
    
    # Convert SMILES to graphs once
    print("    Converting SMILES to graphs...", end='')
    train_graphs = [smiles_to_graph(s) for s in train_smiles]
    val_graphs = [smiles_to_graph(s) for s in val_smiles]
    test_graphs = [smiles_to_graph(s) for s in test_smiles]
    
    # Filter out None
    train_graphs = [g for g in train_graphs if g is not None]
    val_graphs = [g for g in val_graphs if g is not None]
    test_graphs = [g for g in test_graphs if g is not None]
    print(f" {len(train_graphs)} train, {len(test_graphs)} test")
    
    # Determine number of node features
    num_node_features = train_graphs[0].x.shape[1] if len(train_graphs) > 0 else 6
    
    for sigma in sigma_levels:
        print(f"      σ={sigma:.1f}...", end='')
        
        # Inject noise
        if sigma == 0.0:
            y_noisy = train_labels[:len(train_graphs)]
        else:
            y_noisy = injector.inject(train_labels[:len(train_graphs)], sigma)
        
        # Attach labels to graphs
        for i, graph in enumerate(train_graphs):
            graph.y = torch.tensor([y_noisy[i]], dtype=torch.float)
        for i, graph in enumerate(val_graphs):
            graph.y = torch.tensor([val_labels[i]], dtype=torch.float)
        for i, graph in enumerate(test_graphs):
            graph.y = torch.tensor([test_labels[i]], dtype=torch.float)
        
        # Create loaders
        train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)
        
        # Train
        preds = train_gnn_model(
            train_loader, val_loader, test_loader,
            model_class, num_node_features, epochs=100
        )
        predictions[sigma] = preds
        print(" done")
    
    return predictions


def run_experiment_graph_gp(train_smiles, train_labels, test_smiles, test_labels,
                            strategy, sigma_levels):
    """Run noise robustness experiment for Graph GP"""
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions = {}
    uncertainties = {}
    
    for sigma in sigma_levels:
        print(f"      σ={sigma:.1f}...", end='')
        
        # Inject noise
        if sigma == 0.0:
            y_noisy = train_labels
        else:
            y_noisy = injector.inject(train_labels, sigma)
        
        # Train GP
        gp_dict = train_gauche_gp(
            train_smiles, y_noisy,
            kernel='weisfeiler_lehman',
            num_epochs=50
        )
        
        # Predict
        gp_results = predict_gauche_gp(gp_dict, test_smiles)
        predictions[sigma] = gp_results['predictions']
        uncertainties[sigma] = gp_results['uncertainties']
        print(" done")
    
    return predictions, uncertainties


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='QM9 Graph Models Noise Robustness')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for this repetition')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of QM9 samples to use')
    args = parser.parse_args()
    
    print("="*80)
    print(f"QM9 Graph Models + NoiseInject (seed={args.random_seed})")
    print("="*80)
    
    # Configuration
    # Only legacy strategy for Phase 1
    strategies = ['legacy']
    sigma_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_dir = Path(__file__).parent.parent / 'results' / 'qm9_graphs' / f'seed_{args.random_seed}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load QM9
    print(f"\nLoading QM9 (n={args.n_samples}, property=homo_lumo_gap)...")
    raw_data = load_qm9(n_samples=args.n_samples, property_idx=4)
    
    # Get scaffold splits
    print("Creating scaffold splits...")
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    train_smiles = splits['train']['smiles']
    train_labels = np.array(splits['train']['labels'])
    val_smiles = splits['val']['smiles']
    val_labels = np.array(splits['val']['labels'])
    test_smiles = splits['test']['smiles']
    test_labels = np.array(splits['test']['labels'])
    
    print(f"Split sizes: Train={len(train_smiles)}, Val={len(val_smiles)}, Test={len(test_smiles)}")
    
    # Storage
    all_results = []
    all_uncertainties = []
    
    # =========================================================================
    # PHASE 1: CORE ROBUSTNESS - Graph Models
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: CORE ROBUSTNESS - Graph Neural Networks")
    print("="*80)
    
    # TODO: uncomment, only for testing
    graph_models = [
        # (GCNRegressor, 'GCN'),
        # (GATRegressor, 'GAT'),
        # (GINRegressor, 'GIN'),
        (MPNNRegressor, 'MPNN')
    ]
    
    for model_class, model_name in graph_models:
        print(f"\n[{model_name}]")
        
        for strategy in strategies:
            print(f"  Strategy: {strategy}")
            
            predictions = run_experiment_gnn(
                train_smiles, train_labels, val_smiles, val_labels,
                test_smiles, test_labels, model_class, strategy,
                sigma_levels, model_name
            )
            
            per_sigma, summary = calculate_noise_metrics(
                test_labels, predictions, metrics=['r2', 'rmse', 'mae']
            )
            per_sigma['model'] = model_name
            per_sigma['rep'] = 'graph'
            per_sigma['strategy'] = strategy
            per_sigma['seed'] = args.random_seed
            all_results.append(per_sigma)
            per_sigma.to_csv(results_dir / f'{model_name}_graph_{strategy}.csv', index=False)
    
    # =========================================================================
    # Graph GP
    # =========================================================================
    print(f"\n[Graph-GP]")
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        
        predictions, uncertainties = run_experiment_graph_gp(
            train_smiles, train_labels, test_smiles, test_labels,
            strategy, sigma_levels
        )
        
        per_sigma, summary = calculate_noise_metrics(
            test_labels, predictions, metrics=['r2', 'rmse', 'mae']
        )
        per_sigma['model'] = 'Graph-GP'
        per_sigma['rep'] = 'graph'
        per_sigma['strategy'] = strategy
        per_sigma['seed'] = args.random_seed
        all_results.append(per_sigma)
        per_sigma.to_csv(results_dir / f'GraphGP_graph_{strategy}.csv', index=False)
        
        # Save uncertainties for Phase 3
        if strategy == 'legacy':
            unc_data = []
            for sigma in sigma_levels:
                for i in range(len(test_labels)):
                    unc_data.append({
                        'sigma': sigma,
                        'sample_idx': i,
                        'y_true': test_labels[i],
                        'y_pred': predictions[sigma][i],
                        'uncertainty': uncertainties[sigma][i],
                        'error': abs(test_labels[i] - predictions[sigma][i])
                    })
            all_uncertainties.append(('Graph-GP', pd.DataFrame(unc_data)))
    
    # =========================================================================
    # PHASE 2: EXPENSIVE METHODS (Limited Testing)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: EXPENSIVE METHODS (Limited Testing)")
    print("="*80)
    
    # Only run on first seed to avoid redundancy
    if args.random_seed == 42:
        # MHG-GNN finetuned + RF (1 run only, very expensive)
        print("\n[MHG-GNN-finetuned + RF] (1 run only, legacy strategy only)")
        strategy = 'legacy'
        
        print("  Fine-tuning MHG-GNN...", end='')
        gnn_model = finetune_gnn(
            train_smiles, train_labels,
            val_smiles, val_labels,
            gnn_type='gcn', hidden_dim=128, epochs=50
        )
        
        mhggnn_train = encode_from_model(gnn_model, train_smiles)
        mhggnn_test = encode_from_model(gnn_model, test_smiles)
        print(" done")
        
        print(f"  Strategy: {strategy}")
        injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
        predictions = {}
        
        for sigma in sigma_levels:
            print(f"    σ={sigma:.1f}...", end='')
            
            if sigma == 0.0:
                y_noisy = train_labels
            else:
                y_noisy = injector.inject(train_labels, sigma)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(mhggnn_train, y_noisy)
            predictions[sigma] = model.predict(mhggnn_test)
            print(" done")
        
        per_sigma, summary = calculate_noise_metrics(
            test_labels, predictions, metrics=['r2', 'rmse', 'mae']
        )
        per_sigma['model'] = 'RF'
        per_sigma['rep'] = 'MHG-GNN-finetuned'
        per_sigma['strategy'] = strategy
        per_sigma['seed'] = args.random_seed
        all_results.append(per_sigma)
        per_sigma.to_csv(results_dir / f'RF_MHGGNN-finetuned_{strategy}.csv', index=False)
    
    # =========================================================================
    # PHASE 3: UNCERTAINTY QUANTIFICATION
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: UNCERTAINTY QUANTIFICATION (Graph-GP, legacy only)")
    print("="*80)
    
    for model_name, unc_df in all_uncertainties:
        unc_df.to_csv(
            results_dir / f'{model_name}_uncertainty_values.csv',
            index=False
        )
        print(f"Saved {model_name} uncertainties")
        
        # Calculate correlation
        print(f"\n{model_name} - Uncertainty-Error Correlation:")
        for sigma in sigma_levels:
            subset = unc_df[unc_df['sigma'] == sigma]
            if len(subset) > 0 and subset['uncertainty'].std() > 0:
                corr = np.corrcoef(subset['uncertainty'], subset['error'])[0, 1]
                print(f"  σ={sigma:.1f}: ρ = {corr:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_dir / 'all_results.csv', index=False)
    
    # Calculate NSI and retention
    summary_table = []
    for (model, rep, strategy), group in combined_df.groupby(['model', 'rep', 'strategy']):
        baseline_rows = group[group['sigma'] == 0.0]
        high_noise_rows = group[group['sigma'] == 0.6]
        
        if len(baseline_rows) > 0 and len(high_noise_rows) > 0:
            baseline = baseline_rows['r2'].values[0]
            high_noise = high_noise_rows['r2'].values[0]
            nsi = (baseline - high_noise) / 0.6
            retention = (high_noise / baseline) * 100 if baseline > 0 else 0
            
            summary_table.append({
                'model': model,
                'rep': rep,
                'strategy': strategy,
                'baseline_r2': baseline,
                'r2_at_0.6': high_noise,
                'NSI': nsi,
                'retention_%': retention,
                'seed': args.random_seed
            })
    
    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(results_dir / 'summary.csv', index=False)
    
    print("\nTop 5 most robust (lowest NSI):")
    top5 = summary_df.nsmallest(5, 'NSI')[['model', 'strategy', 'NSI', 'retention_%']]
    print(top5.to_string(index=False))
    
    print("\nMean performance by model:")
    model_summary = summary_df.groupby('model')[['NSI', 'retention_%']].mean()
    print(model_summary.to_string())
    
    print("\n" + "="*80)
    print(f"COMPLETE - Results saved to {results_dir}/")
    print(f"Run this script 4 more times with different seeds (43, 44, 45, 46)")
    print("="*80)


if __name__ == '__main__':
    main()