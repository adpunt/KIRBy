import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
QM9 Graph Models - Uncertainty Quantification
==============================================

Refactored for parallel execution.

Focused uncertainty quantification for graph neural networks.
Tests Bayesian GNN variants and Graph-GP under noise.

Noise levels: FIXED at [0.0, 0.3, 0.6] for consistency
Parallelize by MODEL, not by noise levels.

Models:
- Graph-GP (Weisfeiler-Lehman kernel)
- GCN-BNN (Bayesian Graph Convolutional Network)
- GAT-BNN (Bayesian Graph Attention Network)
- GIN-BNN (Bayesian Graph Isomorphism Network)
- MPNN-BNN (Bayesian Message Passing Neural Network)

Outputs:
1. MODEL_uncertainty_values.csv - Per-sample uncertainties for analysis
2. MODEL_unique_results.csv - UNIQUE evaluation (which UQ metric is best)

Usage:
    # Run specific model (with ALL noise levels)
    python script.py --model Graph-GP
    python script.py --model GCN-BNN
    
    # Run all (sequential)
    python script.py --all
    
    # Parallel execution
    python script.py --model Graph-GP &
    python script.py --model GCN-BNN &
    python script.py --model GAT-BNN &
    wait
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from scipy.stats import spearmanr
import sys
import yaml
import tempfile
from pathlib import Path
from rdkit import Chem

# Bayesian NN support
try:
    import torchbnn as bnn
    from torchhk.transform import transform_model, transform_layer
    HAS_BNN = True
except ImportError:
    HAS_BNN = False
    print("ERROR: torchbnn/torchhk required for this script")
    print("Install with: pip install torchbnn torchhk")
    sys.exit(1)

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.representations.molecular import (
    train_gauche_gp,
    predict_gauche_gp
)

# NoiseInject imports
from noiseInject import NoiseInjectorRegression

# UNIQUE imports - ADDED
try:
    from unique.pipeline import Pipeline
    HAS_UNIQUE = True
except ImportError:
    HAS_UNIQUE = False
    print("WARNING: UNIQUE not installed")


# =============================================================================
# GRAPH CONVERSION
# =============================================================================

def smiles_to_graph(smiles, y_value=None):
    """Convert SMILES to PyTorch Geometric Data object"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    
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
    
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_features = [
            float(bond.GetBondTypeAsDouble()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]
        
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
# GNN MODELS
# =============================================================================

class GCNRegressor(nn.Module):
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
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_node_features, hidden_dim, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
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
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.initial_projection = nn.Linear(num_node_features, hidden_dim)
        self.msg_layers = nn.ModuleList()
        for i in range(num_layers):
            self.msg_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        self.update_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.update_layers.append(nn.GRUCell(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.initial_projection(x)
        for layer in range(self.num_layers):
            row, col = edge_index
            messages = torch.cat([x[row], x[col]], dim=1)
            messages = self.msg_layers[layer](messages)
            messages = F.relu(messages)
            aggregated = torch.zeros(x.size(0), messages.size(1), device=x.device)
            aggregated.index_add_(0, row, messages)
            x = self.update_layers[layer](aggregated, x)
            if layer < self.num_layers - 1:
                x = F.dropout(x, p=0.2, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()


# =============================================================================
# BAYESIAN TRANSFORMATIONS
# =============================================================================

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


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_bayesian_gnn(train_loader, val_loader, test_loader, model_class,
                       num_node_features, epochs=100, lr=1e-3):
    """
    Train Bayesian GNN and return predictions with uncertainty
    
    Returns:
        predictions: Mean predictions
        uncertainties: Prediction uncertainties (epistemic)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create standard model
    model = model_class(num_node_features).to(device)
    
    # Apply Bayesian transformation to last layer
    model = apply_bayesian_transformation_last_layer(model)
    model.to(device)
    
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
    
    # Test predictions with uncertainty
    model.eval()
    num_samples = 100
    all_predictions = []
    
    for _ in range(num_samples):
        batch_predictions = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                batch_predictions.extend(out.cpu().numpy().tolist())
        all_predictions.append(batch_predictions)
    
    all_predictions = np.array(all_predictions)
    predictions = all_predictions.mean(axis=0)
    uncertainties = all_predictions.std(axis=0)
    
    return predictions, uncertainties


# =============================================================================
# UNCERTAINTY ANALYSIS
# =============================================================================

def analyze_uncertainty(y_true, y_pred, uncertainties, sigma, model_name):
    """
    Analyze uncertainty quality
    
    Returns:
        Dict with correlation, calibration metrics
    """
    errors = np.abs(y_true - y_pred)
    
    # Uncertainty-error correlation (Spearman)
    if uncertainties.std() > 0:
        corr, pval = spearmanr(uncertainties, errors)
    else:
        corr, pval = 0.0, 1.0
    
    # Calibration: RMSE of bins
    n_bins = 10
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_calibration = []
    
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_unc = uncertainties[mask].mean()
            bin_err = errors[mask].mean()
            bin_calibration.append(abs(bin_unc - bin_err))
    
    calibration_error = np.mean(bin_calibration) if bin_calibration else 0.0
    
    return {
        'model': model_name,
        'sigma': sigma,
        'uncertainty_error_corr': corr,
        'correlation_pval': pval,
        'calibration_error': calibration_error,
        'mean_uncertainty': uncertainties.mean(),
        'std_uncertainty': uncertainties.std(),
        'mean_error': errors.mean(),
        'std_error': errors.std()
    }


# =============================================================================
# UNIQUE INTEGRATION - ADDED
# =============================================================================

def run_unique_analysis_graphs(y_true, y_pred, uncertainties, sigma, model_name, results_dir):
    """
    Run UNIQUE for graph models (simpler - just epistemic uncertainty)
    
    Args:
        y_true: True labels
        y_pred: Predictions
        uncertainties: Uncertainty values (std dev)
        sigma: Noise level
        model_name: Model name
        results_dir: Output directory
    
    Returns:
        Dict with UNIQUE results or None
    """
    if not HAS_UNIQUE:
        return None
    
    try:
        n_samples = len(y_true)
        
        # Create DataFrame
        unique_df = pd.DataFrame({
            'ID': [f'sample_{i}' for i in range(n_samples)],
            'labels': y_true,
            'predictions': y_pred,
            'which_set': ['TEST'] * n_samples,
            'variance': uncertainties ** 2,  # Convert std to variance
        })
        
        # Config
        config = {
            'data_path': None,
            'output_path': None,
            'id_column_name': 'ID',
            'labels_column_name': 'labels',
            'predictions_column_name': 'predictions',
            'which_set_column_name': 'which_set',
            'model_name': model_name,
            'problem_type': 'regression',
            'mode': 'compact',
            'inputs_list': [
                {'ModelInputType': {'column_name': 'variance'}}
            ],
            'error_models_list': [],
            'individual_plots': False,
            'summary_plots': False,
            'save_plots': False,
            'evaluate_test_only': True,
            'display_outputs': False,
            'n_bootstrap': 50,
            'verbose': False
        }
        
        # Run UNIQUE
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            data_file = tmpdir / 'data.csv'
            unique_df.to_csv(data_file, index=False)
            
            output_dir = tmpdir / 'output'
            output_dir.mkdir()
            
            config['data_path'] = str(data_file)
            config['output_path'] = str(output_dir)
            
            config_file = tmpdir / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            pipeline = Pipeline.from_config(str(config_file))
            uq_outputs, eval_results = pipeline.fit()
            
            # Save results
            uq_file = results_dir / f'unique_{model_name}_sigma{sigma:.1f}.csv'
            pd.DataFrame(uq_outputs).to_csv(uq_file, index=False)
            
            # Extract metrics
            best_method = 'variance'
            spearman = 0.0
            if 'ranking_metrics' in eval_results:
                ranking = eval_results['ranking_metrics']
                if ranking and 'variance' in ranking:
                    spearman = ranking['variance'].get('spearman_correlation', 0.0)
            
            return {
                'unique_best_metric': best_method,
                'unique_spearman': spearman
            }
            
    except Exception as e:
        print(f"  UNIQUE error: {e}")
        return None


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def save_model_results(sample_data, unique_results, model_name, results_dir):
    """
    Save both per-sample uncertainty data and UNIQUE evaluation results
    """
    if not sample_data and not unique_results:
        return
    
    # Save per-sample uncertainty values
    if sample_data:
        sample_df = pd.DataFrame(sample_data)
        sample_file = results_dir / f'{model_name}_uncertainty_values.csv'
        sample_df.to_csv(sample_file, index=False)
        print(f"  ✓ Saved per-sample data: {sample_file.name} ({len(sample_df)} rows)")
    
    # Save UNIQUE results
    if unique_results:
        unique_df = pd.DataFrame(unique_results)
        unique_file = results_dir / f'{model_name}_unique_results.csv'
        unique_df.to_csv(unique_file, index=False)
        print(f"  ✓ Saved UNIQUE results: {unique_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='QM9 Graph Uncertainty Quantification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific model (with all noise levels: 0.0, 0.3, 0.6)
  python script.py --model Graph-GP
  python script.py --model GCN-BNN
  
  # Run all models (sequential)
  python script.py --all
  
  # Parallel execution
  python script.py --model Graph-GP &
  python script.py --model GCN-BNN &
  python script.py --model GAT-BNN &
  wait
        """
    )
    parser.add_argument('--model', type=str,
                       choices=['Graph-GP', 'GCN-BNN', 'GAT-BNN', 'GIN-BNN', 'MPNN-BNN', 'all'],
                       help='Which model to run')
    parser.add_argument('--all', action='store_true',
                       help='Run all models (sequential)')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--n-samples', type=int, default=10000)
    args = parser.parse_args()
    
    print("="*80)
    print(f"QM9 Graph Models - Uncertainty Quantification (seed={args.random_seed})")
    print("="*80)
    
    # Configuration
    strategy = 'legacy'
    sigma_levels = [0.0, 0.3, 0.6]  # FIXED - consistent with other experiments
    results_dir = Path(__file__).parent.parent / 'results' / 'qm9_graphs_uncertainty' / f'seed_{args.random_seed}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define models
    all_models = ['Graph-GP', 'GCN-BNN', 'GAT-BNN', 'GIN-BNN', 'MPNN-BNN']
    
    # Parse which models to run
    if args.all:
        models_to_run = all_models
    else:
        if not args.model:
            parser.error("Must specify --model (or use --all)")
        models_to_run = all_models if args.model == 'all' else [args.model]
    
    print(f"\nConfiguration:")
    print(f"  Models to run: {models_to_run}")
    print(f"  Noise levels: {sigma_levels} (FIXED)")
    print(f"  Random seed: {args.random_seed}")
    
    # Load QM9 once
    print(f"\nLoading QM9 (n={args.n_samples})...")
    raw_data = load_qm9(n_samples=args.n_samples, property_idx=4)
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    train_smiles = splits['train']['smiles']
    train_labels = np.array(splits['train']['labels'])
    val_smiles = splits['val']['smiles']
    val_labels = np.array(splits['val']['labels'])
    test_smiles = splits['test']['smiles']
    test_labels = np.array(splits['test']['labels'])
    
    print(f"Splits: Train={len(train_smiles)}, Val={len(val_smiles)}, Test={len(test_smiles)}")
    
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    
    # Convert to graphs once if needed for BNN models
    if any('BNN' in m for m in models_to_run):
        print("\nConverting SMILES to graphs...")
        train_graphs = [smiles_to_graph(s, y) for s, y in zip(train_smiles, train_labels)]
        train_graphs = [g for g in train_graphs if g is not None]
        val_graphs = [smiles_to_graph(s, y) for s, y in zip(val_smiles, val_labels)]
        val_graphs = [g for g in val_graphs if g is not None]
        test_graphs = [smiles_to_graph(s, y) for s, y in zip(test_smiles, test_labels)]
        test_graphs = [g for g in test_graphs if g is not None]
        print(f"  Converted: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")
    
    # Run each model
    for model_name in models_to_run:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")
        
        all_sample_data = []
        unique_summaries = []
        
        for sigma in sigma_levels:
            print(f"  σ={sigma:.1f}...", end='', flush=True)
            
            y_noisy = injector.inject(train_labels, sigma) if sigma > 0 else train_labels
            
            # Run model
            if model_name == 'Graph-GP':
                gp_dict = train_gauche_gp(train_smiles, y_noisy, kernel='weisfeiler_lehman', num_epochs=50)
                gp_results = predict_gauche_gp(gp_dict, test_smiles)
                predictions = gp_results['predictions']
                uncertainties = gp_results['uncertainties']
            
            else:  # BNN models
                if model_name == 'GCN-BNN':
                    model_class = GCNRegressor
                elif model_name == 'GAT-BNN':
                    model_class = GATRegressor
                elif model_name == 'GIN-BNN':
                    model_class = GINRegressor
                elif model_name == 'MPNN-BNN':
                    model_class = MPNNRegressor
                
                # Update train graphs with noisy labels
                train_graphs_noisy = [smiles_to_graph(s, y) for s, y in zip(train_smiles, y_noisy)]
                train_graphs_noisy = [g for g in train_graphs_noisy if g is not None]
                
                # Train model
                model = model_class(
                    in_channels=6,
                    hidden_channels=64,
                    num_layers=3,
                    dropout=0.2
                ).to('cuda' if torch.cuda.is_available() else 'cpu')
                
                model = train_bayesian_gnn(
                    model, train_graphs_noisy, val_graphs,
                    epochs=50, batch_size=32
                )
                
                # Predict with uncertainty
                predictions, uncertainties = predict_with_uncertainty_gnn(
                    model, test_graphs, n_samples=30
                )
            
            # Store per-sample data
            for i in range(len(test_labels)):
                all_sample_data.append({
                    'model': model_name,
                    'sigma': sigma,
                    'sample_id': i,
                    'y_true_original': test_labels[i],
                    'y_pred_mean': predictions[i],
                    'total_uncertainty': uncertainties[i],
                })
            
            # Run UNIQUE (pass single uncertainty array)
            unique_res = run_unique_analysis_graphs(
                test_labels, predictions, uncertainties, sigma,
                model_name, results_dir
            )
            
            if unique_res:
                unique_summaries.append(unique_res)
            
            # Quick analysis for printing
            errors = np.abs(test_labels - predictions)
            corr, _ = spearmanr(uncertainties, errors)
            print(f" ρ={corr:.3f}")
        
        # Save results
        save_model_results(all_sample_data, unique_summaries, model_name, results_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nResults saved in: {results_dir}/")
    print(f"  Per-sample data: MODEL_uncertainty_values.csv")
    print(f"  UNIQUE results: MODEL_unique_results.csv")

if __name__ == '__main__':
    main()