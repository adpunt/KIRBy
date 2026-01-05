import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
QM9 Graph Models - Uncertainty Quantification
==============================================

Focused uncertainty quantification for graph neural networks.
Tests Bayesian GNN variants and Graph-GP under noise.

Purpose: Evaluate which uncertainty quantification methods are most reliable
when training data contains noise.

Models:
- Graph-GP (Weisfeiler-Lehman kernel)
- GCN-BNN (Bayesian Graph Convolutional Network)
- GAT-BNN (Bayesian Graph Attention Network)
- GIN-BNN (Bayesian Graph Isomorphism Network)
- MPNN-BNN (Bayesian Message Passing Neural Network)

Noise Strategy: Legacy (Gaussian) only
Noise Levels: σ ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} (reduced for speed)

Analysis:
- Uncertainty-error correlation per noise level
- Calibration analysis
- Uncertainty decomposition
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

def main():
    parser = argparse.ArgumentParser(description='QM9 Graph Uncertainty Quantification')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--n-samples', type=int, default=10000)
    args = parser.parse_args()
    
    print("="*80)
    print(f"QM9 Graph Models - Uncertainty Quantification (seed={args.random_seed})")
    print("="*80)
    
    # Configuration
    strategy = 'legacy'
    sigma_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Reduced for speed
    results_dir = Path(__file__).parent.parent / 'results' / 'qm9_graphs_uncertainty' / f'seed_{args.random_seed}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load QM9
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
    
    all_uncertainty_data = []
    all_analysis = []
    
    # =========================================================================
    # Model 1: Graph-GP
    # =========================================================================
    print("\n" + "="*80)
    print("Graph-GP (Weisfeiler-Lehman Kernel)")
    print("="*80)
    
    for sigma in sigma_levels:
        print(f"  σ={sigma:.1f}...", end='')
        
        y_noisy = injector.inject(train_labels, sigma) if sigma > 0 else train_labels
        
        gp_dict = train_gauche_gp(
            train_smiles, y_noisy,
            kernel='weisfeiler_lehman',
            num_epochs=50
        )
        
        gp_results = predict_gauche_gp(gp_dict, test_smiles)
        predictions = gp_results['predictions']
        uncertainties = gp_results['uncertainties']
        
        # Store detailed data
        for i in range(len(test_labels)):
            all_uncertainty_data.append({
                'model': 'Graph-GP',
                'sigma': sigma,
                'sample_idx': i,
                'y_true': test_labels[i],
                'y_pred': predictions[i],
                'uncertainty': uncertainties[i],
                'error': abs(test_labels[i] - predictions[i])
            })
        
        # Analyze
        analysis = analyze_uncertainty(test_labels, predictions, uncertainties, sigma, 'Graph-GP')
        all_analysis.append(analysis)
        
        # UNIQUE analysis - ADDED
        unique_res = run_unique_analysis_graphs(
            test_labels, predictions, uncertainties, sigma,
            'Graph-GP', results_dir
        )
        if unique_res:
            all_analysis[-1].update(unique_res)
        
        print(f" ρ={analysis['uncertainty_error_corr']:.3f}")
    
    # =========================================================================
    # Model 2-5: Bayesian GNNs
    # =========================================================================
    print("\n" + "="*80)
    print("Bayesian Graph Neural Networks")
    print("="*80)
    
    graph_models = [
        (GCNRegressor, 'GCN-BNN'),
        (GATRegressor, 'GAT-BNN'),
        (GINRegressor, 'GIN-BNN'),
        (MPNNRegressor, 'MPNN-BNN')
    ]
    
    for model_class, model_name in graph_models:
        print(f"\n[{model_name}]")
        
        # Convert SMILES to graphs
        print("  Converting SMILES...", end='')
        train_graphs = [smiles_to_graph(s) for s in train_smiles]
        val_graphs = [smiles_to_graph(s) for s in val_smiles]
        test_graphs = [smiles_to_graph(s) for s in test_smiles]
        
        train_graphs = [g for g in train_graphs if g is not None]
        val_graphs = [g for g in val_graphs if g is not None]
        test_graphs = [g for g in test_graphs if g is not None]
        print(f" {len(train_graphs)} train, {len(test_graphs)} test")
        
        num_node_features = train_graphs[0].x.shape[1]
        
        for sigma in sigma_levels:
            print(f"  σ={sigma:.1f}...", end='')
            
            y_noisy = injector.inject(train_labels[:len(train_graphs)], sigma) if sigma > 0 else train_labels[:len(train_graphs)]
            
            # Attach labels
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
            predictions, uncertainties = train_bayesian_gnn(
                train_loader, val_loader, test_loader,
                model_class, num_node_features, epochs=100
            )
            
            # Store detailed data
            for i in range(len(test_labels)):
                all_uncertainty_data.append({
                    'model': model_name,
                    'sigma': sigma,
                    'sample_idx': i,
                    'y_true': test_labels[i],
                    'y_pred': predictions[i],
                    'uncertainty': uncertainties[i],
                    'error': abs(test_labels[i] - predictions[i])
                })
            
            # Analyze
            analysis = analyze_uncertainty(test_labels, predictions, uncertainties, sigma, model_name)
            all_analysis.append(analysis)
            
            # UNIQUE analysis - ADDED
            unique_res = run_unique_analysis_graphs(
                test_labels, predictions, uncertainties, sigma,
                model_name, results_dir
            )
            if unique_res:
                all_analysis[-1].update(unique_res)
            
            print(f" ρ={analysis['uncertainty_error_corr']:.3f}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save detailed uncertainty data
    unc_df = pd.DataFrame(all_uncertainty_data)
    unc_df.to_csv(results_dir / 'uncertainty_values.csv', index=False)
    print(f"Saved: {results_dir / 'uncertainty_values.csv'}")
    
    # Save analysis
    analysis_df = pd.DataFrame(all_analysis)
    analysis_df.to_csv(results_dir / 'uncertainty_analysis.csv', index=False)
    print(f"Saved: {results_dir / 'uncertainty_analysis.csv'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nUncertainty-Error Correlation by Model:")
    print(analysis_df.pivot_table(values='uncertainty_error_corr', 
                                   index='model', columns='sigma'))
    
    print("\nMean Correlation by Model (across noise levels):")
    print(analysis_df.groupby('model')['uncertainty_error_corr'].mean().sort_values(ascending=False))
    
    print("\nBest model at high noise (σ=0.6):")
    high_noise = analysis_df[analysis_df['sigma'] == 0.6].sort_values('uncertainty_error_corr', ascending=False)
    print(high_noise[['model', 'uncertainty_error_corr', 'calibration_error']].head())
    
    # UNIQUE summary if available
    if 'unique_spearman' in analysis_df.columns:
        print("\nUNIQUE Spearman Correlations:")
        print(analysis_df.pivot_table(values='unique_spearman',
                                      index='model', columns='sigma'))
    
    print("\n" + "="*80)
    print(f"COMPLETE - Results saved to {results_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()