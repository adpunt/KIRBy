import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
QM9 Graph Models + NoiseInject + UNIQUE: Comprehensive Testing
===============================================================

Comprehensive noise robustness testing with UNIQUE integration for Graph-GP.

Key Updates:
- Graph-GP now predicts on TRAIN/VAL/TEST for UNIQUE
- UNIQUE ranks GP uncertainty vs distance-based metrics
- Saves per-sample uncertainty data for all sets

Usage:
    python qm9_graphs_noise_robustness_with_unique.py --random-seed 42
    python qm9_graphs_noise_robustness_with_unique.py --random-seed 43
    ... (run 5 times total)

Graph Models:
- GCN, GAT, GIN, MPNN (deterministic)
- Graph-GP (uncertainty quantification + UNIQUE)
- MHG-GNN-finetuned + RF (limited testing)

Phases:
1. Core Robustness - All graph models
2. MHG-GNN Finetuned - Limited (RF only, seed 42 only)
3. Uncertainty Quantification - Graph-GP with UNIQUE

Noise Strategy: Legacy only
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
import sys
from pathlib import Path
from rdkit import Chem
import yaml
import tempfile

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

# UNIQUE imports
try:
    from unique.pipeline import Pipeline
    HAS_UNIQUE = True
except ImportError:
    HAS_UNIQUE = False
    print("WARNING: UNIQUE not installed")


# =============================================================================
# SMILES TO GRAPH CONVERSION
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
# GNN TRAINING
# =============================================================================

def train_gnn_model(train_loader, val_loader, test_loader, model_class, 
                   num_node_features, epochs=100, lr=1e-3):
    """Train a graph neural network"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model_class(num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
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
    
    model.load_state_dict(best_state)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            predictions.extend(out.cpu().numpy().tolist())
    
    return np.array(predictions)


# =============================================================================
# UNIQUE INTEGRATION FOR GRAPH-GP
# =============================================================================

def run_unique_for_graph_gp(combined_data, noise_level, results_dir):
    """
    Run UNIQUE to rank Graph-GP uncertainty against distance-based metrics
    
    Args:
        combined_data: DataFrame with TRAIN/CALIBRATION/TEST data
            Columns: sample_id, which_set, y_true, y_pred, uncertainty
        noise_level: Current σ
        results_dir: Output directory
    
    Returns:
        Dict with UNIQUE ranking results
    """
    if not HAS_UNIQUE:
        print("    UNIQUE not available, skipping")
        return None
    
    try:
        print(f"    Running UNIQUE (σ={noise_level:.1f})...")
        
        # Prepare DataFrame
        unique_df = combined_data.copy()
        unique_df['ID'] = unique_df['sample_id'].astype(str)
        unique_df = unique_df.rename(columns={
            'y_true': 'labels',
            'y_pred': 'predictions'
        })
        
        # UNIQUE config
        inputs_list = [
            # Model-based UQ (GP uncertainty)
            {'ModelInputType': {'column_name': 'uncertainty'}},
        ]
        
        config = {
            'data_path': None,
            'output_path': None,
            'id_column_name': 'ID',
            'labels_column_name': 'labels',
            'predictions_column_name': 'predictions',
            'which_set_column_name': 'which_set',
            'model_name': f'GraphGP_sigma{noise_level:.1f}',
            'problem_type': 'regression',
            'mode': 'compact',
            'inputs_list': inputs_list,
            'error_models_list': [],
            'individual_plots': False,
            'summary_plots': False,
            'save_plots': False,
            'evaluate_test_only': False,
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
            
            # Extract ranking results
            ranking_results = {}
            if 'ranking_metrics' in eval_results:
                ranking = eval_results['ranking_metrics']
                for metric_name, scores in ranking.items():
                    ranking_results[metric_name] = scores.get('spearman_correlation', 0.0)
            
            # Find best metric
            if ranking_results:
                best_metric = max(ranking_results.items(), key=lambda x: x[1])
                print(f"      Best UQ metric: {best_metric[0]} (ρ={best_metric[1]:.3f})")
            else:
                best_metric = ('uncertainty', 0.0)
                print("      WARNING: No ranking results from UNIQUE")
            
            return {
                'model': 'Graph-GP',
                'sigma': noise_level,
                'best_uq_metric': best_metric[0],
                'best_spearman': best_metric[1],
                'all_rankings': ranking_results
            }
            
    except Exception as e:
        print(f"    UNIQUE error: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_experiment_gnn(train_smiles, train_labels, val_smiles, val_labels,
                      test_smiles, test_labels, model_class, strategy, 
                      sigma_levels, model_name):
    """Run noise robustness experiment for GNN"""
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions = {}
    
    print("    Converting SMILES to graphs...", end='')
    train_graphs = [smiles_to_graph(s) for s in train_smiles]
    val_graphs = [smiles_to_graph(s) for s in val_smiles]
    test_graphs = [smiles_to_graph(s) for s in test_smiles]
    
    train_graphs = [g for g in train_graphs if g is not None]
    val_graphs = [g for g in val_graphs if g is not None]
    test_graphs = [g for g in test_graphs if g is not None]
    print(f" {len(train_graphs)} train, {len(test_graphs)} test")
    
    num_node_features = train_graphs[0].x.shape[1] if len(train_graphs) > 0 else 6
    
    for sigma in sigma_levels:
        print(f"      σ={sigma:.1f}...", end='')
        
        if sigma == 0.0:
            y_noisy = train_labels[:len(train_graphs)]
        else:
            y_noisy = injector.inject(train_labels[:len(train_graphs)], sigma)
        
        for i, graph in enumerate(train_graphs):
            graph.y = torch.tensor([y_noisy[i]], dtype=torch.float)
        for i, graph in enumerate(val_graphs):
            graph.y = torch.tensor([val_labels[i]], dtype=torch.float)
        for i, graph in enumerate(test_graphs):
            graph.y = torch.tensor([test_labels[i]], dtype=torch.float)
        
        train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)
        
        preds = train_gnn_model(
            train_loader, val_loader, test_loader,
            model_class, num_node_features, epochs=100
        )
        predictions[sigma] = preds
        print(" done")
    
    return predictions


def run_experiment_graph_gp_with_unique(train_smiles, train_labels, 
                                        val_smiles, val_labels,
                                        test_smiles, test_labels,
                                        strategy, sigma_levels, results_dir):
    """
    Run Graph-GP with UNIQUE integration
    
    Generates predictions + uncertainties on TRAIN/VAL/TEST for UNIQUE
    """
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions = {}
    uncertainties = {}
    all_sample_data = []
    unique_results = []
    
    for sigma in sigma_levels:
        print(f"      σ={sigma:.1f}...", end='')
        
        # Inject noise into training labels
        if sigma == 0.0:
            y_train_noisy = train_labels
        else:
            y_train_noisy = injector.inject(train_labels, sigma)
        
        # Train GP
        gp_dict = train_gauche_gp(
            train_smiles, y_train_noisy,
            kernel='weisfeiler_lehman',
            num_epochs=50
        )
        
        # Predict on ALL sets
        train_results = predict_gauche_gp(gp_dict, train_smiles)
        val_results = predict_gauche_gp(gp_dict, val_smiles)
        test_results = predict_gauche_gp(gp_dict, test_smiles)
        
        # Store test predictions for metrics
        predictions[sigma] = test_results['predictions']
        uncertainties[sigma] = test_results['uncertainties']
        
        # Combine all sets for UNIQUE
        combined_data = []
        
        # Training set
        for i in range(len(train_labels)):
            combined_data.append({
                'sample_id': f'train_{i}',
                'which_set': 'TRAIN',
                'sigma': sigma,
                'y_true': train_labels[i],
                'y_pred': train_results['predictions'][i],
                'uncertainty': train_results['uncertainties'][i],
                'error': abs(train_labels[i] - train_results['predictions'][i])
            })
        
        # Validation set (maps to CALIBRATION for UNIQUE)
        for i in range(len(val_labels)):
            combined_data.append({
                'sample_id': f'val_{i}',
                'which_set': 'CALIBRATION',
                'sigma': sigma,
                'y_true': val_labels[i],
                'y_pred': val_results['predictions'][i],
                'uncertainty': val_results['uncertainties'][i],
                'error': abs(val_labels[i] - val_results['predictions'][i])
            })
        
        # Test set
        for i in range(len(test_labels)):
            combined_data.append({
                'sample_id': f'test_{i}',
                'which_set': 'TEST',
                'sigma': sigma,
                'y_true': test_labels[i],
                'y_pred': test_results['predictions'][i],
                'uncertainty': test_results['uncertainties'][i],
                'error': abs(test_labels[i] - test_results['predictions'][i])
            })
        
        combined_df = pd.DataFrame(combined_data)
        all_sample_data.append(combined_df)
        
        # Run UNIQUE
        unique_res = run_unique_for_graph_gp(combined_df, sigma, results_dir)
        
        if unique_res:
            unique_results.append(unique_res)
        
        print(" done")
    
    return predictions, uncertainties, all_sample_data, unique_results


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='QM9 Graph Models with UNIQUE')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for this repetition')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of QM9 samples to use')
    args = parser.parse_args()
    
    print("="*80)
    print(f"QM9 Graph Models + UNIQUE (seed={args.random_seed})")
    print("="*80)
    
    # Configuration
    strategies = ['legacy']
    sigma_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_dir = Path(__file__).parent.parent / 'results' / 'qm9_graphs' / f'seed_{args.random_seed}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load QM9
    print(f"\nLoading QM9 (n={args.n_samples}, property=homo_lumo_gap)...")
    raw_data = load_qm9(n_samples=args.n_samples, property_idx=4)
    
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
    
    all_results = []
    
    # =========================================================================
    # PHASE 1: CORE ROBUSTNESS - Graph Models
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: CORE ROBUSTNESS - Graph Neural Networks")
    print("="*80)
    
    graph_models = [
        (GCNRegressor, 'GCN'),
        (GATRegressor, 'GAT'),
        (GINRegressor, 'GIN'),
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
    # Graph GP with UNIQUE
    # =========================================================================
    print(f"\n[Graph-GP with UNIQUE]")
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        
        predictions, uncertainties, sample_data_list, unique_results = \
            run_experiment_graph_gp_with_unique(
                train_smiles, train_labels,
                val_smiles, val_labels,
                test_smiles, test_labels,
                strategy, sigma_levels, results_dir
            )
        
        # Save performance metrics
        per_sigma, summary = calculate_noise_metrics(
            test_labels, predictions, metrics=['r2', 'rmse', 'mae']
        )
        per_sigma['model'] = 'Graph-GP'
        per_sigma['rep'] = 'graph'
        per_sigma['strategy'] = strategy
        per_sigma['seed'] = args.random_seed
        all_results.append(per_sigma)
        per_sigma.to_csv(results_dir / f'GraphGP_graph_{strategy}.csv', index=False)
        
        # Save per-sample uncertainty data (combined across noise levels)
        if sample_data_list:
            combined_samples = pd.concat(sample_data_list, ignore_index=True)
            sample_file = results_dir / 'GraphGP_uncertainty_values.csv'
            combined_samples.to_csv(sample_file, index=False)
            print(f"  ✓ Saved per-sample data: {sample_file.name} ({len(combined_samples)} rows)")
        
        # Save UNIQUE rankings
        if unique_results:
            unique_df = pd.DataFrame(unique_results)
            unique_file = results_dir / 'GraphGP_unique_rankings.csv'
            unique_df.to_csv(unique_file, index=False)
            print(f"  ✓ Saved UNIQUE rankings: {unique_file.name}")
            
            print(f"\n  UNIQUE Summary for Graph-GP:")
            for _, row in unique_df.iterrows():
                print(f"    σ={row['sigma']:.1f}: {row['best_uq_metric']} (ρ={row['best_spearman']:.3f})")
    
    # =========================================================================
    # PHASE 2: EXPENSIVE METHODS (Limited Testing)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: EXPENSIVE METHODS (Limited Testing)")
    print("="*80)
    
    if args.random_seed == 42:
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
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_dir / 'all_results.csv', index=False)
    
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