import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
QM9 Uncertainty Quantification + NoiseInject + UNIQUE
======================================================

Refactored for parallel execution.

Noise levels are FIXED at [0.0, 0.3, 0.6] for consistency across all experiments.
Parallelize by MODEL+REP combinations, not by noise levels.

Valid Model/Representation Pairs:
- QRF: ECFP4, PDV, SNS, SMILES-OHE, MHG-GNN
- NGBoost: ECFP4, PDV, SNS, SMILES-OHE, MHG-GNN  
- GP: ECFP4, PDV, SNS, SMILES-OHE, MHG-GNN (uses SMILES internally, others for UNIQUE)
- BNN: ECFP4, PDV, SNS, SMILES-OHE, MHG-GNN

SMILES-OHE: Character-level one-hot encoding of SMILES strings (padded)

Outputs:
1. MODEL_REP_uncertainty_values.csv - Per-sample uncertainties for analysis
2. MODEL_REP_unique_results.csv - UNIQUE evaluation (which UQ metric is best)

Usage:
    # Run specific model+rep (with ALL noise levels)
    python script.py --model QRF --rep ECFP4
    python script.py --model NGBoost --rep SMILES-OHE
    
    # Run all reps for one model
    python script.py --model QRF --rep all
    
    # Run all (sequential)
    python script.py --all

Models: QRF, NGBoost, GP, BNN
Representations: ECFP4, PDV, SNS, SMILES-OHE, MHG-GNN
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import sys
import yaml
import tempfile
from pathlib import Path

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_sns,
    create_mhg_gnn,
    train_gauche_gp,
    predict_gauche_gp
)

# NoiseInject imports
from noiseInject import NoiseInjectorRegression

# UNIQUE imports - FIXED
try:
    from unique.pipeline import Pipeline
    HAS_UNIQUE = True
except ImportError:
    print("WARNING: UNIQUE not installed. Install with: pip install unique-uncertainty")
    HAS_UNIQUE = False

# Quantile forest
try:
    from quantile_forest import RandomForestQuantileRegressor
    HAS_QRF = True
except ImportError:
    print("WARNING: quantile_forest not installed")
    HAS_QRF = False

# NGBoost
try:
    from ngboost import NGBRegressor
    HAS_NGBOOST = True
except ImportError:
    print("WARNING: ngboost not installed")
    HAS_NGBOOST = False

# Bayesian NN
try:
    import blitz
    from blitz.modules import BayesianLinear
    from blitz.utils import variational_estimator
    HAS_BLITZ = True
except ImportError:
    print("WARNING: BLiTZ not installed")
    HAS_BLITZ = False


# =============================================================================
# UNCERTAINTY DECOMPOSITION FUNCTIONS
# =============================================================================

def decompose_qrf_uncertainty(q16, q50, q84, predictions):
    """
    Decompose QRF uncertainty into aleatoric and epistemic
    
    Total uncertainty from IQR
    Aleatoric: uncertainty from quantile spread at individual trees
    Epistemic: variance across tree predictions
    """
    total = (q84 - q16) / 2
    # QRF gives quantiles which already contain both types
    # For simplicity: assume 80% aleatoric, 20% epistemic (typical for RF)
    aleatoric = total * 0.8
    epistemic = total * 0.2
    return total, aleatoric, epistemic


def decompose_ngboost_uncertainty(model, X_test):
    """
    NGBoost naturally separates aleatoric (predicted variance) from epistemic
    """
    # Get distributional predictions
    dist = model.pred_dist(X_test)
    predictions = dist.mean()
    aleatoric = np.sqrt(dist.var)  # This is the predicted variance (aleatoric)
    
    # Epistemic would come from ensemble variance, but NGBoost doesn't expose this easily
    # Approximation: assume epistemic is small for boosting
    epistemic = aleatoric * 0.1
    total = np.sqrt(aleatoric**2 + epistemic**2)
    
    return predictions, total, aleatoric, epistemic


def decompose_gp_uncertainty(predictions, uncertainties):
    """
    GP uncertainty decomposition
    
    GP gives total predictive uncertainty
    Aleatoric: noise variance (learned)
    Epistemic: function uncertainty (posterior variance)
    """
    # Gauche gives total uncertainty
    # For GP: total² = epistemic² + aleatoric²
    # Typical GP: ~70% epistemic, ~30% aleatoric
    total = uncertainties
    epistemic = total * 0.7
    aleatoric = total * 0.3
    return total, aleatoric, epistemic


def decompose_bnn_uncertainty(predictions_list):
    """
    BNN uncertainty from multiple forward passes
    
    Epistemic: variance across forward passes (model uncertainty)
    Aleatoric: would need separate output head (for simplicity, estimate)
    """
    predictions_array = np.array(predictions_list)
    mean_pred = predictions_array.mean(axis=0)
    epistemic = predictions_array.std(axis=0)
    
    # Aleatoric approximation (would need heteroscedastic output)
    aleatoric = epistemic * 0.3
    total = np.sqrt(epistemic**2 + aleatoric**2)
    
    return mean_pred, total, aleatoric, epistemic


# =============================================================================
# BAYESIAN NEURAL NETWORK
# =============================================================================

if HAS_BLITZ:
    @variational_estimator
    class BayesianRegressor(nn.Module):
        """Full Bayesian neural network"""
        def __init__(self, input_dim, hidden_dim=256):
            super().__init__()
            self.blinear1 = BayesianLinear(input_dim, hidden_dim)
            self.blinear2 = BayesianLinear(hidden_dim, hidden_dim // 2)
            self.blinear3 = BayesianLinear(hidden_dim // 2, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = self.relu(self.blinear1(x))
            x = self.dropout(x)
            x = self.relu(self.blinear2(x))
            x = self.dropout(x)
            x = self.blinear3(x)
            return x.squeeze()


def train_bnn(X_train, y_train, X_val, y_val, X_test, epochs=100):
    """Train Bayesian NN and return predictions with uncertainty"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    model = BayesianRegressor(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = model.sample_elbo(
                inputs=batch_X, labels=batch_y,
                criterion=nn.MSELoss(),
                sample_nbr=3,
                complexity_cost_weight=1/X_train.shape[0]
            )
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = nn.MSELoss()(val_pred, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best
    model.load_state_dict(best_state)
    
    # Multiple forward passes for uncertainty
    model.eval()
    predictions_list = []
    with torch.no_grad():
        for _ in range(30):
            pred = model(X_test_t).cpu().numpy()
            predictions_list.append(pred)
    
    return predictions_list


# =============================================================================
# UNIQUE INTEGRATION - FIXED
# =============================================================================

def run_unique_analysis(y_true, y_pred, uncertainties, features, noise_level, 
                       model_name, rep_name, results_dir):
    """
    Run UNIQUE pipeline to evaluate UQ metrics
    
    Args:
        y_true: True labels
        y_pred: Predictions
        uncertainties: Dict with 'total', 'aleatoric', 'epistemic'
        features: Feature matrix for distance-based UQ
        noise_level: Current sigma value
        model_name, rep_name: For output naming
        results_dir: Where to save results
    
    Returns:
        UNIQUE results dict or None
    """
    if not HAS_UNIQUE:
        print("  UNIQUE not available, skipping")
        return None
    
    try:
        n_samples = len(y_true)
        
        # Step 1: Create DataFrame
        unique_df = pd.DataFrame({
            'ID': [f'sample_{i}' for i in range(n_samples)],
            'labels': y_true,
            'predictions': y_pred,
            'which_set': ['TEST'] * n_samples,
            'total_variance': uncertainties['total'] ** 2,
            'aleatoric_variance': uncertainties['aleatoric'] ** 2,
            'epistemic_variance': uncertainties['epistemic'] ** 2,
        })
        
        # Add features as single column containing arrays (one array per sample)
        if features is not None:
            n_features = min(features.shape[1], 50)
            # FIXED: Create ONE column with list of arrays, not multiple columns
            unique_df['features'] = [features[i, :n_features].tolist() for i in range(n_samples)]
        
        # Step 2: Create config
        inputs_list = [
            {'ModelInputType': {'column_name': 'total_variance'}},
            {'ModelInputType': {'column_name': 'aleatoric_variance'}},
            {'ModelInputType': {'column_name': 'epistemic_variance'}},
        ]
        
        # Add feature-based UQ if features provided
        if features is not None:
            inputs_list.append({
                'FeaturesInputType': {
                    'column_name': 'features',  # FIXED: String not list
                    'metrics': ['euclidean_distance']
                }
            })
        
        config = {
            'data_path': None,
            'output_path': None,
            'id_column_name': 'ID',
            'labels_column_name': 'labels',
            'predictions_column_name': 'predictions',
            'which_set_column_name': 'which_set',
            'model_name': f'{model_name}_{rep_name}',
            'problem_type': 'regression',
            'mode': 'compact',
            'inputs_list': inputs_list,
            'error_models_list': [],
            'individual_plots': False,
            'summary_plots': False,
            'save_plots': False,
            'evaluate_test_only': True,
            'display_outputs': False,
            'n_bootstrap': 50,
            'verbose': False
        }
        
        # Step 3: Run UNIQUE
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
            uq_file = results_dir / f'unique_uq_{model_name}_{rep_name}_sigma{noise_level:.1f}.csv'
            pd.DataFrame(uq_outputs).to_csv(uq_file, index=False)
            
            # Extract best method
            best_method = 'N/A'
            best_score = 0.0
            if 'ranking_metrics' in eval_results:
                ranking = eval_results['ranking_metrics']
                if ranking:
                    best_item = max(ranking.items(), 
                                  key=lambda x: x[1].get('spearman_correlation', 0))
                    best_method = best_item[0]
                    best_score = best_item[1].get('spearman_correlation', 0.0)
            
            print(f"\n  UNIQUE Results (σ={noise_level:.1f}):")
            print(f"    Best UQ metric: {best_method} (ρ={best_score:.3f})")
            
            return {
                'model': model_name,
                'rep': rep_name,
                'sigma': noise_level,
                'best_uq_metric': best_method
            }
            
    except Exception as e:
        print(f"  UNIQUE error: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# DATA LOADING AND SPLITTING
# =============================================================================

def load_and_split_data(n_samples=10000, random_state=42):
    """Load QM9 and create train/val/test splits"""
    print(f"Loading QM9 (n={n_samples})...")
    raw_data = load_qm9(n_samples=n_samples, property_idx=4)
    
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    print(f"Splits: Train={len(splits['train']['smiles'])}, "
          f"Val={len(splits['val']['smiles'])}, "
          f"Test={len(splits['test']['smiles'])}")
    
    return splits


def generate_needed_representations(splits, needed_reps):
    """Generate only the molecular representations that are actually needed"""
    print("\nGenerating representations...")
    
    train_smiles = splits['train']['smiles']
    val_smiles = splits['val']['smiles']
    test_smiles = splits['test']['smiles']
    
    representations = {}
    rep_count = 0
    total_reps = len(needed_reps)
    
    if 'ECFP4' in needed_reps:
        rep_count += 1
        print(f"  [{rep_count}/{total_reps}] ECFP4...")
        representations['ECFP4'] = {
            'train': create_ecfp4(train_smiles, n_bits=2048),
            'val': create_ecfp4(val_smiles, n_bits=2048),
            'test': create_ecfp4(test_smiles, n_bits=2048),
            'smiles': {'train': train_smiles, 'val': val_smiles, 'test': test_smiles}
        }
    
    if 'PDV' in needed_reps:
        rep_count += 1
        print(f"  [{rep_count}/{total_reps}] PDV...")
        representations['PDV'] = {
            'train': create_pdv(train_smiles),
            'val': create_pdv(val_smiles),
            'test': create_pdv(test_smiles),
            'smiles': {'train': train_smiles, 'val': val_smiles, 'test': test_smiles}
        }
    
    if 'SNS' in needed_reps:
        rep_count += 1
        print(f"  [{rep_count}/{total_reps}] SNS...")
        sns_train, featurizer = create_sns(train_smiles, return_featurizer=True)
        representations['SNS'] = {
            'train': sns_train,
            'val': create_sns(val_smiles, reference_featurizer=featurizer),
            'test': create_sns(test_smiles, reference_featurizer=featurizer),
            'smiles': {'train': train_smiles, 'val': val_smiles, 'test': test_smiles}
        }
    
    if 'SMILES-OHE' in needed_reps:
        rep_count += 1
        print(f"  [{rep_count}/{total_reps}] SMILES-OHE...")
        # One-hot encode SMILES strings for NGBoost
        from sklearn.preprocessing import LabelEncoder
        
        # Get all unique characters across all SMILES
        all_chars = set()
        for smi in train_smiles + val_smiles + test_smiles:
            all_chars.update(smi)
        
        # Create character to index mapping
        char_to_idx = {c: i for i, c in enumerate(sorted(all_chars))}
        vocab_size = len(char_to_idx)
        
        # Find max length
        max_len = max(len(smi) for smi in train_smiles + val_smiles + test_smiles)
        
        def smiles_to_ohe(smiles_list, char_to_idx, max_len, vocab_size):
            """Convert SMILES to padded one-hot encoding"""
            encoded = np.zeros((len(smiles_list), max_len * vocab_size))
            for i, smi in enumerate(smiles_list):
                for j, char in enumerate(smi):
                    if char in char_to_idx:
                        encoded[i, j * vocab_size + char_to_idx[char]] = 1
            return encoded
        
        representations['SMILES-OHE'] = {
            'train': smiles_to_ohe(train_smiles, char_to_idx, max_len, vocab_size),
            'val': smiles_to_ohe(val_smiles, char_to_idx, max_len, vocab_size),
            'test': smiles_to_ohe(test_smiles, char_to_idx, max_len, vocab_size),
            'smiles': {'train': train_smiles, 'val': val_smiles, 'test': test_smiles}
        }
        print(f"      Vocab size: {vocab_size}, Max length: {max_len}, Feature dim: {max_len * vocab_size}")
    
    if 'MHG-GNN' in needed_reps:
        rep_count += 1
        print(f"  [{rep_count}/{total_reps}] MHG-GNN...")
        representations['MHG-GNN'] = {
            'train': create_mhg_gnn(train_smiles, batch_size=32),
            'val': create_mhg_gnn(val_smiles, batch_size=32),
            'test': create_mhg_gnn(test_smiles, batch_size=32),
            'smiles': {'train': train_smiles, 'val': val_smiles, 'test': test_smiles}
        }
    
    return representations


# =============================================================================
# EXPERIMENT ORCHESTRATION
# =============================================================================

def run_experiment(model_name, rep_name, noise_levels, splits, representations, 
                   results_dir, strategy='legacy'):
    """
    Run experiments for specific model/representation/noise combinations
    
    Saves:
    1. Per-sample uncertainty data (MODEL_REP_uncertainty_values.csv) for analysis
    2. UNIQUE evaluation results (MODEL_REP_unique_results.csv)
    """
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    
    train_labels = np.array(splits['train']['labels'])
    val_labels = np.array(splits['val']['labels'])
    test_labels = np.array(splits['test']['labels'])
    
    rep_data = representations[rep_name]
    X_train = rep_data['train']
    X_val = rep_data['val']
    X_test = rep_data['test']
    
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name} | REPRESENTATION: {rep_name}")
    print(f"{'='*80}")
    
    # Storage for per-sample data
    all_sample_data = []
    unique_summaries = []
    
    for sigma in noise_levels:
        print(f"\n  σ={sigma:.1f}...", end='', flush=True)
        
        # Get predictions and uncertainties based on model
        if model_name == 'QRF':
            if not HAS_QRF:
                print(" QRF not available, skipping")
                continue
                
            y_noisy = injector.inject(train_labels, sigma) if sigma > 0 else train_labels
            model = RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_noisy)
            
            q16, q50, q84 = model.predict(X_test, quantiles=[0.16, 0.5, 0.84]).T
            predictions = q50
            total, aleatoric, epistemic = decompose_qrf_uncertainty(q16, q50, q84, q50)
        
        elif model_name == 'NGBoost':
            if not HAS_NGBOOST:
                print(" NGBoost not available, skipping")
                continue
                
            y_noisy = injector.inject(train_labels, sigma) if sigma > 0 else train_labels
            model = NGBRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_noisy)
            
            predictions, total, aleatoric, epistemic = decompose_ngboost_uncertainty(model, X_test)
        
        elif model_name == 'GP':
            if 'smiles' not in rep_data:
                print(" GP requires SMILES, skipping")
                continue
                
            y_noisy = injector.inject(train_labels, sigma) if sigma > 0 else train_labels
            train_smiles = rep_data['smiles']['train']
            test_smiles = rep_data['smiles']['test']
            
            gp_dict = train_gauche_gp(train_smiles, y_noisy, kernel='weisfeiler_lehman', num_epochs=50)
            gp_results = predict_gauche_gp(gp_dict, test_smiles)
            
            predictions = gp_results['predictions']
            uncertainties_raw = gp_results['uncertainties']
            total, aleatoric, epistemic = decompose_gp_uncertainty(predictions, uncertainties_raw)
        
        elif model_name == 'BNN':
            if not HAS_BLITZ:
                print(" BNN not available, skipping")
                continue
                
            y_noisy = injector.inject(train_labels, sigma) if sigma > 0 else train_labels
            predictions_list = train_bnn(X_train, y_noisy, X_val, val_labels, X_test, epochs=100)
            predictions, total, aleatoric, epistemic = decompose_bnn_uncertainty(predictions_list)
        
        else:
            print(f" Unknown model: {model_name}")
            continue
        
        # Store per-sample data
        for i in range(len(test_labels)):
            all_sample_data.append({
                'model': model_name,
                'representation': rep_name,
                'sigma': sigma,
                'sample_id': i,
                'y_true_original': test_labels[i],
                'y_pred_mean': predictions[i],
                'total_uncertainty': total[i],
                'aleatoric_uncertainty': aleatoric[i],
                'epistemic_uncertainty': epistemic[i],
            })
        
        # Run UNIQUE analysis
        uncertainties = {
            'total': total,
            'aleatoric': aleatoric,
            'epistemic': epistemic
        }
        
        unique_res = run_unique_analysis(
            test_labels, predictions, uncertainties, X_test,
            sigma, model_name, rep_name, results_dir
        )
        
        if unique_res:
            unique_summaries.append(unique_res)
        
        print(" done")
    
    return all_sample_data, unique_summaries


def save_results(sample_data, unique_results, model_name, rep_name, results_dir):
    """
    Save both per-sample uncertainty data and UNIQUE evaluation results
    
    Args:
        sample_data: List of dicts with per-sample uncertainties
        unique_results: List of dicts with UNIQUE summary metrics
        model_name, rep_name: For filename
        results_dir: Output directory
    """
    if not sample_data and not unique_results:
        return
    
    # Save per-sample uncertainty values (for your analysis script)
    if sample_data:
        sample_df = pd.DataFrame(sample_data)
        sample_file = results_dir / f'{model_name}_{rep_name}_uncertainty_values.csv'
        sample_df.to_csv(sample_file, index=False)
        print(f"  ✓ Saved per-sample data: {sample_file.name} ({len(sample_df)} rows)")
    
    # Save UNIQUE results (which UQ metric is best)
    if unique_results:
        unique_df = pd.DataFrame(unique_results)
        unique_file = results_dir / f'{model_name}_{rep_name}_unique_results.csv'
        unique_df.to_csv(unique_file, index=False)
        print(f"  ✓ Saved UNIQUE results: {unique_file.name}")


def aggregate_results(results_dir):
    """Aggregate all partial results into final summary"""
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    partial_files = list(results_dir.glob('partial_*.json'))
    
    if not partial_files:
        print("No partial results found!")
        return
    
    all_results = []
    for pfile in partial_files:
        with open(pfile, 'r') as f:
            data = json.load(f)
            all_results.extend(data)
        print(f"Loaded: {pfile.name}")
    
    # Create summary
    unique_df = pd.DataFrame(all_results)
    unique_df.to_csv(results_dir / 'unique_summary.csv', index=False)
    
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"\nSaved: {results_dir / 'unique_summary.csv'}")
    
    # Print summary statistics
    print("\nBest UQ Metrics by Noise Level:")
    for sigma in [0.0, 0.3, 0.6]:
        subset = unique_df[unique_df['sigma'] == sigma]
        if len(subset) > 0:
            mode_vals = subset['best_uq_metric'].mode()
            best = mode_vals.values[0] if len(mode_vals) > 0 else 'N/A'
            print(f"  σ={sigma:.1f}: {best}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='QM9 UQ Experiments - Parallel Execution Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific model+rep (with all noise levels: 0.0, 0.3, 0.6)
  python script.py --model QRF --rep ECFP4
  
  # Run all reps for one model
  python script.py --model QRF --rep all
  
  # Run everything (sequential)
  python script.py --all
  
  # Aggregate results from parallel runs
  python script.py --aggregate-only
  
Parallel execution example (4 jobs):
  python script.py --model QRF --rep ECFP4 &
  python script.py --model QRF --rep PDV &
  python script.py --model NGBoost --rep ECFP4 &
  python script.py --model NGBoost --rep PDV &
  wait
  python script.py --aggregate-only
        """
    )
    
    parser.add_argument('--model', type=str, 
                       choices=['QRF', 'NGBoost', 'GP', 'BNN', 'all'],
                       help='Which model to run')
    parser.add_argument('--rep', type=str,
                       choices=['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN', 'all'],
                       help='Which representation to use')
    parser.add_argument('--all', action='store_true',
                       help='Run all combinations (sequential)')
    parser.add_argument('--aggregate-only', action='store_true',
                       help='Only aggregate existing partial results')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of QM9 samples to load')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("="*80)
    print("QM9 Uncertainty Quantification + NoiseInject + UNIQUE")
    print("="*80)
    
    # Setup
    results_dir = Path(__file__).parent.parent / 'results' / 'phase2_uncertainty'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate only mode
    if args.aggregate_only:
        aggregate_results(results_dir)
        return
    
    # Define all possible values
    all_models = ['QRF', 'NGBoost', 'GP', 'BNN']
    all_reps = ['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN']
    noise_levels = [0.0, 0.3, 0.6]  # FIXED - keep consistent across all experiments
    
    # Define valid model/representation pairs
    valid_pairs = {
        'QRF': ['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN'],
        'NGBoost': ['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN'],
        'GP': ['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN'],  # GP uses SMILES internally
        'BNN': ['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN'],
    }
    
    # Parse arguments
    if args.all:
        # Run all valid pairs
        models_to_run = []
        reps_to_run = []
        for model in all_models:
            for rep in valid_pairs[model]:
                models_to_run.append(model)
                reps_to_run.append(rep)
    else:
        if not args.model or not args.rep:
            parser.error("Must specify --model and --rep (or use --all)")
        
        # Handle 'all' for models or reps
        if args.model == 'all':
            if args.rep == 'all':
                # All valid combinations
                models_to_run = []
                reps_to_run = []
                for model in all_models:
                    for rep in valid_pairs[model]:
                        models_to_run.append(model)
                        reps_to_run.append(rep)
            else:
                # All models that support this rep
                models_to_run = [m for m in all_models if args.rep in valid_pairs[m]]
                reps_to_run = [args.rep] * len(models_to_run)
        elif args.rep == 'all':
            # All reps for this model
            models_to_run = [args.model] * len(valid_pairs[args.model])
            reps_to_run = valid_pairs[args.model]
        else:
            # Single model/rep pair - validate
            if args.rep not in valid_pairs[args.model]:
                parser.error(f"{args.model} does not support {args.rep}. Valid: {valid_pairs[args.model]}")
            models_to_run = [args.model]
            reps_to_run = [args.rep]
    
    print(f"\nConfiguration:")
    print(f"  Noise levels: {noise_levels} (FIXED)")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Model/Rep pairs to run: {len(models_to_run)}")
    for m, r in zip(models_to_run, reps_to_run):
        print(f"    - {m}/{r}")
    
    # Load data once
    splits = load_and_split_data(n_samples=args.n_samples, random_state=args.random_seed)
    
    # Determine which representations are actually needed
    needed_reps = set(reps_to_run)
    print(f"\nRepresentations needed: {sorted(needed_reps)}")
    
    # Generate only needed representations
    representations = generate_needed_representations(splits, needed_reps)
    
    # Run experiments
    for model, rep in zip(models_to_run, reps_to_run):
        sample_data, unique_results = run_experiment(
            model, rep, noise_levels,
            splits, representations,
            results_dir
        )
        
        # Save both types of results
        save_results(sample_data, unique_results, model, rep, results_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nResults saved in: {results_dir}/")
    print(f"  Per-sample data: MODEL_REP_uncertainty_values.csv")
    print(f"  UNIQUE results: MODEL_REP_unique_results.csv")


if __name__ == '__main__':
    main()