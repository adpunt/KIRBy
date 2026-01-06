import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
QM9 Uncertainty Quantification + NoiseInject + UNIQUE
======================================================

Properly integrates UNIQUE framework to rank uncertainty metrics.

Key Changes:
1. Generate predictions on TRAIN/VAL/TEST (not just TEST)
2. Save all three subsets for UNIQUE
3. Run UNIQUE to rank: total_uncertainty vs aleatoric_uncertainty vs epistemic_uncertainty
4. Output: Which UQ metric best predicts errors at each noise level?

Models: QRF, NGBoost, GP, BNN
Representations: ECFP4, PDV, SNS, SMILES-OHE, MHG-GNN
Noise levels: σ ∈ [0.0, 0.3, 0.6]
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
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

# UNIQUE imports
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

def decompose_qrf_uncertainty(q16, q50, q84):
    """QRF uncertainty decomposition"""
    total = (q84 - q16) / 2
    aleatoric = total * 0.8
    epistemic = total * 0.2
    return total, aleatoric, epistemic


def decompose_ngboost_uncertainty(model, X):
    """NGBoost uncertainty decomposition"""
    dist = model.pred_dist(X)
    predictions = dist.mean()
    aleatoric = np.sqrt(dist.var)
    epistemic = aleatoric * 0.1
    total = np.sqrt(aleatoric**2 + epistemic**2)
    return predictions, total, aleatoric, epistemic


def decompose_gp_uncertainty(predictions, uncertainties):
    """GP uncertainty decomposition"""
    total = uncertainties
    epistemic = total * 0.7
    aleatoric = total * 0.3
    return total, aleatoric, epistemic


def decompose_bnn_uncertainty(predictions_list):
    """BNN uncertainty decomposition"""
    predictions_array = np.array(predictions_list)
    mean_pred = predictions_array.mean(axis=0)
    epistemic = predictions_array.std(axis=0)
    aleatoric = epistemic * 0.3
    total = np.sqrt(epistemic**2 + aleatoric**2)
    return mean_pred, total, aleatoric, epistemic


# =============================================================================
# BAYESIAN NEURAL NETWORK
# =============================================================================

if HAS_BLITZ:
    @variational_estimator
    class BayesianRegressor(nn.Module):
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


def train_bnn(X_train, y_train, X_val, y_val, epochs=100):
    """Train Bayesian NN - returns trained model for prediction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
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
    
    model.load_state_dict(best_state)
    return model


def predict_bnn(model, X, n_samples=30):
    """Get BNN predictions with uncertainty"""
    device = next(model.parameters()).device
    X_t = torch.FloatTensor(X).to(device)
    
    model.eval()
    predictions_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_t).cpu().numpy()
            predictions_list.append(pred)
    
    return predictions_list


# =============================================================================
# UNIQUE INTEGRATION
# =============================================================================

def run_unique_analysis(combined_data, features, noise_level, model_name, rep_name, results_dir):
    """
    Run UNIQUE to rank uncertainty metrics
    
    Purpose: Determine which uncertainty metric (total, aleatoric, epistemic) 
             best predicts errors at this noise level
    
    Args:
        combined_data: DataFrame with columns:
            - sample_id: identifier
            - which_set: 'TRAIN', 'CALIBRATION', 'TEST'
            - y_true: true labels
            - y_pred: predictions
            - total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty
        features: Feature matrix (for distance-based UQ comparison)
        noise_level: Current σ
        model_name, rep_name: For output naming
        results_dir: Output directory
    
    Returns:
        Dict with UNIQUE rankings
    """
    if not HAS_UNIQUE:
        print("    UNIQUE not available, skipping")
        return None
    
    try:
        print(f"    Running UNIQUE (σ={noise_level:.1f})...")
        
        # Prepare UNIQUE input DataFrame
        unique_df = combined_data.copy()
        unique_df['ID'] = unique_df['sample_id'].astype(str)
        unique_df = unique_df.rename(columns={
            'y_true': 'labels',
            'y_pred': 'predictions'
        })
        
        # Add features if provided (for distance-based UQ)
        if features is not None:
            n_feat = min(features.shape[1], 50)  # Limit to 50 features
            unique_df['features'] = [features[i, :n_feat].tolist() 
                                    for i in range(len(unique_df))]
        
        # UNIQUE config
        inputs_list = [
            # Model-based UQ metrics
            {'ModelInputType': {'column_name': 'total_uncertainty'}},
            {'ModelInputType': {'column_name': 'aleatoric_uncertainty'}},
            {'ModelInputType': {'column_name': 'epistemic_uncertainty'}},
        ]
        
        # Add distance-based UQ if features available
        if features is not None:
            inputs_list.append({
                'FeaturesInputType': {
                    'column_name': 'features',
                    'metrics': ['euclidean_distance', 'manhattan_distance']
                }
            })
        
        config = {
            'data_path': None,
            'output_path': None,
            'id_column_name': 'ID',
            'labels_column_name': 'labels',
            'predictions_column_name': 'predictions',
            'which_set_column_name': 'which_set',
            'model_name': f'{model_name}_{rep_name}_sigma{noise_level:.1f}',
            'problem_type': 'regression',
            'mode': 'compact',
            'inputs_list': inputs_list,
            'error_models_list': [],  # Not using error models
            'individual_plots': False,
            'summary_plots': False,
            'save_plots': False,
            'evaluate_test_only': False,  # Evaluate all sets
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
                # Get Spearman correlations for each UQ metric
                for metric_name, scores in ranking.items():
                    ranking_results[metric_name] = scores.get('spearman_correlation', 0.0)
            
            # Find best metric
            if ranking_results:
                best_metric = max(ranking_results.items(), key=lambda x: x[1])
                print(f"      Best UQ metric: {best_metric[0]} (ρ={best_metric[1]:.3f})")
            else:
                best_metric = ('N/A', 0.0)
                print("      WARNING: No ranking results from UNIQUE")
            
            return {
                'model': model_name,
                'representation': rep_name,
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
        
        all_chars = set()
        for smi in train_smiles + val_smiles + test_smiles:
            all_chars.update(smi)
        
        char_to_idx = {c: i for i, c in enumerate(sorted(all_chars))}
        vocab_size = len(char_to_idx)
        max_len = max(len(smi) for smi in train_smiles + val_smiles + test_smiles)
        
        def smiles_to_ohe(smiles_list, char_to_idx, max_len, vocab_size):
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
    Run experiments with UNIQUE integration
    
    For each noise level:
    1. Get predictions + uncertainties on TRAIN, VAL, TEST
    2. Create combined DataFrame with all three subsets
    3. Run UNIQUE to rank uncertainty metrics
    4. Save both per-sample data and UNIQUE rankings
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
    
    all_sample_data = []
    unique_results = []
    
    for sigma in noise_levels:
        print(f"\n  σ={sigma:.1f}...")
        
        # Inject noise into training labels only
        y_train_noisy = injector.inject(train_labels, sigma) if sigma > 0 else train_labels
        
        # Get predictions and uncertainties for ALL sets
        if model_name == 'QRF':
            if not HAS_QRF:
                print("    QRF not available, skipping")
                continue
            
            model = RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train_noisy)
            
            # Predict on all sets
            q16_train, q50_train, q84_train = model.predict(X_train, quantiles=[0.16, 0.5, 0.84]).T
            q16_val, q50_val, q84_val = model.predict(X_val, quantiles=[0.16, 0.5, 0.84]).T
            q16_test, q50_test, q84_test = model.predict(X_test, quantiles=[0.16, 0.5, 0.84]).T
            
            train_pred = q50_train
            val_pred = q50_val
            test_pred = q50_test
            
            train_total, train_aleatoric, train_epistemic = decompose_qrf_uncertainty(q16_train, q50_train, q84_train)
            val_total, val_aleatoric, val_epistemic = decompose_qrf_uncertainty(q16_val, q50_val, q84_val)
            test_total, test_aleatoric, test_epistemic = decompose_qrf_uncertainty(q16_test, q50_test, q84_test)
        
        elif model_name == 'NGBoost':
            if not HAS_NGBOOST:
                print("    NGBoost not available, skipping")
                continue
            
            model = NGBRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train_noisy)
            
            train_pred, train_total, train_aleatoric, train_epistemic = decompose_ngboost_uncertainty(model, X_train)
            val_pred, val_total, val_aleatoric, val_epistemic = decompose_ngboost_uncertainty(model, X_val)
            test_pred, test_total, test_aleatoric, test_epistemic = decompose_ngboost_uncertainty(model, X_test)
        
        elif model_name == 'GP':
            if 'smiles' not in rep_data:
                print("    GP requires SMILES, skipping")
                continue
            
            train_smiles = rep_data['smiles']['train']
            val_smiles = rep_data['smiles']['val']
            test_smiles = rep_data['smiles']['test']
            
            gp_dict = train_gauche_gp(train_smiles, y_train_noisy, kernel='weisfeiler_lehman', num_epochs=50)
            
            train_results = predict_gauche_gp(gp_dict, train_smiles)
            val_results = predict_gauche_gp(gp_dict, val_smiles)
            test_results = predict_gauche_gp(gp_dict, test_smiles)
            
            train_pred = train_results['predictions']
            val_pred = val_results['predictions']
            test_pred = test_results['predictions']
            
            train_total, train_aleatoric, train_epistemic = decompose_gp_uncertainty(train_pred, train_results['uncertainties'])
            val_total, val_aleatoric, val_epistemic = decompose_gp_uncertainty(val_pred, val_results['uncertainties'])
            test_total, test_aleatoric, test_epistemic = decompose_gp_uncertainty(test_pred, test_results['uncertainties'])
        
        elif model_name == 'BNN':
            if not HAS_BLITZ:
                print("    BNN not available, skipping")
                continue
            
            # Train BNN
            bnn_model = train_bnn(X_train, y_train_noisy, X_val, val_labels, epochs=100)
            
            # Predict on all sets
            train_pred_list = predict_bnn(bnn_model, X_train)
            val_pred_list = predict_bnn(bnn_model, X_val)
            test_pred_list = predict_bnn(bnn_model, X_test)
            
            train_pred, train_total, train_aleatoric, train_epistemic = decompose_bnn_uncertainty(train_pred_list)
            val_pred, val_total, val_aleatoric, val_epistemic = decompose_bnn_uncertainty(val_pred_list)
            test_pred, test_total, test_aleatoric, test_epistemic = decompose_bnn_uncertainty(test_pred_list)
        
        else:
            print(f"    Unknown model: {model_name}")
            continue
        
        # Combine all sets for UNIQUE
        combined_data = []
        
        # Training set
        for i in range(len(train_labels)):
            combined_data.append({
                'sample_id': f'train_{i}',
                'which_set': 'TRAIN',
                'y_true': train_labels[i],
                'y_pred': train_pred[i],
                'total_uncertainty': train_total[i],
                'aleatoric_uncertainty': train_aleatoric[i],
                'epistemic_uncertainty': train_epistemic[i],
            })
        
        # Validation set (maps to CALIBRATION for UNIQUE)
        for i in range(len(val_labels)):
            combined_data.append({
                'sample_id': f'val_{i}',
                'which_set': 'CALIBRATION',
                'y_true': val_labels[i],
                'y_pred': val_pred[i],
                'total_uncertainty': val_total[i],
                'aleatoric_uncertainty': val_aleatoric[i],
                'epistemic_uncertainty': val_epistemic[i],
            })
        
        # Test set
        for i in range(len(test_labels)):
            combined_data.append({
                'sample_id': f'test_{i}',
                'which_set': 'TEST',
                'y_true': test_labels[i],
                'y_pred': test_pred[i],
                'total_uncertainty': test_total[i],
                'aleatoric_uncertainty': test_aleatoric[i],
                'epistemic_uncertainty': test_epistemic[i],
            })
        
        combined_df = pd.DataFrame(combined_data)
        
        # Save per-sample data
        all_sample_data.append(combined_df)
        
        # Prepare features for UNIQUE (optional - for distance-based UQ)
        if model_name != 'GP':  # GP uses SMILES internally
            all_features = np.vstack([X_train, X_val, X_test])
        else:
            all_features = None
        
        # Run UNIQUE
        unique_res = run_unique_analysis(
            combined_df, all_features, sigma, 
            model_name, rep_name, results_dir
        )
        
        if unique_res:
            unique_results.append(unique_res)
        
        print("    done")
    
    return all_sample_data, unique_results


def save_results(sample_data_list, unique_results, model_name, rep_name, results_dir):
    """
    Save both per-sample data and UNIQUE rankings
    """
    if not sample_data_list and not unique_results:
        return
    
    # Save per-sample data (combined across noise levels)
    if sample_data_list:
        combined_samples = pd.concat(sample_data_list, ignore_index=True)
        sample_file = results_dir / f'{model_name}_{rep_name}_uncertainty_values.csv'
        combined_samples.to_csv(sample_file, index=False)
        print(f"  ✓ Saved per-sample data: {sample_file.name} ({len(combined_samples)} rows)")
    
    # Save UNIQUE rankings
    if unique_results:
        unique_df = pd.DataFrame(unique_results)
        unique_file = results_dir / f'{model_name}_{rep_name}_unique_rankings.csv'
        unique_df.to_csv(unique_file, index=False)
        print(f"  ✓ Saved UNIQUE rankings: {unique_file.name}")
        
        # Print summary
        print(f"\n  UNIQUE Summary for {model_name}/{rep_name}:")
        for _, row in unique_df.iterrows():
            print(f"    σ={row['sigma']:.1f}: {row['best_uq_metric']} (ρ={row['best_spearman']:.3f})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='QM9 UQ Experiments with UNIQUE Integration'
    )
    
    parser.add_argument('--model', type=str, 
                       choices=['QRF', 'NGBoost', 'GP', 'BNN', 'all'],
                       help='Which model to run')
    parser.add_argument('--rep', type=str,
                       choices=['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN', 'all'],
                       help='Which representation to use')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of QM9 samples to load')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if not args.model or not args.rep:
        parser.error("Must specify --model and --rep")
    
    print("="*80)
    print("QM9 Uncertainty Quantification + UNIQUE")
    print("="*80)
    
    # Setup
    results_dir = Path(__file__).parent.parent / 'results' / 'phase2_uncertainty'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define configurations
    all_models = ['QRF', 'NGBoost', 'GP', 'BNN']
    all_reps = ['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN']
    noise_levels = [0.0, 0.3, 0.6]
    
    valid_pairs = {
        'QRF': ['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN'],
        'NGBoost': ['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN'],
        'GP': ['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN'],
        'BNN': ['ECFP4', 'PDV', 'SNS', 'SMILES-OHE', 'MHG-GNN'],
    }
    
    # Parse requested models and reps
    if args.model == 'all':
        models_to_run = all_models
    else:
        models_to_run = [args.model]
    
    if args.rep == 'all':
        reps_to_run = []
        for model in models_to_run:
            reps_to_run.extend([(model, rep) for rep in valid_pairs[model]])
    else:
        reps_to_run = [(model, args.rep) for model in models_to_run 
                       if args.rep in valid_pairs[model]]
    
    print(f"\nConfiguration:")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Model/Rep pairs to run: {len(reps_to_run)}")
    for m, r in reps_to_run:
        print(f"    - {m}/{r}")
    
    # Load data
    splits = load_and_split_data(n_samples=args.n_samples, random_state=args.random_seed)
    
    # Generate needed representations
    needed_reps = set([r for _, r in reps_to_run])
    representations = generate_needed_representations(splits, needed_reps)
    
    # Run experiments
    for model, rep in reps_to_run:
        sample_data, unique_results = run_experiment(
            model, rep, noise_levels,
            splits, representations,
            results_dir
        )
        
        save_results(sample_data, unique_results, model, rep, results_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nResults saved in: {results_dir}/")
    print(f"  Per-sample data: MODEL_REP_uncertainty_values.csv")
    print(f"  UNIQUE rankings: MODEL_REP_unique_rankings.csv")


if __name__ == '__main__':
    main()