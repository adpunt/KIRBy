import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
ESOL + NoiseInject: Full Model-Representation Matrix
=====================================================

Tests noise robustness on ESOL solubility dataset with FULL coverage matching QM9 experiments.
NO repetitions (n=1) - single run per configuration.

Purpose: Demonstrate cross-dataset consistency and test new KIRBy representations

Model-Representation Matrix:
- Representations: ECFP4, PDV, SNS, MHG-GNN-pretrained (4 total)
- Models per rep: RF, QRF, XGBoost, GP, DNN×4 (baseline, full-BNN, last-layer-BNN, var-BNN) (8 total)
- Total configurations: 4 reps × 8 models = 32

Noise Strategies: legacy, outlier, quantile, hetero, threshold, valprop (6 total)
Noise Levels: σ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} (11 levels)

Expected results:
- Same model-rep rankings as QM9
- Same "representation > model architecture" variance decomposition
- New KIRBy reps (MHG-GNN) competitive with baselines
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Bayesian neural network imports
try:
    import bayesian_torch.layers as bnn
    from bayesian_torch.models.dnn_to_bnn import transform_model, transform_layer
    HAS_BAYESIAN_TORCH = True
except ImportError:
    print("WARNING: bayesian-torch not installed, BNN experiments will be skipped")
    HAS_BAYESIAN_TORCH = False

# XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    print("WARNING: xgboost not installed, XGBoost experiments will be skipped")
    HAS_XGBOOST = False

# Gaussian Process
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    HAS_GP = True
except ImportError:
    print("WARNING: sklearn.gaussian_process not available, GP experiments will be skipped")
    HAS_GP = False

# Quantile forest
try:
    from quantile_forest import RandomForestQuantileRegressor
    HAS_QRF = True
except ImportError:
    print("WARNING: quantile_forest not installed, QRF experiments will be skipped")
    HAS_QRF = False

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.datasets.esol import load_esol_combined
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_sns,
    create_mhg_gnn
)

# NoiseInject imports
from noiseInject import (
    NoiseInjectorRegression, 
    calculate_noise_metrics
)


# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================

class DeterministicRegressor(nn.Module):
    """Standard feedforward neural network for regression"""
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
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()


# =============================================================================
# BAYESIAN TRANSFORMATIONS
# =============================================================================

if HAS_BAYESIAN_TORCH:
    def apply_bayesian_transformation(model):
        """
        Converts an existing PyTorch model's Linear layers to Bayesian Linear layers.
        
        Parameters
        ----------
        model : nn.Module
            The PyTorch model to be transformed.
            
        Returns
        -------
        model : nn.Module
            The transformed model with Bayesian layers.
        """
        # Convert Linear -> BayesLinear
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
        """
        Replaces only the final nn.Linear layer in the model with a Bayesian Linear layer.
        Uses torchhk-style transform_layer to apply the conversion.
        
        Parameters
        ----------
        model : nn.Module
            Your PyTorch model with at least one nn.Linear layer.
            
        Returns
        -------
        model : nn.Module
            The modified model with the final nn.Linear replaced by bnn.BayesLinear.
        """
        last_linear_name = None
        last_linear_module = None
        
        # Find the last nn.Linear layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                last_linear_name = name
                last_linear_module = module
                break
        
        if last_linear_module is None:
            raise ValueError("No nn.Linear layer found to replace.")
        
        # Build Bayesian version of the final layer
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
        
        # Helper: assign new module to its place in the model
        def set_nested_attr(obj, attr_path, value):
            attrs = attr_path.split(".")
            for a in attrs[:-1]:
                obj = getattr(obj, a)
            setattr(obj, attrs[-1], value)
        
        # Replace the final linear layer
        set_nested_attr(model, last_linear_name, bayesian_layer)
        return model

    def apply_bayesian_transformation_last_layer_variational(model):
        """
        Converts the last Linear layer of a PyTorch model to a Bayesian Linear layer
        (VBLL - Variational Bayesian Last Layer) while keeping the rest of the model deterministic.
        
        Parameters
        ----------
        model : nn.Module
            The PyTorch model to be transformed.
            
        Returns
        -------
        model : nn.Module
            The transformed model with the last layer replaced by a Bayesian layer.
        """
        last_linear_name = None
        last_linear_module = None
        
        # Identify the last nn.Linear layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                last_linear_name = name
                last_linear_module = module
                break
        
        if last_linear_module is None:
            raise ValueError("No nn.Linear layer found to replace.")
        
        # Transform using torchhk-style util
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
        
        # Helper for recursive attribute setting
        def set_nested_attr(obj, attr_path, value):
            attrs = attr_path.split(".")
            for a in attrs[:-1]:
                obj = getattr(obj, a)
            setattr(obj, attrs[-1], value)
        
        # Replace in the model
        set_nested_attr(model, last_linear_name, bayesian_layer)
        return model


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_neural_model(X_train, y_train, X_val, y_val, X_test, 
                       model_type='deterministic', epochs=100, lr=1e-3):
    """
    Train a neural network model (deterministic or Bayesian)
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (for early stopping)
        X_test: Test data
        model_type: 'deterministic', 'full-bnn', 'last-layer-bnn', or 'var-bnn'
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        predictions: Test predictions
        uncertainties: Test uncertainties (None for deterministic)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Initialize model
    model = DeterministicRegressor(X_train.shape[1]).to(device)
    
    # Apply Bayesian transformation if requested
    if HAS_BAYESIAN_TORCH and model_type != 'deterministic':
        if model_type == 'full-bnn':
            model = apply_bayesian_transformation(model)
        elif model_type == 'last-layer-bnn':
            model = apply_bayesian_transformation_last_layer(model)
        elif model_type == 'var-bnn':
            model = apply_bayesian_transformation_last_layer_variational(model)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    is_bayesian = model_type != 'deterministic'
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
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
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
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
    if is_bayesian:
        # Multiple forward passes for uncertainty
        predictions_list = []
        for _ in range(30):
            with torch.no_grad():
                pred = model(X_test_t).cpu().numpy()
                predictions_list.append(pred)
        
        predictions_array = np.array(predictions_list)
        predictions = predictions_array.mean(axis=0)
        uncertainties = predictions_array.std(axis=0)
        
        return predictions, uncertainties
    else:
        with torch.no_grad():
            predictions = model(X_test_t).cpu().numpy()
            return predictions, None


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment_tree_model(X_train, y_train, X_test, y_test, 
                              model_fn, strategy, sigma_levels):
    """
    Run noise robustness experiment for tree-based model
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_fn: Function that returns initialized model
        strategy: Noise strategy name
        sigma_levels: List of sigma values to test
    
    Returns:
        predictions: Dict mapping sigma -> predictions
        uncertainties: Dict mapping sigma -> uncertainties (or None)
    """
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions = {}
    uncertainties = {}
    
    for sigma in sigma_levels:
        # Inject noise
        if sigma == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, sigma)
        
        # Train model
        model = model_fn()
        model.fit(X_train, y_noisy)
        
        # Get predictions
        if hasattr(model, 'predict') and 'Quantile' in str(type(model)):
            # QRF - get quantiles
            q16, q50, q84 = model.predict(X_test, quantiles=[0.16, 0.5, 0.84]).T
            predictions[sigma] = q50
            uncertainties[sigma] = (q84 - q16) / 2
        elif hasattr(model, 'predict') and 'GaussianProcess' in str(type(model)):
            # GP - get mean and std
            pred_mean, pred_std = model.predict(X_test, return_std=True)
            predictions[sigma] = pred_mean
            uncertainties[sigma] = pred_std
        else:
            # Standard model
            predictions[sigma] = model.predict(X_test)
            uncertainties[sigma] = None
    
    return predictions, uncertainties


def run_experiment_neural(X_train, y_train, X_val, y_val, X_test, y_test,
                         model_type, strategy, sigma_levels):
    """Run noise robustness experiment for neural network"""
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions = {}
    uncertainties = {}
    
    for sigma in sigma_levels:
        # Inject noise
        if sigma == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, sigma)
        
        # Train
        preds, uncs = train_neural_model(
            X_train, y_noisy, X_val, y_val, X_test,
            model_type=model_type, epochs=100
        )
        
        predictions[sigma] = preds
        uncertainties[sigma] = uncs
    
    return predictions, uncertainties


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("="*80)
    print("ESOL + NoiseInject: Full Model-Representation Matrix")
    print("="*80)

    # Configuration
    strategies = ['legacy', 'outlier', 'quantile', 'hetero', 'threshold', 'valprop']
    sigma_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_dir = Path('results/esol')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ESOL
    print("\nLoading ESOL dataset...")
    data = load_esol_combined(splitter='scaffold')
    
    train_smiles = data['train']['smiles']
    train_labels = np.array(data['train']['labels'])
    test_smiles = data['test']['smiles']
    test_labels = np.array(data['test']['labels'])
    
    # Create validation split for neural models
    n_val = len(train_smiles) // 5
    val_smiles = train_smiles[:n_val]
    val_labels = train_labels[:n_val]
    train_smiles_fit = train_smiles[n_val:]
    train_labels_fit = train_labels[n_val:]
    
    print(f"Split sizes: Train={len(train_smiles_fit)}, Val={len(val_smiles)}, Test={len(test_smiles)}")
    
    # Storage for all results
    all_results = []
    all_uncertainties = []
    
    # =========================================================================
    # REPRESENTATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING MOLECULAR REPRESENTATIONS")
    print("="*80)
    
    print("ECFP4...")
    ecfp4_train = create_ecfp4(train_smiles_fit, n_bits=2048)
    ecfp4_val = create_ecfp4(val_smiles, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    
    print("PDV...")
    pdv_train = create_pdv(train_smiles_fit)
    pdv_val = create_pdv(val_smiles)
    pdv_test = create_pdv(test_smiles)
    
    print("SNS...")
    sns_train, sns_featurizer = create_sns(train_smiles_fit, return_featurizer=True)
    sns_val = create_sns(val_smiles, reference_featurizer=sns_featurizer)
    sns_test = create_sns(test_smiles, reference_featurizer=sns_featurizer)
    
    print("MHG-GNN (pretrained)...")
    mhggnn_train = create_mhg_gnn(train_smiles_fit)
    mhggnn_test = create_mhg_gnn(test_smiles)
    
    # =========================================================================
    # EXPERIMENTS
    # =========================================================================
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)
    
    # Define experiment configurations
    experiments = [
        # RF experiments
        ('RF', 'ECFP4', ecfp4_train, None, ecfp4_test,
         lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), None),
        ('RF', 'PDV', pdv_train, None, pdv_test,
         lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), None),
        ('RF', 'SNS', sns_train, None, sns_test,
         lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), None),
        ('RF', 'MHG-GNN-pretrained', mhggnn_train, None, mhggnn_test,
         lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), None),
    ]
    
    # QRF experiments
    if HAS_QRF:
        experiments.extend([
            ('QRF', 'ECFP4', ecfp4_train, None, ecfp4_test,
             lambda: RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1), None),
            ('QRF', 'PDV', pdv_train, None, pdv_test,
             lambda: RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1), None),
        ])
    
    # XGBoost experiments
    if HAS_XGBOOST:
        experiments.extend([
            ('XGBoost', 'ECFP4', ecfp4_train, None, ecfp4_test,
             lambda: XGBRegressor(n_estimators=100, random_state=42), None),
            ('XGBoost', 'PDV', pdv_train, None, pdv_test,
             lambda: XGBRegressor(n_estimators=100, random_state=42), None),
        ])
    
    # GP experiments
    if HAS_GP:
        experiments.extend([
            ('GP', 'ECFP4', ecfp4_train, None, ecfp4_test,
             lambda: GaussianProcessRegressor(
                 kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(noise_level=1.0),
                 alpha=1e-10, random_state=42, n_restarts_optimizer=10
             ), None),
            ('GP', 'PDV', pdv_train, None, pdv_test,
             lambda: GaussianProcessRegressor(
                 kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(noise_level=1.0),
                 alpha=1e-10, random_state=42, n_restarts_optimizer=10
             ), None),
        ])
    
    # Neural network experiments
    for model_type in ['deterministic', 'full-bnn', 'last-layer-bnn', 'var-bnn']:
        if model_type != 'deterministic' and not HAS_BAYESIAN_TORCH:
            continue
        
        model_name = {'deterministic': 'DNN', 'full-bnn': 'Full-BNN', 
                     'last-layer-bnn': 'LastLayer-BNN', 'var-bnn': 'Var-BNN'}[model_type]
        
        experiments.extend([
            (model_name, 'ECFP4', ecfp4_train, ecfp4_val, ecfp4_test, None, model_type),
            (model_name, 'PDV', pdv_train, pdv_val, pdv_test, None, model_type),
            (model_name, 'SNS', sns_train, sns_val, sns_test, None, model_type),
        ])
    
    # Run all experiments
    for idx, (model_name, rep_name, X_train, X_val, X_test, model_fn, model_type) in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] {model_name} + {rep_name}...")
        
        for strategy in strategies:
            print(f"  Strategy: {strategy}")
            
            if model_fn is not None:
                # Tree-based or GP model
                predictions, uncertainties = run_experiment_tree_model(
                    X_train, train_labels_fit, X_test, test_labels,
                    model_fn, strategy, sigma_levels
                )
            else:
                # Neural network
                predictions, uncertainties = run_experiment_neural(
                    X_train, train_labels_fit, X_val, val_labels, X_test, test_labels,
                    model_type, strategy, sigma_levels
                )
            
            per_sigma, summary = calculate_noise_metrics(
                test_labels, predictions, metrics=['r2', 'rmse', 'mae']
            )
            per_sigma['model'] = model_name
            per_sigma['rep'] = rep_name
            per_sigma['strategy'] = strategy
            all_results.append(per_sigma)
            
            filename = f"{model_name.replace('-', '')}_{rep_name.replace('-', '')}_{strategy}.csv"
            per_sigma.to_csv(results_dir / filename, index=False)
            
            # Save uncertainties for legacy strategy
            if strategy == 'legacy' and uncertainties[0.0] is not None:
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
                all_uncertainties.append((model_name, rep_name, pd.DataFrame(unc_data)))
    
    # =========================================================================
    # UNCERTAINTY QUANTIFICATION
    # =========================================================================
    print("\n" + "="*80)
    print("UNCERTAINTY QUANTIFICATION (legacy strategy only)")
    print("="*80)
    
    # Save all uncertainty data
    for model_name, rep_name, unc_df in all_uncertainties:
        filename = f'{model_name.replace("-", "")}_{rep_name.replace("-", "")}_uncertainty_values.csv'
        unc_df.to_csv(results_dir / filename, index=False)
        print(f"Saved {model_name} + {rep_name} uncertainties")
        
        # Calculate correlation between uncertainty and error per sigma
        print(f"\n{model_name} + {rep_name} - Uncertainty-Error Correlation:")
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
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_dir / 'all_results.csv', index=False)
    
    # Calculate NSI and retention using σ_max = 0.6
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
                'retention_%': retention
            })
    
    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(results_dir / 'summary.csv', index=False)
    
    print("\nTop 10 most robust (lowest NSI):")
    top10 = summary_df.nsmallest(10, 'NSI')[['model', 'rep', 'strategy', 'NSI', 'retention_%']]
    print(top10.to_string(index=False))
    
    print("\nResults by strategy (mean across models/reps):")
    strategy_summary = summary_df.groupby('strategy')[['NSI', 'retention_%']].mean()
    print(strategy_summary.to_string())
    
    print("\nResults by representation (mean across models/strategies):")
    rep_summary = summary_df.groupby('rep')[['NSI', 'retention_%']].mean()
    print(rep_summary.to_string())
    
    print("\n" + "="*80)
    print("COMPLETE - Results saved to results/esol/")
    print("="*80)


if __name__ == '__main__':
    main()