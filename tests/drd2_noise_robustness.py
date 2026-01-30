import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
DRD2 Ki + NoiseInject: Full Model-Representation Matrix
=======================================================

Tests noise robustness on DRD2 Ki dataset with FULL coverage matching QM9 experiments.
NO repetitions (n=1) - single run per configuration.

Purpose: Demonstrate noise robustness on high-quality Ki data (contrast to COX-2 IC50)

Dataset notes:
  - DRD2 Ki from ChEMBL following Landrum's quality filters
  - Ki data is MORE reproducible than IC50 (Landrum 2024: R²=0.88 vs 0.51)
  - DRD2 (Dopamine D2 receptor) is heavily studied with abundant Ki data
  - Clinical relevance: antipsychotics (haloperidol, risperidone), Parkinson's

Model-Representation Matrix:
- Representations: ECFP4, PDV, SNS, MHG-GNN-pretrained (4 total)
- Models per rep: RF, QRF, XGBoost, GP, DNN×4 (baseline, full-BNN, last-layer-BNN, var-BNN) (8 total)
- Total configurations: 4 reps × 8 models = 32

Noise Strategies: legacy, outlier, quantile, hetero, threshold, valprop (6 total)
Noise Levels: σ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} (11 levels)

Expected results:
- Higher baseline performance than COX-2 due to cleaner Ki data
- Same model-rep rankings as other datasets
- Interesting comparison: does cleaner data show different noise robustness patterns?
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
from kirby.datasets.drd2 import load_drd2_chembl, get_drd2_splits
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
        """
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
        """
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
        """
        Converts the last Linear layer to a Bayesian Linear layer (VBLL).
        """
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

def train_neural_model(X_train, y_train, X_val, y_val, X_test, 
                       model_type='deterministic', epochs=100, lr=1e-3):
    """
    Train a neural network model (deterministic or Bayesian)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    model = DeterministicRegressor(X_train.shape[1]).to(device)
    
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
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
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
    if is_bayesian:
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
    """
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions = {}
    uncertainties = {}
    
    for sigma in sigma_levels:
        if sigma == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, sigma)
        
        model = model_fn()
        model.fit(X_train, y_noisy)
        
        if hasattr(model, 'predict') and 'Quantile' in str(type(model)):
            q16, q50, q84 = model.predict(X_test, quantiles=[0.16, 0.5, 0.84]).T
            predictions[sigma] = q50
            uncertainties[sigma] = (q84 - q16) / 2
        elif hasattr(model, 'predict') and 'GaussianProcess' in str(type(model)):
            pred_mean, pred_std = model.predict(X_test, return_std=True)
            predictions[sigma] = pred_mean
            uncertainties[sigma] = pred_std
        else:
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
        if sigma == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, sigma)
        
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
    print("DRD2 Ki + NoiseInject: Full Model-Representation Matrix")
    print("="*80)
    print()
    print("Dataset: DRD2 Ki from ChEMBL (Landrum quality filters)")
    print("Note: Ki data is highly reproducible (Landrum 2024: R²=0.88 between assays)")
    print("      Contrast with COX-2 IC50 to study noise robustness across data quality")
    print()

    # Configuration
    strategies = ['legacy', 'outlier', 'quantile', 'hetero', 'threshold', 'valprop']
    sigma_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_dir = Path('results/drd2')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load DRD2 Ki
    print("Loading DRD2 Ki dataset from ChEMBL...")
    data = load_drd2_chembl()
    splits = get_drd2_splits(data, splitter='scaffold')
    
    train_smiles = splits['train']['smiles']
    train_labels = np.array(splits['train']['labels'])
    test_smiles = splits['test']['smiles']
    test_labels = np.array(splits['test']['labels'])
    
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
    mhggnn_val = create_mhg_gnn(val_smiles)
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
            ('QRF', 'SNS', sns_train, None, sns_test,
             lambda: RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1), None),
            ('QRF', 'MHG-GNN-pretrained', mhggnn_train, None, mhggnn_test,
             lambda: RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1), None),
        ])
    
    # XGBoost experiments
    if HAS_XGBOOST:
        experiments.extend([
            ('XGBoost', 'ECFP4', ecfp4_train, None, ecfp4_test,
             lambda: XGBRegressor(n_estimators=100, random_state=42), None),
            ('XGBoost', 'PDV', pdv_train, None, pdv_test,
             lambda: XGBRegressor(n_estimators=100, random_state=42), None),
            ('XGBoost', 'SNS', sns_train, None, sns_test,
             lambda: XGBRegressor(n_estimators=100, random_state=42), None),
            ('XGBoost', 'MHG-GNN-pretrained', mhggnn_train, None, mhggnn_test,
             lambda: XGBRegressor(n_estimators=100, random_state=42), None),
        ])
    
    # GP experiments (only on smaller representations due to computational cost)
    if HAS_GP:
        experiments.extend([
            ('GP', 'PDV', pdv_train, None, pdv_test,
             lambda: GaussianProcessRegressor(
                 kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(noise_level=1.0),
                 alpha=1e-10, random_state=42, n_restarts_optimizer=5
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
            (model_name, 'MHG-GNN-pretrained', mhggnn_train, mhggnn_val, mhggnn_test, None, model_type),
        ])
    
    # Run all experiments
    for idx, (model_name, rep_name, X_train, X_val, X_test, model_fn, model_type) in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] {model_name} + {rep_name}...")
        
        for strategy in strategies:
            print(f"  Strategy: {strategy}")
            
            if model_fn is not None:
                predictions, uncertainties = run_experiment_tree_model(
                    X_train, train_labels_fit, X_test, test_labels,
                    model_fn, strategy, sigma_levels
                )
            else:
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
    
    for model_name, rep_name, unc_df in all_uncertainties:
        filename = f'{model_name.replace("-", "")}_{rep_name.replace("-", "")}_uncertainty_values.csv'
        unc_df.to_csv(results_dir / filename, index=False)
        print(f"Saved {model_name} + {rep_name} uncertainties")
        
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
    
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_dir / 'all_results.csv', index=False)
    
    # Calculate Noise Degradation Slope (NDS) via linear regression
    # NDS = slope of R² vs sigma (negative = degradation with noise)
    # Also track baseline_r2 for later thresholding (avoid "robust because bad")
    from scipy.stats import linregress
    
    summary_table = []
    for (model, rep, strategy), group in combined_df.groupby(['model', 'rep', 'strategy']):
        group_sorted = group.sort_values('sigma')
        sigmas = group_sorted['sigma'].values
        r2_values = group_sorted['r2'].values
        rmse_values = group_sorted['rmse'].values
        
        baseline_r2 = r2_values[0]
        baseline_rmse = rmse_values[0]
        
        # Linear regression: R² vs sigma
        slope_r2, intercept_r2, r_value, p_value, std_err = linregress(sigmas, r2_values)
        # Linear regression: RMSE vs sigma  
        slope_rmse, _, _, _, _ = linregress(sigmas, rmse_values)
        
        summary_table.append({
            'model': model,
            'rep': rep,
            'strategy': strategy,
            'baseline_r2': baseline_r2,
            'baseline_rmse': baseline_rmse,
            'NDS_r2': slope_r2,  # negative = degrades with noise
            'NDS_rmse': slope_rmse,  # positive = degrades with noise
            'r2_fit': r_value**2,  # how linear is the degradation
        })
    
    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(results_dir / 'summary.csv', index=False)
    
    print("\nNoise Degradation Slope (NDS) Summary:")
    print("  NDS_r2: slope of R² vs σ (more negative = faster degradation)")
    print("  NDS_rmse: slope of RMSE vs σ (more positive = faster degradation)")
    print("  baseline_r2: R² at σ=0 (for thresholding poor performers)")
    
    print("\nTop 10 by NDS_r2 (least negative = most robust):")
    top10 = summary_df.nlargest(10, 'NDS_r2')[['model', 'rep', 'strategy', 'baseline_r2', 'NDS_r2']]
    print(top10.to_string(index=False))
    
    print("\nResults by strategy (mean across models/reps):")
    strategy_summary = summary_df.groupby('strategy')[['baseline_r2', 'NDS_r2', 'NDS_rmse']].mean()
    print(strategy_summary.to_string())
    
    print("\nResults by representation (mean across models/strategies):")
    rep_summary = summary_df.groupby('rep')[['baseline_r2', 'NDS_r2', 'NDS_rmse']].mean()
    print(rep_summary.to_string())
    
    print("\nResults by model (mean across reps/strategies):")
    model_summary = summary_df.groupby('model')[['baseline_r2', 'NDS_r2', 'NDS_rmse']].mean()
    print(model_summary.to_string())
    
    print("\n" + "="*80)
    print("COMPLETE - Results saved to results/drd2/")
    print("="*80)


if __name__ == '__main__':
    main()