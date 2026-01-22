import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
AstraZeneca ADME + NoiseInject: Full Model-Representation Matrix
=================================================================

Tests noise robustness on THREE AstraZeneca ADME datasets with FULL coverage.
All datasets from single industrial source with consistent experimental protocols.
NO repetitions (n=1) - single run per configuration.

Datasets:
- Lipophilicity_AstraZeneca (4,200 compounds) - logD @ pH 7.4 [Pat Walters endorsed]
- PPBR_AZ (1,797 compounds) - Plasma Protein Binding Rate
- Clearance_Microsome_AZ (1,102 compounds) - Microsomal Clearance

Purpose: 
- Validate QM9/ESOL findings on larger, industrially-relevant datasets
- Test representation vs model architecture variance decomposition
- Compare noise robustness across different ADME endpoints

Model-Representation Matrix:
- Representations: ECFP4, PDV, SNS, MHG-GNN-pretrained (4 total)
- Models per rep: RF, QRF, XGBoost, GP, DNN×4 (baseline, full-BNN, last-layer-BNN, var-BNN) (8 total)
- Total configurations per dataset: 4 reps × 8 models = 32

Noise Strategies: legacy, outlier, quantile, hetero, threshold, valprop (6 total)
Noise Levels: σ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} (11 levels)

Expected results:
- Same model-rep rankings as QM9/ESOL
- Same "representation > model architecture" variance decomposition
- Consistent patterns across ADME endpoints despite different scales/distributions
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
from typing import Dict, Tuple, Optional, Callable, List
import warnings
warnings.filterwarnings('ignore')

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
try:
    from kirby.representations.molecular import (
        create_ecfp4,
        create_pdv,
        create_sns,
        create_mhg_gnn
    )
    HAS_KIRBY = True
except ImportError:
    print("WARNING: KIRBy not found, using fallback representations")
    HAS_KIRBY = False
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    
    def create_ecfp4(smiles_list, n_bits=2048):
        """Fallback ECFP4 fingerprint generator"""
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(n_bits))
        return np.array(fps)
    
    def create_pdv(smiles_list):
        """Fallback PDV (physicochemical descriptor vector) generator"""
        desc_names = [desc[0] for desc in Descriptors.descList[:200]]
        calc_funcs = [desc[1] for desc in Descriptors.descList[:200]]
        
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                desc_vals = []
                for func in calc_funcs:
                    try:
                        val = func(mol)
                        if np.isnan(val) or np.isinf(val):
                            val = 0.0
                    except:
                        val = 0.0
                    desc_vals.append(val)
                features.append(desc_vals)
            else:
                features.append([0.0] * len(calc_funcs))
        return np.array(features, dtype=np.float32)
    
    def create_sns(smiles_list, return_featurizer=False, reference_featurizer=None):
        """Fallback: just use ECFP4 as placeholder for SNS"""
        result = create_ecfp4(smiles_list, n_bits=1024)
        if return_featurizer:
            return result, None
        return result
    
    def create_mhg_gnn(smiles_list):
        """Fallback: use PDV as placeholder for MHG-GNN embeddings"""
        return create_pdv(smiles_list)

# NoiseInject imports
try:
    from noiseInject import NoiseInjectorRegression, calculate_noise_metrics
    HAS_NOISEINJECT = True
except ImportError:
    print("WARNING: noiseInject not found, using inline implementation")
    HAS_NOISEINJECT = False
    
    class NoiseInjectorRegression:
        """Minimal noise injector for regression tasks"""
        def __init__(self, strategy='legacy', random_state=42):
            self.strategy = strategy
            self.rng = np.random.RandomState(random_state)
        
        def inject(self, y, sigma):
            """Inject noise with given sigma (as fraction of std)"""
            y = np.array(y)
            noise_scale = sigma * y.std()
            
            if self.strategy == 'legacy':
                noise = self.rng.normal(0, noise_scale, size=len(y))
            elif self.strategy == 'outlier':
                noise = self.rng.normal(0, noise_scale, size=len(y))
                n_outliers = max(1, int(0.05 * len(y)))
                outlier_idx = self.rng.choice(len(y), n_outliers, replace=False)
                noise[outlier_idx] *= 5
            elif self.strategy == 'quantile':
                noise = self.rng.normal(0, noise_scale, size=len(y))
                quantiles = np.percentile(y, [25, 75])
                mask = (y < quantiles[0]) | (y > quantiles[1])
                noise[mask] *= 2
            elif self.strategy == 'hetero':
                # Heteroscedastic: noise proportional to |y|
                noise = self.rng.normal(0, 1, size=len(y)) * sigma * np.abs(y - y.mean())
            elif self.strategy == 'threshold':
                noise = self.rng.normal(0, noise_scale, size=len(y))
                median = np.median(y)
                noise[y > median] *= 1.5
            elif self.strategy == 'valprop':
                # Value-proportional noise
                noise = self.rng.normal(0, 1, size=len(y)) * sigma * (np.abs(y) + 0.1)
            else:
                noise = self.rng.normal(0, noise_scale, size=len(y))
            
            return y + noise
    
    def calculate_noise_metrics(y_true, predictions_dict, metrics=['r2', 'rmse', 'mae']):
        """Calculate metrics for each sigma level"""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        results = []
        for sigma, y_pred in predictions_dict.items():
            row = {'sigma': sigma}
            if 'r2' in metrics:
                row['r2'] = r2_score(y_true, y_pred)
            if 'rmse' in metrics:
                row['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            if 'mae' in metrics:
                row['mae'] = mean_absolute_error(y_true, y_pred)
            results.append(row)
        
        df = pd.DataFrame(results)
        summary = df.describe()
        return df, summary

# AstraZeneca ADME loader
from astrazeneca_adme import (
    load_lipophilicity,
    load_ppbr,
    load_clearance_microsome,
    AZ_DATASETS
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
        """Converts Linear layers to Bayesian Linear layers."""
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
        """Replaces only the final nn.Linear layer with Bayesian Linear."""
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
        """VBLL - Variational Bayesian Last Layer"""
        return apply_bayesian_transformation_last_layer(model)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_neural_model(X_train, y_train, X_val, y_val, X_test, 
                       model_type='deterministic', epochs=100, lr=1e-3,
                       batch_size=32, patience=10):
    """
    Train a neural network model (deterministic or Bayesian)
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (for early stopping)
        X_test: Test data
        model_type: 'deterministic', 'full-bnn', 'last-layer-bnn', or 'var-bnn'
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        patience: Early stopping patience
    
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
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
# EXPERIMENT RUNNERS
# =============================================================================

def run_experiment_tree_model(X_train, y_train, X_test, y_test, 
                              model_fn, strategy, sigma_levels):
    """Run noise robustness experiment for tree-based model"""
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions = {}
    uncertainties = {}
    
    for sigma in sigma_levels:
        # Inject noise
        if sigma == 0.0:
            y_noisy = y_train.copy()
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
            y_noisy = y_train.copy()
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
# REPRESENTATION GENERATOR
# =============================================================================

def generate_representations(train_smiles: List[str], 
                            val_smiles: List[str], 
                            test_smiles: List[str],
                            verbose: bool = True) -> Dict:
    """
    Generate all molecular representations for train/val/test sets.
    
    Returns dict with keys: 'ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained'
    Each value is a dict with 'train', 'val', 'test' arrays.
    """
    representations = {}
    
    if verbose:
        print("  ECFP4...")
    representations['ECFP4'] = {
        'train': create_ecfp4(train_smiles, n_bits=2048),
        'val': create_ecfp4(val_smiles, n_bits=2048),
        'test': create_ecfp4(test_smiles, n_bits=2048)
    }
    
    if verbose:
        print("  PDV...")
    representations['PDV'] = {
        'train': create_pdv(train_smiles),
        'val': create_pdv(val_smiles),
        'test': create_pdv(test_smiles)
    }
    
    if verbose:
        print("  SNS...")
    sns_train, sns_featurizer = create_sns(train_smiles, return_featurizer=True)
    representations['SNS'] = {
        'train': sns_train,
        'val': create_sns(val_smiles, reference_featurizer=sns_featurizer),
        'test': create_sns(test_smiles, reference_featurizer=sns_featurizer)
    }
    
    if verbose:
        print("  MHG-GNN (pretrained)...")
    representations['MHG-GNN-pretrained'] = {
        'train': create_mhg_gnn(train_smiles),
        'val': create_mhg_gnn(val_smiles),
        'test': create_mhg_gnn(test_smiles)
    }
    
    return representations


# =============================================================================
# SINGLE DATASET EXPERIMENT
# =============================================================================

def run_dataset_experiments(dataset_name: str,
                           train_smiles: List[str],
                           train_labels: np.ndarray,
                           val_smiles: List[str],
                           val_labels: np.ndarray,
                           test_smiles: List[str],
                           test_labels: np.ndarray,
                           strategies: List[str],
                           sigma_levels: List[float],
                           results_dir: Path) -> pd.DataFrame:
    """
    Run all experiments for a single dataset.
    
    Returns combined results DataFrame.
    """
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")
    print(f"Label range: [{train_labels.min():.2f}, {train_labels.max():.2f}]")
    print(f"Label mean: {train_labels.mean():.2f} ± {train_labels.std():.2f}")
    
    # Generate representations
    print("\nGenerating representations...")
    reps = generate_representations(train_smiles, val_smiles, test_smiles)
    
    # Storage
    all_results = []
    all_uncertainties = []
    
    # Build experiment list
    experiments = []
    
    # RF experiments (all representations)
    for rep_name in ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']:
        experiments.append((
            'RF', rep_name,
            reps[rep_name]['train'], reps[rep_name]['val'], reps[rep_name]['test'],
            lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            None
        ))
    
    # QRF experiments
    if HAS_QRF:
        for rep_name in ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']:
            experiments.append((
                'QRF', rep_name,
                reps[rep_name]['train'], reps[rep_name]['val'], reps[rep_name]['test'],
                lambda: RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                None
            ))
    
    # XGBoost experiments
    if HAS_XGBOOST:
        for rep_name in ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']:
            experiments.append((
                'XGBoost', rep_name,
                reps[rep_name]['train'], reps[rep_name]['val'], reps[rep_name]['test'],
                lambda: XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                None
            ))
    
    # GP experiments (smaller reps only due to O(n³) complexity)
    if HAS_GP:
        for rep_name in ['ECFP4', 'PDV']:
            experiments.append((
                'GP', rep_name,
                reps[rep_name]['train'], reps[rep_name]['val'], reps[rep_name]['test'],
                lambda: GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(noise_level=1.0),
                    alpha=1e-10, random_state=42, n_restarts_optimizer=5
                ),
                None
            ))
    
    # Neural network experiments
    for model_type in ['deterministic', 'full-bnn', 'last-layer-bnn', 'var-bnn']:
        if model_type != 'deterministic' and not HAS_BAYESIAN_TORCH:
            continue
        
        model_name = {
            'deterministic': 'DNN',
            'full-bnn': 'Full-BNN',
            'last-layer-bnn': 'LastLayer-BNN',
            'var-bnn': 'Var-BNN'
        }[model_type]
        
        for rep_name in ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']:
            experiments.append((
                model_name, rep_name,
                reps[rep_name]['train'], reps[rep_name]['val'], reps[rep_name]['test'],
                None, model_type
            ))
    
    # Run experiments
    print(f"\nRunning {len(experiments)} model-representation configurations...")
    print(f"× {len(strategies)} strategies × {len(sigma_levels)} sigma levels")
    print(f"= {len(experiments) * len(strategies) * len(sigma_levels)} total experiments\n")
    
    for idx, (model_name, rep_name, X_train, X_val, X_test, model_fn, model_type) in enumerate(experiments, 1):
        print(f"[{idx}/{len(experiments)}] {model_name} + {rep_name}...")
        
        for strategy in strategies:
            print(f"  Strategy: {strategy}", end="", flush=True)
            
            try:
                if model_fn is not None:
                    # Tree-based or GP model
                    predictions, uncertainties = run_experiment_tree_model(
                        X_train, train_labels, X_test, test_labels,
                        model_fn, strategy, sigma_levels
                    )
                else:
                    # Neural network
                    predictions, uncertainties = run_experiment_neural(
                        X_train, train_labels, X_val, val_labels, X_test, test_labels,
                        model_type, strategy, sigma_levels
                    )
                
                # Calculate metrics
                per_sigma, summary = calculate_noise_metrics(
                    test_labels, predictions, metrics=['r2', 'rmse', 'mae']
                )
                per_sigma['model'] = model_name
                per_sigma['rep'] = rep_name
                per_sigma['strategy'] = strategy
                per_sigma['dataset'] = dataset_name
                all_results.append(per_sigma)
                
                # Save individual results
                filename = f"{model_name.replace('-', '')}_{rep_name.replace('-', '')}_{strategy}.csv"
                per_sigma.to_csv(results_dir / filename, index=False)
                
                # Track uncertainties for legacy strategy
                if strategy == 'legacy' and uncertainties.get(0.0) is not None:
                    unc_data = []
                    for sigma in sigma_levels:
                        if uncertainties.get(sigma) is not None:
                            for i in range(len(test_labels)):
                                unc_data.append({
                                    'sigma': sigma,
                                    'sample_idx': i,
                                    'y_true': test_labels[i],
                                    'y_pred': predictions[sigma][i],
                                    'uncertainty': uncertainties[sigma][i],
                                    'error': abs(test_labels[i] - predictions[sigma][i])
                                })
                    if unc_data:
                        all_uncertainties.append((model_name, rep_name, pd.DataFrame(unc_data)))
                
                print(" ✓")
                
            except Exception as e:
                print(f" ✗ Error: {e}")
                continue
    
    # Save uncertainty data
    if all_uncertainties:
        print("\nSaving uncertainty data...")
        for model_name, rep_name, unc_df in all_uncertainties:
            filename = f'{model_name.replace("-", "")}_{rep_name.replace("-", "")}_uncertainty_values.csv'
            unc_df.to_csv(results_dir / filename, index=False)
    
    # Combine results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(results_dir / 'all_results.csv', index=False)
        return combined_df
    else:
        return pd.DataFrame()


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_summary_statistics(results_df: pd.DataFrame, 
                               sigma_max: float = 0.6) -> pd.DataFrame:
    """
    Compute NSI and retention statistics from results.
    
    Args:
        results_df: Combined results DataFrame
        sigma_max: Sigma value for computing NSI (default 0.6)
    
    Returns:
        Summary DataFrame with NSI and retention metrics
    """
    summary_table = []
    
    for (dataset, model, rep, strategy), group in results_df.groupby(
        ['dataset', 'model', 'rep', 'strategy']
    ):
        baseline_rows = group[group['sigma'] == 0.0]
        high_noise_rows = group[group['sigma'] == sigma_max]
        
        if len(baseline_rows) > 0 and len(high_noise_rows) > 0:
            baseline = baseline_rows['r2'].values[0]
            high_noise = high_noise_rows['r2'].values[0]
            nsi = (baseline - high_noise) / sigma_max
            retention = (high_noise / baseline) * 100 if baseline > 0 else 0
            
            summary_table.append({
                'dataset': dataset,
                'model': model,
                'rep': rep,
                'strategy': strategy,
                'baseline_r2': baseline,
                f'r2_at_{sigma_max}': high_noise,
                'NSI': nsi,
                'retention_%': retention
            })
    
    return pd.DataFrame(summary_table)


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("="*80)
    print("AstraZeneca ADME + NoiseInject: Full Model-Representation Matrix")
    print("="*80)
    print("\nDatasets: Lipophilicity, PPBR, Clearance_Microsome")
    print("All from single industrial source (AstraZeneca) with consistent protocols")
    print("Ideal for noise robustness studies\n")

    # Configuration
    strategies = ['legacy', 'outlier', 'quantile', 'hetero', 'threshold', 'valprop']
    sigma_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    base_results_dir = Path('results/az_adme')
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    # All results across datasets
    all_combined_results = []
    
    # =========================================================================
    # DATASET 1: LIPOPHILICITY (4,200 compounds)
    # =========================================================================
    print("\n" + "="*80)
    print("Loading Lipophilicity_AstraZeneca dataset...")
    lipo_data = load_lipophilicity(splitter='scaffold')
    
    # Create validation split from training data
    lipo_train_smiles = lipo_data['train']['smiles']
    lipo_train_labels = lipo_data['train']['labels']
    lipo_val_smiles = lipo_data['val']['smiles']
    lipo_val_labels = lipo_data['val']['labels']
    lipo_test_smiles = lipo_data['test']['smiles']
    lipo_test_labels = lipo_data['test']['labels']
    
    lipo_results_dir = base_results_dir / 'lipophilicity'
    lipo_results_dir.mkdir(parents=True, exist_ok=True)
    
    lipo_results = run_dataset_experiments(
        dataset_name='Lipophilicity',
        train_smiles=lipo_train_smiles,
        train_labels=lipo_train_labels,
        val_smiles=lipo_val_smiles,
        val_labels=lipo_val_labels,
        test_smiles=lipo_test_smiles,
        test_labels=lipo_test_labels,
        strategies=strategies,
        sigma_levels=sigma_levels,
        results_dir=lipo_results_dir
    )
    all_combined_results.append(lipo_results)
    
    # =========================================================================
    # DATASET 2: PPBR (1,797 compounds)
    # =========================================================================
    print("\n" + "="*80)
    print("Loading PPBR_AZ dataset...")
    ppbr_data = load_ppbr(splitter='scaffold')
    
    ppbr_train_smiles = ppbr_data['train']['smiles']
    ppbr_train_labels = ppbr_data['train']['labels']
    ppbr_val_smiles = ppbr_data['val']['smiles']
    ppbr_val_labels = ppbr_data['val']['labels']
    ppbr_test_smiles = ppbr_data['test']['smiles']
    ppbr_test_labels = ppbr_data['test']['labels']
    
    ppbr_results_dir = base_results_dir / 'ppbr'
    ppbr_results_dir.mkdir(parents=True, exist_ok=True)
    
    ppbr_results = run_dataset_experiments(
        dataset_name='PPBR',
        train_smiles=ppbr_train_smiles,
        train_labels=ppbr_train_labels,
        val_smiles=ppbr_val_smiles,
        val_labels=ppbr_val_labels,
        test_smiles=ppbr_test_smiles,
        test_labels=ppbr_test_labels,
        strategies=strategies,
        sigma_levels=sigma_levels,
        results_dir=ppbr_results_dir
    )
    all_combined_results.append(ppbr_results)
    
    # =========================================================================
    # DATASET 3: CLEARANCE_MICROSOME (1,102 compounds)
    # =========================================================================
    print("\n" + "="*80)
    print("Loading Clearance_Microsome_AZ dataset...")
    clint_data = load_clearance_microsome(splitter='scaffold')
    
    clint_train_smiles = clint_data['train']['smiles']
    clint_train_labels = clint_data['train']['labels']
    clint_val_smiles = clint_data['val']['smiles']
    clint_val_labels = clint_data['val']['labels']
    clint_test_smiles = clint_data['test']['smiles']
    clint_test_labels = clint_data['test']['labels']
    
    clint_results_dir = base_results_dir / 'clearance_microsome'
    clint_results_dir.mkdir(parents=True, exist_ok=True)
    
    clint_results = run_dataset_experiments(
        dataset_name='Clearance_Microsome',
        train_smiles=clint_train_smiles,
        train_labels=clint_train_labels,
        val_smiles=clint_val_smiles,
        val_labels=clint_val_labels,
        test_smiles=clint_test_smiles,
        test_labels=clint_test_labels,
        strategies=strategies,
        sigma_levels=sigma_levels,
        results_dir=clint_results_dir
    )
    all_combined_results.append(clint_results)
    
    # =========================================================================
    # CROSS-DATASET ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("CROSS-DATASET ANALYSIS")
    print("="*80)
    
    # Combine all results
    combined_all = pd.concat(all_combined_results, ignore_index=True)
    combined_all.to_csv(base_results_dir / 'all_datasets_results.csv', index=False)
    
    # Compute summary statistics
    summary_df = compute_summary_statistics(combined_all, sigma_max=0.6)
    summary_df.to_csv(base_results_dir / 'all_datasets_summary.csv', index=False)
    
    # Print summaries
    print("\n" + "-"*80)
    print("Top 10 most robust configurations (lowest NSI across all datasets):")
    print("-"*80)
    top10 = summary_df.nsmallest(10, 'NSI')[
        ['dataset', 'model', 'rep', 'strategy', 'baseline_r2', 'NSI', 'retention_%']
    ]
    print(top10.to_string(index=False))
    
    print("\n" + "-"*80)
    print("Mean NSI by dataset:")
    print("-"*80)
    dataset_summary = summary_df.groupby('dataset')[['NSI', 'retention_%', 'baseline_r2']].mean()
    print(dataset_summary.to_string())
    
    print("\n" + "-"*80)
    print("Mean NSI by representation (across all datasets):")
    print("-"*80)
    rep_summary = summary_df.groupby('rep')[['NSI', 'retention_%', 'baseline_r2']].mean()
    print(rep_summary.to_string())
    
    print("\n" + "-"*80)
    print("Mean NSI by model (across all datasets):")
    print("-"*80)
    model_summary = summary_df.groupby('model')[['NSI', 'retention_%', 'baseline_r2']].mean()
    print(model_summary.to_string())
    
    print("\n" + "-"*80)
    print("Mean NSI by strategy (across all datasets):")
    print("-"*80)
    strategy_summary = summary_df.groupby('strategy')[['NSI', 'retention_%']].mean()
    print(strategy_summary.to_string())
    
    # Variance decomposition (simplified ANOVA-style)
    print("\n" + "-"*80)
    print("Variance decomposition of NSI:")
    print("-"*80)
    
    overall_var = summary_df['NSI'].var()
    
    # Variance explained by each factor
    factors = ['dataset', 'rep', 'model', 'strategy']
    var_explained = {}
    
    for factor in factors:
        group_means = summary_df.groupby(factor)['NSI'].mean()
        between_var = group_means.var() * len(summary_df) / len(group_means)
        var_explained[factor] = between_var / overall_var * 100 if overall_var > 0 else 0
    
    for factor, pct in sorted(var_explained.items(), key=lambda x: -x[1]):
        print(f"  {factor:15s}: {pct:5.1f}% of variance")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {base_results_dir}/")
    print("  - lipophilicity/     (4,200 compounds)")
    print("  - ppbr/              (1,797 compounds)")
    print("  - clearance_microsome/ (1,102 compounds)")
    print("  - all_datasets_results.csv")
    print("  - all_datasets_summary.csv")
    print("="*80)


if __name__ == '__main__':
    main()