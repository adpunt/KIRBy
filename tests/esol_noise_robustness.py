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
- Models per rep: RF, QRF, XGBoost, NGBoost, DNN×4 (baseline, full-BNN, last-layer-BNN, var-BNN), Gauche GP (9 total)
- Total configurations: 4 reps × 9 models = 36

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

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.datasets.esol import load_esol_combined
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_sns,
    create_mhg_gnn,
    train_gauche_gp,
    predict_gauche_gp
)

# NoiseInject imports
from noiseInject import (
    NoiseInjectorRegression, 
    calculate_noise_metrics
)

# Quantile forest
try:
    from quantile_forest import RandomForestQuantileRegressor
    HAS_QRF = True
except ImportError:
    print("WARNING: quantile_forest not installed, QRF experiments will be skipped")
    HAS_QRF = False

# Bayesian neural network
try:
    import blitz
    from blitz.modules import BayesianLinear
    from blitz.utils import variational_estimator
    HAS_BLITZ = True
except ImportError:
    print("WARNING: BLiTZ not installed, BNN experiments will be skipped")
    HAS_BLITZ = False


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


if HAS_BLITZ:
    @variational_estimator
    class BayesianRegressor(nn.Module):
        """Full Bayesian neural network with weight uncertainty"""
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


def train_neural_model(X_train, y_train, X_val, y_val, X_test, 
                       model_class, epochs=100, lr=1e-3, is_bayesian=False):
    """
    Train a neural network model (deterministic or Bayesian)
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (for early stopping)
        X_test: Test data
        model_class: DeterministicRegressor or BayesianRegressor
        epochs: Number of training epochs
        lr: Learning rate
        is_bayesian: Whether this is a Bayesian model
    
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
    model = model_class(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
            
            if is_bayesian:
                # Bayesian forward pass
                loss = model.sample_elbo(
                    inputs=batch_X,
                    labels=batch_y,
                    criterion=nn.MSELoss(),
                    sample_nbr=3,
                    complexity_cost_weight=1/X_train.shape[0]
                )
            else:
                # Standard forward pass
                pred = model(batch_X)
                loss = nn.MSELoss()(pred, batch_y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            if is_bayesian:
                val_pred = model(X_val_t)
            else:
                val_pred = model(X_val_t)
            val_loss = nn.MSELoss()(val_pred, y_val_t).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Test predictions
    model.eval()
    with torch.no_grad():
        if is_bayesian:
            # Multiple forward passes for uncertainty
            predictions_list = []
            for _ in range(30):
                pred = model(X_test_t).cpu().numpy()
                predictions_list.append(pred)
            
            predictions_array = np.array(predictions_list)
            predictions = predictions_array.mean(axis=0)
            uncertainties = predictions_array.std(axis=0)
            
            return predictions, uncertainties
        else:
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
        else:
            # Standard model
            predictions[sigma] = model.predict(X_test)
            uncertainties[sigma] = None
    
    return predictions, uncertainties


def run_experiment_gp(train_smiles, y_train, test_smiles, y_test,
                     strategy, sigma_levels):
    """Run noise robustness experiment for Gaussian Process"""
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions = {}
    uncertainties = {}
    
    for sigma in sigma_levels:
        # Inject noise
        if sigma == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, sigma)
        
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
    
    return predictions, uncertainties


def run_experiment_neural(X_train, y_train, X_val, y_val, X_test, y_test,
                         model_class, strategy, sigma_levels, is_bayesian=False):
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
            model_class, epochs=100, is_bayesian=is_bayesian
        )
        
        predictions[sigma] = preds
        uncertainties[sigma] = uncs
    
    return predictions, uncertainties


# =============================================================================
# MAIN SCRIPT
# =============================================================================


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("="*80)
    print("ESOL + NoiseInject: Strategic Model-Representation Pairs")
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
    # PHASE 1: CORE ROBUSTNESS
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: CORE ROBUSTNESS - Strategic Pairs")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # Pair 1: RF + ECFP4 (baseline from paper)
    # -------------------------------------------------------------------------
    print("\n[1/9] RF + ECFP4...")
    ecfp4_train = create_ecfp4(train_smiles_fit, n_bits=2048)
    ecfp4_val = create_ecfp4(val_smiles, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, _ = run_experiment_tree_model(
            ecfp4_train, train_labels_fit, ecfp4_test, test_labels,
            lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            strategy, sigma_levels
        )
        
        per_sigma, summary = calculate_noise_metrics(
            test_labels, predictions, metrics=['r2', 'rmse', 'mae']
        )
        per_sigma['model'] = 'RF'
        per_sigma['rep'] = 'ECFP4'
        per_sigma['strategy'] = strategy
        all_results.append(per_sigma)
        per_sigma.to_csv(results_dir / f'RF_ECFP4_{strategy}.csv', index=False)
    
    # -------------------------------------------------------------------------
    # Pair 2: QRF + PDV (baseline from paper)
    # -------------------------------------------------------------------------
    if HAS_QRF:
        print("\n[2/9] QRF + PDV...")
        pdv_train = create_pdv(train_smiles_fit)
        pdv_val = create_pdv(val_smiles)
        pdv_test = create_pdv(test_smiles)
        
        for strategy in strategies:
            print(f"  Strategy: {strategy}")
            predictions, uncertainties = run_experiment_tree_model(
                pdv_train, train_labels_fit, pdv_test, test_labels,
                lambda: RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                strategy, sigma_levels
            )
            
            per_sigma, summary = calculate_noise_metrics(
                test_labels, predictions, metrics=['r2', 'rmse', 'mae']
            )
            per_sigma['model'] = 'QRF'
            per_sigma['rep'] = 'PDV'
            per_sigma['strategy'] = strategy
            all_results.append(per_sigma)
            per_sigma.to_csv(results_dir / f'QRF_PDV_{strategy}.csv', index=False)
            
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
                all_uncertainties.append(('QRF', 'PDV', pd.DataFrame(unc_data)))
    else:
        print("\n[2/9] QRF + PDV... SKIPPED (quantile_forest not installed)")
        pdv_train = create_pdv(train_smiles_fit)
        pdv_val = create_pdv(val_smiles)
        pdv_test = create_pdv(test_smiles)
    
    # -------------------------------------------------------------------------
    # Pair 3: GP + PDV (best performer from paper)
    # -------------------------------------------------------------------------
    print("\n[3/9] Gauche GP + PDV...")
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, uncertainties = run_experiment_gp(
            train_smiles_fit, train_labels_fit, test_smiles, test_labels,
            strategy, sigma_levels
        )
        
        per_sigma, summary = calculate_noise_metrics(
            test_labels, predictions, metrics=['r2', 'rmse', 'mae']
        )
        per_sigma['model'] = 'GP'
        per_sigma['rep'] = 'PDV'
        per_sigma['strategy'] = strategy
        all_results.append(per_sigma)
        per_sigma.to_csv(results_dir / f'GP_PDV_{strategy}.csv', index=False)
        
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
            all_uncertainties.append(('GP', 'PDV', pd.DataFrame(unc_data)))
    
    # -------------------------------------------------------------------------
    # Pair 4: RF + MHG-GNN pretrained (new with KIRBy)
    # -------------------------------------------------------------------------
    print("\n[4/7] RF + MHG-GNN (pretrained)...")
    mhggnn_train = create_mhg_gnn(train_smiles_fit, batch_size=32)
    mhggnn_test = create_mhg_gnn(test_smiles, batch_size=32)
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, _ = run_experiment_tree_model(
            mhggnn_train, train_labels_fit, mhggnn_test, test_labels,
            lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            strategy, sigma_levels
        )
        
        per_sigma, summary = calculate_noise_metrics(
            test_labels, predictions, metrics=['r2', 'rmse', 'mae']
        )
        per_sigma['model'] = 'RF'
        per_sigma['rep'] = 'MHG-GNN-pretrained'
        per_sigma['strategy'] = strategy
        all_results.append(per_sigma)
        per_sigma.to_csv(results_dir / f'RF_MHGGNN-pretrained_{strategy}.csv', index=False)
    
    # -------------------------------------------------------------------------
    # Pair 7: DNN + ECFP4 (neural baseline)
    # -------------------------------------------------------------------------
    print("\n[5/7] DNN + ECFP4...")
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, _ = run_experiment_neural(
            ecfp4_train, train_labels_fit, ecfp4_val, val_labels, ecfp4_test, test_labels,
            DeterministicRegressor, strategy, sigma_levels, is_bayesian=False
        )
        
        per_sigma, summary = calculate_noise_metrics(
            test_labels, predictions, metrics=['r2', 'rmse', 'mae']
        )
        per_sigma['model'] = 'DNN'
        per_sigma['rep'] = 'ECFP4'
        per_sigma['strategy'] = strategy
        all_results.append(per_sigma)
        per_sigma.to_csv(results_dir / f'DNN_ECFP4_{strategy}.csv', index=False)
    
    # -------------------------------------------------------------------------
    # Pair 8: DNN + PDV (neural baseline)
    # -------------------------------------------------------------------------
    print("\n[6/7] DNN + PDV...")
    
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        predictions, _ = run_experiment_neural(
            pdv_train, train_labels_fit, pdv_val, val_labels, pdv_test, test_labels,
            DeterministicRegressor, strategy, sigma_levels, is_bayesian=False
        )
        
        per_sigma, summary = calculate_noise_metrics(
            test_labels, predictions, metrics=['r2', 'rmse', 'mae']
        )
        per_sigma['model'] = 'DNN'
        per_sigma['rep'] = 'PDV'
        per_sigma['strategy'] = strategy
        all_results.append(per_sigma)
        per_sigma.to_csv(results_dir / f'DNN_PDV_{strategy}.csv', index=False)
    
    # -------------------------------------------------------------------------
    # Pair 9: Full-BNN + PDV (probabilistic neural)
    # -------------------------------------------------------------------------
    if HAS_BLITZ:
        print("\n[7/7] Full-BNN + PDV...")
        
        for strategy in strategies:
            print(f"  Strategy: {strategy}")
            predictions, uncertainties = run_experiment_neural(
                pdv_train, train_labels_fit, pdv_val, val_labels, pdv_test, test_labels,
                BayesianRegressor, strategy, sigma_levels, is_bayesian=True
            )
            
            per_sigma, summary = calculate_noise_metrics(
                test_labels, predictions, metrics=['r2', 'rmse', 'mae']
            )
            per_sigma['model'] = 'Full-BNN'
            per_sigma['rep'] = 'PDV'
            per_sigma['strategy'] = strategy
            all_results.append(per_sigma)
            per_sigma.to_csv(results_dir / f'FullBNN_PDV_{strategy}.csv', index=False)
            
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
                all_uncertainties.append(('Full-BNN', 'PDV', pd.DataFrame(unc_data)))
    else:
        print("\n[9/9] Full-BNN + PDV... SKIPPED (BLiTZ not installed)")
    
    # =========================================================================
    # PHASE 2: PROBABILISTIC COMPARISON
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: PROBABILISTIC COMPARISON (legacy strategy only)")
    print("="*80)
    print("Comparison results already collected in Phase 1:")
    print("  - RF vs QRF on PDV")
    print("  - DNN vs Full-BNN on PDV")
    
    # =========================================================================
    # PHASE 3: UNCERTAINTY QUANTIFICATION
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: UNCERTAINTY QUANTIFICATION (legacy strategy only)")
    print("="*80)
    
    # Save all uncertainty data
    for model_name, rep_name, unc_df in all_uncertainties:
        unc_df.to_csv(
            results_dir / f'{model_name}_{rep_name}_uncertainty_values.csv',
            index=False
        )
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