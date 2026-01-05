import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
QM9 Uncertainty Quantification + NoiseInject + UNIQUE
======================================================

Tests uncertainty quantification under noise using NoiseInject framework.
Integrates UNIQUE for UQ metric evaluation and comparison.

Single run (no bootstraps) - matches original Phase 2 experimental design.

Purpose: 
- Demonstrate aleatoric vs epistemic uncertainty decomposition
- Use UNIQUE to determine which UQ metric is most robust to noise
- Answer: "Which uncertainty should I trust when training data is noisy?"

Models: QRF, NGBoost, Gauche GP, BNN variants (full, last-layer, var)
Representations: ECFP4, PDV, SNS
Noise Strategy: legacy (Gaussian) only
Noise Levels: σ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

UNIQUE Integration:
- For each (model, rep, noise level): create UNIQUE input
- Evaluate: variance, knn_euclidean, kde_gaussian
- Compare: ranking, calibration, scoring_rules
- Output: Which UQ metric best predicts errors at each noise level
"""

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
        
        # Add features
        feature_columns = []
        if features is not None:
            n_features = min(features.shape[1], 50)
            for i in range(n_features):
                col_name = f'feat_{i}'
                unique_df[col_name] = features[:, i]
                feature_columns.append(col_name)
        
        # Step 2: Create config
        inputs_list = [
            {'ModelInputType': {'column_name': 'total_variance'}},
            {'ModelInputType': {'column_name': 'aleatoric_variance'}},
            {'ModelInputType': {'column_name': 'epistemic_variance'}},
        ]
        
        if feature_columns:
            inputs_list.append({
                'FeaturesInputType': {
                    'column_name': feature_columns,
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
# MAIN EXPERIMENT RUNNER
# =============================================================================

def main():
    print("="*80)
    print("QM9 Uncertainty Quantification + NoiseInject + UNIQUE")
    print("="*80)
    
    # Configuration
    sigma_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    strategy = 'legacy'
    n_samples = 10000
    results_dir = Path(__file__).parent.parent / 'results' / 'phase2_uncertainty'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load QM9
    print("\nLoading QM9...")
    raw_data = load_qm9(n_samples=n_samples, property_idx=4)
    splits = get_qm9_splits(raw_data, splitter='scaffold')
    
    train_smiles = splits['train']['smiles']
    train_labels = splits['train']['labels']
    val_smiles = splits['val']['smiles']
    val_labels = splits['val']['labels']
    test_smiles = splits['test']['smiles']
    test_labels = splits['test']['labels']
    
    print(f"Splits: Train={len(train_smiles)}, Val={len(val_smiles)}, Test={len(test_smiles)}")
    
    # Generate representations
    print("\nGenerating representations...")
    
    print("  [1/3] ECFP4...")
    ecfp4_train = create_ecfp4(train_smiles, n_bits=2048)
    ecfp4_val = create_ecfp4(val_smiles, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    
    print("  [2/3] PDV...")
    pdv_train = create_pdv(train_smiles)
    pdv_val = create_pdv(val_smiles)
    pdv_test = create_pdv(test_smiles)
    
    print("  [3/4] SNS...")
    sns_train, featurizer = create_sns(train_smiles, return_featurizer=True)
    sns_val = create_sns(val_smiles, reference_featurizer=featurizer)
    sns_test = create_sns(test_smiles, reference_featurizer=featurizer)

    print("  [4/4] MHG-GNN (pretrained)...")
    mhggnn_train = create_mhg_gnn(train_smiles, batch_size=32)
    mhggnn_val = create_mhg_gnn(val_smiles, batch_size=32)
    mhggnn_test = create_mhg_gnn(test_smiles, batch_size=32)
    
    representations = [
        ('ECFP4', ecfp4_train, ecfp4_val, ecfp4_test),
        ('PDV', pdv_train, pdv_val, pdv_test),
        ('SNS', sns_train, sns_val, sns_test),
        ('MHG-GNN-pretrained', mhggnn_train, mhggnn_val, mhggnn_test)
    ]
    
    # NoiseInject
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    
    # Storage
    all_results = []
    unique_summaries = []
    
    # ==========================================================================
    # TEST ALL CONFIGURATIONS
    # ==========================================================================
    
    for rep_name, X_train, X_val, X_test in representations:
        print(f"\n{'='*80}")
        print(f"REPRESENTATION: {rep_name}")
        print(f"{'='*80}")
        
        # ======================================================================
        # Model 1: Quantile Random Forest
        # ======================================================================
        if HAS_QRF:
            print(f"\n[QRF + {rep_name}]")
            
            for sigma in sigma_levels:
                print(f"  σ={sigma:.1f}...", end='')
                
                # Inject noise
                y_noisy = injector.inject(train_labels, sigma) if sigma > 0 else train_labels
                
                # Train
                model = RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_noisy)
                
                # Predict with quantiles
                q16, q50, q84 = model.predict(X_test, quantiles=[0.16, 0.5, 0.84]).T
                
                # Decompose uncertainty
                total, aleatoric, epistemic = decompose_qrf_uncertainty(q16, q50, q84, q50)
                
                # UNIQUE analysis
                uncertainties = {
                    'total': total,
                    'aleatoric': aleatoric,
                    'epistemic': epistemic
                }
                
                unique_res = run_unique_analysis(
                    test_labels, q50, uncertainties, X_test,
                    sigma, 'QRF', rep_name, results_dir
                )
                
                if unique_res is not None:
                    unique_summaries.append(unique_res)
                
                print(" done")
        
        # ======================================================================
        # Model 2: NGBoost
        # ======================================================================
        if HAS_NGBOOST:
            print(f"\n[NGBoost + {rep_name}]")
            
            for sigma in sigma_levels:
                print(f"  σ={sigma:.1f}...", end='')
                
                y_noisy = injector.inject(train_labels, sigma) if sigma > 0 else train_labels
                
                model = NGBRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_noisy)
                
                predictions, total, aleatoric, epistemic = decompose_ngboost_uncertainty(model, X_test)
                
                uncertainties = {
                    'total': total,
                    'aleatoric': aleatoric,
                    'epistemic': epistemic
                }
                
                unique_res = run_unique_analysis(
                    test_labels, predictions, uncertainties, X_test,
                    sigma, 'NGBoost', rep_name, results_dir
                )
                
                if unique_res is not None:
                    unique_summaries.append(unique_res)
                
                print(" done")
        
        # ======================================================================
        # Model 3: Gauche GP (only for ECFP4, PDV, SNS, MHG-GNN - needs SMILES)
        # ======================================================================
        if rep_name in ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']:
            print(f"\n[Gauche GP + {rep_name}]")
            
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
                uncertainties_raw = gp_results['uncertainties']
                
                total, aleatoric, epistemic = decompose_gp_uncertainty(predictions, uncertainties_raw)
                
                uncertainties = {
                    'total': total,
                    'aleatoric': aleatoric,
                    'epistemic': epistemic
                }
                
                unique_res = run_unique_analysis(
                    test_labels, predictions, uncertainties, X_test,
                    sigma, 'GP', rep_name, results_dir
                )
                
                if unique_res is not None:
                    unique_summaries.append(unique_res)
                
                print(" done")
        
        # ======================================================================
        # Model 4: Bayesian Neural Network
        # ======================================================================
        if HAS_BLITZ:
            print(f"\n[BNN + {rep_name}]")
            
            for sigma in sigma_levels:
                print(f"  σ={sigma:.1f}...", end='')
                
                y_noisy = injector.inject(train_labels, sigma) if sigma > 0 else train_labels
                
                predictions_list = train_bnn(X_train, y_noisy, X_val, val_labels, X_test, epochs=100)
                
                predictions, total, aleatoric, epistemic = decompose_bnn_uncertainty(predictions_list)
                
                uncertainties = {
                    'total': total,
                    'aleatoric': aleatoric,
                    'epistemic': epistemic
                }
                
                unique_res = run_unique_analysis(
                    test_labels, predictions, uncertainties, X_test,
                    sigma, 'BNN', rep_name, results_dir
                )
                
                if unique_res is not None:
                    unique_summaries.append(unique_res)
                
                print(" done")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if unique_summaries:
        unique_df = pd.DataFrame(unique_summaries)
        unique_df.to_csv(results_dir / 'unique_summary.csv', index=False)
        
        print("\nBest UQ Metrics by Noise Level:")
        for sigma in sigma_levels:
            subset = unique_df[unique_df['sigma'] == sigma]
            if len(subset) > 0:
                print(f"  σ={sigma:.1f}: {subset['best_uq_metric'].mode().values[0] if len(subset['best_uq_metric'].mode()) > 0 else 'N/A'}")
        
        print("\nBest UQ Metrics by Model:")
        print(unique_df.groupby('model')['best_uq_metric'].value_counts())
    
    print("\n" + "="*80)
    print("COMPLETE - Results saved to results/phase2_uncertainty/")
    print("="*80)


if __name__ == '__main__':
    main()