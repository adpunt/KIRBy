import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
hERG Cardiotoxicity + NoiseInject: Full Model-Representation Matrix
====================================================================

Tests noise robustness on hERG classification dataset with FULL coverage matching QM9 experiments.
NO repetitions (n=1) - single run per configuration.

Purpose: Demonstrate cross-dataset consistency on classification task and test new KIRBy representations

Model-Representation Matrix:
- Representations: ECFP4, PDV, SNS, MHG-GNN-pretrained (4 total)
- Models per rep: RF, XGBoost, GP, DNN (baseline), DNN (full-BNN), DNN (last-layer-BNN), DNN (var-BNN) (7 total)  
- Total configurations: 4 reps × 7 models = 28

Noise Strategies: uniform, class_imbalance, binary_asymmetric (3 total)
Noise Levels: flip_prob ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} (11 levels)

Expected results:
- Same model-rep rankings as QM9/ESOL
- Classification-specific noise behavior
- New KIRBy reps competitive with baselines

Note: Uses FLuID dataset with provided train/test split
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.calibration import CalibratedClassifierCV
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
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("WARNING: xgboost not installed, XGBoost experiments will be skipped")
    HAS_XGBOOST = False

# Gaussian Process
try:
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    HAS_GP = True
except ImportError:
    print("WARNING: sklearn.gaussian_process not available, GP experiments will be skipped")
    HAS_GP = False

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.datasets.herg import load_herg
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_sns,
    create_mhg_gnn
)

# NoiseInject imports
from noiseInject import (
    NoiseInjectorClassification,
    calculate_classification_metrics
)


# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================

class DeterministicClassifier(nn.Module):
    """Standard feedforward neural network for binary classification"""
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

def train_neural_classifier(X_train, y_train, X_val, y_val, X_test, 
                            model_type='deterministic', epochs=100, lr=1e-3):
    """
    Train a neural network classifier (deterministic or Bayesian)
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (for early stopping)
        X_test: Test data
        model_type: 'deterministic', 'full-bnn', 'last-layer-bnn', or 'var-bnn'
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        predictions: Test predictions (0/1)
        probabilities: Test prediction probabilities
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Initialize model
    model = DeterministicClassifier(X_train.shape[1]).to(device)
    
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
    
    # Calculate class weights for imbalanced data
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
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
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
        
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
        probabilities_list = []
        for _ in range(30):
            with torch.no_grad():
                logits = model(X_test_t).cpu().numpy()
                probs = torch.sigmoid(torch.FloatTensor(logits)).numpy()
                probabilities_list.append(probs)
        
        probabilities = np.mean(probabilities_list, axis=0)
        predictions = (probabilities > 0.5).astype(int)
    else:
        with torch.no_grad():
            logits = model(X_test_t).cpu().numpy()
            probabilities = torch.sigmoid(torch.FloatTensor(logits)).numpy()
            predictions = (probabilities > 0.5).astype(int)
    
    return predictions, probabilities


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment_tree_model(X_train, y_train, X_test, y_test,
                              model_fn, strategy, flip_prob_levels):
    """
    Run noise robustness experiment for tree-based classifier
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_fn: Function that returns initialized model
        strategy: Noise strategy name
        flip_prob_levels: List of flip probability values to test
    
    Returns:
        predictions: Dict mapping flip_prob -> predictions
        probabilities: Dict mapping flip_prob -> probabilities
    """
    injector = NoiseInjectorClassification(strategy=strategy, random_state=42)
    predictions = {}
    probabilities = {}
    
    for flip_prob in flip_prob_levels:
        # Inject noise
        if flip_prob == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, flip_prob)
        
        # Train model
        model = model_fn()
        model.fit(X_train, y_noisy)
        
        # Get predictions and probabilities
        predictions[flip_prob] = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities[flip_prob] = model.predict_proba(X_test)[:, 1]
        else:
            probabilities[flip_prob] = predictions[flip_prob].astype(float)
    
    return predictions, probabilities


def run_experiment_neural(X_train, y_train, X_val, y_val, X_test, y_test,
                          model_type, strategy, flip_prob_levels):
    """Run noise robustness experiment for neural classifier"""
    injector = NoiseInjectorClassification(strategy=strategy, random_state=42)
    predictions = {}
    probabilities = {}
    
    for flip_prob in flip_prob_levels:
        # Inject noise
        if flip_prob == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, flip_prob)
        
        # Train
        preds, probs = train_neural_classifier(
            X_train, y_noisy, X_val, y_val, X_test,
            model_type=model_type, epochs=100
        )
        
        predictions[flip_prob] = preds
        probabilities[flip_prob] = probs
    
    return predictions, probabilities


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("="*80)
    print("hERG + NoiseInject: Full Model-Representation Matrix")
    print("="*80)

    # Configuration
    strategies = ['uniform', 'class_imbalance', 'binary_asymmetric']
    flip_prob_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_dir = Path('results/herg')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load hERG
    print("\nLoading hERG dataset...")
    print("Loading FLuID training set...")
    train_data = load_herg(source='fluid', use_test=False)
    print("Loading FLuID test set...")
    test_data = load_herg(source='fluid', use_test=True)

    train_smiles = train_data['smiles']
    train_labels = np.array(train_data['labels'])
    test_smiles = test_data['smiles']
    test_labels = np.array(test_data['labels'])
    
    # Create validation split for neural models
    n_val = len(train_smiles) // 5
    val_smiles = train_smiles[:n_val]
    val_labels = train_labels[:n_val]
    train_smiles_fit = train_smiles[n_val:]
    train_labels_fit = train_labels[n_val:]
    
    print(f"Split sizes: Train={len(train_smiles_fit)}, Val={len(val_smiles)}, Test={len(test_smiles)}")
    print(f"Class distribution - Train: {np.bincount(train_labels_fit)}, Test: {np.bincount(test_labels)}")
    
    # Storage for all results
    all_results = []
    all_calibration_data = []
    
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
         lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'), None),
        ('RF', 'PDV', pdv_train, None, pdv_test,
         lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'), None),
        ('RF', 'SNS', sns_train, None, sns_test,
         lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'), None),
        ('RF', 'MHG-GNN-pretrained', mhggnn_train, None, mhggnn_test,
         lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'), None),
    ]
    
    # XGBoost experiments
    if HAS_XGBOOST:
        experiments.extend([
            ('XGBoost', 'ECFP4', ecfp4_train, None, ecfp4_test,
             lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'), None),
            ('XGBoost', 'PDV', pdv_train, None, pdv_test,
             lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'), None),
        ])
    
    # GP experiments
    if HAS_GP:
        experiments.extend([
            ('GP', 'ECFP4', ecfp4_train, None, ecfp4_test,
             lambda: GaussianProcessClassifier(kernel=ConstantKernel(1.0) * RBF(1.0), random_state=42, n_jobs=-1), None),
            ('GP', 'PDV', pdv_train, None, pdv_test,
             lambda: GaussianProcessClassifier(kernel=ConstantKernel(1.0) * RBF(1.0), random_state=42, n_jobs=-1), None),
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
                # Tree-based model
                predictions, probabilities = run_experiment_tree_model(
                    X_train, train_labels_fit, X_test, test_labels,
                    model_fn, strategy, flip_prob_levels
                )
            else:
                # Neural network
                predictions, probabilities = run_experiment_neural(
                    X_train, train_labels_fit, X_val, val_labels, X_test, test_labels,
                    model_type, strategy, flip_prob_levels
                )
            
            per_flip, summary, per_class = calculate_classification_metrics(
                test_labels, predictions, probabilities
            )
            per_flip['model'] = model_name
            per_flip['rep'] = rep_name
            per_flip['strategy'] = strategy
            all_results.append(per_flip)
            
            filename = f"{model_name.replace('-', '')}_{rep_name.replace('-', '')}_{strategy}.csv"
            per_flip.to_csv(results_dir / filename, index=False)
            
            # Save calibration data for uniform strategy
            if strategy == 'uniform' and model_name in ['RF', 'DNN', 'Full-BNN']:
                calib_data = []
                for flip_prob in flip_prob_levels:
                    for i in range(len(test_labels)):
                        calib_data.append({
                            'flip_prob': flip_prob,
                            'sample_idx': i,
                            'y_true': test_labels[i],
                            'y_pred': predictions[flip_prob][i],
                            'probability': probabilities[flip_prob][i]
                        })
                all_calibration_data.append((model_name, rep_name, pd.DataFrame(calib_data)))
    
    # =========================================================================
    # CALIBRATION ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("PROBABILITY CALIBRATION ANALYSIS (uniform strategy only)")
    print("="*80)
    
    # Save all calibration data
    for model_name, rep_name, calib_df in all_calibration_data:
        filename = f'{model_name.replace("-", "")}_{rep_name.replace("-", "")}_calibration_values.csv'
        calib_df.to_csv(results_dir / filename, index=False)
        print(f"Saved {model_name} + {rep_name} calibration data")
        
        # Calculate Brier score per flip_prob
        print(f"\n{model_name} + {rep_name} - Brier Score:")
        for flip_prob in flip_prob_levels:
            subset = calib_df[calib_df['flip_prob'] == flip_prob]
            if len(subset) > 0:
                brier = np.mean((subset['probability'] - subset['y_true'])**2)
                print(f"  flip_prob={flip_prob:.1f}: Brier = {brier:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_dir / 'all_results.csv', index=False)
    
    # Find AUC column name
    auc_col = None
    for col in combined_df.columns:
        if 'auc' in col.lower() and 'pr' not in col.lower():
            auc_col = col
            break
    
    if auc_col is None:
        print("\nWARNING: Could not find AUC column.")
        return
    
    print(f"\nUsing AUC column: {auc_col}")
    
    # Calculate NSI and retention using flip_prob_max = 0.6
    summary_table = []
    for (model, rep, strategy), group in combined_df.groupby(['model', 'rep', 'strategy']):
        baseline_rows = group[group['flip_prob'] == 0.0]
        high_noise_rows = group[group['flip_prob'] == 0.6]
        
        if len(baseline_rows) > 0 and len(high_noise_rows) > 0:
            baseline_acc = baseline_rows['accuracy'].values[0]
            high_noise_acc = high_noise_rows['accuracy'].values[0]
            baseline_auc = baseline_rows[auc_col].values[0]
            high_noise_auc = high_noise_rows[auc_col].values[0]
            
            nsi_acc = (baseline_acc - high_noise_acc) / 0.6
            retention_acc = (high_noise_acc / baseline_acc) * 100 if baseline_acc > 0 else 0
            nsi_auc = (baseline_auc - high_noise_auc) / 0.6
            retention_auc = (high_noise_auc / baseline_auc) * 100 if baseline_auc > 0 else 0
            
            summary_table.append({
                'model': model,
                'rep': rep,
                'strategy': strategy,
                'baseline_accuracy': baseline_acc,
                'accuracy_at_0.6': high_noise_acc,
                'NSI_accuracy': nsi_acc,
                'retention_%_accuracy': retention_acc,
                'baseline_auc': baseline_auc,
                'auc_at_0.6': high_noise_auc,
                'NSI_auc': nsi_auc,
                'retention_%_auc': retention_auc
            })
    
    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(results_dir / 'summary.csv', index=False)
    
    print("\nTop 10 most robust by accuracy (lowest NSI):")
    top10_acc = summary_df.nsmallest(10, 'NSI_accuracy')[
        ['model', 'rep', 'strategy', 'NSI_accuracy', 'retention_%_accuracy']
    ]
    print(top10_acc.to_string(index=False))
    
    print("\nTop 10 most robust by AUC (lowest NSI):")
    top10_auc = summary_df.nsmallest(10, 'NSI_auc')[
        ['model', 'rep', 'strategy', 'NSI_auc', 'retention_%_auc']
    ]
    print(top10_auc.to_string(index=False))
    
    print("\nResults by strategy (mean across models/reps):")
    strategy_summary = summary_df.groupby('strategy')[
        ['NSI_accuracy', 'retention_%_accuracy', 'NSI_auc', 'retention_%_auc']
    ].mean()
    print(strategy_summary.to_string())
    
    print("\nResults by representation (mean across models/strategies):")
    rep_summary = summary_df.groupby('rep')[
        ['NSI_accuracy', 'retention_%_accuracy', 'NSI_auc', 'retention_%_auc']
    ].mean()
    print(rep_summary.to_string())
    
    print("\n" + "="*80)
    print("COMPLETE - Results saved to results/herg/")
    print("="*80)

if __name__ == '__main__':
    main()