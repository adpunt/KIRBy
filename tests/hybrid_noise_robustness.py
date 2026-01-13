#!/usr/bin/env python3
"""
QM9 Hybrid Representation Noise Robustness Experiment
======================================================

Comprehensive evaluation of hybrid molecular representations under label noise.

PHASE 1: Hybrid Optimization
- Base representations: ECFP4, PDV, MHG-GNN
- Budgets: 25, 50, 100, 200, 300, 500
- Allocation strategies: greedy, performance_weighted
- All 2-way and 3-way combinations
- Select top 5 hybrids per allocation strategy (10 total)

PHASE 2: Noise Robustness Evaluation
- 13 representations: 10 hybrids + 3 base reps
- Noise levels: σ ∈ {0, 0.15, 0.3, 0.45, 0.6, 0.75}
- Noise strategies: legacy, outlier, quantile, hetero, threshold, valprop
- Models: RF, XGBoost, NGBoost, DNN, MLP
- Metrics: R², MAE, RMSE, retention rate, NSI

Dataset: QM9 HOMO-LUMO gap, 10k samples, scaffold split
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import argparse
import json
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# KIRBy imports
from kirby.datasets.qm9 import load_qm9, get_qm9_splits
from kirby.representations.molecular import create_ecfp4, create_pdv, create_mhg_gnn
from kirby.hybrid import create_hybrid, apply_feature_selection
from noiseInject import NoiseInjectorRegression

# Optional imports
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    print("WARNING: xgboost not installed, XGBoost experiments will be skipped")
    HAS_XGBOOST = False

try:
    from ngboost import NGBRegressor
    HAS_NGBOOST = True
except ImportError:
    print("WARNING: ngboost not installed, NGBoost experiments will be skipped")
    HAS_NGBOOST = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    print("WARNING: torch not installed, DNN experiments will be skipped")
    HAS_TORCH = False


# =============================================================================
# CONFIGURATION
# =============================================================================

N_SAMPLES = 5000
RANDOM_SEED = 42

# Hybrid optimization
BUDGETS = [25, 50, 100, 200, 300, 500]
ALLOCATION_STRATEGIES = ['greedy', 'performance_weighted']
TOP_K_PER_STRATEGY = 5

# Noise experiment
NOISE_LEVELS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
NOISE_STRATEGIES = ['legacy', 'outlier', 'quantile', 'hetero', 'threshold', 'valprop']

# Models to test
MODEL_NAMES = ['RF', 'XGBoost', 'NGBoost', 'DNN', 'MLP']


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HybridConfig:
    """Configuration for a hybrid representation"""
    name: str
    combo: Tuple[str, ...]  # ('ecfp4', 'pdv') or ('ecfp4', 'pdv', 'mhggnn')
    budget: int
    allocation: str
    val_r2: float = 0.0
    feature_info: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.name, self.combo, self.budget, self.allocation))


@dataclass
class NoiseResult:
    """Result from a single noise experiment"""
    representation: str
    rep_type: str  # 'base' or 'hybrid'
    model: str
    noise_strategy: str
    sigma: float
    
    r2: float
    mae: float
    rmse: float
    
    n_train: int
    n_test: int
    time_seconds: float
    
    def to_dict(self):
        return asdict(self)


# =============================================================================
# NEURAL NETWORK MODEL
# =============================================================================

if HAS_TORCH:
    class DNNRegressor(nn.Module):
        """Simple feedforward neural network for regression"""
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
    
    def train_dnn(X_train, y_train, X_val, y_val, X_test, epochs=100, lr=1e-3):
        """Train DNN and return test predictions"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val_scaled).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        
        model = DNNRegressor(X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        if best_state is not None:
            model.load_state_dict(best_state)
        
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_t).cpu().numpy()
        
        return predictions


# =============================================================================
# MODEL FACTORY
# =============================================================================

def get_model(model_name: str):
    """Return initialized model by name"""
    if model_name == 'RF':
        return RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    elif model_name == 'XGBoost':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        return XGBRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
    elif model_name == 'NGBoost':
        if not HAS_NGBOOST:
            raise ImportError("NGBoost not installed")
        return NGBRegressor(n_estimators=100, random_state=RANDOM_SEED, verbose=False)
    elif model_name == 'MLP':
        return MLPRegressor(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_SEED,
            verbose=False
        )
    elif model_name == 'DNN':
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed")
        return 'DNN'  # Special case - handled separately
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_predict(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray) -> np.ndarray:
    """Train model and return test predictions"""
    
    # Scale data for non-tree models
    if model_name in ['MLP', 'NGBoost']:
        scaler = StandardScaler()
        X_train_use = scaler.fit_transform(X_train)
        X_test_use = scaler.transform(X_test)
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    if model_name == 'DNN':
        return train_dnn(X_train, y_train, X_val, y_val, X_test)
    
    model = get_model(model_name)
    model.fit(X_train_use, y_train)
    return model.predict(X_test_use)


# =============================================================================
# PHASE 1: HYBRID OPTIMIZATION
# =============================================================================

def generate_hybrid_configs() -> List[Tuple[Tuple[str, ...], int, str]]:
    """Generate all hybrid configurations to test"""
    base_reps = ['ecfp4', 'pdv', 'mhggnn']
    configs = []
    
    # 2-way combinations
    for combo in combinations(base_reps, 2):
        for budget in BUDGETS:
            for allocation in ALLOCATION_STRATEGIES:
                configs.append((combo, budget, allocation))
    
    # 3-way combination
    combo_3way = tuple(base_reps)
    for budget in BUDGETS:
        for allocation in ALLOCATION_STRATEGIES:
            configs.append((combo_3way, budget, allocation))
    
    return configs


def evaluate_hybrid_config(
    combo: Tuple[str, ...],
    budget: int,
    allocation: str,
    base_reps_train: Dict[str, np.ndarray],
    base_reps_val: Dict[str, np.ndarray],
    train_labels: np.ndarray,
    val_labels: np.ndarray
) -> Tuple[float, Dict]:
    """Evaluate a single hybrid configuration and return val R² and feature_info"""
    
    # Select only the reps in this combo
    reps_train = {k: base_reps_train[k] for k in combo}
    reps_val = {k: base_reps_val[k] for k in combo}
    
    # Create hybrid
    hybrid_train, feature_info = create_hybrid(
        reps_train, train_labels,
        allocation_method=allocation,
        budget=budget,
        validation_split=0.2,
        apply_filters=False
    )
    
    # Apply same selection to validation
    hybrid_val = apply_feature_selection(reps_val, feature_info)
    
    # Quick RF evaluation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(hybrid_train)
    X_val_scaled = scaler.transform(hybrid_val)
    
    model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train_scaled, train_labels)
    val_pred = model.predict(X_val_scaled)
    
    val_r2 = r2_score(val_labels, val_pred)
    
    return val_r2, feature_info


def run_hybrid_optimization(
    base_reps_train: Dict[str, np.ndarray],
    base_reps_val: Dict[str, np.ndarray],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    results_dir: Path
) -> Dict[str, List[HybridConfig]]:
    """
    Phase 1: Test all hybrid configurations and select top K per allocation strategy.
    
    Returns:
        Dict mapping allocation strategy -> list of top K HybridConfig objects
    """
    print("\n" + "="*80)
    print("PHASE 1: HYBRID OPTIMIZATION")
    print("="*80)
    
    configs = generate_hybrid_configs()
    print(f"Testing {len(configs)} hybrid configurations...")
    
    all_results = []
    
    for i, (combo, budget, allocation) in enumerate(configs):
        combo_str = '+'.join(combo)
        name = f"{combo_str}_b{budget}_{allocation}"
        
        print(f"  [{i+1}/{len(configs)}] {name}...", end=' ', flush=True)
        
        try:
            start = time.time()
            val_r2, feature_info = evaluate_hybrid_config(
                combo, budget, allocation,
                base_reps_train, base_reps_val,
                train_labels, val_labels
            )
            elapsed = time.time() - start
            
            config = HybridConfig(
                name=name,
                combo=combo,
                budget=budget,
                allocation=allocation,
                val_r2=val_r2,
                feature_info=feature_info
            )
            all_results.append(config)
            
            print(f"R²={val_r2:.4f} ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"FAILED: {e}")
            continue
    
    # Save all hybrid results
    hybrid_df = pd.DataFrame([
        {'name': c.name, 'combo': '+'.join(c.combo), 'budget': c.budget, 
         'allocation': c.allocation, 'val_r2': c.val_r2}
        for c in all_results
    ])
    hybrid_df.to_csv(results_dir / 'phase1_all_hybrids.csv', index=False)
    
    # Select top K per allocation strategy
    top_hybrids = {}
    for allocation in ALLOCATION_STRATEGIES:
        strategy_results = [c for c in all_results if c.allocation == allocation]
        strategy_results.sort(key=lambda x: x.val_r2, reverse=True)
        top_hybrids[allocation] = strategy_results[:TOP_K_PER_STRATEGY]
        
        print(f"\nTop {TOP_K_PER_STRATEGY} hybrids ({allocation}):")
        for j, config in enumerate(top_hybrids[allocation], 1):
            print(f"  {j}. {config.name}: R²={config.val_r2:.4f}")
    
    # Save top hybrids
    top_hybrid_data = []
    for allocation, configs in top_hybrids.items():
        for rank, config in enumerate(configs, 1):
            top_hybrid_data.append({
                'allocation': allocation,
                'rank': rank,
                'name': config.name,
                'combo': '+'.join(config.combo),
                'budget': config.budget,
                'val_r2': config.val_r2
            })
    
    top_df = pd.DataFrame(top_hybrid_data)
    top_df.to_csv(results_dir / 'phase1_top_hybrids.csv', index=False)
    
    # Save feature_info for top hybrids as pickle files
    feature_info_dir = results_dir / 'feature_info'
    feature_info_dir.mkdir(exist_ok=True)
    
    for allocation, configs in top_hybrids.items():
        for config in configs:
            pkl_path = feature_info_dir / f"{config.name}_feature_info.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(config.feature_info, f)
    
    print(f"\nSaved feature_info for {sum(len(v) for v in top_hybrids.values())} top hybrids")
    
    return top_hybrids


# =============================================================================
# PHASE 2: NOISE ROBUSTNESS EVALUATION
# =============================================================================

def prepare_representations(
    top_hybrids: Dict[str, List[HybridConfig]],
    base_reps_train: Dict[str, np.ndarray],
    base_reps_val: Dict[str, np.ndarray],
    base_reps_test: Dict[str, np.ndarray],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    results_dir: Optional[Path] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Prepare all representations for noise evaluation.
    
    NOTE: Keeps train/val separate to avoid data leakage during early stopping.
    Val is used for early stopping in neural networks, test for final evaluation.
    
    Returns:
        Dict mapping rep_name -> {'train': X_train, 'val': X_val, 'test': X_test, 'type': 'base'/'hybrid'}
    """
    print("\nPreparing representations for noise evaluation...")
    
    representations = {}
    
    # Add base representations (keep train/val separate)
    for rep_name in ['ecfp4', 'pdv', 'mhggnn']:
        representations[rep_name] = {
            'train': base_reps_train[rep_name],
            'val': base_reps_val[rep_name],
            'test': base_reps_test[rep_name],
            'type': 'base'
        }
        print(f"  Base: {rep_name} - train shape {base_reps_train[rep_name].shape}")
    
    # Add top hybrids
    for allocation, configs in top_hybrids.items():
        for config in configs:
            reps_train = {k: base_reps_train[k] for k in config.combo}
            reps_val = {k: base_reps_val[k] for k in config.combo}
            reps_test = {k: base_reps_test[k] for k in config.combo}
            
            # Check if feature_info is stored in config
            if config.feature_info and len(config.feature_info) > 0:
                # Use stored feature_info
                hybrid_train = apply_feature_selection(reps_train, config.feature_info)
                hybrid_val = apply_feature_selection(reps_val, config.feature_info)
                hybrid_test = apply_feature_selection(reps_test, config.feature_info)
            elif results_dir is not None:
                # Try to load from pickle file
                pkl_path = results_dir / 'feature_info' / f"{config.name}_feature_info.pkl"
                if pkl_path.exists():
                    with open(pkl_path, 'rb') as f:
                        feature_info = pickle.load(f)
                    hybrid_train = apply_feature_selection(reps_train, feature_info)
                    hybrid_val = apply_feature_selection(reps_val, feature_info)
                    hybrid_test = apply_feature_selection(reps_test, feature_info)
                else:
                    # Recreate hybrid on training data only
                    hybrid_train, feature_info = create_hybrid(
                        reps_train, train_labels,
                        allocation_method=config.allocation,
                        budget=config.budget,
                        validation_split=0.2,
                        apply_filters=False
                    )
                    hybrid_val = apply_feature_selection(reps_val, feature_info)
                    hybrid_test = apply_feature_selection(reps_test, feature_info)
            else:
                # Recreate hybrid on training data only
                hybrid_train, feature_info = create_hybrid(
                    reps_train, train_labels,
                    allocation_method=config.allocation,
                    budget=config.budget,
                    validation_split=0.2,
                    apply_filters=False
                )
                hybrid_val = apply_feature_selection(reps_val, feature_info)
                hybrid_test = apply_feature_selection(reps_test, feature_info)
            
            representations[config.name] = {
                'train': hybrid_train,
                'val': hybrid_val,
                'test': hybrid_test,
                'type': 'hybrid'
            }
            print(f"  Hybrid: {config.name} - train shape {hybrid_train.shape}")
    
    return representations


def run_single_noise_experiment(
    rep_name: str,
    rep_type: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train_clean: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    noise_strategy: str,
    sigma: float
) -> Optional[NoiseResult]:
    """Run a single noise experiment"""
    
    # Inject noise
    if sigma == 0.0:
        y_train_noisy = y_train_clean.copy()
    else:
        injector = NoiseInjectorRegression(strategy=noise_strategy, random_state=RANDOM_SEED)
        y_train_noisy = injector.inject(y_train_clean, sigma)
    
    try:
        start = time.time()
        predictions = train_and_predict(model_name, X_train, y_train_noisy, X_val, y_val, X_test)
        elapsed = time.time() - start
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        return NoiseResult(
            representation=rep_name,
            rep_type=rep_type,
            model=model_name,
            noise_strategy=noise_strategy,
            sigma=sigma,
            r2=r2,
            mae=mae,
            rmse=rmse,
            n_train=len(y_train_clean),
            n_test=len(y_test),
            time_seconds=elapsed
        )
    except Exception as e:
        print(f"      ERROR: {e}")
        return None


def run_noise_experiments(
    representations: Dict[str, Dict[str, np.ndarray]],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    results_dir: Path
) -> pd.DataFrame:
    """
    Phase 2: Run all noise robustness experiments.
    """
    print("\n" + "="*80)
    print("PHASE 2: NOISE ROBUSTNESS EVALUATION")
    print("="*80)
    
    # Calculate total experiments
    n_reps = len(representations)
    n_models = len([m for m in MODEL_NAMES if _model_available(m)])
    n_strategies = len(NOISE_STRATEGIES)
    n_sigmas = len(NOISE_LEVELS)
    total = n_reps * n_models * n_strategies * n_sigmas
    
    print(f"Total experiments: {n_reps} reps × {n_models} models × {n_strategies} strategies × {n_sigmas} σ = {total}")
    
    all_results = []
    exp_count = 0
    
    for rep_name, rep_data in representations.items():
        X_train = rep_data['train']
        X_val = rep_data['val']
        X_test = rep_data['test']
        rep_type = rep_data['type']
        
        print(f"\n[{rep_name}] ({rep_type})")
        
        for model_name in MODEL_NAMES:
            if not _model_available(model_name):
                continue
            
            print(f"  Model: {model_name}")
            
            for noise_strategy in NOISE_STRATEGIES:
                print(f"    Strategy: {noise_strategy}", end=' ', flush=True)
                
                strategy_results = []
                for sigma in NOISE_LEVELS:
                    exp_count += 1
                    
                    result = run_single_noise_experiment(
                        rep_name, rep_type,
                        X_train, X_val, X_test,
                        train_labels, val_labels, test_labels,
                        model_name, noise_strategy, sigma
                    )
                    
                    if result is not None:
                        all_results.append(result.to_dict())
                        strategy_results.append(result)
                
                # Print summary for this strategy
                if strategy_results:
                    r2_baseline = strategy_results[0].r2
                    r2_final = strategy_results[-1].r2
                    print(f"R²: {r2_baseline:.3f} → {r2_final:.3f}")
                else:
                    print("FAILED")
                
                # Periodic checkpoint
                if exp_count % 100 == 0:
                    checkpoint_df = pd.DataFrame(all_results)
                    checkpoint_df.to_csv(results_dir / 'phase2_checkpoint.csv', index=False)
    
    # Final save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / 'phase2_all_results.csv', index=False)
    
    print(f"\n✓ Completed {len(all_results)} experiments")
    
    return results_df


def _model_available(model_name: str) -> bool:
    """Check if model is available"""
    if model_name == 'XGBoost':
        return HAS_XGBOOST
    elif model_name == 'NGBoost':
        return HAS_NGBOOST
    elif model_name == 'DNN':
        return HAS_TORCH
    return True


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_noise_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate NSI and retention rate for each representation/model/strategy combination.
    
    NSI = (R²_clean - R²_noisy) / σ_max
    Retention = R²_σmax / R²_clean × 100
    AUC = Area under the R² vs σ curve (higher is better)
    """
    print("\n" + "="*80)
    print("CALCULATING NOISE METRICS")
    print("="*80)
    
    sigma_max = max(NOISE_LEVELS)
    
    summary_rows = []
    
    for (rep, model, strategy), group in results_df.groupby(['representation', 'model', 'noise_strategy']):
        baseline_row = group[group['sigma'] == 0.0]
        maxnoise_row = group[group['sigma'] == sigma_max]
        
        if len(baseline_row) == 0 or len(maxnoise_row) == 0:
            continue
        
        r2_baseline = baseline_row['r2'].values[0]
        r2_maxnoise = maxnoise_row['r2'].values[0]
        
        mae_baseline = baseline_row['mae'].values[0]
        mae_maxnoise = maxnoise_row['mae'].values[0]
        
        # NSI: Noise Sensitivity Index
        nsi = (r2_baseline - r2_maxnoise) / sigma_max if sigma_max > 0 else 0
        
        # Retention: % of performance retained at max noise
        retention = (r2_maxnoise / r2_baseline * 100) if r2_baseline > 0 else 0
        
        # MAE increase
        mae_increase = (mae_maxnoise - mae_baseline) / mae_baseline * 100 if mae_baseline > 0 else 0
        
        # AUC: Area under the R² vs σ curve (trapezoidal integration)
        sorted_group = group.sort_values('sigma')
        sigmas = sorted_group['sigma'].values
        r2s = sorted_group['r2'].values
        auc = np.trapz(r2s, sigmas)  # Higher is better (more robust)
        
        # Normalized AUC (as fraction of perfect AUC = baseline * sigma_max)
        perfect_auc = r2_baseline * sigma_max
        auc_normalized = (auc / perfect_auc * 100) if perfect_auc > 0 else 0
        
        # Get rep_type
        rep_type = group['rep_type'].values[0]
        
        # Get R² at each noise level for the full curve
        r2_at_levels = {}
        for sigma in NOISE_LEVELS:
            sigma_row = group[group['sigma'] == sigma]
            if len(sigma_row) > 0:
                r2_at_levels[f'r2_at_{sigma}'] = sigma_row['r2'].values[0]
        
        row_data = {
            'representation': rep,
            'rep_type': rep_type,
            'model': model,
            'noise_strategy': strategy,
            'r2_baseline': r2_baseline,
            'r2_maxnoise': r2_maxnoise,
            'mae_baseline': mae_baseline,
            'mae_maxnoise': mae_maxnoise,
            'nsi': nsi,
            'retention_pct': retention,
            'mae_increase_pct': mae_increase,
            'auc': auc,
            'auc_normalized_pct': auc_normalized
        }
        row_data.update(r2_at_levels)
        
        summary_rows.append(row_data)
    
    summary_df = pd.DataFrame(summary_rows)
    
    return summary_df


def print_summary(summary_df: pd.DataFrame, results_dir: Path):
    """Print and save summary statistics"""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall best by retention
    print("\n=== TOP 10 BY RETENTION (averaged across strategies) ===")
    avg_retention = summary_df.groupby(['representation', 'model'])['retention_pct'].mean().reset_index()
    avg_retention = avg_retention.sort_values('retention_pct', ascending=False).head(10)
    print(avg_retention.to_string(index=False))
    
    # Best by NSI (lowest is better)
    print("\n=== TOP 10 BY NSI (lowest, averaged across strategies) ===")
    avg_nsi = summary_df.groupby(['representation', 'model'])['nsi'].mean().reset_index()
    avg_nsi = avg_nsi.sort_values('nsi', ascending=True).head(10)
    print(avg_nsi.to_string(index=False))
    
    # Best by AUC (highest is better)
    print("\n=== TOP 10 BY AUC (highest, averaged across strategies) ===")
    avg_auc = summary_df.groupby(['representation', 'model'])['auc_normalized_pct'].mean().reset_index()
    avg_auc = avg_auc.sort_values('auc_normalized_pct', ascending=False).head(10)
    print(avg_auc.to_string(index=False))
    
    # By representation type
    print("\n=== COMPARISON: BASE vs HYBRID (mean across all) ===")
    type_comparison = summary_df.groupby('rep_type')[['nsi', 'retention_pct', 'auc_normalized_pct', 'r2_baseline']].mean()
    print(type_comparison.to_string())
    
    # By noise strategy
    print("\n=== BY NOISE STRATEGY (mean across all models/reps) ===")
    strategy_stats = summary_df.groupby('noise_strategy')[['nsi', 'retention_pct', 'auc_normalized_pct']].mean()
    strategy_stats = strategy_stats.sort_values('retention_pct', ascending=False)
    print(strategy_stats.to_string())
    
    # By model
    print("\n=== BY MODEL (mean across all reps/strategies) ===")
    model_stats = summary_df.groupby('model')[['nsi', 'retention_pct', 'auc_normalized_pct', 'r2_baseline']].mean()
    model_stats = model_stats.sort_values('retention_pct', ascending=False)
    print(model_stats.to_string())
    
    # By representation
    print("\n=== BY REPRESENTATION (mean across all models/strategies) ===")
    rep_stats = summary_df.groupby('representation')[['nsi', 'retention_pct', 'auc_normalized_pct', 'r2_baseline']].mean()
    rep_stats = rep_stats.sort_values('retention_pct', ascending=False)
    print(rep_stats.to_string())
    
    # Create pivot table of R² degradation curves
    print("\n=== R² DEGRADATION CURVES (by representation, averaged across models/strategies) ===")
    r2_cols = [col for col in summary_df.columns if col.startswith('r2_at_')]
    if r2_cols:
        curve_data = summary_df.groupby('representation')[r2_cols].mean()
        print(curve_data.to_string())
        curve_data.to_csv(results_dir / 'r2_degradation_curves.csv')
    
    # Save summary
    summary_df.to_csv(results_dir / 'summary_metrics.csv', index=False)
    
    # Save aggregated views
    avg_retention.to_csv(results_dir / 'summary_by_retention.csv', index=False)
    avg_nsi.to_csv(results_dir / 'summary_by_nsi.csv', index=False)
    avg_auc.to_csv(results_dir / 'summary_by_auc.csv', index=False)
    
    # Save per-strategy breakdown
    for strategy in NOISE_STRATEGIES:
        strategy_df = summary_df[summary_df['noise_strategy'] == strategy]
        strategy_summary = strategy_df.groupby(['representation', 'model'])[['nsi', 'retention_pct', 'r2_baseline']].mean()
        strategy_summary = strategy_summary.reset_index().sort_values('retention_pct', ascending=False)
        strategy_summary.to_csv(results_dir / f'summary_strategy_{strategy}.csv', index=False)


# =============================================================================
# MAIN
# =============================================================================

def main(skip_phase1: bool = False, phase1_results: str = None):
    """
    Main entry point.
    
    Args:
        skip_phase1: If True, skip hybrid optimization and load from file
        phase1_results: Path to existing phase1 results directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results_qm9_hybrid_noise_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("QM9 HYBRID REPRESENTATION NOISE ROBUSTNESS EXPERIMENT")
    print("="*80)
    print(f"Dataset: QM9 HOMO-LUMO gap, {N_SAMPLES} samples, scaffold split")
    print(f"Base representations: ECFP4, PDV, MHG-GNN")
    print(f"Budgets: {BUDGETS}")
    print(f"Allocation strategies: {ALLOCATION_STRATEGIES}")
    print(f"Noise levels: {NOISE_LEVELS}")
    print(f"Noise strategies: {NOISE_STRATEGIES}")
    print(f"Models: {[m for m in MODEL_NAMES if _model_available(m)]}")
    print(f"Results: {results_dir}")
    print("="*80)
    
    # Save configuration
    config = {
        'n_samples': N_SAMPLES,
        'random_seed': RANDOM_SEED,
        'budgets': BUDGETS,
        'allocation_strategies': ALLOCATION_STRATEGIES,
        'top_k_per_strategy': TOP_K_PER_STRATEGY,
        'noise_levels': NOISE_LEVELS,
        'noise_strategies': NOISE_STRATEGIES,
        'models': [m for m in MODEL_NAMES if _model_available(m)],
        'timestamp': timestamp
    }
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load QM9 data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    print("Loading QM9 dataset...")
    raw_data = load_qm9(n_samples=N_SAMPLES, property_idx=4, random_seed=RANDOM_SEED)
    splits = get_qm9_splits(raw_data, splitter='scaffold', random_seed=RANDOM_SEED)
    
    train_smiles = splits['train']['smiles']
    train_labels = np.array(splits['train']['labels'])
    val_smiles = splits['val']['smiles']
    val_labels = np.array(splits['val']['labels'])
    test_smiles = splits['test']['smiles']
    test_labels = np.array(splits['test']['labels'])
    
    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")
    print(f"Property: {raw_data['property_name']}")
    
    # Create base representations
    print("\nCreating base representations...")
    t0 = time.time()
    
    print("  ECFP4...")
    ecfp4_train = create_ecfp4(train_smiles, n_bits=2048)
    ecfp4_val = create_ecfp4(val_smiles, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    
    print("  PDV...")
    pdv_train = create_pdv(train_smiles)
    pdv_val = create_pdv(val_smiles)
    pdv_test = create_pdv(test_smiles)
    
    print("  MHG-GNN...")
    mhggnn_train = create_mhg_gnn(train_smiles)
    mhggnn_val = create_mhg_gnn(val_smiles)
    mhggnn_test = create_mhg_gnn(test_smiles)
    
    print(f"  Done ({time.time() - t0:.1f}s)")
    print(f"  ECFP4:   {ecfp4_train.shape}")
    print(f"  PDV:     {pdv_train.shape}")
    print(f"  MHG-GNN: {mhggnn_train.shape}")
    
    base_reps_train = {'ecfp4': ecfp4_train, 'pdv': pdv_train, 'mhggnn': mhggnn_train}
    base_reps_val = {'ecfp4': ecfp4_val, 'pdv': pdv_val, 'mhggnn': mhggnn_val}
    base_reps_test = {'ecfp4': ecfp4_test, 'pdv': pdv_test, 'mhggnn': mhggnn_test}
    
    # Phase 1: Hybrid optimization
    if skip_phase1 and phase1_results:
        print(f"\nLoading Phase 1 results from {phase1_results}...")
        phase1_dir = Path(phase1_results)
        
        # Load top hybrids from file and recreate HybridConfig objects
        top_df = pd.read_csv(phase1_dir / 'phase1_top_hybrids.csv')
        top_hybrids = {}
        
        for allocation in ALLOCATION_STRATEGIES:
            alloc_rows = top_df[top_df['allocation'] == allocation]
            top_hybrids[allocation] = []
            
            for _, row in alloc_rows.iterrows():
                # Try to load feature_info from pickle
                pkl_path = phase1_dir / 'feature_info' / f"{row['name']}_feature_info.pkl"
                feature_info = {}
                if pkl_path.exists():
                    with open(pkl_path, 'rb') as f:
                        feature_info = pickle.load(f)
                
                config = HybridConfig(
                    name=row['name'],
                    combo=tuple(row['combo'].split('+')),
                    budget=int(row['budget']),
                    allocation=allocation,
                    val_r2=row['val_r2'],
                    feature_info=feature_info
                )
                top_hybrids[allocation].append(config)
        
        print(f"  Loaded {sum(len(v) for v in top_hybrids.values())} hybrid configurations")
    else:
        top_hybrids = run_hybrid_optimization(
            base_reps_train, base_reps_val,
            train_labels, val_labels,
            results_dir
        )
    
    # Prepare all representations (keeps train/val separate)
    representations = prepare_representations(
        top_hybrids,
        base_reps_train, base_reps_val, base_reps_test,
        train_labels, val_labels,
        results_dir=results_dir
    )
    
    # Phase 2: Noise robustness evaluation
    results_df = run_noise_experiments(
        representations,
        train_labels, val_labels, test_labels,
        results_dir
    )
    
    # Calculate and print summary metrics
    summary_df = calculate_noise_metrics(results_df)
    print_summary(summary_df, results_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Results saved to: {results_dir}")
    print(f"  - phase1_all_hybrids.csv: All hybrid configurations tested")
    print(f"  - phase1_top_hybrids.csv: Top {TOP_K_PER_STRATEGY} hybrids per strategy")
    print(f"  - phase2_all_results.csv: All noise experiment results")
    print(f"  - summary_metrics.csv: NSI and retention metrics")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QM9 Hybrid Noise Robustness Experiment')
    parser.add_argument('--skip-phase1', action='store_true',
                       help='Skip Phase 1 (hybrid optimization) and load from existing results')
    parser.add_argument('--phase1-results', type=str, default=None,
                       help='Path to existing Phase 1 results directory')
    args = parser.parse_args()
    
    main(skip_phase1=args.skip_phase1, phase1_results=args.phase1_results)