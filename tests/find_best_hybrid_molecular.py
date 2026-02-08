#!/usr/bin/env python3
"""
Find Best Hybrid Molecular Representation

A flexible CLI tool for finding the optimal hybrid molecular representation,
with multi-model evaluation and reusable blueprints.

Key Outputs:
1. BEST OVERALL: Highest performing model+hybrid combination
2. BEST BY DIMENSION: Best config at each dimension tier (50, 100, 200, 500)
3. BEST BY MODEL: Optimal hybrid for each model type (RF, XGBoost, Ridge, MLP, KNN)
4. PERFORMANCE MATRIX: Full Model × Dimension grid
5. BASELINE COMPARISON: Hybrid vs full-dimension individual representations

The output is a minimal blueprint (reps + method + budget + importance) that can
be applied to new data - no need to store feature indices.

Usage:
    # Quick local test
    python find_best_hybrid_molecular.py esol --reps fast --search quick

    # Medium search with HuggingFace models
    python find_best_hybrid_molecular.py herg --reps fast+hf --search medium

    # Full search with all models
    python find_best_hybrid_molecular.py data.csv --smiles-col SMILES --target-col pIC50 \\
        --reps all --search full
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings('ignore')


# =============================================================================
# SCAFFOLD UTILITIES
# =============================================================================

def get_murcko_scaffold(smiles: str) -> str:
    """Get Murcko scaffold for a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return smiles


def assign_scaffold_groups(smiles_list: List[str]) -> Tuple[np.ndarray, int]:
    """
    Assign scaffold group IDs to molecules for GroupKFold.

    Returns:
        tuple: (groups array, number of unique scaffolds)
    """
    scaffolds = [get_murcko_scaffold(smi) for smi in smiles_list]
    unique_scaffolds = list(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    groups = np.array([scaffold_to_id[s] for s in scaffolds])
    return groups, len(unique_scaffolds)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Representation presets
REP_PRESETS = {
    'fast': ['ecfp4', 'maccs', 'pdv'],
    'fast+hf': ['ecfp4', 'maccs', 'pdv', 'chemberta', 'molformer', 'selformer'],
    'all': [
        'ecfp4', 'maccs', 'pdv', 'mol2vec', 'chemberta', 'molformer',
        'selformer', 'chembert', 'molclr', 'graphmvp', 'smited',
        'mhg_gnn', 'grover', 'graph_kernel'
    ]
}

# Models to evaluate
# TODO: Add more models - SVR/SVC, LightGBM, CatBoost, GaussianProcess, ElasticNet, GBM
MODELS = {
    'rf': {
        'regressor': lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'classifier': lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    },
    'xgboost': {
        'regressor': lambda: _get_xgb_regressor(),
        'classifier': lambda: _get_xgb_classifier(),
    },
    'ridge': {
        'regressor': lambda: Ridge(alpha=1.0, random_state=42),
        'classifier': lambda: LogisticRegression(C=1.0, random_state=42, max_iter=1000, n_jobs=-1),
    },
    'mlp': {
        'regressor': lambda: MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
        'classifier': lambda: MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
    },
    'knn': {
        'regressor': lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        'classifier': lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    },
}

# Dimension tiers
DIMENSION_TIERS = [50, 100, 200, 500]

# Allocation methods
ALLOCATION_METHODS = ['greedy', 'performance_weighted', 'fixed', 'mrmr']
ALLOCATION_METHODS_HIGH_COMPUTE = ['greedy_feature']  # Feature-level, slower but more precise

# Importance methods by computation cost
IMPORTANCE_METHODS_FAST = [
    'random_forest',      # MDI/Gini importance
    'treeshap',           # TreeSHAP (handles correlations better)
    'xgboost_gain',       # XGBoost gain per split
    'xgboost_weight',     # XGBoost split count
    'xgboost_cover',      # XGBoost samples covered
    'lightgbm_gain',      # LightGBM gain
    'lightgbm_split',     # LightGBM split count
]

IMPORTANCE_METHODS_MEDIUM = [
    'permutation',        # Permutation importance (model-agnostic)
    'integrated_gradients',  # Path integral (neural net)
    'deeplift',           # DeepLIFT attribution
    'deepliftshap',       # DeepLIFT + SHAP
    'gradientshap',       # Gradient + SHAP sampling
]

IMPORTANCE_METHODS_SLOW = [
    'kernelshap',         # KernelSHAP (any model, slow)
    'lime',               # LIME local approximation
    'drop_column',        # Retrain without feature (gold standard)
    'boruta',             # Shadow feature selection (statistically sound)
]

# Combined for reference
IMPORTANCE_METHODS_ALL = IMPORTANCE_METHODS_FAST + IMPORTANCE_METHODS_MEDIUM + IMPORTANCE_METHODS_SLOW

# Search configurations
SEARCH_CONFIGS = {
    'quick': {
        # 1 config - fast sanity check
        'methods': ['greedy'],
        'budgets': [100],
        'importance_methods': ['random_forest'],
        'models': ['rf'],
        'n_folds': 3,
    },
    'balanced': {
        # 16 configs - tests dimension tiers with multiple importance methods
        'methods': ['greedy', 'performance_weighted'],
        'budgets': [50, 100, 200, 500],
        'importance_methods': ['random_forest', 'treeshap'],
        'models': ['rf'],
        'n_folds': 5,
    },
    'medium': {
        # 54 configs - adds model comparison + more importance methods
        'methods': ['greedy', 'performance_weighted'],
        'budgets': [50, 100, 200],
        'importance_methods': ['random_forest', 'treeshap', 'permutation'],
        'models': ['rf', 'xgboost', 'ridge'],
        'n_folds': 5,
    },
    'precise': {
        # 32 configs - feature-level greedy (HIGH COMPUTATION)
        # Use for small budgets where precision matters
        'methods': ['greedy_feature'],
        'budgets': [25, 50, 75, 100],
        'importance_methods': ['random_forest', 'treeshap', 'permutation', 'integrated_gradients'],
        'models': ['rf', 'xgboost'],
        'n_folds': 5,
    },
    'importance_compare': {
        # 28 configs - compare ALL fast importance methods
        'methods': ['greedy'],
        'budgets': [50, 100],
        'importance_methods': IMPORTANCE_METHODS_FAST,
        'models': ['rf', 'xgboost'],
        'n_folds': 5,
    },
    'full': {
        # 240 configs - exhaustive search with more importance methods
        'methods': ['greedy', 'performance_weighted', 'fixed'],
        'budgets': [50, 100, 200, 500],
        'importance_methods': ['random_forest', 'treeshap', 'permutation', 'xgboost_gain'],
        'models': ['rf', 'xgboost', 'ridge', 'mlp', 'knn'],
        'n_folds': 5,
    },
    'exhaustive': {
        # ALL methods (VERY HIGH COMPUTATION) - for final analysis
        'methods': ['greedy', 'greedy_feature', 'performance_weighted', 'mrmr'],
        'budgets': [50, 100],
        'importance_methods': IMPORTANCE_METHODS_FAST + ['permutation'],
        'models': ['rf', 'xgboost'],
        'n_folds': 5,
    },
}


def _get_xgb_regressor():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


def _get_xgb_classifier():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


@dataclass
class Blueprint:
    """Minimal recipe for recreating a hybrid."""
    reps: List[str]
    method: str
    budget: int
    importance: str

    def to_dict(self):
        return {
            'reps': self.reps,
            'method': self.method,
            'budget': self.budget,
            'importance': self.importance,
        }


@dataclass
class ConfigResult:
    """Result for a single configuration."""
    model: str
    method: str
    budget: int
    importance: str
    score_mean: float
    score_std: float
    n_features: int
    blueprint: Blueprint
    time_seconds: float


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(name_or_path: str, smiles_col: str = 'smiles',
                 target_col: str = 'target') -> Tuple[List[str], np.ndarray, str]:
    """Load dataset from built-in name or CSV path."""
    from rdkit import Chem

    name_lower = name_or_path.lower()

    if name_lower == 'esol':
        try:
            from kirby.datasets.esol import load_esol
            splits = load_esol(splitter='scaffold')
            smiles = (splits['train']['smiles'] + splits['val']['smiles'] + splits['test']['smiles'])
            labels = np.concatenate([splits['train']['labels'], splits['val']['labels'], splits['test']['labels']])
            return smiles, labels, 'regression'
        except ImportError:
            from deepchem.molnet import load_delaney
            _, datasets, _ = load_delaney(featurizer='Raw', splitter=None)
            return list(datasets[0].ids), datasets[0].y.flatten(), 'regression'

    elif name_lower == 'herg':
        try:
            from kirby.datasets.herg import load_herg_tdc
            data = load_herg_tdc()
            return data['smiles'], data['labels'], 'classification'
        except ImportError:
            from tdc.single_pred import Tox
            data = Tox(name='hERG', label_name='hERG_at_10uM')
            df = data.get_data()
            return df['Drug'].tolist(), df['Y'].values, 'classification'

    elif name_lower == 'logd':
        # OpenADMET-LogD dataset from HuggingFace
        import requests
        from pathlib import Path

        cache_dir = Path.home() / '.cache' / 'kirby'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / 'openadmet_train.csv'

        if not cached.exists():
            url = ("https://huggingface.co/datasets/openadmet/"
                   "openadmet-expansionrx-challenge-data/resolve/main/expansion_data_train.csv")
            print(f"  Downloading OpenADMET data from HuggingFace...")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            cached.write_bytes(resp.content)

        df = pd.read_csv(cached)
        logd_col = next((c for c in df.columns if 'LogD' in c), None)
        if logd_col is None:
            raise ValueError("Cannot find LogD column in OpenADMET data")

        valid_smiles, valid_labels = [], []
        for smi, label in zip(df['SMILES'], df[logd_col]):
            if pd.isna(smi) or pd.isna(label):
                continue
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_smiles.append(str(smi))
                valid_labels.append(float(label))

        print(f"  Loaded {len(valid_smiles)} molecules from OpenADMET-LogD")
        return valid_smiles, np.array(valid_labels), 'regression'

    elif name_or_path.endswith('.csv') or os.path.exists(name_or_path):
        print(f"Loading custom dataset from {name_or_path}...")
        df = pd.read_csv(name_or_path)

        smiles_col_found = None
        target_col_found = None
        for col in df.columns:
            if col.lower() == smiles_col.lower():
                smiles_col_found = col
            if col.lower() == target_col.lower():
                target_col_found = col

        if smiles_col_found is None:
            raise ValueError(f"SMILES column '{smiles_col}' not found. Available: {list(df.columns)}")
        if target_col_found is None:
            raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

        valid_smiles, valid_labels = [], []
        for smi, label in zip(df[smiles_col_found], df[target_col_found]):
            if pd.isna(smi) or pd.isna(label):
                continue
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_smiles.append(str(smi))
                valid_labels.append(label)

        labels = np.array(valid_labels)

        if np.issubdtype(labels.dtype, np.integer) or (
            np.issubdtype(labels.dtype, np.floating) and
            np.allclose(labels, labels.astype(int)) and
            len(np.unique(labels)) < 20
        ):
            task_type = 'classification'
        else:
            task_type = 'regression'

        print(f"  Loaded {len(valid_smiles)} valid molecules, task: {task_type}")
        return valid_smiles, labels, task_type

    else:
        raise ValueError(f"Unknown dataset: {name_or_path}")


# =============================================================================
# REPRESENTATION GENERATION
# =============================================================================

def get_representations(smiles: List[str], preset_or_list: str) -> Dict[str, np.ndarray]:
    """Generate representations based on preset or explicit list."""
    from kirby.representations.molecular import (
        create_ecfp4, create_maccs, create_pdv, create_mol2vec,
        create_chemberta, create_molformer, create_selformer,
        create_chembert, create_molclr, create_graphmvp, create_smited,
        create_mhg_gnn, create_grover, create_graph_kernel
    )

    if preset_or_list in REP_PRESETS:
        rep_names = REP_PRESETS[preset_or_list]
    else:
        rep_names = [r.strip().lower() for r in preset_or_list.split(',')]

    print(f"\nGenerating {len(rep_names)} representations...")

    rep_funcs = {
        'ecfp4': create_ecfp4,
        'maccs': create_maccs,
        'pdv': create_pdv,
        'mol2vec': create_mol2vec,
        'chemberta': create_chemberta,
        'molformer': create_molformer,
        'selformer': create_selformer,
        'chembert': create_chembert,
        'molclr': create_molclr,
        'graphmvp': create_graphmvp,
        'smited': create_smited,
        'mhg_gnn': create_mhg_gnn,
        'grover': create_grover,
        'graph_kernel': create_graph_kernel,
    }

    representations = {}
    for rep_name in rep_names:
        if rep_name not in rep_funcs:
            print(f"  WARNING: Unknown representation '{rep_name}', skipping")
            continue

        try:
            start = time.time()
            print(f"  Generating {rep_name}...", end=' ', flush=True)
            rep = rep_funcs[rep_name](smiles)
            elapsed = time.time() - start
            print(f"done ({rep.shape[1]} features, {elapsed:.1f}s)")
            representations[rep_name] = rep
        except Exception as e:
            print(f"FAILED: {e}")

    if not representations:
        raise ValueError("No representations could be generated!")

    return representations


def get_augmentations(smiles: List[str]) -> Dict[str, np.ndarray]:
    """Generate augmented features (graph topology, spectral)."""
    from kirby.representations.molecular import compute_graph_topology, compute_spectral_features

    print("\nGenerating augmented features...")
    augmentations = {}

    # Graph topology features
    try:
        print("  Generating graph_topology...", end=' ', flush=True)
        start = time.time()
        topo = compute_graph_topology(smiles)
        elapsed = time.time() - start
        print(f"done ({topo.shape[1]} features, {elapsed:.1f}s)")
        augmentations['graph_topology'] = topo
    except Exception as e:
        print(f"FAILED: {e}")

    # Spectral features
    try:
        print("  Generating spectral...", end=' ', flush=True)
        start = time.time()
        spec = compute_spectral_features(smiles, k=10)
        elapsed = time.time() - start
        print(f"done ({spec.shape[1]} features, {elapsed:.1f}s)")
        augmentations['spectral'] = spec
    except Exception as e:
        print(f"FAILED: {e}")

    return augmentations


# =============================================================================
# BASELINE EVALUATION
# =============================================================================

def evaluate_baselines(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    task_type: str,
    models_to_eval: List[str],
    scaffold_groups: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate full-dimension baselines for each representation and model.

    Uses scaffold-based CV splits for chemically meaningful evaluation.

    Returns dict: {rep_name: {model_name: score}}
    """
    print("\nEvaluating full-dimension baselines (scaffold CV)...")

    baselines = {}

    # Individual reps
    for rep_name, X in base_reps.items():
        baselines[rep_name] = {'dimension': X.shape[1]}

        for model_name in models_to_eval:
            score = evaluate_single_config(
                X, labels, task_type, model_name, scaffold_groups, n_folds, seed
            )
            baselines[rep_name][model_name] = round(score, 4)

        print(f"  {rep_name} (dim={X.shape[1]}): " +
              ", ".join(f"{m}={baselines[rep_name][m]:.3f}" for m in models_to_eval))

    # All concatenated
    X_concat = np.hstack(list(base_reps.values()))
    baselines['all_concat'] = {'dimension': X_concat.shape[1]}

    for model_name in models_to_eval:
        score = evaluate_single_config(
            X_concat, labels, task_type, model_name, scaffold_groups, n_folds, seed
        )
        baselines['all_concat'][model_name] = round(score, 4)

    print(f"  all_concat (dim={X_concat.shape[1]}): " +
          ", ".join(f"{m}={baselines['all_concat'][m]:.3f}" for m in models_to_eval))

    return baselines


def evaluate_single_config(
    X: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    model_name: str,
    scaffold_groups: np.ndarray,
    n_folds: int,
    seed: int
) -> float:
    """Evaluate a single X matrix with a single model using scaffold CV."""
    gkf = GroupKFold(n_splits=n_folds)
    scores = []

    for train_idx, val_idx in gkf.split(X, labels, groups=scaffold_groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if task_type == 'classification':
            model = MODELS[model_name]['classifier']()
            model.fit(X_train_scaled, y_train)
            score = accuracy_score(y_val, model.predict(X_val_scaled))
        else:
            model = MODELS[model_name]['regressor']()
            model.fit(X_train_scaled, y_train)
            score = r2_score(y_val, model.predict(X_val_scaled))

        scores.append(score)

    return np.mean(scores)


# =============================================================================
# HYBRID EVALUATION
# =============================================================================

def evaluate_hybrid_config(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    task_type: str,
    method: str,
    budget: int,
    importance_method: str,
    model_name: str,
    scaffold_groups: np.ndarray,
    n_folds: int,
    seed: int = 42,
    augmentations: Optional[Dict[str, np.ndarray]] = None
) -> ConfigResult:
    """Evaluate a single hybrid configuration with a single model using scaffold CV."""
    from kirby.hybrid import create_hybrid, apply_feature_selection

    start_time = time.time()

    gkf = GroupKFold(n_splits=n_folds)
    fold_scores = []
    fold_n_features = []

    # Get a sample X for splitting (all reps have same n_samples)
    sample_X = next(iter(base_reps.values()))

    for train_idx, val_idx in gkf.split(sample_X, labels, groups=scaffold_groups):
        train_reps = {k: v[train_idx] for k, v in base_reps.items()}
        val_reps = {k: v[val_idx] for k, v in base_reps.items()}
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        # Split augmentations if provided
        train_augs = None
        val_augs = None
        if augmentations:
            train_augs = {k: v[train_idx] for k, v in augmentations.items()}
            val_augs = {k: v[val_idx] for k, v in augmentations.items()}

        try:
            X_train, feature_info = create_hybrid(
                train_reps, train_labels,
                allocation_method=method,
                budget=budget,
                importance_method=importance_method,
                augmentations=train_augs,
                augmentation_strategy='greedy_ablation' if train_augs else None,
            )
            X_val = apply_feature_selection(val_reps, feature_info, augmentations=val_augs)
        except Exception as e:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if task_type == 'classification':
            model = MODELS[model_name]['classifier']()
            model.fit(X_train_scaled, train_labels)
            score = accuracy_score(val_labels, model.predict(X_val_scaled))
        else:
            model = MODELS[model_name]['regressor']()
            model.fit(X_train_scaled, train_labels)
            score = r2_score(val_labels, model.predict(X_val_scaled))

        fold_scores.append(score)
        fold_n_features.append(X_train.shape[1])

    if not fold_scores:
        raise ValueError("All folds failed!")

    elapsed = time.time() - start_time

    blueprint = Blueprint(
        reps=list(base_reps.keys()),
        method=method,
        budget=budget,
        importance=importance_method,
    )

    return ConfigResult(
        model=model_name,
        method=method,
        budget=budget,
        importance=importance_method,
        score_mean=np.mean(fold_scores),
        score_std=np.std(fold_scores),
        n_features=int(np.mean(fold_n_features)),
        blueprint=blueprint,
        time_seconds=elapsed,
    )


def run_search(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    task_type: str,
    scaffold_groups: np.ndarray,
    search_mode: str,
    seed: int = 42,
    augmentations: Optional[Dict[str, np.ndarray]] = None,
    models_override: Optional[List[str]] = None
) -> List[ConfigResult]:
    """Run grid search over all configurations using scaffold CV.

    Args:
        models_override: If provided, use these models instead of search config defaults.
    """
    config = SEARCH_CONFIGS[search_mode]
    models_to_use = models_override if models_override else config['models']
    results = []

    total_configs = (
        len(config['methods']) *
        len(config['budgets']) *
        len(config['importance_methods']) *
        len(models_to_use)
    )

    aug_str = " +augment" if augmentations else ""
    print(f"\nSearching {total_configs} configurations ({search_mode} mode, scaffold CV{aug_str})...")

    config_idx = 0
    for method in config['methods']:
        for budget in config['budgets']:
            for importance in config['importance_methods']:
                for model_name in models_to_use:
                    config_idx += 1

                    try:
                        result = evaluate_hybrid_config(
                            base_reps=base_reps,
                            labels=labels,
                            task_type=task_type,
                            method=method,
                            budget=budget,
                            importance_method=importance,
                            model_name=model_name,
                            scaffold_groups=scaffold_groups,
                            n_folds=config['n_folds'],
                            seed=seed,
                            augmentations=augmentations,
                        )
                        results.append(result)

                        metric = 'Acc' if task_type == 'classification' else 'R²'
                        print(f"  [{config_idx}/{total_configs}] {model_name}/{method}/{importance}/b={budget}: "
                              f"{metric}={result.score_mean:.4f}±{result.score_std:.3f}, dim={result.n_features}")

                    except Exception as e:
                        print(f"  [{config_idx}/{total_configs}] {model_name}/{method}/{importance}/b={budget}: FAILED - {e}")

    return results


# =============================================================================
# RESULT AGGREGATION
# =============================================================================

def aggregate_results(
    results: List[ConfigResult],
    baselines: Dict[str, Dict[str, float]],
    models_evaluated: List[str]
) -> Dict[str, Any]:
    """Aggregate results into best-by-dimension, best-by-model, and matrices."""

    if not results:
        return {}

    # Best overall
    best_overall = max(results, key=lambda x: x.score_mean)

    # Best by dimension
    best_by_dimension = {}
    for tier in DIMENSION_TIERS:
        eligible = [r for r in results if r.n_features <= tier]
        if eligible:
            best = max(eligible, key=lambda x: x.score_mean)
            best_by_dimension[tier] = best

    # Best by model
    best_by_model = {}
    for model_name in models_evaluated:
        model_results = [r for r in results if r.model == model_name]
        if model_results:
            best = max(model_results, key=lambda x: x.score_mean)
            best_by_model[model_name] = best

    # Performance matrix: model × dimension
    performance_matrix = {}
    for model_name in models_evaluated:
        performance_matrix[model_name] = {}
        for tier in DIMENSION_TIERS:
            eligible = [r for r in results if r.model == model_name and r.n_features <= tier]
            if eligible:
                best = max(eligible, key=lambda x: x.score_mean)
                performance_matrix[model_name][str(tier)] = round(best.score_mean, 4)

    return {
        'best_overall': best_overall,
        'best_by_dimension': best_by_dimension,
        'best_by_model': best_by_model,
        'performance_matrix': performance_matrix,
        'baselines': baselines,
    }


# =============================================================================
# OUTPUT
# =============================================================================

def print_summary(aggregated: Dict[str, Any], task_type: str, models_evaluated: List[str]):
    """Print formatted summary to console."""
    metric = 'R²' if task_type == 'regression' else 'Accuracy'

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Best overall
    best = aggregated['best_overall']
    print(f"\n=== BEST OVERALL ===")
    print(f"Model: {best.model}, Method: {best.method}, Importance: {best.importance}, Budget: {best.budget}")
    print(f"{metric}: {best.score_mean:.4f} ± {best.score_std:.4f}, Dimension: {best.n_features}")

    # Best by model
    print(f"\n=== BEST BY MODEL ===")
    print(f"{'Model':<10} {'Dim':<6} {'Method':<15} {'Importance':<15} {metric:<14} {'Budget'}")
    print("-" * 80)
    for model_name, result in aggregated['best_by_model'].items():
        bp = result.blueprint
        print(f"{model_name:<10} {result.n_features:<6} {result.method:<15} {result.importance:<15} "
              f"{result.score_mean:.4f}±{result.score_std:.2f}   {bp.budget}")

    # Best by dimension
    print(f"\n=== BEST BY DIMENSION ===")
    print(f"{'Dim':<8} {'Model':<10} {'Method':<15} {'Importance':<15} {metric:<12}")
    print("-" * 70)
    for tier, result in sorted(aggregated['best_by_dimension'].items()):
        print(f"≤{tier:<7} {result.model:<10} {result.method:<15} {result.importance:<15} {result.score_mean:.4f}±{result.score_std:.2f}")

    # Performance matrix
    print(f"\n=== PERFORMANCE MATRIX ({metric}) ===")
    tiers = [str(t) for t in DIMENSION_TIERS]
    header = f"{'Model':<12}" + "".join(f"{'≤'+t:<10}" for t in tiers)
    print(header)
    print("-" * len(header))
    for model_name in models_evaluated:
        if model_name in aggregated['performance_matrix']:
            row = f"{model_name:<12}"
            for tier in tiers:
                score = aggregated['performance_matrix'][model_name].get(tier, None)
                if score is not None:
                    row += f"{score:<10.4f}"
                else:
                    row += f"{'--':<10}"
            print(row)

    # Baselines comparison
    print(f"\n=== BASELINE COMPARISON (full dimension) ===")
    baselines = aggregated['baselines']
    header = f"{'Baseline':<15} {'Dim':<8}" + "".join(f"{m:<10}" for m in models_evaluated)
    print(header)
    print("-" * len(header))
    for baseline_name, baseline_scores in baselines.items():
        row = f"{baseline_name:<15} {baseline_scores.get('dimension', '?'):<8}"
        for m in models_evaluated:
            score = baseline_scores.get(m, None)
            if score is not None:
                row += f"{score:<10.4f}"
            else:
                row += f"{'--':<10}"
        print(row)

    # Improvement summary
    print(f"\n=== HYBRID IMPROVEMENT ===")
    for model_name, result in aggregated['best_by_model'].items():
        hybrid_score = result.score_mean

        # Find best individual baseline for this model
        best_baseline_name = None
        best_baseline_score = -np.inf
        for baseline_name, baseline_scores in baselines.items():
            if baseline_name == 'all_concat':
                continue
            if model_name in baseline_scores and baseline_scores[model_name] > best_baseline_score:
                best_baseline_score = baseline_scores[model_name]
                best_baseline_name = baseline_name

        if best_baseline_name:
            improvement = hybrid_score - best_baseline_score
            baseline_dim = baselines[best_baseline_name].get('dimension', '?')
            print(f"  {model_name}: Hybrid({result.n_features}d) vs {best_baseline_name}({baseline_dim}d): "
                  f"{hybrid_score:.4f} vs {best_baseline_score:.4f} ({improvement:+.4f})")

    print("=" * 80)


def save_results(
    aggregated: Dict[str, Any],
    dataset_name: str,
    search_mode: str,
    reps_tested: List[str],
    output_path: str,
    base_reps: Optional[Dict[str, np.ndarray]] = None,
    labels: Optional[np.ndarray] = None
):
    """Save minimal blueprint JSON with optional importance weights."""

    def result_to_dict(r: ConfigResult, include_weights: bool = False):
        d = {
            'model': r.model,
            'score_mean': round(r.score_mean, 4),
            'score_std': round(r.score_std, 4),
            'dimension': r.n_features,
            'blueprint': r.blueprint.to_dict(),
        }

        # Optionally add feature importance weights
        if include_weights and base_reps is not None and labels is not None:
            try:
                _, feature_info = create_hybrid(
                    base_reps, labels,
                    allocation_method=r.blueprint.method,
                    budget=r.blueprint.budget,
                    importance_method=r.blueprint.importance,
                )
                # Extract weights in hybrid feature order
                weights_by_rep = {}
                for rep_name in feature_info:
                    if rep_name in ('allocation', 'greedy_history', 'greedy_feature_history'):
                        continue
                    if isinstance(feature_info[rep_name], dict) and 'importance_scores' in feature_info[rep_name]:
                        scores = feature_info[rep_name]['importance_scores']
                        indices = feature_info[rep_name]['selected_indices']
                        weights_by_rep[rep_name] = {
                            'indices': indices.tolist() if hasattr(indices, 'tolist') else list(indices),
                            'scores': scores.tolist() if hasattr(scores, 'tolist') else list(scores),
                        }
                d['feature_weights'] = weights_by_rep
            except Exception as e:
                print(f"  Warning: Could not extract weights for {r.model}: {e}")

        return d

    output = {
        'metadata': {
            'dataset': dataset_name,
            'search_mode': search_mode,
            'reps_available': reps_tested,
            'timestamp': datetime.now().isoformat(),
        },
        # Include weights for best_overall (most commonly used)
        'best_overall': result_to_dict(aggregated['best_overall'], include_weights=True),
        'best_by_dimension': {
            str(tier): result_to_dict(r)
            for tier, r in aggregated['best_by_dimension'].items()
        },
        'best_by_model': {
            model: result_to_dict(r, include_weights=True)
            for model, r in aggregated['best_by_model'].items()
        },
        'performance_matrix': aggregated['performance_matrix'],
        'baselines': aggregated['baselines'],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Also save a Python blueprint file
    blueprint_path = output_path.replace('.json', '_blueprint.py')
    save_blueprint_code(aggregated, reps_tested, blueprint_path)


def save_blueprint_code(aggregated: Dict[str, Any], reps_tested: List[str], output_path: str):
    """Generate importable Python code for reusing the hybrids."""

    lines = [
        '"""',
        'Auto-generated hybrid blueprints.',
        '',
        'Usage:',
        '    from results_blueprint import create_rf_hybrid',
        '    X = create_rf_hybrid(smiles_list, labels)',
        '"""',
        '',
        'from kirby.hybrid import create_hybrid, apply_feature_selection',
        'from kirby.representations.molecular import (',
        '    create_ecfp4, create_maccs, create_pdv,',
        '    create_chemberta, create_molformer, create_selformer,',
        '    create_chembert, create_mol2vec,',
        ')',
        '',
        '',
        '# Representation generators',
        'REP_GENERATORS = {',
        "    'ecfp4': create_ecfp4,",
        "    'maccs': create_maccs,",
        "    'pdv': create_pdv,",
        "    'chemberta': create_chemberta,",
        "    'molformer': create_molformer,",
        "    'selformer': create_selformer,",
        "    'chembert': create_chembert,",
        "    'mol2vec': create_mol2vec,",
        '}',
        '',
        '',
    ]

    # Generate function for each model
    for model_name, result in aggregated['best_by_model'].items():
        bp = result.blueprint

        lines.extend([
            f"def create_{model_name}_hybrid(smiles_list, labels):",
            f'    """',
            f'    Create the optimal hybrid for {model_name.upper()} model.',
            f'    ',
            f'    Blueprint: method={bp.method}, budget={bp.budget}, importance={bp.importance}',
            f'    Expected dimension: ~{result.n_features}',
            f'    Expected score: {result.score_mean:.4f}',
            f'    """',
            f'    # Generate required representations',
            f'    reps = {{}}',
        ])

        for rep_name in bp.reps:
            lines.append(f"    reps['{rep_name}'] = REP_GENERATORS['{rep_name}'](smiles_list)")

        lines.extend([
            f'    ',
            f'    # Create hybrid with optimal settings',
            f'    X_hybrid, feature_info = create_hybrid(',
            f'        reps, labels,',
            f"        allocation_method='{bp.method}',",
            f'        budget={bp.budget},',
            f"        importance_method='{bp.importance}',",
            f'    )',
            f'    ',
            f'    return X_hybrid, feature_info',
            f'',
            f'',
        ])

    # Write file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Blueprint code saved to: {output_path}")


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def compare_importance_methods(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    methods: List[str] = ['random_forest', 'permutation'],
) -> Dict[str, Any]:
    """
    Compare how different importance methods rank features.

    Returns correlation between methods and top feature overlap.
    """
    from kirby.hybrid import compute_feature_importance
    from scipy.stats import spearmanr, kendalltau

    print("\n=== IMPORTANCE METHOD COMPARISON ===")

    # Compute importance for each method
    all_importances = {}
    for method in methods:
        print(f"  Computing {method} importance...", end=' ', flush=True)
        try:
            imp = compute_feature_importance(base_reps, labels, method=method)
            # Concatenate all reps
            concat_imp = np.concatenate([imp[rep] for rep in sorted(imp.keys())])
            all_importances[method] = concat_imp
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    if len(all_importances) < 2:
        return {'error': 'Need at least 2 methods to compare'}

    # Compute correlations between methods
    method_names = list(all_importances.keys())
    correlations = {}

    for i, m1 in enumerate(method_names):
        for m2 in method_names[i+1:]:
            imp1 = all_importances[m1]
            imp2 = all_importances[m2]

            # Spearman rank correlation
            spearman_r, spearman_p = spearmanr(imp1, imp2)
            # Kendall tau
            kendall_t, kendall_p = kendalltau(imp1, imp2)

            key = f"{m1}_vs_{m2}"
            correlations[key] = {
                'spearman_r': round(spearman_r, 4),
                'spearman_p': round(spearman_p, 6),
                'kendall_tau': round(kendall_t, 4),
                'kendall_p': round(kendall_p, 6),
            }
            print(f"  {m1} vs {m2}: Spearman r={spearman_r:.3f}, Kendall τ={kendall_t:.3f}")

    # Compare top-k overlap
    top_k_values = [10, 25, 50, 100]
    top_k_overlap = {}

    for k in top_k_values:
        overlap_scores = []
        for i, m1 in enumerate(method_names):
            for m2 in method_names[i+1:]:
                top1 = set(np.argsort(all_importances[m1])[-k:])
                top2 = set(np.argsort(all_importances[m2])[-k:])
                overlap = len(top1 & top2) / k
                overlap_scores.append(overlap)
        top_k_overlap[f'top_{k}'] = round(np.mean(overlap_scores), 3)

    print(f"  Top-k overlap: {top_k_overlap}")

    return {
        'methods': method_names,
        'correlations': correlations,
        'top_k_overlap': top_k_overlap,
        'importances': {m: imp.tolist() for m, imp in all_importances.items()},
    }


def generate_explanations(
    best_result: ConfigResult,
    base_reps: Dict[str, np.ndarray],
    smiles: List[str],
    labels: np.ndarray,
    output_path: str
):
    """Generate and save feature explanations for the best configuration."""
    from kirby.hybrid import create_hybrid
    from kirby.importance.hybrid import explain_hybrid, format_explanation_summary

    print("\n=== GENERATING EXPLANATIONS ===")

    bp = best_result.blueprint

    # Recreate the hybrid to get feature_info
    print(f"  Recreating hybrid: method={bp.method}, budget={bp.budget}")
    _, feature_info = create_hybrid(
        base_reps, labels,
        allocation_method=bp.method,
        budget=bp.budget,
        importance_method=bp.importance,
    )

    # Generate explanations
    print("  Mapping features to interpretations...")
    explanations = explain_hybrid(
        feature_info=feature_info,
        smiles_list=smiles[:100],  # Use subset for ECFP SMARTS extraction
        top_k=30,
    )

    # Print summary
    print("\n" + format_explanation_summary(explanations, max_features=15))

    # Compare importance methods
    importance_comparison = compare_importance_methods(base_reps, labels)

    # Save to file
    explain_path = output_path.replace('.json', '_explanations.json')

    # Convert to JSON-serializable format
    explain_output = {
        'model': best_result.model,
        'blueprint': bp.to_dict(),
        'by_representation': explanations['by_representation'],
        'top_features': [
            {
                'rank': i + 1,
                'rep': f['rep_name'],
                'local_idx': f['local_idx'],
                'importance': round(f['importance'], 6),
                'description': f['interpretation'].get('description', ''),
                'type': f['interpretation'].get('type', ''),
            }
            for i, f in enumerate(explanations['top_features'])
        ],
        'importance_method_comparison': {
            'correlations': importance_comparison.get('correlations', {}),
            'top_k_overlap': importance_comparison.get('top_k_overlap', {}),
        },
    }

    import json
    with open(explain_path, 'w') as f:
        json.dump(explain_output, f, indent=2)

    print(f"\nExplanations saved to: {explain_path}")

    return explanations


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Find the optimal hybrid molecular representation for each model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s esol --reps fast --search quick
  %(prog)s herg --reps fast+hf --search medium
  %(prog)s data.csv --smiles-col SMILES --target-col pIC50 --reps all --search full

Output:
  - Best overall: single best model+hybrid combination
  - Best by dimension: optimal at each tier (50, 100, 200, 500)
  - Best by model: optimal hybrid for RF, XGBoost, Ridge, MLP, KNN
  - Performance matrix: full Model × Dimension grid
  - Baseline comparison: hybrid vs full-dimension baselines
        """
    )

    parser.add_argument('dataset', type=str, help="'esol', 'herg', or path to CSV")
    parser.add_argument('--smiles-col', type=str, default='smiles')
    parser.add_argument('--target-col', type=str, default='target')
    parser.add_argument('--reps', type=str, default='fast',
                       help="'fast', 'fast+hf', 'all', or comma-separated list")
    parser.add_argument('--search', type=str, default='quick',
                       choices=['quick', 'balanced', 'medium', 'precise', 'importance_compare', 'full', 'exhaustive'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--explain', action='store_true',
                       help='Generate feature explanations for best configs')
    parser.add_argument('--augment', action='store_true',
                       help='Include augmented features (graph topology, spectral)')
    parser.add_argument('--model', type=str, default=None,
                       help="Override models: 'rf', 'xgboost', 'ridge', 'mlp', 'knn', or comma-separated")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    smiles, labels, task_type = load_dataset(
        args.dataset, args.smiles_col, args.target_col
    )
    print(f"  {len(smiles)} molecules, task: {task_type}")

    # Compute scaffold groups for CV
    print("  Assigning scaffold groups...")
    scaffold_groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  {n_scaffolds} unique scaffolds")

    # Generate representations
    base_reps = get_representations(smiles, args.reps)

    # Generate augmentations if requested
    augmentations = None
    if args.augment:
        augmentations = get_augmentations(smiles)

    # Get search config
    search_config = SEARCH_CONFIGS[args.search]

    # Determine models to evaluate (CLI override or search config default)
    if args.model:
        models_to_eval = [m.strip() for m in args.model.split(',')]
        # Validate model names
        invalid = [m for m in models_to_eval if m not in MODELS]
        if invalid:
            print(f"ERROR: Unknown model(s): {invalid}")
            print(f"Valid models: {list(MODELS.keys())}")
            return 1
        print(f"Model override: {models_to_eval}")
    else:
        models_to_eval = search_config['models']

    # Evaluate baselines
    baselines = evaluate_baselines(
        base_reps, labels, task_type, models_to_eval, scaffold_groups,
        n_folds=search_config['n_folds'], seed=args.seed
    )

    # Run hybrid search
    results = run_search(
        base_reps, labels, task_type, scaffold_groups, args.search, args.seed,
        augmentations=augmentations,
        models_override=models_to_eval if args.model else None
    )

    # Aggregate
    aggregated = aggregate_results(results, baselines, models_to_eval)

    # Print summary
    print_summary(aggregated, task_type, models_to_eval)

    # Save results
    if args.output:
        output_path = args.output
    else:
        dataset_name = Path(args.dataset).stem if os.path.exists(args.dataset) else args.dataset
        output_path = f"results_{dataset_name}_{args.search}.json"

    save_results(aggregated, args.dataset, args.search, list(base_reps.keys()), output_path,
                 base_reps=base_reps, labels=labels)

    # Generate explanations if requested
    if args.explain and aggregated.get('best_overall'):
        generate_explanations(
            best_result=aggregated['best_overall'],
            base_reps=base_reps,
            smiles=smiles,
            labels=labels,
            output_path=output_path,
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
