import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', message='.*experimental_relax_shapes.*')
warnings.filterwarnings('ignore', message='.*reduce_retracing.*')

"""
Baseline Noise Robustness: OpenADMET-LogD, OpenADMET-Caco2 Efflux, FLuID hERG
===============================================================================

Tests noise robustness on three defensible baseline datasets with the full
model-representation matrix matching the DRD2/QM9 experiments.

NO repetitions (n=1) - single run per configuration (matching DRD2 protocol).

Datasets:
  1. OpenADMET-LogD (REGRESSION) — 7309 molecules, lipophilicity
     Source: ExpansionRx via HuggingFace (CC-BY-4.0)
     Baseline: R²=0.805 (LGBM scaffold CV from benchmark)
     Notes: Untransformed. Most ECFP-learnable ADMET endpoint.

  2. OpenADMET-Caco2_Efflux (REGRESSION) — 3777 molecules, P-gp efflux ratio
     Source: ExpansionRx via HuggingFace (CC-BY-4.0)
     Baseline: R²=0.703 (LGBM scaffold CV from benchmark)
     Notes: log10 transformed. Functional ADME endpoint.

  3. FLuID hERG (CLASSIFICATION) — Lhasa Limited benchmark
     Source: kirby.datasets.herg (source='fluid')
     Notes: Binary blocker/non-blocker. Pre-split train/test.

Model-Representation Matrix (per dataset):
  Representations: ECFP4, PDV, SNS, MHG-GNN-pretrained (4 total)
  Regression models: RF, QRF, XGBoost, GP(PDV only), DNN, Full-BNN,
                     LastLayer-BNN, Var-BNN (8 total)
  Classification models: RF, XGBoost, GP(PDV only), DNN, Full-BNN,
                         LastLayer-BNN, Var-BNN (7 total)
  Total: 4 reps × ~8 models × 6 strategies × 3 datasets

Noise:
  Strategies: legacy, outlier, quantile, hetero, threshold, valprop (6)
  Sigma levels: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (11)

Key output metric:
  NDS (Noise Degradation Slope): linear slope of R² (or AUC) vs σ
  More negative = degrades faster with noise = less robust

Usage:
  python run_baselines_noise.py
  python run_baselines_noise.py --datasets logd caco2
  python run_baselines_noise.py --datasets herg
  python run_baselines_noise.py --openadmet-csv /path/to/cached/expansion_data_train.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
)
from scipy.stats import linregress, spearmanr
import sys
import time
import requests
from pathlib import Path

# RDKit for scaffold splitting and SMILES standardization
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.logger().setLevel(RDLogger.ERROR)


def standardise_smiles(smi):
    """Canonicalise SMILES, keep largest fragment, remove salts.

    Matches test_datasets.py standardization protocol.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Largest fragment (salt removal)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

# ── Optional imports ──────────────────────────────────────────────────────
try:
    import torchbnn as bnn
    from bayesian_torch.models.dnn_to_bnn import transform_model, transform_layer
    HAS_BAYESIAN_TORCH = True
except ImportError:
    print("WARNING: torchbnn or bayesian-torch not installed, BNN experiments will be skipped")
    HAS_BAYESIAN_TORCH = False

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("WARNING: xgboost not installed, XGBoost experiments will be skipped")
    HAS_XGBOOST = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    HAS_GP = True
except ImportError:
    print("WARNING: sklearn.gaussian_process not available, GP experiments will be skipped")
    HAS_GP = False

try:
    from quantile_forest import RandomForestQuantileRegressor
    HAS_QRF = True
except ImportError:
    print("WARNING: quantile_forest not installed, QRF experiments will be skipped")
    HAS_QRF = False

# ── KIRBy imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_sns,
    create_mhg_gnn
)
from noiseInject import NoiseInjectorRegression, calculate_noise_metrics

# Try classification noise injector; fall back to our own implementation
try:
    from noiseInject import NoiseInjectorClassification
    HAS_NOISE_CLASSIFICATION = True
except ImportError:
    HAS_NOISE_CLASSIFICATION = False

# hERG data loader
try:
    from kirby.datasets.herg import load_herg
    HAS_HERG = True
except ImportError:
    print("WARNING: kirby.datasets.herg not available, hERG experiments will be skipped")
    HAS_HERG = False


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

STRATEGIES = ['legacy', 'outlier', 'quantile', 'hetero', 'threshold', 'valprop']
SIGMA_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
GP_MAX_N = 2000  # Subsample for GP (O(n³) cost)
CACHE_DIR = Path('data_cache')


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def download_openadmet(csv_path=None):
    """Download or load OpenADMET-ExpansionRx data.

    Returns the full DataFrame with all endpoints.
    """
    if csv_path and Path(csv_path).exists():
        print(f"  Loading cached OpenADMET data from {csv_path}")
        return pd.read_csv(csv_path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / 'openadmet_train.csv'

    if cached.exists():
        print(f"  Loading cached OpenADMET data from {cached}")
        return pd.read_csv(cached)

    url = (
        "https://huggingface.co/datasets/openadmet/"
        "openadmet-expansionrx-challenge-data/resolve/main/expansion_data_train.csv"
    )
    print(f"  Downloading OpenADMET data from HuggingFace...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    print(f"  Saved to {cached}")
    return pd.read_csv(cached)


def scaffold_split(smiles_list, test_frac=0.2, random_state=42):
    """Bemis-Murcko scaffold-based train/test split.

    Returns (train_indices, test_indices) as numpy arrays.
    """
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False)
            except Exception:
                scaf = smi
        else:
            scaf = smi
        scaffolds.setdefault(scaf, []).append(i)

    rng = np.random.RandomState(random_state)
    groups = list(scaffolds.values())
    rng.shuffle(groups)

    n_total = len(smiles_list)
    n_test = int(n_total * test_frac)

    test_idx, train_idx = [], []
    for group in groups:
        if len(test_idx) < n_test:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    return np.array(train_idx), np.array(test_idx)


def load_openadmet_endpoint(df, endpoint_col, log_transform=False):
    """Extract a single endpoint from OpenADMET, scaffold-split it.

    Matches test_datasets.py protocol:
      - Standardizes SMILES (canonicalize, salt removal)
      - Deduplicates by taking median target per canonical SMILES
      - Scaffold split: 80 train / 20 test

    Returns dict with keys: train_smiles, train_labels, val_smiles, val_labels,
                            test_smiles, test_labels
    """
    # Find SMILES column
    smiles_col = next(c for c in df.columns if c.upper() == 'SMILES')

    sub = df[[smiles_col, endpoint_col]].dropna(subset=[endpoint_col]).copy()
    sub[endpoint_col] = pd.to_numeric(sub[endpoint_col], errors='coerce')
    sub = sub.dropna()

    if len(sub) < 100:
        raise ValueError(f"Only {len(sub)} valid rows for {endpoint_col}")

    # Standardize SMILES (matching test_datasets.py protocol)
    sub['std_smiles'] = sub[smiles_col].apply(standardise_smiles)
    sub = sub.dropna(subset=['std_smiles'])

    # Deduplicate by canonical SMILES: take median target (matching test_datasets.py)
    sub = sub.groupby('std_smiles').agg({endpoint_col: 'median'}).reset_index()

    if len(sub) < 100:
        raise ValueError(f"Only {len(sub)} unique molecules for {endpoint_col}")

    smiles_arr = sub['std_smiles'].values
    labels_arr = sub[endpoint_col].values.astype(np.float64)

    if log_transform:
        labels_arr = np.log10(np.clip(labels_arr, 1e-10, None))
        valid = np.isfinite(labels_arr)
        smiles_arr = smiles_arr[valid]
        labels_arr = labels_arr[valid]

    if len(smiles_arr) < 100:
        raise ValueError(f"Only {len(smiles_arr)} valid after transform for {endpoint_col}")

    # Scaffold split: 80 train / 20 test
    train_idx, test_idx = scaffold_split(smiles_arr.tolist(), test_frac=0.2,
                                         random_state=42)

    train_smiles = smiles_arr[train_idx]
    train_labels = labels_arr[train_idx]
    test_smiles = smiles_arr[test_idx]
    test_labels = labels_arr[test_idx]

    # Carve validation from train (20% of train = 16% of total)
    n_val = len(train_smiles) // 5
    val_smiles = train_smiles[:n_val]
    val_labels = train_labels[:n_val]
    train_smiles_fit = train_smiles[n_val:]
    train_labels_fit = train_labels[n_val:]

    return {
        'train_smiles': train_smiles_fit,
        'train_labels': train_labels_fit,
        'val_smiles': val_smiles,
        'val_labels': val_labels,
        'test_smiles': test_smiles,
        'test_labels': test_labels,
    }


def load_herg_fluid():
    """Load FLuID hERG from kirby with its canonical train/test split.

    Returns dict with same keys as load_openadmet_endpoint.
    """
    print("  Loading FLuID training set...")
    train_data = load_herg(source='fluid', use_test=False)
    print("  Loading FLuID test set...")
    test_data = load_herg(source='fluid', use_test=True)

    train_smiles = train_data['smiles']
    train_labels = train_data['labels']
    test_smiles = test_data['smiles']
    test_labels = test_data['labels']

    # Carve validation from train
    n_val = len(train_smiles) // 5
    val_smiles = train_smiles[:n_val]
    val_labels = train_labels[:n_val]
    train_smiles_fit = train_smiles[n_val:]
    train_labels_fit = train_labels[n_val:]

    return {
        'train_smiles': train_smiles_fit,
        'train_labels': train_labels_fit,
        'val_smiles': val_smiles,
        'val_labels': val_labels,
        'test_smiles': test_smiles,
        'test_labels': test_labels,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION NOISE INJECTOR (fallback if noiseInject doesn't have one)
# ═══════════════════════════════════════════════════════════════════════════

class _ClassificationNoiseInjector:
    """Label-flipping noise for binary classification.

    Mirrors the NoiseInjectorRegression interface but operates on {0,1} labels.
    Each strategy creates a meaningfully different flipping pattern.
    """

    def __init__(self, strategy='legacy', random_state=42):
        self.strategy = strategy
        self.random_state = random_state

    def inject(self, labels, sigma):
        rng = np.random.RandomState(self.random_state)
        noisy = labels.copy().astype(float)
        n = len(labels)

        if sigma == 0.0:
            return noisy.astype(labels.dtype)

        if self.strategy == 'legacy':
            # Symmetric uniform flip
            flip_mask = rng.random(n) < (sigma * 0.5)
            noisy[flip_mask] = 1.0 - noisy[flip_mask]

        elif self.strategy == 'outlier':
            # Heavier flipping on the minority class
            pos_rate = labels.mean()
            minority = 1 if pos_rate < 0.5 else 0
            min_mask = labels == minority
            maj_mask = ~min_mask
            noisy[min_mask] = np.where(
                rng.random(min_mask.sum()) < sigma * 0.6,
                1.0 - noisy[min_mask], noisy[min_mask])
            noisy[maj_mask] = np.where(
                rng.random(maj_mask.sum()) < sigma * 0.3,
                1.0 - noisy[maj_mask], noisy[maj_mask])

        elif self.strategy == 'quantile':
            # Flip a fixed fraction of randomly chosen labels
            n_flip = int(n * sigma * 0.5)
            idx = rng.permutation(n)[:n_flip]
            noisy[idx] = 1.0 - noisy[idx]

        elif self.strategy == 'hetero':
            # Class-conditional: positives flip faster
            pos_mask = labels == 1
            neg_mask = ~pos_mask
            noisy[pos_mask] = np.where(
                rng.random(pos_mask.sum()) < sigma * 0.6,
                0.0, noisy[pos_mask])
            noisy[neg_mask] = np.where(
                rng.random(neg_mask.sum()) < sigma * 0.4,
                1.0, noisy[neg_mask])

        elif self.strategy == 'threshold':
            # No noise until sigma > 0.3, then ramps up
            if sigma > 0.3:
                flip_prob = (sigma - 0.3) / 0.7 * 0.5
                flip_mask = rng.random(n) < flip_prob
                noisy[flip_mask] = 1.0 - noisy[flip_mask]

        elif self.strategy == 'valprop':
            # Noise scaled by class prevalence
            pos_rate = max(labels.mean(), 0.01)
            neg_rate = 1.0 - pos_rate
            pos_mask = labels == 1
            neg_mask = ~pos_mask
            noisy[pos_mask] = np.where(
                rng.random(pos_mask.sum()) < sigma * neg_rate * 0.5,
                0.0, noisy[pos_mask])
            noisy[neg_mask] = np.where(
                rng.random(neg_mask.sum()) < sigma * pos_rate * 0.5,
                1.0, noisy[neg_mask])

        return noisy.astype(labels.dtype)


def get_classification_injector(strategy, random_state=42):
    """Return a classification noise injector (prefer library, fall back)."""
    if HAS_NOISE_CLASSIFICATION:
        return NoiseInjectorClassification(strategy=strategy,
                                           random_state=random_state)
    return _ClassificationNoiseInjector(strategy=strategy,
                                        random_state=random_state)


# ═══════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK MODELS
# ═══════════════════════════════════════════════════════════════════════════

class DeterministicRegressor(nn.Module):
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


# ═══════════════════════════════════════════════════════════════════════════
# BAYESIAN TRANSFORMATIONS (identical to DRD2 script)
# ═══════════════════════════════════════════════════════════════════════════

if HAS_BAYESIAN_TORCH:
    def apply_bayesian_transformation(model):
        transform_model(
            model, nn.Linear, bnn.BayesLinear,
            args={"prior_mu": 0, "prior_sigma": 0.1,
                  "in_features": ".in_features",
                  "out_features": ".out_features", "bias": ".bias"},
            attrs={"weight_mu": ".weight"})
        return model

    def apply_bayesian_transformation_last_layer(model):
        last_name, last_mod = None, None
        for name, mod in reversed(list(model.named_modules())):
            if isinstance(mod, nn.Linear):
                last_name, last_mod = name, mod
                break
        if last_mod is None:
            raise ValueError("No nn.Linear found")
        bl = transform_layer(
            last_mod, nn.Linear, bnn.BayesLinear,
            args={"prior_mu": 0, "prior_sigma": 0.1,
                  "in_features": ".in_features",
                  "out_features": ".out_features", "bias": ".bias"},
            attrs={"weight_mu": ".weight"})

        def _set(obj, path, val):
            parts = path.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)

        _set(model, last_name, bl)
        return model

    # var-bnn uses same transformation as last-layer-bnn in the DRD2 script
    apply_bayesian_transformation_variational = apply_bayesian_transformation_last_layer


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def train_neural_regression(X_train, y_train, X_val, y_val, X_test,
                            model_type='deterministic', epochs=100, lr=1e-3):
    """Train a regression neural network (deterministic or Bayesian).
    Returns (predictions, uncertainties_or_None)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    model = DeterministicRegressor(X_train.shape[1]).to(device)

    if HAS_BAYESIAN_TORCH and model_type != 'deterministic':
        if model_type == 'full-bnn':
            model = apply_bayesian_transformation(model)
        elif model_type == 'last-layer-bnn':
            model = apply_bayesian_transformation_last_layer(model)
        elif model_type == 'var-bnn':
            model = apply_bayesian_transformation_variational(model)

    is_bayesian = model_type != 'deterministic'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    best_val, patience_ctr, patience = float('inf'), 0, 10
    best_state = None

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_v), y_v).item()
        if vl < best_val:
            best_val = vl
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    if is_bayesian:
        preds = np.array([model(X_te).detach().cpu().numpy() for _ in range(30)])
        return preds.mean(0), preds.std(0)
    with torch.no_grad():
        return model(X_te).cpu().numpy(), None


def train_neural_classification(X_train, y_train, X_val, y_val, X_test,
                                model_type='deterministic', epochs=100, lr=1e-3):
    """Train a binary classification neural network.
    Returns (predicted_labels, predicted_probabilities, uncertainties_or_None)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    model = DeterministicRegressor(X_train.shape[1]).to(device)  # outputs logit

    if HAS_BAYESIAN_TORCH and model_type != 'deterministic':
        if model_type == 'full-bnn':
            model = apply_bayesian_transformation(model)
        elif model_type == 'last-layer-bnn':
            model = apply_bayesian_transformation_last_layer(model)
        elif model_type == 'var-bnn':
            model = apply_bayesian_transformation_variational(model)

    is_bayesian = model_type != 'deterministic'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    best_val, patience_ctr, patience = float('inf'), 0, 10
    best_state = None

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_v), y_v).item()
        if vl < best_val:
            best_val = vl
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    if is_bayesian:
        logits_all = np.array(
            [model(X_te).detach().cpu().numpy() for _ in range(30)])
        mean_logit = logits_all.mean(0)
        probs = 1.0 / (1.0 + np.exp(-mean_logit))
        preds = (probs >= 0.5).astype(int)
        uncs = logits_all.std(0)
        return preds, probs, uncs

    with torch.no_grad():
        logits = model(X_te).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return preds, probs, None


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def run_regression_tree_experiment(X_train, y_train, X_test, y_test,
                                   model_fn, strategy, sigma_levels):
    """Noise robustness for a tree-based regression model."""
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions, uncertainties = {}, {}

    for sigma in sigma_levels:
        y_noisy = y_train if sigma == 0.0 else injector.inject(y_train, sigma)
        mdl = model_fn()
        mdl.fit(X_train, y_noisy)

        if 'Quantile' in str(type(mdl)):
            q16, q50, q84 = mdl.predict(X_test, quantiles=[0.16, 0.5, 0.84]).T
            predictions[sigma] = q50
            uncertainties[sigma] = (q84 - q16) / 2
        elif 'GaussianProcess' in str(type(mdl)):
            pm, ps = mdl.predict(X_test, return_std=True)
            predictions[sigma] = pm
            uncertainties[sigma] = ps
        else:
            predictions[sigma] = mdl.predict(X_test)
            uncertainties[sigma] = None

    return predictions, uncertainties


def run_regression_neural_experiment(X_train, y_train, X_val, y_val,
                                     X_test, y_test,
                                     model_type, strategy, sigma_levels):
    """Noise robustness for a neural regression model."""
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    predictions, uncertainties = {}, {}

    for sigma in sigma_levels:
        y_noisy = y_train if sigma == 0.0 else injector.inject(y_train, sigma)
        preds, uncs = train_neural_regression(
            X_train, y_noisy, X_val, y_val, X_test,
            model_type=model_type, epochs=100)
        predictions[sigma] = preds
        uncertainties[sigma] = uncs

    return predictions, uncertainties


def run_classification_tree_experiment(X_train, y_train, X_test, y_test,
                                       model_fn, strategy, sigma_levels):
    """Noise robustness for a tree-based classification model."""
    injector = get_classification_injector(strategy, random_state=42)
    predictions, probabilities, uncertainties = {}, {}, {}

    for sigma in sigma_levels:
        y_noisy = y_train if sigma == 0.0 else injector.inject(y_train, sigma)
        mdl = model_fn()
        mdl.fit(X_train, y_noisy)

        preds = mdl.predict(X_test)
        predictions[sigma] = preds

        if hasattr(mdl, 'predict_proba'):
            probabilities[sigma] = mdl.predict_proba(X_test)[:, 1]
        elif 'GaussianProcess' in str(type(mdl)):
            probabilities[sigma] = mdl.predict_proba(X_test)[:, 1]
        else:
            probabilities[sigma] = preds.astype(float)

        uncertainties[sigma] = None

    return predictions, probabilities, uncertainties


def run_classification_neural_experiment(X_train, y_train, X_val, y_val,
                                          X_test, y_test,
                                          model_type, strategy, sigma_levels):
    """Noise robustness for a neural classification model."""
    injector = get_classification_injector(strategy, random_state=42)
    predictions, probabilities, uncertainties = {}, {}, {}

    for sigma in sigma_levels:
        y_noisy = y_train if sigma == 0.0 else injector.inject(y_train, sigma)
        preds, probs, uncs = train_neural_classification(
            X_train, y_noisy.astype(np.float32), X_val, y_val.astype(np.float32),
            X_test, model_type=model_type, epochs=100)
        predictions[sigma] = preds
        probabilities[sigma] = probs
        uncertainties[sigma] = uncs

    return predictions, probabilities, uncertainties


# ═══════════════════════════════════════════════════════════════════════════
# METRICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def compute_regression_metrics(y_true, predictions_dict):
    """Per-sigma regression metrics DataFrame."""
    rows = []
    for sigma in sorted(predictions_dict):
        yp = predictions_dict[sigma]
        rows.append({
            'sigma': sigma,
            'r2': r2_score(y_true, yp),
            'rmse': np.sqrt(mean_squared_error(y_true, yp)),
            'mae': mean_absolute_error(y_true, yp),
            'spearman': spearmanr(y_true, yp).correlation
                        if np.std(yp) > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def compute_classification_metrics(y_true, predictions_dict, probabilities_dict):
    """Per-sigma classification metrics DataFrame."""
    rows = []
    for sigma in sorted(predictions_dict):
        yp = predictions_dict[sigma]
        yprob = probabilities_dict.get(sigma)
        try:
            auc = roc_auc_score(y_true, yprob) if yprob is not None else np.nan
        except ValueError:
            auc = np.nan
        rows.append({
            'sigma': sigma,
            'accuracy': accuracy_score(y_true, yp),
            'auc': auc,
            'f1': f1_score(y_true, yp, zero_division=0),
            'mcc': matthews_corrcoef(y_true, yp),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# REPRESENTATION GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_representations(train_smiles, val_smiles, test_smiles):
    """Generate all four representations for a dataset.

    Returns dict mapping rep_name -> (X_train, X_val, X_test).
    """
    reps = {}

    print("  ECFP4...", flush=True)
    reps['ECFP4'] = (
        create_ecfp4(train_smiles, n_bits=2048),
        create_ecfp4(val_smiles, n_bits=2048),
        create_ecfp4(test_smiles, n_bits=2048),
    )
    print(f"    done: {reps['ECFP4'][0].shape}", flush=True)

    print("  PDV...", flush=True)
    reps['PDV'] = (
        create_pdv(train_smiles),
        create_pdv(val_smiles),
        create_pdv(test_smiles),
    )
    print(f"    done: {reps['PDV'][0].shape}", flush=True)

    print("  SNS...", flush=True)
    sns_train, sns_feat = create_sns(train_smiles, return_featurizer=True)
    sns_val = create_sns(val_smiles, reference_featurizer=sns_feat)
    sns_test = create_sns(test_smiles, reference_featurizer=sns_feat)
    reps['SNS'] = (sns_train, sns_val, sns_test)
    print(f"    done: {sns_train.shape}", flush=True)

    print("  MHG-GNN (pretrained)...", flush=True)
    try:
        reps['MHG-GNN-pretrained'] = (
            create_mhg_gnn(train_smiles),
            create_mhg_gnn(val_smiles),
            create_mhg_gnn(test_smiles),
        )
        print(f"    done: {reps['MHG-GNN-pretrained'][0].shape}", flush=True)
    except Exception as e:
        print(f"    FAILED: {e} — skipping MHG-GNN", flush=True)

    return reps


# ═══════════════════════════════════════════════════════════════════════════
# BUILD EXPERIMENT LIST
# ═══════════════════════════════════════════════════════════════════════════

def build_regression_experiments(reps, train_labels, val_labels, test_labels):
    """Return list of (model_name, rep_name, X_train, X_val, X_test,
                        model_fn_or_None, neural_type_or_None)."""
    exps = []

    for rname, (Xtr, Xv, Xte) in reps.items():
        # RF
        exps.append(('RF', rname, Xtr, None, Xte,
                      lambda: RandomForestRegressor(n_estimators=100,
                                                    random_state=42, n_jobs=-1),
                      None))
        # QRF
        if HAS_QRF:
            exps.append(('QRF', rname, Xtr, None, Xte,
                          lambda: RandomForestQuantileRegressor(
                              n_estimators=100, random_state=42, n_jobs=-1),
                          None))
        # XGBoost
        if HAS_XGBOOST:
            exps.append(('XGBoost', rname, Xtr, None, Xte,
                          lambda: XGBRegressor(n_estimators=100,
                                               random_state=42),
                          None))
        # GP — PDV only, subsample if large
        if HAS_GP and rname == 'PDV':
            _Xtr = Xtr
            _y = train_labels
            if len(Xtr) > GP_MAX_N:
                idx = np.random.RandomState(42).choice(
                    len(Xtr), GP_MAX_N, replace=False)
                _Xtr = Xtr[idx]
                _y = train_labels[idx]
            exps.append(('GP', rname, _Xtr, None, Xte,
                          lambda: GaussianProcessRegressor(
                              kernel=ConstantKernel(1.0) * RBF(1.0)
                                     + WhiteKernel(noise_level=1.0),
                              alpha=1e-10, random_state=42,
                              n_restarts_optimizer=5),
                          None))

        # Neural models
        for mtype, mname in [('deterministic', 'DNN'),
                              ('full-bnn', 'Full-BNN'),
                              ('last-layer-bnn', 'LastLayer-BNN'),
                              ('var-bnn', 'Var-BNN')]:
            if mtype != 'deterministic' and not HAS_BAYESIAN_TORCH:
                continue
            exps.append((mname, rname, Xtr, Xv, Xte, None, mtype))

    return exps


def build_classification_experiments(reps, train_labels, val_labels,
                                      test_labels):
    """Same structure as regression but with classifiers."""
    exps = []

    for rname, (Xtr, Xv, Xte) in reps.items():
        # RF (classifier)
        exps.append(('RF', rname, Xtr, None, Xte,
                      lambda: RandomForestClassifier(
                          n_estimators=100, random_state=42,
                          n_jobs=-1, class_weight='balanced'),
                      None))
        # XGBoost (classifier)
        if HAS_XGBOOST:
            scale = ((train_labels == 0).sum() /
                     max((train_labels == 1).sum(), 1))
            exps.append(('XGBoost', rname, Xtr, None, Xte,
                          lambda s=scale: XGBClassifier(
                              n_estimators=100, random_state=42,
                              scale_pos_weight=s),
                          None))
        # GP classifier — PDV only, subsample
        if HAS_GP and rname == 'PDV':
            _Xtr = Xtr
            _y = train_labels
            if len(Xtr) > GP_MAX_N:
                idx = np.random.RandomState(42).choice(
                    len(Xtr), GP_MAX_N, replace=False)
                _Xtr = Xtr[idx]
                _y = train_labels[idx]
            exps.append(('GP', rname, _Xtr, None, Xte,
                          lambda: GaussianProcessClassifier(
                              kernel=ConstantKernel(1.0) * RBF(1.0),
                              random_state=42, n_restarts_optimizer=3),
                          None))

        # Neural models
        for mtype, mname in [('deterministic', 'DNN'),
                              ('full-bnn', 'Full-BNN'),
                              ('last-layer-bnn', 'LastLayer-BNN'),
                              ('var-bnn', 'Var-BNN')]:
            if mtype != 'deterministic' and not HAS_BAYESIAN_TORCH:
                continue
            exps.append((mname, rname, Xtr, Xv, Xte, None, mtype))

    return exps


# ═══════════════════════════════════════════════════════════════════════════
# DATASET RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_dataset(dataset_name, task_type, data, results_dir):
    """Run the full noise robustness experiment matrix for one dataset.

    Args:
        dataset_name: e.g. 'OpenADMET-LogD'
        task_type: 'regression' or 'classification'
        data: dict with train_smiles, train_labels, val_smiles, val_labels,
              test_smiles, test_labels
        results_dir: Path for output CSVs

    Returns:
        summary_df with NDS values
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_smiles = data['train_smiles']
    train_labels = data['train_labels']
    val_smiles = data['val_smiles']
    val_labels = data['val_labels']
    test_smiles = data['test_smiles']
    test_labels = data['test_labels']

    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name} ({task_type})")
    print(f"{'='*80}")
    print(f"  Train: {len(train_smiles)}, Val: {len(val_smiles)}, "
          f"Test: {len(test_smiles)}")
    if task_type == 'classification':
        print(f"  Train blockers: {train_labels.sum()} "
              f"({100*train_labels.mean():.1f}%)")
        print(f"  Test  blockers: {test_labels.sum()} "
              f"({100*test_labels.mean():.1f}%)")
    else:
        print(f"  Train y range: [{train_labels.min():.2f}, "
              f"{train_labels.max():.2f}], std={train_labels.std():.3f}")
        print(f"  Test  y range: [{test_labels.min():.2f}, "
              f"{test_labels.max():.2f}], std={test_labels.std():.3f}")

    # ── Generate representations ──────────────────────────────────────
    print(f"\nGENERATING REPRESENTATIONS for {dataset_name}")
    reps = generate_representations(train_smiles, val_smiles, test_smiles)

    # ── Build experiment list ─────────────────────────────────────────
    if task_type == 'regression':
        experiments = build_regression_experiments(
            reps, train_labels, val_labels, test_labels)
    else:
        experiments = build_classification_experiments(
            reps, train_labels, val_labels, test_labels)

    print(f"\n{len(experiments)} model-rep configs × {len(STRATEGIES)} strategies "
          f"= {len(experiments) * len(STRATEGIES)} experiment runs")

    # ── Run experiments ───────────────────────────────────────────────
    all_per_sigma = []
    all_uncertainties = []

    for idx, (model_name, rep_name, X_train, X_val_rep, X_test,
              model_fn, model_type) in enumerate(experiments, 1):

        print(f"\n[{idx}/{len(experiments)}] {model_name} + {rep_name}...",
              flush=True)

        # For GP subsampled datasets, need matching labels
        if model_name == 'GP' and len(X_train) < len(train_labels):
            # GP was subsampled; reconstruct matching labels
            gp_n = len(X_train)
            gp_idx = np.random.RandomState(42).choice(
                len(train_labels), gp_n, replace=False)
            y_train_for_exp = train_labels[gp_idx]
        else:
            y_train_for_exp = train_labels

        for strategy in STRATEGIES:
            print(f"  Strategy: {strategy}", flush=True)

            try:
                if task_type == 'regression':
                    # ── Regression experiment ─────────────────────────
                    if model_fn is not None:
                        predictions, uncertainties = \
                            run_regression_tree_experiment(
                                X_train, y_train_for_exp, X_test,
                                test_labels, model_fn, strategy,
                                SIGMA_LEVELS)
                    else:
                        predictions, uncertainties = \
                            run_regression_neural_experiment(
                                X_train, y_train_for_exp,
                                X_val_rep, val_labels,
                                X_test, test_labels,
                                model_type, strategy, SIGMA_LEVELS)

                    per_sigma = compute_regression_metrics(
                        test_labels, predictions)

                else:
                    # ── Classification experiment ─────────────────────
                    if model_fn is not None:
                        predictions, probabilities, uncertainties = \
                            run_classification_tree_experiment(
                                X_train, y_train_for_exp, X_test,
                                test_labels, model_fn, strategy,
                                SIGMA_LEVELS)
                    else:
                        predictions, probabilities, uncertainties = \
                            run_classification_neural_experiment(
                                X_train, y_train_for_exp,
                                X_val_rep, val_labels,
                                X_test, test_labels,
                                model_type, strategy, SIGMA_LEVELS)

                    per_sigma = compute_classification_metrics(
                        test_labels, predictions, probabilities)

                per_sigma['model'] = model_name
                per_sigma['rep'] = rep_name
                per_sigma['strategy'] = strategy
                per_sigma['dataset'] = dataset_name
                all_per_sigma.append(per_sigma)

                # Save per-experiment CSV
                fname = (f"{model_name.replace('-', '')}_"
                         f"{rep_name.replace('-', '')}_{strategy}.csv")
                per_sigma.to_csv(results_dir / fname, index=False)

                # Save uncertainty data for legacy strategy
                if (strategy == 'legacy' and
                        uncertainties.get(0.0) is not None):
                    unc_rows = []
                    for sigma in SIGMA_LEVELS:
                        for i in range(len(test_labels)):
                            unc_rows.append({
                                'sigma': sigma,
                                'sample_idx': i,
                                'y_true': test_labels[i],
                                'y_pred': (predictions[sigma][i]
                                           if task_type == 'regression'
                                           else probabilities[sigma][i]),
                                'uncertainty': uncertainties[sigma][i],
                            })
                    unc_df = pd.DataFrame(unc_rows)
                    ufname = (f"{model_name.replace('-', '')}_"
                              f"{rep_name.replace('-', '')}"
                              f"_uncertainty_values.csv")
                    unc_df.to_csv(results_dir / ufname, index=False)
                    all_uncertainties.append(
                        (model_name, rep_name, unc_df))

            except Exception as e:
                print(f"  ERROR in {model_name}+{rep_name}+{strategy}: "
                      f"{e}", flush=True)
                continue

    if not all_per_sigma:
        print(f"ERROR: No results for {dataset_name}")
        return pd.DataFrame()

    # ── Combine and compute NDS ───────────────────────────────────────
    combined = pd.concat(all_per_sigma, ignore_index=True)
    combined.to_csv(results_dir / 'all_results.csv', index=False)

    # Primary metric for NDS
    metric_col = 'r2' if task_type == 'regression' else 'auc'
    degrade_col = 'rmse' if task_type == 'regression' else 'f1'

    summary_rows = []
    for (model, rep, strat), grp in combined.groupby(
            ['model', 'rep', 'strategy']):
        grp = grp.sort_values('sigma')
        sigmas = grp['sigma'].values
        primary = grp[metric_col].values
        secondary = grp[degrade_col].values

        baseline_primary = primary[0]
        baseline_secondary = secondary[0]

        sl_p, _, rv, _, _ = linregress(sigmas, primary)
        sl_s, _, _, _, _ = linregress(sigmas, secondary)

        summary_rows.append({
            'dataset': dataset_name,
            'task': task_type,
            'model': model,
            'rep': rep,
            'strategy': strat,
            f'baseline_{metric_col}': baseline_primary,
            f'baseline_{degrade_col}': baseline_secondary,
            f'NDS_{metric_col}': sl_p,
            f'NDS_{degrade_col}': sl_s,
            'r2_fit': rv ** 2,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / 'summary.csv', index=False)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"NDS SUMMARY: {dataset_name}")
    print(f"{'─'*80}")
    print(f"  NDS_{metric_col}: slope of {metric_col} vs σ "
          f"(more negative = faster degradation)")
    print(f"  baseline_{metric_col}: performance at σ=0")

    print(f"\n  Top 10 most robust (NDS_{metric_col}, least negative):")
    top = summary_df.nlargest(10, f'NDS_{metric_col}')[
        ['model', 'rep', 'strategy',
         f'baseline_{metric_col}', f'NDS_{metric_col}']]
    print(top.to_string(index=False))

    print(f"\n  By representation (mean across models/strategies):")
    rep_agg = summary_df.groupby('rep')[
        [f'baseline_{metric_col}', f'NDS_{metric_col}']].mean()
    print(rep_agg.to_string())

    print(f"\n  By model (mean across reps/strategies):")
    mod_agg = summary_df.groupby('model')[
        [f'baseline_{metric_col}', f'NDS_{metric_col}']].mean()
    print(mod_agg.to_string())

    print(f"\n  By strategy (mean across models/reps):")
    strat_agg = summary_df.groupby('strategy')[
        [f'baseline_{metric_col}', f'NDS_{metric_col}']].mean()
    print(strat_agg.to_string())

    # ── Uncertainty summary ───────────────────────────────────────────
    if all_uncertainties:
        print(f"\n  Uncertainty-Error Correlations (legacy strategy):")
        for mname, rname, udf in all_uncertainties:
            print(f"    {mname} + {rname}:")
            for sigma in SIGMA_LEVELS:
                sub = udf[udf['sigma'] == sigma]
                if len(sub) > 0 and sub['uncertainty'].std() > 0:
                    corr = np.corrcoef(
                        sub['uncertainty'],
                        np.abs(sub['y_true'] - sub['y_pred']))[0, 1]
                    print(f"      σ={sigma:.1f}: ρ = {corr:.4f}")

    return summary_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Baseline Noise Robustness: OpenADMET + FLuID hERG')
    parser.add_argument('--datasets', nargs='+',
                        choices=['logd', 'caco2', 'herg', 'all'],
                        default=['all'],
                        help='Which datasets to test (default: all)')
    parser.add_argument('--openadmet-csv', type=str, default=None,
                        help='Path to cached OpenADMET CSV '
                             '(skips download)')
    parser.add_argument('--results-root', type=str, default='results',
                        help='Root directory for results')
    args = parser.parse_args()

    ds_list = (['logd', 'caco2', 'herg'] if 'all' in args.datasets
               else args.datasets)

    print("=" * 80)
    print("Baseline Noise Robustness Experiment")
    print("=" * 80)
    print(f"  Datasets: {ds_list}")
    print(f"  Representations: ECFP4, PDV, SNS, MHG-GNN-pretrained")
    print(f"  Noise strategies: {STRATEGIES}")
    print(f"  Sigma levels: {SIGMA_LEVELS}")
    print(f"  Models: RF, QRF, XGBoost, GP(PDV), DNN"
          + (", Full-BNN, LastLayer-BNN, Var-BNN" if HAS_BAYESIAN_TORCH
             else " (BNNs skipped — no bayesian-torch)"))
    print()

    all_summaries = []

    # ── OpenADMET data (shared download) ──────────────────────────────
    openadmet_df = None
    if 'logd' in ds_list or 'caco2' in ds_list:
        print("Downloading/loading OpenADMET-ExpansionRx dataset...")
        openadmet_df = download_openadmet(csv_path=args.openadmet_csv)
        print(f"  Shape: {openadmet_df.shape}")
        print(f"  Columns: {list(openadmet_df.columns)}")

    # ── 1. OpenADMET-LogD ─────────────────────────────────────────────
    if 'logd' in ds_list and openadmet_df is not None:
        # Find LogD column
        logd_col = next((c for c in openadmet_df.columns if 'LogD' in c),
                        None)
        if logd_col is None:
            print("ERROR: Cannot find LogD column in OpenADMET data")
        else:
            print(f"\nPreparing OpenADMET-LogD (column: {logd_col})...")
            n_valid = openadmet_df[logd_col].notna().sum()
            print(f"  {n_valid} molecules with LogD values")

            data = load_openadmet_endpoint(openadmet_df, logd_col,
                                            log_transform=False)
            summary = run_dataset(
                'OpenADMET-LogD', 'regression', data,
                Path(args.results_root) / 'openadmet_logd')
            all_summaries.append(summary)

    # ── 2. OpenADMET-Caco2 Efflux ────────────────────────────────────
    if 'caco2' in ds_list and openadmet_df is not None:
        # Find Caco-2 Efflux column
        caco2_col = next(
            (c for c in openadmet_df.columns
             if 'Caco' in c and 'Efflux' in c), None)
        if caco2_col is None:
            print("ERROR: Cannot find Caco-2 Efflux column in OpenADMET data")
        else:
            print(f"\nPreparing OpenADMET-Caco2_Efflux (column: {caco2_col})...")
            n_valid = openadmet_df[caco2_col].notna().sum()
            print(f"  {n_valid} molecules with Caco-2 Efflux values")

            data = load_openadmet_endpoint(openadmet_df, caco2_col,
                                            log_transform=True)
            summary = run_dataset(
                'OpenADMET-Caco2_Efflux', 'regression', data,
                Path(args.results_root) / 'openadmet_caco2')
            all_summaries.append(summary)

    # ── 3. FLuID hERG ─────────────────────────────────────────────────
    if 'herg' in ds_list:
        if not HAS_HERG:
            print("ERROR: kirby.datasets.herg not available — skipping hERG")
        else:
            print("\nPreparing FLuID hERG...")
            data = load_herg_fluid()
            summary = run_dataset(
                'hERG-FLuID', 'classification', data,
                Path(args.results_root) / 'herg_fluid')
            all_summaries.append(summary)

    # ═════════════════════════════════════════════════════════════════════
    # CROSS-DATASET COMPARISON
    # ═════════════════════════════════════════════════════════════════════
    if len(all_summaries) > 1:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_path = Path(args.results_root) / 'combined_summary.csv'
        combined_summary.to_csv(combined_path, index=False)

        print("\n" + "=" * 80)
        print("CROSS-DATASET COMPARISON")
        print("=" * 80)

        # Compare representations across datasets
        print("\nBy representation × dataset (mean NDS across models/strategies):")
        for ds_name, ds_grp in combined_summary.groupby('dataset'):
            task = ds_grp['task'].iloc[0]
            nds_col = 'NDS_r2' if task == 'regression' else 'NDS_auc'
            base_col = ('baseline_r2' if task == 'regression'
                        else 'baseline_auc')
            if nds_col not in ds_grp.columns:
                continue
            print(f"\n  {ds_name} ({task}):")
            agg = ds_grp.groupby('rep')[[base_col, nds_col]].mean()
            print(agg.to_string())

        # Compare models across datasets
        print("\nBy model × dataset (mean NDS across reps/strategies):")
        for ds_name, ds_grp in combined_summary.groupby('dataset'):
            task = ds_grp['task'].iloc[0]
            nds_col = 'NDS_r2' if task == 'regression' else 'NDS_auc'
            base_col = ('baseline_r2' if task == 'regression'
                        else 'baseline_auc')
            if nds_col not in ds_grp.columns:
                continue
            print(f"\n  {ds_name} ({task}):")
            agg = ds_grp.groupby('model')[[base_col, nds_col]].mean()
            print(agg.to_string())

    print("\n" + "=" * 80)
    print("COMPLETE — Results saved to", args.results_root)
    print("=" * 80)


if __name__ == '__main__':
    main()