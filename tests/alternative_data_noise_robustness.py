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
Validation Noise Robustness: 3 Regression Datasets with Scaffold CV
====================================================================

Tests noise robustness on three validated regression datasets that pass
the ECFP4+RF baseline (R² > 0.5).

Datasets (all REGRESSION):
  1. OpenADMET-LogD — 7309 molecules, lipophilicity
     Baseline: RF R²=0.69, LGBM R²=0.81

  2. OpenADMET-Caco2_Efflux — 3777 molecules, P-gp efflux ratio
     Baseline: RF R²=0.66, LGBM R²=0.70

  3. ChEMBL-hERG-Ki — 1415 molecules, hERG binding affinity (pKi)
     Baseline: RF R²=0.54, LGBM R²=0.56

Split: 5-fold scaffold CV (Murcko scaffolds)
  - Groups molecules by scaffold
  - Holds out entire scaffold groups per fold
  - Tests generalization to novel chemotypes

Model-Representation Matrix:
  Representations: ECFP4, PDV, SNS, MHG-GNN-pretrained (4 total)
  Models: RF, QRF, XGBoost, GP(PDV only), DNN, Full-BNN, LastLayer-BNN, Var-BNN (8 total)
  Total: 4 reps × 8 models × 6 strategies × 3 datasets × 5 folds

Noise Strategies (regression): legacy, outlier, quantile, hetero, threshold, valprop (6)
Sigma levels: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (11)

Usage:
  python alternative_data_noise_robustness.py
  python alternative_data_noise_robustness.py --datasets logd caco2 herg
  python alternative_data_noise_robustness.py --results-root results/validation
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import linregress, spearmanr
import sys
import time
import requests
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.logger().setLevel(RDLogger.ERROR)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

STRATEGIES = ['legacy', 'outlier', 'quantile', 'hetero', 'threshold', 'valprop']
SIGMA_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_FOLDS = 5
GP_MAX_N = 2000
CACHE_DIR = Path('data_cache')


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

try:
    import torchbnn as bnn
    from bayesian_torch.models.dnn_to_bnn import transform_model, transform_layer
    HAS_BAYESIAN_TORCH = True
except ImportError:
    print("WARNING: torchbnn or bayesian-torch not installed, BNN experiments will be skipped")
    HAS_BAYESIAN_TORCH = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    print("WARNING: xgboost not installed, XGBoost experiments will be skipped")
    HAS_XGBOOST = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
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

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_sns,
    create_mhg_gnn
)
from noiseInject import NoiseInjectorRegression


# ═══════════════════════════════════════════════════════════════════════════
# SMILES / SCAFFOLD UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def standardise_smiles(smi):
    """Canonicalise SMILES, keep largest fragment, remove salts."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_scaffold(smi):
    """Get Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return smi


def assign_scaffold_groups(smiles_list):
    """Assign scaffold group IDs to molecules."""
    scaffolds = [get_scaffold(smi) for smi in smiles_list]
    unique_scaffolds = list(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    groups = np.array([scaffold_to_id[s] for s in scaffolds])
    return groups, len(unique_scaffolds)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def download_openadmet(csv_path=None):
    """Download or load OpenADMET-ExpansionRx data."""
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


def fetch_chembl_herg_ki():
    """Extract hERG (CHEMBL240) Ki data via ChEMBL REST API."""
    print("  Fetching ChEMBL hERG Ki data...")

    cached = CACHE_DIR / "chembl_herg_ki.csv"
    if cached.exists():
        print(f"  [cached] {cached}")
        return pd.read_csv(cached)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_ID = "CHEMBL240"
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

    all_records = []
    offset = 0
    limit = 1000

    while True:
        params = {
            "target_chembl_id": TARGET_ID,
            "standard_type": "Ki",
            "pchembl_value__isnull": "false",
            "standard_relation": "=",
            "data_validity_comment__isnull": "true",
            "limit": limit,
            "offset": offset,
            "format": "json",
        }

        print(f"    Querying ChEMBL API (offset={offset})...")
        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"    ChEMBL API request failed: {e}")
            break

        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            all_records.append({
                "canonical_smiles": act.get("canonical_smiles"),
                "pchembl_value": act.get("pchembl_value"),
                "assay_type": act.get("assay_type"),
            })

        page_meta = data.get("page_meta", {})
        if page_meta.get("next") is None:
            break
        offset += limit
        time.sleep(0.5)

    if not all_records:
        raise RuntimeError("No hERG Ki records retrieved from ChEMBL!")

    df = pd.DataFrame(all_records)
    print(f"    Retrieved {len(df)} raw hERG Ki records")

    # Filter to binding assays
    if "assay_type" in df.columns:
        df = df[df["assay_type"] == "B"].copy()

    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value", "canonical_smiles"])

    # Deduplicate: median pchembl per compound
    grouped = df.groupby("canonical_smiles")["pchembl_value"]
    medians = grouped.median().reset_index()
    stds = grouped.std().reset_index().rename(columns={"pchembl_value": "std"})
    merged = medians.merge(stds, on="canonical_smiles")

    # Remove high-variance compounds (std > 1.0 log unit)
    merged = merged[(merged["std"].isna()) | (merged["std"] <= 1.0)]

    result = merged[["canonical_smiles", "pchembl_value"]].copy()
    result.columns = ["SMILES", "pKi"]

    result.to_csv(cached, index=False)
    print(f"    Final hERG Ki dataset: {len(result)} compounds")

    return result


def load_openadmet_endpoint(df, endpoint_col, log_transform=False):
    """Extract a single endpoint from OpenADMET, standardize, deduplicate."""
    smiles_col = next(c for c in df.columns if c.upper() == 'SMILES')

    sub = df[[smiles_col, endpoint_col]].dropna(subset=[endpoint_col]).copy()
    sub[endpoint_col] = pd.to_numeric(sub[endpoint_col], errors='coerce')
    sub = sub.dropna()

    # Standardize SMILES
    sub['std_smiles'] = sub[smiles_col].apply(standardise_smiles)
    sub = sub.dropna(subset=['std_smiles'])

    # Deduplicate by canonical SMILES: take median target
    sub = sub.groupby('std_smiles').agg({endpoint_col: 'median'}).reset_index()

    smiles_arr = sub['std_smiles'].values
    labels_arr = sub[endpoint_col].values.astype(np.float64)

    if log_transform:
        labels_arr = np.log10(np.clip(labels_arr, 1e-10, None))
        valid = np.isfinite(labels_arr)
        smiles_arr = smiles_arr[valid]
        labels_arr = labels_arr[valid]

    return smiles_arr, labels_arr


def load_chembl_herg():
    """Load ChEMBL hERG Ki as regression dataset."""
    df = fetch_chembl_herg_ki()

    # Standardize SMILES
    df['std_smiles'] = df['SMILES'].apply(standardise_smiles)
    df = df.dropna(subset=['std_smiles'])

    smiles_arr = df['std_smiles'].values
    labels_arr = df['pKi'].values.astype(np.float64)

    return smiles_arr, labels_arr


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

    apply_bayesian_transformation_variational = apply_bayesian_transformation_last_layer


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def train_neural_regression(X_train, y_train, X_val, y_val, X_test,
                            model_type='deterministic', epochs=100, lr=1e-3):
    """Train regression neural network. Returns (predictions, uncertainties)."""
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


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def run_tree_experiment(X_train, y_train, X_test, y_test, model_fn, strategy, sigma_levels):
    """Noise robustness for tree-based regression model."""
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


def run_neural_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                          model_type, strategy, sigma_levels):
    """Noise robustness for neural regression model."""
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


def compute_metrics(y_true, predictions_dict):
    """Per-sigma regression metrics DataFrame."""
    rows = []
    for sigma in sorted(predictions_dict):
        yp = predictions_dict[sigma]
        rows.append({
            'sigma': sigma,
            'r2': r2_score(y_true, yp),
            'rmse': np.sqrt(mean_squared_error(y_true, yp)),
            'mae': mean_absolute_error(y_true, yp),
            'spearman': spearmanr(y_true, yp).correlation if np.std(yp) > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# REPRESENTATION GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_representations(smiles_list):
    """Generate all representations for a SMILES list. Returns dict of arrays."""
    import gc
    reps = {}

    print("  ECFP4...", flush=True)
    reps['ECFP4'] = create_ecfp4(smiles_list, n_bits=2048)
    print(f"    done: {reps['ECFP4'].shape}", flush=True)

    print("  PDV...", flush=True)
    reps['PDV'] = create_pdv(smiles_list)
    print(f"    done: {reps['PDV'].shape}", flush=True)

    print("  SNS...", flush=True)
    reps['SNS'], _ = create_sns(smiles_list, return_featurizer=True)
    print(f"    done: {reps['SNS'].shape}", flush=True)

    print("  MHG-GNN (pretrained)...", flush=True)
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reps['MHG-GNN-pretrained'] = create_mhg_gnn(smiles_list, batch_size=32)
        print(f"    done: {reps['MHG-GNN-pretrained'].shape}", flush=True)
    except Exception as e:
        print(f"    FAILED: {e} — skipping MHG-GNN", flush=True)

    return reps


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_dataset(dataset_name, smiles, labels, results_dir):
    """Run full scaffold CV experiment for one dataset."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f"  N molecules: {len(smiles)}")
    print(f"  y range: [{labels.min():.3f}, {labels.max():.3f}], std={labels.std():.3f}")

    # Assign scaffold groups
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  N scaffolds: {n_scaffolds}")

    # Generate representations for ALL molecules
    print(f"\nGenerating representations for {len(smiles)} molecules...")
    reps = generate_representations(smiles)

    # Build experiment configs
    experiments = []
    for rname in reps.keys():
        # RF
        experiments.append(('RF', rname,
            lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), None))
        # QRF
        if HAS_QRF:
            experiments.append(('QRF', rname,
                lambda: RandomForestQuantileRegressor(n_estimators=100, random_state=42, n_jobs=-1), None))
        # XGBoost
        if HAS_XGBOOST:
            experiments.append(('XGBoost', rname,
                lambda: XGBRegressor(n_estimators=100, random_state=42), None))
        # GP (PDV only)
        if HAS_GP and rname == 'PDV':
            experiments.append(('GP', rname,
                lambda: GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(noise_level=1.0),
                    alpha=1e-10, random_state=42, n_restarts_optimizer=5), None))
        # Neural models
        for mtype, mname in [('deterministic', 'DNN'),
                              ('full-bnn', 'Full-BNN'),
                              ('last-layer-bnn', 'LastLayer-BNN'),
                              ('var-bnn', 'Var-BNN')]:
            if mtype != 'deterministic' and not HAS_BAYESIAN_TORCH:
                continue
            experiments.append((mname, rname, None, mtype))

    print(f"\n{len(experiments)} model-rep configs × {len(STRATEGIES)} strategies × {N_FOLDS} folds")

    # Scaffold CV
    gkf = GroupKFold(n_splits=N_FOLDS)
    all_fold_results = []
    all_uncertainties = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(smiles, labels, groups)):
        print(f"\n{'─'*40}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'─'*40}")
        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

        # Split labels
        y_train_full = labels[train_idx]
        y_test = labels[test_idx]

        # Carve validation from train (20% of train)
        n_val = len(train_idx) // 5
        val_idx_local = np.arange(n_val)
        train_idx_local = np.arange(n_val, len(train_idx))
        y_train = y_train_full[train_idx_local]
        y_val = y_train_full[val_idx_local]

        # Run experiments
        for exp_idx, (model_name, rep_name, model_fn, model_type) in enumerate(experiments, 1):
            print(f"\n  [{exp_idx}/{len(experiments)}] {model_name} + {rep_name}...", flush=True)

            # Get representation splits
            X_full = reps[rep_name]
            X_train_full = X_full[train_idx]
            X_test = X_full[test_idx]
            X_train = X_train_full[train_idx_local]
            X_val = X_train_full[val_idx_local]

            # Subsample for GP if needed
            if model_name == 'GP' and len(X_train) > GP_MAX_N:
                gp_idx = np.random.RandomState(42).choice(len(X_train), GP_MAX_N, replace=False)
                X_train_gp = X_train[gp_idx]
                y_train_gp = y_train[gp_idx]
            else:
                X_train_gp = X_train
                y_train_gp = y_train

            for strategy in STRATEGIES:
                print(f"    Strategy: {strategy}", flush=True)

                try:
                    if model_fn is not None:
                        if model_name == 'GP':
                            predictions, uncertainties = run_tree_experiment(
                                X_train_gp, y_train_gp, X_test, y_test,
                                model_fn, strategy, SIGMA_LEVELS)
                        else:
                            predictions, uncertainties = run_tree_experiment(
                                X_train, y_train, X_test, y_test,
                                model_fn, strategy, SIGMA_LEVELS)
                    else:
                        predictions, uncertainties = run_neural_experiment(
                            X_train, y_train, X_val, y_val, X_test, y_test,
                            model_type, strategy, SIGMA_LEVELS)

                    per_sigma = compute_metrics(y_test, predictions)
                    per_sigma['model'] = model_name
                    per_sigma['rep'] = rep_name
                    per_sigma['strategy'] = strategy
                    per_sigma['fold'] = fold_idx
                    per_sigma['dataset'] = dataset_name
                    all_fold_results.append(per_sigma)

                    # Save uncertainty data (legacy strategy only)
                    if strategy == 'legacy' and uncertainties.get(0.0) is not None:
                        unc_rows = []
                        for sigma in SIGMA_LEVELS:
                            for i in range(len(y_test)):
                                unc_rows.append({
                                    'sigma': sigma,
                                    'sample_idx': i,
                                    'y_true': y_test[i],
                                    'y_pred': predictions[sigma][i],
                                    'uncertainty': uncertainties[sigma][i],
                                    'fold': fold_idx,
                                })
                        all_uncertainties.append((model_name, rep_name, pd.DataFrame(unc_rows)))

                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    continue

    if not all_fold_results:
        print(f"ERROR: No results for {dataset_name}")
        return pd.DataFrame()

    # Combine results across folds
    combined = pd.concat(all_fold_results, ignore_index=True)
    combined.to_csv(results_dir / 'all_results.csv', index=False)

    # Aggregate across folds and compute NDS
    summary_rows = []
    for (model, rep, strat), grp in combined.groupby(['model', 'rep', 'strategy']):
        # Average across folds first, then compute NDS
        fold_avgs = grp.groupby('sigma')[['r2', 'rmse', 'mae', 'spearman']].mean().reset_index()
        fold_avgs = fold_avgs.sort_values('sigma')

        sigmas = fold_avgs['sigma'].values
        r2_vals = fold_avgs['r2'].values
        rmse_vals = fold_avgs['rmse'].values

        baseline_r2 = r2_vals[0]
        baseline_rmse = rmse_vals[0]

        sl_r2, _, rv, _, _ = linregress(sigmas, r2_vals)
        sl_rmse, _, _, _, _ = linregress(sigmas, rmse_vals)

        # Retention at sigma=0.5 and 1.0
        r2_at_05 = fold_avgs[fold_avgs['sigma'] == 0.5]['r2'].values
        r2_at_10 = fold_avgs[fold_avgs['sigma'] == 1.0]['r2'].values
        retention_05 = r2_at_05[0] / baseline_r2 if len(r2_at_05) > 0 and baseline_r2 > 0 else np.nan
        retention_10 = r2_at_10[0] / baseline_r2 if len(r2_at_10) > 0 and baseline_r2 > 0 else np.nan

        # CV std of baseline R²
        baseline_std = grp[grp['sigma'] == 0.0]['r2'].std()

        summary_rows.append({
            'dataset': dataset_name,
            'model': model,
            'rep': rep,
            'strategy': strat,
            'baseline_r2': baseline_r2,
            'baseline_r2_std': baseline_std,
            'baseline_rmse': baseline_rmse,
            'NDS_r2': sl_r2,
            'NDS_rmse': sl_rmse,
            'retention_0.5': retention_05,
            'retention_1.0': retention_10,
            'r2_fit': rv ** 2,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / 'summary.csv', index=False)

    # Save uncertainty data
    for model_name, rep_name, unc_df in all_uncertainties:
        unc_combined = unc_df.groupby(['sigma', 'sample_idx']).agg({
            'y_true': 'first',
            'y_pred': 'mean',
            'uncertainty': 'mean',
        }).reset_index()
        fname = f"{model_name.replace('-', '')}_{rep_name.replace('-', '')}_uncertainty_values.csv"
        unc_combined.to_csv(results_dir / fname, index=False)

    # Print summary
    print(f"\n{'─'*80}")
    print(f"SUMMARY: {dataset_name}")
    print(f"{'─'*80}")

    working = summary_df[summary_df['baseline_r2'] >= 0.3]
    print(f"  Working configs (baseline R² >= 0.3): {len(working)}/{len(summary_df)}")

    if len(working) > 0:
        print(f"\n  Top 5 most robust (highest NDS):")
        top5 = working.nlargest(5, 'NDS_r2')
        for _, row in top5.iterrows():
            print(f"    {row['model']:12} + {row['rep']:20} + {row['strategy']:10}: "
                  f"NDS={row['NDS_r2']:.4f}, baseline={row['baseline_r2']:.3f}±{row['baseline_r2_std']:.3f}")

        print(f"\n  By representation (mean across models/strategies):")
        rep_agg = working.groupby('rep')[['baseline_r2', 'NDS_r2', 'retention_0.5']].mean()
        print(rep_agg.round(4).to_string())

        print(f"\n  By model (mean across reps/strategies):")
        mod_agg = working.groupby('model')[['baseline_r2', 'NDS_r2', 'retention_0.5']].mean()
        print(mod_agg.round(4).to_string())

    return summary_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Validation Noise Robustness: 3 Regression Datasets with Scaffold CV')
    parser.add_argument('--datasets', nargs='+',
                        choices=['logd', 'caco2', 'herg', 'all'],
                        default=['all'],
                        help='Which datasets to test (default: all)')
    parser.add_argument('--openadmet-csv', type=str, default=None,
                        help='Path to cached OpenADMET CSV')
    parser.add_argument('--results-root', type=str, default='results/validation',
                        help='Root directory for results')
    args = parser.parse_args()

    ds_list = ['logd', 'caco2', 'herg'] if 'all' in args.datasets else args.datasets

    print("=" * 80)
    print("Validation Noise Robustness Experiment")
    print("=" * 80)
    print(f"  Datasets: {ds_list}")
    print(f"  Split: {N_FOLDS}-fold scaffold CV")
    print(f"  Representations: ECFP4, PDV, SNS, MHG-GNN-pretrained")
    print(f"  Noise strategies: {STRATEGIES}")
    print(f"  Sigma levels: {SIGMA_LEVELS}")
    print(f"  Models: RF, QRF, XGBoost, GP(PDV), DNN"
          + (", Full-BNN, LastLayer-BNN, Var-BNN" if HAS_BAYESIAN_TORCH else ""))
    print()

    all_summaries = []

    # Load OpenADMET data if needed
    openadmet_df = None
    if 'logd' in ds_list or 'caco2' in ds_list:
        print("Loading OpenADMET-ExpansionRx dataset...")
        openadmet_df = download_openadmet(csv_path=args.openadmet_csv)
        print(f"  Shape: {openadmet_df.shape}")

    # 1. OpenADMET-LogD
    if 'logd' in ds_list and openadmet_df is not None:
        logd_col = next((c for c in openadmet_df.columns if 'LogD' in c), None)
        if logd_col:
            print(f"\nPreparing OpenADMET-LogD (column: {logd_col})...")
            smiles, labels = load_openadmet_endpoint(openadmet_df, logd_col, log_transform=False)
            print(f"  {len(smiles)} molecules")
            summary = run_dataset('OpenADMET-LogD', smiles, labels,
                                  Path(args.results_root) / 'logd')
            all_summaries.append(summary)
        else:
            print("ERROR: Cannot find LogD column")

    # 2. OpenADMET-Caco2_Efflux
    if 'caco2' in ds_list and openadmet_df is not None:
        caco2_col = next((c for c in openadmet_df.columns if 'Caco' in c and 'Efflux' in c), None)
        if caco2_col:
            print(f"\nPreparing OpenADMET-Caco2_Efflux (column: {caco2_col})...")
            smiles, labels = load_openadmet_endpoint(openadmet_df, caco2_col, log_transform=True)
            print(f"  {len(smiles)} molecules")
            summary = run_dataset('OpenADMET-Caco2_Efflux', smiles, labels,
                                  Path(args.results_root) / 'caco2')
            all_summaries.append(summary)
        else:
            print("ERROR: Cannot find Caco2 Efflux column")

    # 3. ChEMBL-hERG-Ki
    if 'herg' in ds_list:
        print("\nPreparing ChEMBL-hERG-Ki...")
        smiles, labels = load_chembl_herg()
        print(f"  {len(smiles)} molecules")
        summary = run_dataset('ChEMBL-hERG-Ki', smiles, labels,
                              Path(args.results_root) / 'herg')
        all_summaries.append(summary)

    # Cross-dataset summary
    if len(all_summaries) > 1:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(Path(args.results_root) / 'combined_summary.csv', index=False)

        print("\n" + "=" * 80)
        print("CROSS-DATASET SUMMARY")
        print("=" * 80)

        for ds in combined['dataset'].unique():
            ds_data = combined[combined['dataset'] == ds]
            working = ds_data[ds_data['baseline_r2'] >= 0.3]
            print(f"\n  {ds}: {len(working)}/{len(ds_data)} configs pass baseline >= 0.3")

    print("\n" + "=" * 80)
    print(f"COMPLETE — Results saved to {args.results_root}")
    print("=" * 80)


if __name__ == '__main__':
    main()
