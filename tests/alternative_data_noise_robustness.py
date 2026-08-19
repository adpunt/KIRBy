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
  Models: RF, QRF, XGBoost, NGBoost, LightGBM, SVM, GP(PDV only),
          DNN, BNN-Full, VBLL-Full,                    (NN-α family — DNN base)
          MLP, MLP-BNN-Full, MLP-VBLL-Full             (NN-β family — MLP base)
          (13 total; BNN-Last implemented but not run by default)
  Total: 4 reps × 13 models × 6 strategies × 3 datasets × 5 folds

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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
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
    # Use torchhk (same library as qsar_qm_models/models/models.py:28) so the
    # BayesLinear conversion is byte-for-byte identical between QM9 and
    # validation pipelines. bayesian_torch has a different transform_model
    # signature and was producing divergent BNN-α rows.
    from torchhk import transform_model, transform_layer
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
    # GP on PDV is implemented with gauche/gpytorch so the model exactly matches
    # qsar_qm_models/models/models.py:504 (the canonical Gauche ExactGP class).
    # This makes the validation 'GP' rows directly comparable to QM9 gauche_rbf
    # rows. Previously KIRBy used sklearn.GaussianProcessRegressor — functionally
    # equivalent (scaled RBF + learned observation noise) but a different library.
    import gpytorch
    import gauche  # noqa: F401  (verifies the gauche package is importable)
    try:
        from botorch.fit import fit_gpytorch_mll as _fit_gpytorch
    except ImportError:
        from botorch import fit_gpytorch_model as _fit_gpytorch
    HAS_GP = True
except ImportError as _e:
    print(f"WARNING: gauche/gpytorch/botorch not available ({_e}); GP experiments will be skipped")
    HAS_GP = False

try:
    from quantile_forest import RandomForestQuantileRegressor
    HAS_QRF = True
except ImportError:
    print("WARNING: quantile_forest not installed, QRF experiments will be skipped")
    HAS_QRF = False

try:
    from ngboost import NGBRegressor
    HAS_NGBOOST = True
except ImportError:
    print("WARNING: ngboost not installed, NGBoost experiments will be skipped")
    HAS_NGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    print("WARNING: lightgbm not installed, LightGBM experiments will be skipped")
    HAS_LGB = False

try:
    from sklearn.svm import SVR
    HAS_SVM = True
except ImportError:
    print("WARNING: sklearn.svm not available, SVM experiments will be skipped")
    HAS_SVM = False

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_sns,
    create_mhg_gnn
)
from kirby.noise_robustness import robustness_metrics
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
    unique_scaffolds = sorted(set(scaffolds))
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
    """DNN matching the qsar_qm_models BNN-Full / VBLL-Full base architecture.

    Hidden sizes [128, 64], ReLU, Dropout(0.2), NO BatchNorm — mirrors
    qsar_qm_models/scripts/investigate_pdv.py DEFAULT_PARAMS['dnn'] and the
    DNNModel class in that file. Required for QM9 and validation BNN-α /
    VBLL-α rows to be directly comparable.
    """
    def __init__(self, input_dim, hidden_size1=128, hidden_size2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze()


class MLPRegressor(nn.Module):
    """MLP matching qsar_qm_models/models/models.py:483-502 MLPRegressor (NN-β base).

    Linear(input→32) → ReLU → Linear(32→32) → ReLU → Dropout(0.2) → Linear(32→1).
    Used as the base for BNN-β (`mlp_bnn_full`) and VBLL-β (`mlp_vbll`) in the
    paper Wilcoxon comparisons.
    """
    def __init__(self, input_dim, hidden_size=32, num_hidden_layers=2, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.dropout(x)
        return self.output_layer(x).squeeze()


if HAS_BAYESIAN_TORCH:
    def apply_bayesian_transformation(model):
        """Full BNN: replace every nn.Linear with torchbnn BayesLinear.

        Mirrors qsar_qm_models/models/models.py:1046 apply_bayesian_transformation.
        """
        transform_model(
            model, nn.Linear, bnn.BayesLinear,
            args={"prior_mu": 0, "prior_sigma": 0.1,
                  "in_features": ".in_features",
                  "out_features": ".out_features", "bias": ".bias"},
            attrs={"weight_mu": ".weight"})
        return model

    def apply_bayesian_transformation_last_layer(model):
        """Last-layer BNN: replace only the final nn.Linear with BayesLinear.

        Kept for completeness; not used by default (see model dispatch below).
        Mirrors qsar_qm_models/models/models.py:1076.
        """
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


# ─── Variational Bayesian Last Layer (full-network variant) ───────────────────
# Mirrors qsar_qm_models/models/models.py:1131-1349 verbatim — VBLLLayer,
# VBLLLoss, and the full-network transformation. NOT wrapped in HAS_BAYESIAN_TORCH
# because VBLL is a pure PyTorch implementation and has no torchbnn dependency.

class VBLLLayer(nn.Module):
    """Variational Bayesian Last Layer (Harrison 2024).

    Mean-field variational posterior q(W) = N(weight_mu, diag(exp(weight_log_sigma)^2))
    over the weight matrix, standard normal prior p(W) = N(0, I), trained via the
    reparameterization trick.
    """
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -3.0))

        # Learned observation noise (aleatoric)
        self.log_noise_var = nn.Parameter(torch.tensor(-2.0))

        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @property
    def noise_var(self):
        return torch.exp(self.log_noise_var)

    def kl_divergence(self):
        prior_var = self.prior_sigma ** 2
        w_var = torch.exp(2.0 * self.weight_log_sigma)
        kl_w = 0.5 * torch.sum(
            w_var / prior_var
            + ((self.prior_mu - self.weight_mu) ** 2) / prior_var
            - 1.0
            + math.log(prior_var) - 2.0 * self.weight_log_sigma
        )
        b_var = torch.exp(2.0 * self.bias_log_sigma)
        kl_b = 0.5 * torch.sum(
            b_var / prior_var
            + ((self.prior_mu - self.bias_mu) ** 2) / prior_var
            - 1.0
            + math.log(prior_var) - 2.0 * self.bias_log_sigma
        )
        return kl_w + kl_b

    def forward(self, x):
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)


class VBLLLoss(nn.Module):
    """ELBO loss for VBLL: Gaussian NLL (learned noise) + sum(KL) / n_data."""
    def __init__(self, model, n_data):
        super().__init__()
        self.model = model
        self.n_data = n_data
        self.vbll_layers = [m for m in model.modules() if isinstance(m, VBLLLayer)]
        self.output_layer = self.vbll_layers[-1] if self.vbll_layers else None

    def forward(self, pred, target):
        if self.output_layer is None:
            return nn.MSELoss()(pred, target)
        noise_var = self.output_layer.noise_var
        nll = 0.5 * torch.log(noise_var) + 0.5 * ((pred - target) ** 2) / noise_var
        nll = nll.mean()
        kl = sum(layer.kl_divergence() for layer in self.vbll_layers)
        return nll + kl / self.n_data


def apply_bayesian_transformation_full_variational(model):
    """Replace ALL nn.Linear layers with VBLLLayer. Initializes each VBLLLayer
    weight_mu from the corresponding pretrained Linear weights.

    Mirrors qsar_qm_models/models/models.py:1295.
    """
    linear_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            linear_layers.append((name, mod))

    def _set(obj, path, val):
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)

    for name, lin in linear_layers:
        vbll = VBLLLayer(lin.in_features, lin.out_features)
        with torch.no_grad():
            vbll.weight_mu.copy_(lin.weight.data)
            if lin.bias is not None:
                vbll.bias_mu.copy_(lin.bias.data)
        _set(model, name, vbll)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# GAUSSIAN PROCESS (gauche / gpytorch — mirrors qsar models.py:504)
# ═══════════════════════════════════════════════════════════════════════════

if HAS_GP:
    class _GaucheExactGP(gpytorch.models.ExactGP):
        """ExactGP with a ScaleKernel wrapper. Mirrors qsar_qm_models
        models.py:504 'Gauche' class verbatim — same mean/cov structure so
        the validation GP rows are directly comparable to QM9 gauche_rbf rows.
        """
        def __init__(self, train_x, train_y, likelihood, kernel_class):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel_class())

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x))


    class GaussianProcessGauche:
        """sklearn-compatible wrapper around the gauche/gpytorch ExactGP so the
        existing run_tree_experiment GP path works unchanged. Default kernel is
        RBF (matches qsar's gauche_rbf — PDV-only in this pipeline).

        Name intentionally contains 'GaussianProcess' so the dispatch substring
        check in run_tree_experiment (`'GaussianProcess' in str(type(mdl))`)
        keeps routing predictions through the (mean, std) branch.
        """
        def __init__(self, kernel='rbf', init_noise=1e-3,
                     init_outputscale=1.0, max_fit_iter=None):
            self.kernel_name = kernel
            self.init_noise = init_noise
            self.init_outputscale = init_outputscale
            self.max_fit_iter = max_fit_iter
            self.model_ = None
            self.likelihood_ = None

        def _kernel_class(self):
            if self.kernel_name == 'rbf':
                return gpytorch.kernels.RBFKernel
            if self.kernel_name == 'tanimoto':
                # Lazy import — only needed if caller switches to Tanimoto.
                from gauche.kernels.fingerprint_kernels.tanimoto_kernel \
                    import TanimotoKernel
                return TanimotoKernel
            raise ValueError(f"unknown kernel {self.kernel_name!r}")

        def fit(self, X, y):
            Xt = torch.from_numpy(np.asarray(X, dtype=np.float64))
            yt = torch.from_numpy(np.asarray(y, dtype=np.float64))
            self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood(
                noise=self.init_noise)
            self.model_ = _GaucheExactGP(Xt, yt, self.likelihood_,
                                         self._kernel_class())
            # Match qsar's init: scale-kernel outputscale starts at 1.0
            self.model_.covar_module.outputscale = self.init_outputscale
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood_, self.model_)
            _fit_gpytorch(mll)
            return self

        def predict(self, X, return_std=False):
            self.model_.eval()
            self.likelihood_.eval()
            Xt = torch.from_numpy(np.asarray(X, dtype=np.float64))
            with torch.no_grad():
                preds = self.model_(Xt)
                mean = preds.mean.detach().cpu().numpy()
                var = preds.variance.detach().cpu().numpy()
            if return_std:
                return mean, np.sqrt(np.clip(var, 1e-12, None))
            return mean


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def train_neural_regression(X_train, y_train, X_val, y_val, X_test,
                            model_type='deterministic', epochs=100, lr=1e-3):
    """Train regression neural network. Returns (predictions, uncertainties).

    Targets are z-score normalized for training (fit on the training labels
    only) and predictions/uncertainties are inverse-transformed back to the
    raw label scale before returning. Without this, MSE on raw-scale labels
    diverges for wide-range datasets (e.g. hERG) and high-dim reps (e.g.
    MHG-GNN-pretrained). Mirrors qsar_qm_models, which trains NNs on
    normalized targets (process_and_train.py --normalize, models.py:2854).
    Tree models are scale-invariant in y and do not need this.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize targets — fit on TRAIN ONLY, no leakage from val/test.
    y_scaler = StandardScaler()
    y_train_norm = y_scaler.fit_transform(
        np.asarray(y_train, dtype=float).reshape(-1, 1)).ravel()
    y_val_norm = y_scaler.transform(
        np.asarray(y_val, dtype=float).reshape(-1, 1)).ravel()
    y_scale = float(y_scaler.scale_[0])  # std used to rescale predicted uncertainty

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train_norm).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val_norm).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    # arch: 'dnn' (DNN, NN-α: [128, 64]) or 'mlp' (MLP, NN-β: [32, 32]).
    # Model types prefixed with 'mlp-' use the MLP base; others use DNN.
    if model_type.startswith('mlp-'):
        arch = 'mlp'
        subtype = model_type[len('mlp-'):]   # 'deterministic' | 'full-bnn' | 'full-vbll'
        model = MLPRegressor(X_train.shape[1]).to(device)
    else:
        arch = 'dnn'
        subtype = model_type
        model = DeterministicRegressor(X_train.shape[1]).to(device)

    if subtype == 'full-bnn':
        if not HAS_BAYESIAN_TORCH:
            raise RuntimeError(f"Full BNN ({arch}) requested but torchbnn not available")
        model = apply_bayesian_transformation(model)
    elif subtype == 'last-layer-bnn':
        if not HAS_BAYESIAN_TORCH:
            raise RuntimeError(f"Last-layer BNN ({arch}) requested but torchbnn not available")
        model = apply_bayesian_transformation_last_layer(model)
    elif subtype == 'full-vbll':
        model = apply_bayesian_transformation_full_variational(model).to(device)

    is_bayesian = subtype != 'deterministic'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # VBLL uses its own ELBO loss; everything else uses MSE.
    if subtype == 'full-vbll':
        criterion = VBLLLoss(model, n_data=len(X_train))
    else:
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

    # Predictions are on the normalized scale — inverse-transform back to raw
    # label units so metrics line up with the tree-model R²/RMSE.
    def _to_raw(arr):
        return y_scaler.inverse_transform(
            np.asarray(arr, dtype=float).reshape(-1, 1)).ravel()

    if is_bayesian:
        preds_norm = np.array([model(X_te).detach().cpu().numpy() for _ in range(30)])
        mean_raw = _to_raw(preds_norm.mean(0))
        # std scales by the label std only (mean-shift cancels in a std).
        std_raw = preds_norm.std(0) * y_scale
        return mean_raw, std_raw
    with torch.no_grad():
        return _to_raw(model(X_te).cpu().numpy()), None


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def run_tree_experiment(X_train, y_train, X_test, y_test, model_fn, strategy, sigma_levels):
    """Noise robustness for tree-based regression model."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
        elif 'NGBRegressor' in str(type(mdl)):
            dist = mdl.pred_dist(X_test)
            predictions[sigma] = dist.loc
            uncertainties[sigma] = dist.scale
        else:
            predictions[sigma] = mdl.predict(X_test)
            uncertainties[sigma] = None

    return predictions, uncertainties


def run_neural_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                          model_type, strategy, sigma_levels):
    """Noise robustness for neural regression model."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

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

def generate_representations(smiles_list, rep_filter=None):
    """Generate all representations for a SMILES list. Returns dict of arrays."""
    import gc
    reps = {}
    wanted = set(rep_filter) if rep_filter else {'ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained'}

    if 'ECFP4' in wanted:
        print("  ECFP4...", flush=True)
        reps['ECFP4'] = create_ecfp4(smiles_list, n_bits=2048)
        print(f"    done: {reps['ECFP4'].shape}", flush=True)

    if 'PDV' in wanted:
        print("  PDV...", flush=True)
        reps['PDV'] = create_pdv(smiles_list)
        print(f"    done: {reps['PDV'].shape}", flush=True)

    if 'SNS' in wanted:
        print("  SNS...", flush=True)
        reps['SNS'], _ = create_sns(smiles_list, return_featurizer=True)
        print(f"    done: {reps['SNS'].shape}", flush=True)

    if 'MHG-GNN-pretrained' in wanted:
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

def run_dataset(dataset_name, smiles, labels, results_dir, model_filter=None, rep_filter=None, sigma_levels=None,
                gp_kernel='rbf', gp_reps=None):
    """Run full scaffold CV experiment for one dataset."""
    if sigma_levels is None:
        sigma_levels = SIGMA_LEVELS
    # Reps on which to run the GP (gauche/gpytorch). Default {'PDV'} preserves
    # the original PDV-only RBF behaviour; pass gp_reps to run GP on more reps
    # (e.g. all four) so a single consistent GP model spans every rep and can
    # enter the cross-rep ANOVA.
    gp_rep_set = set(gp_reps) if gp_reps else {'PDV'}
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
    reps = generate_representations(smiles, rep_filter=rep_filter)

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
        # NGBoost
        if HAS_NGBOOST:
            experiments.append(('NGBoost', rname,
                lambda: NGBRegressor(n_estimators=500, learning_rate=0.01, random_state=42,
                                     verbose=False), None))
        # LightGBM
        if HAS_LGB:
            experiments.append(('LightGBM', rname,
                lambda: LGBMRegressor(n_estimators=100, random_state=42, verbose=-1,
                                      n_jobs=-1), None))
        # SVM — match QM9 defaults exactly: fixed RBF, C=1.0, gamma='scale',
        # for every representation. No per-rep kernel switching, no Optuna
        # tuning, so SVM is the same model on QM9 and on the validation
        # datasets and is free of the kernel--representation confound.
        if HAS_SVM:
            experiments.append(('SVM', rname,
                lambda: SVR(C=1.0, gamma='scale', kernel='rbf'), None))
        # GP — gauche/gpytorch ExactGP with learned GaussianLikelihood noise,
        # mirroring qsar_qm_models/models/models.py:504. Kernel selectable via
        # gp_kernel (rbf default — valid on every rep, matches QM9 gauche_rbf;
        # tanimoto only valid on binary fingerprints, e.g. ECFP4). Reps selectable
        # via gp_reps (default {'PDV'} preserves the original behaviour).
        # Display name encodes the kernel so the figure pipeline can map
        # RBF -> gauche_rbf and Tanimoto -> gauche (val_model_map update needed).
        if HAS_GP and rname in gp_rep_set:
            gp_name = 'GP' if gp_kernel == 'rbf' else f'GP-{gp_kernel.capitalize()}'
            experiments.append((gp_name, rname,
                lambda k=gp_kernel: GaussianProcessGauche(
                    kernel=k, init_noise=1e-3, init_outputscale=1.0), None))
        # Neural models. Display names match the val_model_map in
        # qsar_qm_models/scripts/generate_paper_figures.py:_normalize_validation_names
        # so rows merge into the canonical QM9 model namespace:
        #   DNN              -> dnn          (NN-α deterministic)
        #   BNN-Full         -> dnn_bnn_full (BNN-α, full BNN on DNN base)
        #   VBLL-Full        -> dnn_vbll     (VBLL-α, full VBLL on DNN base)
        #   MLP              -> mlp          (NN-β deterministic)
        #   MLP-BNN-Full     -> mlp_bnn_full (BNN-β, full BNN on MLP base)
        #   MLP-VBLL-Full    -> mlp_vbll     (VBLL-β, full VBLL on MLP base)
        # BNN-Last is implemented but not run by default (user directive).
        for mtype, mname in [('deterministic', 'DNN'),
                              ('full-bnn', 'BNN-Full'),
                              ('full-vbll', 'VBLL-Full'),
                              ('mlp-deterministic', 'MLP'),
                              ('mlp-full-bnn', 'MLP-BNN-Full'),
                              ('mlp-full-vbll', 'MLP-VBLL-Full')]:
            if mtype != 'deterministic' and not HAS_BAYESIAN_TORCH:
                continue
            experiments.append((mname, rname, None, mtype))

    # Filter to specific models if requested
    if model_filter:
        filter_set = set(model_filter)
        experiments = [(n, r, fn, mt) for n, r, fn, mt in experiments if n in filter_set]
        print(f"  Filtered to models: {sorted(filter_set)}")

    print(f"\n{len(experiments)} model-rep configs × {len(STRATEGIES)} strategies × {len(sigma_levels)} sigmas × {N_FOLDS} folds")

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
                                model_fn, strategy, sigma_levels)
                        else:
                            predictions, uncertainties = run_tree_experiment(
                                X_train, y_train, X_test, y_test,
                                model_fn, strategy, sigma_levels)
                    else:
                        predictions, uncertainties = run_neural_experiment(
                            X_train, y_train, X_val, y_val, X_test, y_test,
                            model_type, strategy, sigma_levels)

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
                        for sigma in sigma_levels:
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

        # Incremental save after each fold (protection against timeout)
        if all_fold_results:
            _partial = pd.concat(all_fold_results, ignore_index=True)
            _partial_path = results_dir / 'all_results_partial.csv'
            _partial.to_csv(_partial_path, index=False)
            print(f"  [Checkpoint] Saved {len(_partial)} rows after fold {fold_idx + 1}")

    if not all_fold_results:
        print(f"ERROR: No results for {dataset_name}")
        return pd.DataFrame()

    # Combine results across folds
    combined = pd.concat(all_fold_results, ignore_index=True)

    # Merge with existing results if running a subset of models/reps
    existing_path = results_dir / 'all_results.csv'
    if (model_filter or rep_filter) and existing_path.exists():
        existing = pd.read_csv(existing_path)
        # Remove old rows ONLY for the exact model+rep combos we're re-running
        if model_filter and rep_filter:
            mask = existing['model'].isin(model_filter) & existing['rep'].isin(rep_filter)
        elif model_filter:
            mask = existing['model'].isin(model_filter)
        else:
            mask = existing['rep'].isin(rep_filter)
        existing = existing[~mask]
        combined = pd.concat([existing, combined], ignore_index=True)
        print(f"  Merged with existing results: {len(combined)} total rows")

    combined.to_csv(results_dir / 'all_results.csv', index=False)

    # Aggregate across folds and compute curve-shape robustness metrics
    summary_rows = []
    for (model, rep, strat), grp in combined.groupby(['model', 'rep', 'strategy']):
        # Average across folds first, then summarise the degradation curve
        fold_avgs = grp.groupby('sigma')[['r2', 'rmse', 'mae', 'spearman']].mean().reset_index()
        fold_avgs = fold_avgs.sort_values('sigma')

        sigmas = fold_avgs['sigma'].values
        r2_vals = fold_avgs['r2'].values
        rmse_vals = fold_avgs['rmse'].values

        baseline_r2 = r2_vals[0]
        baseline_rmse = rmse_vals[0]

        # Curve-shape robustness from R² retention (no linear assumption).
        # auc_norm is PRIMARY (higher = more robust); weibull_* are supplementary.
        rob = robustness_metrics(sigmas, r2_vals)

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
            'auc_norm': rob['auc_norm'],
            'weibull_tau': rob['weibull_tau'],
            'weibull_beta': rob['weibull_beta'],
            'retention_0.5': retention_05,
            'retention_1.0': retention_10,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Merge with existing summary if running a subset of models/reps
    existing_summary = results_dir / 'summary.csv'
    if (model_filter or rep_filter) and existing_summary.exists():
        existing = pd.read_csv(existing_summary)
        if model_filter and rep_filter:
            mask = existing['model'].isin(model_filter) & existing['rep'].isin(rep_filter)
        elif model_filter:
            mask = existing['model'].isin(model_filter)
        else:
            mask = existing['rep'].isin(rep_filter)
        existing = existing[~mask]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)

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
        print(f"\n  Top 5 most robust (highest auc_norm):")
        top5 = working.nlargest(5, 'auc_norm')
        for _, row in top5.iterrows():
            print(f"    {row['model']:12} + {row['rep']:20} + {row['strategy']:10}: "
                  f"auc_norm={row['auc_norm']:.4f}, baseline={row['baseline_r2']:.3f}±{row['baseline_r2_std']:.3f}")

        print(f"\n  By representation (mean across models/strategies):")
        rep_agg = working.groupby('rep')[['baseline_r2', 'auc_norm', 'retention_0.5']].mean()
        print(rep_agg.round(4).to_string())

        print(f"\n  By model (mean across reps/strategies):")
        mod_agg = working.groupby('model')[['baseline_r2', 'auc_norm', 'retention_0.5']].mean()
        print(mod_agg.round(4).to_string())

    return summary_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Validation Noise Robustness: 3 Regression Datasets with Scaffold CV')
    parser.add_argument('--datasets', nargs='+',
                        choices=['logd', 'caco2', 'herg_ki', 'all'],
                        default=['all'],
                        help='Which datasets to test (default: all)')
    parser.add_argument('--openadmet-csv', type=str, default=None,
                        help='Path to cached OpenADMET CSV')
    parser.add_argument('--results-root', type=str, default='results/validation',
                        help='Root directory for results')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Only run these models (e.g. --models NGBoost SVM LightGBM)')
    parser.add_argument('--reps', nargs='+', default=None,
                        choices=['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained'],
                        help='Only run these reps (e.g. --reps PDV). Merges with existing results.')
    parser.add_argument('--sigmas', nargs='+', type=float, default=None,
                        help='Only run these sigma levels (e.g. --sigmas 0.0 0.5 1.0). Default: all 11.')
    parser.add_argument('--gp-kernel', type=str, default='rbf', choices=['rbf', 'tanimoto'],
                        help="GP (gauche) kernel. 'rbf' (default) is valid on all reps and matches "
                             "QM9 gauche_rbf; 'tanimoto' is only valid on binary fingerprints (ECFP4).")
    parser.add_argument('--gp-reps', nargs='+', default=None,
                        choices=['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained'],
                        help='Reps on which to run the GP. Default: PDV only (original behaviour). '
                             'Pass all four to run GP on every rep so it can enter the cross-rep ANOVA.')
    args = parser.parse_args()

    ds_list = ['logd', 'caco2', 'herg_ki'] if 'all' in args.datasets else args.datasets
    sigma_levels = args.sigmas if args.sigmas else SIGMA_LEVELS

    print("=" * 80)
    print("Validation Noise Robustness Experiment")
    print("=" * 80)
    print(f"  Datasets: {ds_list}")
    print(f"  Split: {N_FOLDS}-fold scaffold CV")
    rep_list = ", ".join(args.reps) if args.reps else "ECFP4, PDV, SNS, MHG-GNN-pretrained"
    print(f"  Representations: {rep_list}")
    print(f"  Noise strategies: {STRATEGIES}")
    print(f"  Sigma levels: {SIGMA_LEVELS}")
    model_list = ("RF, QRF, XGBoost, NGBoost, LightGBM, SVM, GP(PDV), "
                  "DNN, VBLL-Full, MLP, MLP-VBLL-Full")
    if HAS_BAYESIAN_TORCH:
        model_list += ", BNN-Full, MLP-BNN-Full"
    if args.models:
        model_list = ", ".join(args.models)
    print(f"  Models: {model_list}")
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
                                  Path(args.results_root) / 'logd',
                                  model_filter=args.models, rep_filter=args.reps,
                                  sigma_levels=sigma_levels,
                                  gp_kernel=args.gp_kernel, gp_reps=args.gp_reps)
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
                                  Path(args.results_root) / 'caco2',
                                  model_filter=args.models, rep_filter=args.reps,
                                  sigma_levels=sigma_levels,
                                  gp_kernel=args.gp_kernel, gp_reps=args.gp_reps)
            all_summaries.append(summary)
        else:
            print("ERROR: Cannot find Caco2 Efflux column")

    # 3. ChEMBL-hERG-Ki
    if 'herg_ki' in ds_list:
        print("\nPreparing ChEMBL-hERG-Ki...")
        smiles, labels = load_chembl_herg()
        print(f"  {len(smiles)} molecules")
        summary = run_dataset('ChEMBL-hERG-Ki', smiles, labels,
                              Path(args.results_root) / 'herg',
                              model_filter=args.models, rep_filter=args.reps,
                              gp_kernel=args.gp_kernel, gp_reps=args.gp_reps)
        all_summaries.append(summary)

    # Cross-dataset summary
    if len(all_summaries) > 1:
        combined = pd.concat(all_summaries, ignore_index=True)

        # Merge with existing combined_summary if running a subset
        combined_path = Path(args.results_root) / 'combined_summary.csv'
        if (args.models or args.reps) and combined_path.exists():
            existing = pd.read_csv(combined_path)
            if args.models and args.reps:
                mask = existing['model'].isin(args.models) & existing['rep'].isin(args.reps)
            elif args.models:
                mask = existing['model'].isin(args.models)
            else:
                mask = existing['rep'].isin(args.reps)
            existing = existing[~mask]
            combined = pd.concat([existing, combined], ignore_index=True)

        combined.to_csv(combined_path, index=False)

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
