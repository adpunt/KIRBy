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
  Representations: ECFP4, PDV, SNS, MHG-GNN-pretrained, Avalon, ChemBERTa (6 total)
  Models: RF, QRF, XGBoost, NGBoost, LightGBM, SVM, GP(PDV only),
          DNN, BNN-Full, VBLL-Full,                    (NN-α family — DNN base)
          MLP, MLP-BNN-Full, MLP-VBLL-Full             (NN-β family — MLP base)
          (13 total; BNN-Last implemented but not run by default)
  Total: 4 reps × 13 models × 7 conditions × 3 datasets × 5 folds

Noise Strategies (regression): legacy, outlier, quantile, hetero, threshold, valprop (6)
Sigma levels: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (11)

Usage:
  python alternative_data_noise_robustness.py
  python alternative_data_noise_robustness.py --datasets logd caco2 herg
  python alternative_data_noise_robustness.py --results-root results/validation
"""

import argparse
import zlib

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
# SHARED MODEL SPEC
# ═══════════════════════════════════════════════════════════════════════════
# Every default this pipeline shares with QM9 lives in ONE file, in the
# qsar_qm_models checkout, and is IMPORTED rather than restated. Restating it
# here is what let the boosting learning rate, the quantile-forest tree count
# and the NN-beta width drift apart under one name each.
#
# No vendored copy and no silent fallback: if the checkout cannot be found this
# raises, because a second copy that quietly goes stale is the defect, not the
# fix. Every job script already runs `. <qsar_qm_models>/setup.sh` before this
# module is imported, so the checkout is present by construction.

def _load_model_defaults():
    candidates = []
    env = os.environ.get('QSAR_QM_MODELS_ROOT')
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates += [
        here.parent.parent.parent / 'qsar_qm_models',   # sibling checkout
        Path('/data/stat-cadd/scat9264/qsar_qm_models'),
        Path('/data/stat-ecr/scat9264/qsar_qm_models'),
    ]
    for root in candidates:
        spec_file = Path(root) / 'models' / 'model_defaults.py'
        if spec_file.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location('model_defaults', spec_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"  model spec: {spec_file}  version {mod.SPEC_VERSION}  "
                  f"hash {mod.spec_hash()}")
            return mod
    raise RuntimeError(
        "Cannot find models/model_defaults.py, the shared parameter spec. This "
        "pipeline must build the same models as QM9 and refuses to guess them. "
        "Set QSAR_QM_MODELS_ROOT to the qsar_qm_models checkout, or place it "
        f"beside the KIRBy checkout. Looked in: "
        + ", ".join(str(c) for c in candidates))


MODEL_SPEC = _load_model_defaults()
SKLEARN_DEFAULTS = MODEL_SPEC.SKLEARN_DEFAULTS
GP_DEFAULTS = MODEL_SPEC.GP_DEFAULTS
NEURAL_DEFAULTS = MODEL_SPEC.NEURAL_DEFAULTS
BAYESIAN_DEFAULTS = MODEL_SPEC.BAYESIAN_DEFAULTS
sklearn_params = MODEL_SPEC.sklearn_params
should_standardise = MODEL_SPEC.should_standardise


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# The noise conditions, named exactly as in `noiseInject.CONDITIONS` and in
# `rust/src/main.rs`, so the experimental and QM9 halves of the study can be read
# on one axis. The six they replace were one condition at six doses: at a common
# nominal setting they delivered between 0.49x and 2.00x the same amount of
# noise, and their entire apparent severity ordering was that. See
# NOISE_DESIGN.md sections 1, 2 and 2a.
#
# Shape, at matched amount:      gaussian, student_t_nu10/5/3, laplace
# Targeting, at matched amount:  grouped_wider, grouped_shifted, outlier_p01/05/10
# Not dose-matched, own axis:    censoring_10 ... censoring_50
NOISE_CONDITIONS = ['gaussian', 'student_t_nu5', 'laplace',
                    'grouped_wider', 'grouped_shifted', 'outlier_p05',
                    'censoring']

# Which conditions need a scaffold group assignment to be injectable at all.
GROUPED_CONDITIONS = ('grouped_wider', 'grouped_shifted')

# Models that emit a PER-MOLECULE uncertainty. QRF via its quantile spread, GP
# via posterior sd, NGBoost via its predicted distribution, and the four
# Bayesian/variational nets via the spread of NEURAL_DEFAULTS mc_passes
# stochastic forward passes.
# DNN and MLP are deterministic and return None, so cross-fitting them would
# buy nothing. Kept as one list so the runner, the roster and the SLURM
# generator cannot drift apart.
UNCERTAINTY_MODELS = ['QRF', 'NGBoost', 'GP',
                      'BNN-Full', 'VBLL-Full', 'MLP-BNN-Full', 'MLP-VBLL-Full']
# The noise LEVEL is now the amount of noise DELIVERED -- the root-mean-square
# error added, in the label's own units, which for these three datasets is log
# units. Every condition solves for the internal scale that hits it, so a level
# means the same thing across conditions. It did not before: the number was a
# raw scale each condition interpreted its own way.
#
# In log units these are anchored to published assay error (NOISE_DESIGN.md
# section 6.4): logD 0.15 within a laboratory, Caco-2 0.35 between eleven
# laboratories, hERG 0.54. Each grid brackets one unit of real error and runs to
# roughly twice it. Report the resulting fraction of the label spread alongside,
# because one unit of real error is 0.13 of the spread on logD but 0.76 on
# Caco-2 -- a factor of six, and a single shared ladder would mean six different
# experiments.
NOISE_LEVELS_BY_DATASET = {
    'OpenADMET-LogD':          [0.0, 0.15, 0.3, 0.5, 0.75, 1.0],
    'OpenADMET-Caco2_Efflux':  [0.0, 0.1, 0.2, 0.35, 0.5, 0.7],
    'ChEMBL-hERG-Ki':          [0.0, 0.15, 0.3, 0.54, 0.8, 1.1],
}
# Censoring is swept on its own axis -- the fraction of labels clipped -- because
# it has no variance parameter and is not zero-mean. Its LEVEL is that fraction,
# so 0.0 really is a clean baseline. Grid set by the range-finding run: damage
# accelerates smoothly, becoming serious around 20% and severe past 30%, and
# Svensson et al. report 25-63% of labels censored in real industrial assays.
CENSORING_LEVELS = [0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50]

# Fallback for a dataset with no anchored grid of its own.
NOISE_LEVELS = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0]
N_FOLDS = 5
# From the shared spec. Uncapped, to match QM9, which fits its GP on the whole
# training set (models.py stacks train+val, 9,000 molecules at -n 10000). A
# 2,000-molecule cap made this a DIFFERENT model, not the same model on less
# data. Set max_train_n in model_defaults.py to reinstate a cap; it then
# applies to every GP variant, not just the one named exactly 'GP'.
GP_MAX_N = GP_DEFAULTS['max_train_n']
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


def _ngboost_kwargs():
    """Resolve the shared spec's Dist and Score NAMES to the classes.

    QM9 passes Dist=Normal and Score=MLE explicitly while this pipeline used to
    pass neither, so the two agreed only for as long as ngboost's defaults
    happened to match. They are now named in one place and resolved here, and
    scripts/audit_pipeline_parity.py asserts the names still point where they
    used to.
    """
    from ngboost.distns import Normal
    from ngboost.scores import MLE
    params = dict(SKLEARN_DEFAULTS['ngboost'])
    dist_name, score_name = params.pop('dist'), params.pop('score')
    params['Dist'] = {'Normal': Normal}[dist_name]
    params['Score'] = {'MLE': MLE}[score_name]
    return params

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
    create_mhg_gnn,
    create_avalon,
    create_chemberta
)
from kirby.noise_robustness import robustness_metrics
from noiseInject import CONDITIONS, NoiseInjectorRegression


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
    """Assign scaffold group IDs to molecules. Used for the CV fold geometry."""
    scaffolds = [get_scaffold(smi) for smi in smiles_list]
    unique_scaffolds = sorted(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    groups = np.array([scaffold_to_id[s] for s in scaffolds])
    return groups, len(unique_scaffolds)


def assign_noise_groups(smiles_list):
    """Scaffold groups for NOISE INJECTION, with acyclic molecules as singletons.

    Deliberately separate from `assign_scaffold_groups`, which defines the CV
    fold geometry: changing that would silently move every fold in the study.

    The difference is NOISE_DESIGN.md section 2a rule 2. `get_scaffold` returns
    an empty string for an acyclic molecule, so every acyclic molecule lands in
    one bucket. For splitting that is merely coarse; for grouped noise it is
    wrong, because a single group-level draw would then move the whole bucket at
    once. Measured on QM9, where 32.2% of molecules are acyclic, that swings the
    realised dose from 0.795 to 1.493 against a target of 1.0. Each acyclic
    molecule therefore becomes its own group.

    Returns (groups, n_groups, largest_group_share).
    """
    scaffolds = []
    for i, smi in enumerate(smiles_list):
        core = get_scaffold(smi)
        scaffolds.append(f'__acyclic_{i}' if core == '' else core)
    unique_scaffolds = sorted(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    groups = np.array([scaffold_to_id[s] for s in scaffolds])
    largest = float(np.bincount(groups).max() / len(groups))
    return groups, len(unique_scaffolds), largest


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
    # The cached CSV has been written with different label column names by
    # different versions of fetch_chembl_herg_ki (a stale cache carrying
    # 'pChEMBL' rather than 'pKi' raised KeyError and killed the whole run).
    # Accept any of the known spellings and fail with a readable message.
    label_col = next((c for c in ('pKi', 'pChEMBL', 'pchembl_value', 'pchembl')
                      if c in df.columns), None)
    if label_col is None:
        raise KeyError(
            f"hERG label column not found. Looked for pKi/pChEMBL/pchembl_value; "
            f"the file has: {list(df.columns)}")
    labels_arr = df[label_col].values.astype(np.float64)

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

    Built at hidden_size=128 by the caller, matching QM9: Linear(input→128) →
    ReLU → Linear(128→128) → ReLU → Dropout(0.2) → Linear(128→1). The class
    default below is 32 and must NOT be relied on — QM9 passes 128.
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
        def __init__(self, kernel=None, init_noise=None,
                     init_outputscale=None, max_fit_iter=None):
            self.kernel_name = GP_DEFAULTS['kernel'] if kernel is None else kernel
            self.init_noise = (GP_DEFAULTS['likelihood_noise']
                               if init_noise is None else init_noise)
            self.init_outputscale = (GP_DEFAULTS['outputscale']
                                     if init_outputscale is None else init_outputscale)
            self.max_fit_iter = (GP_DEFAULTS['fallback_adam_iters']
                                 if max_fit_iter is None else max_fit_iter)
            self.model_ = None
            self.likelihood_ = None
            # 'botorch' or 'adam_fallback'. Written into every results row as
            # gp_fit_method: the fallback fits a DIFFERENT optimisation and used
            # to leave no trace outside this attribute.
            self.fit_method_ = None
            self.fit_fallback_ = None

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
            # QM9 computed an outputscale and never applied it, so its
            # ScaleKernel actually started at gpytorch's softplus(0) ~ 0.693.
            # Both sides now apply the spec value.
            if GP_DEFAULTS['apply_outputscale']:
                self.model_.covar_module.outputscale = self.init_outputscale
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood_, self.model_)
            # botorch >= ~0.12 requires its fitter's argument to be a botorch
            # Model (it calls transform_inputs); _GaucheExactGP is a plain
            # gpytorch ExactGP, so on newer botorch this raises
            #   '_GaucheExactGP' object has no attribute 'transform_inputs'
            # and -- because run_dataset catches per-condition exceptions -- the
            # whole GP arm produced NOTHING while the job still exited 0.
            # Fall back to a plain gpytorch Adam loop, which is version-proof.
            try:
                _fit_gpytorch(mll)
                self.fit_method_ = 'botorch'
            except Exception as e:
                self.fit_fallback_ = f"{type(e).__name__}: {e}"
                self.fit_method_ = 'adam_fallback'
                print(f"      [gp] botorch fitter refused this model "
                      f"({self.fit_fallback_}); using the Adam fallback. "
                      f"Recorded as gp_fit_method=adam_fallback.", flush=True)
                self.model_.train()
                self.likelihood_.train()
                opt = torch.optim.Adam(self.model_.parameters(),
                                       lr=GP_DEFAULTS['fallback_adam_lr'])
                n_iter = self.max_fit_iter
                for _ in range(n_iter):
                    opt.zero_grad()
                    loss = -mll(self.model_(Xt), yt)
                    loss.backward()
                    opt.step()
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
                            model_type='deterministic', epochs=None, lr=None):
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
    _t = NEURAL_DEFAULTS['training']
    epochs = _t['epochs'] if epochs is None else epochs
    lr = _t['lr'] if lr is None else lr

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
        # Width 128 to match QM9, which passes hidden_size=128 explicitly
        # (qsar models.py:2531,:2613) and which paper.tex:216 states as [128,128].
        # The class default of 32 was being taken here, giving 16x fewer
        # hidden parameters under the same model name.
        model = MLPRegressor(
            X_train.shape[1],
            hidden_size=NEURAL_DEFAULTS['mlp']['hidden_size'],
            num_hidden_layers=NEURAL_DEFAULTS['mlp']['num_hidden_layers'],
            dropout_rate=NEURAL_DEFAULTS['mlp']['dropout_rate']).to(device)
    else:
        arch = 'dnn'
        subtype = model_type
        h1, h2 = NEURAL_DEFAULTS['dnn']['hidden_sizes']
        model = DeterministicRegressor(X_train.shape[1], h1, h2).to(device)

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
    # Batch size 32, from the shared spec. This pipeline used 64 while QM9 used
    # 32 -- not listed in the original hand audit at all.
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=_t['batch_size'],
                        shuffle=True)

    best_val, patience_ctr = float('inf'), 0
    patience = _t['patience']
    tolerance = _t['improvement_tolerance']
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
        if vl < best_val - tolerance:
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
        # 100 passes to match QM9 (models.py:2077 and six other sites) and
        # paper.tex:216, which already claims 100.
        preds_norm = np.array([model(X_te).detach().cpu().numpy()
                               for _ in range(_t['mc_passes'])])
        mean_raw = _to_raw(preds_norm.mean(0))
        # std scales by the label std only (mean-shift cancels in a std).
        std_raw = preds_norm.std(0) * y_scale
        return mean_raw, std_raw
    with torch.no_grad():
        return _to_raw(model(X_te).cpu().numpy()), None


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def _oof_predict(fit_predict, X, y_noisy, n_folds, groups=None, seed=42):
    """Out-of-fold predictions and uncertainties over the TRAINING molecules.

    Splits the training rows into `n_folds` parts, fits on the rest and scores
    the held-out part, rotating. Every training molecule therefore receives a
    prediction from a model that never saw ITS label.

    This is the only construction under which "does predicted uncertainty
    track which labels were corrupted?" is answerable: scoring a molecule with
    a model that fitted its corrupted label measures memorisation, not
    uncertainty (a GP has zero posterior variance at its own training inputs;
    a forest has fitted those exact rows).

    The noise is injected ONCE over the whole training set before this is
    called, so a molecule carries the same corruption in whichever fold it
    lands.

    `groups` are the Murcko-scaffold ids of the training rows. When supplied the
    inner split is scaffold-grouped, matching the outer CV: without it the fit
    set contains close analogues of every held-out molecule, so out-of-fold
    uncertainties come from an interpolation regime while the test set is an
    extrapolation regime, and the two are not on the same scale.

    `fit_predict(X_fit, y_fit, X_score) -> (mean, std_or_None)`.

    Returns (oof_mean, oof_unc, n_folds_ok) — the caller must check n_folds_ok,
    because a silently truncated out-of-fold pass looks exactly like a complete
    one in the output file.
    """
    n = len(y_noisy)
    oof_mean = np.full(n, np.nan)
    oof_unc = np.full(n, np.nan)

    if groups is not None and len(np.unique(groups)) >= n_folds:
        splitter = GroupKFold(n_splits=n_folds)
        folds = [(tr, te) for tr, te in splitter.split(X, y_noisy, groups)]
    else:
        if groups is not None:
            print(f"      [oof] only {len(np.unique(groups))} scaffolds for {n_folds} folds "
                  f"— falling back to a random split", flush=True)
        order = np.random.RandomState(seed).permutation(n)
        parts = np.array_split(order, n_folds)
        folds = [(np.setdiff1d(order, held), held) for held in parts]

    n_ok = 0
    for keep, held in folds:
        try:
            m, u = fit_predict(X[keep], y_noisy[keep], X[held])
        except Exception as e:
            print(f"      [oof] fold failed: {type(e).__name__}: {e}", flush=True)
            continue
        oof_mean[held] = np.asarray(m, dtype=float).ravel()
        if u is not None:
            oof_unc[held] = np.asarray(u, dtype=float).ravel()
        n_ok += 1
    if n_ok < len(folds):
        print(f"      [oof] WARNING {n_ok}/{len(folds)} inner folds succeeded", flush=True)
    return oof_mean, oof_unc, n_ok


def _tree_mean_std(mdl, X_score):
    """Per-molecule mean and uncertainty from a fitted tree/GP/NGBoost model.

    Single definition, used for both the held-out test set and the
    out-of-fold training predictions, so the two are never computed
    differently.
    """
    if 'Quantile' in str(type(mdl)):
        q16, q50, q84 = mdl.predict(X_score, quantiles=[0.16, 0.5, 0.84]).T
        return q50, (q84 - q16) / 2
    if 'GaussianProcess' in str(type(mdl)):
        pm, ps = mdl.predict(X_score, return_std=True)
        return pm, ps
    if 'NGBRegressor' in str(type(mdl)):
        dist = mdl.pred_dist(X_score)
        return dist.loc, dist.scale
    return mdl.predict(X_score), None


def _injector_for(condition, level_index, stream='training'):
    """A fresh injector for one condition at one level of the grid.

    Two properties this exists to guarantee, both learned the hard way.

    REPRODUCIBILITY. `NoiseInjectorRegression` advances its generator on every
    draw, so re-using one injector across the level grid makes the realisation at
    a level depend on how many levels ran before it -- and a run of a subset of
    the grid would then not reproduce the corresponding rows of a full run. The
    seed here depends on the condition and the level, and on nothing else.

    SEPARATE STREAMS. The validation split draws from its own stream, so
    corrupting validation cannot change the training corruption. That was already
    true (two hard-coded seeds); it is now derived rather than hard-coded, so it
    stays true when a condition or a level is added.
    """
    base = 42 if stream == 'training' else 1337
    # zlib.crc32, NOT hash(): Python randomises string hashing per process, so
    # hash() here would have made the noise different on every run while the
    # docstring above promised the opposite.
    digest = zlib.crc32(condition.encode('utf-8'))
    seed = (base * 1000003 + digest + level_index * 7919) & 0x7FFFFFFF
    return NoiseInjectorRegression.from_condition(condition, random_state=seed)


def run_tree_experiment(X_train, y_train, X_test, y_test, model_fn, condition, sigma_levels,
                        oof_folds=0, train_groups=None, noise_groups=None,
                        test_noise_groups=None, provenance=None, rep_name=None):
    """Noise robustness for tree-based regression model.

    Returns (predictions, uncertainties, extras). `extras` carries everything
    needed to ask whether uncertainty tracks the injected noise:
      train_noise_scale[sigma]  per-molecule noise SD actually applied in training
      train_epsilon[sigma]      the noise actually added to each training label
      test_noise_scale[sigma]   the noise scale a TEST molecule's region would
                                receive, using the TRAINING distribution's
                                cut-points -- the model never saw this, so it is
                                the target for "did the model learn where the
                                data is unreliable?"
      oof_mean/oof_unc[sigma]   out-of-fold training predictions (if oof_folds>0)
    """
    # Standardise only the representations the shared spec names. This used to
    # scale EVERYTHING, including binary fingerprints -- which QM9 does not do,
    # and which costs an RBF-kernel support vector machine about 0.6 R-squared
    # because the rare bits get huge magnitudes and dominate every distance.
    if rep_name is None:
        raise ValueError(
            "run_tree_experiment needs rep_name so the standardisation "
            "decision can be reported against a representation. The decision "
            "itself is made from the features, not the name.")
    if should_standardise(X_train, rep_name):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    predictions, uncertainties = {}, {}
    extras = {'train_noise_scale': {}, 'train_epsilon': {}, 'test_noise_scale': {},
              'test_noise_pattern': {}, 'train_noise_pattern': {},
              'test_noise_pattern_pred': {}, 'train_noise_pattern_pred': {},
              'oof_mean': {}, 'oof_unc': {}, 'oof_folds_ok': {},
              'gp_fit_method': ''}

    # The SHAPE of the noise -- which molecules a condition hits hardest -- taken
    # at a fixed reference level, so it is the same column at every level
    # INCLUDING zero. This is what makes the confound testable: for the
    # label-keyed conditions the noise scale is a deterministic function of the
    # label, and a model's uncertainty may already track the label for reasons
    # that have nothing to do with noise. Correlating uncertainty against this
    # pattern on a model trained at level 0 measures exactly that pre-existing
    # association, and the honest effect is (correlation at the level) minus
    # (correlation at zero).
    #
    # The pattern is reproducible because the scale map is drawn from a
    # separately seeded generator, so it does not depend on how many shape draws
    # have happened -- see `_selection_rng` in noiseInject.
    pattern_injector = _injector_for(condition, 0)
    tr_pattern = pattern_injector.noise_scale(y_train, 1.0, reference=y_train,
                                              groups=noise_groups,
                                              reference_groups=noise_groups)
    # reference_groups=noise_groups, NOT test_noise_groups: the affected groups
    # must be the ones the TRAINING labels were exposed to, then looked up by
    # group id for the test molecules. Re-running the selection over the test
    # set's own groups picks a different set -- measured on a 40-group split,
    # two of eight corrupted groups went unmarked -- and question B would then
    # be scored against an injection that never happened.
    te_pattern = pattern_injector.noise_scale(y_test, 1.0, reference=y_train,
                                              groups=test_noise_groups,
                                              reference_groups=noise_groups)

    for sigma in sigma_levels:
        # A FRESH injector per level, seeded from the level index. The generator
        # advances with every draw, so re-using one would make the realisation
        # at a level depend on how many levels were run before it -- and a run
        # of a subset of the grid would then not reproduce the full run.
        injector = _injector_for(condition, sigma_levels.index(sigma))
        result = injector.inject_verbose(y_train, sigma, groups=noise_groups)
        y_noisy, tr_scale, eps = result.y_noisy, result.noise_scale, result.epsilon
        # noise_scale consumes no randomness and re-seeds its own selection, so
        # calling it here cannot perturb the draw above.
        te_scale = injector.noise_scale(y_test, sigma, reference=y_train,
                                        groups=test_noise_groups,
                                        reference_groups=noise_groups)
        if provenance is not None:
            provenance[sigma] = result.as_row()

        mdl = model_fn()
        mdl.fit(X_train, y_noisy)
        # Which optimiser actually fitted the GP -- '' for every other model.
        # The Adam fallback is a different fit, so it has to reach the results.
        if getattr(mdl, 'fit_method_', None):
            extras['gp_fit_method'] = mdl.fit_method_
        mean, std = _tree_mean_std(mdl, X_test)
        predictions[sigma] = mean
        uncertainties[sigma] = std

        extras['train_noise_scale'][sigma] = tr_scale
        extras['train_epsilon'][sigma] = eps
        extras['test_noise_scale'][sigma] = te_scale
        extras['train_noise_pattern'][sigma] = tr_pattern
        extras['test_noise_pattern'][sigma] = te_pattern
        # The sham ceiling: the same pattern computed from what the model
        # PREDICTED rather than the true label. If uncertainty tracks this just
        # as strongly, the model is following its own prediction, not the noise.
        # Computed here because quantile and outlier need the training reference,
        # which is not recoverable from the output file once folds 1-4 carry no
        # training rows.
        extras['test_noise_pattern_pred'][sigma] = pattern_injector.noise_scale(
            np.asarray(predictions[sigma], dtype=float), 1.0, reference=y_train,
            groups=test_noise_groups, reference_groups=noise_groups)
        if sigma in extras['oof_mean']:
            _om = np.asarray(extras['oof_mean'][sigma], dtype=float)
            _om = np.where(np.isfinite(_om), _om, float(np.nanmean(_om)) if np.isfinite(_om).any() else 0.0)
            extras['train_noise_pattern_pred'][sigma] = pattern_injector.noise_scale(
                _om, 1.0, reference=y_train, groups=noise_groups,
                reference_groups=noise_groups)

        if oof_folds and oof_folds > 1:
            def _fp(Xf, yf, Xs):
                m = model_fn()
                m.fit(Xf, yf)
                return _tree_mean_std(m, Xs)
            om, ou, n_ok = _oof_predict(_fp, X_train, y_noisy, oof_folds,
                                        groups=train_groups)
            extras['oof_mean'][sigma] = om
            extras['oof_unc'][sigma] = ou
            extras['oof_folds_ok'][sigma] = n_ok

    return predictions, uncertainties, extras


def run_neural_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                          model_type, condition, sigma_levels, oof_folds=0,
                          train_groups=None, noise_validation=True,
                          noise_groups=None, test_noise_groups=None,
                          val_noise_groups=None, provenance=None, rep_name=None):
    """Noise robustness for neural regression model. See run_tree_experiment
    for the contents of `extras`."""
    # Same policy as the tree path and as QM9 -- see the note there.
    if rep_name is None:
        raise ValueError(
            "run_neural_experiment needs rep_name so the standardisation "
            "decision can be reported against a representation. The decision "
            "itself is made from the features, not the name.")
    if should_standardise(X_train, rep_name):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    predictions, uncertainties = {}, {}
    extras = {'train_noise_scale': {}, 'train_epsilon': {}, 'test_noise_scale': {},
              'test_noise_pattern': {}, 'train_noise_pattern': {},
              'test_noise_pattern_pred': {}, 'train_noise_pattern_pred': {},
              'oof_mean': {}, 'oof_unc': {}, 'oof_folds_ok': {},
              'gp_fit_method': ''}

    # See run_tree_experiment for why the reference pattern is recorded, and why
    # a fresh injector is built per level.
    pattern_injector = _injector_for(condition, 0)
    tr_pattern = pattern_injector.noise_scale(y_train, 1.0, reference=y_train,
                                              groups=noise_groups,
                                              reference_groups=noise_groups)
    # reference_groups=noise_groups, NOT test_noise_groups: the affected groups
    # must be the ones the TRAINING labels were exposed to, then looked up by
    # group id for the test molecules. Re-running the selection over the test
    # set's own groups picks a different set -- measured on a 40-group split,
    # two of eight corrupted groups went unmarked -- and question B would then
    # be scored against an injection that never happened.
    te_pattern = pattern_injector.noise_scale(y_test, 1.0, reference=y_train,
                                              groups=test_noise_groups,
                                              reference_groups=noise_groups)

    for sigma in sigma_levels:
        injector = _injector_for(condition, sigma_levels.index(sigma))
        result = injector.inject_verbose(y_train, sigma, groups=noise_groups)
        y_noisy, tr_scale, eps = result.y_noisy, result.noise_scale, result.epsilon
        te_scale = injector.noise_scale(y_test, sigma, reference=y_train,
                                        groups=test_noise_groups,
                                        reference_groups=noise_groups)
        if provenance is not None:
            provenance[sigma] = result.as_row()

        # FIX: the validation labels used for early stopping were left CLEAN,
        # so the four Bayesian nets were explicitly selected NOT to fit the
        # injected noise -- suppressing the very quantity this run measures, and
        # only for the neural half of the roster. They now carry the same
        # condition at the same level, drawn from an INDEPENDENT generator so
        # the training draw is untouched.
        if noise_validation and sigma > 0.0:
            # reference=y_train matters for censoring: `inject` would otherwise
            # take the cut-point from y_val itself, so validation would carry a
            # different noise geometry from training and test.
            y_val_used = _injector_for(condition, sigma_levels.index(sigma),
                                       stream='validation').inject_verbose(
                y_val, sigma, reference=y_train, groups=val_noise_groups,
                reference_groups=noise_groups).y_noisy
        else:
            y_val_used = y_val

        preds, uncs = train_neural_regression(
            X_train, y_noisy, X_val, y_val_used, X_test,
            model_type=model_type, epochs=100)
        predictions[sigma] = preds
        uncertainties[sigma] = uncs

        extras['train_noise_scale'][sigma] = tr_scale
        extras['train_epsilon'][sigma] = eps
        extras['test_noise_scale'][sigma] = te_scale
        extras['train_noise_pattern'][sigma] = tr_pattern
        extras['test_noise_pattern'][sigma] = te_pattern
        # The sham ceiling: the same pattern computed from what the model
        # PREDICTED rather than the true label. If uncertainty tracks this just
        # as strongly, the model is following its own prediction, not the noise.
        # Computed here because quantile and outlier need the training reference,
        # which is not recoverable from the output file once folds 1-4 carry no
        # training rows.
        extras['test_noise_pattern_pred'][sigma] = pattern_injector.noise_scale(
            np.asarray(predictions[sigma], dtype=float), 1.0, reference=y_train,
            groups=test_noise_groups, reference_groups=noise_groups)
        if sigma in extras['oof_mean']:
            _om = np.asarray(extras['oof_mean'][sigma], dtype=float)
            _om = np.where(np.isfinite(_om), _om, float(np.nanmean(_om)) if np.isfinite(_om).any() else 0.0)
            extras['train_noise_pattern_pred'][sigma] = pattern_injector.noise_scale(
                _om, 1.0, reference=y_train, groups=noise_groups,
                reference_groups=noise_groups)

        if oof_folds and oof_folds > 1:
            def _fp(Xf, yf, Xs):
                # Carve the early-stopping split from THIS inner fold's own fit
                # rows. Reusing the outer validation set would early-stop against
                # molecules sharing scaffolds with the rows being scored (26-42%
                # overlap on the real datasets), so the out-of-fold uncertainty
                # would not be on the same extrapolation footing as the test set.
                nv = max(1, len(yf) // 5)
                return train_neural_regression(Xf[nv:], yf[nv:], Xf[:nv], yf[:nv], Xs,
                                               model_type=model_type, epochs=100)
            # FIX: train_neural_regression consumes the GLOBAL torch generator
            # (weight init + shuffled DataLoader). Without this snapshot the
            # extra out-of-fold fits would shift the stream, so the MAIN model at
            # every later sigma would be initialised differently and this job's
            # R2/auc_norm would silently disagree with a run made without
            # --oof-folds.
            _tstate = torch.get_rng_state()
            _cstate = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
            try:
                om, ou, n_ok = _oof_predict(_fp, X_train, y_noisy, oof_folds,
                                            groups=train_groups)
            finally:
                torch.set_rng_state(_tstate)
                if _cstate is not None:
                    torch.cuda.set_rng_state_all(_cstate)
            extras['oof_mean'][sigma] = om
            extras['oof_unc'][sigma] = ou
            extras['oof_folds_ok'][sigma] = n_ok

    return predictions, uncertainties, extras


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

# The settled representation set (RERUN_PLAN.md 13.7). Avalon and ChemBERTa were
# added 2026-08-26; SNS stays because the paper describes it as ECFP4's
# collision-free counterpart.
ALL_REPS = ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained', 'Avalon', 'ChemBERTa']


def generate_representations(smiles_list, rep_filter=None):
    """Generate all representations for a SMILES list. Returns dict of arrays."""
    import gc
    reps = {}
    wanted = set(rep_filter) if rep_filter else set(ALL_REPS)

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

    if 'Avalon' in wanted:
        print("  Avalon...", flush=True)
        reps['Avalon'] = create_avalon(smiles_list, n_bits=2048)
        print(f"    done: {reps['Avalon'].shape}", flush=True)

    if 'ChemBERTa' in wanted:
        print("  ChemBERTa...", flush=True)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            reps['ChemBERTa'] = create_chemberta(smiles_list, batch_size=32)
            print(f"    done: {reps['ChemBERTa'].shape}", flush=True)
        except Exception as e:
            print(f"    FAILED: {e} — skipping ChemBERTa", flush=True)

    return reps


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def _flush_uncertainties(all_uncertainties, results_dir, dataset_name):
    """Write the accumulated per-molecule uncertainty frames to disk.

    Called after EVERY outer fold, not only at the end. The out-of-fold pass
    multiplies the fit count by (1 + oof_folds), so a task is far more likely to
    hit its wall clock than before -- and previously everything this run exists
    to produce was held in memory until all five folds had finished, so a
    timeout at fold 5 destroyed all of it while leaving an all_results_partial
    file that made the job look productive.

    Rewriting the same rows is harmless: the merge below keys on
    (split, noise_type, fold, sigma) and replaces rather than duplicates.
    """
    results_dir = Path(results_dir)
    by_config = {}
    for model_name, rep_name, unc_df in all_uncertainties:
        by_config.setdefault((model_name, rep_name), []).append(unc_df)

    for (model_name, rep_name), frames in by_config.items():
        unc_all = pd.concat(frames, ignore_index=True)
        unc_all['dataset'] = dataset_name
        unc_all['model'] = model_name
        unc_all['rep'] = rep_name
        fname = f"{model_name.replace('-', '')}_{rep_name.replace('-', '')}_uncertainty_values.csv"
        out_path = results_dir / fname
        if out_path.exists():
            try:
                prev = pd.read_csv(out_path)
                keys = [c for c in ('split', 'noise_type', 'fold', 'sigma')
                        if c in prev.columns and c in unc_all.columns]
                if keys:
                    new_keys = set(map(tuple, unc_all[keys].drop_duplicates().values))
                    prev = prev[~prev[keys].apply(tuple, axis=1).isin(new_keys)]
                unc_all = pd.concat([prev, unc_all], ignore_index=True)
            except Exception as e:
                print(f"  WARN: could not merge existing {fname}: {e}")
        tmp = out_path.with_suffix('.csv.tmp')
        unc_all.to_csv(tmp, index=False)
        os.replace(tmp, out_path)   # atomic: a killed job never leaves a half file
        print(f"  Saved {fname}: {len(unc_all)} rows "
              f"({unc_all['split'].nunique() if 'split' in unc_all else 1} split(s), "
              f"{unc_all['noise_type'].nunique() if 'noise_type' in unc_all else 1} condition(s), "
              f"{unc_all['fold'].nunique() if 'fold' in unc_all else 1} fold(s))")


def run_dataset(dataset_name, smiles, labels, results_dir, model_filter=None, rep_filter=None, sigma_levels=None,
                gp_kernel='rbf', gp_reps=None, unc_conditions='gaussian', oof_folds=0,
                conditions=None, oof_outer_folds=None, noise_validation=True):
    """Run full scaffold CV experiment for one dataset."""
    if sigma_levels is None:
        # Levels are anchored to this dataset's published assay error, in log
        # units -- see NOISE_LEVELS_BY_DATASET.
        sigma_levels = NOISE_LEVELS_BY_DATASET.get(dataset_name, NOISE_LEVELS)
    # Which noise conditions this invocation runs. A subset lets one job cover
    # one condition, which is how the uncertainty runs are split across the queue.
    conditions = list(conditions) if conditions else list(NOISE_CONDITIONS)
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"unknown noise condition(s) {unknown}. Known: {sorted(CONDITIONS)}. "
            "The six old strategy names (legacy, quantile, threshold, hetero, "
            "valprop, outlier) were deleted in noiseInject 1.0.0 -- they were one "
            "condition at six doses. See NOISE_DESIGN.md.")

    # Scaffold groups for the two grouped conditions. Separate from the CV
    # grouping: acyclic molecules are singletons here (section 2a rule 2).
    noise_groups_all, n_noise_groups, largest_share = assign_noise_groups(smiles)
    if any(c in GROUPED_CONDITIONS for c in conditions):
        print(f"  noise groups: {n_noise_groups} scaffolds, largest holds "
              f"{largest_share:.1%} of molecules")

    # Report which conditions produce a CONSTANT per-molecule noise scale on this
    # dataset. Those cannot answer "does uncertainty track WHERE the noise is" --
    # a rank correlation against a constant is undefined, not zero -- and it is
    # far better to see that here than after the cluster time is spent.
    for _c in conditions:
        _inj = NoiseInjectorRegression.from_condition(_c, random_state=0)
        _sc = _inj.noise_scale(labels, 1.0, reference=labels, groups=noise_groups_all)
        if len(np.unique(np.round(_sc, 12))) == 1:
            _note = ("expected: a shape-only condition gives every molecule the "
                     "same scale, so question B is UNDEFINED here, not zero"
                     if _inj.scale_is_degenerate is not False and
                     CONDITIONS[_c]['strategy'] == 'uniform'
                     else "DEGENERATE on this dataset -- question B cannot be "
                          "answered for this condition")
            print(f"  NOTE condition '{_c}': noise scale is CONSTANT ({_note})")
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
            lambda: RandomForestRegressor(random_state=42, n_jobs=-1,
                                          **sklearn_params('rf')), None))
        # QRF
        if HAS_QRF:
            experiments.append(('QRF', rname,
                lambda: RandomForestQuantileRegressor(random_state=42, n_jobs=-1,
                                                      **sklearn_params('qrf')), None))
        # XGBoost
        if HAS_XGBOOST:
            experiments.append(('XGBoost', rname,
                # learning_rate is the one that bit: unset falls through to the
                # booster default of 0.3, not the wrapper-documented 0.1.
                # Measured: 0.3 costs up to 0.05 R2 under injected noise.
                lambda: XGBRegressor(random_state=42,
                                     **sklearn_params('xgboost')), None))
        # NGBoost
        if HAS_NGBOOST:
            experiments.append(('NGBoost', rname,
                lambda: NGBRegressor(random_state=42, verbose=False,
                                     **_ngboost_kwargs()), None))
        # LightGBM
        if HAS_LGB:
            experiments.append(('LightGBM', rname,
                lambda: LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1,
                                      **sklearn_params('lightgbm')), None))
        # SVM — match QM9 defaults exactly: fixed RBF, C=1.0, gamma='scale',
        # for every representation. No per-rep kernel switching, no Optuna
        # tuning, so SVM is the same model on QM9 and on the validation
        # datasets and is free of the kernel--representation confound.
        if HAS_SVM:
            experiments.append(('SVM', rname,
                lambda: SVR(**sklearn_params('svm')), None))
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
        # A requested model that cannot be built must STOP the job. Previously an
        # absent optional package (gpytorch, quantile_forest, bayesian-torch)
        # only printed a warning on the first page of the log, the experiment
        # list came out empty, every fold looped over nothing, and the run
        # exited having produced no rows. That is exactly what happened to jobs
        # 12822693 and 12822694 on 2026-08-19.
        if not experiments:
            raise RuntimeError(
                f"No experiments to run: requested models {sorted(filter_set)} but none "
                f"could be constructed. Check the optional-import warnings at the top of "
                f"this log - the packages for those models are missing from this "
                f"interpreter ({sys.executable}).")
        built = {n for n, _, _, _ in experiments}
        missing = filter_set - built
        if missing:
            raise RuntimeError(
                f"Requested models {sorted(missing)} could not be constructed in this "
                f"environment ({sys.executable}). Refusing to run a partial roster - "
                f"fix the environment or drop them from --models explicitly.")
        print(f"  Filtered to models: {sorted(filter_set)}")

    print(f"\n{len(experiments)} model-rep configs × {len(conditions)} conditions "
          f"({', '.join(conditions)}) × {len(sigma_levels)} levels × {N_FOLDS} folds")

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

        # Scaffold ids of the training rows, so the inner cross-fitting split can
        # be scaffold-grouped exactly like the outer CV.
        groups_full = np.asarray(groups)[train_idx]
        groups_train = groups_full[train_idx_local]

        # Run experiments
        for exp_idx, (model_name, rep_name, model_fn, model_type) in enumerate(experiments, 1):
            print(f"\n  [{exp_idx}/{len(experiments)}] {model_name} + {rep_name}...", flush=True)

            # Get representation splits
            X_full = reps[rep_name]
            X_train_full = X_full[train_idx]
            X_test = X_full[test_idx]
            X_train = X_train_full[train_idx_local]
            X_val = X_train_full[val_idx_local]

            # Subsample for GP if needed. `gp_row_idx` keeps the position of
            # each retained molecule in the FULL training array, so the GP's
            # out-of-fold rows can be labelled with the same molecule index every
            # other model uses. Without it, joining GP to QRF on
            # (dataset, fold, sample_idx) pairs completely different molecules.
            if model_name.startswith('GP') and GP_MAX_N and len(X_train) > GP_MAX_N:
                gp_row_idx = np.random.RandomState(42).choice(
                    len(X_train), GP_MAX_N, replace=False)
                X_train_gp = X_train[gp_row_idx]
                y_train_gp = y_train[gp_row_idx]
            else:
                gp_row_idx = np.arange(len(X_train))
                X_train_gp = X_train
                y_train_gp = y_train

            for condition in conditions:
                # Censoring is not dose-matched, so it is swept on its own axis:
                # the level is the FRACTION of labels clipped, not an amount of
                # noise. Sweeping it over the dose grid would clip the same
                # fraction at every level, including the clean baseline.
                levels = (CENSORING_LEVELS
                          if CONDITIONS[condition]['strategy'] == 'censoring'
                          else sigma_levels)
                print(f"    Condition: {condition}  levels {levels}", flush=True)
                # The noise groups for THIS fold's splits, taken from the
                # dataset-wide assignment so a molecule keeps its group whichever
                # fold it lands in.
                ng_train = noise_groups_all[train_idx][train_idx_local]
                ng_val = noise_groups_all[train_idx][val_idx_local]
                ng_test = noise_groups_all[test_idx]
                provenance = {}

                try:
                    # Cross-fitting only ever helps a model that emits a
                    # per-molecule uncertainty; it is wasted compute otherwise.
                    #
                    # It is also the ONLY expensive part of this run. The five
                    # scaffold folds are trained regardless (they produce the R2
                    # numbers), so saving TEST uncertainty for all five is free.
                    # Cross-fitting, by contrast, multiplies the fits, so it can
                    # be restricted to the first N outer folds: one outer fold
                    # still gives every training molecule an out-of-fold score.
                    _oof = oof_folds if model_name in UNCERTAINTY_MODELS else 0
                    if _oof and oof_outer_folds is not None and fold_idx >= oof_outer_folds:
                        _oof = 0
                    if model_fn is not None:
                        if model_name == 'GP':
                            predictions, uncertainties, extras = run_tree_experiment(
                                X_train_gp, y_train_gp, X_test, y_test,
                                model_fn, condition, levels, oof_folds=_oof,
                                train_groups=groups_train[gp_row_idx],
                                noise_groups=ng_train[gp_row_idx],
                                test_noise_groups=ng_test,
                                provenance=provenance, rep_name=rep_name)
                            y_train_used = y_train_gp
                            row_index = gp_row_idx
                        else:
                            predictions, uncertainties, extras = run_tree_experiment(
                                X_train, y_train, X_test, y_test,
                                model_fn, condition, levels, oof_folds=_oof,
                                train_groups=groups_train,
                                noise_groups=ng_train,
                                test_noise_groups=ng_test,
                                provenance=provenance, rep_name=rep_name)
                            y_train_used = y_train
                            row_index = np.arange(len(y_train))
                    else:
                        predictions, uncertainties, extras = run_neural_experiment(
                            X_train, y_train, X_val, y_val, X_test, y_test,
                            model_type, condition, levels, oof_folds=_oof,
                            train_groups=groups_train,
                            noise_validation=noise_validation,
                            noise_groups=ng_train,
                            test_noise_groups=ng_test,
                            val_noise_groups=ng_val,
                            provenance=provenance, rep_name=rep_name)
                        y_train_used = y_train
                        row_index = np.arange(len(y_train))

                    per_sigma = compute_metrics(y_test, predictions)
                    per_sigma['model'] = model_name
                    per_sigma['rep'] = rep_name
                    per_sigma['noise_type'] = condition
                    per_sigma['fold'] = fold_idx
                    per_sigma['dataset'] = dataset_name
                    # What the noise ACTUALLY delivered, per level. Nothing
                    # recorded this before, which is the single reason the
                    # dose confound survived the life of the project.
                    # RERUN_PLAN.md section 5.2 -- what every result row must
                    # carry. `shape_name` and `targeting_name` are here as well
                    # as `noise_type` so a row can be read without parsing the
                    # condition name, and `seed` and the clean label mean and
                    # spread so a row can be reproduced from itself.
                    _prov_cols = ('shape_name', 'targeting_name',
                                  'unit_dose_g', 'solved_scale',
                                  'target_dose_label_units',
                                  'realised_dose_label_units',
                                  'realised_dose_fraction_of_spread',
                                  'mean_epsilon', 'affected_molecule_fraction',
                                  'effective_n', 'censoring_limit', 'n_groups',
                                  'largest_group_share',
                                  'clean_label_mean', 'clean_label_sd',
                                  'seed', 'scale_is_degenerate')
                    for _col in _prov_cols:
                        per_sigma[_col] = per_sigma['sigma'].map(
                            lambda _s: provenance.get(_s, {}).get(_col))
                    # Provenance: which shared parameter spec built these models.
                    # A results CSV must be traceable to the numbers behind it.
                    for _k, _v in MODEL_SPEC.provenance_columns().items():
                        per_sigma[_k] = _v
                    # Which optimiser actually fitted the GP. The Adam fallback
                    # is a different fit and used to leave no trace on disk.
                    per_sigma['gp_fit_method'] = extras.get('gp_fit_method', '')
                    all_fold_results.append(per_sigma)

                    # ---- Uncertainty output -------------------------------
                    # Was: one condition only, and the writer then overwrote the
                    # file once per fold so only the LAST fold survived and
                    # neither the condition nor the fold reached disk. Both fixed.
                    save_this = (unc_conditions == 'all') or (condition == 'gaussian')
                    if save_this and uncertainties.get(0.0) is not None:
                        # (B) TEST molecules: does the model become less certain
                        # in the regions the condition corrupts? `noise_scale` is
                        # what the molecule's region WOULD receive -- the model
                        # never saw it, so this is a genuine prediction target.
                        unc_rows = []
                        for sigma in levels:
                            ts = extras['test_noise_scale'].get(sigma)
                            tp = extras['test_noise_pattern'].get(sigma)
                            tpp = extras['test_noise_pattern_pred'].get(sigma)
                            for i in range(len(y_test)):
                                unc_rows.append({
                                    'split': 'test',
                                    'noise_type': condition,
                                    'sigma': sigma,
                                    'fold': fold_idx,
                                    'sample_idx': i,
                                    # sample_idx is a WITHIN-FOLD position, so it
                                    # repeats across folds and identifies nothing
                                    # on its own. mol_idx is the row in the full
                                    # dataset: it is what joins a molecule across
                                    # folds, conditions and models.
                                    'mol_idx': int(test_idx[i]),
                                    'y_true': y_test[i],
                                    'y_pred': predictions[sigma][i],
                                    'uncertainty': uncertainties[sigma][i],
                                    'noise_scale': (float(ts[i]) if ts is not None else np.nan),
                                    'noise_pattern': (float(tp[i]) if tp is not None else np.nan),
                                    'noise_pattern_pred': (float(tpp[i]) if tpp is not None else np.nan),
                                    'injected_noise': 0.0,   # test labels are never corrupted
                                    # Emitted on EVERY row, test included. If the column
                                    # appeared only where out-of-fold rows exist, task files
                                    # would have different column sets and the streaming merge
                                    # -- which writes one header and appends -- would shift
                                    # every later column silently.
                                    'oof_folds_ok': -1,
                                })
                        all_uncertainties.append((model_name, rep_name, pd.DataFrame(unc_rows)))

                        # (A) TRAINING molecules, scored out-of-fold: do the
                        # molecules whose labels were actually corrupted come
                        # back as the uncertain ones? `injected_noise` here is
                        # the epsilon the injector really drew, not a regression
                        # estimate.
                        if extras['oof_unc']:
                            tr_rows = []
                            for sigma in levels:
                                ou = extras['oof_unc'].get(sigma)
                                om = extras['oof_mean'].get(sigma)
                                if ou is None:
                                    continue
                                eps = extras['train_epsilon'][sigma]
                                sc = extras['train_noise_scale'][sigma]
                                pt = extras['train_noise_pattern'][sigma]
                                ptp = extras['train_noise_pattern_pred'].get(sigma)
                                if not np.isfinite(ou).any():
                                    # Every inner fold failed. Writing this block
                                    # would put an all-blank cell into the output
                                    # that the coverage report counts as present.
                                    print(f"      [oof] sigma={sigma}: no finite "
                                          f"out-of-fold values — block skipped",
                                          flush=True)
                                    continue
                                n_ok = extras['oof_folds_ok'].get(sigma, 0)
                                for i in range(len(y_train_used)):
                                    tr_rows.append({
                                        'split': 'train_oof',
                                        'noise_type': condition,
                                        'sigma': sigma,
                                        'fold': fold_idx,
                                        'oof_folds_ok': n_ok,
                                        'sample_idx': int(row_index[i]),
                                        'mol_idx': int(train_idx[train_idx_local[int(row_index[i])]]),
                                        'y_true': y_train_used[i],
                                        'y_pred': om[i],
                                        'uncertainty': ou[i],
                                        'noise_scale': float(sc[i]),
                                        'noise_pattern': float(pt[i]),
                                        'noise_pattern_pred': (float(ptp[i])
                                                               if ptp is not None else np.nan),
                                        'injected_noise': float(eps[i]),
                                    })
                            if tr_rows:
                                all_uncertainties.append(
                                    (model_name, rep_name, pd.DataFrame(tr_rows)))

                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    continue

        # Incremental save after each fold (protection against timeout)
        if all_uncertainties:
            _flush_uncertainties(all_uncertainties, results_dir, dataset_name)
        if all_fold_results:
            _partial = pd.concat(all_fold_results, ignore_index=True)
            _partial_path = results_dir / 'all_results_partial.csv'
            _partial.to_csv(_partial_path, index=False)
            print(f"  [Checkpoint] Saved {len(_partial)} rows after fold {fold_idx + 1}")

    if not all_fold_results:
        # Every (model, condition) raised. Returning quietly here made the job
        # exit 0 with no output, which is indistinguishable from success in
        # sacct and would have been discovered only at merge time.
        raise RuntimeError(
            f"No results produced for {dataset_name} — every model/condition "
            f"combination raised. Check the per-condition ERROR lines above.")

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
    for (model, rep, strat), grp in combined.groupby(['model', 'rep', 'noise_type']):
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
            'noise_type': strat,
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

    # Save uncertainty data (final pass; also written incrementally per fold).
    _flush_uncertainties(all_uncertainties, results_dir, dataset_name)

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
            print(f"    {row['model']:12} + {row['rep']:20} + {row['noise_type']:10}: "
                  f"auc_norm={row['auc_norm']:.4f}, baseline={row['baseline_r2']:.3f}±{row['baseline_r2_std']:.3f}")

        print(f"\n  By representation (mean across models/conditions):")
        rep_agg = working.groupby('rep')[['baseline_r2', 'auc_norm', 'retention_0.5']].mean()
        print(rep_agg.round(4).to_string())

        print(f"\n  By model (mean across reps/conditions):")
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
                        choices=ALL_REPS,
                        help='Only run these reps (e.g. --reps PDV). Merges with existing results.')
    parser.add_argument('--sigmas', nargs='+', type=float, default=None,
                        help='Only run these sigma levels (e.g. --sigmas 0.0 0.5 1.0). Default: all 11.')
    parser.add_argument('--gp-kernel', type=str, default='rbf', choices=['rbf', 'tanimoto'],
                        help="GP (gauche) kernel. 'rbf' (default) is valid on all reps and matches "
                             "QM9 gauche_rbf; 'tanimoto' is only valid on binary fingerprints (ECFP4).")
    parser.add_argument('--gp-reps', nargs='+', default=None,
                        choices=ALL_REPS,
                        help='Reps on which to run the GP. Default: PDV only (original behaviour). '
                             'Pass all four to run GP on every rep so it can enter the cross-rep ANOVA.')
    parser.add_argument('--conditions', nargs='+', default=None,
                        choices=sorted(CONDITIONS),
                        help='Only run these noise conditions (default: NOISE_CONDITIONS). '
                             'Splitting a run by condition is how the '
                             'uncertainty jobs are spread across the queue. Every condition '
                             'delivers the SAME amount of noise at a given level, so any '
                             'difference between them is a difference of shape.')
    parser.add_argument('--unc-conditions', type=str, default='gaussian',
                        choices=['gaussian', 'all'],
                        help="Which noise conditions get their per-molecule uncertainty written "
                             "to disk. 'gaussian' (default) is the cheap case. 'all' is needed to "
                             "ask whether a model learns WHERE the data is unreliable, since the "
                             "shape-only conditions give every molecule the same noise scale and "
                             "so have no pattern to learn -- for those the question is UNDEFINED, "
                             "not zero. grouped_wider and the censoring conditions are the ones "
                             "with a pattern to find.")
    parser.add_argument('--no-noise-validation', action='store_true',
                        help="Leave the early-stopping validation labels CLEAN. Default is to "
                             "corrupt them with the same condition and level as training: "
                             "otherwise the neural models are explicitly selected NOT to fit the "
                             "injected noise, which suppresses the quantity being measured and "
                             "does so for the neural half of the roster only.")
    parser.add_argument('--oof-outer-folds', type=int, default=None,
                        help="Cross-fit only the first N of the 5 scaffold folds (default: all "
                             "of them). The 5 folds are trained either way, so TEST-side "
                             "uncertainty is free for all 5; this only limits the expensive "
                             "out-of-fold TRAINING pass. --oof-outer-folds 1 cuts the added "
                             "cost about threefold and still gives every training molecule an "
                             "out-of-fold score, just without a spread across folds.")
    parser.add_argument('--oof-folds', type=int, default=0,
                        help="Cross-fit the training set into this many folds and save an "
                             "out-of-fold uncertainty for every TRAINING molecule (0 = off, "
                             "the default). Needed to ask whether uncertainty flags the labels "
                             "that were actually corrupted: scoring a molecule with a model that "
                             "fitted its corrupted label measures memorisation, not uncertainty. "
                             "Costs this many extra fits per noise level, and only applies to the "
                             "models that emit a per-molecule uncertainty.")
    args = parser.parse_args()

    ds_list = ['logd', 'caco2', 'herg_ki'] if 'all' in args.datasets else args.datasets
    # None means: use each dataset's own anchored grid (NOISE_LEVELS_BY_DATASET).
    sigma_levels = args.sigmas if args.sigmas else None

    print("=" * 80)
    print("Validation Noise Robustness Experiment")
    print("=" * 80)
    print(f"  Datasets: {ds_list}")
    print(f"  Split: {N_FOLDS}-fold scaffold CV")
    rep_list = ", ".join(args.reps) if args.reps else ", ".join(ALL_REPS)
    print(f"  Representations: {rep_list}")
    print(f"  Noise conditions: {args.conditions if args.conditions else NOISE_CONDITIONS}")
    print(f"  Noise levels: {sigma_levels if sigma_levels else 'per dataset, anchored to assay error'}")
    model_list = ("RF, QRF, XGBoost, NGBoost, LightGBM, SVM, GP(PDV), "
                  "DNN, VBLL-Full, MLP, MLP-VBLL-Full")
    if HAS_BAYESIAN_TORCH:
        model_list += ", BNN-Full, MLP-BNN-Full"
    if args.models:
        model_list = ", ".join(args.models)
    print(f"  Models: {model_list}")
    print(f"  Uncertainty saved for conditions: {args.unc_conditions}")
    print(f"  Out-of-fold training uncertainty: "
          f"{args.oof_folds if args.oof_folds > 1 else 'off'}"
          f"{' folds, models: ' + ','.join(UNCERTAINTY_MODELS) if args.oof_folds > 1 else ''}"
          f"{'' if not args.oof_folds or args.oof_outer_folds is None else f', on the first {args.oof_outer_folds} of {N_FOLDS} scaffold folds'}")
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
                                  gp_kernel=args.gp_kernel, gp_reps=args.gp_reps,
                                  unc_conditions=args.unc_conditions, oof_folds=args.oof_folds,
                                  conditions=args.conditions,
                                  oof_outer_folds=args.oof_outer_folds,
                                  noise_validation=not args.no_noise_validation,)
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
                                  gp_kernel=args.gp_kernel, gp_reps=args.gp_reps,
                                  unc_conditions=args.unc_conditions, oof_folds=args.oof_folds,
                                  conditions=args.conditions,
                                  oof_outer_folds=args.oof_outer_folds,
                                  noise_validation=not args.no_noise_validation,)
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
                              sigma_levels=sigma_levels,   # was omitted: --sigmas was ignored for hERG
                              gp_kernel=args.gp_kernel, gp_reps=args.gp_reps,
                              unc_conditions=args.unc_conditions, oof_folds=args.oof_folds,
                              conditions=args.conditions,
                              oof_outer_folds=args.oof_outer_folds,
                              noise_validation=not args.no_noise_validation,)
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
