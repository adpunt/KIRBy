#!/usr/bin/env python3
"""
Hybrid Effectiveness Analysis

In-depth analysis of KIRBy hybrid representations:
1. Individual representation performance baselines
2. Hybrid vs individual comparison
3. Allocation strategy comparison (greedy vs fixed vs performance-weighted)
4. Augmentation impact analysis
5. Feature importance distributions
6. Cross-validation stability

Uses real molecular datasets for meaningful results.

Usage:
    python tests/hybrid_effectiveness_analysis.py                    # Full analysis
    python tests/hybrid_effectiveness_analysis.py --quick            # Quick version (fewer samples)
    python tests/hybrid_effectiveness_analysis.py --dataset esol     # Specific dataset
"""

import sys
import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for the analysis."""
    n_samples: int = 1000         # Max samples to use (increased)
    test_size: float = 0.2        # Test split ratio
    n_cv_folds: int = 5           # Cross-validation folds
    random_state: int = 42
    budget: int = 500             # Feature budget for hybrid (increased from 100)
    step_size: int = 50           # Greedy step size (increased)
    augmentation_budget: int = 50 # Features per augmentation (increased)

    # Which representations to test - ALL of them
    base_reps: List[str] = field(default_factory=lambda: [
        'ecfp4', 'maccs', 'pdv', 'sns'  # Added SNS
    ])
    pretrained_reps: List[str] = field(default_factory=lambda: [
        'chemberta', 'molformer', 'mol2vec'  # Added mol2vec
    ])
    augmentations: List[str] = field(default_factory=lambda: [
        'graph_topology', 'spectral', 'subgraph_counts', 'graph_distances'
    ])


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class Dataset:
    name: str
    smiles: List[str]
    labels: np.ndarray
    task_type: str = 'regression'


def load_dataset(name: str, max_samples: int = 1000) -> Dataset:
    """Load a molecular dataset.

    Available datasets:
    - esol, lipophilicity, freesolv: DeepChem MolNet
    - genentech_hlm, genentech_solubility: Genentech Computational-ADME
    - az_solubility: AstraZeneca ChEMBL Solubility
    - herg: ChEMBL hERG Ki
    """
    from rdkit import Chem
    import requests
    from pathlib import Path

    CACHE_DIR = Path("data_cache")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def extract_smiles(data):
        """Extract SMILES from DeepChem dataset, handling various formats."""
        ids = data.ids
        if isinstance(ids[0], str):
            return list(ids)
        return [x[0] if hasattr(x, '__iter__') and not isinstance(x, str) else str(x) for x in ids]

    def validate_smiles(smiles_list, labels):
        """Filter to only valid SMILES."""
        valid_smiles = []
        valid_labels = []
        for smi, label in zip(smiles_list, labels):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and len(smi) > 1:
                valid_smiles.append(smi)
                valid_labels.append(label)
        return valid_smiles, np.array(valid_labels)

    def standardise_smiles(smi: str):
        """Canonicalise SMILES, keep largest fragment."""
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

    def download_file(url: str, dest: Path) -> Path:
        """Download a file if not cached."""
        if dest.exists():
            return dest
        print(f"  Downloading {dest.name}...")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest

    smiles = None
    labels = None

    # ══════════════════════════════════════════════════════════════════════
    # DeepChem MolNet datasets
    # ══════════════════════════════════════════════════════════════════════
    if name in ['esol', 'lipophilicity', 'freesolv']:
        try:
            if name == 'esol':
                from deepchem.molnet import load_delaney
                tasks, datasets, _ = load_delaney(featurizer='Raw', splitter=None)
            elif name == 'lipophilicity':
                from deepchem.molnet import load_lipo
                tasks, datasets, _ = load_lipo(featurizer='Raw', splitter=None)
            elif name == 'freesolv':
                from deepchem.molnet import load_freesolv
                tasks, datasets, _ = load_freesolv(featurizer='Raw', splitter=None)

            data = datasets[0]
            smiles = extract_smiles(data)
            labels = data.y.flatten()
        except ImportError:
            print(f"DeepChem not available, using fallback data for {name}")
            return _load_fallback_data(name, max_samples)

    # ══════════════════════════════════════════════════════════════════════
    # Genentech Computational-ADME datasets
    # ══════════════════════════════════════════════════════════════════════
    elif name.startswith('genentech_'):
        csv_url = ("https://raw.githubusercontent.com/molecularinformatics/"
                   "Computational-ADME/main/ADME_public_set_3521.csv")
        dest = CACHE_DIR / "genentech_adme_3521.csv"
        download_file(csv_url, dest)

        df = pd.read_csv(dest)

        # Find SMILES column (case-insensitive)
        smiles_col = None
        for col in df.columns:
            if col.upper() == 'SMILES':
                smiles_col = col
                break
        if smiles_col is None:
            raise ValueError(f"No SMILES column found in Genentech data. Columns: {list(df.columns)}")

        # Map dataset name to column (exact column names from CSV)
        endpoint_map = {
            'genentech_hlm': 'LOG HLM_CLint (mL/min/kg)',
            'genentech_rlm': 'LOG RLM_CLint (mL/min/kg)',
            'genentech_solubility': 'LOG SOLUBILITY PH 6.8 (ug/mL)',
            'genentech_hppb': 'LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)',
            'genentech_rppb': 'LOG PLASMA PROTEIN BINDING (RAT) (% unbound)',
            'genentech_mdr1': 'LOG MDR1-MDCK ER (B-A/A-B)',
        }

        if name not in endpoint_map:
            target_col = 'LOG HLM_CLint (mL/min/kg)'
        else:
            target_col = endpoint_map[name]

        if target_col not in df.columns:
            # Try partial match
            matches = [c for c in df.columns if name.split('_')[1].upper() in c.upper()]
            target_col = matches[0] if matches else df.select_dtypes(include=[np.number]).columns[0]

        print(f"  Using SMILES column: {smiles_col}, target column: {target_col}")
        df = df.dropna(subset=[smiles_col, target_col])
        df['std_smiles'] = df[smiles_col].apply(standardise_smiles)
        df = df.dropna(subset=['std_smiles'])
        df = df.groupby('std_smiles').agg({target_col: 'median'}).reset_index()

        smiles = df['std_smiles'].tolist()
        labels = df[target_col].values

    # ══════════════════════════════════════════════════════════════════════
    # AstraZeneca ChEMBL Solubility
    # ══════════════════════════════════════════════════════════════════════
    elif name == 'az_solubility':
        csv_url = "https://data.dgl.ai/dataset/AstraZeneca_ChEMBL_Solubility.csv"
        dest = CACHE_DIR / "az_chembl_solubility.csv"
        download_file(csv_url, dest)

        df = pd.read_csv(dest)
        smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
        sol_col = df.select_dtypes(include=[np.number]).columns[0]

        df = df.dropna(subset=[smiles_col, sol_col])
        df['std_smiles'] = df[smiles_col].apply(standardise_smiles)
        df = df.dropna(subset=['std_smiles'])

        # Apply log10 transform (values are in nM)
        df[sol_col] = np.log10(np.clip(df[sol_col].values, 1e-10, None))
        df = df.groupby('std_smiles').agg({sol_col: 'median'}).reset_index()

        smiles = df['std_smiles'].tolist()
        labels = df[sol_col].values

    # ══════════════════════════════════════════════════════════════════════
    # ChEMBL hERG Ki (cached extraction)
    # ══════════════════════════════════════════════════════════════════════
    elif name == 'herg':
        cached = CACHE_DIR / "chembl_herg_ki.csv"
        if not cached.exists():
            print("  Fetching hERG Ki from ChEMBL API (this may take a minute)...")
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
                    "limit": limit, "offset": offset,
                }
                try:
                    resp = requests.get(base_url, params=params, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    print(f"  ChEMBL API error: {e}")
                    break

                activities = data.get("activities", [])
                if not activities:
                    break

                for act in activities:
                    all_records.append({
                        "SMILES": act.get("canonical_smiles"),
                        "pChEMBL": act.get("pchembl_value"),
                    })

                if data.get("page_meta", {}).get("next") is None:
                    break
                offset += limit
                import time
                time.sleep(0.3)

            if all_records:
                df = pd.DataFrame(all_records)
                df = df.dropna()
                df['pChEMBL'] = pd.to_numeric(df['pChEMBL'], errors='coerce')
                df = df.dropna()
                df = df.groupby('SMILES')['pChEMBL'].median().reset_index()
                df.to_csv(cached, index=False)

        df = pd.read_csv(cached)
        df['std_smiles'] = df['SMILES'].apply(standardise_smiles)
        df = df.dropna(subset=['std_smiles'])

        smiles = df['std_smiles'].tolist()
        labels = df['pChEMBL'].values

    # ══════════════════════════════════════════════════════════════════════
    # OpenADMET-ExpansionRx (HuggingFace)
    # ══════════════════════════════════════════════════════════════════════
    elif name.startswith('openadmet_'):
        # Try raw dataset first (has more data), fall back to train
        raw_url = ("https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data/"
                   "resolve/main/expansion_data_raw.csv")
        raw_dest = CACHE_DIR / "openadmet_raw.csv"
        train_url = ("https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data/"
                     "resolve/main/expansion_data_train.csv")
        train_dest = CACHE_DIR / "openadmet_train.csv"

        try:
            download_file(raw_url, raw_dest)
            dest = raw_dest
        except Exception:
            download_file(train_url, train_dest)
            dest = train_dest

        df = pd.read_csv(dest)

        # Find SMILES column
        smiles_col = None
        for col in df.columns:
            if col.upper() == 'SMILES':
                smiles_col = col
                break
        if smiles_col is None:
            raise ValueError(f"No SMILES column in OpenADMET. Columns: {list(df.columns)}")

        # Map endpoint names
        endpoint_map = {
            'openadmet_logd': 'LogD',
            'openadmet_ksol': 'KSOL',
            'openadmet_hlm': 'HLM CLint',
            'openadmet_mlm': 'MLM CLint',
            'openadmet_caco2': 'Caco-2 Permeability Papp',
            'openadmet_caco2_efflux': 'Caco-2 Permeability Efflux',
        }

        if name not in endpoint_map:
            target_pattern = 'LogD'
        else:
            target_pattern = endpoint_map[name]

        # Find target column
        target_col = None
        for col in df.columns:
            if target_pattern in col:
                target_col = col
                break
        if target_col is None:
            raise ValueError(f"Target column '{target_pattern}' not found. Columns: {list(df.columns)}")

        print(f"  Using SMILES column: {smiles_col}, target column: {target_col}")
        df = df.dropna(subset=[smiles_col, target_col])

        # Convert target to numeric (raw dataset has strings)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col])

        df['std_smiles'] = df[smiles_col].apply(standardise_smiles)
        df = df.dropna(subset=['std_smiles'])

        # Apply log transform for some endpoints (matches test_datasets.py)
        if name in ['openadmet_ksol', 'openadmet_hlm', 'openadmet_mlm',
                    'openadmet_caco2', 'openadmet_caco2_efflux']:
            df[target_col] = np.log10(np.clip(df[target_col].values, 1e-10, None))
            # Remove infinites after transform
            df = df[np.isfinite(df[target_col])].reset_index(drop=True)

        df = df.groupby('std_smiles').agg({target_col: 'median'}).reset_index()

        smiles = df['std_smiles'].tolist()
        labels = df[target_col].values

    else:
        raise ValueError(f"Unknown dataset: {name}. Available: esol, lipophilicity, freesolv, "
                        "genentech_hlm, genentech_solubility, az_solubility, herg, openadmet_logd")

    if smiles is None:
        return _load_fallback_data(name, max_samples)

    # Validate SMILES
    n_before = len(smiles)
    smiles, labels = validate_smiles(smiles, labels)
    n_after = len(smiles)
    if n_before != n_after:
        print(f"  Filtered {n_before - n_after} invalid SMILES")

    # Subsample if needed
    if len(smiles) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(smiles), max_samples, replace=False)
        smiles = [smiles[i] for i in idx]
        labels = labels[idx]

    print(f"Loaded {name}: {len(smiles)} molecules")
    return Dataset(name, smiles, labels)


def _load_fallback_data(name: str, max_samples: int) -> Dataset:
    """Generate synthetic-ish data when DeepChem unavailable."""
    # Use a set of real SMILES with synthetic labels
    base_smiles = [
        'CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'CC(C)CO',
        'c1ccccc1', 'c1ccc(O)cc1', 'c1ccc(N)cc1', 'c1ccc(C)cc1',
        'CC(=O)O', 'CCC(=O)O', 'CCCC(=O)O', 'CC(C)(C)C(=O)O',
        'CCN', 'CCCN', 'CCCCN', 'CC(C)N', 'c1ccc(CN)cc1',
        'CCOC(=O)C', 'CCOC(=O)CC', 'COC(=O)c1ccccc1',
        'CC(=O)Nc1ccccc1', 'c1ccc2ccccc2c1', 'c1ccc2c(c1)cccc2',
        'CCCCCc1ccccc1', 'CCOc1ccccc1', 'COc1ccc(O)cc1',
    ]

    # Duplicate and vary to reach max_samples
    np.random.seed(42)
    smiles = []
    while len(smiles) < max_samples:
        smiles.extend(base_smiles)
    smiles = smiles[:max_samples]

    # Generate synthetic labels based on simple properties
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    labels = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Combine a few descriptors + noise for synthetic property
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            val = logp + 0.01 * mw + np.random.randn() * 0.5
        else:
            val = np.random.randn()
        labels.append(val)

    return Dataset(f"{name}_synthetic", smiles, np.array(labels))


# =============================================================================
# REPRESENTATION GENERATION
# =============================================================================

def generate_representations(smiles: List[str], config: AnalysisConfig,
                            include_pretrained: bool = True) -> Tuple[Dict, Dict]:
    """
    Generate all representations for the given SMILES.

    Returns:
        base_reps: Dict of base representations
        augmentations: Dict of augmentation features
    """
    from kirby.representations.molecular import (
        create_ecfp4, create_maccs, create_pdv, create_sns,
        create_chemberta, create_molformer, create_mol2vec,
        compute_graph_topology, compute_spectral_features,
        compute_subgraph_counts, compute_graph_distances
    )

    base_reps = {}
    augmentations = {}

    print(f"\nGenerating representations for {len(smiles)} molecules...")

    # Base representations (fast fingerprints)
    for rep_name in config.base_reps:
        t0 = time.time()
        try:
            if rep_name == 'ecfp4':
                base_reps['ecfp4'] = create_ecfp4(smiles)
            elif rep_name == 'maccs':
                base_reps['maccs'] = create_maccs(smiles)
            elif rep_name == 'pdv':
                base_reps['pdv'] = create_pdv(smiles)
            elif rep_name == 'sns':
                base_reps['sns'] = create_sns(smiles)
            print(f"  {rep_name}: {base_reps[rep_name].shape} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  {rep_name}: FAILED ({e})")

    # Pretrained representations (slower, frozen embeddings)
    # Use small batch size to reduce memory usage
    pretrained_batch_size = 8  # Reduced from 32 to avoid OOM
    if include_pretrained:
        for rep_name in config.pretrained_reps:
            t0 = time.time()
            try:
                if rep_name == 'chemberta':
                    base_reps['chemberta'] = create_chemberta(smiles, batch_size=pretrained_batch_size)
                elif rep_name == 'molformer':
                    base_reps['molformer'] = create_molformer(smiles, batch_size=pretrained_batch_size)
                elif rep_name == 'mol2vec':
                    base_reps['mol2vec'] = create_mol2vec(smiles)
                print(f"  {rep_name}: {base_reps[rep_name].shape} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  {rep_name}: FAILED ({e})")

    # Augmentations
    print("\nGenerating augmentations...")
    for aug_name in config.augmentations:
        t0 = time.time()
        if aug_name == 'graph_topology':
            augmentations['graph_topology'] = compute_graph_topology(smiles)
        elif aug_name == 'spectral':
            augmentations['spectral'] = compute_spectral_features(smiles, k=10)
        elif aug_name == 'subgraph_counts':
            augmentations['subgraph_counts'] = compute_subgraph_counts(smiles)
        elif aug_name == 'graph_distances':
            augmentations['graph_distances'] = compute_graph_distances(smiles)
        print(f"  {aug_name}: {augmentations[aug_name].shape} ({time.time()-t0:.1f}s)")

    return base_reps, augmentations


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class EvaluationResult:
    """Results from a single evaluation."""
    name: str
    r2_train: float
    r2_test: float
    rmse_test: float
    mae_test: float
    n_features: int
    cv_r2_mean: float = None
    cv_r2_std: float = None


def evaluate_representation(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    name: str,
    n_cv_folds: int = 5,
    random_state: int = 42,
    use_lgbm: bool = True
) -> EvaluationResult:
    """Evaluate a single representation using LightGBM or RandomForest."""

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model - use LightGBM by default (much better performance)
    if use_lgbm:
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=random_state,
                verbose=-1,
            )
            cv_model = lgb.LGBMRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                num_leaves=63, min_child_samples=10, subsample=0.8,
                colsample_bytree=0.3, reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, random_state=random_state, verbose=-1,
            )
        except ImportError:
            use_lgbm = False

    if not use_lgbm:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        cv_model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)

    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation on training set
    cv_scores = cross_val_score(
        cv_model, X_train_scaled, y_train, cv=n_cv_folds, scoring='r2'
    )

    return EvaluationResult(
        name=name,
        r2_train=r2_train,
        r2_test=r2_test,
        rmse_test=rmse_test,
        mae_test=mae_test,
        n_features=X_train.shape[1],
        cv_r2_mean=cv_scores.mean(),
        cv_r2_std=cv_scores.std()
    )


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_individual_representations(
    base_reps: Dict[str, np.ndarray],
    augmentations: Dict[str, np.ndarray],
    labels: np.ndarray,
    config: AnalysisConfig
) -> List[EvaluationResult]:
    """Analyze each representation individually."""

    print("\n" + "=" * 70)
    print("INDIVIDUAL REPRESENTATION ANALYSIS")
    print("=" * 70)

    results = []

    # Split data
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=config.test_size,
        random_state=config.random_state
    )
    y_train, y_test = labels[train_idx], labels[test_idx]

    # Evaluate each base representation
    print("\n[Base Representations]")
    for name, X in base_reps.items():
        X_train, X_test = X[train_idx], X[test_idx]
        result = evaluate_representation(
            X_train, X_test, y_train, y_test, name,
            config.n_cv_folds, config.random_state
        )
        results.append(result)
        print(f"  {name:15s}: R²={result.r2_test:.4f} (CV: {result.cv_r2_mean:.4f}±{result.cv_r2_std:.4f}) | {result.n_features} features")

    # Evaluate each augmentation
    print("\n[Augmentations (standalone)]")
    for name, X in augmentations.items():
        X_train, X_test = X[train_idx], X[test_idx]
        result = evaluate_representation(
            X_train, X_test, y_train, y_test, f"aug_{name}",
            config.n_cv_folds, config.random_state
        )
        results.append(result)
        print(f"  {name:15s}: R²={result.r2_test:.4f} (CV: {result.cv_r2_mean:.4f}±{result.cv_r2_std:.4f}) | {result.n_features} features")

    # Simple concatenation baseline
    print("\n[Concatenation Baselines]")

    # All base reps concatenated
    X_all_base = np.hstack(list(base_reps.values()))
    X_train, X_test = X_all_base[train_idx], X_all_base[test_idx]
    result = evaluate_representation(
        X_train, X_test, y_train, y_test, "concat_all_base",
        config.n_cv_folds, config.random_state
    )
    results.append(result)
    print(f"  {'all_base':15s}: R²={result.r2_test:.4f} (CV: {result.cv_r2_mean:.4f}±{result.cv_r2_std:.4f}) | {result.n_features} features")

    # All base + all augmentations
    X_all = np.hstack(list(base_reps.values()) + list(augmentations.values()))
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    result = evaluate_representation(
        X_train, X_test, y_train, y_test, "concat_all",
        config.n_cv_folds, config.random_state
    )
    results.append(result)
    print(f"  {'all_base+aug':15s}: R²={result.r2_test:.4f} (CV: {result.cv_r2_mean:.4f}±{result.cv_r2_std:.4f}) | {result.n_features} features")

    return results


def analyze_hybrid_strategies(
    base_reps: Dict[str, np.ndarray],
    augmentations: Dict[str, np.ndarray],
    labels: np.ndarray,
    config: AnalysisConfig
) -> List[EvaluationResult]:
    """Compare different hybrid creation strategies."""

    from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection

    print("\n" + "=" * 70)
    print("HYBRID STRATEGY COMPARISON")
    print("=" * 70)

    results = []

    # Split data
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=config.test_size,
        random_state=config.random_state
    )
    y_train, y_test = labels[train_idx], labels[test_idx]

    # Prepare train/test splits for reps
    base_train = {k: v[train_idx] for k, v in base_reps.items()}
    base_test = {k: v[test_idx] for k, v in base_reps.items()}
    aug_train = {k: v[train_idx] for k, v in augmentations.items()}
    aug_test = {k: v[test_idx] for k, v in augmentations.items()}

    strategies = [
        # (name, allocation_method, aug_strategy, kwargs)
        ("greedy_no_aug", "greedy", "none", {}),
        ("greedy_all_aug", "greedy", "all", {}),
        ("greedy_ablation", "greedy", "greedy_ablation", {}),
        ("fixed_no_aug", "fixed", "none", {"n_per_rep": config.budget // len(base_reps)}),
        ("fixed_all_aug", "fixed", "all", {"n_per_rep": config.budget // len(base_reps)}),
    ]

    print(f"\nBudget: {config.budget} features | Augmentation budget: {config.augmentation_budget}/aug")
    print("-" * 70)

    for name, alloc_method, aug_strategy, kwargs in strategies:
        t0 = time.time()
        try:
            # Create hybrid on training data
            hybrid_train, feature_info = create_hybrid(
                base_train, y_train,
                allocation_method=alloc_method,
                budget=config.budget,
                step_size=config.step_size,
                augmentations=aug_train if aug_strategy != "none" else None,
                augmentation_strategy=aug_strategy,
                augmentation_budget=config.augmentation_budget,
                **kwargs
            )

            # Apply same selection to test
            hybrid_test = apply_feature_selection(base_test, feature_info)

            # Add augmentation features if kept
            if aug_strategy != "none" and 'augmentation_info' in feature_info:
                aug_feats = apply_augmentation_selection(aug_test, feature_info['augmentation_info'])
                if aug_feats is not None:
                    hybrid_test = np.hstack([hybrid_test, aug_feats])

            # Evaluate
            result = evaluate_representation(
                hybrid_train, hybrid_test, y_train, y_test, name,
                config.n_cv_folds, config.random_state
            )
            results.append(result)

            # Show allocation details
            if 'allocation' in feature_info:
                alloc_str = ", ".join([f"{k}:{v}" for k, v in feature_info['allocation'].items() if v > 0])
            else:
                alloc_str = "N/A"

            kept_aug = feature_info.get('kept_augmentations', [])
            aug_str = f"aug:[{','.join(kept_aug)}]" if kept_aug else "no_aug"

            print(f"  {name:20s}: R²={result.r2_test:.4f} (CV:{result.cv_r2_mean:.4f}±{result.cv_r2_std:.4f}) | "
                  f"{result.n_features} feats | {alloc_str} | {aug_str} ({time.time()-t0:.1f}s)")

        except Exception as e:
            print(f"  {name:20s}: FAILED - {e}")

    return results


def analyze_feature_importance(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    config: AnalysisConfig
) -> Dict:
    """Analyze feature importance distributions across representations."""

    from kirby.hybrid import create_hybrid

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Create hybrid with greedy allocation
    hybrid, feature_info = create_hybrid(
        base_reps, labels,
        allocation_method='greedy',
        budget=config.budget,
        step_size=config.step_size
    )

    print(f"\nGreedy allocation results (budget={config.budget}):")
    print("-" * 50)

    allocation = feature_info.get('allocation', {})
    total_allocated = sum(allocation.values())

    importance_stats = {}

    for rep_name, n_features in allocation.items():
        if n_features == 0:
            continue

        rep_info = feature_info.get(rep_name, {})
        importances = rep_info.get('importance_scores', np.array([]))

        total_features = base_reps[rep_name].shape[1]
        pct_selected = 100 * n_features / total_features if total_features > 0 else 0
        pct_budget = 100 * n_features / total_allocated if total_allocated > 0 else 0

        importance_stats[rep_name] = {
            'n_selected': n_features,
            'total_features': total_features,
            'pct_selected': pct_selected,
            'pct_budget': pct_budget,
            'importance_mean': importances.mean() if len(importances) > 0 else 0,
            'importance_std': importances.std() if len(importances) > 0 else 0,
        }

        print(f"  {rep_name:12s}: {n_features:4d}/{total_features:4d} selected ({pct_selected:5.1f}%) | "
              f"{pct_budget:5.1f}% of budget | imp_mean={importance_stats[rep_name]['importance_mean']:.4f}")

    # Show greedy history if available
    if 'greedy_history' in feature_info:
        history = feature_info['greedy_history']
        print(f"\nGreedy iteration history ({len(history)} iterations):")
        for h in history[-5:]:  # Show last 5
            print(f"  Iter {h['iteration']}: +{h['rep']} -> R²={h['val_r2']:.4f}")

    return importance_stats


def create_summary_table(all_results: List[EvaluationResult]) -> pd.DataFrame:
    """Create a summary DataFrame of all results."""

    data = []
    for r in all_results:
        data.append({
            'Method': r.name,
            'R² (test)': r.r2_test,
            'R² (CV mean)': r.cv_r2_mean,
            'R² (CV std)': r.cv_r2_std,
            'RMSE': r.rmse_test,
            'MAE': r.mae_test,
            'N Features': r.n_features,
        })

    df = pd.DataFrame(data)
    df = df.sort_values('R² (test)', ascending=False)
    return df


# =============================================================================
# MAIN
# =============================================================================

AVAILABLE_DATASETS = [
    'esol', 'lipophilicity', 'freesolv',  # DeepChem MolNet
    'genentech_hlm', 'genentech_rlm', 'genentech_solubility',  # Genentech ADME
    'genentech_hppb', 'genentech_rppb', 'genentech_mdr1',
    'az_solubility',  # AstraZeneca
    'openadmet_logd', 'openadmet_ksol', 'openadmet_hlm',  # OpenADMET-ExpansionRx
    'openadmet_caco2_efflux',  # Caco-2 Efflux (smaller dataset)
    'herg',  # ChEMBL hERG Ki
]


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Effectiveness Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  DeepChem MolNet:     esol, lipophilicity, freesolv
  Genentech ADME:      genentech_hlm, genentech_rlm, genentech_solubility,
                       genentech_hppb, genentech_rppb, genentech_mdr1
  AstraZeneca:         az_solubility
  ChEMBL:              herg

Examples:
  python hybrid_effectiveness_analysis.py --dataset herg
  python hybrid_effectiveness_analysis.py --dataset genentech_hlm --budget 500
  python hybrid_effectiveness_analysis.py --all-datasets --no-pretrained
        """
    )
    parser.add_argument('--dataset', type=str, default='esol',
                       choices=AVAILABLE_DATASETS,
                       help='Dataset to use (default: esol)')
    parser.add_argument('--all-datasets', action='store_true',
                       help='Run on all available datasets')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with fewer samples (100)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Skip pretrained models (faster)')
    parser.add_argument('--budget', type=int, default=500,
                       help='Feature budget for hybrid (default: 500)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Max samples to use (default: 1000, 0 or negative = use all)')
    parser.add_argument('--all-samples', action='store_true',
                       help='Use all available samples (no limit)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file for results')

    args = parser.parse_args()

    # Determine samples (0 or --all-samples means use all)
    if args.all_samples or (args.samples is not None and args.samples <= 0):
        n_samples = 999999  # Effectively unlimited
    elif args.samples:
        n_samples = args.samples
    elif args.quick:
        n_samples = 100
    else:
        n_samples = 1000

    # Configure with larger budgets (no aggressive restrictions)
    config = AnalysisConfig(
        n_samples=n_samples,
        budget=args.budget,
        step_size=50,  # Larger steps for faster greedy
        augmentation_budget=50,  # More augmentation features
    )

    # Determine which datasets to run
    if args.all_datasets:
        datasets_to_run = AVAILABLE_DATASETS
    else:
        datasets_to_run = [args.dataset]

    print("=" * 70)
    print("KIRBy HYBRID EFFECTIVENESS ANALYSIS")
    print("=" * 70)
    print(f"Datasets:    {', '.join(datasets_to_run)}")
    print(f"Max samples: {'all' if config.n_samples >= 999999 else config.n_samples}")
    print(f"Budget:      {config.budget}")
    print(f"CV folds:    {config.n_cv_folds}")
    print(f"Step size:   {config.step_size}")
    print(f"Aug budget:  {config.augmentation_budget}")
    print(f"Pretrained:  {not args.no_pretrained}")

    all_dataset_results = []

    for dataset_name in datasets_to_run:
        print("\n" + "#" * 70)
        print(f"# DATASET: {dataset_name}")
        print("#" * 70)

        try:
            # Load data
            dataset = load_dataset(dataset_name, config.n_samples)

            # Generate representations
            base_reps, augmentations = generate_representations(
                dataset.smiles, config,
                include_pretrained=not args.no_pretrained
            )

            all_results = []

            # Individual representation analysis
            results = analyze_individual_representations(
                base_reps, augmentations, dataset.labels, config
            )
            all_results.extend(results)

            # Hybrid strategy comparison
            results = analyze_hybrid_strategies(
                base_reps, augmentations, dataset.labels, config
            )
            all_results.extend(results)

            # Feature importance analysis
            importance_stats = analyze_feature_importance(
                base_reps, dataset.labels, config
            )

            # Summary for this dataset
            print("\n" + "=" * 70)
            print(f"SUMMARY TABLE - {dataset_name}")
            print("=" * 70)

            df = create_summary_table(all_results)
            df['dataset'] = dataset_name
            print(df.to_string(index=False))
            all_dataset_results.append(df)

            # Insights for this dataset
            print("\n" + "-" * 70)
            print(f"KEY INSIGHTS - {dataset_name}")
            print("-" * 70)

            best = df.iloc[0]
            print(f"\n  Best method:     {best['Method']} (R²={best['R² (test)']:.4f})")

            # Compare hybrid vs best individual
            hybrid_results = df[df['Method'].str.contains('greedy|fixed')]
            individual_results = df[~df['Method'].str.contains('hybrid|greedy|fixed|concat|aug_')]

            if len(hybrid_results) > 0 and len(individual_results) > 0:
                best_hybrid = hybrid_results.iloc[0]['R² (test)']
                best_individual = individual_results.iloc[0]['R² (test)']
                improvement = best_hybrid - best_individual

                print(f"  Best hybrid:     {hybrid_results.iloc[0]['Method']} (R²={best_hybrid:.4f})")
                print(f"  Best individual: {individual_results.iloc[0]['Method']} (R²={best_individual:.4f})")
                print(f"  Improvement:     {improvement:+.4f} R²")

        except Exception as e:
            print(f"  ERROR on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Combined summary across all datasets
    if len(all_dataset_results) > 1:
        print("\n" + "=" * 70)
        print("CROSS-DATASET SUMMARY")
        print("=" * 70)

        combined_df = pd.concat(all_dataset_results, ignore_index=True)

        # Compute average improvement across datasets
        improvements = []
        for ds_name in datasets_to_run:
            ds_df = combined_df[combined_df['dataset'] == ds_name]
            hybrid_best = ds_df[ds_df['Method'].str.contains('greedy|fixed')]['R² (test)'].max()
            indiv_best = ds_df[~ds_df['Method'].str.contains('hybrid|greedy|fixed|concat|aug_')]['R² (test)'].max()
            if pd.notna(hybrid_best) and pd.notna(indiv_best):
                improvements.append(hybrid_best - indiv_best)
                print(f"  {ds_name:25s}: hybrid={hybrid_best:.4f}, individual={indiv_best:.4f}, Δ={hybrid_best-indiv_best:+.4f}")

        if improvements:
            print(f"\n  Average hybrid improvement: {np.mean(improvements):+.4f} R²")

        # Save combined results
        if args.output:
            combined_df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
    elif args.output and all_dataset_results:
        all_dataset_results[0].to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
