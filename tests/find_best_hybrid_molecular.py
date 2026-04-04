#!/usr/bin/env python3
"""
Find Best Hybrid Molecular Representation

Thin molecular wrapper around kirby.find_best_hybrid. Handles SMILES loading,
molecular representation generation, scaffold grouping, and output formatting.
The actual search logic lives in the domain-agnostic library.

Datasets (all regression):
  - logd:  OpenADMET-LogD (7309 molecules, lipophilicity)
  - caco2: OpenADMET-Caco2_Efflux (3777 molecules, P-gp efflux ratio, log-transformed)
  - herg_ki: ChEMBL-hERG-Ki (1415 molecules, hERG binding affinity pKi)
  - qm9:   QM9 HOMO-LUMO gap (2000 molecules, quantum mechanical property)

Usage:
    # Quick local test
    python find_best_hybrid_molecular.py logd --reps fast --search fast

    # Medium search with HuggingFace models
    python find_best_hybrid_molecular.py caco2 --reps fast+hf --search medium

    # Full search with all models
    python find_best_hybrid_molecular.py data.csv --smiles-col SMILES --target-col pIC50 \\
        --reps all --search long
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

# Block stale system bottleneck (Anaconda 2022.05, compiled against old numpy)
sys.modules.setdefault('bottleneck', None)

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from kirby.find_best_hybrid import find_best_hybrid, SearchResult, Blueprint

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
    """Assign scaffold group IDs to molecules for GroupKFold."""
    scaffolds = [get_murcko_scaffold(smi) for smi in smiles_list]
    unique_scaffolds = sorted(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    groups = np.array([scaffold_to_id[s] for s in scaffolds])
    return groups, len(unique_scaffolds)


def assign_mw_groups(smiles_list: List[str], n_groups: int = 5) -> Tuple[np.ndarray, int]:
    """Assign molecular-weight percentile groups for GroupKFold.

    Bins molecules into n_groups equal-count bins by ExactMolWt.
    Invalid SMILES get group 0.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mws = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mws.append(Descriptors.ExactMolWt(mol) if mol is not None else 0.0)
    mws = np.array(mws)

    percentiles = np.linspace(0, 100, n_groups + 1)[1:-1]
    bins = np.percentile(mws, percentiles)
    groups = np.digitize(mws, bins)
    return groups, int(len(np.unique(groups)))


def assign_spectral_groups(smiles_list: List[str], n_groups: int = 5) -> Tuple[np.ndarray, int]:
    """Assign spectral clustering groups based on Tanimoto similarity for GroupKFold.

    Computes Morgan fingerprints, builds a pairwise Tanimoto similarity matrix,
    then runs spectral clustering.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from sklearn.cluster import SpectralClustering

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles('C'), 2, nBits=1024))

    n = len(fps)
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim_matrix[i] = sims

    clustering = SpectralClustering(
        n_clusters=n_groups, affinity='precomputed', random_state=42,
        assign_labels='kmeans',
    )
    groups = clustering.fit_predict(sim_matrix)
    return groups, int(len(np.unique(groups)))


# =============================================================================
# CONFIGURATION
# =============================================================================

REP_PRESETS = {
    'fast': ['ecfp4', 'maccs', 'pdv'],
    'fast+hf': ['ecfp4', 'maccs', 'pdv', 'chemberta', 'molformer', 'selformer'],
    'all': [
        'ecfp4', 'maccs', 'avalon', 'atom_pair', 'top_torsion', 'rdkfp',
        'pdv', 'mordred',
        'coulomb_matrix', 'usr', 'usrcat', 'pharmacophore3d', 'tda', 'e3fp',
        'mol2vec', 'chemberta', 'molformer',
        'selformer', 'chembert', 'molclr', 'graphmvp', 'smited',
        'mhg_gnn', 'grover', 'graph_kernel',
        'clamp', 'rmat', 'map4', 'unimol_v2',
        'minimol', 'gp_molformer', 'mole',
        # Per-fold factory reps (moved from base_reps to rep_factories):
        'pdv_binary_mean', 'pdv_rank', 'sns',
        'chemberta_pca50', 'molformer_pca50', 'selformer_pca50',
        'molclr_pca50', 'smited_pca50', 'mhg_gnn_pca50', 'grover_pca50',
        'mordred_pca50',
    ],
}

# Reps that require per-fold fitting (data-dependent transforms).
# These are created as rep_factories instead of base_reps to prevent leakage.
FACTORY_REPS = {
    'pdv_binary_mean', 'pdv_rank', 'sns',
    'chemberta_pca50', 'molformer_pca50', 'selformer_pca50',
    'molclr_pca50', 'smited_pca50', 'mhg_gnn_pca50', 'grover_pca50',
    'mordred_pca50',
}

# Map old search names to new speed tiers for backward compat
_SEARCH_NAME_MAP = {
    'quick': 'fast',
    'balanced': 'fast',
    'medium': 'medium',
    'precise': 'medium',
    'importance_compare': 'medium',
    'full': 'long',
    'exhaustive': 'long',
    # Direct speed tier names also work
    'fast': 'fast',
    'long': 'long',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def _standardise_smiles(smi: str) -> Optional[str]:
    """Canonicalise SMILES via library standardizer."""
    from kirby.representations.molecular import standardize_smiles
    return standardize_smiles(smi)


def _download_openadmet() -> pd.DataFrame:
    """Download or load cached OpenADMET-ExpansionRx data."""
    import requests

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

    return pd.read_csv(cached)


def _load_openadmet_endpoint(df: pd.DataFrame, endpoint_col: str,
                             log_transform: bool = False,
                             clip: bool = True) -> Tuple[List[str], np.ndarray]:
    """Extract a single endpoint from OpenADMET, standardize, deduplicate."""
    smiles_col = next(c for c in df.columns if c.upper() == 'SMILES')

    sub = df[[smiles_col, endpoint_col]].dropna(subset=[endpoint_col]).copy()
    sub[endpoint_col] = pd.to_numeric(sub[endpoint_col], errors='coerce')
    sub = sub.dropna()

    sub['std_smiles'] = sub[smiles_col].apply(_standardise_smiles)
    sub = sub.dropna(subset=['std_smiles'])

    # Deduplicate by canonical SMILES: take median target
    sub = sub.groupby('std_smiles').agg({endpoint_col: 'median'}).reset_index()

    smiles_arr = sub['std_smiles'].tolist()
    labels_arr = sub[endpoint_col].values.astype(np.float64)

    if log_transform:
        labels_arr = np.log10(np.clip(labels_arr, 1e-10, None))
        valid = np.isfinite(labels_arr)
        smiles_arr = [s for s, v in zip(smiles_arr, valid) if v]
        labels_arr = labels_arr[valid]

    # Clip extreme outliers (1st / 99th percentile)
    if clip and len(labels_arr) > 10:
        p1, p99 = np.percentile(labels_arr, [1, 99])
        labels_arr = np.clip(labels_arr, p1, p99)

    return smiles_arr, labels_arr


def _fetch_chembl_herg_ki() -> Tuple[List[str], np.ndarray]:
    """Fetch ChEMBL hERG (CHEMBL240) Ki data as regression (pKi). Cached."""
    import requests
    import time as _time

    cache_dir = Path.home() / '.cache' / 'kirby'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / 'chembl_herg_ki.csv'

    if cached.exists():
        print(f"  Loading cached ChEMBL hERG Ki from {cached}")
        df = pd.read_csv(cached)
        return df['SMILES'].tolist(), df['pKi'].values

    print("  Fetching ChEMBL hERG Ki data...")
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    all_records = []
    offset = 0
    limit = 1000

    while True:
        params = {
            "target_chembl_id": "CHEMBL240",
            "standard_type": "Ki",
            "pchembl_value__isnull": "false",
            "standard_relation": "=",
            "data_validity_comment__isnull": "true",
            "limit": limit, "offset": offset, "format": "json",
        }
        print(f"    Querying ChEMBL API (offset={offset})...")
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        activities = data.get("activities", [])
        if not activities:
            break
        for act in activities:
            all_records.append({
                "canonical_smiles": act.get("canonical_smiles"),
                "pchembl_value": act.get("pchembl_value"),
                "assay_type": act.get("assay_type"),
            })

        if data.get("page_meta", {}).get("next") is None:
            break
        offset += limit
        _time.sleep(0.5)

    if not all_records:
        raise RuntimeError("No hERG Ki records retrieved from ChEMBL!")

    df = pd.DataFrame(all_records)
    if "assay_type" in df.columns:
        df = df[df["assay_type"] == "B"].copy()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value", "canonical_smiles"])

    # Deduplicate: median pKi per compound, remove high-variance (std > 1.0)
    grouped = df.groupby("canonical_smiles")["pchembl_value"]
    medians = grouped.median().reset_index()
    stds = grouped.std().reset_index().rename(columns={"pchembl_value": "std"})
    merged = medians.merge(stds, on="canonical_smiles")
    merged = merged[(merged["std"].isna()) | (merged["std"] <= 1.0)]

    # Standardize SMILES
    merged['std_smiles'] = merged['canonical_smiles'].apply(_standardise_smiles)
    merged = merged.dropna(subset=['std_smiles'])

    result = merged[['std_smiles', 'pchembl_value']].copy()
    result.columns = ['SMILES', 'pKi']
    result.to_csv(cached, index=False)
    print(f"    Cached {len(result)} compounds to {cached}")

    return result['SMILES'].tolist(), result['pKi'].values


# Polaris ADME-Fang endpoints (biogen/adme-fang-v1, 3521 molecules)
POLARIS_ADME_TARGETS = {
    'adme_hlm': ('LOG_HLM_CLint', 'human liver microsome clearance'),
    'adme_rlm': ('LOG_RLM_CLint', 'rat liver microsome clearance'),
    'adme_mdr1': ('LOG_MDR1-MDCK_ER', 'MDR1-MDCK efflux ratio'),
    'adme_solubility': ('LOG_SOLUBILITY', 'aqueous solubility'),
}


def _load_polaris_adme(target_key: str):
    """Load a Polaris ADME-Fang endpoint. Caches the raw table as CSV.

    Returns:
        (smiles, labels) — filtered to valid RDKit SMILES with non-NaN targets.
    """
    from rdkit import Chem

    col_name, description = POLARIS_ADME_TARGETS[target_key]

    cache_dir = Path.home() / '.cache' / 'kirby'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / 'polaris_adme_fang.csv'

    if cached.exists():
        df = pd.read_csv(cached)
    else:
        import polaris as po
        print("  Downloading Polaris ADME-Fang dataset...")
        ds = po.load_dataset("biogen/adme-fang-v1")
        table = ds.table
        rows = []
        for i in range(ds.n_rows):
            row = {col: table[col][i] for col in ds.columns}
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(cached, index=False)
        print(f"  Cached {len(df)} rows to {cached}")

    # Filter to valid SMILES + non-NaN target
    df = df.dropna(subset=['SMILES', col_name])
    smiles, labels = [], []
    for smi, val in zip(df['SMILES'], df[col_name]):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            smiles.append(str(smi))
            labels.append(float(val))

    return smiles, np.array(labels)


def load_dataset(name_or_path: str, smiles_col: str = 'smiles',
                 target_col: str = 'target',
                 clip: bool = True) -> Tuple[List[str], np.ndarray, str]:
    """Load dataset from built-in name or CSV path.

    All datasets are post-processed with InChIKey deduplication to prevent
    the same molecule (different SMILES notation) appearing in train+test.

    Built-in datasets (all regression):
      - logd:  OpenADMET-LogD (~7309 molecules)
      - caco2: OpenADMET-Caco2_Efflux (~3777 molecules, log-transformed)
      - herg_ki: ChEMBL-hERG-Ki (~1415 molecules, pKi)
      - qm9:   QM9 HOMO-LUMO gap (~2000 molecules, normalized)
      - adme_hlm:  Polaris ADME-Fang — human liver microsome clearance (~3087 molecules)
      - adme_rlm:  Polaris ADME-Fang — rat liver microsome clearance (~3054 molecules)
      - adme_mdr1: Polaris ADME-Fang — MDR1-MDCK efflux ratio (~2642 molecules)
      - adme_solubility: Polaris ADME-Fang — aqueous solubility (~2173 molecules)
    """
    from rdkit import Chem
    from kirby.representations.molecular import deduplicate_by_inchikey

    name_lower = name_or_path.lower()

    if name_lower in POLARIS_ADME_TARGETS:
        col_name, description = POLARIS_ADME_TARGETS[name_lower]
        smiles, labels = _load_polaris_adme(name_lower)
        print(f"  Loaded {len(smiles)} molecules from Polaris ADME-Fang ({description})")
        smiles, labels, task_type = smiles, labels, 'regression'

    elif name_lower == 'logd':
        df = _download_openadmet()
        logd_col = next((c for c in df.columns if 'LogD' in c), None)
        if logd_col is None:
            raise ValueError("Cannot find LogD column in OpenADMET data")
        smiles, labels = _load_openadmet_endpoint(df, logd_col, log_transform=False, clip=clip)
        print(f"  Loaded {len(smiles)} molecules from OpenADMET-LogD")
        smiles, labels, task_type = smiles, labels, 'regression'

    elif name_lower == 'caco2':
        df = _download_openadmet()
        caco2_col = next((c for c in df.columns if 'Caco' in c and 'Efflux' in c), None)
        if caco2_col is None:
            raise ValueError("Cannot find Caco2 Efflux column in OpenADMET data")
        smiles, labels = _load_openadmet_endpoint(df, caco2_col, log_transform=True, clip=clip)
        print(f"  Loaded {len(smiles)} molecules from OpenADMET-Caco2_Efflux")
        smiles, labels, task_type = smiles, labels, 'regression'

    elif name_lower in ('herg_ki',):
        smiles, labels = _fetch_chembl_herg_ki()
        print(f"  Loaded {len(smiles)} molecules from ChEMBL-hERG-Ki (pKi)")
        smiles, labels, task_type = smiles, labels, 'regression'

    elif name_lower == 'herg':
        from kirby.datasets.herg import load_herg_tdc
        data = load_herg_tdc()
        smiles = data['smiles']
        labels = data['labels']
        print(f"  Loaded {len(smiles)} molecules from TDC-hERG (binary blocker clf)")
        smiles, labels, task_type = smiles, labels, 'classification'

    elif name_lower == 'qm9':
        from kirby.datasets.qm9 import load_qm9
        qm9_data = load_qm9(n_samples=10000, property_idx=4, random_seed=42)
        smiles = qm9_data['smiles']
        labels = qm9_data['labels']  # Raw values; StandardScaler applied per fold
        print(f"  Loaded {len(smiles)} molecules from QM9 (HOMO-LUMO gap)")
        smiles, labels, task_type = smiles, labels, 'regression'

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

        smiles = valid_smiles
        print(f"  Loaded {len(smiles)} valid molecules, task: {task_type}")

    else:
        raise ValueError(f"Unknown dataset: {name_or_path}. "
                         f"Built-in options: logd, caco2, herg_ki, herg, qm9, "
                         f"adme_hlm, adme_rlm, adme_mdr1, adme_solubility")

    # InChIKey deduplication: same molecule with different SMILES notation
    # → prevent data leakage across train/test folds
    n_before = len(smiles)
    smiles, labels = deduplicate_by_inchikey(smiles, labels)
    n_removed = n_before - len(smiles)
    if n_removed > 0:
        print(f"  InChIKey dedup: {n_before} → {len(smiles)} ({n_removed} duplicates removed)")

    return smiles, labels, task_type


# =============================================================================
# REPRESENTATION GENERATION
# =============================================================================

def _rep_cache_path(smiles: List[str], rep_name: str) -> Path:
    """Get cache file path for a representation, keyed by SMILES hash.

    Respects KIRBY_CACHE_DIR env var (set by setup.sh on the server to
    redirect caching to /data/).  Falls back to ~/.cache/kirby.
    """
    import hashlib
    smiles_hash = hashlib.sha256('\n'.join(smiles).encode()).hexdigest()[:16]
    base = Path(os.environ.get('KIRBY_CACHE_DIR', Path.home() / '.cache' / 'kirby'))
    cache_dir = base / 'reps' / smiles_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    # graph_kernel_filtered replaces old graph_kernel cache (quality filter now default)
    if rep_name == 'graph_kernel':
        rep_name = 'graph_kernel_filtered'
    return cache_dir / f'{rep_name}.npy'


def get_representations(smiles: List[str], preset_or_list: str,
                        cache: bool = True) -> Dict[str, np.ndarray]:
    """Generate representations based on preset or explicit list.

    Args:
        smiles: List of SMILES strings.
        preset_or_list: Preset name ('fast', 'fast+hf', 'all') or comma-separated rep names.
        cache: If True, cache representations to ~/.cache/kirby/reps/ and load from
               cache on subsequent calls with the same SMILES. Default True.
    """
    from kirby.representations.molecular import (
        create_ecfp4, create_maccs, create_avalon, create_pdv, create_mordred,
        create_atom_pair, create_top_torsion, create_rdkfp,
        create_coulomb_matrix, create_usr, create_usrcat,
        create_pharmacophore3d, create_tda, create_e3fp,
        create_mol2vec,
        create_chemberta, create_molformer, create_selformer,
        create_chembert, create_molclr, create_graphmvp, create_smited,
        create_mhg_gnn, create_grover, create_graph_kernel,
        create_unimol, create_schnet, create_dimenetpp, create_clamp,
        create_rmat, create_map4, create_unimol_v2,
        create_minimol, create_gp_molformer, create_mole,
    )

    if preset_or_list in REP_PRESETS:
        rep_names = REP_PRESETS[preset_or_list]
    else:
        rep_names = [r.strip().lower() for r in preset_or_list.split(',')]

    # Filter out factory reps — these are handled by get_rep_factories()
    base_rep_names = [r for r in rep_names if r not in FACTORY_REPS]

    print(f"\nGenerating {len(base_rep_names)} base representations "
          f"(+ {len(rep_names) - len(base_rep_names)} per-fold factory reps)...")

    rep_funcs = {
        'ecfp4': create_ecfp4,
        'maccs': create_maccs,
        'avalon': create_avalon,
        'atom_pair': create_atom_pair,
        'top_torsion': create_top_torsion,
        'rdkfp': create_rdkfp,
        'pdv': create_pdv,
        'mordred': create_mordred,
        'coulomb_matrix': create_coulomb_matrix,
        'usr': create_usr,
        'usrcat': create_usrcat,
        'pharmacophore3d': create_pharmacophore3d,
        'tda': create_tda,
        'e3fp': create_e3fp,
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
        'unimol': create_unimol,
        'schnet': create_schnet,
        'dimenetpp': create_dimenetpp,
        'clamp': create_clamp,
        'rmat': create_rmat,
        'map4': create_map4,
        'unimol_v2': create_unimol_v2,
        'minimol': create_minimol,
        'gp_molformer': create_gp_molformer,
        'mole': create_mole,
    }

    representations = {}
    for rep_name in base_rep_names:
        if rep_name not in rep_funcs:
            print(f"  WARNING: Unknown representation '{rep_name}', skipping")
            continue

        # Try loading from cache
        if cache:
            cache_file = _rep_cache_path(smiles, rep_name)
            if cache_file.exists():
                rep = np.load(cache_file)
                print(f"  {rep_name}: loaded from cache ({rep.shape[1]} features)")
                representations[rep_name] = rep
                continue

        try:
            start = time.time()
            print(f"  Generating {rep_name}...", end=' ', flush=True)
            rep = rep_funcs[rep_name](smiles)
            elapsed = time.time() - start
            print(f"done ({rep.shape[1]} features, {elapsed:.1f}s)")
            representations[rep_name] = rep

            # Save to cache
            if cache:
                cache_file = _rep_cache_path(smiles, rep_name)
                np.save(cache_file, rep)
        except Exception as e:
            print(f"FAILED: {e}")

    if not representations:
        raise ValueError("No representations could be generated!")

    return representations


def get_rep_factories(smiles: List[str], preset_or_list: str) -> Dict[str, Any]:
    """Create per-fold rep factories for data-dependent representations.

    These reps require fitting on training data (PCA, rank, binary mean, SNS)
    and are passed as rep_factories to find_best_hybrid() to prevent leakage.

    Args:
        smiles: List of SMILES strings
        preset_or_list: Preset name or comma-separated rep names

    Returns:
        Dict of {rep_name: factory_callable} for reps that need per-fold fitting
    """
    from kirby.representations.molecular import (
        create_chemberta, create_molformer, create_selformer,
        create_molclr, create_smited, create_mhg_gnn, create_grover,
        create_mordred,
        make_pdv_binary_mean_factory, make_pdv_rank_factory,
        make_pca_factory, make_sns_factory,
    )

    if preset_or_list in REP_PRESETS:
        rep_names = REP_PRESETS[preset_or_list]
    else:
        rep_names = [r.strip().lower() for r in preset_or_list.split(',')]

    # Only create factories for requested reps that are in FACTORY_REPS
    requested_factories = [r for r in rep_names if r in FACTORY_REPS]
    if not requested_factories:
        return {}

    print(f"\nCreating {len(requested_factories)} per-fold rep factories...")

    # Map factory rep names to their factory constructors
    # PCA factories need the base_fn for their full-data embedding
    pca_base_fns = {
        'chemberta_pca50': create_chemberta,
        'molformer_pca50': create_molformer,
        'selformer_pca50': create_selformer,
        'molclr_pca50': create_molclr,
        'smited_pca50': create_smited,
        'mhg_gnn_pca50': create_mhg_gnn,
        'grover_pca50': create_grover,
        'mordred_pca50': create_mordred,
    }

    factories = {}
    for rep_name in requested_factories:
        try:
            start = time.time()
            print(f"  Setting up {rep_name} factory...", end=' ', flush=True)

            if rep_name == 'pdv_binary_mean':
                factories[rep_name] = make_pdv_binary_mean_factory(smiles)
            elif rep_name == 'pdv_rank':
                factories[rep_name] = make_pdv_rank_factory(smiles)
            elif rep_name == 'sns':
                factories[rep_name] = make_sns_factory(smiles)
            elif rep_name in pca_base_fns:
                factories[rep_name] = make_pca_factory(
                    smiles, base_fn=pca_base_fns[rep_name]
                )
            else:
                print(f"WARNING: no factory for '{rep_name}'")
                continue

            elapsed = time.time() - start
            print(f"done ({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    return factories


def get_augmentations(smiles: List[str]) -> Dict[str, np.ndarray]:
    """Generate augmented features (graph topology, spectral)."""
    from kirby.representations.molecular import compute_graph_topology, compute_spectral_features

    print("\nGenerating augmented features...")
    augmentations = {}

    try:
        print("  Generating graph_topology...", end=' ', flush=True)
        start = time.time()
        topo = compute_graph_topology(smiles)
        elapsed = time.time() - start
        print(f"done ({topo.shape[1]} features, {elapsed:.1f}s)")
        augmentations['graph_topology'] = topo
    except Exception as e:
        print(f"FAILED: {e}")

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
# OUTPUT FORMATTING
# =============================================================================

def _get(result, key):
    """Access result field by key — works with both dicts and objects."""
    if isinstance(result, dict):
        return result[key]
    return getattr(result, key)


_METRIC_DISPLAY = {
    'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE',
    'accuracy': 'Accuracy', 'auc': 'AUC',
    'spearman': 'Spearman ρ', 'pearson': 'Pearson r',
    'ci': 'CI', 'f1_weighted': 'F1w',
}


def print_summary(result: SearchResult, task_type: str):
    """Print formatted summary to console."""
    metric = _METRIC_DISPLAY.get(result.metric_name, result.metric_name)
    lower_is_better = result.lower_is_better
    dimension_tiers = [50, 100, 200, 500]

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    if not result.best_by_model:
        print("\nNo results found.")
        return

    # Best by model
    print(f"\n=== BEST BY MODEL ===")
    models_evaluated = list(result.best_by_model.keys())
    print(f"{'Model':<10} {'Dim':<6} {'Method':<20} {'Importance':<18} {metric:<14} {'Budget'}")
    print("-" * 85)
    for model_name, r in result.best_by_model.items():
        bp = _get(r, 'blueprint')
        print(f"{model_name:<10} {_get(r, 'n_features'):<6} {bp.allocation_method:<20} "
              f"{bp.importance_method:<18} "
              f"{_get(r, 'score_mean'):.4f}+/-{_get(r, 'score_std'):.2f}   {bp.budget}")

    # Best by dimension
    print(f"\n=== BEST BY DIMENSION ===")
    print(f"{'Dim':<8} {'Model':<10} {'Method':<20} {'Importance':<18} {metric:<12}")
    print("-" * 75)
    for tier, r in sorted(result.best_by_dimension.items()):
        bp = _get(r, 'blueprint')
        print(f"<={tier:<7} {_get(r, 'model'):<10} {bp.allocation_method:<20} "
              f"{bp.importance_method:<18} "
              f"{_get(r, 'score_mean'):.4f}+/-{_get(r, 'score_std'):.2f}")

    # Performance matrix
    print(f"\n=== PERFORMANCE MATRIX ({metric}) ===")
    tiers = [str(t) for t in dimension_tiers]
    header = f"{'Model':<12}" + "".join(f"{'<='+t:<10}" for t in tiers)
    print(header)
    print("-" * len(header))
    for model_name in models_evaluated:
        if model_name in result.performance_matrix:
            row = f"{model_name:<12}"
            for tier in tiers:
                score = result.performance_matrix[model_name].get(tier, None)
                if score is not None:
                    row += f"{score:<10.4f}"
                else:
                    row += f"{'--':<10}"
            print(row)

    # Baselines comparison
    print(f"\n=== BASELINE COMPARISON (full dimension) ===")
    baselines = result.baselines
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

    # Secondary metrics for baselines
    sec_keys = set()
    for baseline_scores in baselines.values():
        if '_secondary' in baseline_scores:
            sec_keys.update(baseline_scores['_secondary'].keys())
    if sec_keys:
        print(f"\n=== SECONDARY METRICS (baselines) ===")
        # Group by metric name
        sec_metric_names = sorted(set(k.split('_', 1)[1] for k in sec_keys if '_' in k))
        for sec_metric in sec_metric_names:
            display_name = _METRIC_DISPLAY.get(sec_metric, sec_metric)
            print(f"\n  {display_name}:")
            header = f"  {'Baseline':<15} {'Dim':<8}" + "".join(f"{m:<10}" for m in models_evaluated)
            print(header)
            for baseline_name, baseline_scores in baselines.items():
                sec = baseline_scores.get('_secondary', {})
                row = f"  {baseline_name:<15} {baseline_scores.get('dimension', '?'):<8}"
                for m in models_evaluated:
                    key = f'{m}_{sec_metric}'
                    val = sec.get(key, {}).get('mean')
                    if val is not None:
                        row += f"{val:<10.4f}"
                    else:
                        row += f"{'--':<10}"
                print(row)

    # Top-3 base representations (ranked by RF)
    if baselines:
        metric_name = result.metric_name
        rf_scores = []
        for rep_name, model_scores in baselines.items():
            if rep_name.startswith('_'):
                continue
            rf_score = model_scores.get('rf')
            if rf_score is not None and not np.isnan(rf_score):
                rf_scores.append((rep_name, rf_score))
        rf_scores.sort(key=lambda x: x[1], reverse=not lower_is_better)
        top3 = rf_scores[:3]
        if top3:
            print(f"\n  TOP-3 BASE REPS (by RF {metric}):")
            for i, (rname, score) in enumerate(top3, 1):
                print(f"    {i}. {rname:<25} {score:.4f}")

    # Improvement summary
    print(f"\n=== HYBRID IMPROVEMENT ===")
    for model_name, r in result.best_by_model.items():
        hybrid_score = _get(r, 'score_mean')
        hybrid_dim = _get(r, 'n_features')

        best_baseline_name = None
        best_baseline_score = None
        for baseline_name, baseline_scores in baselines.items():
            if baseline_name == 'all_concat':
                continue
            if model_name not in baseline_scores:
                continue
            score = baseline_scores[model_name]
            if best_baseline_score is None:
                best_baseline_score = score
                best_baseline_name = baseline_name
            elif (lower_is_better and score < best_baseline_score) or \
                 (not lower_is_better and score > best_baseline_score):
                best_baseline_score = score
                best_baseline_name = baseline_name

        if best_baseline_name is not None:
            improvement = hybrid_score - best_baseline_score
            baseline_dim = baselines[best_baseline_name].get('dimension', '?')
            better = improvement < 0 if lower_is_better else improvement > 0
            sign = '' if better else '(!)'
            print(f"  {model_name}: Hybrid({hybrid_dim}d) vs {best_baseline_name}({baseline_dim}d): "
                  f"{hybrid_score:.4f} vs {best_baseline_score:.4f} ({improvement:+.4f}) {sign}")

    print("=" * 80)


def save_results(
    result: SearchResult,
    dataset_name: str,
    search_mode: str,
    reps_tested: List[str],
    output_path: str,
    base_reps: Optional[Dict[str, np.ndarray]] = None,
    labels: Optional[np.ndarray] = None
):
    """Save minimal blueprint JSON with optional importance weights."""
    from kirby.hybrid import create_hybrid

    def result_to_dict(r, include_weights: bool = False):
        bp = _get(r, 'blueprint')
        d = {
            'model': _get(r, 'model'),
            'score_mean': round(_get(r, 'score_mean'), 4),
            'score_std': round(_get(r, 'score_std'), 4),
            'dimension': _get(r, 'n_features'),
            'blueprint': bp.to_dict(),
        }

        if include_weights and base_reps is not None and labels is not None:
            try:
                _, feature_info = create_hybrid(
                    base_reps, labels,
                    allocation_method=bp.allocation_method,
                    budget=bp.budget,
                    importance_method=bp.importance_method,
                )
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
                print(f"  Warning: Could not extract weights for {_get(r, 'model')}: {e}")

        return d

    output = {
        'metadata': {
            'dataset': dataset_name,
            'search_mode': search_mode,
            'reps_available': reps_tested,
            'timestamp': datetime.now().isoformat(),
        },
        'best_by_dimension': {
            str(tier): result_to_dict(r)
            for tier, r in result.best_by_dimension.items()
        },
        'best_by_model': {
            model: result_to_dict(r, include_weights=True)
            for model, r in result.best_by_model.items()
        },
        'performance_matrix': result.performance_matrix,
        'baselines': result.baselines,
    }

    # Extract top-3 base reps by RF score
    lower_is_better = result.lower_is_better
    rf_scores = []
    for rep_name, model_scores in result.baselines.items():
        if rep_name.startswith('_'):
            continue
        rf_score = model_scores.get('rf')
        if rf_score is not None and not np.isnan(rf_score):
            rf_scores.append((rep_name, rf_score))
    rf_scores.sort(key=lambda x: x[1], reverse=not lower_is_better)
    output['top3_base_reps'] = [
        {'rep': name, 'rf_score': round(score, 4)} for name, score in rf_scores[:3]
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    blueprint_path = output_path.replace('.json', '_blueprint.py')
    save_blueprint_code(result, reps_tested, blueprint_path)


def save_blueprint_code(result: SearchResult, reps_tested: List[str], output_path: str):
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
        '    create_pdv_binary_mean, create_pdv_rank,',
        '    create_mol2vec,',
        '    create_chemberta, create_molformer, create_selformer,',
        '    create_chembert, create_molclr, create_graphmvp, create_smited,',
        '    create_mhg_gnn, create_grover, create_graph_kernel,',
        '    create_unimol, create_schnet,',
        ')',
        '',
        '',
        '# Representation generators',
        'REP_GENERATORS = {',
        "    'ecfp4': create_ecfp4,",
        "    'maccs': create_maccs,",
        "    'pdv': create_pdv,",
        "    'pdv_binary_mean': create_pdv_binary_mean,",
        "    'pdv_rank': create_pdv_rank,",
        "    'mol2vec': create_mol2vec,",
        "    'chemberta': create_chemberta,",
        "    'molformer': create_molformer,",
        "    'selformer': create_selformer,",
        "    'chembert': create_chembert,",
        "    'molclr': create_molclr,",
        "    'graphmvp': create_graphmvp,",
        "    'smited': create_smited,",
        "    'mhg_gnn': create_mhg_gnn,",
        "    'grover': create_grover,",
        "    'graph_kernel': create_graph_kernel,",
        "    'unimol': create_unimol,",
        "    'schnet': create_schnet,",
        '}',
        '',
        '',
    ]

    for model_name, r in result.best_by_model.items():
        bp = _get(r, 'blueprint')
        n_features = _get(r, 'n_features')
        score_mean = _get(r, 'score_mean')

        lines.extend([
            f"def create_{model_name}_hybrid(smiles_list, labels):",
            f'    """',
            f'    Create the optimal hybrid for {model_name.upper()} model.',
            f'    ',
            f'    Blueprint: method={bp.allocation_method}, budget={bp.budget}, importance={bp.importance_method}',
            f'    Expected dimension: ~{n_features}',
            f'    Expected score: {score_mean:.4f}',
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
            f"        allocation_method='{bp.allocation_method}',",
            f'        budget={bp.budget},',
            f"        importance_method='{bp.importance_method}',",
            f'    )',
            f'    ',
            f'    return X_hybrid, feature_info',
            f'',
            f'',
        ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Blueprint code saved to: {output_path}")


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def compare_importance_methods(
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    methods: List[str] = ['treeshap', 'permutation'],
) -> Dict[str, Any]:
    """Compare how different importance methods rank features."""
    from kirby.hybrid import compute_feature_importance
    from scipy.stats import spearmanr, kendalltau

    print("\n=== IMPORTANCE METHOD COMPARISON ===")

    all_importances = {}
    for method in methods:
        print(f"  Computing {method} importance...", end=' ', flush=True)
        try:
            imp = compute_feature_importance(base_reps, labels, method=method)
            concat_imp = np.concatenate([imp[rep] for rep in sorted(imp.keys())])
            all_importances[method] = concat_imp
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    if len(all_importances) < 2:
        return {'error': 'Need at least 2 methods to compare'}

    method_names = list(all_importances.keys())
    correlations = {}

    for i, m1 in enumerate(method_names):
        for m2 in method_names[i+1:]:
            imp1 = all_importances[m1]
            imp2 = all_importances[m2]
            spearman_r, spearman_p = spearmanr(imp1, imp2)
            kendall_t, kendall_p = kendalltau(imp1, imp2)

            key = f"{m1}_vs_{m2}"
            correlations[key] = {
                'spearman_r': round(spearman_r, 4),
                'spearman_p': round(spearman_p, 6),
                'kendall_tau': round(kendall_t, 4),
                'kendall_p': round(kendall_p, 6),
            }
            print(f"  {m1} vs {m2}: Spearman r={spearman_r:.3f}, Kendall tau={kendall_t:.3f}")

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
    best_result: dict,
    base_reps: Dict[str, np.ndarray],
    smiles: List[str],
    labels: np.ndarray,
    output_path: str
):
    """Generate and save feature explanations for the best configuration."""
    from kirby.hybrid import create_hybrid
    from kirby.importance.hybrid import explain_hybrid, format_explanation_summary

    print("\n=== GENERATING EXPLANATIONS ===")

    bp = _get(best_result, 'blueprint')

    print(f"  Recreating hybrid: method={bp.allocation_method}, budget={bp.budget}")
    _, feature_info = create_hybrid(
        base_reps, labels,
        allocation_method=bp.allocation_method,
        budget=bp.budget,
        importance_method=bp.importance_method,
    )

    print("  Mapping features to interpretations...")
    explanations = explain_hybrid(
        feature_info=feature_info,
        smiles_list=smiles[:100],
        top_k=30,
    )

    print("\n" + format_explanation_summary(explanations, max_features=15))

    importance_comparison = compare_importance_methods(base_reps, labels)

    explain_path = output_path.replace('.json', '_explanations.json')
    Path(explain_path).parent.mkdir(parents=True, exist_ok=True)

    explain_output = {
        'model': _get(best_result, 'model'),
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
  %(prog)s logd --reps fast --search fast
  %(prog)s caco2 --reps fast+hf --search medium
  %(prog)s data.csv --smiles-col SMILES --target-col pIC50 --reps all --search long

Output:
  - Best overall: single best model+hybrid combination
  - Best by dimension: optimal at each tier (50, 100, 200, 500)
  - Best by model: optimal hybrid per model type
  - Performance matrix: full Model x Dimension grid
  - Baseline comparison: hybrid vs full-dimension baselines
        """
    )

    parser.add_argument('dataset', type=str, help="'logd', 'caco2', 'herg_ki', 'qm9', 'adme_hlm', 'adme_rlm', 'adme_mdr1', 'adme_solubility', or path to CSV")
    parser.add_argument('--smiles-col', type=str, default='smiles')
    parser.add_argument('--target-col', type=str, default='target')
    parser.add_argument('--reps', type=str, default='fast',
                       help="'fast', 'fast+hf', 'all', or comma-separated list")
    parser.add_argument('--search', type=str, default='fast',
                       choices=list(_SEARCH_NAME_MAP.keys()),
                       help="Search speed: 'fast', 'medium', 'long' (or legacy: quick, full, etc.)")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--explain', action='store_true',
                       help='Generate feature explanations for best configs')
    parser.add_argument('--augment', action='store_true',
                       help='Include augmented features (graph topology, spectral)')
    parser.add_argument('--allocation', type=str, default=None,
                       help="Override allocation methods: comma-separated from "
                            "'greedy', 'performance_weighted', 'softmax_T5', "
                            "'dominant_greedy', 'two_stage', 'stability_selection'")
    parser.add_argument('--model', type=str, default=None,
                       help="Override models: 'rf', 'xgboost', 'ridge', 'mlp', 'knn', etc. or comma-separated")
    parser.add_argument('--save-weights', action='store_true',
                       help='Fit best config on full data and save feature weights')
    parser.add_argument('--weight-method', type=str, default='lightgbm_gain',
                       help="Importance method for feature weights (default: lightgbm_gain)")
    parser.add_argument('--test-rep-subsets', action='store_true',
                       help='Also test leave-one-rep-out subsets (when >=3 reps)')
    parser.add_argument('--n-folds', type=int, default=None,
                       help='Override number of CV folds (default: from speed tier)')
    parser.add_argument('--finetune', type=str, default=None,
                       help='Comma-separated fine-tuning methods: '
                            'lora_chemberta, lora_molformer, '
                            'freeze_chemberta, freeze_molformer, '
                            'gnn_molclr, gnn_graphmvp, '
                            'contrastive_chemberta, contrastive_molformer, '
                            'grover_ft, ft_schnet, ft_dimenetpp, ft_chemprop, mock_lora')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                       help='Fine-tuning epochs per fold (default: 5)')
    parser.add_argument('--lora-rank', type=int, default=8,
                       help='LoRA rank (default: 8)')
    parser.add_argument('--no-ft-cache', action='store_true',
                       help='Disable fine-tuning cache')
    parser.add_argument('--tune', action='store_true',
                       help='Tune model HPs via RandomizedSearchCV before hybrid search')
    parser.add_argument('--tune-iter', type=int, default=20,
                       help='RandomizedSearchCV iterations per model (default: 20)')
    parser.add_argument('--no-clip', action='store_true',
                       help='Disable outlier clipping (1st/99th percentile) for OpenADMET endpoints')

    args = parser.parse_args()

    # Map search name to speed tier
    speed = _SEARCH_NAME_MAP.get(args.search, args.search)

    # Load dataset (molecular-specific)
    print(f"Loading dataset: {args.dataset}")
    smiles, labels, task_type = load_dataset(
        args.dataset, args.smiles_col, args.target_col,
        clip=not args.no_clip,
    )
    print(f"  {len(smiles)} molecules, task: {task_type}")

    # Compute scaffold groups for CV (molecular-specific)
    print("  Assigning scaffold groups...")
    scaffold_groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  {n_scaffolds} unique scaffolds")

    # Generate representations (molecular-specific)
    base_reps = get_representations(smiles, args.reps)

    # Create per-fold factories for data-dependent reps (prevents leakage)
    leakage_free_factories = get_rep_factories(smiles, args.reps)

    # TODO: augmentations not yet wired into find_best_hybrid library
    if args.augment:
        print("WARNING: --augment not yet supported in library mode, ignoring")

    # Resolve model override
    models_arg = None
    if args.model:
        models_arg = [m.strip() for m in args.model.split(',')]

    # Resolve allocation override
    alloc_arg = None
    if args.allocation:
        alloc_arg = [a.strip() for a in args.allocation.split(',')]

    # Build fine-tuning factories
    rep_factories = None
    if args.finetune:
        ft_methods = [m.strip() for m in args.finetune.split(',')]
        rep_factories = {}
        ft_cache = not args.no_ft_cache
        dataset_name = Path(args.dataset).stem if os.path.exists(args.dataset) else args.dataset

        for method in ft_methods:
            if method == 'mock_lora':
                from kirby.finetuning._base import mock_factory
                rep_factories['mock_lora'] = mock_factory(n_features=64, seed=args.seed)
            elif method.startswith('lora_'):
                from kirby.finetuning.molecular_lora import make_molecular_lora_factory
                model_key = method.replace('lora_', '')
                rep_factories[method] = make_molecular_lora_factory(
                    smiles_list=smiles,
                    dataset_name=dataset_name,
                    base_model=model_key,
                    lora_rank=args.lora_rank,
                    epochs=args.finetune_epochs,
                    cache=ft_cache,
                )
            elif method.startswith('freeze_'):
                from kirby.finetuning.molecular_freeze import make_molecular_freeze_factory
                model_key = method.replace('freeze_', '')
                rep_factories[method] = make_molecular_freeze_factory(
                    smiles_list=smiles,
                    dataset_name=dataset_name,
                    base_model=model_key,
                    epochs=args.finetune_epochs,
                    cache=ft_cache,
                )
            elif method.startswith('gnn_'):
                from kirby.finetuning.molecular_gnn import make_molecular_gnn_factory
                model_key = method.replace('gnn_', '')
                rep_factories[method] = make_molecular_gnn_factory(
                    smiles_list=smiles,
                    dataset_name=dataset_name,
                    base_model=model_key,
                    epochs=args.finetune_epochs,
                    cache=ft_cache,
                )
            elif method.startswith('contrastive_'):
                from kirby.finetuning.molecular_contrastive import make_molecular_contrastive_factory
                model_key = method.replace('contrastive_', '')
                rep_factories[method] = make_molecular_contrastive_factory(
                    smiles_list=smiles,
                    dataset_name=dataset_name,
                    base_model=model_key,
                    lora_rank=args.lora_rank,
                    epochs=args.finetune_epochs,
                    cache=ft_cache,
                )
            elif method == 'grover_ft':
                from kirby.finetuning.molecular_grover import make_molecular_grover_factory
                rep_factories[method] = make_molecular_grover_factory(
                    smiles_list=smiles,
                    dataset_name=dataset_name,
                    epochs=args.finetune_epochs,
                    cache=ft_cache,
                )
            elif method in ('ft_schnet', 'ft_dimenetpp'):
                from kirby.finetuning.molecular_3d import make_molecular_3d_factory
                model_3d = method.replace('ft_', '')
                rep_factories[method] = make_molecular_3d_factory(
                    smiles_list=smiles,
                    dataset_name=dataset_name,
                    base_model=model_3d,
                    epochs=100,
                    cache=ft_cache,
                )
            elif method == 'ft_chemprop':
                from kirby.finetuning.molecular_chemprop import make_molecular_chemprop_factory
                rep_factories[method] = make_molecular_chemprop_factory(
                    smiles_list=smiles,
                    dataset_name=dataset_name,
                    epochs=args.finetune_epochs,
                    task_type=task_type,
                    cache=ft_cache,
                )
            else:
                print(f"  WARNING: Unknown fine-tuning method '{method}', skipping")

        if rep_factories:
            print(f"\nFine-tuning factories: {list(rep_factories.keys())}")
        else:
            rep_factories = None

    # Merge leakage-free factories with fine-tuning factories
    if leakage_free_factories:
        if rep_factories is None:
            rep_factories = {}
        rep_factories.update(leakage_free_factories)

    # Hyperparameter tuning
    if args.tune:
        from kirby.find_best_hybrid import tune_models
        print("\n=== Hyperparameter Tuning ===")
        tuned_models = tune_models(
            base_reps=base_reps, labels=labels, groups=scaffold_groups,
            task_type=task_type, speed=speed, n_iter=args.tune_iter,
            seed=args.seed,
        )
        models_arg = tuned_models

    # Call domain-agnostic search
    result = find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=scaffold_groups,
        models=models_arg,
        metric='rmse' if task_type == 'regression' else 'accuracy',
        secondary_metrics=['r2', 'spearman'] if task_type == 'regression' else None,
        task_type=task_type,
        speed=speed,
        n_folds=args.n_folds,
        allocation_methods=alloc_arg,
        return_fitted=args.save_weights,
        weight_method=args.weight_method if args.save_weights else None,
        test_rep_subsets=args.test_rep_subsets,
        seed=args.seed,
        rep_factories=rep_factories,
    )

    # Print summary (molecular-specific formatting)
    print_summary(result, task_type)

    # Save results
    if args.output:
        output_path = args.output
    else:
        dataset_name = Path(args.dataset).stem if os.path.exists(args.dataset) else args.dataset
        tune_suffix = '_tuned' if args.tune else ''
        output_path = f"results_{dataset_name}_{args.search}{tune_suffix}.json"

    save_results(result, args.dataset, args.search, list(base_reps.keys()), output_path,
                 base_reps=base_reps, labels=labels)

    # Generate explanations if requested (iterate over models)
    if args.explain and result.best_by_model:
        for model_name, best_result in result.best_by_model.items():
            print(f"\n--- Explanations for model: {model_name} ---")
            generate_explanations(
                best_result=best_result,
                base_reps=base_reps,
                smiles=smiles,
                labels=labels,
                output_path=output_path.replace('.json', f'_{model_name}.json'),
            )

    return 0


if __name__ == '__main__':
    sys.exit(main())
