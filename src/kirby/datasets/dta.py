"""
Drug-Target Affinity (DTA) Dataset Loader

Loads DAVIS and KIBA benchmark datasets directly from DeepDTA GitHub,
and BindingDB_Kd via TDC (Therapeutics Data Commons).
No external dependencies beyond numpy/pandas/requests (+ PyTDC for BindingDB).

DAVIS: 68 drugs × 442 kinases, Kd values (nM → pKd)
KIBA: 2,111 drugs × 229 proteins, integrated KIBA scores (~118K valid pairs)
BindingDB_Kd: ~52K drug-target pairs (10,665 drugs × 1,413 proteins), Kd from BindingDB

Note on expected performance:
    Published SOTA DTA CI (0.90+) uses warm-start splits with deep interaction
    models (DeepDTA, GraphDTA) that learn drug-protein attention — a different
    paradigm. KIRBy uses cold-target/cold-drug GroupKFold with linear/tree models.
    Cold-target CI 0.82-0.87 is consistent with published cold-target ranges.
    This is by design: the thesis evaluates representation quality, not architecture.

Reference:
- Davis et al. (2011) Nature Biotechnology (DAVIS kinase binding)
- Tang et al. (2014) Bioinformatics (KIBA score)
- Öztürk et al. (2018) Bioinformatics (DeepDTA, data source)
- Liu et al. (2007) Bioinformatics (BindingDB)
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from kirby.paths import data_dir

_BASE_URL = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"

_DATASETS = {
    'davis': {
        'property': 'Kd binding affinity (pKd)',
        'description': '68 drugs × 442 kinases, Kd (nM, log-transformed to pKd)',
        'transform': 'pkd',  # -log10(Kd_nM / 1e9)
    },
    'kiba': {
        'property': 'KIBA integrated score',
        'description': '~118K drug-target pairs (from 2111×229 matrix), integrated binding score. '
                       'DeepDTA Y matrix already has higher = stronger binding (range ~0-17).',
        'transform': None,  # No transform needed — DeepDTA values are already affinity-scaled
    },
    'bindingdb_kd': {
        'property': 'Kd binding affinity (pKd)',
        'description': '~52K drug-target pairs (10,665 drugs × 1,413 proteins), Kd from BindingDB',
        'transform': 'pkd',
        'source': 'tdc',
    },
    'metz': {
        'property': 'Ki binding affinity (already log-scaled by TDC)',
        'description': '~35K drug-target pairs (~1,400 drugs × ~170 kinases), '
                       'Metz kinase profiling Ki. TDC ships values pre-log-transformed '
                       '(higher = stronger), so no pKd transform is applied.',
        'transform': None,  # TDC Metz Y is already log-scaled affinity (like KIBA)
        'source': 'tdc',
    },
}


def _cache_dir() -> Path:
    """Get cache directory."""
    return data_dir('dta')


def _download_file(url: str, dest: Path) -> None:
    """Download a file from URL to local path."""
    import requests
    print(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def _download_deepdta(name: str, cache: Path) -> Tuple[dict, dict, np.ndarray]:
    """Download DAVIS or KIBA from DeepDTA GitHub and return raw data.

    Returns:
        (ligands_dict, proteins_dict, Y_matrix)
    """
    raw_dir = cache / f'{name}_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    ligands_path = raw_dir / 'ligands_can.txt'
    proteins_path = raw_dir / 'proteins.txt'
    y_path = raw_dir / 'Y'

    if not ligands_path.exists():
        _download_file(f"{_BASE_URL}/{name}/ligands_can.txt", ligands_path)
    if not proteins_path.exists():
        _download_file(f"{_BASE_URL}/{name}/proteins.txt", proteins_path)
    if not y_path.exists():
        _download_file(f"{_BASE_URL}/{name}/Y", y_path)

    ligands = json.loads(ligands_path.read_text())
    proteins = json.loads(proteins_path.read_text())
    Y = pickle.loads(y_path.read_bytes(), encoding='latin1')
    Y = np.array(Y, dtype=np.float64)

    return ligands, proteins, Y


def _load_from_tdc(name_lower: str, info: dict, cache: Path) -> Tuple[List[str], List[str], np.ndarray, List[str], List[str]]:
    """Load a DTA dataset from TDC (Therapeutics Data Commons).

    Returns:
        (drug_smiles, protein_seqs, labels, drug_ids, protein_ids)
    """
    cached = cache / f'{name_lower}_processed.npz'
    cached_meta = cache / f'{name_lower}_meta.csv'

    if cached.exists() and cached_meta.exists():
        print(f"  Loading cached {name_lower} dataset...")
        data = np.load(cached, allow_pickle=True)
        meta_df = pd.read_csv(cached_meta)
        # Ensure IDs are strings (pandas may read numeric IDs as float)
        drug_ids = meta_df['Drug_ID'].fillna('unknown').astype(str).tolist()
        target_ids = meta_df['Target_ID'].fillna('unknown').astype(str).tolist()
        return (
            meta_df['Drug'].tolist(),
            meta_df['Target'].tolist(),
            data['labels'],
            drug_ids,
            target_ids,
        )

    try:
        from tdc.multi_pred import DTI
    except ImportError:
        raise ImportError("TDC required: pip install PyTDC")

    # Map dataset names to TDC names
    # TDC name strings are case-insensitive (TDC lowercases internally before
    # lookup). 'Metz' is the documented TDC DTI benchmark identifier for the
    # Metz et al. kinase profiling set (resolves to 'metz'). Confirmed via TDC
    # docs/source DTI group listing; some older PyTDC builds may not ship it,
    # in which case DTI(name='Metz') raises a clear TDC error.
    tdc_name_map = {
        'bindingdb_kd': 'BindingDB_Kd',
        'metz': 'Metz',
    }
    tdc_name = tdc_name_map.get(name_lower, name_lower)

    print(f"  Loading {tdc_name} from TDC...")
    dti = DTI(name=tdc_name)
    df = dti.get_data()

    # TDC DTI columns: Drug, Target, Y, Drug_ID, Target_ID
    df = df.dropna(subset=['Drug', 'Target', 'Y'])
    df = df[df['Y'] > 0]  # Filter non-positive Kd values (can't log-transform)

    drug_smiles = df['Drug'].tolist()
    protein_seqs = df['Target'].tolist()
    labels = df['Y'].values.astype(np.float64)

    # Use TDC-provided IDs, or derive from unique values
    if 'Drug_ID' in df.columns:
        drug_ids = df['Drug_ID'].fillna('unknown').astype(str).tolist()
    else:
        unique_drugs = sorted(set(drug_smiles))
        drug_map = {s: f'drug_{i}' for i, s in enumerate(unique_drugs)}
        drug_ids = [drug_map[s] for s in drug_smiles]

    if 'Target_ID' in df.columns:
        # TDC BindingDB has NaN Target_IDs for some entries — derive from sequence
        has_nan = df['Target_ID'].isna().any()
        if has_nan:
            # For entries with NaN Target_ID, use a hash of the protein sequence
            seq_to_id = {}
            for _, row in df.iterrows():
                tid = row['Target_ID']
                seq = row['Target']
                if pd.notna(tid):
                    seq_to_id[seq] = str(tid)
                elif seq not in seq_to_id:
                    seq_to_id[seq] = f'seq_{hash(seq) % 10**8}'
            protein_ids = [seq_to_id[s] for s in protein_seqs]
        else:
            protein_ids = df['Target_ID'].astype(str).tolist()
    else:
        unique_prots = sorted(set(protein_seqs))
        prot_map = {s: f'prot_{i}' for i, s in enumerate(unique_prots)}
        protein_ids = [prot_map[s] for s in protein_seqs]

    # Re-canonicalize SMILES
    from rdkit import Chem
    valid_mask = []
    for i, smi in enumerate(drug_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            drug_smiles[i] = Chem.MolToSmiles(mol, canonical=True)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    # Drop entries with invalid SMILES
    if not all(valid_mask):
        valid_mask = np.array(valid_mask)
        n_invalid = (~valid_mask).sum()
        drug_smiles = [s for s, v in zip(drug_smiles, valid_mask) if v]
        protein_seqs = [s for s, v in zip(protein_seqs, valid_mask) if v]
        drug_ids = [s for s, v in zip(drug_ids, valid_mask) if v]
        protein_ids = [s for s, v in zip(protein_ids, valid_mask) if v]
        labels = labels[valid_mask]
        print(f"  Dropped {n_invalid} entries with invalid SMILES")

    # Transform labels
    if info['transform'] == 'pkd':
        labels = -np.log10(labels / 1e9)

    n_drugs = len(set(drug_ids))
    n_proteins = len(set(protein_ids))
    print(f"  Loaded from TDC: {len(labels)} pairs from "
          f"{n_drugs} drugs × {n_proteins} proteins")

    # Cache processed data
    np.savez_compressed(cached, labels=labels)
    pd.DataFrame({
        'Drug': drug_smiles,
        'Target': protein_seqs,
        'Drug_ID': drug_ids,
        'Target_ID': protein_ids,
    }).to_csv(cached_meta, index=False)
    print(f"  Cached to {cached}")

    return drug_smiles, protein_seqs, labels, drug_ids, protein_ids


def load_dataset(name: str) -> Dict:
    """Load a DTA benchmark dataset.

    Args:
        name: 'davis', 'kiba', or 'bindingdb_kd'

    Returns:
        dict with keys:
            drug_smiles: List[str] — SMILES strings for each pair
            protein_seqs: List[str] — amino acid sequences for each pair
            labels: np.ndarray — binding affinity values
            task_type: 'regression'
            metadata: {drug_ids, protein_ids, n_drugs, n_proteins}
    """
    name_lower = name.lower()
    if name_lower not in _DATASETS:
        raise ValueError(
            f"Unknown DTA dataset: {name}. Available: {list(_DATASETS.keys())}"
        )

    info = _DATASETS[name_lower]
    cache = _cache_dir()

    # TDC-sourced datasets use a separate loader
    if info.get('source') == 'tdc':
        drug_smiles, protein_seqs, labels, drug_ids, protein_ids = \
            _load_from_tdc(name_lower, info, cache)
    else:
        # DeepDTA-sourced datasets (davis, kiba)
        cached = cache / f'{name_lower}_processed.npz'
        cached_meta = cache / f'{name_lower}_meta.csv'

        if cached.exists() and cached_meta.exists():
            print(f"  Loading cached {name_lower} dataset...")
            data = np.load(cached, allow_pickle=True)
            meta_df = pd.read_csv(cached_meta)
            drug_smiles = meta_df['Drug'].tolist()
            protein_seqs = meta_df['Target'].tolist()
            labels = data['labels']
            drug_ids = meta_df['Drug_ID'].tolist()
            protein_ids = meta_df['Target_ID'].tolist()
        else:
            ligands, proteins, Y = _download_deepdta(name_lower, cache)

            drug_keys = list(ligands.keys())
            protein_keys = list(proteins.keys())

            # Y matrix: rows = drugs, cols = proteins
            assert Y.shape[0] == len(drug_keys), \
                f"Y rows ({Y.shape[0]}) != drugs ({len(drug_keys)})"
            assert Y.shape[1] == len(protein_keys), \
                f"Y cols ({Y.shape[1]}) != proteins ({len(protein_keys)})"

            # Flatten to pairs, skipping NaN (KIBA has 75% missing)
            drug_smiles, protein_seqs, labels = [], [], []
            drug_ids, protein_ids = [], []

            for i, dk in enumerate(drug_keys):
                for j, pk in enumerate(protein_keys):
                    val = Y[i, j]
                    if np.isnan(val):
                        continue
                    drug_smiles.append(ligands[dk])
                    protein_seqs.append(proteins[pk])
                    drug_ids.append(dk)
                    protein_ids.append(pk)
                    labels.append(val)

            labels = np.array(labels, dtype=np.float64)

            # Re-canonicalize drug SMILES for consistency
            from rdkit import Chem
            for i, smi in enumerate(drug_smiles):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    drug_smiles[i] = Chem.MolToSmiles(mol, canonical=True)

            # Transform labels
            if info['transform'] == 'pkd':
                # DAVIS: Kd (nM) → pKd
                labels = -np.log10(labels / 1e9)
            elif info['transform'] == 'kiba':
                # KIBA: negate so higher = stronger binding (like pKd)
                labels = -labels

            print(f"  Downloaded {name_lower}: {len(labels)} pairs from "
                  f"{len(drug_keys)} drugs × {len(protein_keys)} proteins")

            # Cache processed data
            np.savez_compressed(cached, labels=labels)
            pd.DataFrame({
                'Drug': drug_smiles,
                'Target': protein_seqs,
                'Drug_ID': drug_ids,
                'Target_ID': protein_ids,
            }).to_csv(cached_meta, index=False)
            print(f"  Cached to {cached}")

    # Filter invalid entries
    valid = np.isfinite(labels)
    valid &= np.array([isinstance(s, str) and len(s) > 0 for s in drug_smiles])
    valid &= np.array([isinstance(s, str) and len(s) > 10 for s in protein_seqs])
    n_filtered = (~valid).sum()
    if n_filtered > 0:
        drug_smiles = [s for s, v in zip(drug_smiles, valid) if v]
        protein_seqs = [s for s, v in zip(protein_seqs, valid) if v]
        drug_ids = [s for s, v in zip(drug_ids, valid) if v]
        protein_ids = [s for s, v in zip(protein_ids, valid) if v]
        labels = labels[valid]
        print(f"  Filtered {n_filtered} invalid entries")

    n_drugs = len(set(drug_ids))
    n_proteins = len(set(protein_ids))

    print(f"  Loaded {name_lower}: {len(labels)} pairs, "
          f"{n_drugs} drugs × {n_proteins} proteins")
    print(f"  Label range: [{labels.min():.2f}, {labels.max():.2f}], "
          f"mean={labels.mean():.2f}")

    return {
        'drug_smiles': drug_smiles,
        'protein_seqs': protein_seqs,
        'labels': labels,
        'task_type': 'regression',
        'metadata': {
            'drug_ids': drug_ids,
            'protein_ids': protein_ids,
            'n_drugs': n_drugs,
            'n_proteins': n_proteins,
            'description': info['description'],
        },
    }


def assign_cold_target_groups(protein_ids: List[str]) -> Tuple[np.ndarray, int]:
    """Assign group IDs by protein for cold-target GroupKFold.

    Ensures that the same protein never appears in both train and test,
    preventing data leakage from memorizing protein-specific binding patterns.

    Args:
        protein_ids: List of protein identifiers (one per pair).

    Returns:
        (group_ids, n_unique_proteins)
    """
    unique_proteins = sorted(set(protein_ids))
    protein_to_id = {p: i for i, p in enumerate(unique_proteins)}
    groups = np.array([protein_to_id[p] for p in protein_ids])
    return groups, len(unique_proteins)


def assign_cold_drug_groups(drug_ids: List[str]) -> Tuple[np.ndarray, int]:
    """Assign group IDs by drug for cold-drug GroupKFold.

    Ensures that the same drug never appears in both train and test,
    preventing data leakage from memorizing drug-specific binding patterns.

    Args:
        drug_ids: List of drug identifiers (one per pair).

    Returns:
        (group_ids, n_unique_drugs)
    """
    unique_drugs = sorted(set(drug_ids))
    drug_to_id = {d: i for i, d in enumerate(unique_drugs)}
    groups = np.array([drug_to_id[d] for d in drug_ids])
    return groups, len(unique_drugs)
