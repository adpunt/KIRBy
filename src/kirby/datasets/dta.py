"""
Drug-Target Affinity (DTA) Dataset Loader

Loads DAVIS and KIBA benchmark datasets directly from DeepDTA GitHub.
No external dependencies beyond numpy/pandas/requests.

DAVIS: 68 drugs × 442 kinases, Kd values (nM → pKd)
KIBA: 2,111 drugs × 229 proteins, integrated KIBA scores (~118K valid pairs)

Reference:
- Davis et al. (2011) Nature Biotechnology (DAVIS kinase binding)
- Tang et al. (2014) Bioinformatics (KIBA score)
- Öztürk et al. (2018) Bioinformatics (DeepDTA, data source)
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

_BASE_URL = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"

_DATASETS = {
    'davis': {
        'property': 'Kd binding affinity (pKd)',
        'description': '68 drugs × 442 kinases, Kd (nM, log-transformed to pKd)',
        'transform': 'pkd',  # -log10(Kd_nM / 1e9)
    },
    'kiba': {
        'property': 'KIBA integrated score',
        'description': '~118K drug-target pairs (from 2111×229 matrix), integrated binding score',
        'transform': None,  # already in KIBA units
    },
}


def _cache_dir() -> Path:
    """Get cache directory, preferring /Volumes/seagate if available."""
    seagate = Path('/Volumes/seagate/kirby/dta')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache = Path.home() / '.cache' / 'kirby' / 'dta'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


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


def load_dataset(name: str) -> Dict:
    """Load a DTA benchmark dataset.

    Args:
        name: 'davis' or 'kiba'

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

        # Transform DAVIS Kd (nM) → pKd
        if info['transform'] == 'pkd':
            labels = -np.log10(labels / 1e9)

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
