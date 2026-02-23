"""
Drug-Target Affinity (DTA) Dataset Loader

Loads DAVIS and KIBA benchmark datasets via PyTDC (Therapeutics Data Commons).

DAVIS: 30,056 drug-target pairs, 68 drugs × 442 kinases, Kd values
KIBA: ~117K drug-target pairs, integrated KIBA scores

Reference:
- Huang et al. (2021) "Therapeutics Data Commons" NeurIPS Datasets & Benchmarks
- Davis et al. (2011) Nature Biotechnology (DAVIS kinase binding)
- Tang et al. (2014) Bioinformatics (KIBA score)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


_DATASETS = {
    'davis': {
        'tdc_name': 'DAVIS',
        'property': 'Kd binding affinity',
        'description': '68 drugs × 442 kinases, Kd (nM, log-transformed)',
    },
    'kiba': {
        'tdc_name': 'KIBA',
        'property': 'KIBA integrated score',
        'description': '~117K drug-target pairs, integrated binding score',
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
            metadata: {drug_ids, protein_ids, protein_families, n_drugs, n_proteins}
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
        try:
            from tdc.multi_pred import DTI
        except ImportError:
            raise ImportError(
                "DTA datasets require PyTDC: pip install PyTDC"
            )

        print(f"  Downloading {name_lower} via PyTDC...")
        dti = DTI(name=info['tdc_name'], path=str(cache))
        df = dti.get_data()

        # PyTDC columns: Drug_ID, Drug, Target_ID, Target, Y
        drug_smiles = df['Drug'].tolist()
        protein_seqs = df['Target'].tolist()
        labels = df['Y'].values.astype(np.float64)
        drug_ids = df['Drug_ID'].tolist()
        protein_ids = df['Target_ID'].tolist()

        # Cache processed data
        np.savez_compressed(cached, labels=labels)
        df[['Drug', 'Target', 'Drug_ID', 'Target_ID']].to_csv(cached_meta, index=False)
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

    # Compute unique drug/protein counts
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
