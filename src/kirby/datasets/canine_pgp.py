"""Canine ABCB1 (P-glycoprotein) substrate / inhibitor data loader.

Underlies the breed-MDR1 (ABCB1-1Δ) story for collies & related breeds —
loss-of-function frameshift means substrates that are tolerated in
non-affected dogs (and humans) cause severe CNS toxicity in MDR1-deficient
dogs. Standard Lhasa-relevant vet ADMET endpoint.

Like ``canine_cyp.py`` this is mainly a literature CSV at
``data/raw/canine_pgp.csv``; ChEMBL ABCB1 dog data is sparse. Resolution
order per ``organism``:

1. CSV at ``data/raw/canine_pgp{_<organism>}.csv`` if present.
2. ChEMBL target ID from ``docs/cross_species_targets_resolved.json``
   (gene key ``ABCB1``) → live activity query.
3. Documented skip with sparse-data log.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from kirby.utils.species import resolve_species
from kirby.datasets.canine_cyp import (
    SPARSE_THRESHOLD,
    _load_chembl,  # reusable single-target ChEMBL fetch
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESOLVED_JSON = _REPO_ROOT / 'docs' / 'cross_species_targets_resolved.json'
DEFAULT_DATA_RAW_DIR = _REPO_ROOT / 'data' / 'raw'

_GENE = 'ABCB1'


def _resolved_target_id(organism: str) -> Optional[str]:
    if not _RESOLVED_JSON.exists():
        return None
    try:
        data = json.loads(_RESOLVED_JSON.read_text())
    except Exception:
        return None
    entry = data.get(_GENE, {}).get(organism)
    if not entry:
        return None
    if entry.get('flag') in ('none', 'skip', 'error'):
        return None
    return entry.get('best')


def _csv_candidates(organism: str, data_raw_dir: Path) -> List[Path]:
    base = data_raw_dir / 'canine_pgp'
    return [
        base.with_name(f'{base.name}_{organism}.csv'),
        base.with_suffix('.csv'),
    ]


def _load_csv(organism: str, data_raw_dir: Path) -> Optional[pd.DataFrame]:
    for path in _csv_candidates(organism, data_raw_dir):
        if path.exists():
            df = pd.read_csv(path)
            if 'smiles' not in df.columns or 'label' not in df.columns:
                raise ValueError(
                    f"{path} must contain 'smiles' and 'label' columns; "
                    f"found {list(df.columns)}."
                )
            df = df.dropna(subset=['smiles', 'label']).copy()
            df.attrs['source_path'] = str(path)
            return df
    return None


def load_canine_pgp(organism: str = 'dog',
                    prefer: str = 'csv',
                    csv_dir: Optional[Path] = None) -> Dict:
    """Load canine P-glycoprotein (ABCB1) substrate / inhibitor data.

    Args:
        organism: Species key from ``SPECIES_REGISTRY``. Default 'dog'.
            'human' is supported via the resolved ABCB1 entry for
            cross-species comparison.
        prefer: 'csv' (default) checks the hand-curated CSV first and
            falls back to ChEMBL; 'chembl' inverts the order.
        csv_dir: Override for ``data/raw`` lookup (testing).

    Returns:
        dict with the same shape as ``canine_cyp.load_canine_cyp``.
    """
    resolve_species(organism)

    data_raw_dir = Path(csv_dir) if csv_dir is not None else DEFAULT_DATA_RAW_DIR

    csv_df: Optional[pd.DataFrame] = None
    chembl_df: Optional[pd.DataFrame] = None

    if prefer == 'csv':
        csv_df = _load_csv(organism, data_raw_dir)
        if csv_df is None or len(csv_df) == 0:
            tid = _resolved_target_id(organism)
            if tid is not None:
                chembl_df = _load_chembl(tid, 'ABCB1', organism)
    elif prefer == 'chembl':
        tid = _resolved_target_id(organism)
        if tid is not None:
            chembl_df = _load_chembl(tid, 'ABCB1', organism)
        if chembl_df is None or len(chembl_df) == 0:
            csv_df = _load_csv(organism, data_raw_dir)
    else:
        raise ValueError(f"prefer must be 'csv' or 'chembl', got '{prefer}'")

    if csv_df is not None and len(csv_df) > 0:
        df = csv_df
        source = 'csv'
    elif chembl_df is not None and len(chembl_df) > 0:
        df = chembl_df
        source = 'chembl'
    else:
        print(
            f"  WARNING: no canine P-gp / ABCB1 data found for "
            f"organism='{organism}'. Checked CSV "
            f"{_csv_candidates(organism, data_raw_dir)} "
            f"and resolved JSON gene='ABCB1'. "
            f"Documented skip."
        )
        return {
            'smiles': [],
            'labels': np.array([], dtype=float),
            'organism': organism,
            'gene': _GENE,
            'source': 'none',
            'source_path': None,
            'n_compounds': 0,
        }

    n = len(df)
    if n < SPARSE_THRESHOLD:
        print(
            f"  WARNING: only {n} compounds for canine P-gp / ABCB1 "
            f"(organism='{organism}', source='{source}') — below the "
            f"{SPARSE_THRESHOLD}-row inclusion threshold. "
            f"Documented skip."
        )

    smiles = df['smiles'].astype(str).tolist()
    labels = np.asarray(df['label'].values, dtype=float)
    result: Dict = {
        'smiles': smiles,
        'labels': labels,
        'organism': organism,
        'gene': _GENE,
        'source': source,
        'source_path': df.attrs.get('source_path'),
        'n_compounds': n,
    }
    for col in df.columns:
        if col in ('smiles', 'label'):
            continue
        result.setdefault('extras', {})[col] = df[col].tolist()
    return result
