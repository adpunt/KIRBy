"""
ITsFlexible CDR Flexibility Dataset Loader

Downloads CDR flexibility classification data from Spoendlin et al. 2025.
Binary classification: flexible (labels=1) vs rigid (labels=0) CDR3 loops.

A CDR3 loop is "flexible" if it adopts >= 2 conformational clusters
(RMSD > 1.25A between clusters) across available PDB structures.

Two modes:
  - cdr:   CDR-specific test data (CDRH3/CDRL3/CDRA3/CDRB3) with full Fv
            sequences. Works with antibody reps. Unique by CDR3 sequence.
  - loops: General protein loop training data (train+val+test). Works with
            protein reps only (no VH/VL, just loop sequences).

Reference:
  Spoendlin et al. (2025). "Predicting the conformational flexibility of
  antibody and T-cell receptor CDRs." Nature Machine Intelligence.
  GitHub: https://github.com/oxpig/ITsFlexible
  Zenodo: https://zenodo.org/records/15032263
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

_GITHUB_RAW = (
    "https://raw.githubusercontent.com/oxpig/ITsFlexible/main/data"
)

# CDR test files (have Fv sequences)
CDR_FILES = {
    'CDRH3': 'CDRH3_test_align_Fv.csv',
    'CDRL3': 'CDRL3_test_align_Fv.csv',
    'CDRA3': 'CDRA3_test_align_Fv.csv',
    'CDRB3': 'CDRB3_test_align_Fv.csv',
}

# General loop files (no Fv sequences)
LOOP_FILES = {
    'train': 'train_align_loop.csv',
    'val': 'val_align_loop.csv',
    'test': 'test_align_loop.csv',
}


def _cache_dir() -> Path:
    seagate = Path('/Volumes/seagate/kirby/itsflexible')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache = Path.home() / '.cache' / 'kirby' / 'itsflexible'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_csv(filename: str) -> pd.DataFrame:
    """Download CSV from ITsFlexible GitHub repo."""
    import requests

    cache = _cache_dir()
    cached = cache / filename

    if cached.exists():
        return pd.read_csv(cached)

    url = f"{_GITHUB_RAW}/{filename}"
    print(f"  Downloading {filename}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cached.write_bytes(resp.content)

    return pd.read_csv(cached)


def load_itsflexible(
    mode: str = 'cdr',
    cdr_types: Optional[List[str]] = None,
) -> Dict:
    """Load ITsFlexible CDR flexibility dataset.

    Args:
        mode: 'cdr' for CDR-specific data with Fv sequences (antibody reps),
              'loops' for general protein loop data (protein reps only).
        cdr_types: which CDR types to include (default: ['CDRH3', 'CDRL3']).
                   Only used in 'cdr' mode. Options: CDRH3, CDRL3, CDRA3, CDRB3.

    Returns:
        For mode='cdr':
          dict with heavy_seqs, light_seqs, labels, cdr_seqs, cdr_types_arr,
                   task_type='classification', n_classes=2
        For mode='loops':
          dict with loop_seqs, labels, task_type='classification', n_classes=2
    """
    if mode == 'cdr':
        return _load_cdr(cdr_types or ['CDRH3', 'CDRL3'])
    elif mode == 'loops':
        return _load_loops()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'cdr' or 'loops'.")


def _load_cdr(cdr_types: List[str]) -> Dict:
    """Load CDR-specific test data with full Fv sequences."""
    all_heavy, all_light, all_labels = [], [], []
    all_cdr_seqs, all_cdr_types = [], []

    for cdr_type in cdr_types:
        if cdr_type not in CDR_FILES:
            print(f"  WARNING: Unknown CDR type '{cdr_type}', skipping")
            continue

        df = _download_csv(CDR_FILES[cdr_type])

        # Find CDR sequence column (named after CDR type)
        cdr_col = cdr_type
        if cdr_col not in df.columns:
            # Try case variations
            for col in df.columns:
                if col.lower() == cdr_type.lower():
                    cdr_col = col
                    break

        if cdr_col not in df.columns:
            print(f"  WARNING: CDR column '{cdr_type}' not found in {CDR_FILES[cdr_type]}")
            continue

        # Filter to rows with valid labels and Fv FIRST, then dedup.
        # (Dedup before filter could lose valid rows if the first occurrence
        # of a CDR3 sequence has NaN labels but a later duplicate is valid.)
        mask = df['labels'].notna() & df['Fv'].notna()
        df_valid = df[mask]
        df_unique = df_valid.drop_duplicates(subset=[cdr_col], keep='first').copy()

        n_total = len(df_unique)
        n_flex = int((df_unique['labels'] == 1).sum())
        print(f"  {cdr_type}: {n_total} unique sequences "
              f"({n_flex} flexible, {n_total - n_flex} rigid)")

        for _, row in df_unique.iterrows():
            fv = str(row['Fv'])
            cdr_seq = str(row[cdr_col])
            label = int(row['labels'])

            # Split Fv into VH and VL (separated by underscore)
            parts = fv.split('_')
            if len(parts) >= 2:
                vh = parts[0]
                vl = parts[1]
            else:
                vh = fv
                vl = ''

            all_heavy.append(vh)
            all_light.append(vl)
            all_labels.append(label)
            all_cdr_seqs.append(cdr_seq)
            all_cdr_types.append(cdr_type)

    if not all_labels:
        raise RuntimeError("No CDR data loaded. Check CDR type names and data availability.")

    labels = np.array(all_labels, dtype=np.int64)
    n = len(labels)
    n_flex = int((labels == 1).sum())

    print(f"  Total: {n} unique CDR sequences, "
          f"{n_flex} flexible ({100*n_flex/n:.1f}%), "
          f"{n - n_flex} rigid ({100*(n-n_flex)/n:.1f}%)")

    return {
        'heavy_seqs': all_heavy,
        'light_seqs': all_light,
        'labels': labels,
        'cdr_seqs': all_cdr_seqs,
        'cdr_types': all_cdr_types,
        'task_type': 'classification',
        'n_classes': 2,
        'is_intra_family': False,
        'property_name': 'CDR flexibility (flexible=1, rigid=0)',
        'mutations': [[] for _ in range(n)],
        'wildtype_heavy': None,
        'wildtype_light': None,
    }


def _load_loops() -> Dict:
    """Load general protein loop training data."""
    dfs = []
    for split_name, filename in LOOP_FILES.items():
        df = _download_csv(filename)
        df['_split'] = split_name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Deduplicate by sequence
    df_unique = df.drop_duplicates(subset=['seq'], keep='first').copy()
    df_unique = df_unique[df_unique['labels'].notna()]

    seqs = df_unique['seq'].tolist()
    labels = df_unique['labels'].values.astype(np.int64)
    n = len(labels)
    n_flex = int((labels == 1).sum())

    print(f"  General loops: {n} unique sequences, "
          f"{n_flex} flexible ({100*n_flex/n:.1f}%), "
          f"{n - n_flex} rigid ({100*(n-n_flex)/n:.1f}%)")

    return {
        'loop_seqs': seqs,
        'labels': labels,
        'task_type': 'classification',
        'n_classes': 2,
        'property_name': 'Loop flexibility (flexible=1, rigid=0)',
    }
