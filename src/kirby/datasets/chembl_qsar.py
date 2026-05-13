"""
ChEMBL QSAR Activity Cliff Dataset Loader

Downloads 3 curated target datasets from Dablander et al. 2023:
  - D2:   Dopamine D2 receptor (6333 compounds, Ki in nM)
  - FXa:  Factor Xa (3605 compounds, Ki in nM)
  - Mpro: SARS-CoV-2 main protease (1924 compounds, IC50 in uM)

Raw activity values are converted to pActivity = -log10(activity_M) for
consistent scale across targets. MMP pairs are loaded separately and
activity cliff delta is computed from the converted pActivity values.

Activity cliff = MMP pair where |Delta pActivity| > threshold (default 2.0).

Reference:
  Dablander et al. (2023). "Exploring QSAR models for activity-cliff prediction."
  Journal of Cheminformatics 15:47.
  https://github.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

_GITHUB_RAW = (
    "https://raw.githubusercontent.com/MarkusFerdinandDablander/"
    "QSAR-activity-cliff-experiments/main/data"
)

# Target name → directory name + CSV filenames (all repos use the same names)
# activity_col: exact column name in molecule_data_clean.csv
# activity_unit: 'nM' or 'uM' — used for pActivity conversion
TARGETS = {
    'd2': {
        'dir': 'chembl_dopamine_d2',
        'compound_csv': 'molecule_data_clean.csv',
        'mmp_csv': 'MMP_data_clean.csv',
        'activity_col': 'Ki [nM]',
        'activity_unit': 'nM',
        'label': 'Dopamine D2 receptor',
        'expected_n': 6333,
    },
    'fxa': {
        'dir': 'chembl_factor_xa',
        'compound_csv': 'molecule_data_clean.csv',
        'mmp_csv': 'MMP_data_clean.csv',
        'activity_col': 'Ki [nM]',
        'activity_unit': 'nM',
        'label': 'Factor Xa',
        'expected_n': 3605,
    },
    'mpro': {
        'dir': 'postera_sars_cov_2_mpro',
        'compound_csv': 'molecule_data_clean.csv',
        'mmp_csv': 'MMP_data_clean.csv',
        'activity_col': 'f_avg_IC50 [uM]',
        'activity_unit': 'uM',
        'label': 'SARS-CoV-2 Mpro',
        'expected_n': 1924,
    },
}


def _cache_dir() -> Path:
    """Cache directory for downloaded data."""
    seagate = Path('/Volumes/seagate/kirby/chembl_qsar')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    cache = cache_root / 'chembl_qsar'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_csv(target_dir: str, filename: str) -> pd.DataFrame:
    """Download CSV from Dablander GitHub repo."""
    import requests

    cache = _cache_dir() / target_dir
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / filename

    if cached.exists():
        return pd.read_csv(cached)

    url = f"{_GITHUB_RAW}/{target_dir}/{filename}"
    print(f"  Downloading {filename}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cached.write_bytes(resp.content)

    return pd.read_csv(cached)


def load_chembl_qsar(
    target: str = 'd2',
    ac_threshold: float = 2.0,
) -> Dict:
    """Load a ChEMBL QSAR dataset for activity cliff analysis.

    Args:
        target: 'd2' (dopamine D2), 'fxa' (Factor Xa), or 'mpro' (SARS-CoV-2 Mpro).
        ac_threshold: pIC50 difference threshold for activity cliff definition.

    Returns:
        dict with:
          smiles: list of SMILES strings
          labels: np.array of pIC50 values (regression target)
          task_type: 'regression'
          mmp_pairs: list of (idx_i, idx_j, delta_pic50) tuples (for AC analysis)
          n_activity_cliffs: number of ACs (|delta_pic50| > threshold)
          target_name: human-readable target name
    """
    target = target.lower()
    if target not in TARGETS:
        raise ValueError(
            f"Unknown target: {target}. Available: {sorted(TARGETS.keys())}")

    info = TARGETS[target]

    # Load compound data
    df = _download_csv(info['dir'], info['compound_csv'])

    # Columns: SMILES (first col) + activity (known column name)
    smiles_col = 'SMILES'
    activity_col = info['activity_col']

    if smiles_col not in df.columns:
        # Fallback: first column
        smiles_col = df.columns[0]
    if activity_col not in df.columns:
        raise ValueError(
            f"Activity column '{activity_col}' not found in {info['compound_csv']}. "
            f"Available columns: {list(df.columns)}")

    # Clean data: drop rows with missing SMILES or activity
    mask = df[smiles_col].notna() & df[activity_col].notna()
    df = df[mask].copy()

    # Remove zero/negative activity values before log transform
    raw_activity = df[activity_col].values.astype(np.float64)
    valid = raw_activity > 0
    df = df[valid].copy()
    raw_activity = raw_activity[valid]

    smiles = df[smiles_col].tolist()

    # Convert raw activity to pActivity = -log10(activity_in_M)
    # Ki [nM] → M: divide by 1e9.  IC50 [uM] → M: divide by 1e6.
    unit = info['activity_unit']
    if unit == 'nM':
        activity_M = raw_activity * 1e-9
    elif unit == 'uM':
        activity_M = raw_activity * 1e-6
    else:
        raise ValueError(f"Unknown activity unit: {unit}")

    labels = -np.log10(activity_M)

    print(f"  {info['label']}: {len(smiles)} compounds, "
          f"pActivity range [{labels.min():.2f}, {labels.max():.2f}] "
          f"(from {activity_col})")

    # Load MMP pairs
    # MMP CSV has: smiles_1, smiles_2, transformation, constant_part
    # Activity cliff delta is computed from compound pActivity values.
    mmp_pairs = []
    n_acs = 0
    try:
        mmp_df = _download_csv(info['dir'], info['mmp_csv'])

        # Build SMILES → index mapping
        smi_to_idx = {s: i for i, s in enumerate(smiles)}

        # Parse MMP pairs from smiles_1 and smiles_2 columns
        col1 = 'smiles_1'
        col2 = 'smiles_2'
        if col1 not in mmp_df.columns or col2 not in mmp_df.columns:
            # Fallback: first two columns with 'smiles' in name
            smi_cols = [c for c in mmp_df.columns if 'smiles' in c.lower()]
            if len(smi_cols) >= 2:
                col1, col2 = smi_cols[0], smi_cols[1]
            else:
                raise ValueError(f"Cannot find smiles columns in MMP CSV. "
                                 f"Columns: {list(mmp_df.columns)}")

        for _, row in mmp_df.iterrows():
            s1 = row[col1]
            s2 = row[col2]
            i = smi_to_idx.get(s1)
            j = smi_to_idx.get(s2)
            if i is not None and j is not None:
                delta = abs(labels[i] - labels[j])
                mmp_pairs.append((i, j, float(delta)))
                if delta >= ac_threshold:
                    n_acs += 1

        print(f"  {len(mmp_pairs)} MMP pairs, {n_acs} activity cliffs "
              f"(|delta pActivity| >= {ac_threshold})")
    except Exception as e:
        print(f"  MMP loading failed: {e}")

    return {
        'smiles': smiles,
        'labels': labels,
        'task_type': 'regression',
        'target_name': info['label'],
        'target_id': target,
        'mmp_pairs': mmp_pairs,
        'n_activity_cliffs': n_acs,
        'ac_threshold': ac_threshold,
    }
