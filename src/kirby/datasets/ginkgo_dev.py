"""
Ginkgo Developability Dataset Loader

Downloads the GDPa1 antibody developability benchmark from Ginkgo Bioworks.
242 antibodies with 9 biophysical property measurements and pre-computed 5-fold splits.

Properties (5 core + 4 additional):
  Core: HIC, Tm2, Titer, PR_CHO, AC-SINS_pH7.4
  Additional: Purity, SEC %Monomer, SMAC, HAC

Each property becomes a separate regression dataset (e.g. 'ginkgo_hic', 'ginkgo_tm2').
Fold assignments from hierarchical clustering preserve structural diversity across folds.

Reference:
  van Niekerk et al., "2025 Ginkgo Datapoints Antibody Developability Competition
  outcomes: limited model performance and a call for data standardization" (2026)
  https://github.com/ginkgobioworks/abdev-benchmark
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from kirby.paths import data_dir

# Map KIRBy dataset names → CSV column names
GINKGO_PROPERTIES = {
    'ginkgo_hic':         'HIC',
    'ginkgo_tm2':         'Tm2',
    'ginkgo_titer':       'Titer',
    'ginkgo_pr_cho':      'PR_CHO',
    'ginkgo_acsins':      'AC-SINS_pH7.4',
    'ginkgo_purity':      'Purity',
    'ginkgo_aggregation': 'SEC %Monomer',
    'ginkgo_colloidal':   'SMAC',
    'ginkgo_heparin':     'HAC',
}


def _cache_dir() -> Path:
    """Cache dir."""
    return data_dir('ginkgo_dev')


def _download_ginkgo() -> pd.DataFrame:
    """Download or load cached GDPa1 CSV."""
    import requests

    cache = _cache_dir()
    cached = cache / 'GDPa1_v1.2_20250814.csv'

    if cached.exists():
        return pd.read_csv(cached)

    # Try HuggingFace resolve URL first, then GitHub raw
    urls = [
        "https://huggingface.co/datasets/ginkgo-datapoints/GDPa1/resolve/main/GDPa1_v1.2_20250814.csv",
        "https://raw.githubusercontent.com/ginkgobioworks/abdev-benchmark/main/data/GDPa1_v1.2_20250814.csv",
    ]

    for url in urls:
        try:
            print(f"  Downloading GDPa1 from {url}...")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            cached.write_bytes(resp.content)
            return pd.read_csv(cached)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    raise RuntimeError(
        "Could not download GDPa1 dataset. Please download manually from "
        "https://huggingface.co/datasets/ginkgo-datapoints/GDPa1 "
        f"and place CSV at {cached}"
    )


def load_ginkgo_dataset(name: str = 'ginkgo_hic') -> Dict:
    """Load one Ginkgo developability property as a regression dataset.

    Args:
        name: KIRBy dataset name (e.g. 'ginkgo_hic', 'ginkgo_tm2').

    Returns:
        dict with: heavy_seqs, light_seqs, labels, task_type, fold_assignments,
                   is_intra_family, property_name, mutations, wildtype
    """
    if name not in GINKGO_PROPERTIES:
        raise ValueError(
            f"Unknown Ginkgo property: {name}. "
            f"Available: {sorted(GINKGO_PROPERTIES.keys())}"
        )

    col = GINKGO_PROPERTIES[name]
    df = _download_ginkgo()

    # Pre-computed 5-fold cluster assignments (0-4)
    fold_col = 'hierarchical_cluster_IgG_isotype_stratified_fold'

    # Filter rows with valid property values, sequences, AND fold IDs.
    # fold_col is included in the mask so the .astype(int) below cannot crash
    # on a NaN fold (rows without a cluster assignment can't be CV-split
    # anyway). Without this guard astype(int) raises ValueError: cannot
    # convert float NaN to integer.
    mask = (df[col].notna()
            & df['vh_protein_sequence'].notna()
            & df['vl_protein_sequence'].notna()
            & df[fold_col].notna())
    df = df[mask].copy()

    heavy_seqs = df['vh_protein_sequence'].tolist()
    light_seqs = df['vl_protein_sequence'].tolist()
    labels = df[col].values.astype(np.float64)

    fold_assignments = df[fold_col].values.astype(int)

    n = len(heavy_seqs)
    print(f"  Ginkgo {col}: {n} samples, "
          f"label range [{labels.min():.3f}, {labels.max():.3f}]")

    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'task_type': 'regression',
        'fold_assignments': fold_assignments,
        'is_intra_family': False,
        'property_name': f'Ginkgo developability: {col}',
        'mutations': [[] for _ in range(n)],
        'wildtype_heavy': None,
        'wildtype_light': None,
    }
