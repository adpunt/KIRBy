"""
Ab-CoV Dataset Loader

Therapeutic antibodies against SARS-CoV-2 with IC50/Kd measurements.
Inter-family dataset (diverse antibodies, no single wildtype) — no delta reps.

Data from Rawat et al. (2022), available within FLAb:
  - rawat2022abcov_kd.csv: ~1,780 antibodies with Kd (nM) and -log(Kd)
  - rawat2022abcov_ic50.csv: IC50 measurements

Reference:
- Rawat et al. (2022). "Ab-CoV: SARS-CoV-2 antibody database"
"""

import numpy as np
from typing import Dict, Optional

from kirby.datasets.flab import _download_flab_csv


ABCOV_DATASETS = {
    'abcov_kd': {
        'path': 'binding/rawat2022abcov_kd.csv',
        'property': 'Binding (-log Kd)',
    },
    'abcov_ic50': {
        'path': 'binding/rawat2022abcov_ic50.csv',
        'property': 'Neutralization (IC50)',
    },
}


def load_abcov_dataset(
    name: str = 'abcov_kd',
    max_samples: int = 5000,
) -> Dict:
    """Load Ab-CoV dataset.

    Args:
        name: 'abcov_kd' or 'abcov_ic50'
        max_samples: Cap sample count (unlikely to trigger, ~1780 samples).

    Returns:
        dict with keys: heavy_seqs, light_seqs, labels, task_type,
                        wildtype_heavy (None), wildtype_light (None),
                        mutations (empty), is_intra_family (False),
                        property_name.
    """
    if name not in ABCOV_DATASETS:
        raise ValueError(f"Unknown Ab-CoV dataset: {name}. Options: {list(ABCOV_DATASETS.keys())}")

    info = ABCOV_DATASETS[name]
    df = _download_flab_csv(info['path'])

    # Filter valid
    valid = df['heavy'].notna() & df['fitness'].notna()
    valid &= df['heavy'].str.len() > 10
    df = df[valid].copy()
    df['fitness'] = df['fitness'].astype(float)
    df = df.dropna(subset=['fitness'])

    heavy_seqs = df['heavy'].tolist()
    light_seqs = None
    if 'light' in df.columns:
        light_valid = df['light'].notna() & (df['light'].str.len() > 10)
        if light_valid.sum() > len(heavy_seqs) * 0.5:
            light_seqs = df['light'].tolist()
            light_seqs = [s if isinstance(s, str) and len(s) > 10 else '' for s in light_seqs]

    labels = df['fitness'].values.astype(np.float64)

    # Subsample if needed
    if len(heavy_seqs) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(heavy_seqs), max_samples, replace=False)
        idx = sorted(idx)
        heavy_seqs = [heavy_seqs[i] for i in idx]
        if light_seqs:
            light_seqs = [light_seqs[i] for i in idx]
        labels = labels[idx]

    print(f"  Ab-CoV {name}: {len(heavy_seqs)} antibodies, "
          f"{'paired' if light_seqs else 'heavy-only'}, "
          f"property={info['property']}")
    print(f"  Fitness range: [{labels.min():.3f}, {labels.max():.3f}]")

    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'task_type': 'regression',
        'wildtype_heavy': None,
        'wildtype_light': None,
        'mutations': [[] for _ in heavy_seqs],
        'is_intra_family': False,
        'property_name': info['property'],
    }
