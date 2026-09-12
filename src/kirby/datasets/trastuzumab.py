"""
Trastuzumab-HER2 Binding Dataset Loader

Downloads and parses the Trastuzumab CDRH3 variant dataset from Chinery et al. 2024.
524K CDRH3 variants with binary binding labels (binder vs non-binder).

All variants share the trastuzumab framework — only the 10-residue CDRH3 varies.
Full VH is reconstructed by splicing each variant into the framework.

Two presets:
  - tz_her2_binary:     Binary classification (label: 0=non-binder, 1=binder)
  - tz_her2_multiclass: 3-class classification (low=0, mid=1, high=2)

Reference:
  Chinery et al., "Baselining the Buzz: Trastuzumab-HER2 Affinity, and Beyond" (2024)
  https://github.com/lewis-chinery/Tz_her2_affinity_and_beyond
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from kirby.paths import data_dir

# Trastuzumab framework sequences (verified from source repo utils.py)
VH_PRE = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR"
CDRH3_WT = "WGGDGFYAMD"  # 10-residue wildtype CDRH3
VH_POST = "YWGQGTLVTVSS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"

# Full wildtype VH for reference
VH_WT = VH_PRE + CDRH3_WT + VH_POST  # 120 residues

_GITHUB_RAW = (
    "https://raw.githubusercontent.com/lewis-chinery/Tz_her2_affinity_and_beyond"
    "/main/data/affinity_data/her2/her2_aff_large/processed/remove_overlap"
    "/random_split/0.7_0.15_0.15"
)

CLASS_TO_INT = {'low': 0, 'mid': 1, 'high': 2}


def _cache_dir() -> Path:
    """Get cache directory."""
    return data_dir('trastuzumab')


def _download_csv(split: str) -> pd.DataFrame:
    """Download a split CSV from GitHub (or load from cache)."""
    import requests

    cache = _cache_dir()
    cached = cache / f'{split}.csv'

    if cached.exists():
        return pd.read_csv(cached)

    url = f"{_GITHUB_RAW}/{split}.csv"
    print(f"  Downloading trastuzumab {split}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    return pd.read_csv(cached)


def _extract_mutations(cdrh3: str, wt: str = CDRH3_WT) -> List[Dict]:
    """Extract mutations by comparing CDRH3 variant to wildtype."""
    mutations = []
    # Offset into full VH where CDRH3 starts
    cdrh3_offset = len(VH_PRE)
    min_len = min(len(cdrh3), len(wt))
    for i in range(min_len):
        if cdrh3[i] != wt[i]:
            mutations.append({
                'pos': cdrh3_offset + i,
                'wt_aa': wt[i],
                'mut_aa': cdrh3[i],
                'chain': 'heavy',
            })
    if len(cdrh3) > len(wt):
        for i in range(len(wt), len(cdrh3)):
            mutations.append({
                'pos': cdrh3_offset + i,
                'wt_aa': '-',
                'mut_aa': cdrh3[i],
                'chain': 'heavy',
            })
    elif len(cdrh3) < len(wt):
        for i in range(len(cdrh3), len(wt)):
            mutations.append({
                'pos': cdrh3_offset + i,
                'wt_aa': wt[i],
                'mut_aa': '-',
                'chain': 'heavy',
            })
    return mutations


def load_trastuzumab_dataset(
    preset: str = 'binary',
    max_samples: int = 5000,
) -> Dict:
    """Load the Trastuzumab-HER2 binding dataset.

    Args:
        preset: 'binary' (binder vs non-binder) or 'multiclass' (low/mid/high).
        max_samples: Subsample to this size (stratified by label + edit distance).

    Returns:
        dict with keys:
            heavy_seqs: List[str] - reconstructed full VH sequences
            light_seqs: List[str] - trastuzumab VL (identical for all)
            labels: np.ndarray - integer class labels
            task_type: str - 'classification'
            wildtype_heavy: str - wildtype VH
            wildtype_light: str - trastuzumab VL
            mutations: List[List[Dict]] - per-sample mutation lists
            is_intra_family: bool - True (all share trastuzumab framework)
            property_name: str - description of the task
            edit_distances: np.ndarray - edit distance from wildtype CDRH3
            n_classes: int - number of classes
    """
    # Load and merge all splits
    dfs = []
    for split in ['train', 'val', 'test']:
        df = _download_csv(split)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Validate columns
    required = {'seq', 'class', 'label', 'edit_distance'}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}, got {list(df.columns)}")

    # Drop rows with missing/invalid sequences
    df = df[df['seq'].notna() & (df['seq'].str.len() == 10)].copy()

    # Parse labels
    if preset == 'binary':
        labels = df['label'].values.astype(np.int64)
        property_name = 'HER2 binding (binary: binder=1, non-binder=0)'
        n_classes = 2
    elif preset == 'multiclass':
        df['class_int'] = df['class'].map(CLASS_TO_INT)
        df = df.dropna(subset=['class_int'])
        labels = df['class_int'].values.astype(np.int64)
        property_name = 'HER2 binding affinity class (low=0, mid=1, high=2)'
        n_classes = 3
    else:
        raise ValueError(f"Unknown preset: {preset}. Use 'binary' or 'multiclass'.")

    edit_distances = df['edit_distance'].values.astype(np.int64)
    cdrh3_seqs = df['seq'].tolist()

    print(f"  Trastuzumab ({preset}): {len(cdrh3_seqs)} total samples")
    for c in range(n_classes):
        n_c = (labels == c).sum()
        print(f"    Class {c}: {n_c} ({100*n_c/len(labels):.1f}%)")

    # Stratified subsampling on (label, edit_distance) to preserve distributions
    if len(cdrh3_seqs) > max_samples:
        print(f"  Subsampling {len(cdrh3_seqs)} -> {max_samples}...")
        rng = np.random.RandomState(42)

        # Create strata from label + edit_distance
        strata = labels * 100 + edit_distances  # unique combo key
        unique_strata = np.unique(strata)

        indices = []
        # Allocate proportionally to each stratum
        for s in unique_strata:
            s_idx = np.where(strata == s)[0]
            n_take = max(1, int(len(s_idx) / len(labels) * max_samples))
            n_take = min(n_take, len(s_idx))
            indices.extend(rng.choice(s_idx, n_take, replace=False).tolist())

        # Trim or fill to exact target
        if len(indices) > max_samples:
            indices = rng.choice(indices, max_samples, replace=False).tolist()
        elif len(indices) < max_samples:
            remaining = max_samples - len(indices)
            available = list(set(range(len(labels))) - set(indices))
            if available:
                indices.extend(rng.choice(available,
                                          min(remaining, len(available)),
                                          replace=False).tolist())
        indices = sorted(indices[:max_samples])

        cdrh3_seqs = [cdrh3_seqs[i] for i in indices]
        labels = labels[indices]
        edit_distances = edit_distances[indices]

        print(f"  After subsampling: {len(cdrh3_seqs)} samples")
        for c in range(n_classes):
            n_c = (labels == c).sum()
            print(f"    Class {c}: {n_c} ({100*n_c/len(labels):.1f}%)")

    # Reconstruct full VH sequences
    heavy_seqs = [VH_PRE + cdrh3 + VH_POST for cdrh3 in cdrh3_seqs]
    light_seqs = [VL] * len(heavy_seqs)

    # Extract mutations per sample
    mutations = [_extract_mutations(cdrh3) for cdrh3 in cdrh3_seqs]
    n_mutated = sum(1 for m in mutations if len(m) > 0)
    avg_muts = np.mean([len(m) for m in mutations])
    print(f"  {n_mutated}/{len(heavy_seqs)} samples have mutations, "
          f"avg {avg_muts:.1f} mutations/sample")
    print(f"  Edit distance range: [{edit_distances.min()}, {edit_distances.max()}]")

    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'task_type': 'classification',
        'wildtype_heavy': VH_WT,
        'wildtype_light': VL,
        'mutations': mutations,
        'is_intra_family': True,
        'property_name': property_name,
        'edit_distances': edit_distances,
        'n_classes': n_classes,
    }
