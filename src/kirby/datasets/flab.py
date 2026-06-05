"""
FLAb (Fitness Landscape for Antibodies) Dataset Loader

Downloads and parses antibody fitness datasets from the FLAb benchmark.
GitHub: https://github.com/Graylab/FLAb

FLAb CSV format: heavy, light, <assay_name>, fitness
All datasets provide full VH/VL sequences with continuous fitness measurements.

Intra-family datasets (SSM/DMS of one wildtype) support delta representations.

ISS-78 (EXPECTED RESULT, not a bug — flag for write-up):
  Antibody binding/expression/thermostability tasks are near-unpredictable per
  recent benchmarks. The FLAb paper itself evaluated ~30 models and found no
  significant correlation on most of its datasets. So near-zero or negative
  correlation/R² here is an EXPECTED outcome of the task's difficulty, NOT a
  defect in this loader or the pipeline.

Reference:
- Graylab/FLAb: A benchmark of antibody fitness landscapes (2024)
"""

import io
import os
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_GITHUB_RAW = "https://raw.githubusercontent.com/Graylab/FLAb/main/data"

# Preset datasets: name → (path, property description, is_intra_family)
# Selected for: >200 samples, continuous targets, spanning different properties
FLAB_DATASETS = {
    'flab_expression_koenig': {
        'path': 'expression/koenig2017mutational_er_g6.csv',
        'property': 'Expression (enrichment ratio)',
        'intra_family': True,
    },
    'flab_binding_koenig': {
        'path': 'binding/koenig2017mutational_kd_g6.csv',
        'property': 'Binding (Kd)',
        'intra_family': True,
    },
    'flab_thermo_jain': {
        'path': 'thermostability/jain2017biophysical_Tm.csv',
        'property': 'Thermostability (Tm, °C)',
        'intra_family': False,
    },
    'flab_aggregation_jain': {
        'path': 'aggregation/jain2017biophysical_ACSINS.csv',
        'property': 'Aggregation (AC-SINS, nm)',
        'intra_family': False,
    },
    'flab_thermo_hie': {
        'path': 'thermostability/hie2023efficient_S309_Tm.csv',
        'property': 'Thermostability (Tm, °C)',
        'intra_family': True,
    },
    'flab_binding_rosace': {
        'path': 'binding/rosace2023automated_kd_adalimumab.csv',
        'property': 'Binding (Kd)',
        'intra_family': True,
    },
    # ── FLAb2 expansion (Dec 2025) ──────────────────────────────────────────
    'flab_binding_phillips_h3': {
        'path': 'binding/phillips2021binding_cr9114_h3_kd.csv',
        'property': 'Binding (SPR Kd)',
        'intra_family': True,
    },
    'flab_binding_adams': {
        'path': 'binding/adams2017measuring_4420-fluorescein_kd-titeseq.csv',
        'property': 'Binding (Kd)',
        'intra_family': True,
    },
    'flab_expression_adams': {
        'path': 'expression/adams2017measuring_4420-fluorescein_exp_er.csv',
        'property': 'Expression (enrichment ratio)',
        'intra_family': True,
    },
    'flab_thermo_tresanco': {
        'path': 'thermostability/tresanco2023nbthermo_tm.csv',
        'property': 'Thermostability (Tm)',
        'intra_family': False,
    },
    'flab_thermo_hutchinson': {
        'path': 'thermostability/hutchinson2023enhancement_top200tm1_igg.csv',
        'property': 'Thermostability (Tm)',
        'intra_family': False,
    },
    'flab_immunogenicity_marks': {
        'path': 'immunogenicity/marks2021humanization_immunogenicity.csv',
        'property': 'Immunogenicity',
        'intra_family': False,
    },
    'flab_aggregation_jain_hicrt': {
        'path': 'aggregation/jain2017biophyscial_HICRT.csv',
        'property': 'Aggregation (HIC RT)',
        'intra_family': False,
    },
    'flab_polyreact_jain_bvp': {
        'path': 'polyreactivity/jain2017biophysical_BVPELISA.csv',
        'property': 'Polyreactivity (BVP ELISA)',
        'intra_family': False,
    },
    'flab_polyreact_jain_cic': {
        'path': 'polyreactivity/jain2017biophysical_CICRT.csv',
        'property': 'Polyreactivity (CIC RT)',
        'intra_family': False,
    },
    'flab_pk_jain_fcrnrt': {
        'path': 'pharmacokinetics/jain2023identifying_FcRnRelRT3.csv',
        'property': 'PK (FcRn retention time)',
        'intra_family': False,
    },
    'flab_pk_jain_heprt': {
        'path': 'pharmacokinetics/jain2023identifying_HEPRT3.csv',
        'property': 'PK (heparin retention time)',
        'intra_family': False,
    },
    'flab_aggregation_jain_sas': {
        'path': 'aggregation/jain2017biophysical_SAS.csv',
        'property': 'Aggregation (SAS)',
        'intra_family': False,
    },
}


def _cache_dir() -> Path:
    """Get cache directory, preferring /Volumes/seagate if available."""
    seagate = Path('/Volumes/seagate/kirby/flab')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    cache = cache_root / 'flab'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_flab_csv(relative_path: str) -> pd.DataFrame:
    """Download a FLAb CSV from GitHub (or load from cache)."""
    import requests

    cache = _cache_dir()
    # Flatten path for cache filename
    cache_name = relative_path.replace('/', '_')
    cached = cache / cache_name

    if cached.exists():
        return pd.read_csv(cached)

    url = f"{_GITHUB_RAW}/{relative_path}"
    print(f"  Downloading FLAb: {relative_path}...")

    is_zip = relative_path.endswith('.csv.zip')
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    if is_zip:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)
        # Cache the extracted CSV
        df.to_csv(cached.with_suffix('').with_suffix('.csv') if is_zip else cached, index=False)
    else:
        cached.write_bytes(resp.content)
        df = pd.read_csv(cached)

    return df


def _identify_wildtype(heavy_seqs: List[str], light_seqs: Optional[List[str]]) -> Tuple[str, Optional[str]]:
    """Identify wildtype as the most common sequence (mode)."""
    from collections import Counter
    wt_heavy = Counter(heavy_seqs).most_common(1)[0][0]
    wt_light = None
    if light_seqs:
        wt_light = Counter(light_seqs).most_common(1)[0][0]
    return wt_heavy, wt_light


def _extract_mutations(seq: str, wt: str) -> List[Dict]:
    """Extract mutations by pairwise comparison to wildtype.

    Returns list of dicts: [{'pos': int, 'wt_aa': str, 'mut_aa': str}, ...]
    Handles equal-length sequences (SSM/DMS) and length mismatches (truncation).
    """
    mutations = []
    min_len = min(len(seq), len(wt))
    for i in range(min_len):
        if seq[i] != wt[i]:
            mutations.append({'pos': i, 'wt_aa': wt[i], 'mut_aa': seq[i]})
    # If lengths differ, count as insertion/deletion at the end
    if len(seq) > len(wt):
        for i in range(len(wt), len(seq)):
            mutations.append({'pos': i, 'wt_aa': '-', 'mut_aa': seq[i]})
    elif len(seq) < len(wt):
        for i in range(len(seq), len(wt)):
            mutations.append({'pos': i, 'wt_aa': wt[i], 'mut_aa': '-'})
    return mutations


def load_flab_dataset(
    name: str,
    data_dir: Optional[str] = None,
    max_samples: int = 5000,
) -> Dict:
    """Load a FLAb dataset by preset name or custom path.

    Args:
        name: Preset name (e.g. 'flab_expression_koenig') or path within FLAb
              (e.g. 'binding/hie2023efficient_CoV2_S309_Kd.csv').
        data_dir: Local directory with pre-downloaded FLAb data (optional).
        max_samples: Subsample large datasets to this size (stratified).

    Returns:
        dict with keys:
            heavy_seqs: List[str] - heavy chain sequences
            light_seqs: Optional[List[str]] - light chain sequences (or None)
            labels: np.ndarray - continuous fitness values
            task_type: str - always 'regression'
            wildtype_heavy: str - wildtype heavy chain
            wildtype_light: Optional[str] - wildtype light chain
            mutations: List[List[Dict]] - per-sample mutation lists
            is_intra_family: bool - whether this is an intra-family dataset
            property_name: str - what the fitness measures
    """
    # Resolve preset name
    if name in FLAB_DATASETS:
        info = FLAB_DATASETS[name]
        relative_path = info['path']
        is_intra_family = info['intra_family']
        property_name = info['property']
    else:
        # Treat as a custom FLAb path
        relative_path = name
        is_intra_family = True  # assume intra-family by default
        property_name = 'Unknown'

    # Load CSV
    if data_dir:
        local_path = Path(data_dir) / relative_path
        if local_path.exists():
            df = pd.read_csv(local_path)
        else:
            df = _download_flab_csv(relative_path)
    else:
        df = _download_flab_csv(relative_path)

    # Parse columns — FLAb format: heavy, light, <assay>, fitness
    if 'heavy' not in df.columns:
        raise ValueError(f"Expected 'heavy' column. Found: {list(df.columns)}")
    if 'fitness' not in df.columns:
        raise ValueError(f"Expected 'fitness' column. Found: {list(df.columns)}")

    # Filter valid entries
    valid = df['heavy'].notna() & df['fitness'].notna()
    valid &= df['heavy'].str.len() > 10
    df = df[valid].copy()

    # Parse fitness as numeric
    df['fitness'] = pd.to_numeric(df['fitness'], errors='coerce')
    df = df.dropna(subset=['fitness'])

    # Heavy sequences
    heavy_seqs = df['heavy'].tolist()

    # Light sequences (optional — VHH/nanobody datasets may lack them)
    light_seqs = None
    if 'light' in df.columns:
        light_valid = df['light'].notna() & (df['light'].str.len() > 10)
        if light_valid.sum() > len(heavy_seqs) * 0.5:
            light_seqs = df['light'].tolist()
            # Replace missing/short with empty
            light_seqs = [s if isinstance(s, str) and len(s) > 10 else '' for s in light_seqs]

    labels = df['fitness'].values.astype(np.float64)

    print(f"  FLAb {name}: {len(heavy_seqs)} samples, "
          f"{'paired' if light_seqs else 'heavy-only'}, "
          f"property={property_name}")
    print(f"  Fitness range: [{labels.min():.3f}, {labels.max():.3f}], "
          f"mean={labels.mean():.3f}, std={labels.std():.3f}")

    # Subsample if too large (stratified by fitness quantiles)
    if len(heavy_seqs) > max_samples:
        print(f"  Subsampling {len(heavy_seqs)} → {max_samples}...")
        rng = np.random.RandomState(42)
        # Stratified: bin fitness into 10 quantiles, sample proportionally
        n_bins = 10
        bins = pd.qcut(labels, q=n_bins, labels=False, duplicates='drop')
        indices = []
        per_bin = max_samples // n_bins
        for b in range(n_bins):
            bin_idx = np.where(bins == b)[0]
            n_take = min(per_bin, len(bin_idx))
            indices.extend(rng.choice(bin_idx, n_take, replace=False).tolist())
        # Fill remainder
        remaining = max_samples - len(indices)
        if remaining > 0:
            all_idx = set(range(len(heavy_seqs)))
            available = list(all_idx - set(indices))
            indices.extend(rng.choice(available, min(remaining, len(available)),
                                      replace=False).tolist())
        indices = sorted(indices[:max_samples])

        heavy_seqs = [heavy_seqs[i] for i in indices]
        if light_seqs:
            light_seqs = [light_seqs[i] for i in indices]
        labels = labels[indices]
        print(f"  After subsampling: {len(heavy_seqs)} samples")

    # Identify wildtype and extract mutations
    wt_heavy, wt_light = _identify_wildtype(heavy_seqs, light_seqs)

    mutations = []
    for i, seq in enumerate(heavy_seqs):
        muts = _extract_mutations(seq, wt_heavy)
        if light_seqs and wt_light:
            light_muts = _extract_mutations(light_seqs[i], wt_light)
            # Tag light chain mutations
            for m in light_muts:
                m['chain'] = 'light'
            for m in muts:
                m['chain'] = 'heavy'
            muts.extend(light_muts)
        else:
            for m in muts:
                m['chain'] = 'heavy'
        mutations.append(muts)

    n_mutated = sum(1 for m in mutations if len(m) > 0)
    avg_muts = np.mean([len(m) for m in mutations])
    print(f"  Wildtype heavy: {wt_heavy[:30]}...")
    print(f"  {n_mutated}/{len(heavy_seqs)} samples have mutations, "
          f"avg {avg_muts:.1f} mutations/sample")

    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'task_type': 'regression',
        'wildtype_heavy': wt_heavy,
        'wildtype_light': wt_light,
        'mutations': mutations,
        'is_intra_family': is_intra_family,
        'property_name': property_name,
    }
