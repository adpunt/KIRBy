"""
Graphinity Synthetic DDG Dataset Loader

Downloads synthetic ΔΔG data from the Graphinity project (Hummer et al. 2025).
Single-point mutations at antibody-antigen interfaces from SAbDab structures,
scored by FoldX (~943K) or Rosetta Flex ddG (~21K).

Three presets:
  - foldx:     FoldX ΔΔG values (subsampled to max_samples)
  - rosetta:   Rosetta Flex ddG values
  - both:      Combined FoldX + Rosetta (subsampled)

Plus an augmented preset that combines synthetic + experimental DDG:
  - augmented: Experimental DDG (ddg_combined) + Graphinity synthetic

Sequences are extracted from PDB files using pdb_utils.
Groups = PDB complex ID (for leave-one-complex-out GroupKFold).

Reference:
  Hummer et al., "Investigating the Volume and Diversity of Data Needed for
  Generalizable Antibody-Antigen ΔΔG Prediction" (2025, Nature Comp. Sci.)
  Zenodo: https://doi.org/10.5281/zenodo.15384945
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ZENODO_BASE = "https://zenodo.org/records/15384945/files"
_FOLDX_CSV = "Synthetic_FoldX_ddG_942723.csv"
_FLEXDDG_CSV = "Synthetic_FlexddG_ddG_20829.csv"


def _cache_dir() -> Path:
    """Cache dir, preferring /Volumes/seagate if available."""
    seagate = Path('/Volumes/seagate/kirby/graphinity')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    cache = cache_root / 'graphinity'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_csv(filename: str) -> pd.DataFrame:
    """Download or load cached Graphinity CSV from Zenodo."""
    import requests

    cache = _cache_dir()
    cached = cache / filename

    if cached.exists():
        return pd.read_csv(cached)

    url = f"{_ZENODO_BASE}/{filename}"
    print(f"  Downloading {filename} from Zenodo...")
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()
    with open(cached, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Downloaded: {cached}")
    return pd.read_csv(cached)


def _parse_complex_column(complex_str: str) -> Dict:
    """Parse the complex column mutation descriptor.

    Formats observed in Graphinity CSVs:
      - '{pdb}_{wt_aa}{chain}{resi}{mut_aa}' (e.g. '6q18_SH30A')
      - '{pdb}_{ab_chain}_{ag_chain}_{wt_aa}{chain}{resi}{mut_aa}'
        (e.g. '1bj1_HL_W_YH102A')

    The mutation is always in the LAST underscore-delimited segment:
      {wt_aa}{chain}{residue_number}{mut_aa}

    Returns dict with pdb, wt_aa, chain, position, mut_aa.
    """
    parts = complex_str.split('_')
    pdb = parts[0]
    mut_str = parts[-1]  # Last segment is always the mutation

    # Parse mutation string: {wt_aa}{chain}{resi}{mut_aa}
    # Chain may be multi-character (e.g. 'AA') in some Graphinity entries —
    # P1-4 fix widens the regex. Mutation rows that don't match this shape
    # still fail loudly via the ValueError below.
    m = re.match(r'^([A-Z])([A-Za-z0-9]+)(\d+)([A-Z])$', mut_str)
    if m:
        return {
            'pdb': pdb,
            'wt_aa': m.group(1),
            'chain': m.group(2),
            'position': int(m.group(3)),
            'mut_aa': m.group(4),
        }
    raise ValueError(f"Cannot parse complex column: {complex_str}")


def _classify_chain_anarci(sequence: str) -> Optional[str]:
    """Return 'H', 'L', or None for an antibody V-domain.

    K and L (kappa/lambda) collapse to 'L' (light). Returns None when ANARCI
    can't number the sequence (non-antibody fragment, too short, etc.).
    """
    try:
        from anarci import anarci as run_anarci
    except ImportError:
        return None
    try:
        results = run_anarci([('q', sequence)], scheme='imgt')
    except Exception:
        return None
    details = results[1]
    if not details or not details[0]:
        return None
    chain_type = details[0][0].get('chain_type')
    if chain_type == 'H':
        return 'H'
    if chain_type in ('K', 'L'):
        return 'L'
    return None


def _type_ab_chains(seqs: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Classify a pair of chain sequences as (heavy_seq, light_seq) via ANARCI.

    Returns None when ANARCI can't unambiguously call exactly one H and one L
    (caller falls back to length heuristic).
    """
    types = {cid: _classify_chain_anarci(seq) for cid, seq in seqs.items()}
    h_chains = [cid for cid, t in types.items() if t == 'H']
    l_chains = [cid for cid, t in types.items() if t == 'L']
    if len(h_chains) == 1 and len(l_chains) == 1:
        return seqs[h_chains[0]], seqs[l_chains[0]]
    return None


def _extract_sequences_cached(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract VH/VL sequences for each unique PDB+ab_chain+ag_chain combo.

    Caches the result as a CSV to avoid re-downloading PDBs.
    """
    from kirby.datasets.pdb_utils import download_pdb, extract_chain_sequence

    cache = _cache_dir()
    seq_cache = cache / 'pdb_sequences.csv'

    if seq_cache.exists():
        cached_df = pd.read_csv(seq_cache).fillna('')
        cached_keys = set(zip(cached_df['pdb'], cached_df['ab_chain']))
    else:
        cached_df = pd.DataFrame(columns=['pdb', 'ab_chain', 'heavy_seq', 'light_seq'])
        cached_keys = set()

    # Find unique PDB+chain combos we need
    unique_combos = df[['pdb', 'ab_chain']].drop_duplicates()
    needed = []
    for _, row in unique_combos.iterrows():
        key = (row['pdb'], row['ab_chain'])
        if key not in cached_keys:
            needed.append(key)

    if needed:
        print(f"  Extracting sequences from {len(needed)} PDB structures...")
        new_rows = []
        for i, (pdb_id, ab_chains) in enumerate(needed):
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(needed)} PDBs processed...")
            try:
                pdb_path = download_pdb(pdb_id)
                chain_ids = list(ab_chains) if len(ab_chains) > 1 else [ab_chains]

                # Identify heavy and light chains
                heavy_seq, light_seq = '', ''
                seqs = {}
                for cid in chain_ids:
                    try:
                        seq = extract_chain_sequence(pdb_path, cid)
                        seqs[cid] = seq
                    except Exception:
                        continue

                if len(seqs) == 2:
                    # Type each chain via ANARCI (IMGT V-domain signature).
                    # Falls back to length heuristic if ANARCI unavailable or
                    # both chains fail to number (P1-5 fix).
                    typed = _type_ab_chains(seqs)
                    if typed is not None:
                        heavy_seq, light_seq = typed
                    else:
                        sorted_chains = sorted(seqs.items(),
                                               key=lambda x: len(x[1]),
                                               reverse=True)
                        heavy_seq = sorted_chains[0][1]
                        light_seq = sorted_chains[1][1]
                elif len(seqs) == 1:
                    heavy_seq = list(seqs.values())[0]
                    light_seq = ''

                new_rows.append({
                    'pdb': pdb_id,
                    'ab_chain': ab_chains,
                    'heavy_seq': heavy_seq,
                    'light_seq': light_seq,
                })
            except Exception as e:
                print(f"    WARNING: Failed to process PDB {pdb_id}: {e}")
                new_rows.append({
                    'pdb': pdb_id,
                    'ab_chain': ab_chains,
                    'heavy_seq': '',
                    'light_seq': '',
                })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            cached_df = pd.concat([cached_df, new_df], ignore_index=True)
            cached_df.to_csv(seq_cache, index=False)
            print(f"  Cached sequences for {len(cached_df)} PDB+chain combos")

    return cached_df


def load_graphinity_dataset(
    preset: str = 'foldx',
    max_samples: int = 50000,
) -> Dict:
    """Load Graphinity synthetic DDG dataset.

    Args:
        preset: 'foldx' (FoldX DDG), 'rosetta' (Flex ddG), or 'both'.
        max_samples: Subsample to this size (stratified by PDB).

    Returns:
        dict with: heavy_seqs, light_seqs, labels (float DDG), pdb_ids,
                   task_type='regression', mutations, property_name
    """
    # DDG columns: 'foldx_ddg' in FoldX CSV, 'flexddg_ddg' in FlexddG CSV.
    if preset == 'foldx':
        df = _download_csv(_FOLDX_CSV)
        ddg_col = 'foldx_ddg'
        label = 'FoldX'
    elif preset == 'rosetta':
        df = _download_csv(_FLEXDDG_CSV)
        ddg_col = 'flexddg_ddg'
        label = 'Rosetta Flex ddG'
    elif preset == 'both':
        df_foldx = _download_csv(_FOLDX_CSV)
        df_foldx['_ddg'] = df_foldx['foldx_ddg']
        df_foldx['_source'] = 'foldx'
        df_rosetta = _download_csv(_FLEXDDG_CSV)
        df_rosetta['_ddg'] = df_rosetta['flexddg_ddg']
        df_rosetta['_source'] = 'rosetta'
        # Normalize ab_chain column name
        for d in [df_foldx, df_rosetta]:
            if 'ab_chain' not in d.columns and 'chain_prot1' in d.columns:
                d['ab_chain'] = d['chain_prot1']
            if 'ag_chain' not in d.columns and 'chain_prot2' in d.columns:
                d['ag_chain'] = d['chain_prot2']
        common_cols = ['pdb', 'complex', 'ab_chain', 'ag_chain', '_ddg', '_source']
        df = pd.concat([df_foldx[common_cols], df_rosetta[common_cols]],
                       ignore_index=True)
        ddg_col = '_ddg'
        label = 'FoldX + Rosetta'
    else:
        raise ValueError(f"Unknown preset: {preset}. Use 'foldx', 'rosetta', or 'both'.")

    # Normalize column names (FoldX uses chain_prot1/chain_prot2)
    if 'ab_chain' not in df.columns and 'chain_prot1' in df.columns:
        df['ab_chain'] = df['chain_prot1']
    if 'ag_chain' not in df.columns and 'chain_prot2' in df.columns:
        df['ag_chain'] = df['chain_prot2']

    # Drop rows with missing DDG or PDB
    df = df[df[ddg_col].notna() & df['pdb'].notna() & df['complex'].notna()].copy()
    print(f"  Graphinity ({label}): {len(df)} total mutations")

    # Subsample if needed — stratified by PDB to preserve complex diversity
    if max_samples is not None and len(df) > max_samples:
        rng = np.random.RandomState(42)
        # Sample proportionally from each PDB
        sampled = df.groupby('pdb', group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, int(len(g) / len(df) * max_samples)),
                random_state=rng,
            )
        )
        if len(sampled) > max_samples:
            sampled = sampled.sample(n=max_samples, random_state=rng)
        df = sampled.reset_index(drop=True)
        print(f"  Subsampled to {len(df)} mutations across "
              f"{df['pdb'].nunique()} PDB complexes")

    # Parse mutations from complex column
    mutations_info = []
    for _, row in df.iterrows():
        try:
            info = _parse_complex_column(row['complex'])
            mutations_info.append(info)
        except ValueError:
            mutations_info.append(None)

    # Filter out unparseable rows
    valid = [i for i, m in enumerate(mutations_info) if m is not None]
    if len(valid) < len(df):
        print(f"  WARNING: {len(df) - len(valid)} rows with unparseable complex column")
        df = df.iloc[valid].reset_index(drop=True)
        mutations_info = [mutations_info[i] for i in valid]

    # Extract sequences from PDB files
    seq_df = _extract_sequences_cached(df)
    seq_lookup = {(r['pdb'], r['ab_chain']): (r['heavy_seq'], r['light_seq'])
                  for _, r in seq_df.iterrows()}

    # Build output arrays
    heavy_seqs = []
    light_seqs = []
    labels_list = []
    pdb_ids = []
    mutations = []
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows()):
        key = (row['pdb'], row['ab_chain'])
        seqs = seq_lookup.get(key)
        if seqs is None or not seqs[0]:
            skipped += 1
            continue

        heavy_seq, light_seq = seqs
        info = mutations_info[i]

        heavy_seqs.append(heavy_seq if isinstance(heavy_seq, str) else '')
        light_seqs.append(light_seq if isinstance(light_seq, str) else '')
        labels_list.append(row[ddg_col])
        pdb_ids.append(row['pdb'])
        mutations.append([{
            'pos': info['position'],
            'wt_aa': info['wt_aa'],
            'mut_aa': info['mut_aa'],
            'chain': info['chain'],
        }])

    if skipped:
        print(f"  Skipped {skipped} mutations (missing PDB sequences)")

    labels = np.array(labels_list, dtype=np.float64)
    n = len(heavy_seqs)
    print(f"  Graphinity ({label}): {n} samples, {len(set(pdb_ids))} complexes, "
          f"DDG range [{labels.min():.2f}, {labels.max():.2f}]")

    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'task_type': 'regression',
        'pdb_ids': pdb_ids,
        'mutations': mutations,
        'is_intra_family': False,
        'property_name': f'ΔΔG (kcal/mol, {label} synthetic)',
    }


def load_ddg_augmented_dataset(
    synthetic_preset: str = 'foldx',
    max_synthetic: int = 50000,
) -> Dict:
    """Load experimental DDG (ddg_combined) + Graphinity synthetic, combined.

    Uses experimental DDG as the "core" and augments with synthetic data.
    Groups by PDB complex across both sources for leave-one-complex-out CV.
    """
    from kirby.datasets.ddg_combined import load_ddg_combined_dataset

    # Load experimental
    exp = load_ddg_combined_dataset(ab_ag_only=True)
    n_exp = len(exp['heavy_seqs'])
    print(f"  Experimental DDG: {n_exp} samples")

    # Load synthetic
    syn = load_graphinity_dataset(preset=synthetic_preset, max_samples=max_synthetic)
    n_syn = len(syn['heavy_seqs'])
    print(f"  Synthetic DDG ({synthetic_preset}): {n_syn} samples")

    # Combine
    heavy_seqs = exp['heavy_seqs'] + syn['heavy_seqs']
    light_seqs = exp['light_seqs'] + syn['light_seqs']
    labels = np.concatenate([exp['labels'], syn['labels']])
    pdb_ids = exp.get('pdb_ids', ['exp'] * n_exp) + syn['pdb_ids']
    mutations = (exp.get('mutations', [[] for _ in range(n_exp)])
                 + syn['mutations'])

    n = len(heavy_seqs)
    print(f"  Augmented DDG: {n} total ({n_exp} experimental + {n_syn} synthetic), "
          f"{len(set(pdb_ids))} complexes")

    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'task_type': 'regression',
        'pdb_ids': pdb_ids,
        'mutations': mutations,
        'is_intra_family': False,
        'property_name': f'ΔΔG (kcal/mol, experimental + {synthetic_preset} synthetic)',
    }
