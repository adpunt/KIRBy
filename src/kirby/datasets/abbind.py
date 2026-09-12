"""
AB-Bind Dataset Loader — Antibody-antigen binding affinity changes (ΔΔG).

GitHub: https://github.com/sarahsirin/AB-Bind-Database

~1,101 mutations across 32 PDB complexes. After filtering non-Ab-Ag entries
(1AK4, 1FFW, 1JTG, 1KTZ, 1T83, HM_1KTZ), ~754 Ab-Ag entries remain.

Mutation format: 'H:W98A' → chain H, Trp98→Ala.
Multi-mutations: 'H:D32A,L:K97A' (comma-separated).

Reference:
- Sirin et al. (2016) "AB-Bind: Antibody binding mutational database
  for computational affinity predictions"
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kirby.datasets.pdb_utils import (
    download_pdb, extract_chain_sequence, apply_mutation, build_pdb_resnum_mapping,
)
from kirby.paths import data_dir


# PDB IDs that are NOT antibody-antigen complexes
_NON_ABAG_PDBS = {'1AK4', '1FFW', '1JTG', '1KTZ', '1T83', 'HM_1KTZ'}

_GITHUB_RAW = "https://raw.githubusercontent.com/sarahsirin/AB-Bind-Database/master"

# Regex for mutation format: Chain:WtResNumMut (e.g., H:W98A)
_MUT_RE = re.compile(r'([A-Za-z]):([A-Z])(\d+)([A-Z])')


def _cache_dir() -> Path:
    """Get cache directory."""
    return data_dir('abbind')


def _download_abbind_data() -> pd.DataFrame:
    """Download AB-Bind CSV (or load from cache)."""
    import requests

    cache = _cache_dir()
    cached = cache / 'AB-Bind_experimental_data.csv'

    if cached.exists():
        return pd.read_csv(cached, encoding='latin-1')

    url = f"{_GITHUB_RAW}/AB-Bind_experimental_data.csv"
    print(f"  Downloading AB-Bind data...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    return pd.read_csv(cached, encoding='latin-1')


def _parse_mutations(mutation_str: str) -> List[Dict]:
    """Parse AB-Bind mutation string into structured list.

    Args:
        mutation_str: e.g. 'H:W98A' or 'H:D32A,C:K97A'

    Returns:
        List of dicts: [{'chain': 'H', 'wt_aa': 'W', 'resnum': 98, 'mut_aa': 'A'}, ...]
    """
    mutations = []
    for match in _MUT_RE.finditer(str(mutation_str)):
        chain, wt_aa, resnum, mut_aa = match.groups()
        mutations.append({
            'chain': chain.upper(),
            'wt_aa': wt_aa,
            'resnum': int(resnum),
            'mut_aa': mut_aa,
        })
    return mutations


def _identify_ab_chains(partners_str: str) -> Tuple[set, set]:
    """Identify antibody vs antigen chains from Partners(A_B) column.

    Convention: chains named H and L are antibody. Other chains are antigen.
    The Partners column format is e.g. 'A_HL' or 'HL_VW'.

    Returns:
        (ab_chains, ag_chains) as sets of single-letter chain IDs.
    """
    parts = str(partners_str).split('_')
    all_chains = set()
    for p in parts:
        all_chains.update(p)

    ab_chains = set()
    ag_chains = set()
    for c in all_chains:
        if c in ('H', 'L'):
            ab_chains.add(c)
        else:
            ag_chains.add(c)

    # If no H/L chains found, treat first group as one side and second as other
    if not ab_chains and len(parts) == 2:
        ab_chains = set(parts[0])
        ag_chains = set(parts[1])

    return ab_chains, ag_chains


# Backward-compatible alias (now lives in pdb_utils)
_pdb_resnum_to_seqpos = build_pdb_resnum_mapping


def load_abbind_dataset(
    max_samples: int = 5000,
    antibody_mutations_only: bool = True,
    single_point_only: bool = False,
) -> Dict:
    """Load AB-Bind dataset.

    Args:
        max_samples: Subsample large datasets to this size.
        antibody_mutations_only: If True, only include entries where at least
            one mutation is on an antibody chain (H or L). Default True.
        single_point_only: If True, only include entries with exactly one
            mutation (before antibody filtering). Default False.

    Returns:
        dict with keys: heavy_seqs, light_seqs, labels, task_type,
            wildtype_heavy, wildtype_light, mutations, is_intra_family,
            property_name.
    """
    df = _download_abbind_data()

    # Filter out non-Ab-Ag complexes
    df = df[~df['#PDB'].isin(_NON_ABAG_PDBS)].copy()
    print(f"  AB-Bind: {len(df)} entries after filtering non-Ab-Ag PDBs")

    # Group by PDB for efficient processing
    all_heavy_seqs = []
    all_light_seqs = []
    all_labels = []
    all_mutations = []
    n_skipped = 0

    # Cache PDB data per complex
    pdb_cache = {}  # pdb_id → {'wt_heavy': str, 'wt_light': str, 'resnum_maps': {}}

    for _, row in df.iterrows():
        pdb_id = str(row['#PDB']).strip()
        mutation_str = str(row['Mutation']).strip()
        ddg = float(row['ddG(kcal/mol)'])

        # Parse mutations
        parsed_muts = _parse_mutations(mutation_str)
        if not parsed_muts:
            n_skipped += 1
            continue

        # Filter to single-point mutations only (before antibody filtering)
        if single_point_only and len(parsed_muts) != 1:
            n_skipped += 1
            continue

        # Identify antibody chains
        ab_chains, ag_chains = _identify_ab_chains(row.get('Partners(A_B)', ''))

        # Filter to antibody-side mutations only
        if antibody_mutations_only:
            ab_muts = [m for m in parsed_muts if m['chain'] in ab_chains]
            if not ab_muts:
                n_skipped += 1
                continue
            parsed_muts = ab_muts

        # Get or cache PDB wildtype data
        if pdb_id not in pdb_cache:
            # Skip homology model PDBs (HM_ prefix) — can't download from RCSB
            if pdb_id.startswith('HM_'):
                pdb_cache[pdb_id] = None
                n_skipped += 1
                continue

            try:
                pdb_path = download_pdb(pdb_id)

                wt_heavy = ''
                wt_light = ''
                resnum_maps = {}

                if 'H' in ab_chains:
                    try:
                        wt_heavy = extract_chain_sequence(pdb_path, 'H')
                        resnum_maps['H'] = _pdb_resnum_to_seqpos(pdb_path, 'H')
                    except (ValueError, KeyError):
                        pass
                if 'L' in ab_chains:
                    try:
                        wt_light = extract_chain_sequence(pdb_path, 'L')
                        resnum_maps['L'] = _pdb_resnum_to_seqpos(pdb_path, 'L')
                    except (ValueError, KeyError):
                        pass

                if not wt_heavy and not wt_light:
                    pdb_cache[pdb_id] = None
                else:
                    pdb_cache[pdb_id] = {
                        'wt_heavy': wt_heavy,
                        'wt_light': wt_light,
                        'resnum_maps': resnum_maps,
                        'pdb_path': pdb_path,
                    }
            except Exception as e:
                print(f"    WARNING: Failed to process PDB {pdb_id}: {e}")
                pdb_cache[pdb_id] = None

        cached = pdb_cache.get(pdb_id)
        if cached is None:
            n_skipped += 1
            continue

        wt_heavy = cached['wt_heavy']
        wt_light = cached['wt_light']
        resnum_maps = cached['resnum_maps']

        # Apply mutations to construct mutant sequences
        mut_heavy = wt_heavy
        mut_light = wt_light
        mutation_records = []
        valid = True

        for m in parsed_muts:
            chain = m['chain']
            resnum = m['resnum']
            wt_aa = m['wt_aa']
            mut_aa = m['mut_aa']

            if chain not in resnum_maps:
                valid = False
                break

            resnum_map = resnum_maps[chain]
            if resnum not in resnum_map:
                valid = False
                break

            seq_pos = resnum_map[resnum]

            # Get the appropriate sequence
            if chain == 'H':
                if seq_pos >= len(mut_heavy) or mut_heavy[seq_pos] != wt_aa:
                    valid = False
                    break
                try:
                    mut_heavy = apply_mutation(mut_heavy, seq_pos, mut_aa)
                except (ValueError, IndexError):
                    valid = False
                    break
            elif chain == 'L':
                if seq_pos >= len(mut_light) or mut_light[seq_pos] != wt_aa:
                    valid = False
                    break
                try:
                    mut_light = apply_mutation(mut_light, seq_pos, mut_aa)
                except (ValueError, IndexError):
                    valid = False
                    break
            else:
                valid = False
                break

            mutation_records.append({
                'pos': seq_pos,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'chain': 'heavy' if chain == 'H' else 'light',
                'pdb': pdb_id,
            })

        if not valid or not mutation_records:
            n_skipped += 1
            continue

        all_heavy_seqs.append(mut_heavy)
        all_light_seqs.append(mut_light)
        all_labels.append(ddg)
        all_mutations.append(mutation_records)

    if not all_heavy_seqs:
        raise ValueError("No valid entries found in AB-Bind")

    labels = np.array(all_labels, dtype=np.float64)

    print(f"  Processed: {len(all_heavy_seqs)} valid entries, {n_skipped} skipped")
    print(f"  ΔΔG range: [{labels.min():.3f}, {labels.max():.3f}] kcal/mol, "
          f"mean={labels.mean():.3f}")

    # Check if we have light chains
    has_light = any(len(s) > 0 for s in all_light_seqs)
    light_seqs = all_light_seqs if has_light else None

    # Subsample if needed
    if len(all_heavy_seqs) > max_samples:
        print(f"  Subsampling {len(all_heavy_seqs)} → {max_samples}...")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_heavy_seqs), max_samples, replace=False)
        idx = sorted(idx)
        all_heavy_seqs = [all_heavy_seqs[i] for i in idx]
        if light_seqs:
            light_seqs = [light_seqs[i] for i in idx]
        labels = labels[idx]
        all_mutations = [all_mutations[i] for i in idx]

    # Wildtype: use the most common heavy sequence (mode across PDBs)
    from collections import Counter
    unique_pdb_wts = {}
    for entry in pdb_cache.values():
        if entry and entry['wt_heavy']:
            unique_pdb_wts[entry['wt_heavy']] = unique_pdb_wts.get(entry['wt_heavy'], 0) + 1
    wt_heavy = max(unique_pdb_wts, key=unique_pdb_wts.get) if unique_pdb_wts else None

    return {
        'heavy_seqs': all_heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'task_type': 'regression',
        'wildtype_heavy': wt_heavy,
        'wildtype_light': None,  # multi-PDB, no single wildtype
        'mutations': all_mutations,
        'is_intra_family': False,  # multi-PDB dataset
        'property_name': 'ΔΔG (kcal/mol)',
    }
