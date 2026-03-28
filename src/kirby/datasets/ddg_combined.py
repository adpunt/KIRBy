"""
Combined ΔΔG Dataset — SKEMPI 2.0 + AB-Bind, deduplicated.

Merges two public antibody-antigen ΔΔG databases:
  - SKEMPI 2.0: ~5,000 entries across diverse protein-protein interfaces
    (filtered to Ab-Ag only by default; set ab_ag_only=False for full PP data)
  - AB-Bind: ~754 entries across 32 Ab-Ag complexes (direct DDG measurements)

Deduplication uses (PDB_ID, sorted normalized mutations) as key, preferring
AB-Bind's direct DDG measurements over SKEMPI's Kd-derived values on conflict.

Full SKEMPI (ab_ag_only=False):
  - Includes ALL protein-protein interfaces (~7085 entries before filtering)
  - Barnase-barstar, T4 lysozyme, thrombin inhibitors, enzymes, etc.
  - ~6x more data than Ab-Ag filtered subset
  - Protein sequences returned as heavy_seqs (chain 1) and light_seqs (chain 2)
  - Intended as an EXPERIMENT: does training on more diverse PP data help?
  - Non-antibody sequences will not benefit from antibody-specific LM reps.

AbDesign DDG (load_abdesign_ddg):
  - Stub loader — AbDesign CDR-H3 benchmark data is not in a canonical public
    repository. Raises AbDesignDataNotAvailableError with instructions.

References:
- Jankauskaitė et al. (2019) "SKEMPI 2.0: an updated benchmark of changes in
  protein–protein binding energy, kinetics and thermodynamics upon mutation"
- Sirin et al. (2016) "AB-Bind: Antibody binding mutational database for
  computational affinity predictions"
- Goldenzweig et al. (2016) "Automated Structure-Based Design of a New ABDesign
  Library for CDR-H3 Engineering"
"""

import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from kirby.datasets.pdb_utils import (
    download_pdb, extract_chain_sequence, apply_mutation, build_pdb_resnum_mapping,
)

# Gas constant in kcal/(mol·K)
_R_KCAL = 1.9872036e-3

# SKEMPI mutation regex: WtAA ChainID ResNum MutAA (e.g., YH27A)
_SKEMPI_MUT_RE = re.compile(r'([A-Z])([A-Za-z])(\d+)([A-Z])')

# AB-Bind non-Ab-Ag PDBs (imported from abbind module logic)
_NON_ABAG_PDBS = {'1AK4', '1FFW', '1JTG', '1KTZ', '1T83', 'HM_1KTZ'}

# Keywords suggesting antibody in SKEMPI protein names
_AB_KEYWORDS = {
    'antibody', 'fab', 'scfv', 'igg', 'nanobody', 'vh', 'vhh',
    'immunoglobulin', 'ig ', 'mab', 'anti-',
}

# Keywords suggesting antigen
_AG_KEYWORDS = {
    'antigen', 'lysozyme', 'hen egg', 'hewl', 'vegf', 'egfr', 'her2',
    'rnase', 'barnase', 'barstar', 'cytochrome', 'insulin', 'interferon',
    'interleukin', 'receptor', 'growth factor', 'kinase', 'protease',
}


def _cache_dir() -> Path:
    """Get cache directory for combined DDG data."""
    seagate = Path('/Volumes/seagate/kirby/ddg_combined')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache = Path.home() / '.cache' / 'kirby' / 'ddg_combined'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_skempi() -> pd.DataFrame:
    """Download or load cached SKEMPI 2.0 data."""
    import requests

    cache = _cache_dir()
    cached = cache / 'skempi_v2.csv'

    if cached.exists():
        return pd.read_csv(cached, sep=';')

    url = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
    print(f"  Downloading SKEMPI 2.0 from {url}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    return pd.read_csv(cached, sep=';')


def _is_abag_entry(row: pd.Series) -> bool:
    """Check if a SKEMPI entry is an antibody-antigen complex."""
    # Method 1: Hold_out_type column
    hot = str(row.get('Hold_out_type', '')).lower()
    if 'ab/ag' in hot or 'ab-ag' in hot:
        return True

    # Method 2: keyword matching on protein names
    prot1 = str(row.get('Protein 1', '')).lower()
    prot2 = str(row.get('Protein 2', '')).lower()
    all_text = prot1 + ' ' + prot2

    has_ab = any(kw in all_text for kw in _AB_KEYWORDS)
    has_ag = any(kw in all_text for kw in _AG_KEYWORDS)

    return has_ab and has_ag


def _identify_ab_chains_skempi(
    row: pd.Series, pdb_groups: Tuple[str, str],
) -> Tuple[Set[str], Set[str]]:
    """Identify which chain group is antibody vs antigen.

    Cascade:
      (a) keyword match on protein names
      (b) H/L chain convention
      (c) default to group1 as antibody
    """
    group1_chains = set(pdb_groups[0])
    group2_chains = set(pdb_groups[1])

    prot1 = str(row.get('Protein 1', '')).lower()
    prot2 = str(row.get('Protein 2', '')).lower()

    # (a) Keyword matching
    p1_ab = any(kw in prot1 for kw in _AB_KEYWORDS)
    p2_ab = any(kw in prot2 for kw in _AB_KEYWORDS)

    if p1_ab and not p2_ab:
        return group1_chains, group2_chains
    if p2_ab and not p1_ab:
        return group2_chains, group1_chains

    # (b) H/L chain convention
    all_chains = group1_chains | group2_chains
    if 'H' in all_chains or 'L' in all_chains:
        hl_chains = {c for c in all_chains if c in ('H', 'L')}
        other_chains = all_chains - hl_chains
        return hl_chains, other_chains

    # (c) Default: group1 is antibody
    return group1_chains, group2_chains


def _assign_hl_roles(
    ab_chains: Set[str], pdb_path: Path,
) -> Tuple[Optional[str], Optional[str]]:
    """Map antibody chain IDs to heavy/light roles.

    Uses chain naming convention first (H/L), then sequence length heuristic.

    Returns:
        (heavy_chain_id, light_chain_id) — either may be None.
    """
    if not ab_chains:
        return None, None

    # Direct naming convention
    if 'H' in ab_chains:
        h_chain = 'H'
        l_chain = 'L' if 'L' in ab_chains else None
        return h_chain, l_chain

    # Length heuristic: VH (~120 aa) is typically longer than VL (~107 aa)
    chain_lens = {}
    for c in ab_chains:
        try:
            seq = extract_chain_sequence(pdb_path, c)
            chain_lens[c] = len(seq)
        except (ValueError, KeyError):
            chain_lens[c] = 0

    if not chain_lens:
        return None, None

    sorted_chains = sorted(chain_lens.items(), key=lambda x: -x[1])
    h_chain = sorted_chains[0][0]
    l_chain = sorted_chains[1][0] if len(sorted_chains) > 1 else None

    return h_chain, l_chain


def _parse_skempi_mutations(mutation_str: str) -> List[Dict]:
    """Parse SKEMPI mutation string.

    Format: WtAA ChainID ResNum MutAA, comma-separated for multi-point.
    Example: 'YH27A' or 'YH27A,DL32G'
    """
    mutations = []
    for match in _SKEMPI_MUT_RE.finditer(str(mutation_str)):
        wt_aa, chain, resnum, mut_aa = match.groups()
        mutations.append({
            'chain': chain.upper(),
            'wt_aa': wt_aa,
            'resnum': int(resnum),
            'mut_aa': mut_aa,
        })
    return mutations


def _compute_ddg(kd_mut: float, kd_wt: float, temperature: float) -> Optional[float]:
    """Compute ΔΔG = RT * ln(Kd_mut / Kd_wt) in kcal/mol."""
    if kd_mut <= 0 or kd_wt <= 0 or temperature <= 0:
        return None
    try:
        ddg = _R_KCAL * temperature * math.log(kd_mut / kd_wt)
        return ddg
    except (ValueError, OverflowError):
        return None


def _normalize_mutation_key(
    pdb_id: str, mutations: List[Dict], h_chain: Optional[str], l_chain: Optional[str],
) -> Tuple[str, ...]:
    """Create a normalized deduplication key from mutations.

    Maps raw chain IDs to 'heavy'/'light' roles so SKEMPI and AB-Bind entries
    with different chain naming but same PDB/mutations are recognized as duplicates.
    h_chain/l_chain can be single chain IDs ('H') or comma-separated ('A,B').
    """
    h_chains = set(h_chain.split(',')) if h_chain else set()
    l_chains = set(l_chain.split(',')) if l_chain else set()
    normalized = []
    for m in mutations:
        raw_chain = m['chain']
        if raw_chain in h_chains or raw_chain == 'H':
            role = 'heavy'
        elif raw_chain in l_chains or raw_chain == 'L':
            role = 'light'
        else:
            role = raw_chain  # non-antibody or unknown
        normalized.append(f"{role}:{m['wt_aa']}{m['resnum']}{m['mut_aa']}")

    return (pdb_id.upper(), *sorted(normalized))


def _process_skempi_entries(df: pd.DataFrame) -> List[Dict]:
    """Process SKEMPI 2.0 entries into standardized format.

    Returns list of dicts with keys:
        pdb_id, h_chain, l_chain, mutations, ddg, heavy_seq, light_seq,
        mutation_records, source, dedup_key
    """
    entries = []
    pdb_cache = {}  # pdb_id → {pdb_path, h_chain, l_chain, seqs, resnum_maps}
    n_skipped = 0

    for _, row in df.iterrows():
        # Parse #Pdb column: PDB_group1_group2
        pdb_field = str(row.get('#Pdb', '')).strip()
        parts = pdb_field.split('_')
        if len(parts) < 3:
            n_skipped += 1
            continue

        pdb_id = parts[0].upper()
        pdb_groups = ('_'.join(parts[1:-1]), parts[-1])  # handle multi-char groups
        # Simple case: single-char groups like "1A14_A_B"
        if len(parts) == 3:
            pdb_groups = (parts[1], parts[2])

        # Parse mutations
        mutation_str = str(row.get('Mutation(s)_cleaned', row.get('Mutation(s)_PDB', '')))
        parsed_muts = _parse_skempi_mutations(mutation_str)
        if not parsed_muts:
            n_skipped += 1
            continue

        # Compute DDG from Kd values
        ddg = None

        # Try pre-computed DDG first
        for col in ['DDG', 'ddG']:
            if col in row.index:
                try:
                    val = float(row[col])
                    if np.isfinite(val):
                        ddg = val
                        break
                except (ValueError, TypeError):
                    pass

        # Compute from Kd if no pre-computed DDG
        if ddg is None:
            try:
                kd_mut = float(row.get('Affinity_mut_parsed', np.nan))
                kd_wt = float(row.get('Affinity_wt_parsed', np.nan))
                temp_raw = row.get('Temperature', 298.15)
                temp = float(temp_raw) if pd.notna(temp_raw) else 298.15
                if np.isfinite(kd_mut) and np.isfinite(kd_wt):
                    ddg = _compute_ddg(kd_mut, kd_wt, temp)
            except (ValueError, TypeError):
                pass

        if ddg is None:
            n_skipped += 1
            continue

        # Get or cache PDB data
        if pdb_id not in pdb_cache:
            try:
                pdb_path = download_pdb(pdb_id)
                ab_chains, ag_chains = _identify_ab_chains_skempi(row, pdb_groups)
                h_chain, l_chain = _assign_hl_roles(ab_chains, pdb_path)

                seqs = {}
                resnum_maps = {}

                if h_chain:
                    try:
                        seqs['heavy'] = extract_chain_sequence(pdb_path, h_chain)
                        resnum_maps[h_chain] = build_pdb_resnum_mapping(pdb_path, h_chain)
                    except (ValueError, KeyError):
                        seqs['heavy'] = ''

                if l_chain:
                    try:
                        seqs['light'] = extract_chain_sequence(pdb_path, l_chain)
                        resnum_maps[l_chain] = build_pdb_resnum_mapping(pdb_path, l_chain)
                    except (ValueError, KeyError):
                        seqs['light'] = ''

                if not seqs.get('heavy') and not seqs.get('light'):
                    pdb_cache[pdb_id] = None
                else:
                    pdb_cache[pdb_id] = {
                        'pdb_path': pdb_path,
                        'h_chain': h_chain,
                        'l_chain': l_chain,
                        'seqs': seqs,
                        'resnum_maps': resnum_maps,
                    }
            except Exception as e:
                print(f"    WARNING: SKEMPI PDB {pdb_id}: {e}")
                pdb_cache[pdb_id] = None

        cached = pdb_cache.get(pdb_id)
        if cached is None:
            n_skipped += 1
            continue

        h_chain = cached['h_chain']
        l_chain = cached['l_chain']
        resnum_maps = cached['resnum_maps']
        wt_heavy = cached['seqs'].get('heavy', '')
        wt_light = cached['seqs'].get('light', '')

        # Filter to antibody-side mutations
        ab_chain_ids = set()
        if h_chain:
            ab_chain_ids.add(h_chain)
        if l_chain:
            ab_chain_ids.add(l_chain)

        ab_muts = [m for m in parsed_muts if m['chain'] in ab_chain_ids]
        if not ab_muts:
            n_skipped += 1
            continue

        # Apply mutations
        mut_heavy = wt_heavy
        mut_light = wt_light
        mutation_records = []
        valid = True

        for m in ab_muts:
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

            if chain == h_chain:
                if seq_pos >= len(mut_heavy) or mut_heavy[seq_pos] != wt_aa:
                    valid = False
                    break
                try:
                    mut_heavy = apply_mutation(mut_heavy, seq_pos, mut_aa)
                except (ValueError, IndexError):
                    valid = False
                    break
                role = 'heavy'
            elif chain == l_chain:
                if seq_pos >= len(mut_light) or mut_light[seq_pos] != wt_aa:
                    valid = False
                    break
                try:
                    mut_light = apply_mutation(mut_light, seq_pos, mut_aa)
                except (ValueError, IndexError):
                    valid = False
                    break
                role = 'light'
            else:
                valid = False
                break

            mutation_records.append({
                'pos': seq_pos,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'chain': role,
                'pdb': pdb_id,
            })

        if not valid or not mutation_records:
            n_skipped += 1
            continue

        dedup_key = _normalize_mutation_key(pdb_id, ab_muts, h_chain, l_chain)

        entries.append({
            'pdb_id': pdb_id,
            'h_chain': h_chain,
            'l_chain': l_chain,
            'ddg': ddg,
            'heavy_seq': mut_heavy,
            'light_seq': mut_light,
            'wt_heavy': wt_heavy,
            'wt_light': wt_light,
            'mutation_records': mutation_records,
            'source': 'skempi',
            'dedup_key': dedup_key,
        })

    print(f"  SKEMPI: {len(entries)} valid Ab-Ag entries, {n_skipped} skipped")
    return entries


def _process_skempi_all_entries(df: pd.DataFrame) -> List[Dict]:
    """Process ALL SKEMPI 2.0 entries (protein-protein, not just Ab-Ag).

    Unlike _process_skempi_entries(), this function does NOT require H/L chain
    assignment. It processes ANY protein-protein complex:
      - chain1_seq: sequence from the first chain group (returned as 'heavy_seq')
      - chain2_seq: sequence from the second chain group (returned as 'light_seq')
      - mutations are applied to whichever chain they fall on

    This is intended for the full-SKEMPI experiment where non-antibody PP
    interactions (barnase-barstar, T4-lysozyme, etc.) are included to increase
    training data diversity.

    Returns entries with the same key structure as _process_skempi_entries()
    so they can be directly used in load_ddg_combined_dataset().
    """
    entries = []
    pdb_cache = {}
    n_skipped = 0

    for _, row in df.iterrows():
        pdb_field = str(row.get('#Pdb', '')).strip()
        parts = pdb_field.split('_')
        if len(parts) < 3:
            n_skipped += 1
            continue

        pdb_id = parts[0].upper()
        if len(parts) == 3:
            group1_chains = set(parts[1])
            group2_chains = set(parts[2])
        else:
            # Multi-char groups: split last _ as group2, rest as group1
            group1_chains = set(''.join(parts[1:-1]))
            group2_chains = set(parts[-1])

        mutation_str = str(row.get('Mutation(s)_cleaned', row.get('Mutation(s)_PDB', '')))
        parsed_muts = _parse_skempi_mutations(mutation_str)
        if not parsed_muts:
            n_skipped += 1
            continue

        # Compute DDG
        ddg = None
        for col in ['DDG', 'ddG']:
            if col in row.index:
                try:
                    val = float(row[col])
                    if np.isfinite(val):
                        ddg = val
                        break
                except (ValueError, TypeError):
                    pass

        if ddg is None:
            try:
                kd_mut = float(row.get('Affinity_mut_parsed', np.nan))
                kd_wt = float(row.get('Affinity_wt_parsed', np.nan))
                temp_raw = row.get('Temperature', 298.15)
                temp = float(temp_raw) if pd.notna(temp_raw) else 298.15
                if np.isfinite(kd_mut) and np.isfinite(kd_wt):
                    ddg = _compute_ddg(kd_mut, kd_wt, temp)
            except (ValueError, TypeError):
                pass

        if ddg is None:
            n_skipped += 1
            continue

        # Determine which chains are mutated
        all_ab_chains = group1_chains | group2_chains
        mutated_chains = set(m['chain'] for m in parsed_muts)
        relevant_chains = mutated_chains & all_ab_chains
        if not relevant_chains:
            n_skipped += 1
            continue

        # Cache PDB data: extract sequences for both chain groups
        if pdb_id not in pdb_cache:
            try:
                pdb_path = download_pdb(pdb_id)

                seqs = {}
                resnum_maps = {}
                for chain_id in all_ab_chains:
                    try:
                        seqs[chain_id] = extract_chain_sequence(pdb_path, chain_id)
                        resnum_maps[chain_id] = build_pdb_resnum_mapping(pdb_path, chain_id)
                    except (ValueError, KeyError):
                        seqs[chain_id] = ''

                if not any(seqs.values()):
                    pdb_cache[pdb_id] = None
                else:
                    # Assign "heavy" = first chain group (concatenated), "light" = second
                    def _concat_seqs(chains):
                        return ''.join(seqs.get(c, '') for c in sorted(chains))

                    pdb_cache[pdb_id] = {
                        'pdb_path': pdb_path,
                        'seqs': seqs,
                        'resnum_maps': resnum_maps,
                        'group1_chains': group1_chains,
                        'group2_chains': group2_chains,
                        'wt_heavy': _concat_seqs(group1_chains),
                        'wt_light': _concat_seqs(group2_chains),
                    }
            except Exception as e:
                print(f"    WARNING: SKEMPI (full) PDB {pdb_id}: {e}")
                pdb_cache[pdb_id] = None

        cached = pdb_cache.get(pdb_id)
        if cached is None:
            n_skipped += 1
            continue

        seqs = cached['seqs']
        resnum_maps = cached['resnum_maps']
        wt_heavy = cached['wt_heavy']
        wt_light = cached['wt_light']
        g1_chains = cached['group1_chains']
        g2_chains = cached['group2_chains']

        # Apply mutations to respective chain sequences
        # We track per-chain mutant sequences separately, then concatenate
        mut_seq_per_chain = {c: seqs.get(c, '') for c in all_ab_chains}
        mutation_records = []
        valid = True

        for m in parsed_muts:
            chain = m['chain']
            if chain not in mut_seq_per_chain or chain not in resnum_maps:
                continue  # skip antigen-side mutations silently

            resnum_map = resnum_maps[chain]
            resnum = m['resnum']
            if resnum not in resnum_map:
                valid = False
                break

            seq_pos = resnum_map[resnum]
            cur_seq = mut_seq_per_chain[chain]
            if seq_pos >= len(cur_seq) or cur_seq[seq_pos] != m['wt_aa']:
                valid = False
                break

            try:
                mut_seq_per_chain[chain] = apply_mutation(cur_seq, seq_pos, m['mut_aa'])
            except (ValueError, IndexError):
                valid = False
                break

            role = 'heavy' if chain in g1_chains else 'light'
            mutation_records.append({
                'pos': seq_pos,
                'wt_aa': m['wt_aa'],
                'mut_aa': m['mut_aa'],
                'chain': role,
                'pdb': pdb_id,
            })

        if not valid or not mutation_records:
            n_skipped += 1
            continue

        # Build mutant concatenated sequences for each group
        def _concat_mut(chains):
            return ''.join(mut_seq_per_chain.get(c, seqs.get(c, '')) for c in sorted(chains))

        mut_heavy = _concat_mut(g1_chains)
        mut_light = _concat_mut(g2_chains)

        # For dedup with AB-Bind: map group1 chains → 'heavy', group2 → 'light'
        # so mutation keys like 'H:Y27A' match AB-Bind's 'heavy:Y27A'
        h_for_dedup = ','.join(sorted(g1_chains))
        l_for_dedup = ','.join(sorted(g2_chains))
        dedup_key = _normalize_mutation_key(pdb_id, parsed_muts,
                                            h_for_dedup, l_for_dedup)

        entries.append({
            'pdb_id': pdb_id,
            'h_chain': h_for_dedup,
            'l_chain': l_for_dedup,
            'ddg': ddg,
            'heavy_seq': mut_heavy,
            'light_seq': mut_light,
            'wt_heavy': wt_heavy,
            'wt_light': wt_light,
            'mutation_records': mutation_records,
            'source': 'skempi_full',
            'dedup_key': dedup_key,
        })

    print(f"  SKEMPI (full PP): {len(entries)} valid entries, {n_skipped} skipped")
    return entries


def _process_abbind_entries() -> List[Dict]:
    """Process AB-Bind entries into standardized format for merging.

    Re-uses abbind.py's download + parsing, but adds dedup keys.
    """
    from kirby.datasets.abbind import (
        _download_abbind_data, _parse_mutations, _identify_ab_chains,
        _NON_ABAG_PDBS as ABBIND_NON_ABAG,
    )

    df = _download_abbind_data()
    df = df[~df['#PDB'].isin(ABBIND_NON_ABAG)].copy()

    entries = []
    pdb_cache = {}
    n_skipped = 0

    for _, row in df.iterrows():
        pdb_id = str(row['#PDB']).strip()
        mutation_str = str(row['Mutation']).strip()
        ddg = float(row['ddG(kcal/mol)'])

        parsed_muts = _parse_mutations(mutation_str)
        if not parsed_muts:
            n_skipped += 1
            continue

        ab_chains, ag_chains = _identify_ab_chains(row.get('Partners(A_B)', ''))

        ab_muts = [m for m in parsed_muts if m['chain'] in ab_chains]
        if not ab_muts:
            n_skipped += 1
            continue

        if pdb_id not in pdb_cache:
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
                        resnum_maps['H'] = build_pdb_resnum_mapping(pdb_path, 'H')
                    except (ValueError, KeyError):
                        pass
                if 'L' in ab_chains:
                    try:
                        wt_light = extract_chain_sequence(pdb_path, 'L')
                        resnum_maps['L'] = build_pdb_resnum_mapping(pdb_path, 'L')
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
                print(f"    WARNING: AB-Bind PDB {pdb_id}: {e}")
                pdb_cache[pdb_id] = None

        cached = pdb_cache.get(pdb_id)
        if cached is None:
            n_skipped += 1
            continue

        wt_heavy = cached['wt_heavy']
        wt_light = cached['wt_light']
        resnum_maps = cached['resnum_maps']

        mut_heavy = wt_heavy
        mut_light = wt_light
        mutation_records = []
        valid = True

        for m in ab_muts:
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

        dedup_key = _normalize_mutation_key(pdb_id, ab_muts, 'H', 'L')

        entries.append({
            'pdb_id': pdb_id,
            'h_chain': 'H',
            'l_chain': 'L',
            'ddg': ddg,
            'heavy_seq': mut_heavy,
            'light_seq': mut_light,
            'wt_heavy': wt_heavy,
            'wt_light': wt_light,
            'mutation_records': mutation_records,
            'source': 'abbind',
            'dedup_key': dedup_key,
        })

    print(f"  AB-Bind: {len(entries)} valid entries, {n_skipped} skipped")
    return entries


def load_ddg_combined_dataset(
    max_samples: int = 5000,
    augment_reverse: bool = False,
    ab_ag_only: bool = True,
) -> Dict:
    """Load merged SKEMPI 2.0 + AB-Bind ΔΔG dataset with deduplication.

    Deduplication key: (PDB_ID, sorted normalized mutations).
    On conflict, AB-Bind's direct DDG measurement is preferred.

    Args:
        max_samples: Subsample to this size if exceeded.
        augment_reverse: If True, add reverse mutations (WT→MUT becomes MUT→WT
            with negated DDG). Roughly doubles training data. Forward+reverse
            pairs share the same pdb_id so GroupKFold keeps them together.
        ab_ag_only: If True (default), filter SKEMPI to antibody-antigen
            complexes only. If False, include ALL protein-protein SKEMPI
            entries (~6x more data: barnase-barstar, T4 lysozyme, etc.).
            NOTE: AB-Bind entries are always Ab-Ag regardless of this flag.
            EXPERIMENT OPTION: non-Ab sequences will not benefit from
            antibody-specific LM representations.

    Returns:
        dict with keys: heavy_seqs, light_seqs, labels, task_type,
            wildtype_heavy, wildtype_light, mutations, is_intra_family,
            property_name.
    """
    if ab_ag_only:
        print("  Loading combined ΔΔG dataset (SKEMPI 2.0 Ab-Ag + AB-Bind)...")
    else:
        print("  Loading combined ΔΔG dataset (SKEMPI 2.0 ALL PP + AB-Bind) "
              "[ab_ag_only=False]...")

    # Download and filter SKEMPI
    skempi_df = _download_skempi()

    if ab_ag_only:
        skempi_subset = skempi_df[skempi_df.apply(_is_abag_entry, axis=1)].copy()
        print(f"  SKEMPI: {len(skempi_subset)}/{len(skempi_df)} entries are Ab-Ag")
        skempi_entries = _process_skempi_entries(skempi_subset)
    else:
        print(f"  SKEMPI: using all {len(skempi_df)} protein-protein entries "
              f"(ab_ag_only=False)")
        skempi_entries = _process_skempi_all_entries(skempi_df)

    abbind_entries = _process_abbind_entries()

    # Deduplicate: AB-Bind preferred over SKEMPI on conflict
    combined = {}

    # Add SKEMPI first
    for entry in skempi_entries:
        combined[entry['dedup_key']] = entry

    # AB-Bind overwrites on conflict (preferred source)
    n_overlap = 0
    for entry in abbind_entries:
        if entry['dedup_key'] in combined:
            n_overlap += 1
        combined[entry['dedup_key']] = entry

    print(f"  Deduplication: {n_overlap} overlapping entries "
          f"(AB-Bind preferred)")
    print(f"  Combined: {len(combined)} unique entries "
          f"({len(skempi_entries)} SKEMPI + {len(abbind_entries)} AB-Bind "
          f"- {n_overlap} overlap)")

    entries_list = list(combined.values())

    # Remove extreme ddG outliers BEFORE augmentation (filter on real data,
    # then augment the filtered set to avoid asymmetric pair removal)
    ddg_values = np.array([e['ddg'] for e in entries_list])
    mu, sigma = ddg_values.mean(), ddg_values.std()
    keep = [(abs(v - mu) < 3 * sigma and abs(v) <= 8.0) for v in ddg_values]
    n_outliers = sum(1 for k in keep if not k)
    if n_outliers > 0:
        entries_list = [e for e, k in zip(entries_list, keep) if k]
        print(f"  Outlier filter: removed {n_outliers} entries "
              f"(|ddG|>8 or >3σ from mean)")

    # Reverse mutation augmentation AFTER outlier filtering
    if augment_reverse:
        n_before = len(entries_list)
        reverse_entries = []
        for entry in entries_list:
            wt_h = entry.get('wt_heavy', '')
            wt_l = entry.get('wt_light', '')
            if not wt_h and not wt_l:
                continue
            rev_records = []
            for mr in entry['mutation_records']:
                rev_records.append({
                    'pos': mr['pos'],
                    'wt_aa': mr['mut_aa'],
                    'mut_aa': mr['wt_aa'],
                    'chain': mr['chain'],
                    'pdb': mr['pdb'],
                })
            reverse_entries.append({
                'pdb_id': entry['pdb_id'],  # same PDB → GroupKFold keeps together
                'h_chain': entry['h_chain'],
                'l_chain': entry['l_chain'],
                'ddg': -entry['ddg'],
                'heavy_seq': wt_h,
                'light_seq': wt_l,
                'wt_heavy': entry['heavy_seq'],
                'wt_light': entry['light_seq'],
                'mutation_records': rev_records,
                'source': entry['source'] + '_rev',
                'dedup_key': entry['dedup_key'] + ('_rev',),
            })
        entries_list.extend(reverse_entries)
        print(f"  Reverse augmentation: {n_before} → {len(entries_list)} entries")

    all_heavy_seqs = []
    all_light_seqs = []
    all_labels = []
    all_mutations = []
    all_pdb_ids = []

    for entry in entries_list:
        all_heavy_seqs.append(entry['heavy_seq'] if isinstance(entry['heavy_seq'], str) else '')
        all_light_seqs.append(entry['light_seq'] if isinstance(entry['light_seq'], str) else '')
        all_labels.append(entry['ddg'])
        all_mutations.append(entry['mutation_records'])
        all_pdb_ids.append(entry['pdb_id'])

    if not all_heavy_seqs:
        raise ValueError("No valid entries in combined ΔΔG dataset")

    labels = np.array(all_labels, dtype=np.float64)

    # Check for light chains
    has_light = any(len(s) > 0 for s in all_light_seqs)
    light_seqs = all_light_seqs if has_light else None

    # Subsample if needed
    if len(all_heavy_seqs) > max_samples:
        print(f"  Subsampling {len(all_heavy_seqs)} → {max_samples}...")
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(len(all_heavy_seqs), max_samples, replace=False))
        all_heavy_seqs = [all_heavy_seqs[i] for i in idx]
        if light_seqs:
            light_seqs = [light_seqs[i] for i in idx]
        labels = labels[idx]
        all_mutations = [all_mutations[i] for i in idx]
        all_pdb_ids = [all_pdb_ids[i] for i in idx]

    print(f"  Final: {len(all_heavy_seqs)} entries")
    print(f"  ΔΔG range: [{labels.min():.3f}, {labels.max():.3f}] kcal/mol, "
          f"mean={labels.mean():.3f}")

    return {
        'heavy_seqs': all_heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'task_type': 'regression',
        'wildtype_heavy': None,  # multi-PDB dataset
        'wildtype_light': None,
        'mutations': all_mutations,
        'is_intra_family': False,
        'property_name': 'ΔΔG (kcal/mol)',
        'pdb_ids': all_pdb_ids,
    }


def assign_complex_groups(pdb_ids: List[str]) -> Tuple[np.ndarray, int]:
    """Assign group IDs by PDB complex for leave-one-complex-out GroupKFold."""
    unique_pdbs = sorted(set(pdb_ids))
    pdb_to_id = {p: i for i, p in enumerate(unique_pdbs)}
    groups = np.array([pdb_to_id[p] for p in pdb_ids])
    return groups, len(unique_pdbs)


# ---------------------------------------------------------------------------
# AbDesign stub
# ---------------------------------------------------------------------------

class AbDesignDataNotAvailableError(Exception):
    """Raised when AbDesign DDG data is requested but cannot be located."""


def load_abdesign_ddg(data_dir=None) -> Dict:
    """Load AbDesign CDR-H3 mutant ΔΔG dataset — STUB.

    The AbDesign dataset contains ~658 CDR-H3 mutants with experimentally
    measured binding affinities from the Rosetta-guided antibody design
    pipeline. It is used as an independent held-out test set for DDG
    prediction because its PDB structures do NOT overlap with SKEMPI 2.0
    or AB-Bind.

    Data availability:
        The data has been referenced in several papers but is NOT available
        in a canonical public repository with a stable download URL.

        Possible sources (as of 2026):
          1. Supplementary data of Goldenzweig et al. (2016) PNAS:
             https://www.pnas.org/doi/10.1073/pnas.1605871113
          2. The AbBiBench benchmark repository (if released):
             Search: https://github.com/search?q=AbBiBench
          3. The original AbDesign paper by Lapidoth et al. (2015):
             https://doi.org/10.1002/prot.24779

    To use this dataset once you have the data:
        1. Place a CSV at: <data_dir>/abdesign_ddg/abdesign.csv
        2. Required columns:
             - heavy_seq: mutant VH sequence
             - light_seq: mutant VL sequence (or empty string)
             - wt_heavy: wildtype VH sequence
             - wt_light: wildtype VL sequence (or empty string)
             - ddg: ΔΔG in kcal/mol (positive = destabilizing)
             - pdb_id: PDB ID of the parent structure
        3. Replace this stub with a loader following load_ddg_combined_dataset().

    Args:
        data_dir: Override cache directory (default: /Volumes/seagate or ~/.cache).

    Raises:
        AbDesignDataNotAvailableError: Always (this is a stub).
    """
    cache = Path(data_dir) if data_dir else _cache_dir()
    csv_path = cache / 'abdesign_ddg' / 'abdesign.csv'

    if csv_path.exists():
        # If the user has placed their data, load it
        return _load_abdesign_from_csv(csv_path)

    raise AbDesignDataNotAvailableError(
        "AbDesign CDR-H3 ΔΔG data not found.\n\n"
        "The AbDesign dataset is NOT available in a canonical public repository. "
        "Possible sources:\n"
        "  1. Goldenzweig et al. (2016) PNAS supplementary:\n"
        "     https://www.pnas.org/doi/10.1073/pnas.1605871113\n"
        "  2. Lapidoth et al. (2015) Proteins:\n"
        "     https://doi.org/10.1002/prot.24779\n"
        "  3. AbBiBench benchmark repository (if publicly released):\n"
        "     https://github.com/search?q=AbBiBench\n\n"
        "Once you have the data, place a CSV at:\n"
        f"  {csv_path}\n\n"
        "Required CSV columns: heavy_seq, light_seq, wt_heavy, wt_light, "
        "ddg, pdb_id"
    )


def _load_abdesign_from_csv(csv_path: Path) -> Dict:
    """Load AbDesign ΔΔG data from a user-provided CSV.

    Called automatically by load_abdesign_ddg() if the CSV exists.
    """
    df = pd.read_csv(csv_path)

    required = {'heavy_seq', 'ddg', 'pdb_id'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"AbDesign CSV at {csv_path} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=['heavy_seq', 'ddg']).copy()

    heavy_seqs = list(df['heavy_seq'].astype(str))
    light_seqs = list(df.get('light_seq', '').fillna('').astype(str)) \
        if 'light_seq' in df.columns else [''] * len(heavy_seqs)
    labels = df['ddg'].values.astype(np.float64)
    pdb_ids = list(df['pdb_id'].astype(str))

    print(f"  AbDesign: {len(heavy_seqs)} entries loaded from {csv_path}")
    print(f"  ΔΔG range: [{labels.min():.3f}, {labels.max():.3f}] kcal/mol")

    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs if any(s for s in light_seqs) else None,
        'labels': labels,
        'task_type': 'regression',
        'wildtype_heavy': None,
        'wildtype_light': None,
        'mutations': None,
        'is_intra_family': False,
        'property_name': 'ΔΔG (kcal/mol)',
        'pdb_ids': pdb_ids,
    }
