"""
AbAgym Dataset Loader — Deep Mutational Scanning of antibody-antigen interfaces.

GitHub: https://github.com/3BioCompBio/AbAgym

AbAgym contains 68 DMS experiments, but only 5 mutate the antibody side.
This loader filters to those 5 and constructs mutant sequences from PDB
wildtype + single-site mutations.

The 5 antibody-side experiments:
  1. Ang2_2017_G6        (PDB 4zfg, ~471 interface mutations)
  2. EGFR_2013_Cetuximab  (PDB 1yy9, ~659 interface mutations)
  3. lysozyme_2019_D441   (PDB 1mlc, ~627 interface mutations)
  4. VEGF_2017b_G6        (PDB 4zff, ~513 interface mutations)
  5. HER2_2021_trastuzumab (PDB 8pwh, ~146 interface mutations)

CSV columns: DMS_name, PDB_file, chains, site, wildtype, mutation, mut_names,
    DMS_score, MinMax_normalized_DMS_score, Rank_quartile_normalized_DMS_score,
    closest_interface_atom_distance

Note: AbAgym uses H/L as chain labels, but PDB files may use different IDs.
The loader auto-detects the correct PDB chain by matching wildtype residues.

Reference:
- Deszyński et al. (2024) "AbAgym: Benchmarking of antibody-antigen interfaces"
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kirby.datasets.pdb_utils import download_pdb, extract_chain_sequence, apply_mutation


# The 5 antibody-side DMS experiments.
# pdb_chain_map: AbAgym chain label → actual PDB chain ID.
# Determined by matching wildtype residues at AbAgym sites against PDB residues.
ABAGYM_ANTIBODY_EXPERIMENTS = {
    'Ang2_2017_G6': {
        'pdb': '4zfg',
        'pdb_chain_map': {'H': 'H', 'L': 'L'},
        'description': 'G6 antibody → Angiopoietin-2',
    },
    'EGFR_2013_Cetuximab': {
        'pdb': '1yy9',
        'pdb_chain_map': {'H': 'D', 'L': 'C'},
        'description': 'Cetuximab → EGFR',
    },
    'lysozyme_2019_D441': {
        'pdb': '1mlc',
        'pdb_chain_map': {'H': 'B', 'L': 'A'},
        'description': 'D44.1 antibody → hen egg lysozyme',
    },
    'VEGF_2017b_G6': {
        'pdb': '4zff',
        'pdb_chain_map': {'H': 'H', 'L': 'L'},
        'description': 'G6 antibody → VEGF',
    },
    'HER2_2021_trastuzumab': {
        'pdb': '8pwh',
        'pdb_chain_map': {'H': 'B'},  # only H chain mutated in data
        'description': 'Trastuzumab → HER2',
    },
}

_GITHUB_RAW = "https://raw.githubusercontent.com/3BioCompBio/AbAgym/main"

# Three-letter to one-letter amino acid map (for PDB parsing)
_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def _cache_dir() -> Path:
    """Get cache directory, preferring /Volumes/seagate if available."""
    seagate = Path('/Volumes/seagate/kirby/abagym')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache = Path.home() / '.cache' / 'kirby' / 'abagym'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_abagym_data() -> pd.DataFrame:
    """Download the AbAgym non-redundant interface CSV (or load from cache)."""
    import requests

    cache = _cache_dir()
    cached = cache / 'AbAgym_data_non-redundant_interface.csv'

    if cached.exists():
        return pd.read_csv(cached)

    url = f"{_GITHUB_RAW}/AbAgym_data_non-redundant_interface.csv"
    print(f"  Downloading AbAgym interface data...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    return pd.read_csv(cached)


def _parse_site(site_str: str) -> int:
    """Parse AbAgym site string to integer residue number.

    Handles insertion codes like '100A' → 100.
    """
    s = str(site_str).strip()
    # Strip trailing letters (insertion codes)
    digits = ''.join(c for c in s if c.isdigit() or c == '-')
    if not digits:
        raise ValueError(f"Cannot parse site: {site_str}")
    return int(digits)


def _build_resnum_to_seqpos(pdb_path: Path, chain_id: str) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Map PDB residue number → 0-indexed sequence position for a chain.

    Returns:
        (resnum_to_pos, resnum_to_aa): position mapping and AA verification map.
    """
    from Bio.PDB import PDBParser

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pdb', str(pdb_path))

    chain = structure[0][chain_id]
    resnum_to_pos = {}
    resnum_to_aa = {}
    seq_pos = 0
    for residue in chain.get_residues():
        if residue.get_id()[0] != ' ':
            continue
        resnum = residue.get_id()[1]
        aa = _THREE_TO_ONE.get(residue.get_resname(), '')
        resnum_to_pos[resnum] = seq_pos
        resnum_to_aa[resnum] = aa
        seq_pos += 1

    return resnum_to_pos, resnum_to_aa


def load_abagym_dataset(
    name: str = 'abagym_all',
    max_samples: int = 5000,
) -> Dict:
    """Load AbAgym antibody-side DMS dataset.

    Args:
        name: Dataset name:
            - 'abagym_all': pool all 5 antibody-side experiments
            - 'abagym_<experiment>': individual experiment
              e.g. 'abagym_Ang2_2017_G6', 'abagym_EGFR_2013_Cetuximab'
        max_samples: Subsample large datasets to this size.

    Returns:
        dict with keys: heavy_seqs, light_seqs, labels, task_type,
            wildtype_heavy, wildtype_light, mutations, is_intra_family,
            property_name, interface_distances.
    """
    # Determine which experiments to include
    # Build case-insensitive lookup
    _exp_lower = {k.lower(): k for k in ABAGYM_ANTIBODY_EXPERIMENTS}

    name_stripped = name[len('abagym_'):] if name.lower().startswith('abagym_') else name

    if name.lower() == 'abagym_all':
        experiments = list(ABAGYM_ANTIBODY_EXPERIMENTS.keys())
        is_intra_family = False
    elif name_stripped.lower() in _exp_lower:
        exp_name = _exp_lower[name_stripped.lower()]
        experiments = [exp_name]
        is_intra_family = True  # single DMS experiment
    else:
        available = list(ABAGYM_ANTIBODY_EXPERIMENTS.keys())
        raise ValueError(
            f"Unknown AbAgym experiment: {name_stripped}. "
            f"Options: {available} or use 'abagym_all'"
        )

    # Download data
    df = _download_abagym_data()

    # Filter to selected experiments
    df = df[df['DMS_name'].isin(experiments)].copy()
    if len(df) == 0:
        raise ValueError(f"No data found for experiments: {experiments}")

    print(f"  AbAgym {name}: {len(df)} mutations across {len(experiments)} experiments")

    # Process each experiment
    all_heavy_seqs = []
    all_light_seqs = []
    all_labels = []
    all_mutations = []
    all_distances = []
    all_exp_labels = []  # track experiment name per entry for z-scoring
    wt_heavy_final = None
    wt_light_final = None

    for exp_name in experiments:
        exp_df = df[df['DMS_name'] == exp_name].copy()
        if len(exp_df) == 0:
            continue

        info = ABAGYM_ANTIBODY_EXPERIMENTS[exp_name]
        pdb_id = info['pdb']
        pdb_chain_map = info['pdb_chain_map']
        pdb_path = download_pdb(pdb_id)

        # Get wildtype sequences and residue mappings for each chain
        wt_seqs = {}     # ab_label → sequence
        resnum_maps = {} # ab_label → {resnum → seq_pos}
        resnum_aas = {}  # ab_label → {resnum → aa}

        for ab_label, pdb_chain_id in pdb_chain_map.items():
            wt_seqs[ab_label] = extract_chain_sequence(pdb_path, pdb_chain_id)
            pos_map, aa_map = _build_resnum_to_seqpos(pdb_path, pdb_chain_id)
            resnum_maps[ab_label] = pos_map
            resnum_aas[ab_label] = aa_map

        wt_heavy = wt_seqs.get('H', '')
        wt_light = wt_seqs.get('L', '')

        if wt_heavy_final is None:
            wt_heavy_final = wt_heavy
            wt_light_final = wt_light

        n_added = 0
        # Process each mutation
        for _, row in exp_df.iterrows():
            ab_chain = row['chains']  # 'H' or 'L' in AbAgym labels
            wt_aa = str(row['wildtype']).strip()
            mut_aa = str(row['mutation']).strip()

            if ab_chain not in pdb_chain_map:
                continue

            # Parse site number
            try:
                site = _parse_site(row['site'])
            except ValueError:
                continue

            # Get residue mapping
            resnum_map = resnum_maps.get(ab_chain, {})
            if site not in resnum_map:
                continue
            seq_pos = resnum_map[site]

            # Verify wildtype matches
            pdb_aa = resnum_aas.get(ab_chain, {}).get(site, '')
            if pdb_aa != wt_aa:
                continue  # mismatch, skip

            wt_seq = wt_seqs.get(ab_chain, '')
            if seq_pos >= len(wt_seq):
                continue

            # Construct mutant heavy/light
            try:
                if ab_chain == 'H':
                    mut_heavy = apply_mutation(wt_heavy, seq_pos, mut_aa)
                    mut_light = wt_light
                else:
                    mut_heavy = wt_heavy
                    mut_light = apply_mutation(wt_light, seq_pos, mut_aa)
            except (ValueError, IndexError):
                continue

            all_heavy_seqs.append(mut_heavy)
            all_light_seqs.append(mut_light)

            # Use MinMax normalized score as label
            label = row['MinMax_normalized_DMS_score']
            if pd.isna(label):
                label = row['DMS_score']
            if pd.isna(label):
                continue
            all_labels.append(float(label))
            all_exp_labels.append(exp_name)

            # Record mutation info
            mut_info = {
                'pos': seq_pos,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'chain': 'heavy' if ab_chain == 'H' else 'light',
                'experiment': exp_name,
            }
            all_mutations.append([mut_info])

            # Interface distance (free feature from AbAgym)
            dist = row.get('closest_interface_atom_distance', np.nan)
            all_distances.append(float(dist) if pd.notna(dist) else np.nan)
            n_added += 1

        print(f"    {exp_name}: {n_added} mutations from PDB {pdb_id}")

    if not all_heavy_seqs:
        raise ValueError(f"No valid mutations found for {name}")

    labels = np.array(all_labels, dtype=np.float64)

    # Z-score labels per experiment before concatenation (standard for pooling assays)
    if name.lower() == 'abagym_all' and len(experiments) > 1:
        exp_arr = np.array(all_exp_labels)
        for exp_name in experiments:
            mask = exp_arr == exp_name
            if mask.sum() > 1:
                mu, sigma = labels[mask].mean(), labels[mask].std()
                if sigma > 0:
                    labels[mask] = (labels[mask] - mu) / sigma
                    print(f"    Z-scored {exp_name}: {mask.sum()} entries, "
                          f"mu={mu:.3f}, sigma={sigma:.3f}")

    # Subsample if needed
    if len(all_heavy_seqs) > max_samples:
        print(f"  Subsampling {len(all_heavy_seqs)} → {max_samples}...")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_heavy_seqs), max_samples, replace=False)
        idx = sorted(idx)
        all_heavy_seqs = [all_heavy_seqs[i] for i in idx]
        all_light_seqs = [all_light_seqs[i] for i in idx]
        labels = labels[idx]
        all_mutations = [all_mutations[i] for i in idx]
        all_distances = [all_distances[i] for i in idx]

    # Check if we have light sequences
    has_light = any(len(s) > 0 for s in all_light_seqs)

    print(f"  Final: {len(all_heavy_seqs)} samples, "
          f"{'paired' if has_light else 'heavy-only'}, "
          f"label range [{labels.min():.3f}, {labels.max():.3f}]")

    return {
        'heavy_seqs': all_heavy_seqs,
        'light_seqs': all_light_seqs if has_light else None,
        'labels': labels,
        'task_type': 'regression',
        'wildtype_heavy': wt_heavy_final,
        'wildtype_light': wt_light_final,
        'mutations': all_mutations,
        'is_intra_family': is_intra_family,
        'property_name': 'DMS score (z-scored per experiment)' if name.lower() == 'abagym_all' else 'DMS score (MinMax normalized)',
        'interface_distances': np.array(all_distances, dtype=np.float64),
    }
