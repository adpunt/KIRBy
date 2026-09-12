"""
PDB Utilities — shared by AbAgym and AbBind dataset loaders.

Downloads PDB files from RCSB, extracts chain sequences using BioPython,
and applies single-site mutations.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from kirby.paths import data_dir


def _pdb_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Get PDB cache directory."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    return data_dir('pdb')


def download_pdb(pdb_id: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a PDB file from RCSB, return local path.

    Args:
        pdb_id: 4-character PDB identifier (e.g. '1AHW').
        cache_dir: Override cache directory.

    Returns:
        Path to the local PDB file.
    """
    import requests

    cache = _pdb_cache_dir(cache_dir)
    pdb_id_lower = pdb_id.lower()
    local_path = cache / f'{pdb_id_lower}.pdb'

    if local_path.exists():
        return local_path

    url = f'https://files.rcsb.org/download/{pdb_id_lower}.pdb'
    print(f"  Downloading PDB {pdb_id} from RCSB...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    return local_path


def extract_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    """Extract amino acid sequence from a specific chain in a PDB file.

    Uses BioPython's PDBParser + PPBuilder to build the sequence from
    ATOM records, handling non-standard residues gracefully.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier (e.g. 'H', 'L', 'A').

    Returns:
        Amino acid sequence string.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import PPBuilder

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pdb', str(pdb_path))

    model = structure[0]
    if chain_id not in [c.id for c in model.get_chains()]:
        available = [c.id for c in model.get_chains()]
        raise ValueError(
            f"Chain '{chain_id}' not found in {pdb_path.name}. "
            f"Available: {available}"
        )

    chain = model[chain_id]
    ppb = PPBuilder()
    polypeptides = ppb.build_peptides(chain)

    if not polypeptides:
        raise ValueError(
            f"No polypeptides found for chain '{chain_id}' in {pdb_path.name}"
        )

    # Concatenate all polypeptide fragments (handles chain breaks)
    sequence = ''.join(str(pp.get_sequence()) for pp in polypeptides)
    return sequence


def build_pdb_resnum_mapping(pdb_path: Path, chain_id: str) -> Dict[int, int]:
    """Build mapping from PDB residue number → 0-indexed sequence position.

    Iterates over standard residues (hetfield == ' ') in the given chain
    and maps each PDB residue number to its sequential position.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier (e.g. 'H', 'L', 'A').

    Returns:
        Dict mapping PDB residue numbers to 0-indexed sequence positions.
    """
    from Bio.PDB import PDBParser

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pdb', str(pdb_path))

    chain = structure[0][chain_id]
    mapping = {}
    seq_pos = 0
    for residue in chain.get_residues():
        hetfield = residue.get_id()[0]
        if hetfield != ' ':
            continue
        resnum = residue.get_id()[1]
        mapping[resnum] = seq_pos
        seq_pos += 1

    return mapping


def apply_mutation(wt_seq: str, position: int, mut_aa: str) -> str:
    """Apply a single-site mutation to a sequence (0-indexed position).

    Args:
        wt_seq: Wildtype amino acid sequence.
        position: 0-indexed position to mutate.
        mut_aa: Single-letter amino acid to substitute.

    Returns:
        Mutant sequence string.
    """
    if position < 0 or position >= len(wt_seq):
        raise ValueError(
            f"Position {position} out of range for sequence of length {len(wt_seq)}"
        )
    return wt_seq[:position] + mut_aa + wt_seq[position + 1:]
