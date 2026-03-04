"""
PDBBind Dataset Loader

Loads PDBBind v2020 refined and core sets for structure-based binding affinity
prediction. Uses a sequence-based approach: extracts protein sequences and
ligand SMILES from the index files. Returns the same dict format as DTA
loaders → plugs directly into the existing DTA pipeline.

Data requires manual download from pdbbind.org.cn (free registration).

Splits:
- Random: standard train/val/test by PDB ID
- LP-PDBBind: leak-proof splits via protein sequence clustering (30% identity)

Reference:
- Wang et al. (2004) J. Med. Chem. (PDBBind)
- Su et al. (2019) PNAS (PDBBind v2019)
- LP-PDBBind: Volkov et al. (2022) J. Chem. Inf. Model.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def _find_data_dir() -> Optional[Path]:
    """Find PDBBind data directory from env var or standard paths."""
    if 'KIRBY_PDBBIND_DIR' in os.environ:
        d = Path(os.environ['KIRBY_PDBBIND_DIR'])
        if d.exists():
            return d

    candidates = [
        Path('/Volumes/seagate/kirby/pdbbind'),
        Path.home() / '.cache' / 'kirby' / 'pdbbind',
        Path.home() / 'data' / 'pdbbind',
        Path('/data/stat-cadd/scat9264/KIRBy/data/pdbbind'),
    ]
    for d in candidates:
        if d.exists():
            return d

    return None


def _parse_index_file(index_path: Path) -> List[Dict]:
    """Parse PDBBind INDEX file.

    PDBBind index format (tab-separated, comments start with #):
    PDB_code  resolution  release_year  -logKd/Ki  Kd/Ki=value  reference  ligand_name

    Returns list of dicts with pdb_id, affinity, unit, value_str.
    """
    entries = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdb_id = parts[0].lower()
            try:
                neg_log_affinity = float(parts[3])
            except ValueError:
                continue
            entries.append({
                'pdb_id': pdb_id,
                'affinity': neg_log_affinity,  # -log(Kd/Ki) in M
            })
    return entries


def _extract_protein_seq(pdb_dir: Path, pdb_id: str) -> Optional[str]:
    """Extract protein sequence from PDBBind pocket file or full PDB.

    Tries: {pdb_id}/{pdb_id}_pocket.pdb, then {pdb_id}/{pdb_id}_protein.pdb
    Extracts CA atoms to build sequence.
    """
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }

    for suffix in ['_protein.pdb', '_pocket.pdb']:
        pdb_file = pdb_dir / pdb_id / f'{pdb_id}{suffix}'
        if not pdb_file.exists():
            continue

        seq = []
        seen_residues = set()
        with open(pdb_file) as f:
            for line in f:
                if line.startswith(('ATOM  ', 'HETATM')):
                    atom_name = line[12:16].strip()
                    if atom_name != 'CA':
                        continue
                    res_name = line[17:20].strip()
                    chain = line[21]
                    res_num = line[22:27].strip()
                    res_key = (chain, res_num)
                    if res_key in seen_residues:
                        continue
                    seen_residues.add(res_key)
                    aa = three_to_one.get(res_name)
                    if aa:
                        seq.append(aa)

        if seq:
            return ''.join(seq)

    return None


def _extract_ligand_smiles(pdb_dir: Path, pdb_id: str) -> Optional[str]:
    """Extract ligand SMILES from PDBBind SDF/MOL2 file.

    Tries: {pdb_id}/{pdb_id}_ligand.sdf, then {pdb_id}/{pdb_id}_ligand.mol2
    Uses RDKit for conversion.
    """
    from rdkit import Chem

    # Try SDF first
    sdf_file = pdb_dir / pdb_id / f'{pdb_id}_ligand.sdf'
    if sdf_file.exists():
        try:
            mol = Chem.MolFromMolFile(str(sdf_file), removeHs=True, sanitize=True)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            pass

    # Try MOL2
    mol2_file = pdb_dir / pdb_id / f'{pdb_id}_ligand.mol2'
    if mol2_file.exists():
        try:
            mol = Chem.MolFromMol2File(str(mol2_file), removeHs=True, sanitize=True)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            pass

    return None


def load_dataset(name: str, data_dir: Optional[str] = None) -> Dict:
    """Load a PDBBind dataset subset.

    Args:
        name: 'pdbbind_refined' or 'pdbbind_core'
        data_dir: Path to PDBBind data directory. If None, searches standard
                  locations and KIRBY_PDBBIND_DIR env var.

    Returns:
        dict matching DTA format:
            drug_smiles: List[str]
            protein_seqs: List[str]
            labels: np.ndarray — -log(Kd/Ki) in M
            task_type: 'regression'
            metadata: {pdb_ids, n_complexes}
    """
    if data_dir:
        base_dir = Path(data_dir)
    else:
        base_dir = _find_data_dir()

    if base_dir is None or not base_dir.exists():
        raise FileNotFoundError(
            "PDBBind data not found. Download from pdbbind.org.cn and set "
            "KIRBY_PDBBIND_DIR environment variable, or place data in "
            "~/.cache/kirby/pdbbind/"
        )

    # Determine subset
    name_lower = name.lower()
    if 'core' in name_lower:
        subset = 'coreset'
    elif 'refined' in name_lower:
        subset = 'refined-set'
    else:
        raise ValueError(f"Unknown PDBBind subset: {name}. Use 'pdbbind_refined' or 'pdbbind_core'")

    # Find index file
    index_candidates = [
        base_dir / subset / 'index' / 'INDEX_general_PL_data.2020',
        base_dir / subset / 'index' / 'INDEX_refined_data.2020',
        base_dir / subset / 'index' / 'INDEX_core_data.2020',
        base_dir / 'index' / 'INDEX_general_PL_data.2020',
        base_dir / 'index' / 'INDEX_refined_data.2020',
    ]

    index_path = None
    for p in index_candidates:
        if p.exists():
            index_path = p
            break

    if index_path is None:
        # Try to find any INDEX file
        for f in base_dir.rglob('INDEX_*'):
            index_path = f
            break

    if index_path is None:
        raise FileNotFoundError(
            f"No PDBBind INDEX file found in {base_dir}. Expected in "
            f"{subset}/index/ directory."
        )

    print(f"  Loading PDBBind {subset} from {index_path}")
    entries = _parse_index_file(index_path)
    print(f"  Found {len(entries)} entries in index file")

    # Find the directory with PDB subdirectories
    pdb_dir = base_dir / subset
    if not (pdb_dir / entries[0]['pdb_id']).exists():
        # Try base_dir directly
        if (base_dir / entries[0]['pdb_id']).exists():
            pdb_dir = base_dir
        else:
            # Search for the directory
            for d in base_dir.iterdir():
                if d.is_dir() and (d / entries[0]['pdb_id']).exists():
                    pdb_dir = d
                    break

    # Extract sequences and SMILES
    drug_smiles = []
    protein_seqs = []
    labels = []
    pdb_ids = []
    skipped = 0

    for entry in entries:
        pid = entry['pdb_id']

        seq = _extract_protein_seq(pdb_dir, pid)
        if seq is None or len(seq) < 10:
            skipped += 1
            continue

        smi = _extract_ligand_smiles(pdb_dir, pid)
        if smi is None:
            skipped += 1
            continue

        protein_seqs.append(seq)
        drug_smiles.append(smi)
        labels.append(entry['affinity'])
        pdb_ids.append(pid)

    labels = np.array(labels, dtype=np.float64)

    print(f"  PDBBind {subset}: {len(labels)} complexes loaded "
          f"({skipped} skipped due to missing files)")
    print(f"  Label range: [{labels.min():.2f}, {labels.max():.2f}], "
          f"mean={labels.mean():.2f}")

    return {
        'drug_smiles': drug_smiles,
        'protein_seqs': protein_seqs,
        'labels': labels,
        'task_type': 'regression',
        'metadata': {
            'pdb_ids': pdb_ids,
            'n_complexes': len(labels),
            'subset': subset,
        },
    }
