"""USPTO reaction dataset for CGR-SMILES pretraining.

~1M atom-mapped reactions from US patent grants (1976-2016).
Source: Daniel Lowe, Figshare 5104873. Atom mapping via Indigo.

These reactions are ALREADY atom-mapped, so no rxnmapper is needed
for CGR-SMILES corpus generation.

The data is distributed as a 7z archive on Figshare. This module
auto-downloads and extracts it (requires py7zr), or accepts a
manually-placed .rsmi file.
"""

import os
from pathlib import Path
from typing import List, Optional


def _cache_dir() -> Path:
    """Cache directory for USPTO data."""
    seagate = Path('/Volumes/seagate/kirby/uspto')
    if seagate.parent.parent.exists():
        return seagate
    data_root = Path(os.environ.get(
        'KIRBY_DATA_ROOT', os.path.expanduser('~/.cache/kirby/data')))
    server = data_root / 'uspto'
    if server.parent.exists():
        return server
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                str(Path.home() / '.cache' / 'kirby'))) / 'uspto'


# Figshare download URL for the grants SMILES 7z archive
_FIGSHARE_GRANTS_7Z_URL = (
    'https://figshare.com/ndownloader/files/8664379'
)
_RSMI_FILENAME = '1976_Sep2016_USPTOgrants_smiles.rsmi'


def download_uspto_grants(force: bool = False) -> Path:
    """Download and extract USPTO grants reaction SMILES from Figshare.

    Source: Lowe, D. (2017). Chemical reactions from US patents (1976-Sep2016).
    Figshare. https://doi.org/10.6084/m9.figshare.5104873.v1

    Returns path to the extracted .rsmi file.
    Requires: pip install py7zr
    """
    cache = _cache_dir()
    rsmi_path = cache / _RSMI_FILENAME

    if rsmi_path.exists() and not force:
        return rsmi_path

    # Check for manually-placed file
    if rsmi_path.exists():
        return rsmi_path

    # Download 7z archive
    archive_path = cache / 'grants_smiles.7z'
    if not archive_path.exists() or force:
        import requests
        print(f"  Downloading USPTO grants archive to {archive_path} ...")
        cache.mkdir(parents=True, exist_ok=True)

        resp = requests.get(_FIGSHARE_GRANTS_7Z_URL, stream=True, timeout=300)
        resp.raise_for_status()

        tmp = archive_path.with_suffix('.tmp')
        total = 0
        with open(tmp, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)
                if total % (50 * 1024 * 1024) == 0:
                    print(f"    {total / 1e6:.0f} MB downloaded...")
        os.replace(str(tmp), str(archive_path))
        print(f"  Downloaded {total / 1e6:.1f} MB")

    # Extract .rsmi file from 7z archive
    print(f"  Extracting {_RSMI_FILENAME} from archive...")
    try:
        import py7zr
    except ImportError:
        raise ImportError(
            "USPTO download requires py7zr for 7z extraction: "
            "pip install py7zr\n"
            f"Alternatively, manually extract {_RSMI_FILENAME} to {rsmi_path}"
        )

    with py7zr.SevenZipFile(str(archive_path), mode='r') as z:
        # Find the .rsmi file in the archive
        names = z.getnames()
        rsmi_names = [n for n in names if n.endswith('.rsmi')
                      and 'grants' in n.lower()]
        if not rsmi_names:
            rsmi_names = [n for n in names if n.endswith('.rsmi')]
        if not rsmi_names:
            raise FileNotFoundError(
                f"No .rsmi file found in archive. Contents: {names}")

        target = rsmi_names[0]
        z.extract(path=str(cache), targets=[target])

        # Move extracted file to expected location if needed
        extracted = cache / target
        if extracted != rsmi_path and extracted.exists():
            os.replace(str(extracted), str(rsmi_path))

    print(f"  Extracted to {rsmi_path}")
    return rsmi_path


def load_uspto_stereo(max_reactions: Optional[int] = None) -> List[str]:
    """Load atom-mapped reaction SMILES from USPTO grants.

    Auto-downloads from Figshare if not cached.

    Args:
        max_reactions: Limit number of reactions (None = all ~1M).

    Returns:
        List of atom-mapped reaction SMILES strings.
    """
    path = download_uspto_grants()
    print(f"  Parsing USPTO from {path} ...")

    rxn_smiles = []
    with open(path) as f:
        # Try to detect format from first line
        first_line = f.readline().strip()

        # Check if first line is a header (contains column names, not SMILES)
        if '>>' not in first_line:
            pass  # skip header
        else:
            # First line is data, parse it
            smi = _extract_rxn_smiles(first_line)
            if smi:
                rxn_smiles.append(smi)

        for line in f:
            smi = _extract_rxn_smiles(line.strip())
            if smi:
                rxn_smiles.append(smi)
            if max_reactions and len(rxn_smiles) >= max_reactions:
                break

    print(f"  Loaded {len(rxn_smiles)} reactions")
    return rxn_smiles


def _extract_rxn_smiles(line: str) -> Optional[str]:
    """Extract reaction SMILES from a line, handling multiple formats.

    Supports:
    - Tab-separated with SMILES in any column (finds the one with >>)
    - Plain reaction SMILES (one per line)
    """
    if not line or '>>' not in line:
        return None

    # If tab-separated, find the column containing >>
    if '\t' in line:
        for part in line.split('\t'):
            part = part.strip()
            if '>>' in part:
                return part
        return None

    # Plain SMILES
    return line.strip()
