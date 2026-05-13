"""
Humanness Classification Dataset Loader

Binary classification: human vs non-human antibody VH sequences.

Two presets:
  - species:     Human VH (label=1) vs mouse+rhesus+camelid VH (label=0)
                 Sourced from OAS test sets in the humanness evaluation archive
                 (Zenodo 10562968). Stratified subsample to max_samples.
  - therapeutic: 196 human therapeutics (1) vs 353 non-human (0)
                 From IMGT/mAb-DB via BioPhi (Prihoda et al. 2022).
                 TSV + FASTA from GitHub. Small dataset (549 total).

References:
  Marks et al., "Humanization of antibodies using a machine learning approach" (2021)
  Ramon et al., "AbNatiV: VQ-VAE-based assessment of antibody nativeness" (2024)
  Prihoda et al., "BioPhi: A platform for antibody design, humanization" (2022)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Species preset: OAS-derived test sets from Zenodo
_ZENODO_URL = (
    "https://zenodo.org/records/10562968/files/"
    "train_test_data_immunogenicity_0.0.1.tar.gz"
)

# Therapeutic preset: BioPhi IMGT/mAb-DB humanness benchmark
_BIOPHI_BASE = (
    "https://raw.githubusercontent.com/Merck/BioPhi-2021-publication"
    "/main/data/tasks/humanness"
)
_BIOPHI_TSV = f"{_BIOPHI_BASE}/imgt_mab_db.tsv"
_BIOPHI_FASTA = f"{_BIOPHI_BASE}/imgt_mab_db.fa"


def _cache_dir() -> Path:
    """Cache dir, preferring /Volumes/seagate if available."""
    seagate = Path('/Volumes/seagate/kirby/humanness')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    cache = cache_root / 'humanness'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _parse_fasta(fasta_path: Path) -> Dict[str, str]:
    """Parse a FASTA file, return {header: sequence} dict."""
    sequences = {}
    current_header = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences[current_header] = ''.join(current_seq)
                current_header = line[1:].strip()
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_header is not None:
        sequences[current_header] = ''.join(current_seq)
    return sequences


def _parse_fasta_list(fasta_path: Path) -> List[str]:
    """Parse a FASTA file, return list of sequences (ignoring headers)."""
    return list(_parse_fasta(fasta_path).values())


def _find_fasta_files(root: Path, species: str) -> List[Path]:
    """Recursively find FASTA files matching a species name."""
    species_lower = species.lower()
    matches = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        if not any(name_lower.endswith(ext) for ext in
                   ('.fasta', '.fa', '.fsa', '.fas', '.fna', '.faa')):
            continue
        path_str = str(p).lower()
        if species_lower in path_str:
            matches.append(p)
    return sorted(matches)


def _find_heavy_fasta(root: Path, species: str) -> Optional[Path]:
    """Find the heavy chain FASTA file for a species."""
    candidates = _find_fasta_files(root, species)
    # Prefer files with 'heavy' or 'vh' in name, exclude 'light'/'vl'
    heavy = [f for f in candidates
             if ('heavy' in f.name.lower() or 'vh' in f.name.lower())
             and 'light' not in f.name.lower() and 'vl' not in f.name.lower()]
    if heavy:
        return heavy[0]
    if len(candidates) == 1:
        return candidates[0]
    in_heavy_dir = [f for f in candidates if 'heavy' in str(f.parent).lower()]
    if in_heavy_dir:
        return in_heavy_dir[0]
    return candidates[0] if candidates else None


def load_humanness_dataset(
    preset: str = 'species',
    max_samples: int = 10000,
) -> Dict:
    """Load humanness classification dataset.

    Args:
        preset: 'species' (human vs mouse+rhesus+camelid VH) or
                'therapeutic' (human vs non-human therapeutics).
        max_samples: Maximum total samples for species preset.

    Returns:
        dict with: heavy_seqs, light_seqs, labels (int 0/1),
                   task_type='classification', n_classes=2, is_intra_family=False
    """
    if preset == 'species':
        root = _download_and_extract_species()
        return _load_species(root, max_samples)
    elif preset == 'therapeutic':
        return _load_therapeutic()
    else:
        raise ValueError(f"Unknown preset: {preset}. Use 'species' or 'therapeutic'.")


# =========================================================================
# Species preset — Zenodo OAS test data
# =========================================================================

def _download_and_extract_species() -> Path:
    """Download humanness evaluation archive from Zenodo, extract, return root dir."""
    import requests
    import tarfile

    cache = _cache_dir()
    archive = cache / 'train_test_data_immunogenicity_0.0.1.tar.gz'
    extracted = cache / 'extracted'

    if extracted.exists() and any(extracted.iterdir()):
        return extracted

    if not archive.exists():
        print(f"  Downloading humanness species data from Zenodo (1.3 GB)...")
        resp = requests.get(_ZENODO_URL, timeout=600, stream=True)
        resp.raise_for_status()
        with open(archive, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Download complete: {archive}")

    print(f"  Extracting archive...")
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(extracted)
    print(f"  Extracted to {extracted}")

    return extracted


def _load_species(root: Path, max_samples: int) -> Dict:
    """Load species classification: human VH (1) vs non-human VH (0)."""
    rng = np.random.RandomState(42)

    species_map = {
        'human': 1,
        'mouse': 0,
        'rhesus': 0,
        'camelid': 0,
    }

    all_seqs = []
    all_labels = []
    per_species_counts = {}

    for species, label in species_map.items():
        fasta_path = _find_heavy_fasta(root, species)
        if fasta_path is None:
            print(f"  WARNING: No FASTA found for {species}, skipping")
            continue

        seqs = _parse_fasta_list(fasta_path)
        print(f"  {species}: {len(seqs)} VH sequences from {fasta_path.name}")
        per_species_counts[species] = len(seqs)
        all_seqs.extend(seqs)
        all_labels.extend([label] * len(seqs))

    if not all_seqs:
        raise RuntimeError(
            f"No FASTA files found in {root}. "
            "Expected species-specific VH FASTA files (human, mouse, rhesus, camelid). "
            "Archive structure may have changed — check contents manually."
        )

    labels = np.array(all_labels, dtype=np.int64)

    # Stratified subsample: balance across species
    if len(all_seqs) > max_samples:
        n_species = len(per_species_counts)
        per_species_target = max_samples // n_species
        indices = []
        offset = 0
        for species in species_map:
            count = per_species_counts.get(species, 0)
            if count == 0:
                continue
            n_take = min(per_species_target, count)
            species_idx = np.arange(offset, offset + count)
            chosen = rng.choice(species_idx, n_take, replace=False)
            indices.extend(chosen.tolist())
            offset += count

        if len(indices) > max_samples:
            indices = rng.choice(indices, max_samples, replace=False).tolist()
        indices = sorted(indices)

        all_seqs = [all_seqs[i] for i in indices]
        labels = labels[indices]

    n = len(all_seqs)
    n_pos = int((labels == 1).sum())
    print(f"  Humanness (species): {n} samples, "
          f"{n_pos} human ({100*n_pos/n:.1f}%), "
          f"{n - n_pos} non-human ({100*(n-n_pos)/n:.1f}%)")

    return {
        'heavy_seqs': all_seqs,
        'light_seqs': [''] * n,
        'labels': labels,
        'task_type': 'classification',
        'n_classes': 2,
        'is_intra_family': False,
        'property_name': 'Humanness (species: human=1, non-human=0)',
        'mutations': [[] for _ in range(n)],
        'wildtype_heavy': None,
        'wildtype_light': None,
    }


# =========================================================================
# Therapeutic preset — BioPhi IMGT/mAb-DB (549 therapeutics)
# =========================================================================

def _load_therapeutic() -> Dict:
    """Load therapeutic classification from BioPhi IMGT/mAb-DB data.

    Downloads TSV (species labels) + FASTA (VH/VL sequences) from
    the BioPhi-2021-publication GitHub repo.

    FASTA headers: >{INN_Number}_seq1 (VH), >{INN_Number}_seq2 (VL)
    TSV columns: INN_Number, Antibody, Species
    Species: Human, Humanized, Mouse, Chimeric, etc.
    Label: Human → 1, everything else → 0.
    """
    import requests

    cache = _cache_dir()
    tsv_path = cache / 'imgt_mab_db.tsv'
    fasta_path = cache / 'imgt_mab_db.fa'

    # Download TSV and FASTA
    for url, path in [(_BIOPHI_TSV, tsv_path), (_BIOPHI_FASTA, fasta_path)]:
        if not path.exists():
            print(f"  Downloading {path.name}...")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)

    # Parse TSV: INN_Number, Antibody, Species
    df = pd.read_csv(tsv_path, sep='\t')

    # Parse FASTA: headers like >{INN_Number}_seq1 (VH), >{INN_Number}_seq2 (VL)
    fasta = _parse_fasta(fasta_path)

    # Build per-antibody VH sequences keyed by INN_Number
    vh_by_inn = {}
    for header, seq in fasta.items():
        if '_seq1' in header:
            inn = header.split('_seq1')[0]
            vh_by_inn[inn] = seq

    # Match TSV entries to VH sequences
    heavy_seqs = []
    labels_list = []
    for _, row in df.iterrows():
        inn = str(row['INN_Number'])
        species = str(row.get('Species', '')).strip()
        vh = vh_by_inn.get(inn)
        if vh is None or not species:
            continue

        heavy_seqs.append(vh)
        # Human = 1, everything else (Humanized, Mouse, Chimeric) = 0
        label = 1 if species.lower() == 'human' else 0
        labels_list.append(label)

    labels = np.array(labels_list, dtype=np.int64)
    n = len(heavy_seqs)
    n_pos = int((labels == 1).sum())

    print(f"  Humanness (therapeutic, BioPhi IMGT): {n} samples, "
          f"{n_pos} human, {n - n_pos} non-human")

    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': [''] * n,
        'labels': labels,
        'task_type': 'classification',
        'n_classes': 2,
        'is_intra_family': False,
        'property_name': 'Humanness (therapeutic: human=1, non-human=0)',
        'mutations': [[] for _ in range(n)],
        'wildtype_heavy': None,
        'wildtype_light': None,
    }
