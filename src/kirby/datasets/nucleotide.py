"""
Nucleotide (Genomics) Dataset Loaders

Loads DNA/RNA sequence classification tasks from:
  - NT Revised: 18 tasks from InstaDeepAI (HuggingFace)
  - Genomic Benchmarks: 4 tasks from HuggingFace

All loaders return a standard dict:
    sequences: List[str]     — DNA sequences
    labels: np.ndarray       — integer class labels
    task_type: str            — 'classification'
    groups: np.ndarray        — group IDs for GroupKFold
    n_groups: int
    n_classes: int
    property_name: str
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Cache HuggingFace data on seagate
_SEAGATE = Path('/Volumes/seagate/kirby/hf_cache')
if _SEAGATE.parent.parent.exists():
    os.environ.setdefault('HF_HOME', str(_SEAGATE))

# NT Revised task names (18 classification tasks)
NT_TASKS = [
    'promoter_all', 'promoter_tata', 'promoter_no_tata',
    'enhancers', 'enhancers_types',
    'splice_sites_all', 'splice_sites_acceptors', 'splice_sites_donors',
    'H2AFZ', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
    'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K9me3', 'H4K20me1',
]

# Genomic Benchmarks task names (4 selected tasks)
GB_TASKS = [
    'human_nontata_promoters',
    'human_enhancers_cohn',
    'demo_coding_vs_intergenomic_seqs',
    'demo_human_or_worm',
]

_NT_HF_ID = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"


def _extract_chromosome(name: str) -> str:
    """Extract chromosome from NT name field (e.g. 'chr8:16774000-16775000|0' -> 'chr8')."""
    if ':' in name:
        return name.split(':')[0]
    return name


def assign_chromosome_groups(chromosomes: List[str]) -> Tuple[np.ndarray, int]:
    """Map chromosome labels to integer group IDs for GroupKFold."""
    unique_chroms = sorted(set(chromosomes))
    chrom_to_id = {c: i for i, c in enumerate(unique_chroms)}
    groups = np.array([chrom_to_id[c] for c in chromosomes])
    return groups, len(unique_chroms)


def _assign_hash_groups(sequences: List[str], n_groups: int = 5) -> Tuple[np.ndarray, int]:
    """Hash-based grouping for sequences without natural group structure.

    Uses MD5 (deterministic) instead of Python's hash() which is randomized.
    """
    import hashlib
    groups = np.array([
        int.from_bytes(hashlib.md5(s.encode()).digest()[:4], 'big') % n_groups
        for s in sequences
    ])
    return groups, int(len(np.unique(groups)))


def load_nt_revised_task(task_name: str) -> Dict:
    """Load one of 18 NT Revised tasks from HuggingFace.

    The dataset has all 18 tasks in one file with a 'task' column for filtering
    and a 'name' column containing chromosome info (e.g. 'chr8:1234-5678|0').

    Returns standard dict with sequences, labels, groups, task_type.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("NT Revised tasks require: pip install datasets")

    if task_name not in NT_TASKS:
        raise ValueError(
            f"Unknown NT task '{task_name}'. Available: {NT_TASKS}"
        )

    print(f"Loading NT Revised task: {task_name}...")
    ds = load_dataset(_NT_HF_ID)

    # Merge train + test, filter to requested task
    all_sequences = []
    all_labels = []
    all_chroms = []

    for split_name in ['train', 'test']:
        if split_name not in ds:
            continue
        split = ds[split_name]

        # Filter to requested task (use .filter() for speed)
        filtered = split.filter(
            lambda x: x['task'] == task_name,
            num_proc=1,
        )

        if len(filtered) == 0:
            continue

        all_sequences.extend(filtered['sequence'])
        all_labels.extend(filtered['label'])

        # Extract chromosome from name field
        chroms = [_extract_chromosome(n) for n in filtered['name']]
        all_chroms.extend(chroms)

    if not all_sequences:
        raise ValueError(f"No data found for task '{task_name}' in {_NT_HF_ID}")

    labels = np.array(all_labels, dtype=np.int64)
    n_classes = len(np.unique(labels))

    # Use chromosome-based groups for GroupKFold
    groups, n_groups = assign_chromosome_groups(all_chroms)

    print(f"  {len(all_sequences)} sequences, {n_classes} classes, "
          f"{n_groups} chromosome groups, "
          f"seq lengths: {min(len(s) for s in all_sequences)}-"
          f"{max(len(s) for s in all_sequences)} bp")

    return {
        'sequences': all_sequences,
        'labels': labels,
        'task_type': 'classification',
        'groups': groups,
        'n_groups': n_groups,
        'n_classes': n_classes,
        'property_name': f'NT Revised: {task_name}',
    }


def load_genomic_benchmark(name: str) -> Dict:
    """Load a Genomic Benchmarks task from HuggingFace.

    Returns standard dict with sequences, labels, groups, task_type.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Genomic Benchmarks require: pip install datasets")

    if name not in GB_TASKS:
        raise ValueError(
            f"Unknown Genomic Benchmark '{name}'. Available: {GB_TASKS}"
        )

    print(f"Loading Genomic Benchmark: {name}...")

    # Try multiple HuggingFace dataset IDs
    ds = None
    tried_ids = []
    for hf_id in [
        "InstaDeepAI/genomic-benchmarks",
        "katielink/genomic-benchmarks",
    ]:
        try:
            ds = load_dataset(hf_id, name)
            break
        except Exception:
            tried_ids.append(hf_id)
            continue

    if ds is None:
        # Fallback: try the genomic-benchmarks pip package
        try:
            from genomic_benchmarks.loc2seq import download_dataset as gb_download
        except ImportError:
            raise RuntimeError(
                f"Could not load Genomic Benchmark '{name}' from HuggingFace.\n"
                f"Tried: {tried_ids}\n"
                f"Install the genomic-benchmarks package as fallback:\n"
                f"  pip install genomic-benchmarks"
            )

        print(f"  HuggingFace failed, using genomic-benchmarks package...")
        data_dir = Path(gb_download(name, version=0))
        all_sequences = []
        all_labels = []

        for split_name in ['train', 'test']:
            split_dir = data_dir / split_name
            if not split_dir.exists():
                continue
            class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
            for label_idx, class_dir in enumerate(class_dirs):
                for txt_file in sorted(class_dir.glob('*.txt')):
                    seq = txt_file.read_text().strip()
                    if seq:
                        all_sequences.append(seq)
                        all_labels.append(label_idx)
    else:
        all_sequences = []
        all_labels = []

        for split_name in ['train', 'test']:
            if split_name not in ds:
                continue
            split = ds[split_name]
            col_names = split.column_names

            # Detect columns
            seq_col = None
            for candidate in ['sequence', 'seq', 'text', 'dna']:
                if candidate in col_names:
                    seq_col = candidate
                    break
            if seq_col is None:
                raise ValueError(f"No sequence column found in {col_names}")

            label_col = None
            for candidate in ['label', 'labels', 'y', 'target']:
                if candidate in col_names:
                    label_col = candidate
                    break
            if label_col is None:
                raise ValueError(f"No label column found in {col_names}")

            all_sequences.extend(split[seq_col])
            all_labels.extend(split[label_col])

    labels = np.array(all_labels, dtype=np.int64)
    n_classes = len(np.unique(labels))

    # Use hash-based groups
    groups, n_groups = _assign_hash_groups(all_sequences, n_groups=5)

    print(f"  {len(all_sequences)} sequences, {n_classes} classes, "
          f"{n_groups} groups, "
          f"seq lengths: {min(len(s) for s in all_sequences)}-"
          f"{max(len(s) for s in all_sequences)} bp")

    return {
        'sequences': all_sequences,
        'labels': labels,
        'task_type': 'classification',
        'groups': groups,
        'n_groups': n_groups,
        'n_classes': n_classes,
        'property_name': f'Genomic Benchmark: {name}',
    }
