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


def _gb_cache_dir() -> Path:
    """Stable, shared directory for extracted Genomic Benchmarks datasets.

    Preference order:
      1. /Volumes/seagate/kirby/genomic_benchmarks  (local workstation)
      2. $KIRBY_CACHE_DIR/genomic_benchmarks         (server, set in setup.sh)
      3. ~/.genomic_benchmarks                        (library default fallback)

    Using a stable path (instead of letting the library default to a fresh
    location each run) is what lets concurrent jobs and reruns share one
    extracted copy rather than each re-downloading.
    """
    seagate = Path('/Volumes/seagate/kirby/genomic_benchmarks')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    env_root = os.environ.get('KIRBY_CACHE_DIR')
    if env_root:
        cache = Path(env_root) / 'genomic_benchmarks'
        cache.mkdir(parents=True, exist_ok=True)
        return cache
    fallback = Path.home() / '.genomic_benchmarks'
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _gb_dataset_ready(data_dir: Path) -> bool:
    """True iff an extracted GB dataset already exists with usable content.

    The extracted layout is <data_dir>/{train,test}/<class>/<id>.txt. We only
    require at least one split with at least one non-empty .txt file, because
    the loader below merges whatever train/test splits are present.
    """
    if not data_dir.is_dir():
        return False
    for split_name in ('train', 'test'):
        split_dir = data_dir / split_name
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for txt_file in class_dir.glob('*.txt'):
                try:
                    if txt_file.stat().st_size > 0:
                        return True
                except OSError:
                    continue
    return False


def _gb_download_guarded(name: str, gb_download) -> Path:
    """Download/extract a Genomic Benchmark robustly under concurrency.

    The upstream genomic_benchmarks ``download_from_cloud_cache`` hardcodes
    ``force_download=True``: it ALWAYS ``shutil.rmtree``s the destination and
    re-downloads the Zenodo zip, with no "already present" guard. Under many
    concurrent SLURM jobs sharing one cache dir, one job rmtrees/overwrites the
    zip while another is mid-unpack, raising FileNotFoundError / stale file
    handle (cloud_caching.py:28/31/33). It also wastes a full re-download every
    rerun.

    This wrapper:
      * points the library at a STABLE shared cache dir (so reruns/jobs reuse),
      * SKIPS the download entirely if the extracted dataset is already present,
      * serialises the actual download with an inter-process file lock so only
        one job downloads while the rest wait and then reuse the result.

    The returned data is identical to calling gb_download directly (same Zenodo
    zip, same extracted layout); only the download timing/robustness changes.
    """
    cache_dir = _gb_cache_dir()
    data_dir = cache_dir / name

    # Fast path: already extracted by a previous run/job — no download at all.
    if _gb_dataset_ready(data_dir):
        print(f"  Reusing extracted Genomic Benchmark at {data_dir}")
        return data_dir

    # Allow an explicit opt-out (revert to old behaviour) if ever needed.
    if os.environ.get('KIRBY_GB_FORCE_DOWNLOAD') == '1':
        return Path(gb_download(name, version=0, dest_path=cache_dir))

    lock_path = cache_dir / f'.{name}.lock'

    # POSIX file lock (fcntl) — present on the ARC Linux cluster and macOS.
    # Falls back to an unlocked download if locking is unavailable.
    try:
        import fcntl
    except ImportError:
        fcntl = None

    if fcntl is None:
        # No lock available: still skip if ready, else download once.
        if _gb_dataset_ready(data_dir):
            return data_dir
        return Path(gb_download(name, version=0, dest_path=cache_dir))

    lock_file = open(lock_path, 'w')
    try:
        print(f"  Acquiring Genomic Benchmark download lock {lock_path}...")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        # Re-check under the lock: a concurrent job may have just finished.
        if _gb_dataset_ready(data_dir):
            print(f"  Another job already extracted {data_dir}; reusing.")
            return data_dir
        return Path(gb_download(name, version=0, dest_path=cache_dir))
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _assign_hash_groups(sequences: List[str], n_groups: int = 5) -> Tuple[np.ndarray, int]:
    """Hash-based grouping for sequences without natural group structure.

    Uses MD5 (deterministic) instead of Python's hash() which is randomized.

    NOT a biological group: an md5 of the sequence keeps byte-identical
    duplicates together and scatters everything else, so the folds it produces
    are a random partition of the rows wearing a group label. Kept only for
    callers that explicitly want a duplicate-safe arbitrary partition. The four
    Genomic Benchmarks datasets moved off it to `_assign_kmer_groups` on
    2026-09-04; nothing else in the repo calls it.
    """
    import hashlib
    groups = np.array([
        int.from_bytes(hashlib.md5(s.encode()).digest()[:4], 'big') % n_groups
        for s in sequences
    ])
    return groups, int(len(np.unique(groups)))


# k-mer length and cluster count for `_assign_kmer_groups`. 5 gives 1024
# distinct DNA 5-mers, which is small enough to count over 100k sequences in
# seconds and long enough that two unrelated sequences do not share a profile.
# 50 clusters rather than 5: GroupKFold packs whole groups into folds, so with
# exactly one cluster per fold a single large cluster would make one fold most
# of the dataset. 50 clusters lets it balance the fold sizes while every
# sequence still travels with the sequences it shares k-mer content with.
_KMER_K = 5
_KMER_N_CLUSTERS = 50


def _assign_kmer_groups(sequences: List[str], k: int = _KMER_K,
                        n_clusters: int = _KMER_N_CLUSTERS,
                        seed: int = 0) -> Tuple[np.ndarray, int]:
    """Group sequences by shared k-mer content, for datasets with no coordinate.

    Each sequence becomes an L2-normalised count over its k-mers, and those
    profiles are clustered with MiniBatchKMeans. Homologous or overlapping
    sequences share most of their k-mers and land in one cluster, so a fold that
    holds a cluster out holds out the sequences that resemble each other. This
    is what the four Genomic Benchmarks datasets use in place of a chromosome:
    two of them ship no coordinate through any path this loader takes, and the
    other two ship none through HuggingFace either.

    Deterministic: the vectoriser is stateless and the clusterer is seeded.

    Returns (groups, n_groups). n_clusters is capped at the number of sequences.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import HashingVectorizer

    n = len(sequences)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    n_clusters = int(max(1, min(n_clusters, n)))
    if n_clusters == 1:
        return np.zeros(n, dtype=np.int64), 1

    # 4**k buckets, one per possible DNA k-mer, so the hash collides only on
    # sequences containing an ambiguity code. alternate_sign=False keeps the
    # counts non-negative; norm='l2' makes a long sequence and a short one with
    # the same composition land in the same place.
    vec = HashingVectorizer(analyzer='char', ngram_range=(k, k),
                            n_features=4 ** k, alternate_sign=False,
                            norm='l2', lowercase=True)
    X = vec.transform(sequences)

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                         n_init=10, batch_size=1024)
    groups = km.fit_predict(X).astype(np.int64)
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

    # Check if HF dataset has actual sequences (some return coordinates only)
    _hf_has_sequences = False
    if ds is not None:
        _first_split = 'train' if 'train' in ds else list(ds.keys())[0]
        _cols = ds[_first_split].column_names
        _hf_has_sequences = any(c in _cols for c in ['sequence', 'seq', 'text', 'dna'])
        if not _hf_has_sequences:
            print(f"  HuggingFace has coordinate-only columns {_cols}, "
                  f"falling back to genomic-benchmarks package...")
            ds = None

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

        print(f"  Using genomic-benchmarks package...")
        data_dir = _gb_download_guarded(name, gb_download)
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

    # Group by shared k-mer content. The shipped train/test split is read above
    # and then concatenated away, and neither it nor a chromosome coordinate is
    # available for all four of these datasets; clustering the k-mer profiles is
    # what makes the folds hold out sequences that resemble each other rather
    # than an arbitrary fifth of the rows (was md5(sequence) % 5 until
    # 2026-09-04).
    groups, n_groups = _assign_kmer_groups(all_sequences)

    print(f"  {len(all_sequences)} sequences, {n_classes} classes, "
          f"{n_groups} k-mer groups (k={_KMER_K}), "
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
