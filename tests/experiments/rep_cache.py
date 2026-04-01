"""Disk cache for antibody representation embeddings.

Computing transformer-based reps (ESM2, ProtT5, ABlang2) is expensive.
This module caches computed embeddings to disk so they can be reused
across experiments and conditions that share the same sequences.

Cache key = SHA256(sorted sequence hashes + rep_name).
Storage: .npy files in ~/.cache/kirby/antibody_reps/ (or /Volumes/seagate).

Usage:
    from experiments.rep_cache import compute_or_load_reps

    reps = compute_or_load_reps(
        rep_names=['blosum', 'esm2', 'prot_t5'],
        heavy_seqs=data['heavy_seqs'],
        light_seqs=data['light_seqs'],
        mutations=data.get('mutations'),
        dataset_tag='graphinity_foldx_5k',  # for logging
    )
"""

import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def _cache_dir() -> Path:
    """Cache directory, preferring /Volumes/seagate if available."""
    seagate = Path('/Volumes/seagate/kirby/antibody_reps')
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate
    cache = Path.home() / '.cache' / 'kirby' / 'antibody_reps'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _sequence_hash(heavy_seqs: List[str], light_seqs: List[str],
                   mutations: Optional[List] = None) -> str:
    """Hash sequences to create a stable cache key component."""
    h = hashlib.sha256()
    for s in heavy_seqs:
        h.update(s.encode('utf-8'))
    for s in light_seqs:
        h.update(s.encode('utf-8'))
    if mutations:
        for m in mutations:
            h.update(str(m).encode('utf-8'))
    return h.hexdigest()[:16]


def compute_or_load_reps(
    rep_names: List[str],
    heavy_seqs: List[str],
    light_seqs: List[str],
    mutations: Optional[List] = None,
    dataset_tag: str = '',
) -> Dict[str, np.ndarray]:
    """Compute or load cached antibody representations.

    Args:
        rep_names: List of rep names to compute (e.g. ['blosum', 'esm2']).
        heavy_seqs: Heavy chain sequences.
        light_seqs: Light chain sequences.
        mutations: Optional mutation descriptors (for MUTATION_REPS).
        dataset_tag: Human-readable tag for logging.

    Returns:
        Dict of {rep_name: np.ndarray} for successfully computed reps.
    """
    from kirby.representations.antibody import ANTIBODY_REPS
    from kirby.representations.mutation import MUTATION_REPS

    cache = _cache_dir()
    seq_hash = _sequence_hash(heavy_seqs, light_seqs, mutations)
    n = len(heavy_seqs)

    reps = {}
    for rep_name in rep_names:
        rep_fn = ANTIBODY_REPS.get(rep_name) or MUTATION_REPS.get(rep_name)
        if rep_fn is None:
            continue

        cache_file = cache / f"{rep_name}_{seq_hash}.npy"

        if cache_file.exists():
            try:
                X = np.load(cache_file)
                if X.ndim == 2 and X.shape[0] == n:
                    reps[rep_name] = X
                    print(f"  {rep_name}: {X.shape[1]}D (cached)"
                          f"{f' [{dataset_tag}]' if dataset_tag else ''}")
                    continue
                else:
                    print(f"  {rep_name}: cache shape mismatch "
                          f"({X.shape} vs n={n}), recomputing")
            except Exception:
                pass

        # Compute fresh
        try:
            if rep_name in MUTATION_REPS and mutations is not None:
                X = rep_fn(heavy_seqs, light_seqs, mutations)
            elif rep_name in ANTIBODY_REPS:
                X = ANTIBODY_REPS[rep_name](heavy_seqs, light_seqs)
            else:
                # Mutation-only rep but no mutations available
                continue

            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 2 and X.shape[0] == n:
                np.save(cache_file, X)
                reps[rep_name] = X
                print(f"  {rep_name}: {X.shape[1]}D (computed + cached)"
                      f"{f' [{dataset_tag}]' if dataset_tag else ''}")
            else:
                print(f"  {rep_name}: unexpected shape {X.shape}")
        except Exception as e:
            print(f"  {rep_name}: FAILED ({e})")

    return reps
