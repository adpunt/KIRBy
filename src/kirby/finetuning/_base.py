"""
Base utilities for fine-tuning integration.

Provides caching, device detection, and fold-level cache keys.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np


def get_device() -> str:
    """Detect best available device: CUDA > MPS > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def get_cache_dir() -> Path:
    """Return the cache directory for fine-tuned representations.

    Respects KIRBY_CACHE_DIR env var (server: /data/.../kirby).
    Falls back to ~/.cache/kirby/finetuned_reps.
    """
    base = os.environ.get('KIRBY_CACHE_DIR')
    if base:
        d = Path(base) / 'finetuned_reps'
    else:
        d = Path.home() / '.cache' / 'kirby' / 'finetuned_reps'
    d.mkdir(parents=True, exist_ok=True)
    return d


def fold_cache_key(
    dataset_name: str,
    rep_name: str,
    train_idx: np.ndarray,
    hyperparams: dict,
) -> str:
    """Deterministic SHA256 hash for a (dataset, rep, fold, hyperparams) combo.

    Changing any hyperparameter or fold split produces a new cache entry.
    """
    h = hashlib.sha256()
    h.update(dataset_name.encode())
    h.update(rep_name.encode())
    h.update(train_idx.tobytes())
    # Sort hyperparams for determinism
    for k in sorted(hyperparams.keys()):
        h.update(f"{k}={hyperparams[k]}".encode())
    return h.hexdigest()


def load_or_compute(
    cache_key: str,
    fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached fold embeddings or compute and cache them.

    Args:
        cache_key: SHA256 hash from fold_cache_key().
        fn: Callable that returns (train_embeddings, val_embeddings).
        cache: Whether to use caching.

    Returns:
        (train_embs, val_embs) as numpy arrays.
    """
    if cache:
        cache_path = get_cache_dir() / f'{cache_key}.npz'
        if cache_path.exists():
            data = np.load(cache_path)
            return data['train'], data['val']

    train_embs, val_embs = fn()

    if cache:
        cache_path = get_cache_dir() / f'{cache_key}.npz'
        np.savez(cache_path, train=train_embs, val=val_embs)

    return train_embs, val_embs


def mock_factory(
    n_features: int = 64,
    seed: int = 42,
) -> Callable:
    """Create a mock factory for testing plumbing without a GPU.

    Returns random embeddings of the specified dimension.
    """
    def factory(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(seed)
        train_embs = rng.randn(len(train_idx), n_features).astype(np.float32)
        val_embs = rng.randn(len(val_idx), n_features).astype(np.float32)
        return train_embs, val_embs

    return factory
