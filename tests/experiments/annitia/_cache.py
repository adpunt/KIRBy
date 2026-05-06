"""Disk cache for expensive ANNITIA feature builders / importance scores.

Cache key = sha256(repr({key_dict, version})). Stored as
    <cache_dir>/<sha[:2]>/<sha>.pkl

Each entry is a pickle of {'value': obj, 'meta': dict, 'created_at': float}.
The meta dict is the original `key_dict` plus `version` (for invalidation).

The cache is intentionally simple:
    - pickle, not numpy, so the same call site can cache arrays AND lightweight
      dicts (e.g. fitted FPCA basis + projections).
    - atomic write via tempfile rename so partial writes never corrupt entries.
    - no LRU eviction: feature builders are deterministic, the cache is
      append-only, and a stray .pkl is harmless.

Used by:
    - Phase A feature builders (FPCA, GP, shapelets, wavelets) — slow.
    - Phase D.1 importance method registry — TreeSHAP / KernelSHAP per fold.

NOT used by the cheap `views.py` builders (snapshot/slope/etc) — they're fast
enough that disk roundtrip would be a regression.

Usage:
    from experiments.annitia._cache import cached_call, hash_data

    h = hash_data(X)             # stable hash of input data
    out = cached_call(
        key={'data': h, 'kind': 'fpca', 'n_components': 10, 'var': 'alt'},
        fn=lambda: _fit_fpca(X, ...),
        version=1,
    )
"""

import hashlib
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


_DEFAULT_CACHE_DIR = (
    os.environ.get('ANNITIA_CACHE_DIR')
    or str(Path.home() / '.cache' / 'annitia')
)


def _resolve_cache_dir(cache_dir: Optional[str]) -> Path:
    p = Path(cache_dir) if cache_dir is not None else Path(_DEFAULT_CACHE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stable_repr(obj: Any) -> str:
    """A repr that is stable under dict-key ordering and tolerates ndarrays.

    Plain repr() on dicts can vary if keys are inserted in different orders
    (3.7+ preserves order, so a different call site can give a different hash
    for an equivalent payload). This function sorts dict keys recursively and
    summarises ndarrays by their content hash.
    """
    if isinstance(obj, dict):
        return '{' + ','.join(
            f'{_stable_repr(k)}:{_stable_repr(obj[k])}'
            for k in sorted(obj.keys(), key=lambda x: str(x))
        ) + '}'
    if isinstance(obj, (list, tuple)):
        return '[' + ','.join(_stable_repr(v) for v in obj) + ']'
    if isinstance(obj, np.ndarray):
        return f'arr:{hash_data(obj)}'
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return repr(obj)
    # Fallback for arbitrary types — only safe if repr is stable
    return repr(obj)


def hash_data(*arrays: Any) -> str:
    """Stable sha256 over a sequence of values (arrays or other hashable).

    For ndarrays, hashes the dtype + shape + raw bytes. For other types,
    falls back to _stable_repr.
    """
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, np.ndarray):
            h.update(str(a.dtype).encode('utf-8'))
            h.update(str(a.shape).encode('utf-8'))
            # contiguous bytes — copy if not already
            h.update(np.ascontiguousarray(a).tobytes())
        else:
            h.update(_stable_repr(a).encode('utf-8'))
        h.update(b'|')
    return h.hexdigest()


def _key_to_path(cache_dir: Path, key: Dict, version: int) -> Path:
    payload = _stable_repr({'key': key, 'version': version})
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    sub = cache_dir / digest[:2]
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f'{digest}.pkl'


def cached_call(key: Dict,
                fn: Callable[[], Any],
                version: int = 1,
                cache_dir: Optional[str] = None,
                force: bool = False,
                verbose: bool = False) -> Any:
    """Run `fn()` once per unique (key, version), persist result to disk.

    Args:
        key: arbitrary dict identifying this computation (e.g. data hash,
            builder kind, hyperparams). Sorted internally for stability.
        fn: zero-arg callable producing the value to cache.
        version: bump to invalidate prior entries with the same key.
        cache_dir: defaults to $ANNITIA_CACHE_DIR or ~/.cache/annitia.
        force: bypass existing entry, recompute and overwrite.

    Returns the cached or freshly-computed value.
    """
    cache_dir_p = _resolve_cache_dir(cache_dir)
    path = _key_to_path(cache_dir_p, key, version)
    if path.exists() and not force:
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
            if verbose:
                print(f'  [cache HIT] {path.name[:12]}  '
                      f'kind={key.get("kind", "?")}')
            return payload['value']
        except Exception as e:
            if verbose:
                print(f'  [cache CORRUPT] {path.name[:12]} ({e}); recomputing')
    t0 = time.time()
    value = fn()
    elapsed = time.time() - t0
    payload = {'value': value, 'meta': dict(key), 'version': version,
               'created_at': time.time(), 'elapsed_sec': elapsed}
    # Atomic write
    fd, tmp = tempfile.mkstemp(prefix='annitia_cache_', dir=str(cache_dir_p))
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if verbose:
        print(f'  [cache MISS] {path.name[:12]}  '
              f'kind={key.get("kind", "?")}  ({elapsed:.2f}s)')
    return value


def cache_clear(key: Optional[Dict] = None,
                version: Optional[int] = None,
                cache_dir: Optional[str] = None) -> int:
    """Delete cache entries. If `key` is None, deletes the whole cache dir.
    Returns the number of files removed."""
    cache_dir_p = _resolve_cache_dir(cache_dir)
    if key is None:
        n = 0
        for p in cache_dir_p.rglob('*.pkl'):
            p.unlink()
            n += 1
        return n
    path = _key_to_path(cache_dir_p, key, version=version or 1)
    if path.exists():
        path.unlink()
        return 1
    return 0


def cache_info(cache_dir: Optional[str] = None) -> Dict:
    """Quick stats: # entries, total bytes, location."""
    cache_dir_p = _resolve_cache_dir(cache_dir)
    n = 0
    total = 0
    for p in cache_dir_p.rglob('*.pkl'):
        n += 1
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return {'cache_dir': str(cache_dir_p), 'n_entries': n,
            'total_bytes': total}
