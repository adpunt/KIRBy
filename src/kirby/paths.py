"""Where kirby reads and writes things on disk.

ONE place, so no module names a particular machine's hardware. Every path below
comes from an environment variable with a portable default; nothing is guessed
from a mount point that exists on one laptop.

  KIRBY_CACHE_DIR   everything kirby caches: downloaded datasets, computed
                    representations, structures, the result databases.
                    Default ~/.cache/kirby.
  KIRBY_DATA_DIR    downloaded datasets and structure caches only, when those
                    belong somewhere other than the rest of the cache — a
                    large external volume, a scratch filesystem, a shared
                    read-only mount. Defaults to KIRBY_CACHE_DIR.
  KIRBY_MODELS_DIR  pretrained model weights. Default ~/kirby_models.

`extra_search_paths` exists for the loaders that will use a prepared copy of a
dataset if one is present and download it otherwise. Give it the name of an
environment variable holding colon-separated directories, highest priority
first; it returns the ones that exist.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def cache_root() -> Path:
    """The root under which kirby caches everything."""
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def data_root() -> Path:
    """The root for downloaded datasets and structure caches."""
    explicit = os.environ.get('KIRBY_DATA_DIR')
    return Path(explicit) if explicit else cache_root()


def data_dir(*parts: str, create: bool = True) -> Path:
    """A named directory under the data root, created unless create=False."""
    p = data_root().joinpath(*parts)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(*parts: str, create: bool = True) -> Path:
    """A named directory under the cache root, created unless create=False."""
    p = cache_root().joinpath(*parts)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    """Where pretrained model weights live."""
    explicit = os.environ.get('KIRBY_MODELS_DIR')
    if explicit:
        return Path(explicit)
    return Path.home() / 'kirby_models'


def extra_search_paths(env_var: str) -> List[Path]:
    """Directories named by `env_var`, colon-separated, that exist.

    For loaders that prefer an already-prepared copy of a dataset when one is
    reachable and fall back to downloading it. Empty when the variable is unset,
    which is the portable default.
    """
    raw = os.environ.get(env_var, '')
    out = []
    for chunk in raw.split(os.pathsep):
        chunk = chunk.strip()
        if not chunk:
            continue
        p = Path(chunk)
        if p.exists():
            out.append(p)
    return out
