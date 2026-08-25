"""Raw ingredient blocks for the similarity recipes, with on-disk caching.

A recipe is built by gluing together up to four blocks of numbers, one row per
molecule. This module builds those blocks and caches them, so a recipe that
needs 3D conformers doesn't regenerate them on every run.

  count          ecfp4 counting fingerprint, 1024 columns
  desc           the 208 raw RDKit descriptors
  shape          pharmacophore fingerprint from a 3D conformer, 49 columns
  fine           [atompair counts (1024) | torsion bits (1024)], 2048 columns

Cache layout deliberately matches `find_best_hybrid_molecular._rep_cache_path`
(`$KIRBY_CACHE_DIR/reps/<smiles-hash>/<name>.npy`), so the `shape` block reuses
the framework's existing `pharmacophore3d.npy` file when one has already been
built for the same molecules instead of regenerating conformers.

Nothing here reads activity labels, so caching a block over all rows is not
leakage — the leakage-sensitive step is *column selection*, which happens
per fold in `selection.py` / `recipes.py`.
"""
import hashlib
import os
from pathlib import Path

import numpy as np


def _reps_cache_dir(smiles_list):
    """Same directory the framework's own rep cache uses, for the same molecules."""
    h = hashlib.sha256('\n'.join(smiles_list).encode()).hexdigest()[:16]
    base = Path(os.environ.get('KIRBY_CACHE_DIR', Path.home() / '.cache' / 'kirby'))
    d = base / 'reps' / h
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cached(smiles_list, name, build_fn, use_cache=True):
    """Load `name` for these molecules from disk, else build it and store it."""
    if not use_cache:
        return np.asarray(build_fn(smiles_list), dtype=np.float64)
    path = _reps_cache_dir(smiles_list) / f'{name}.npy'
    if path.exists():
        try:
            return np.asarray(np.load(path), dtype=np.float64)
        except Exception:
            pass  # corrupt/partial file -> rebuild over it
    X = np.asarray(build_fn(smiles_list), dtype=np.float64)
    # Temp name must itself end in .npy — np.save silently appends the extension
    # otherwise, and the rename would then target a file that was never written.
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npy')
    np.save(tmp, X)
    os.replace(tmp, path)
    return X


def get_block(kind, smiles_list, use_cache=True):
    """Return the raw (uncentred, unselected) matrix for one ingredient block.

    Args:
        kind: 'count', 'desc', 'shape' or 'fine'.
        smiles_list: the molecules, in the dataset's row order.
        use_cache: read/write the on-disk cache (default True).

    Returns:
        (n_molecules, n_columns) float64 array.
    """
    from kirby.representations.molecular import (
        create_ecfp4_count, create_desc208, create_pharmacophore3d,
        create_atompair_count, create_top_torsion,
    )
    smiles_list = list(smiles_list)

    if kind == 'count':
        return _cached(smiles_list, 'ecfp4_count', create_ecfp4_count, use_cache)
    if kind == 'desc':
        return _cached(smiles_list, 'desc208', create_desc208, use_cache)
    if kind == 'shape':
        # Shares the framework's cache file: same builder, same molecules.
        return _cached(smiles_list, 'pharmacophore3d', create_pharmacophore3d,
                       use_cache)
    if kind == 'fine':
        ap = _cached(smiles_list, 'atompair_count', create_atompair_count, use_cache)
        # Torsion is forced to 1024 bits to match the recipe's block geometry;
        # create_top_torsion defaults to 2048, so this gets its own cache name.
        to = _cached(smiles_list, 'top_torsion_nb1024',
                     lambda s: create_top_torsion(s, n_bits=1024), use_cache)
        return np.hstack([ap, to])
    raise ValueError(f"unknown block kind: {kind!r}")
