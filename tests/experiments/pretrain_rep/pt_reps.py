"""PT-Rep Phase 0 — base-rep computation for PT corpora.

The registry (`pt_registry`) hands back raw entities. `find_best_hybrid` needs
`{rep_name: matrix}`. This module is the bridge.

Key fact (confirmed by reading `_shared.load_data`): the representation
functions are NOT bound to a named dataset. Antibody reps are a dict of
`name -> fn(heavy_seqs, light_seqs)`; molecular reps come from
`get_molecular_representations(smiles, preset)`. The only thing tied to a
"dataset name" is the on-disk cache folder. So computing reps for a PT corpus
is just: call the same rep functions on the PT entity list, and cache under a
PT-specific key so PT reps never collide with FT reps.

`compute_pt_reps(pt_ds, rep_names)` does exactly that, reusing `_shared`'s
`_cache_rep` / `_cache_rep_dict` so the cache format stays identical to the
rest of the codebase.

Cache key
---------
`pt_cache_key(pt_ds)` encodes source + task + a content hash of the *ordered*
dedup keys. Any change to the corpus — size cap, quality filter, leakage
removal — changes the hash, so a stale cache can never be silently reused
(see the cache-collision incident in project memory).

The heavy foundation-model reps (ESM, AntiBERTy, ...) run on the server exactly
as they do for FT datasets — same functions, same compute. This module adds no
new modelling code, only the corpus→reps plumbing.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Optional

import numpy as np

from experiments.pretrain_rep.pt_registry import PTDataset


def pt_cache_key(pt_ds: PTDataset) -> str:
    """Stable, content-sensitive cache-folder name for a PT corpus's reps.

    Format: ``pt__{source}__{task}__n{N}__{hash10}``. The hash covers the
    ordered dedup keys, so two corpora that differ by even one row (or row
    order) get distinct keys — no stale-cache reuse across cap / quality /
    leakage changes.
    """
    h = hashlib.sha1('\x00'.join(pt_ds.dedup_keys).encode('utf-8')).hexdigest()
    task = (pt_ds.task or 'default').replace('/', '_')
    return f"pt__{pt_ds.source}__{task}__n{len(pt_ds)}__{h[:10]}"


def _compute_with_fns(
    pt_ds: PTDataset,
    rep_fns: Dict[str, Callable[[PTDataset], np.ndarray]],
    cache_key: str,
    cache: bool,
    verbose: bool,
) -> Dict[str, np.ndarray]:
    """Run a uniform ``{name: fn(pt_ds) -> matrix}`` mapping, optionally cached.

    This is the single cache path; the domain branches below just build the
    appropriate `rep_fns` dict and delegate here.
    """
    reps: Dict[str, np.ndarray] = {}
    cache_rep = None
    if cache:
        from experiments._shared import _cache_rep
        cache_rep = _cache_rep

    for rn, fn in rep_fns.items():
        try:
            if cache:
                arr = cache_rep(cache_key, rn, lambda _fn=fn: _fn(pt_ds))
            else:
                arr = fn(pt_ds)
        except Exception as e:
            if verbose:
                print(f"  WARNING: PT rep '{rn}' failed: {e}")
            continue
        arr = np.asarray(arr)
        if arr.shape[0] != len(pt_ds):
            raise ValueError(
                f"PT rep '{rn}' returned {arr.shape[0]} rows, "
                f"expected {len(pt_ds)}")
        reps[rn] = arr
        if verbose:
            print(f"    {rn}: {arr.shape[1]}D")
    return reps


def _antibody_rep_fns(
    rep_names: List[str], verbose: bool,
) -> Dict[str, Callable[[PTDataset], np.ndarray]]:
    """Resolve antibody rep names to `fn(pt_ds) -> matrix` callables.

    Reuses `kirby.representations.antibody.ANTIBODY_REPS` — the same functions
    `load_data` uses for FT antibody datasets. Mutation-aware reps are skipped:
    raw PT corpora carry no mutation metadata.
    """
    from kirby.representations.antibody import ANTIBODY_REPS
    try:
        from kirby.representations.mutation import MUTATION_REPS
    except Exception:
        MUTATION_REPS = {}

    fns: Dict[str, Callable[[PTDataset], np.ndarray]] = {}
    for rn in rep_names:
        if rn in ANTIBODY_REPS:
            base = ANTIBODY_REPS[rn]
            fns[rn] = (lambda pt_ds, _f=base:
                       _f(pt_ds.heavy_seqs, pt_ds.light_seqs))
        elif rn in MUTATION_REPS:
            if verbose:
                print(f"  WARNING: PT rep '{rn}' is mutation-aware; PT corpora "
                      f"have no mutation metadata — skipping.")
        else:
            if verbose:
                print(f"  WARNING: unknown antibody rep '{rn}' — skipping.")
    return fns


def _compute_molecular(
    pt_ds: PTDataset,
    rep_names: List[str],
    rep_preset: str,
    cache_key: str,
    cache: bool,
    verbose: bool,
) -> Dict[str, np.ndarray]:
    """Molecular reps via `get_molecular_representations` (bulk function).

    Unlike antibody (per-rep functions), the molecular pipeline computes a
    preset's reps in one call, so this mirrors `load_data`'s `_cache_rep_dict`
    usage rather than the per-rep loop.
    """
    from find_best_hybrid_molecular import get_representations as get_mol_reps

    if cache:
        from experiments._shared import _cache_rep_dict
        reps = _cache_rep_dict(
            cache_key,
            lambda: get_mol_reps(pt_ds.smiles, rep_preset),
            filter_names=rep_names or None,
        )
    else:
        reps = get_mol_reps(pt_ds.smiles, rep_preset)
        if rep_names:
            reps = {k: v for k, v in reps.items() if k in rep_names}

    for rn, arr in reps.items():
        arr = np.asarray(arr)
        if arr.shape[0] != len(pt_ds):
            raise ValueError(
                f"PT rep '{rn}' returned {arr.shape[0]} rows, "
                f"expected {len(pt_ds)}")
        if verbose:
            print(f"    {rn}: {arr.shape[1]}D")
    return reps


def compute_pt_reps(
    pt_ds: PTDataset,
    rep_names: List[str],
    *,
    rep_preset: str = 'full',
    cache: bool = True,
    cache_key: Optional[str] = None,
    rep_fns: Optional[Dict[str, Callable[[PTDataset], np.ndarray]]] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute base representations for a PT corpus.

    Args:
        pt_ds: A `PTDataset` from `pt_registry.get_pt_dataset`.
        rep_names: Which base reps to compute (names from the domain's rep
            registry — `ANTIBODY_REPS` for antibody, the molecular preset's
            generators for molecular).
        rep_preset: Molecular-only — preset passed to
            `get_molecular_representations` (ignored for antibody).
        cache: If True, results are cached under `cache_key` via `_shared`'s
            `_cache_rep` so PT and FT reps share one cache format. PT reps live
            under a PT-specific key and never overwrite FT reps.
        cache_key: Override the cache folder name. Default: `pt_cache_key(pt_ds)`.
        rep_fns: Test / override hook — a `{name: fn(pt_ds) -> matrix}` mapping
            used instead of the real domain rep registry. The real
            foundation-model reps need a GPU and on-disk model weights, so the
            offline smoke test injects trivial fns here.
        verbose: Print per-rep dims and warnings.

    Returns:
        `{rep_name: np.ndarray}` — one matrix per successfully computed rep,
        each with `len(pt_ds)` rows.
    """
    key = cache_key or pt_cache_key(pt_ds)
    if verbose:
        print(f"  PT reps for {pt_ds.source} (n={len(pt_ds)}) -> cache key {key}")

    if rep_fns is not None:
        return _compute_with_fns(pt_ds, rep_fns, key, cache, verbose)

    if pt_ds.domain == 'antibody':
        fns = _antibody_rep_fns(list(rep_names), verbose)
        return _compute_with_fns(pt_ds, fns, key, cache, verbose)

    if pt_ds.domain == 'molecular':
        return _compute_molecular(
            pt_ds, list(rep_names), rep_preset, key, cache, verbose)

    raise ValueError(
        f"compute_pt_reps has no rep pipeline for domain '{pt_ds.domain}'")
