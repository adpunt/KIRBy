#!/usr/bin/env python3
"""Targeted, dry-run-first purge of stale Graphinity cache.

WHY THIS EXISTS
---------------
The Graphinity loader had a mutation-position parsing bug (graphinity.py: a
greedy chain regex stole digits from the residue number, e.g. 'YH102A' parsed
as chain 'H10' / position 2). That corrupts the *mutant sequence*, so EVERY
representation computed from it (not just mutation-aware ones) is wrong.

The bug is now fixed in the loader. BUT the CACHE-1 anti-churn fix makes the
search/baseline cache IGNORE rep content hashes, and the corrected embeddings
have the SAME shape / labels / n_samples as the corrupt ones — so the cache
meta-validation passes and the corrupt cache would be SILENTLY REUSED. The
normal auto-invalidation (rep_shape / labels_hash / n_samples) cannot catch a
values-only change. => the Graphinity cache must be purged manually, once.

WHAT IT TOUCHES
---------------
Only cache entries belonging to the Graphinity datasets (default:
graphinity_foldx, graphinity_rosetta). It matches in two ways:
  (1) the dataset name appears in the file PATH (searches/baselines/hybrid_evals
      /folds all embed dataset_name in the filename), and
  (2) for any pickle that carries a '_cache_meta' dict, dataset_name in the
      target set (covers content-hashed rep caches under reps/).

SAFETY
------
- DRY-RUN BY DEFAULT. It prints exactly what it WOULD delete (path + size +
  why matched) and a total. Nothing is removed without --delete.
- Run on the SERVER (where $KIRBY_CACHE_DIR lives). Review the printed list,
  then re-run with --delete to remove.

USAGE (on the server)
---------------------
  python tests/purge_graphinity_cache.py                 # dry-run, see the list
  python tests/purge_graphinity_cache.py --delete        # after reviewing
  python tests/purge_graphinity_cache.py --targets graphinity_foldx
  python tests/purge_graphinity_cache.py --cache-dir /path/to/.cache/kirby
"""
import argparse
import os
import pickle
import sys
from pathlib import Path


def _cache_base(override: str | None) -> Path:
    if override:
        return Path(override)
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def _meta_dataset_name(path: Path):
    """Return _cache_meta['dataset_name'] if this pickle carries one, else None.

    Guarded: never raises; only peeks at pickles, skips .npy and unreadable
    files. This is how content-hashed rep caches (whose FILENAME has no dataset
    name) are matched.
    """
    if path.suffix != '.pkl':
        return None
    try:
        with open(path, 'rb') as fh:
            obj = pickle.load(fh)
    except Exception:
        return None
    if isinstance(obj, dict):
        meta = obj.get('_cache_meta')
        if isinstance(meta, dict):
            return meta.get('dataset_name')
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', default=None,
                    help='Cache root (default: $KIRBY_CACHE_DIR or ~/.cache/kirby)')
    ap.add_argument('--targets', nargs='+',
                    default=['graphinity_foldx', 'graphinity_rosetta'],
                    help='Dataset names whose cache to purge')
    ap.add_argument('--delete', action='store_true',
                    help='Actually delete (default: dry-run, list only)')
    ap.add_argument('--inspect-meta', action='store_true', default=True,
                    help='Also peek at pickle _cache_meta to match content-hashed '
                         'rep caches (default on; slower on big caches)')
    ap.add_argument('--no-inspect-meta', dest='inspect_meta', action='store_false')
    args = ap.parse_args()

    base = _cache_base(args.cache_dir)
    if not base.exists():
        print(f"Cache dir not found: {base}")
        sys.exit(1)

    targets = set(args.targets)
    print(f"Cache root : {base}")
    print(f"Targets    : {sorted(targets)}")
    print(f"Mode       : {'DELETE' if args.delete else 'DRY-RUN (no deletion)'}")
    print("-" * 70)

    matches = []  # (path, size_bytes, reason)
    n_scanned = 0
    for path in base.rglob('*'):
        if not path.is_file():
            continue
        n_scanned += 1
        name = path.name
        # (1) dataset name in the filename/path
        hit = next((t for t in targets if t in str(path)), None)
        reason = None
        if hit:
            reason = f'path~{hit}'
        elif args.inspect_meta:
            # (2) dataset name in the pickle's _cache_meta
            ds = _meta_dataset_name(path)
            if ds in targets:
                reason = f'meta.dataset_name={ds}'
        if reason:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            matches.append((path, size, reason))

    total = sum(s for _, s, _ in matches)
    for path, size, reason in sorted(matches):
        rel = path.relative_to(base)
        print(f"  {size/1e6:8.2f} MB  {rel}   [{reason}]")
    print("-" * 70)
    print(f"Scanned {n_scanned} files; {len(matches)} match; "
          f"total {total/1e6:.1f} MB")

    if not matches:
        print("Nothing to purge.")
        return
    if not args.delete:
        print("\nDRY-RUN. Review the list above, then re-run with --delete to remove.")
        return

    removed = 0
    for path, _, _ in matches:
        try:
            path.unlink()
            removed += 1
        except OSError as e:
            print(f"  FAILED to delete {path}: {e}")
    print(f"\nDeleted {removed}/{len(matches)} files. "
          f"Graphinity will recompute cleanly on the next run.")


if __name__ == '__main__':
    main()
