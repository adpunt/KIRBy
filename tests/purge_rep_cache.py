#!/usr/bin/env python3
"""Targeted, dry-run-first purge of a dataset's cached representations (and,
optionally, its search / baseline / hybrid caches).

WHY THIS EXISTS
---------------
Representations are cached to disk as `<cache>/rep_cache/<dataset>/<rep>.npy`.
If a loader changes the number of rows it returns (e.g. an older cache has
1412 rows but the loader now returns 7772), the stale `.npy` is silently
reused and the features no longer line up with the labels — silent corruption.
`_finalize_load_data` now RAISES on a row-count mismatch and points here; this
script removes the stale cache so the next run recomputes it cleanly.

It can also purge the pickle caches keyed on the dataset (searches / baselines /
hybrid_evals / hybrid_fold_cache) with --all-caches, for cases where those were
built from the stale reps (same class as purge_graphinity_cache.py).

SAFETY
------
- DRY-RUN BY DEFAULT. Prints exactly what it WOULD remove (path + size + why),
  and a total. Nothing is deleted without --delete.
- Run where $KIRBY_CACHE_DIR lives (the server). Review the list, then --delete.

USAGE
-----
  python tests/purge_rep_cache.py herg                      # dry-run, reps only
  python tests/purge_rep_cache.py herg --delete             # remove rep cache
  python tests/purge_rep_cache.py herg logd --all-caches    # also pkl caches (dry-run)
  python tests/purge_rep_cache.py herg --all-caches --delete
  python tests/purge_rep_cache.py herg --cache-dir /path/to/.cache/kirby
"""
import argparse
import os
import pickle
import sys
from pathlib import Path


def _cache_base(override):
    if override:
        return Path(override)
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def _meta_dataset_name(path):
    """Return _cache_meta['dataset_name'] if this pickle carries one, else None.

    Never raises; only peeks at .pkl files, skips unreadable ones.
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


def _human(nbytes):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if nbytes < 1024 or unit == 'GB':
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('datasets', nargs='+',
                    help='Dataset name(s) to purge (e.g. herg logd)')
    ap.add_argument('--cache-dir', default=None,
                    help='Cache root (default: $KIRBY_CACHE_DIR or ~/.cache/kirby)')
    ap.add_argument('--all-caches', action='store_true',
                    help='Also purge search/baseline/hybrid pkl caches for these '
                         'datasets (default: rep_cache only)')
    ap.add_argument('--delete', action='store_true',
                    help='Actually delete (default: dry-run, print only)')
    args = ap.parse_args()

    base = _cache_base(args.cache_dir)
    targets = set(args.datasets)
    if not base.exists():
        print(f"Cache root does not exist: {base}")
        sys.exit(1)

    to_remove = []  # (path, size, reason)

    # 1) rep_cache/<dataset>/  — whole per-dataset directory
    rep_root = base / 'rep_cache'
    for ds in sorted(targets):
        d = rep_root / ds
        if d.is_dir():
            for f in sorted(d.rglob('*')):
                if f.is_file():
                    to_remove.append((f, f.stat().st_size,
                                      f"rep_cache/{ds}"))

    # 2) pkl caches (optional) — match by dataset name in the path OR in
    #    the pickle's _cache_meta['dataset_name'].
    if args.all_caches:
        pkl_dirs = ['searches', 'baselines', 'hybrid_evals',
                    'hybrid_fold_cache']
        for sub in pkl_dirs:
            d = base / sub
            if not d.is_dir():
                continue
            for f in sorted(d.rglob('*')):
                if not f.is_file():
                    continue
                matched = next((ds for ds in targets if ds in f.name), None)
                why = None
                if matched:
                    why = f"{sub} (name~{matched})"
                else:
                    mds = _meta_dataset_name(f)
                    if mds in targets:
                        why = f"{sub} (meta={mds})"
                if why:
                    to_remove.append((f, f.stat().st_size, why))

    if not to_remove:
        print(f"Nothing to purge for {sorted(targets)} under {base}")
        return

    total = sum(sz for _, sz, _ in to_remove)
    print(f"{'DELETING' if args.delete else 'DRY-RUN — would delete'} "
          f"{len(to_remove)} file(s), {_human(total)}, under {base}:\n")
    for path, sz, reason in to_remove:
        print(f"  [{reason}]  {path}  ({_human(sz)})")

    if not args.delete:
        print(f"\nDry-run only. Re-run with --delete to remove the above.")
        return

    removed = 0
    for path, _, _ in to_remove:
        try:
            path.unlink()
            removed += 1
        except Exception as e:
            print(f"  WARNING: failed to remove {path}: {e}")
    # Clean up now-empty rep_cache/<dataset> dirs.
    for ds in sorted(targets):
        d = rep_root / ds
        if d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
            except Exception:
                pass
    print(f"\nRemoved {removed}/{len(to_remove)} file(s).")


if __name__ == '__main__':
    main()
