#!/usr/bin/env python3
"""Selectively clear cache entries affected by the top_k auto-demote change.

Two caches are checked:

1. hybrid_search/ — SearchResult-level cache from find_best_hybrid(cache=True).
   Deletes entries where ANY result used a top_k allocation (top_2/top_3/top_5)
   AND augmentations were active.

2. hybrid_fold_cache/ — Per-fold feature_info from create_hybrid_cached().
   Deletes entries where the allocation gave any rep 0 features AND
   augmentation_info is present (meaning augmentations were active but the
   zeroed reps were NOT demoted under old behavior).

Usage:
    python scripts/clear_stale_cache.py              # dry run (default)
    python scripts/clear_stale_cache.py --delete     # actually delete
    python scripts/clear_stale_cache.py --cache-dir /path/to/cache  # custom dir
"""

import argparse
import os
import pickle
import sys
from pathlib import Path


TOP_K_METHODS = {'top_2', 'top_3', 'top_5'}


def check_search_cache(cache_dir, delete=False):
    """Check hybrid_search/ for stale SearchResult entries."""
    search_dir = cache_dir / 'hybrid_search'
    if not search_dir.exists():
        print(f"  {search_dir}: not found, skipping")
        return 0, 0

    files = list(search_dir.glob('*.pkl'))
    print(f"  {search_dir}: {len(files)} entries")

    stale = 0
    kept = 0
    for f in files:
        try:
            result = pickle.loads(f.read_bytes())
            # SearchResult has all_results list
            all_results = getattr(result, 'all_results', None)
            if all_results is None:
                kept += 1
                continue

            has_top_k = any(
                r.get('allocation_method') in TOP_K_METHODS
                for r in all_results
            )
            has_augmentation = any(
                r.get('blueprint') and
                getattr(r['blueprint'], 'augmentation_strategy', 'none') != 'none'
                for r in all_results
            )

            if has_top_k and has_augmentation:
                stale += 1
                if delete:
                    f.unlink()
                    print(f"    DELETED: {f.name}")
                else:
                    print(f"    STALE:   {f.name} (top_k + augmentation)")
            else:
                kept += 1
        except Exception as e:
            print(f"    ERROR:   {f.name} ({e})")
            kept += 1

    return stale, kept


def check_fold_cache(cache_dir, delete=False):
    """Check hybrid_fold_cache/ for stale feature_info entries."""
    fold_dir = cache_dir / 'hybrid_fold_cache'
    if not fold_dir.exists():
        print(f"  {fold_dir}: not found, skipping")
        return 0, 0

    files = list(fold_dir.glob('*.pkl'))
    print(f"  {fold_dir}: {len(files)} entries")

    stale = 0
    kept = 0
    for f in files:
        try:
            feature_info = pickle.loads(f.read_bytes())
            if not isinstance(feature_info, dict):
                kept += 1
                continue

            # Check if any rep got 0 features in allocation
            allocation = feature_info.get('allocation', {})
            has_zeroed_rep = any(v == 0 for v in allocation.values())

            # Check if augmentation was active
            has_aug = bool(feature_info.get('augmentation_info'))
            has_kept_aug = bool(feature_info.get('kept_augmentations'))

            if has_zeroed_rep and (has_aug or has_kept_aug):
                stale += 1
                if delete:
                    f.unlink()
                    print(f"    DELETED: {f.name}")
                else:
                    zeroed = [k for k, v in allocation.items() if v == 0]
                    print(f"    STALE:   {f.name} (zeroed: {zeroed})")
            else:
                kept += 1
        except Exception as e:
            print(f"    ERROR:   {f.name} ({e})")
            kept += 1

    return stale, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--delete', action='store_true',
                        help='Actually delete stale entries (default: dry run)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Cache root (default: ~/.cache/kirby or KIRBY_CACHE_DIR)')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(
        os.environ.get('KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))

    mode = "DELETE" if args.delete else "DRY RUN"
    print(f"Cache root: {cache_dir}")
    print(f"Mode: {mode}\n")

    print("--- hybrid_search/ (SearchResult level) ---")
    s1_stale, s1_kept = check_search_cache(cache_dir, args.delete)

    print(f"\n--- hybrid_fold_cache/ (per-fold feature_info) ---")
    s2_stale, s2_kept = check_fold_cache(cache_dir, args.delete)

    total_stale = s1_stale + s2_stale
    total_kept = s1_kept + s2_kept
    print(f"\nSummary: {total_stale} stale, {total_kept} kept")
    if not args.delete and total_stale > 0:
        print(f"Run with --delete to remove stale entries.")


if __name__ == '__main__':
    main()
