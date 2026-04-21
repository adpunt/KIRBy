"""Find (and optionally delete) hybrid_eval cache files corrupted by the
aug_strategy collision bug fixed on 2026-04-21 (commits a13e0f4 + 390f607).

The bug: experiments called `find_best_hybrid(augmentations=None)` without
passing `augmentation_strategy`. Default was `'uncapped'`, so the no-aug
result was written to `*__uncapped.pkl` with `_cache_meta['aug_names']=()`.
Real uncapped runs write the same filename with `aug_names=('paratope',
'structural')` (or similar). Filename collision.

Corruption signature:
  - filename ends with `__uncapped.pkl`
  - AND `_cache_meta['aug_strategy'] == 'uncapped'`
  - AND `_cache_meta['aug_names'] == ()` (empty tuple)

Any file matching all three is the no-aug-result-mislabeled-as-uncapped bug.

Usage:
  # Step 1: scan and write candidate list (no deletion)
  python tests/scripts/find_corrupt_hybrid_evals.py --scan \
      --out corrupt_files.txt

  # Step 2: review the list
  wc -l corrupt_files.txt
  head corrupt_files.txt

  # Step 3: delete files in the list (reads from --in)
  python tests/scripts/find_corrupt_hybrid_evals.py --delete \
      --in corrupt_files.txt
"""
import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Optional


def cache_root() -> Path:
    return Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')
    )) / 'hybrid_evals'


def inspect_file(path: Path):
    """Return one of: 'corrupt', 'healthy', 'unreadable', 'not_uncapped'."""
    if not path.name.endswith('__uncapped.pkl'):
        return 'not_uncapped', None
    try:
        with open(path, 'rb') as f:
            wrapper = pickle.load(f)
    except Exception as e:
        return 'unreadable', str(e)
    if not isinstance(wrapper, dict) or '_cache_meta' not in wrapper:
        return 'unreadable', 'no _cache_meta'
    meta = wrapper['_cache_meta']
    aug_strategy = meta.get('aug_strategy')
    aug_names = meta.get('aug_names', ())
    if aug_strategy == 'uncapped' and tuple(aug_names) == ():
        return 'corrupt', meta
    return 'healthy', meta


def scan(out_path: Path, dataset_filter: Optional[str]):
    root = cache_root()
    if not root.exists():
        print(f'ERROR: cache root not found: {root}', file=sys.stderr)
        sys.exit(1)

    totals = {'corrupt': 0, 'healthy': 0, 'unreadable': 0, 'not_uncapped': 0}
    per_dataset = {}
    corrupt_files = []
    unreadable_files = []

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_filter and dataset_dir.name != dataset_filter:
            continue
        d_counts = {'corrupt': 0, 'healthy': 0, 'unreadable': 0}
        for f in dataset_dir.glob('*__uncapped.pkl'):
            status, info = inspect_file(f)
            totals[status] = totals.get(status, 0) + 1
            if status in d_counts:
                d_counts[status] += 1
            if status == 'corrupt':
                corrupt_files.append(f)
            elif status == 'unreadable':
                unreadable_files.append((f, info))
        per_dataset[dataset_dir.name] = d_counts

    print('Per-dataset (uncapped files only):')
    print(f'  {"dataset":<40} {"corrupt":>8} {"healthy":>8} {"unreadable":>12}')
    for name in sorted(per_dataset):
        c = per_dataset[name]
        if c['corrupt'] == 0 and c['healthy'] == 0 and c['unreadable'] == 0:
            continue
        print(f'  {name:<40} {c["corrupt"]:>8} {c["healthy"]:>8} '
              f'{c["unreadable"]:>12}')

    print('')
    print(f'Total corrupt:    {totals["corrupt"]}')
    print(f'Total healthy:    {totals["healthy"]}')
    print(f'Total unreadable: {totals["unreadable"]}')

    with open(out_path, 'w') as f:
        for p in corrupt_files:
            f.write(str(p) + '\n')
    print(f'\nWrote {len(corrupt_files)} corrupt-file paths to {out_path}')

    if unreadable_files:
        err_path = out_path.with_suffix('.unreadable.txt')
        with open(err_path, 'w') as f:
            for p, err in unreadable_files:
                f.write(f'{p}\t{err}\n')
        print(f'Wrote {len(unreadable_files)} unreadable paths to {err_path}')
        print('NOTE: unreadable files are NOT deleted — inspect manually.')


def delete(in_path: Path):
    if not in_path.exists():
        print(f'ERROR: input list not found: {in_path}', file=sys.stderr)
        sys.exit(1)

    paths = [Path(line.strip()) for line in in_path.read_text().splitlines()
             if line.strip()]
    print(f'Will delete {len(paths)} files listed in {in_path}')

    # Re-verify each path before deleting. Cheap safety net.
    confirmed = 0
    skipped = []
    for p in paths:
        if not p.exists():
            skipped.append((p, 'missing'))
            continue
        status, _ = inspect_file(p)
        if status != 'corrupt':
            skipped.append((p, f'status={status}'))
            continue
        try:
            p.unlink()
            confirmed += 1
        except Exception as e:
            skipped.append((p, f'unlink failed: {e}'))

    print(f'Deleted: {confirmed}')
    print(f'Skipped: {len(skipped)}')
    for p, reason in skipped[:20]:
        print(f'  {p}: {reason}')
    if len(skipped) > 20:
        print(f'  ... and {len(skipped) - 20} more')


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--scan', action='store_true',
                   help='Inspect cache, write corrupt paths to --out')
    g.add_argument('--delete', action='store_true',
                   help='Delete files listed in --in (re-verifies each)')
    ap.add_argument('--out', type=Path, default=Path('corrupt_files.txt'),
                    help='Output path for --scan (default corrupt_files.txt)')
    ap.add_argument('--in', dest='in_path', type=Path,
                    default=Path('corrupt_files.txt'),
                    help='Input list for --delete (default corrupt_files.txt)')
    ap.add_argument('--dataset', default=None,
                    help='Restrict scan to one dataset (subdir name)')
    args = ap.parse_args()

    if args.scan:
        scan(args.out, args.dataset)
    else:
        delete(args.in_path)


if __name__ == '__main__':
    main()
