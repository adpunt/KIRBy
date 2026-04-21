"""Find (and optionally delete) hybrid_eval cache files corrupted by the
aug_strategy collision bug fixed on 2026-04-21 (commits a13e0f4 + 390f607).

The bug: callers passed augmentations=None without setting
augmentation_strategy. Default was 'uncapped', so no-aug results landed in
*__uncapped.pkl with _cache_meta['aug_names']=(). Real uncapped runs write
the same filename with aug_names=('paratope','structural') (or similar).
Filename collision.

Corruption signature (all three must hold):
  1. filename ends with __uncapped.pkl
  2. _cache_meta['aug_strategy'] == 'uncapped'
  3. _cache_meta['aug_names'] == ()

Usage:
  # Step 1: scan (writes corrupt paths to --out, does NOT delete)
  python tests/scripts/find_corrupt_hybrid_evals.py --scan --out corrupt_files.txt

  # Step 2: review
  wc -l corrupt_files.txt
  head corrupt_files.txt

  # Step 3: delete (re-verifies each file before unlinking)
  python tests/scripts/find_corrupt_hybrid_evals.py --delete --in corrupt_files.txt

Flags:
  --workers N    parallel worker processes (default: CPU count)
  --dataset D    restrict to one dataset subdir
  --progress N   print progress every N files (default 500)
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


def cache_root() -> Path:
    return Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')
    )) / 'hybrid_evals'


def inspect_file(path_str: str) -> Tuple[str, str, str]:
    """Worker function. Returns (path, status, info_str).

    status ∈ {'corrupt', 'healthy', 'unreadable'}
    """
    path = Path(path_str)
    try:
        with open(path, 'rb') as f:
            wrapper = pickle.load(f)
    except Exception as e:
        return (path_str, 'unreadable', f'{type(e).__name__}: {e}')
    if not isinstance(wrapper, dict) or '_cache_meta' not in wrapper:
        return (path_str, 'unreadable', 'no _cache_meta key')
    meta = wrapper['_cache_meta']
    aug_strategy = meta.get('aug_strategy')
    aug_names = meta.get('aug_names', ())
    if aug_strategy == 'uncapped' and tuple(aug_names) == ():
        return (path_str, 'corrupt', '')
    return (path_str, 'healthy', '')


def collect_files(root: Path, dataset_filter: Optional[str]) -> List[Path]:
    files = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_filter and dataset_dir.name != dataset_filter:
            continue
        files.extend(dataset_dir.glob('*__uncapped.pkl'))
    return files


def scan(out_path: Path, dataset_filter: Optional[str],
         workers: int, progress_every: int):
    root = cache_root()
    if not root.exists():
        print(f'ERROR: cache root not found: {root}', file=sys.stderr)
        sys.exit(1)

    print(f'Cache root: {root}', flush=True)
    print('Enumerating files...', flush=True)
    t0 = time.time()
    files = collect_files(root, dataset_filter)
    print(f'Found {len(files)} *__uncapped.pkl files in '
          f'{time.time() - t0:.1f}s', flush=True)
    if not files:
        return

    print(f'Inspecting with {workers} workers '
          f'(progress every {progress_every} files)...', flush=True)

    corrupt_files: List[str] = []
    unreadable_files: List[Tuple[str, str]] = []
    per_dataset = {}
    done = 0
    t_start = time.time()

    path_strs = [str(p) for p in files]

    if workers <= 1:
        iterator = (inspect_file(p) for p in path_strs)
    else:
        pool = mp.Pool(workers)
        iterator = pool.imap_unordered(inspect_file, path_strs, chunksize=50)

    try:
        for path_str, status, info in iterator:
            done += 1
            ds_name = Path(path_str).parent.name
            pd = per_dataset.setdefault(
                ds_name, {'corrupt': 0, 'healthy': 0, 'unreadable': 0})
            pd[status] = pd.get(status, 0) + 1
            if status == 'corrupt':
                corrupt_files.append(path_str)
            elif status == 'unreadable':
                unreadable_files.append((path_str, info))
            if done % progress_every == 0 or done == len(files):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(files) - done) / rate if rate > 0 else 0
                print(f'  [{done}/{len(files)}] '
                      f'corrupt={len(corrupt_files)} '
                      f'unreadable={len(unreadable_files)} '
                      f'elapsed={elapsed:.0f}s eta={eta:.0f}s',
                      flush=True)
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    print('', flush=True)
    print('Per-dataset (uncapped files only):')
    print(f'  {"dataset":<40} {"corrupt":>8} {"healthy":>8} '
          f'{"unreadable":>12}')
    for name in sorted(per_dataset):
        c = per_dataset[name]
        print(f'  {name:<40} {c.get("corrupt",0):>8} '
              f'{c.get("healthy",0):>8} {c.get("unreadable",0):>12}')

    print('', flush=True)
    print(f'Total corrupt:    {len(corrupt_files)}')
    print(f'Total healthy:    {sum(c.get("healthy", 0) for c in per_dataset.values())}')
    print(f'Total unreadable: {len(unreadable_files)}')

    with open(out_path, 'w') as f:
        for p in corrupt_files:
            f.write(p + '\n')
    print(f'Wrote {len(corrupt_files)} corrupt paths to {out_path}')

    if unreadable_files:
        err_path = out_path.with_suffix('.unreadable.txt')
        with open(err_path, 'w') as f:
            for p, err in unreadable_files:
                f.write(f'{p}\t{err}\n')
        print(f'Wrote {len(unreadable_files)} unreadable paths to {err_path}')
        print('NOTE: unreadable files are NOT deleted — inspect manually.')


def delete(in_path: Path, workers: int, progress_every: int):
    if not in_path.exists():
        print(f'ERROR: input list not found: {in_path}', file=sys.stderr)
        sys.exit(1)

    paths = [line.strip() for line in in_path.read_text().splitlines()
             if line.strip()]
    print(f'Re-verifying {len(paths)} files before deletion...', flush=True)

    confirmed: List[str] = []
    skipped: List[Tuple[str, str]] = []
    done = 0
    t_start = time.time()

    if workers <= 1:
        iterator = (inspect_file(p) for p in paths)
    else:
        pool = mp.Pool(workers)
        iterator = pool.imap_unordered(inspect_file, paths, chunksize=50)

    try:
        for path_str, status, info in iterator:
            done += 1
            if status == 'corrupt':
                confirmed.append(path_str)
            else:
                skipped.append((path_str, f'status={status} ({info})'))
            if done % progress_every == 0 or done == len(paths):
                print(f'  [{done}/{len(paths)}] verified={len(confirmed)} '
                      f'skipped={len(skipped)} '
                      f'elapsed={time.time() - t_start:.0f}s', flush=True)
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    print('', flush=True)
    print(f'Confirmed corrupt: {len(confirmed)}')
    print(f'Skipped (no longer match signature): {len(skipped)}')

    deleted = 0
    for p in confirmed:
        try:
            Path(p).unlink()
            deleted += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            skipped.append((p, f'unlink failed: {type(e).__name__}: {e}'))
    print(f'Deleted: {deleted}')

    if skipped:
        print(f'Skipped details (first 20):')
        for p, reason in skipped[:20]:
            print(f'  {p}: {reason}')
        if len(skipped) > 20:
            print(f'  ... and {len(skipped) - 20} more')


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--scan', action='store_true')
    g.add_argument('--delete', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('corrupt_files.txt'))
    ap.add_argument('--in', dest='in_path', type=Path,
                    default=Path('corrupt_files.txt'))
    ap.add_argument('--dataset', default=None,
                    help='Restrict scan to one dataset subdir')
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count()),
                    help='Parallel worker processes (default: CPU count)')
    ap.add_argument('--progress', type=int, default=500,
                    help='Print progress every N files (default 500)')
    args = ap.parse_args()

    if args.scan:
        scan(args.out, args.dataset, args.workers, args.progress)
    else:
        delete(args.in_path, args.workers, args.progress)


if __name__ == '__main__':
    main()
