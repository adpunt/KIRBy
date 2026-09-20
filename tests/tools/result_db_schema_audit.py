#!/usr/bin/env python
"""Which cached searches were written under an older column list.

WHAT HAPPENED. `nested_cv_evaluate` opened
result_db/buchwald_hartwig__ocv5f0s42.db, ResultCache._ensure_schema found no
`rep_count` column, and raised. All five outer folds raised the same way, so the
nested arm produced nothing.

`rep_count` became part of the primary key on 2026-09-12 (commit df7c7fe, the
note at result_cache.py:104-109). There is no migration: a file written before
that date cannot be opened at all, and its rows cannot be read out either,
because the column the rows would need was never written.

WHY NOBODY SAW IT. check_precompute_status.py DOES run the same PRAGMA check —
line 334 — but it walks DATASET_REPS and opens `<dataset>.db` by name. The
nested per-fold stores are `<dataset>__ocv<N>f<i>s<seed>.db`, which no dataset
name spells, so the status table never opens them and never reports them.

  <dataset>.db                       ONE search over the whole dataset.
                                     What the precompute fills. What the
                                     per-fold-reselection arm reads.
  <dataset>__ocv5f0s42.db ... f4     FIVE searches, one per outer fold, each
                                     over that fold's training molecules only.
                                     What nested CV reads.

Those are different files holding different searches, so a stale one of either
kind says nothing about the other. This walks every .db in result_db/ and
reports both.

USAGE
    python tests/tools/result_db_schema_audit.py              # report only
    python tests/tools/result_db_schema_audit.py --quarantine # move the stale ones aside

Nothing is deleted. --quarantine MOVES files into a dated directory beside the
cache, so every one of them is recoverable.
"""
import argparse
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from kirby.result_cache import ALL_COLS  # noqa: E402

STAMP = time.strftime('%Y%m%d')

# nested_cv_cache_namespace (find_best_hybrid.py:10016) spells one outer fold's
# inner search exactly this way. Every OTHER namespace is some experiment's own
# condition -- the noise wave writes `<dataset>__n20_as_they_come_none_d0_f2`
# and calling that an outer fold would be wrong.
_OCV = re.compile(r'^ocv(\d+)f(\d+)s(\d+)$')


def describe(stem):
    """What one .db file holds, from its name alone."""
    if '__' not in stem:
        return 'one search over the whole dataset'
    ns = stem.split('__', 1)[1]
    m = _OCV.match(ns)
    if m:
        return (f'nested CV, outer fold {m.group(2)} of {m.group(1)}, '
                f'seed {m.group(3)}')
    return f'a named condition: {ns}' 


def cache_root():
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def inspect(path):
    """(missing columns, row count or None, error or None) for one .db."""
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'
    try:
        have = {r[1] for r in conn.execute('PRAGMA table_info(results)')}
        if not have:
            return None, None, 'no `results` table'
        missing = [c for c in ALL_COLS if c not in have]
        rows = conn.execute('SELECT COUNT(*) FROM results').fetchone()[0]
        return missing, rows, None
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quarantine', action='store_true',
                    help='move every stale file aside (moved, never deleted)')
    ap.add_argument('--dataset',
                    help='only files whose name starts with this dataset')
    args = ap.parse_args()

    db_dir = cache_root() / 'result_db'
    print(f'result_db: {db_dir}')
    if not db_dir.exists():
        print('  does not exist')
        return 0

    files = sorted(db_dir.glob('*.db'))
    if args.dataset:
        files = [f for f in files
                 if f.stem == args.dataset
                 or f.stem.startswith(args.dataset + '__')]
    if not files:
        print('  no .db files match')
        return 0

    ok, stale, broken = [], [], []
    for p in files:
        missing, rows, err = inspect(p)
        if err:
            broken.append((p, err))
        elif missing:
            stale.append((p, missing, rows))
        else:
            ok.append((p, rows))

    print(f'\n{len(files)} file(s): {len(ok)} openable, {len(stale)} stale, '
          f'{len(broken)} unreadable\n')

    if ok:
        print('OPENABLE — the column list matches the code')
        for p, rows in ok:
            print(f'  {p.name:58s} {rows:>8,} configurations  '
                  f'{describe(p.stem)}')
    if stale:
        print('\n⛔ STALE — cannot be opened, and the rows cannot be read out')
        for p, missing, _ in stale:
            age = time.strftime('%Y-%m-%d', time.localtime(p.stat().st_mtime))
            print(f'  {p.name:58s} written {age}  missing '
                  f'{", ".join(missing)}')
            print(f'  {"":58s} {describe(p.stem)}')
    if broken:
        print('\nUNREADABLE for some other reason')
        for p, err in broken:
            print(f'  {p.name:58s} {err}')

    if stale and not args.quarantine:
        print(f'\n{len(stale)} stale file(s). Re-run with --quarantine to move '
              f'them aside, then re-run whatever needs them: the search will '
              f'rebuild from nothing.')
    if stale and args.quarantine:
        dest = cache_root() / f'result_db_stale_{STAMP}'
        dest.mkdir(parents=True, exist_ok=True)
        print(f'\n--- quarantine into {dest} ---')
        for p, _, _ in stale:
            shutil.move(str(p), str(dest / p.name))
            print(f'  moved {p.name}')
        print(f'\n{len(stale)} file(s) moved. Nothing deleted.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
