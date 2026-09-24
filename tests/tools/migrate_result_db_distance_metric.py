#!/usr/bin/env python3
"""Give every result_db file the distance_metric column, filled with 'manhattan'.

WHY. Commit 9c6f7d7 (2026-09-20 18:19) added `distance_metric` to the result
store's identity columns and made a file without it raise StaleResultDB. Every
result_db on the server was written before that, so from then on every
precompute pass and every experiment that opened one failed in seconds:
"was written under an older column list and is missing ['distance_metric']".
Each of those precompute links reported 0 searches written and exited.

WHY 'manhattan' IS THE RIGHT VALUE FOR EVERY EXISTING ROW. The column records
which pairwise distance a search ran under. Nothing recorded it before
2026-09-20. The default in kirby/similarity/distance_config.py is 'manhattan',
and it has been since that file was first committed (c01132d, 2026-09-04).
It changes only through --distance-metric (run_dataset.py, the two
opig_scaffold scripts) or KIRBY_DISTANCE_METRIC in the environment. No SLURM
script in tests/slurm_scripts passes either. So a row is 'manhattan' unless
a job was launched from a shell that exported KIRBY_DISTANCE_METRIC -- check
that before --apply (the command is in the output below).

HOW. The column is part of the PRIMARY KEY, and SQLite cannot change a primary
key in place. So each file is rebuilt: a new file is created by ResultCache
itself, which gives exactly the schema the engine expects, every row is copied
across with distance_metric='manhattan', the row counts are compared, and only
then is the new file moved over the old. The old file is kept beside it as
<dataset>.db.pre_distance_metric. Nothing is deleted.

A file missing any column OTHER than distance_metric is reported and left
untouched: those rows cannot be filled in from a default.

⛔ NO JOB MAY BE WRITING THESE FILES WHILE THIS RUNS. It refuses to --apply if
squeue shows any job of yours RUNNING.

Usage:
    python tests/tools/migrate_result_db_distance_metric.py             # report only
    python tests/tools/migrate_result_db_distance_metric.py --apply
"""
import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

from kirby.result_cache import ALL_COLS, ResultCache  # noqa: E402

FILL = {'distance_metric': 'manhattan'}


def running_jobs():
    try:
        out = subprocess.run(['squeue', '-u', os.environ.get('USER', ''), '-h',
                              '-t', 'R', '-o', '%i %j'],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [l for l in out.stdout.splitlines() if l.strip()]


def columns(path):
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        return [r[1] for r in con.execute('PRAGMA table_info(results)')]
    finally:
        con.close()


def migrate(path, tmp_root):
    ds = path.stem
    have = columns(path)
    new_path = tmp_root / 'result_db' / f'{ds}.db'
    if new_path.exists():
        new_path.unlink()
    ResultCache(ds, cache_dir=tmp_root).close()   # creates the current schema
    src = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    dst = sqlite3.connect(str(new_path))
    try:
        n_src = src.execute('SELECT COUNT(*) FROM results').fetchone()[0]
        copy = [c for c in have if c in ALL_COLS or c == 'created_at']
        cols = copy + list(FILL)
        sql = (f'INSERT INTO results ({", ".join(cols)}) '
               f'VALUES ({", ".join("?" * len(cols))})')
        cur = src.execute(f'SELECT {", ".join(copy)} FROM results')
        while True:
            batch = cur.fetchmany(5000)
            if not batch:
                break
            dst.executemany(sql, [tuple(r) + tuple(FILL.values()) for r in batch])
        dst.commit()
        n_dst = dst.execute('SELECT COUNT(*) FROM results').fetchone()[0]
        ok = dst.execute('PRAGMA integrity_check').fetchone()[0]
    finally:
        src.close()
        dst.close()
    if n_dst != n_src or ok != 'ok':
        raise RuntimeError(f'{ds}: copied {n_dst} of {n_src} rows, '
                           f'integrity {ok!r} -- old file left in place')
    backup = path.with_name(path.name + '.pre_distance_metric')
    if not backup.exists():
        shutil.copy2(path, backup)
    shutil.move(str(new_path), str(path))
    ResultCache(ds, cache_dir=path.parent.parent).close()  # must open cleanly now
    return n_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--cache-dir', default=os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    args = ap.parse_args()

    root = Path(args.cache_dir)
    db_dir = root / 'result_db'
    print(f'result_db   {db_dir}')
    print(f'KIRBY_DISTANCE_METRIC in this shell: '
          f'{os.environ.get("KIRBY_DISTANCE_METRIC", "<unset>")}')
    files = sorted(db_dir.glob('*.db'))
    todo, fine, other = [], [], []
    for p in files:
        try:
            miss = [c for c in ALL_COLS if c not in columns(p)]
        except sqlite3.DatabaseError as e:
            other.append((p, f'unreadable: {e}'))
            continue
        if not miss:
            fine.append(p)
        elif set(miss) <= set(FILL):
            todo.append(p)
        else:
            other.append((p, f'missing {miss}'))

    print(f'\n{len(files)} files: {len(todo)} need distance_metric, '
          f'{len(fine)} already current, {len(other)} cannot be migrated\n')
    for p in todo:
        con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
        n = con.execute('SELECT COUNT(*) FROM results').fetchone()[0]
        con.close()
        print(f'  MIGRATE  {p.stem:28s} {n:8d} rows')
    for p in fine:
        print(f'  current  {p.stem}')
    for p, why in other:
        print(f'  LEFT     {p.stem}: {why}')

    if not args.apply:
        print('\nReport only. Before --apply, confirm no shell that submitted '
              'jobs exported another distance:\n'
              '  grep -n KIRBY_DISTANCE_METRIC ~/.bashrc ~/.bash_profile '
              'setup.sh')
        return
    if os.environ.get('KIRBY_DISTANCE_METRIC', 'manhattan') != 'manhattan':
        sys.exit('KIRBY_DISTANCE_METRIC is set to something other than '
                 'manhattan in this shell. Stopping.')
    run = running_jobs()
    if run is None:
        sys.exit('Could not run squeue to check for running jobs. Stopping.')
    run = [l for l in run if not l.split()[1:2] == ['bash']]
    if run:
        sys.exit('Jobs are RUNNING and may be writing these files:\n  '
                 + '\n  '.join(run) + '\nStopping. Hold or wait for them.')

    tmp_root = root / '_migrate_distance_metric_tmp'
    total = 0
    for p in todo:
        n = migrate(p, tmp_root)
        total += n
        print(f'  done     {p.stem:28s} {n:8d} rows  (old file kept as '
              f'{p.name}.pre_distance_metric)')
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f'\nMigrated {len(todo)} files, {total} rows.')


if __name__ == '__main__':
    main()
