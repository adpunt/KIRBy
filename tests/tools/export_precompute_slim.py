#!/usr/bin/env python3
"""Copy the main predictive precompute rows out of result_db into small files.

Reads  <KIRBY_CACHE_DIR>/result_db/<dataset>.db  read-only and writes
<out>/<dataset>.slim.db with the same `results` table minus the `blueprint`
column (the pickled model blob that makes the full files large). Keeps every
other column, including fold_scores_json, so per-fold scores survive.

Rows kept: tier in --tiers (default general, general_distance), objective
'predictive', aug_strategy in --aug (default none, capped, uncapped). Prints how many rows each file kept, so a
filter that matches nothing is visible rather than silent.

Usage (on the server, after `. setup.sh`):
    python tests/tools/export_precompute_slim.py --out $HOME/slim_precompute \
        caco2 herg_ki buchwald_hartwig
"""
import argparse
import os
import sqlite3
from pathlib import Path


def cache_root():
    env = os.environ.get('KIRBY_CACHE_DIR')
    return Path(env) if env else Path.home() / '.cache' / 'kirby'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tiers', default='general,general_distance')
    ap.add_argument('--aug', default='none,capped,uncapped')
    a = ap.parse_args()
    tiers = [t for t in a.tiers.split(',') if t]
    augs = [t for t in a.aug.split(',') if t]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        src = cache_root() / 'result_db' / f'{ds}.db'
        if not src.exists():
            print(f'{ds}: MISSING {src}')
            continue
        dst = out / f'{ds}.slim.db'
        if dst.exists():
            dst.unlink()
        con = sqlite3.connect(f'file:{src}?mode=ro', uri=True)
        cols = [r[1] for r in con.execute('PRAGMA table_info(results)')
                if r[1] != 'blueprint']
        col_sql = ', '.join(cols)
        marks = ','.join('?' * len(tiers))
        amarks = ','.join('?' * len(augs))
        rows = con.execute(
            f'SELECT {col_sql} FROM results WHERE tier IN ({marks}) '
            f"AND objective = 'predictive' AND aug_strategy IN ({amarks})",
            tiers + augs).fetchall()
        con.close()
        dc = sqlite3.connect(str(dst))
        dc.execute(f'CREATE TABLE results ({col_sql})')
        dc.executemany(
            f'INSERT INTO results VALUES ({",".join("?" * len(cols))})', rows)
        dc.commit()
        dc.close()
        print(f'{ds}: kept {len(rows)} rows -> {dst} '
              f'({dst.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
