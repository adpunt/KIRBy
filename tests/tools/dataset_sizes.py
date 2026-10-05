#!/usr/bin/env python3
"""How many molecules each dataset holds, and what a kNN sweep costs on it.

Why this exists: the cost of one neighbour setting is a distance between every
pair of training molecules, so it grows with the square of the molecule count.
Grouping datasets for a kNN cut therefore needs the molecule count, and the
fast-tier list in scratchpad/precompute_cuts/DECISIONS.md is about the hybrid
SEARCH grid, not about this.

Reads only. Loads each dataset with the 'fast' representation preset, prints the
row count, and writes nothing except the output file if one is asked for.

Usage (inside an interactive session on the server):
    python tests/tools/dataset_sizes.py
    python tests/tools/dataset_sizes.py --out dataset_sizes.txt
    python tests/tools/dataset_sizes.py --datasets herg_ki qm9 kiba
"""
import os
import sys
import time
import argparse
from pathlib import Path

os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# No search may run: this script only counts rows.
os.environ.setdefault('KIRBY_NO_COLD_SEARCH', '1.0')
os.environ.setdefault('KIRBY_COLD_SEARCH_MAX_CONFIGS', '0')

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))


def all_datasets():
    from experiments.complexity.complexity_helpers import DATASET_REPS
    return sorted(set(DATASET_REPS) - {'chembl_qsar', 'davis'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    from experiments import dataset_policy  # noqa: F401  (fills the registry)
    from experiments._shared import load_data

    names = a.datasets or all_datasets()
    lines = []
    header = (f"{'dataset':26s} {'molecules':>9s} {'groups':>7s} "
              f"{'task':>14s} {'pair work vs herg_ki':>20s}")
    print(header, flush=True)
    rows = []
    for ds in names:
        t0 = time.time()
        try:
            data = load_data(ds, rep_names=[])
            n = int(len(data['labels']))
            g = data.get('groups')
            ng = '-' if g is None else str(len(set(g)))
            tt = str(data.get('task_type'))
        except Exception as exc:
            rows.append((ds, None, '-', f'FAILED {type(exc).__name__}', 0.0))
            print(f'{ds:26s} {"-":>9s} {"-":>7s} '
                  f'{"FAILED " + type(exc).__name__:>14s}', flush=True)
            continue
        rows.append((ds, n, ng, tt, time.time() - t0))
        print(f'{ds:26s} {n:9d} {ng:>7s} {tt:>14s}', flush=True)

    base = dict((r[0], r[1]) for r in rows).get('herg_ki')
    print('\n' + header)
    for ds, n, ng, tt, secs in sorted(rows, key=lambda r: -(r[1] or 0)):
        if n is None:
            ratio = '-'
        elif base:
            ratio = f'{(n * n) / (base * base):.1f}x'
        else:
            ratio = '-'
        line = f'{ds:26s} {n if n is not None else "-":>9} {ng:>7s} {tt:>14s} {ratio:>20s}'
        print(line)
        lines.append(line)
    if a.out:
        Path(a.out).write_text(header + '\n' + '\n'.join(lines) + '\n')
        print(f'\nwritten to {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
