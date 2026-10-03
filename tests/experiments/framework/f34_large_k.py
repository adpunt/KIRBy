#!/usr/bin/env python3
"""Neighbour counts above 40, on the 12 (dataset, representation) pairs whose best
setting in f34_full_sweep used k=40, the largest count that sweep tried.

Same code, folds, hybrids and scoring as f34_full_sweep; only the grid changes:

  weighting          rank, inverse_square
  neighbour count    40, 50, 60, 80, 100, 150, 200 (40 again, to tie back)
  coefficients       binary: tanimoto, dice, euclidean
                     count: soergel, tanimoto, manhattan
                     continuous: cosine, correlation, manhattan, minkowski_p0.5

Output goes to tests/results/framework/f34_large_k, one file per task, and a
re-run skips what is finished, exactly as f34_full_sweep does.

Usage on the server (inside an interactive session):
    python tests/experiments/framework/f34_large_k.py --list-tasks tasks.txt
    python tests/experiments/framework/f34_large_k.py --task herg mordred 0 main
"""
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f34_full_sweep as sweep  # noqa: E402
import f18_distance_metric_sweep as f18  # noqa: E402

# The 12 pairs whose winner in f34_full_sweep (all 6 weightings, k 5 to 40,
# every coefficient) was at k=40. Read off f34_full_analyse's output, 2026-10-03.
PAIRS = {
    'adme_mdr1': ('ecfp4', 'graph_kernel', 'grover', 'hybrid_general_distance',
                  'mhg_gnn', 'mordred'),
    'herg': ('graph_kernel', 'hybrid_general', 'hybrid_general_distance',
             'mhg_gnn', 'molformer', 'mordred'),
}

sweep.WEIGHTING = ('rank', 'inverse_square')
sweep.K_VALUES = (40, 50, 60, 80, 100, 150, 200)
sweep.COEFFS = {
    'binary': ('tanimoto', 'dice', 'euclidean'),
    'count': ('soergel', 'tanimoto', 'manhattan'),
    'continuous': ('cosine', 'correlation', 'manhattan', 'minkowski_p0.5'),
}
OUT = sweep.OUT_DEFAULT.parent / 'f34_large_k'


def list_tasks(path):
    tasks = []
    for ds, spaces in PAIRS.items():
        data, labels, _, folds = sweep.load(ds)
        n = len(labels)
        for space in spaces:
            if space in sweep.HYBRIDS:
                kind, d = 'continuous', sweep.f34.BUDGET
            else:
                X = np.asarray(data['reps'][space], dtype=np.float64)
                kind, d = f18.classify_representation(X), X.shape[1]
            parts = ((('main', 1), ('minkowski_p0.5', 20))
                     if kind == 'continuous' else (('all', 1),))
            for fi in range(len(folds)):
                for part, w in parts:
                    tasks.append((n * n * d * w, f'{ds} {space} {fi} {part}'))
    tasks.sort()
    Path(path).write_text('\n'.join(t for _, t in tasks) + '\n')
    print(f'{len(tasks)} tasks written to {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--list-tasks', metavar='PATH')
    ap.add_argument('--task', nargs=4,
                    metavar=('DATASET', 'REP', 'FOLD', 'PART'))
    a = ap.parse_args()
    from experiments import dataset_policy  # noqa: F401  (fills the registry)
    if a.list_tasks:
        return list_tasks(a.list_tasks)
    if a.task:
        ds, space, fi, part = a.task
        sweep.run_task(ds, space, int(fi), part, a.out, 'full')
        return 0
    ap.error('give --list-tasks or --task')


if __name__ == '__main__':
    sys.exit(main())
