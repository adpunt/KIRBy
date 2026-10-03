#!/usr/bin/env python3
"""Does the neighbour count list 10, 20, 40 lose anything, and does the best count
depend on the representation kind or the coefficient?

Reads f34_full_sweep (k 5 to 40). k above 40 is in f34_importance_sweep and its
analysis. Only the shortlisted grid is looked at:

  weighting          rank, inverse_square
  coefficients       binary: tanimoto, dice, euclidean
                     count: soergel, tanimoto, manhattan
                     continuous: cosine, correlation, manhattan, minkowski_p0.5

A score is the mean over the 5 folds, in the dataset's own metric.

Usage:
    python tests/experiments/framework/f34_k_check.py
"""
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[2] / 'results' / 'framework'
N_FOLDS = 5
WEIGHTING = {'rank', 'inverse_square'}
COEFFS = {
    'binary': ('tanimoto', 'dice', 'euclidean'),
    'count': ('soergel', 'tanimoto', 'manhattan'),
    'continuous': ('cosine', 'correlation', 'manhattan', 'minkowski_p0.5'),
}
PICK = (10, 20, 40)


def read(dirname):
    """(dataset, rep) -> {(weighting, coefficient, k): mean over 5 folds}."""
    d = RES / dirname
    folds = defaultdict(dict)
    meta = {}
    if not d.exists():
        return {}, meta
    for f in sorted(d.glob('*/*.jsonl')):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            key = (r['dataset'], r['representation'])
            meta[key] = (r['kind'], r['metric'], r['lower_is_better'])
            if r['coefficient'] not in COEFFS[r['kind']]:
                continue
            for s, v in (r.get('scores') or {}).items():
                w, k = s.split('|')
                if w in WEIGHTING:
                    folds[key + ((w, r['coefficient'], int(k[1:])),)][
                        r['fold']] = v
    out = defaultdict(dict)
    for (ds, rep, setting), f in folds.items():
        if len(f) == N_FOLDS:
            out[(ds, rep)][setting] = float(np.mean(list(f.values())))
    return out, meta


def best(cells, lower, keep=lambda s: True):
    c = {s: v for s, v in cells.items() if keep(s)}
    if not c:
        return None, None
    s = (min if lower else max)(c, key=c.get)
    return s, c[s]


full, meta = read('f34_full_sweep')


def name(s):
    return f'{s[0]}|{s[1]}|k{s[2]}'


print(f'{len(full)} (dataset, representation) pairs with all {N_FOLDS} folds '
      f'in f34_full_sweep.\n')

print('1. Inside the shortlisted weightings and coefficients: the best score at '
      'any k from 5 to 40, against the best at k 10, 20 or 40 only. Gap is in '
      'the dataset metric; 0 means 10/20/40 catches it.')
gaps = defaultdict(list)
for key in sorted(full):
    kind, metric, lower = meta[key]
    s_all, v_all = best(full[key], lower)
    s_pick, v_pick = best(full[key], lower, lambda s: s[2] in PICK)
    gap = abs(v_all - v_pick)
    gaps[kind].append(gap)
    print(f'  {key[0]:10s} {key[1]:24s} {kind:10s} {metric:4s} '
          f'any k {v_all:.4f} ({name(s_all):32s})  10/20/40 {v_pick:.4f}  '
          f'gap {gap:.4f}')
print('  Largest gap per kind: ' + ', '.join(
    f'{k} {max(g):.4f} over {len(g)} pairs' for k, g in sorted(gaps.items())))

print('\n2. The best k for each coefficient, taking the better of the two '
      'weightings. One row per (dataset, representation).')
by_kind_coef = defaultdict(Counter)
for key in sorted(full):
    kind, _, lower = meta[key]
    cells = []
    for c in COEFFS[kind]:
        s, v = best(full[key], lower, lambda s, c=c: s[1] == c)
        if s is None:
            cells.append(f'{c} -')
            continue
        by_kind_coef[(kind, c)][s[2]] += 1
        cells.append(f'{c} k{s[2]} ({s[0][:3]})')
    print(f'  {key[0]:10s} {key[1]:24s} ' + '   '.join(cells))
print('\n   How often each k is best, per kind and coefficient (k 5 to 40):')
for (kind, c), cnt in sorted(by_kind_coef.items()):
    print(f'  {kind:10s} {c:16s} ' + ', '.join(
        f'k{k}: {n}' for k, n in sorted(cnt.items())))

print('\n   How often each k is best, per dataset (all shortlisted coefficients):')
by_ds = defaultdict(Counter)
for key in full:
    kind, _, lower = meta[key]
    for c in COEFFS[kind]:
        s, _ = best(full[key], lower, lambda s, c=c: s[1] == c)
        if s is not None:
            by_ds[key[0]][s[2]] += 1
for ds, cnt in sorted(by_ds.items()):
    print(f'  {ds:10s} ' + ', '.join(f'k{k}: {n}' for k, n in sorted(cnt.items())))

