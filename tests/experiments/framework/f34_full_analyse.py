#!/usr/bin/env python3
"""Read f34_full_sweep's output and say, per (dataset, representation), which kNN
setting scores best, and which setting values ever come near the top.

A setting is one (weighting, coefficient, neighbour count). Its score is the mean
over the 5 folds, in the dataset's own metric. A setting missing a fold is not
ranked, and the count of those is printed.

Usage:
    python tests/experiments/framework/f34_full_analyse.py [results dir]
"""
import sys
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parents[2] / 'results' / 'framework'
           / 'f34_full_sweep')
N_FOLDS = 5
TOP = 5

groups = {}      # (dataset, rep) -> {setting: mean}
info = {}
for ds_dir in sorted(p for p in DIR.iterdir() if p.is_dir()):
    folds = defaultdict(dict)        # (rep, setting) -> {fold: score}
    kinds = defaultdict(set)
    lines = [l for f in sorted(ds_dir.glob('*.jsonl'))
             for l in f.read_text().splitlines()]
    for line in lines:
        r = json.loads(line)
        kinds[r['representation']].add(r['kind'])
        lower = r['lower_is_better']
        metric = r['metric']
        for key, v in (r.get('scores') or {}).items():
            w, k = key.split('|')
            folds[(r['representation'], f'{w}|{r["coefficient"]}|{k}')][
                r['fold']] = v
    by_rep = defaultdict(dict)
    incomplete = defaultdict(int)
    for (rep, setting), f in folds.items():
        if len(f) == N_FOLDS:
            by_rep[rep][setting] = float(np.mean(list(f.values())))
        else:
            incomplete[rep] += 1
    for rep, cells in by_rep.items():
        groups[(ds_dir.name, rep)] = cells
        info[(ds_dir.name, rep)] = (metric, lower, kinds[rep], incomplete[rep])
    for rep in set(incomplete) - set(by_rep):
        info[(ds_dir.name, rep)] = (metric, lower, kinds[rep], incomplete[rep])


n_files = sum(1 for _ in DIR.glob('*/*.jsonl'))
print(f'{DIR}: {n_files} task files, {len(groups)} (dataset, representation) '
      f'groups with all {N_FOLDS} folds')
if not groups:
    sys.exit('Nothing complete to rank. Look at f34_full.log for why the '
             'tasks stopped.')


def ranked(key):
    cells = groups[key]
    lower = info[key][1]
    return sorted(cells, key=lambda s: cells[s] if lower else -cells[s])


print(f'One block per (dataset, representation): the {TOP} best settings '
      f'(weighting|coefficient|neighbour count), mean over {N_FOLDS} folds.\n')
for key in sorted(info):
    metric, lower, kinds, n_inc = info[key]
    note = (f'  [{n_inc} settings missing a fold]' if n_inc else '')
    note += (f'  [kind differs by fold: {sorted(kinds)}]'
             if len(kinds) > 1 else '')
    print(f'{key[0]} / {key[1]}  ({"/".join(sorted(kinds))}, {metric}, '
          f'{"lower" if lower else "higher"} is better){note}')
    if key not in groups:
        print('    nothing complete yet')
        continue
    for s in ranked(key)[:TOP]:
        print(f'    {s:42s} {groups[key][s]:.4f}')

print('\nBest rank each value reached in any (dataset, representation). '
      'Out of the number of settings in that group.')
for ax, name in ((0, 'weighting'), (1, 'coefficient'), (2, 'neighbour count')):
    best = {}
    for key in groups:
        for i, s in enumerate(ranked(key)):
            v = s.split('|')[ax]
            best[v] = min(best.get(v, 10**9), i + 1)
    print(f'  {name}: ' + ', '.join(f'{v} {r}' for v, r in
                                    sorted(best.items(), key=lambda x: x[1])))

print('\nWinner per (dataset, representation), and the gap to second place in '
      'the dataset metric:')
for key in sorted(groups):
    r = ranked(key)
    c = groups[key]
    gap = abs(c[r[0]] - c[r[1]]) if len(r) > 1 else float('nan')
    print(f'  {key[0]:10s} {key[1]:26s} {r[0]:42s} {c[r[0]]:.4f}  '
          f'2nd {c[r[1]]:.4f}  gap {gap:.4f}')


# The shortlist proposed from the first f34 run (base representations only, 3
# datasets). For each group: the best score inside the shortlist, against the
# best score overall. A gap of 0 means the shortlist caught the winner.
SHORTLIST_WEIGHTING = {'inverse_square', 'rank'}
SHORTLIST_K = {'k10', 'k15', 'k20', 'k25'}
SHORTLIST_COEFF = {
    'binary': {'dice', 'cosine', 'manhattan', 'tversky'},
    'count': {'dice', 'cosine', 'manhattan', 'soergel'},
    'continuous': {'minkowski_p0.3', 'minkowski_p0.5', 'cosine'},
}
print('\nDoes the shortlist from the first run catch the winner? Gap = best in '
      'shortlist minus best overall, in the dataset metric (0 = caught).')
for key in sorted(groups):
    kinds = info[key][2]
    allowed = set().union(*(SHORTLIST_COEFF[k] for k in kinds))
    c = groups[key]
    inside = [s for s in ranked(key) if s.split('|')[0] in SHORTLIST_WEIGHTING
              and s.split('|')[1] in allowed and s.split('|')[2] in SHORTLIST_K]
    best = ranked(key)[0]
    if not inside:
        print(f'  {key[0]:10s} {key[1]:26s} no shortlist setting scored')
        continue
    gap = abs(c[best] - c[inside[0]])
    print(f'  {key[0]:10s} {key[1]:26s} gap {gap:.4f}   overall best {best}')
