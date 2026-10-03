#!/usr/bin/env python3
"""Read f34_importance_sweep's output: which (importance weighting, neighbour
weighting, coefficient, k) wins per (dataset, representation), whether importance
weighting beats no weighting, and whether k 10/20/40 is enough.

A setting's score is the mean over the 5 folds, in the dataset's own metric. A
setting missing a fold is not ranked.

Usage:
    python tests/experiments/framework/f34_importance_analyse.py [results dir]
"""
import sys
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parents[2] / 'results' / 'framework'
           / 'f34_importance_sweep')
N_FOLDS = 5
TOP = 5
PICK_K = (10, 20, 40)

folds = defaultdict(dict)     # (ds, rep, setting) -> {fold: score}
meta = {}
n_files = 0
for f in sorted(DIR.glob('*/*.jsonl')):
    n_files += 1
    for line in f.read_text().splitlines():
        r = json.loads(line)
        key = (r['dataset'], r['representation'])
        meta.setdefault(key, (r['kind'], r['metric'], r['lower_is_better']))
        for s, v in (r.get('scores') or {}).items():
            w, k = s.split('|')
            folds[key + ((r['importance'], w, r['coefficient'],
                          int(k[1:])),)][r['fold']] = v
groups = defaultdict(dict)
missing = Counter()
for (ds, rep, setting), f in folds.items():
    if len(f) == N_FOLDS:
        groups[(ds, rep)][setting] = float(np.mean(list(f.values())))
    else:
        missing[(ds, rep)] += 1

print(f'{DIR}: {n_files} task files, {len(groups)} (dataset, representation) '
      f'pairs with at least one setting on all {N_FOLDS} folds')
if not groups:
    sys.exit('Nothing complete to rank yet.')


def name(s):
    return f'{s[0]}|{s[1]}|{s[2]}|k{s[3]}'


def ranked(key, keep=lambda s: True):
    lower = meta[key][2]
    c = groups[key]
    return sorted((s for s in c if keep(s)),
                  key=lambda s: c[s] if lower else -c[s])


def best(key, keep=lambda s: True):
    r = ranked(key, keep)
    return (r[0], groups[key][r[0]]) if r else (None, None)


print(f'\nOne block per (dataset, representation): the {TOP} best settings '
      f'(importance|neighbour weighting|coefficient|k), mean over {N_FOLDS} '
      f'folds.')
for key in sorted(groups):
    kind, metric, lower = meta[key]
    note = f'  [{missing[key]} settings missing a fold]' if missing[key] else ''
    print(f'{key[0]} / {key[1]}  ({kind}, {metric}, '
          f'{"lower" if lower else "higher"} is better){note}')
    for s in ranked(key)[:TOP]:
        print(f'    {name(s):50s} {groups[key][s]:.4f}')

print('\nBest rank each value reached in any (dataset, representation), and how '
      'many pairs it won.')
for ax, label in ((0, 'importance'), (1, 'neighbour weighting'),
                  (2, 'coefficient'), (3, 'k')):
    rank_best, wins = {}, Counter()
    for key in groups:
        r = ranked(key)
        wins[r[0][ax]] += 1
        for i, s in enumerate(r):
            rank_best[s[ax]] = min(rank_best.get(s[ax], 10**9), i + 1)
    print(f'  {label}: ' + ', '.join(
        f'{v} best {rank_best[v]} won {wins[v]}'
        for v in sorted(rank_best, key=lambda v: (rank_best[v], -wins[v]))))

print('\nImportance weighting against none, per pair. Change is best with '
      'importance minus best without, in the dataset metric (negative is '
      'better for rmse, positive for auc).')
for key in sorted(groups):
    s0, v0 = best(key, lambda s: s[0] == 'none')
    s1, v1 = best(key, lambda s: s[0] != 'none')
    if s0 is None or s1 is None:
        continue
    print(f'  {key[0]:10s} {key[1]:24s} none {v0:.4f} ({name(s0):40s})  '
          f'importance {v1:.4f} ({name(s1):46s})  change {v1 - v0:+.4f}')

print('\nk 10, 20 or 40 only, against every k (10 to 200). Gap in the dataset '
      'metric; 0 means 10/20/40 catches the best.')
gaps = defaultdict(list)
for key in sorted(groups):
    s_all, v_all = best(key)
    s_pick, v_pick = best(key, lambda s: s[3] in PICK_K)
    gap = abs(v_all - v_pick)
    gaps[meta[key][0]].append(gap)
    print(f'  {key[0]:10s} {key[1]:24s} any k {v_all:.4f} ({name(s_all):46s}) '
          f' 10/20/40 {v_pick:.4f}  gap {gap:.4f}')
print('  Largest gap per kind: ' + ', '.join(
    f'{k} {max(g):.4f} over {len(g)} pairs' for k, g in sorted(gaps.items())))

print('\nBest k per coefficient and importance variant, counted over pairs '
      '(better of the two neighbour weightings).')
cnt = defaultdict(Counter)
for key in groups:
    kind = meta[key][0]
    for c in {s[2] for s in groups[key]}:
        for imp in {s[0] for s in groups[key]}:
            s, _ = best(key, lambda s, c=c, imp=imp: s[2] == c and s[0] == imp)
            if s is not None:
                cnt[(kind, c, imp)][s[3]] += 1
for (kind, c, imp), n in sorted(cnt.items()):
    print(f'  {kind:10s} {c:16s} {imp:18s} ' + ', '.join(
        f'k{k}: {m}' for k, m in sorted(n.items())))
