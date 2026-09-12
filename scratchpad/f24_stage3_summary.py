"""Pool the f24 stage-3 win counts per dataset and method.

The three rules: decoys outrank the score, win counts per dataset and never a
mean across them, three folds ranks methods and cannot support a quoted delta.
"""
import os, sys, json, glob
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'analysis'))
D = 'tests/experiments/results/framework'

docs = []
for p in sorted(glob.glob(os.path.join(D, 'f24_augmentation_admission_*.json'))):
    if any(k in os.path.basename(p) for k in ('__count', '__timing', '__smoke')):
        continue
    with open(p) as fh:
        docs.append((p, json.load(fh)))

# the no-extras control, keyed across files
ctrl = {}
for _, d in docs:
    for r in d.get('rows') or []:
        if r.get('cell') != 'none' or r.get('control') != 'real':
            continue
        for m, v in (r.get('models') or {}).items():
            if v.get('status') == 'ok':
                ctrl[(d['dataset'], r['placement'], r['budget'],
                      r.get('allocation_method'), m)] = v['mean']

tally = defaultdict(lambda: defaultdict(int))
meta = {}
for _, d in docs:
    ds, sign = d['dataset'], (-1 if d.get('lower_is_better') else 1)
    meta[ds] = (d.get('metric'), d.get('fold_spread'), d.get('n_samples'))
    pairs = defaultdict(dict)
    for r in d.get('rows') or []:
        for m, v in (r.get('models') or {}).items():
            if v.get('status') != 'ok':
                continue
            k = (r['cell'], r.get('margin_multiple'), r['placement'],
                 r['budget'], r.get('allocation_method'), m)
            pairs[k][r['control']] = v
    for k, got in pairs.items():
        cell, mg, pl, bud, alloc, m = k
        if cell == 'none' or 'real' not in got:
            continue
        v = got['real']
        method = cell.split('__')[0]
        t = tally[(ds, method, pl)]
        t['total'] += 1
        clean = (sum(v.get('decoy_noise_per_fold') or []) == 0
                 and sum(v.get('decoy_echo_per_fold') or []) == 0)
        t['clean'] += clean
        b = ctrl.get((ds, pl, bud, alloc, m))
        beats_none = b is not None and sign * (v['mean'] - b) > 0
        t['none'] += beats_none
        beats_rand = ('random' in got
                      and sign * (v['mean'] - got['random']['mean']) > 0)
        t['rand'] += beats_rand
        t['survives'] += (clean and beats_none and beats_rand)

for ds in sorted(tally_ds := {k[0] for k in tally}):
    metric, spread, n = meta[ds]
    print(f'\n{"="*78}\n{ds}   {metric}   n={n}   fold spread {spread:.4f}')
    print(f'  {"placement":9}{"method":11}{"configs":>9}{"no decoys":>11}'
          f'{"beats none":>12}{"beats random":>14}{"ALL THREE":>11}')
    for (d2, method, pl), t in sorted(tally.items()):
        if d2 != ds:
            continue
        print(f'  {pl:9}{method:11}{t["total"]:>9}{t["clean"]:>11}'
              f'{t["none"]:>12}{t["rand"]:>14}{t["survives"]:>11}')

# --- what the survivors look like, and whether they clear the fold spread
print('\n\n' + '='*78)
print('CONFIGURATIONS SURVIVING ALL THREE, and their margin over the fold spread')
surv = defaultdict(list)
for _, d in docs:
    ds, sign = d['dataset'], (-1 if d.get('lower_is_better') else 1)
    spread = d.get('fold_spread') or 0
    pairs = defaultdict(dict)
    for r in d.get('rows') or []:
        for m, v in (r.get('models') or {}).items():
            if v.get('status') == 'ok':
                pairs[(r['cell'], r.get('margin_multiple'), r['placement'],
                       r['budget'], r.get('allocation_method'), m)][r['control']] = v
    for k, got in pairs.items():
        cell, mg, pl, bud, alloc, m = k
        if cell == 'none' or 'real' not in got:
            continue
        v = got['real']
        if sum(v.get('decoy_noise_per_fold') or []) or sum(v.get('decoy_echo_per_fold') or []):
            continue
        b = ctrl.get((ds, pl, bud, alloc, m))
        if b is None:
            continue
        dn = sign * (v['mean'] - b)
        if dn <= 0 or 'random' not in got:
            continue
        dr = sign * (v['mean'] - got['random']['mean'])
        if dr <= 0:
            continue
        surv[ds].append((dn, dn/spread if spread else 0, cell, mg, pl, alloc, m,
                         v.get('mean_kept')))

for ds in sorted(surv):
    rows = sorted(surv[ds], reverse=True)
    over = [r for r in rows if r[1] > 1.0]
    print(f'\n{ds}: {len(rows)} survivors, {len(over)} of them beating no-extras '
          f'by MORE than one fold spread')
    for dn, mult, cell, mg, pl, alloc, m, kept in rows[:12]:
        print(f'   {dn:+.4f} ({mult:4.2f}x spread)  {pl:6} {alloc or "-":20} '
              f'{m:8} kept~{kept:5.1f}  {cell}__m{mg:g}')
