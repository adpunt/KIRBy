import json, glob, os
from collections import defaultdict
D = 'tests/experiments/results/framework'
RANKS = ('residual_correlation', 'rf_importance', 'lightgbm_gain', 'mutual_info')

ctrl = {}
for ds in ('caco2', 'flab_expression_koenig'):
    f = os.path.join(D, f'f24_augmentation_admission_{ds}__single__none.json')
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    for r in d['rows']:
        if r['cell'] == 'none' and r['control'] == 'real':
            for m, v in r['models'].items():
                ctrl[(ds, m)] = v['mean']

for path in sorted(glob.glob(os.path.join(
        D, 'f24_augmentation_admission_*__single__uncapped__*__64-off.json'))):
    d = json.load(open(path))
    ds = d['dataset']
    sign = -1 if d['lower_is_better'] else 1
    src = os.path.basename(path).split('__uncapped__')[1].split('__')[0]
    by = defaultdict(dict)
    for r in d['rows']:
        for m, v in (r.get('models') or {}).items():
            if v.get('status') == 'ok':
                by[(r['cell'], r.get('margin_multiple'), m)][r['control']] = v
    agg = defaultdict(lambda: [None, None, 0.0, 0.0, 0])
    for (cell, mg, m), got in by.items():
        if 'real' not in got:
            continue
        rank = next((x for x in cell.split('__') if x in RANKS), None)
        if rank is None:
            continue
        v = got['real']
        f = len(v['decoy_noise_per_fold'] or [1])
        e = agg[(rank, m)]
        e[2] += sum(v['decoy_noise_per_fold'] or []) / f
        e[3] += v.get('mean_kept', 0)
        e[4] += 1
        if e[0] is None or sign * (v['mean'] - e[0]) > 0:
            e[0], e[1] = v['mean'], cell
    print(f"\n{'='*82}\n{ds}   {d['metric']}   {src} donors   base {d['base_rep']} "
          f"({d['base_width']} features)   wobble {d['fold_spread']:.4f}")
    print(f"  {'model':10}{'ranking rule':22}{'base':>9}{'augmented':>11}"
          f"{'change':>9}{'beats wobble':>14}{'random taken':>14}")
    for m in ('lightgbm', 'rf'):
        b = ctrl.get((ds, m))
        for r in RANKS:
            e = agg.get((r, m))
            if not e or b is None:
                continue
            ch = sign * (e[0] - b)
            print(f"  {m:10}{r:22}{b:>9.4f}{e[0]:>11.4f}{ch:>+9.4f}"
                  f"{('YES' if ch > d['fold_spread'] else 'no'):>14}"
                  f"{e[2]/e[4]:>14.1f}")
