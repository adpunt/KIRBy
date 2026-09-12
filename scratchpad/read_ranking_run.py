import json, sys
from collections import defaultdict
p = sys.argv[1]
d = json.load(open(p))
print(f"rows {len(d['rows'])}  fitting {d['fit_seconds']:.0f}s  "
      f"{d['fit_seconds']/720:.2f} s per fit  spread {d['fold_spread']:.4f} {d['metric']}")
sign = -1 if d['lower_is_better'] else 1
RANKS = ('residual_correlation', 'rf_importance', 'lightgbm_gain',
         'mutual_info', 'deeplift', 'residual_deeplift')
by = defaultdict(dict)
for r in d['rows']:
    m = (r['models'] or {}).get('rf')
    if m and m.get('status') == 'ok':
        by[(r['cell'], r.get('margin_multiple'))][r['control']] = m
print(f"\n{'ranking':22}{'size':10}{'vs random':>11}{'noise taken':>13}{'echo taken':>12}{'kept':>7}")
for (cell, mg), got in sorted(by.items()):
    if 'real' not in got:
        continue
    parts = cell.split('__')
    rank = next((x for x in parts if x in RANKS), '?')
    v = got['real']
    dr = (sign*(v['mean']-got['random']['mean'])) if 'random' in got else None
    n = sum(v.get('decoy_noise_per_fold') or [])
    e = sum(v.get('decoy_echo_per_fold') or [])
    s = f'{dr:+.4f}' if dr is not None else '  -'
    print(f"{rank:22}{parts[-1]:10}{s:>11}{n:>13}{e:>12}{v.get('mean_kept', 0):>7.1f}")
