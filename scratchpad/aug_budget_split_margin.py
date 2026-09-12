"""Read the 2026-08-21 budget-split results off disk and compute the two margins
that decide the augmentation question. No re-running; this is pure analysis.

For every (budget, donor source, supplement fraction) cell, paired per fold:

  vs BASECUT  — does spending part of a FIXED budget on extra features beat
                spending all of it on the base representation's own features?
                This is the brief's question.

  vs SPLITRAND — does the SELECTION beat matched-shape randomly chosen extra
                features? Same width, same budget, same shape. If this is ~0 the
                gain is extra model capacity, not the choosing.

Reported per dataset only. Nothing is averaged across datasets.
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.join(os.environ.get('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache'),
                    'aug_budget_split')

# Direction per dataset: which stored metric is the real one, and its sign.
REAL = {
    'herg_ki': ('rmse', True),
    'logd': ('rmse', True),
    'buchwald_hartwig': ('rmse', True),
    'flab_thermo_tresanco': ('spearman', False),
}


def better(a, b, lower_is_better):
    """Signed improvement of a over b, positive = a is better."""
    return (b - a) if lower_is_better else (a - b)


def load(ds):
    path = os.path.join(ROOT, f'{ds}__lightgbm.json')
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def report(ds):
    o = load(ds)
    if o is None:
        print(f"  {ds}: no stored result")
        return
    metric, lower = REAL[ds]
    rows = o['rows']

    # index by (arm, source, budget, frac, fold)
    idx = {}
    for r in rows:
        idx[(r['arm'], r['source'], r['budget'], r['frac'], r['fold'])] = r[metric]

    basecut = {(r['budget'], r['fold']): r[metric]
               for r in rows if r['arm'] == 'BASECUT'}

    cells = defaultdict(list)
    for r in rows:
        if r['arm'] != 'SPLIT':
            continue
        k = (r['budget'], r['source'], r['frac'])
        f = r['fold']
        sel = r[metric]
        bc = basecut.get((r['budget'], f))
        rnd = idx.get(('SPLITRAND', r['source'], r['budget'], r['frac'], f))
        if bc is None or rnd is None:
            continue
        cells[k].append((better(sel, bc, lower), better(sel, rnd, lower)))

    print(f"\n=== {ds}  (base={o['best_rep']}, {o['base_width']} features, "
          f"{o['n_folds']} folds, {o.get('n_seeds')} seeds, metric={metric}) ===")
    print(f"  {'budget':>6} {'source':<7} {'frac':>5} "
          f"{'vs base-cut':>12} {'won':>5} {'vs random':>11} {'won':>5}")

    out = []
    for k in sorted(cells):
        v = cells[k]
        n = len(v)
        d_bc = sum(a for a, _ in v) / n
        d_rd = sum(b for _, b in v) / n
        w_bc = sum(1 for a, _ in v if a > 0)
        w_rd = sum(1 for _, b in v if b > 0)
        out.append((k, d_bc, w_bc, d_rd, w_rd, n))

    # strongest cells by the brief's question first
    for (b, s, fr), d_bc, w_bc, d_rd, w_rd, n in sorted(
            out, key=lambda t: -t[1])[:12]:
        print(f"  {b:>6} {s:<7} {fr:>5.2f} {d_bc:>+12.4f} {w_bc:>3}/{n} "
              f"{d_rd:>+11.4f} {w_rd:>3}/{n}")

    pos_bc = [t for t in out if t[1] > 0]
    pos_both = [t for t in out if t[1] > 0 and t[3] > 0]
    print(f"  cells where extras beat spending the budget on base features: "
          f"{len(pos_bc)} of {len(out)}")
    print(f"  ... and where selection ALSO beats matched random: "
          f"{len(pos_both)} of {len(out)}")


if __name__ == '__main__':
    targets = sys.argv[1:] or list(REAL)
    for ds in targets:
        report(ds)
