#!/usr/bin/env python
"""Decide whether predicted-structure geometry is worth adding.

    python tests/analysis/analyse_geometry_survey.py tests/results/geometry_survey

Reads the survey result files and answers three separate questions, kept apart
because they have different answers and different strengths of evidence.

1. PRE-SPECIFIED. On each dataset, take the base representation that scores
   best on its own, and add all_geometry to it. One number per dataset, chosen
   before looking at any augmentation result, so it is not inflated by picking
   the winner. A sign test across datasets is the headline.

2. WHOLE GRID. Every base representation x geometry group x model, counted as
   helped or hurt per dataset. Says whether the effect is broad or confined.

3. SELECTION-OPTIMISTIC. The best augmentation per dataset, which is the
   maximum of a few hundred comparisons. Reported for completeness and never
   as an effect size.

Deltas are never averaged across datasets. Datasets are the unit.
"""

import sys
import json
import math
from pathlib import Path
from collections import defaultdict


def sign_test(deltas):
    """Two-sided sign test. Returns (n_pos, n_neg, p). Exact binomial."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, float('nan')
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


def main(results_dir):
    d = Path(results_dir)
    files = sorted(d.glob('geometry_*.json'))
    if not files:
        print(f"No geometry_*.json under {d}")
        return 1

    scored = {}
    for f in files:
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if r.get('status') != 'OK' or not r.get('base_baselines'):
            continue
        scored[r['metadata']['dataset']] = r

    if not scored:
        print(f"{len(files)} files, none scored with base-alone baselines.")
        return 1

    print(f"{len(scored)} dataset(s) with base-alone baselines: "
          f"{', '.join(sorted(scored))}\n")

    # ---------------------------------------------------------------- 1
    print("=" * 70)
    print("1. PRE-SPECIFIED: all_geometry added to each dataset's best base rep")
    print("=" * 70)
    print("   Chosen before seeing any augmentation result, so this is the")
    print("   number that is not inflated by picking a winner.\n")

    pre = {}
    for ds, r in sorted(scored.items()):
        is_clf = r['metadata'].get('task_type') == 'classification'
        key = 'accuracy_mean' if is_clf else 'spearman_mean'
        best_rep, best_model, best_val = None, None, -float('inf')
        for rep, entry in r['base_baselines'].items():
            for m, sc in entry.get('scores', {}).items():
                v = sc.get(key)
                if v is not None and v > best_val:
                    best_rep, best_model, best_val = rep, m, v
        aug = r['augmentations'].get(f'{best_rep}+all_geometry')
        if not aug or best_model not in aug.get('delta', {}):
            print(f"  {ds:<26} best base {best_rep} ({best_model}) "
                  f"{best_val:.4f}  — no all_geometry entry")
            continue
        dv = aug['delta'][best_model]
        pre[ds] = dv
        print(f"  {ds:<26} {best_rep:<14} {best_model:<4} "
              f"{best_val:.4f} -> {best_val + dv:.4f}   delta {dv:+.4f}")

    if pre:
        pos, neg, p = sign_test(list(pre.values()))
        vals = sorted(pre.values())
        mid = vals[len(vals) // 2] if len(vals) % 2 else \
            (vals[len(vals)//2 - 1] + vals[len(vals)//2]) / 2
        print(f"\n  {pos} up, {neg} down of {len(pre)} datasets. "
              f"Sign test p = {p:.4f}. Median delta {mid:+.4f}.")
        print(f"  Range {min(vals):+.4f} to {max(vals):+.4f}.")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 70)
    print("2. WHOLE GRID: every base rep x geometry group x model")
    print("=" * 70)

    per_group_ds = defaultdict(dict)   # group -> dataset -> (helped, hurt)
    for ds, r in scored.items():
        tally = defaultdict(lambda: [0, 0])
        for entry in r['augmentations'].values():
            for m, dv in entry.get('delta', {}).items():
                tally[entry['geometry']][0 if dv > 0 else 1] += 1
        for g, (h, u) in tally.items():
            per_group_ds[g][ds] = (h, u)

    print(f"\n  {'geometry group':<26} {'helped':>7} {'hurt':>6}   "
          f"datasets where it helped more often   sign p")
    for g in sorted(per_group_ds, key=lambda g: -sum(
            v[0] for v in per_group_ds[g].values())):
        h = sum(v[0] for v in per_group_ds[g].values())
        u = sum(v[1] for v in per_group_ds[g].values())
        per_ds = [1 if a > b else -1 for a, b in per_group_ds[g].values()]
        pos, neg, p = sign_test(per_ds)
        print(f"  {g:<26} {h:>7} {u:>6}   {pos:>2} of {pos+neg:<2}"
              f"                          {p:.3f}")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 70)
    print("3. SELECTION-OPTIMISTIC: best augmentation per dataset")
    print("=" * 70)
    print("   Maximum over every representation, group and model. Not an")
    print("   effect size — reported so it is not mistaken for one.\n")
    for ds, r in sorted(scored.items()):
        best = None
        n_comp = 0
        for key, entry in r['augmentations'].items():
            for m, dv in entry.get('delta', {}).items():
                n_comp += 1
                if best is None or dv > best[0]:
                    best = (dv, key, m)
        if best:
            print(f"  {ds:<26} {best[1]:<32} {best[2]:<4} "
                  f"delta {best[0]:+.4f}   (max of {n_comp})")

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 70)
    print("4. GEOMETRY ALONE vs THE BEST REPRESENTATION ALONE")
    print("=" * 70)
    print("   Whether structure features carry signal at all, separately from")
    print("   whether they add to a representation that already works.\n")
    for ds, r in sorted(scored.items()):
        is_clf = r['metadata'].get('task_type') == 'classification'
        key = 'accuracy_mean' if is_clf else 'spearman_mean'
        gbest = max((sc.get(key, -float('inf'))
                     for e in r['geometry_baselines'].values()
                     for sc in e.get('scores', {}).values()), default=float('nan'))
        bbest = max((sc.get(key, -float('inf'))
                     for e in r['base_baselines'].values()
                     for sc in e.get('scores', {}).values()), default=float('nan'))
        print(f"  {ds:<26} geometry alone {gbest:>7.4f}   "
              f"best representation {bbest:>7.4f}   gap {gbest - bbest:+.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else 'tests/results/geometry_survey'))
