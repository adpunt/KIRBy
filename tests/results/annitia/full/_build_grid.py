#!/usr/bin/env python3
"""Aggregate external T2.2 cv_summary into a master grid CSV.

Reads cv_summary.json from --cv-dir and emits nhanes_grid.csv with columns:
    subset, recipe, endpoint, mean_cindex, std_cindex, weighted_LB
where weighted_LB = 0.7 * hep + 0.3 * death (per-recipe, per-subset).
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv-dir', required=True)
    ap.add_argument('--out-csv', required=True)
    args = ap.parse_args()

    p = Path(args.cv_dir) / 'cv_summary.json'
    if not p.exists():
        sys.exit(f'missing: {p}')
    summary = json.loads(p.read_text())
    rows = []
    by_pair = {}  # (subset, recipe) -> {hepatic: c, death: c}
    for ep in ('hepatic', 'death'):
        for r in summary.get('per_condition', {}).get(ep, []):
            subset = r.get('subset')
            recipe = r.get('label')
            mc = r.get('mean_cindex')
            sd = r.get('std_cindex')
            rows.append((subset, recipe, ep, mc, sd))
            by_pair.setdefault((subset, recipe), {})[ep] = mc

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'recipe', 'endpoint', 'mean_cindex',
                    'std_cindex', 'weighted_LB'])
        for subset, recipe, ep, mc, sd in rows:
            pair = by_pair.get((subset, recipe), {})
            hep = pair.get('hepatic')
            dth = pair.get('death')
            wlb = (0.7 * hep + 0.3 * dth) if (hep is not None and dth is not None) else None
            w.writerow([subset, recipe, ep,
                        f'{mc:.4f}' if mc is not None else '',
                        f'{sd:.4f}' if sd is not None else '',
                        f'{wlb:.4f}' if wlb is not None else ''])
    # Print top-5 by weighted_LB
    pairs = []
    for (s, rcp), m in by_pair.items():
        if 'hepatic' in m and 'death' in m and m['hepatic'] is not None and m['death'] is not None:
            pairs.append((s, rcp, m['hepatic'], m['death'],
                          0.7 * m['hepatic'] + 0.3 * m['death']))
    pairs.sort(key=lambda x: x[4], reverse=True)
    print('TOP-5 by weighted_LB:')
    for s, rcp, h, d, wl in pairs[:5]:
        print(f'  {s:18s} {rcp:25s} hep={h:.4f} dth={d:.4f} wLB={wl:.4f}')
    print(f'\nWrote {len(rows)} rows to {args.out_csv}')


if __name__ == '__main__':
    main()
