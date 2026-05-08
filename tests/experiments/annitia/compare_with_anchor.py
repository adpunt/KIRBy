"""Compare eval/sub_*.csv predictions to the existing LB-best 0.9302 anchor sub.

For each fusion method that produced a val_test prediction, compute:
- Spearman ρ with anchor (per endpoint)
- max |Δrank| (largest pointwise rank disagreement)
- patients flipped between top-10% and bottom-10% bands

If a method correlates 0.99+ with anchor, it's nearly identical and submission
won't move LB much (good or bad).

If correlation drops below 0.90, the method is materially different — report
which patients are most affected, since they drive any LB difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

_REPO = Path(__file__).resolve().parents[3]
ANCHOR = _REPO / 'tests/results/annitia/full/sub_master_meta_avg_aware_weighted.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', default='tests/results/annitia/eval')
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not ANCHOR.exists():
        print(f'WARN: anchor not found at {ANCHOR}, looking for fallback...')
        # find any meta_avg_aware_weighted sub
        for p in (_REPO / 'tests/results/annitia/full').glob('sub_master_meta_avg_aware_weighted*.csv'):
            print(f'  using {p}')
            anchor = pd.read_csv(p)
            break
        else:
            return
    else:
        anchor = pd.read_csv(ANCHOR)

    print(f'Anchor: {ANCHOR.name} ({len(anchor)} rows)')

    rows = []
    for sub_p in sorted(eval_dir.glob('sub_*.csv')):
        if 'master' in sub_p.name:
            continue
        method = sub_p.stem.replace('sub_', '')
        df = pd.read_csv(sub_p)
        merged = anchor.merge(df, on='trustii_id', how='inner', suffixes=('_anchor', '_method'))
        if 'risk_hepatic_event_anchor' not in merged.columns:
            print(f'  {method}: missing hep col')
            continue

        for ep, col in [('hepatic', 'risk_hepatic_event'), ('death', 'risk_death')]:
            a = merged[f'{col}_anchor'].values
            m = merged[f'{col}_method'].values
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(m)):
                rows.append(dict(method=method, endpoint=ep, rho=float('nan'),
                                 max_drank=float('nan'), flip_pct=float('nan')))
                continue
            rho = float(spearmanr(a, m).statistic)
            ra = rankdata(a) / len(a)
            rm = rankdata(m) / len(m)
            max_d = float(np.max(np.abs(ra - rm)))
            # Top/bottom decile flip
            top_a = ra >= 0.9
            bot_a = ra <= 0.1
            top_m = rm >= 0.9
            bot_m = rm <= 0.1
            flips = ((top_a & bot_m) | (bot_a & top_m)).sum()
            rows.append(dict(method=method, endpoint=ep, rho=rho,
                             max_drank=max_d, flip_pct=100*flips/len(a),
                             flips=int(flips)))

    rows.sort(key=lambda r: (r['endpoint'], -r['rho']))
    print(f'\n  {"method":30s} {"ep":>9s} {"rho":>7s} {"max_dr":>8s} {"flip%":>8s}')
    for r in rows:
        print(f'  {r["method"]:30s} {r["endpoint"]:>9s} '
              f'{r["rho"]:>7.4f} {r["max_drank"]:>8.4f} {r["flip_pct"]:>8.2f}')

    # Markdown
    md = ['# Anchor correlation report', '',
          f'Anchor: `{ANCHOR.name}` (LB 0.9302)', '',
          '| method | endpoint | Spearman ρ | max Δrank | top↔bot decile flips |',
          '|---|---|---|---|---|']
    for r in rows:
        md.append(f'| {r["method"]} | {r["endpoint"]} | {r["rho"]:.4f} | '
                  f'{r["max_drank"]:.4f} | {r.get("flips", 0)} ({r["flip_pct"]:.2f}%) |')
    (eval_dir / 'anchor_correlation.md').write_text('\n'.join(md))
    (eval_dir / 'anchor_correlation.json').write_text(json.dumps(rows, indent=2))
    print(f'\nWrote {eval_dir}/anchor_correlation.md')


if __name__ == '__main__':
    main()
