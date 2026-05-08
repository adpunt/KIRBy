"""Decision-rule script for safe submission.

For each fusion method, compute a 'trust score' that combines:
- Hold-out C above anchor (signal of real lift)
- Hold-out SE (penalty for noisy lift)
- OOF→hold-out gap (penalty for transfer-overfit signature)
- Adversarial AUC (penalty for distribution shift)
- Anchor correlation (boost if highly correlated, since LB transfer is similar)

Trust = lift(hold) - 1.0 * SE(hold) - 0.5 * |OOF_C - hold_C| - 0.5 * max(0, |advAUC - 0.5| - 0.05)

Higher trust → safer submission. Anything below anchor's trust → don't submit.

Reads tests/results/annitia/eval/comparison.json + anchor_correlation.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', default='tests/results/annitia/eval')
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)

    cmp = json.loads((eval_dir / 'comparison.json').read_text())
    rows = cmp['summary_rows']

    anchor_corr = {}
    p = eval_dir / 'anchor_correlation.json'
    if p.exists():
        for r in json.loads(p.read_text()):
            anchor_corr[(r['method'], r['endpoint'])] = r.get('rho', float('nan'))

    # Build per-method per-endpoint trust
    trust = []
    # find anchor baseline (from method='anchor_rank_mean')
    anchor_rows = {r['endpoint']: r for r in rows if r['method'] == 'anchor_rank_mean'}
    for r in rows:
        ep = r['endpoint']
        c_h = r['c_holdout']
        se_h = r.get('c_holdout_se', 0.0)
        c_t = r['c_train']
        adv = r.get('adv_auc', 0.5)
        rho = anchor_corr.get((r['method'], ep), float('nan'))
        anc = anchor_rows.get(ep)
        anc_c = anc['c_holdout'] if anc else 0.5
        lift = c_h - anc_c
        gap = abs(c_t - c_h)
        adv_pen = 0.5 * max(0, abs((adv if np.isfinite(adv) else 0.5) - 0.5) - 0.05)
        score = lift - 1.0 * se_h - 0.5 * gap - adv_pen
        trust.append(dict(method=r['method'], endpoint=ep,
                          c_train=c_t, c_holdout=c_h, c_holdout_se=se_h,
                          uno_holdout=r.get('uno_holdout', float('nan')),
                          adv_auc=adv, rho_anchor=rho,
                          lift_vs_anchor=lift,
                          gap_train_holdout=gap,
                          trust=float(score)))
    trust.sort(key=lambda r: (r['endpoint'], -r['trust']))

    print(f'\n  {"endpoint":>9s} {"method":25s} {"holdC":>7s} {"±SE":>6s} '
          f'{"liftA":>7s} {"gap":>6s} {"adv":>6s} {"ρanc":>7s} {"trust":>8s}')
    for r in trust:
        print(f'  {r["endpoint"]:>9s} {r["method"]:25s} '
              f'{r["c_holdout"]:>7.4f} {r["c_holdout_se"]:>6.4f} '
              f'{r["lift_vs_anchor"]:>+7.4f} {r["gap_train_holdout"]:>6.4f} '
              f'{r["adv_auc"]:>6.3f} '
              f'{r["rho_anchor"]:>7.4f} {r["trust"]:>+8.4f}')

    # Decision rule: only methods with trust > 0 (strictly above anchor) AND
    # rho_anchor > 0.85 (not radically different from LB-proven anchor) are
    # eligible for submission.
    eligible = [r for r in trust if r['trust'] > 0
                and (np.isnan(r['rho_anchor']) or r['rho_anchor'] > 0.85)
                and r['method'] != 'anchor_rank_mean']
    print(f'\n  --- Eligible (trust > 0, ρ_anchor > 0.85) ---')
    for r in eligible:
        print(f'  {r["endpoint"]:>9s} {r["method"]:25s} trust={r["trust"]:+.4f}')

    md = ['# Trust score report', '',
          'Trust = lift_vs_anchor − SE − 0.5·|OOFgap| − advAUC penalty',
          '', '| endpoint | method | hold C | ±SE | lift vs anchor | OOF gap | adv AUC | ρ anchor | trust |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in trust:
        md.append(f'| {r["endpoint"]} | {r["method"]} | {r["c_holdout"]:.4f} | '
                  f'{r["c_holdout_se"]:.4f} | {r["lift_vs_anchor"]:+.4f} | '
                  f'{r["gap_train_holdout"]:.4f} | {r["adv_auc"]:.3f} | '
                  f'{r["rho_anchor"]:.4f} | {r["trust"]:+.4f} |')
    (eval_dir / 'trust.md').write_text('\n'.join(md))
    (eval_dir / 'trust.json').write_text(json.dumps(trust, indent=2))
    print(f'\nWrote {eval_dir}/trust.md')


if __name__ == '__main__':
    main()
