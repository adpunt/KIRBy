"""Conservative LB prediction from hold-out C with empirical gap calibration.

Past evidence (from leaderboard_history.json):
- rank_median single-variant: OOF 0.9078 → LB 0.9112 (gap −0.003)
- meta_avg_aware_weighted ensemble: OOF unknown → LB 0.9302
- cherry_pick_anomBlend_wapp: OOF 0.9323 → LB 0.898 (gap +0.034)

The gap depends on RECIPE TYPE:
- Pure averaging of N independent base models: gap ≈ 0 (transfer is honest)
- Selection of best base: gap ≈ +0.005-0.015 (mild winner's curse)
- Meta-meta-fitting (anomaly_blend, weighted_anomaly): gap ≈ +0.03-0.05

For our V08-gated methods:
- Hold-out C is by definition not subject to OOF selection bias on hold-out
- BUT we still have inter-method selection bias when we cherry-pick the best
- So expected LB ≈ hold_C - small_selection_bias - noise

Conservative LB band: [hold_C - 2*SE - 0.005, hold_C - 2*SE + 0.005]
Aggressive LB band:  hold_C ± SE (only if we trust hold-out to be representative)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def conservative_band(hold_c, hold_se, recipe_type='ensemble'):
    """Return (low, point, high) LB prediction band.

    recipe_type:
      - 'ensemble':       average of N hold-out-tied methods → small gap
      - 'single_holdout': one method picked by hold-out → mild winner's curse
      - 'meta_meta':      method fits on OOF (cherry-pick, anomaly_blend) → big gap
    """
    if recipe_type == 'ensemble':
        bias = 0.005  # small inter-method selection bias
    elif recipe_type == 'single_holdout':
        bias = 0.015  # canonical winner's curse for max-of-many
    elif recipe_type == 'meta_meta':
        bias = 0.030  # observed empirically on cherry_pick
    else:
        bias = 0.010
    point = hold_c - bias
    low = hold_c - bias - 2 * hold_se
    high = hold_c - bias + 2 * hold_se
    return float(low), float(point), float(high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', default='tests/results/annitia/eval')
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    cmp = json.loads((eval_dir / 'comparison.json').read_text())
    rows = cmp['summary_rows']

    # Combined LB per method (use the same logic as comparison.md)
    combined = {}
    for r in rows:
        m = r['method']
        if m not in combined:
            combined[m] = {}
        combined[m][r['endpoint']] = r

    # Recipe type heuristic: anchor + Hodges-Lehmann + trimmed are 'ensemble',
    # F01 Caruana is 'single_holdout' (because it greedy-selects), F02-F09 vary
    recipe_type = {
        'anchor_rank_mean': 'ensemble',
        'F03_hodges_lehmann': 'ensemble',
        'F09_trimmed_mean': 'ensemble',
        'F02_geomean_rank': 'ensemble',
        'F04_bates_granger': 'single_holdout',  # selects + fits weights
        'F05_stacking_l2': 'single_holdout',    # fits meta-learner
        'F06_diversity_pen': 'single_holdout',
        'F01_caruana_bagged': 'single_holdout',  # bagging dampens but not zero
    }

    print(f'\n  {"method":25s} {"hold LB":>9s} {"±SE":>7s} '
          f'{"recipe":>16s} {"pred LB low":>12s} {"point":>9s} {"high":>9s}')
    rows_md = []
    for m, eps in combined.items():
        if 'hepatic' not in eps or 'death' not in eps:
            continue
        hep = eps['hepatic']
        death = eps['death']
        hold_lb = 0.7 * hep['c_holdout'] + 0.3 * death['c_holdout']
        # combined SE (independent assumption)
        hold_se = float(np.sqrt(0.49 * hep.get('c_holdout_se', 0)**2
                                + 0.09 * death.get('c_holdout_se', 0)**2))
        rt = recipe_type.get(m, 'ensemble')
        low, pt, hi = conservative_band(hold_lb, hold_se, rt)
        print(f'  {m:25s} {hold_lb:>9.4f} {hold_se:>7.4f} {rt:>16s} '
              f'{low:>12.4f} {pt:>9.4f} {hi:>9.4f}')
        rows_md.append((m, hold_lb, hold_se, rt, low, pt, hi))

    rows_md.sort(key=lambda x: -x[5])
    md = ['# Calibrated LB prediction', '',
          'Conservative bands using empirical OOF→LB gap heuristic:',
          '- ensemble: bias = 0.005',
          '- single_holdout: bias = 0.015',
          '- meta_meta: bias = 0.030 (cherry_pick observed)', '',
          '| method | hold LB | ±SE | recipe | pred LB low | point | high |',
          '|---|---|---|---|---|---|---|']
    for m, hl, hs, rt, lo, pt, hi in rows_md:
        md.append(f'| {m} | {hl:.4f} | {hs:.4f} | {rt} | '
                  f'{lo:.4f} | {pt:.4f} | {hi:.4f} |')
    md.append('')
    md.append('## Anchor reference')
    md.append('- Existing LB best `meta_avg_aware_weighted` = 0.9302')
    md.append('- A new submission ONLY makes sense if its `pred LB low` exceeds 0.9302.')
    md.append('- Otherwise the expected outcome is regression or noise — keep the slot.')
    (eval_dir / 'predict_lb.md').write_text('\n'.join(md))
    print(f'\nWrote {eval_dir}/predict_lb.md')


if __name__ == '__main__':
    main()
