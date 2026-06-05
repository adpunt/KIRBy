"""Tool 2: Hold-out fusion-test set (F.36 from GENERALIZATION_IDEAS.md).

Carve off 10% of training (per endpoint, since hep and death have different
valid sets). Score each fusion variant on the 10% holdout — the closest local
analog to LB.

Implementation note: existing 5-fold OOFs are used as base predictions on every
training row, including the 10% holdout. For deterministic fusion variants
(rank_mean, median, trimmed mean, RRF, equal-weight meta), this gives the
correct holdout prediction. For data-driven (NNLS, isotonic, anomaly routing,
cluster routing), the variant_proxy_oof fallback already approximates with the
mean of base OOFs — the holdout score therefore measures the *deterministic*
component only, but ranking across variants is still informative.

Output: tests/results/annitia/full/holdout_fusion_scores.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia               # noqa: E402
from experiments.annitia.survival import concordance                # noqa: E402
from experiments.annitia.oof_to_leaderboard import (                 # noqa: E402
    base_oof_dict, load_consensus_oof, variant_proxy_oof,
)

FULL = REPO / 'tests/results/annitia/full'
HOLDOUT_FRAC = 0.10
SEED = 2026


def holdout_indices(y_struct, valid_mask, frac=HOLDOUT_FRAC, seed=SEED):
    """Return (fit_idx, holdout_idx) on rows where valid_mask is True.
    Stratified on event column."""
    pool = np.flatnonzero(valid_mask)
    events = y_struct['event'][pool]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    fit_local, hold_local = next(sss.split(pool, events))
    return pool[fit_local], pool[hold_local]


def main():
    train = load_annitia()
    h_y = train['hepatic']['y_struct']; h_v = train['hepatic']['valid_mask']
    d_y = train['death']['y_struct'];   d_v = train['death']['valid_mask']

    h_fit, h_hold = holdout_indices(h_y, h_v)
    d_fit, d_hold = holdout_indices(d_y, d_v)
    print(f'[holdout] hep: fit={h_fit.size} hold={h_hold.size} '
          f'(events_hold={int(h_y["event"][h_hold].sum())})')
    print(f'[holdout] death: fit={d_fit.size} hold={d_hold.size} '
          f'(events_hold={int(d_y["event"][d_hold].sum())})')

    consensus = load_consensus_oof()
    base_oofs = base_oof_dict()

    csvs = sorted(FULL.glob('sub_master_*.csv'))
    csvs = [c for c in csvs if not c.name.endswith('.summary.json')]
    print(f'[holdout] {len(csvs)} variants')

    rows = []
    for csv in csvs:
        stem = csv.stem.replace('sub_master_', '')
        h_proxy = variant_proxy_oof(stem, base_oofs['hepatic'], consensus['hepatic'])
        d_proxy = variant_proxy_oof(stem, base_oofs['death'],   consensus['death'])

        # Score on holdout indices only
        h_use = h_hold[np.isfinite(h_proxy[h_hold]) & np.isfinite(h_y['time'][h_hold])]
        d_use = d_hold[np.isfinite(d_proxy[d_hold]) & np.isfinite(d_y['time'][d_hold])]

        h_c = float(concordance(h_y[h_use], h_proxy[h_use])) if h_use.size >= 5 else float('nan')
        d_c = float(concordance(d_y[d_use], d_proxy[d_use])) if d_use.size >= 5 else float('nan')
        if np.isfinite(h_c) and np.isfinite(d_c):
            lb = 0.7 * h_c + 0.3 * d_c
        else:
            lb = float('nan')

        rows.append({
            'variant': stem,
            'n_hep_hold': int(h_use.size),
            'n_death_hold': int(d_use.size),
            'holdout_C_hep': h_c,
            'holdout_C_death': d_c,
            'holdout_LB': lb,
        })
        print(f'  {stem:50s} holdout LB={lb:.4f}  (hep {h_c:.4f} / death {d_c:.4f})')

    df = pd.DataFrame(rows).sort_values('holdout_LB', ascending=False)
    out = FULL / 'holdout_fusion_scores.csv'
    df.to_csv(out, index=False)
    print(f'\n[holdout] wrote {out} ({len(df)} variants)')


if __name__ == '__main__':
    main()
