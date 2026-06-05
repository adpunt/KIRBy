"""Tool 1: Bootstrap C-index CI per candidate variant.

For each sub_master_<variant>.csv we reconstruct the training-side OOF prediction
using `variant_proxy_oof` (from oof_to_leaderboard.py), then bootstrap patients
with replacement (B=1000) and compute hep + death + weighted-LB C-index per
bootstrap. Reports mean / std / 2.5% / 97.5% percentile CI.

Output: tests/results/annitia/full/bootstrap_candidate_ci.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia               # noqa: E402
from experiments.annitia.survival import concordance                # noqa: E402
from experiments.annitia.oof_to_leaderboard import (                 # noqa: E402
    base_oof_dict, load_consensus_oof, variant_proxy_oof,
)

FULL = REPO / 'tests/results/annitia/full'
B = 1000
SEED = 2026


def boot_concordance(y_struct, risk, idx_pool, rng, B):
    """Bootstrap C-index. idx_pool is the index array of valid rows.

    Resampled from idx_pool with replacement. Returns array of length B (NaN if
    a particular sample has 0 events)."""
    out = np.full(B, np.nan)
    n = idx_pool.size
    if n < 5:
        return out
    for b in range(B):
        sel = rng.choice(idx_pool, size=n, replace=True)
        ys = y_struct[sel]
        if not ys['event'].any():
            continue
        try:
            from sksurv.metrics import concordance_index_censored
            c, *_ = concordance_index_censored(ys['event'], ys['time'], risk[sel])
            out[b] = float(c)
        except Exception:
            continue
    return out


def main():
    train = load_annitia()
    h_y = train['hepatic']['y_struct']; h_v = train['hepatic']['valid_mask']
    d_y = train['death']['y_struct'];   d_v = train['death']['valid_mask']

    consensus = load_consensus_oof()
    base_oofs = base_oof_dict()

    csvs = sorted(FULL.glob('sub_master_*.csv'))
    csvs = [c for c in csvs if not c.name.endswith('.summary.json')]
    print(f'[bootstrap] {len(csvs)} candidate variants, B={B}, seed={SEED}')

    # Resume from existing CSV if present
    out_csv = FULL / 'bootstrap_candidate_ci.csv'
    rows = []
    done = set()
    if out_csv.exists():
        try:
            prev = pd.read_csv(out_csv)
            rows = prev.to_dict('records')
            done = set(prev['variant'])
            print(f'[bootstrap] resuming, {len(done)} variants already done')
        except Exception:
            rows = []; done = set()

    rng = np.random.default_rng(SEED)
    for csv in csvs:
        stem = csv.stem.replace('sub_master_', '')
        if stem in done:
            continue
        h_proxy = variant_proxy_oof(stem, base_oofs['hepatic'], consensus['hepatic'])
        d_proxy = variant_proxy_oof(stem, base_oofs['death'],   consensus['death'])

        # Static C-index on full train
        h_finite_mask = h_v & np.isfinite(h_proxy) & np.isfinite(h_y['time'])
        d_finite_mask = d_v & np.isfinite(d_proxy) & np.isfinite(d_y['time'])
        h_idx = np.flatnonzero(h_finite_mask)
        d_idx = np.flatnonzero(d_finite_mask)

        h_static = float(concordance(h_y[h_idx], h_proxy[h_idx])) if h_idx.size >= 5 else float('nan')
        d_static = float(concordance(d_y[d_idx], d_proxy[d_idx])) if d_idx.size >= 5 else float('nan')

        # Bootstrap
        h_boot = boot_concordance(h_y, h_proxy, h_idx, rng, B)
        d_boot = boot_concordance(d_y, d_proxy, d_idx, rng, B)
        # weighted LB per bootstrap (only where both finite)
        w_boot = 0.7 * h_boot + 0.3 * d_boot

        def stats(a):
            ok = np.isfinite(a)
            if ok.sum() < 10:
                return (float('nan'),) * 4
            arr = a[ok]
            return (float(arr.mean()), float(arr.std()),
                    float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

        mh, sh, lh, hh = stats(h_boot)
        md, sd, ld, hd = stats(d_boot)
        mw, sw, lw, hw = stats(w_boot)

        rows.append({
            'variant': stem,
            'n_hep': int(h_idx.size),
            'n_death': int(d_idx.size),
            'static_hep': h_static, 'static_death': d_static,
            'static_weighted_LB': 0.7 * h_static + 0.3 * d_static if np.isfinite(h_static) and np.isfinite(d_static) else float('nan'),
            'mean_hep': mh, 'std_hep': sh, 'ci_hep_low': lh, 'ci_hep_high': hh,
            'mean_death': md, 'std_death': sd, 'ci_death_low': ld, 'ci_death_high': hd,
            'mean_weighted_LB': mw, 'std_weighted_LB': sw,
            'ci_weighted_low': lw, 'ci_weighted_high': hw,
        })
        print(f'  {stem:50s} LB={mw:.4f} ± {sw:.4f}  [{lw:.4f}, {hw:.4f}]', flush=True)

        # Incremental save every variant — protect against time-out
        pd.DataFrame(rows).sort_values('mean_weighted_LB', ascending=False).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows).sort_values('mean_weighted_LB', ascending=False)
    df.to_csv(out_csv, index=False)
    print(f'\n[bootstrap] wrote {out_csv} ({len(df)} variants)')


if __name__ == '__main__':
    main()
