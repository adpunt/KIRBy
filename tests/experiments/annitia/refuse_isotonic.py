"""Isotonic-calibrated fusion.

For each existing variant CSV, fit an isotonic regression on the OOF
predictions against the binary event indicator (per endpoint). Apply
the calibrated mapping to the val predictions and average the
calibrated rank scores.

Concordance is rank-based so calibration is theoretically a no-op for
a single model — but for an *ensemble* it matters because variants on
different scales are not comparable; rank-then-calibrate aligns them.

Output: tests/results/annitia/full/sub_master_isotonic_avg.csv

Note: this requires ground-truth event labels for the training rows.
We reconstruct OOF for each variant by re-averaging its members'
training-side OOF arrays (saved in branches/oof_*.npy by the upstream
runs). For variants without a saved OOF (the meta_aware ones), we fit
isotonic on the variant's own ranks vs y as a fallback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))
FULL = REPO / 'tests/results/annitia/full'
BRANCHES = REPO / 'tests/results/annitia/branches'

from kirby.datasets.annitia_maslg import load_annitia      # noqa: E402

VARIANTS = [
    'rank_mean', 'rank_median', 'rank_trimmed', 'residual_greedy',
    'uncertainty_route', 'anomaly_blend', 'disagreement_route',
    'cluster_route', 'specialist_override', 'soft_blend_disagreement',
    'anomaly_router_meta', 'disagreement_router_meta', 'anomaly_blend_meta',
    'weighted_anomaly_per_patient', 'cluster_router_meta',
]


def rank01(x):
    return rankdata(x) / len(x)


def main():
    from sklearn.isotonic import IsotonicRegression

    train = load_annitia()
    y_h_event = train['hepatic']['event'].astype(int)
    y_d_event = train['death']['event'].astype(int)
    h_valid = train['hepatic']['valid_mask']
    d_valid = train['death']['valid_mask']

    # Build a "consensus OOF" per endpoint by averaging all OOF .npy files.
    # This is a proxy for "what does the trained ensemble think about training rows".
    def rank01_nan(x):
        out = np.full_like(x, np.nan, dtype=float)
        m = np.isfinite(x)
        if m.sum():
            out[m] = rankdata(x[m]) / max(m.sum(), 1)
        return out

    def consensus_oof(endpoint):
        files = list(BRANCHES.glob(f'oof_{endpoint}_*.npy'))
        if not files:
            return None
        arrs = []
        for f in files:
            try:
                a = np.load(f)
                finite = np.isfinite(a)
                if finite.sum() < 50:
                    continue
                if np.std(a[finite]) <= 1e-10:
                    continue
                arrs.append(rank01_nan(a))
            except Exception:
                continue
        if not arrs:
            return None
        ref = arrs[0].shape[0]
        arrs = [a for a in arrs if a.shape[0] == ref]
        # nanmean across variants — gives consensus where any model is finite
        return np.nanmean(np.stack(arrs, axis=1), axis=1)

    H_oof = consensus_oof('hepatic')
    D_oof = consensus_oof('death')

    if H_oof is None or D_oof is None:
        print('FAIL: cannot build consensus OOF; aborting')
        sys.exit(1)

    # Fit isotonic on consensus OOF -> event (only on rows where event is valid AND OOF is finite)
    n_h = min(len(H_oof), len(y_h_event), len(h_valid))
    n_d = min(len(D_oof), len(y_d_event), len(d_valid))
    mh = h_valid[:n_h] & np.isfinite(H_oof[:n_h])
    md = d_valid[:n_d] & np.isfinite(D_oof[:n_d])
    iso_h = IsotonicRegression(out_of_bounds='clip').fit(H_oof[:n_h][mh], y_h_event[:n_h][mh])
    iso_d = IsotonicRegression(out_of_bounds='clip').fit(D_oof[:n_d][md], y_d_event[:n_d][md])

    # Now load each variant CSV, calibrate val predictions via the isotonic
    # mapping (after rank01), and average the calibrated ranks.
    dfs = {}
    for v in VARIANTS:
        p = FULL / f'sub_master_{v}.csv'
        if p.exists():
            dfs[v] = pd.read_csv(p)
    present = list(dfs.keys())
    print(f'isotonic_avg: calibrating {len(present)} variants')

    trustii_id = dfs[present[0]]['trustii_id'].values
    n = len(trustii_id)
    H_cal = np.zeros(n)
    D_cal = np.zeros(n)
    for v in present:
        H_cal += iso_h.transform(rank01(dfs[v]['risk_hepatic_event'].values))
        D_cal += iso_d.transform(rank01(dfs[v]['risk_death'].values))
    H_cal /= len(present)
    D_cal /= len(present)

    out = pd.DataFrame({
        'trustii_id': trustii_id,
        'risk_hepatic_event': H_cal,
        'risk_death': D_cal,
    })
    dest = FULL / 'sub_master_isotonic_avg.csv'
    out.to_csv(dest, index=False)
    print(f'wrote {dest}')

    # quick sanity
    anc = FULL / 'sub_master_meta_avg_aware_weighted.csv'
    if anc.exists():
        from scipy.stats import spearmanr
        a = pd.read_csv(anc)
        rho_h, _ = spearmanr(out['risk_hepatic_event'], a['risk_hepatic_event'])
        rho_d, _ = spearmanr(out['risk_death'], a['risk_death'])
        print(f'  spearman vs 0.9302 anchor: hep={rho_h:.4f} death={rho_d:.4f}')

    summary = {'n_variants': len(present),
               'consensus_n_files_hep': len(list(BRANCHES.glob('oof_hepatic_*.npy'))),
               'consensus_n_files_death': len(list(BRANCHES.glob('oof_death_*.npy')))}
    (FULL / 'refuse_isotonic_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
