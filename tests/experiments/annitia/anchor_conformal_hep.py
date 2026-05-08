"""Conformal calibration applied to anchor's hep predictions specifically.

Replaces anchor's risk_hepatic_event column with conformally-calibrated rank
based on training-side residuals. Death column kept unchanged.

Output: sub_master_anchor_conformal_hep.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'tests'))
from kirby.datasets.annitia_maslg import load_annitia

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
FULL = REPO / 'tests/results/annitia/full'
BRANCHES = REPO / 'tests/results/annitia/branches'
ANCHOR = FULL / 'sub_master_meta_avg_aware_weighted.csv'


def main():
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                        filename=Path(VAL_CSV).name, has_outcomes=False)

    # Use the proxy_train_oof reconstructed by A37 if available
    proxy = FULL / 'proxy_train_oof_meta_avg_aware_weighted_hepatic.npy'
    if not proxy.exists():
        # fallback: consensus mean of base OOFs
        files = []
        for d in (BRANCHES, REPO / 'tests/results/annitia/be',
                   REPO / 'tests/results/annitia/cd', REPO / 'tests/results/annitia/hep_boost',
                   REPO / 'tests/results/annitia/external'):
            if d.exists(): files.extend(d.glob('oof_hepatic_*.npy'))
        ranks = []; ref_n = None
        for f in files:
            try:
                a = np.load(f)
                if ref_n is None: ref_n = a.shape[0]
                if a.shape[0] != ref_n: continue
                if not np.isfinite(a).any(): continue
                if np.std(a[np.isfinite(a)]) < 1e-10: continue
                m = np.isfinite(a)
                r = np.full(a.shape[0], np.nan)
                if m.sum(): r[m] = rankdata(a[m]) / m.sum()
                ranks.append(r)
            except Exception: continue
        anchor_oof_train = np.nanmean(np.stack(ranks, axis=1), axis=1)
        print(f'  using consensus proxy ({len(ranks)} models)')
    else:
        anchor_oof_train = np.load(proxy)
        print(f'  using A37 reconstructed proxy: {proxy.name}')

    # Map val IDs
    val_meta = pd.read_csv(VAL_CSV)[['trustii_id', 'patient_id_anon']]
    g2i = dict(zip(val_meta['patient_id_anon'].astype(str),
                    val_meta['trustii_id'].astype(int)))
    val_int = np.array([g2i[str(g)] for g in val['patient_ids']])
    anchor = pd.read_csv(ANCHOR).set_index('trustii_id').reindex(val_int).reset_index()

    # Fit isotonic regression on (anchor_train_rank, event_indicator) for valid hep rows
    valid = train['hepatic']['valid_mask']
    event = train['hepatic']['y_struct']['event'].astype(float)
    ok = valid & np.isfinite(anchor_oof_train)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(rankdata(anchor_oof_train[ok]) / ok.sum(), event[ok])

    # Apply to val
    anchor_val_rank = rankdata(anchor['risk_hepatic_event'].values) / len(anchor)
    cal = iso.predict(anchor_val_rank)
    print(f'  iso-cal range: [{cal.min():.3f}, {cal.max():.3f}]; mean={cal.mean():.3f}')
    # Convert calibrated probabilities back to ranks (concordance is rank-based)
    cal_rank = rankdata(cal) / len(cal)

    out = pd.DataFrame({
        'trustii_id': val_int,
        'risk_hepatic_event': cal_rank,
        'risk_death': rankdata(anchor['risk_death'].values) / len(anchor),
    }).sort_values('trustii_id').reset_index(drop=True)
    path = FULL / 'sub_master_anchor_conformal_hep.csv'
    out.to_csv(path, index=False)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
