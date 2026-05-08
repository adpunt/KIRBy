"""Mahalanobis-distance-aware meta fusion: per-patient confidence weighting.

Different from existing sub_master_unc_mahal_blend (which is a 2-way blend
between anchor and rank_median). This builds a true meta layer:
  for each val patient, compute Mahalanobis distance to training centroid
  near patients (low Mahal) → trust anchor (high anchor weight)
  far patients (high Mahal) → use defensive aggregation (rank-trimmed mean)

The weight schedule is continuous (sigmoid on standardised Mahalanobis)
not discrete buckets.

Output: sub_master_mahal_meta.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.covariance import EmpiricalCovariance

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'tests'))
from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.views import build_views

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
FULL = REPO / 'tests/results/annitia/full'
ANCHOR = FULL / 'sub_master_meta_avg_aware_weighted.csv'
RANK_TRIMMED = FULL / 'sub_master_rank_trimmed.csv'  # defensive (20% trim)


def r01(x): return rankdata(x) / len(x)


def main():
    print('Loading...')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                        filename=Path(VAL_CSV).name, has_outcomes=False)
    vt = build_views(train, view_names=('snapshot', 'clinical_pro'))
    vv = build_views(val,   view_names=('snapshot', 'clinical_pro'))
    X_tr = np.concatenate([vt[v][0] for v in ('snapshot', 'clinical_pro')], axis=1)
    X_va = np.concatenate([vv[v][0] for v in ('snapshot', 'clinical_pro')], axis=1)

    # Standardise + Mahalanobis from training centroid
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0) + 1e-9
    Xtr_s = (X_tr - mu) / sd; Xva_s = (X_va - mu) / sd
    cov = EmpiricalCovariance().fit(Xtr_s)
    mahal = np.sqrt(cov.mahalanobis(Xva_s))
    # Sigmoid gate: low mahal → 1 (use anchor); high mahal → 0 (use defensive)
    # Standardise mahal by training-set's mahal distribution
    train_mahal = np.sqrt(cov.mahalanobis(Xtr_s))
    z = (mahal - train_mahal.mean()) / (train_mahal.std() + 1e-9)
    anchor_weight = 1.0 / (1.0 + np.exp(z - 0))  # ≈1 when below mean, →0 above 2σ
    print(f'  val mahal: mean={mahal.mean():.2f}, std={mahal.std():.2f}, '
          f'p95={np.percentile(mahal, 95):.2f}')
    print(f'  anchor_weight: mean={anchor_weight.mean():.3f}, '
          f'min={anchor_weight.min():.3f}, max={anchor_weight.max():.3f}')

    # Map val_int_ids
    val_meta = pd.read_csv(VAL_CSV)[['trustii_id', 'patient_id_anon']]
    g2i = dict(zip(val_meta['patient_id_anon'].astype(str),
                    val_meta['trustii_id'].astype(int)))
    val_int = np.array([g2i[str(g)] for g in val['patient_ids']])

    anchor = pd.read_csv(ANCHOR).set_index('trustii_id').reindex(val_int).reset_index()
    defensive = pd.read_csv(RANK_TRIMMED).set_index('trustii_id').reindex(val_int).reset_index()

    h_anc = r01(anchor['risk_hepatic_event'].values)
    d_anc = r01(anchor['risk_death'].values)
    h_def = r01(defensive['risk_hepatic_event'].values)
    d_def = r01(defensive['risk_death'].values)

    h_out = anchor_weight * h_anc + (1 - anchor_weight) * h_def
    d_out = anchor_weight * d_anc + (1 - anchor_weight) * d_def

    out = pd.DataFrame({
        'trustii_id': val_int,
        'risk_hepatic_event': r01(h_out),
        'risk_death': r01(d_out),
    }).sort_values('trustii_id').reset_index(drop=True)
    path = FULL / 'sub_master_mahal_meta.csv'
    out.to_csv(path, index=False)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
