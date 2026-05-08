"""Failure-focused fusion: per-val-patient model weighting by neighbour-failure-rate.

For each val_test patient, find their K nearest training neighbours. For each
base model, compute its OOF C-index on those neighbours. Use that local C as
the model's weight FOR THAT PATIENT. Models that get the patient's "type" of
patient right dominate the prediction; models that fail in this neighbourhood
are downweighted.

Different from existing variants:
  - anomaly_router_meta: discrete bucket routing
  - hard_specialist: single override on top-30 hardest
  - per_patient_specialist: failed because contested patients have no events
    among 5-NN; this version uses C-index on broader neighbourhood (events
    plus censored), so works even where event-only metrics fail.

Output: sub_master_failure_focused.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'tests'))
from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.views import build_views
from experiments.annitia.survival import concordance

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
FULL = REPO / 'tests/results/annitia/full'
RES = REPO / 'tests/results/annitia'
ANCHOR = FULL / 'sub_master_meta_avg_aware_weighted.csv'
K_NN = 30  # bigger neighbourhood so C-index is computable


def r01(x): return rankdata(x) / len(x)


def load_pool(endpoint, kind):  # kind = 'oof' | 'val'
    pool = {}
    for d in ('branches', 'be', 'cd', 'hep_boost', 'external', 'hparam', 'research', 'research_v2'):
        dp = RES / d
        if not dp.exists(): continue
        for f in sorted(dp.glob(f'{kind}_{endpoint}_*.npy')):
            try:
                a = np.load(f)
                if not np.isfinite(a).any(): continue
                if np.std(a[np.isfinite(a)]) < 1e-10: continue
                cid = f.stem.replace(f'{kind}_{endpoint}_', '')
                if cid not in pool: pool[cid] = a
            except Exception: continue
    return pool


def main():
    print('Loading...')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                        filename=Path(VAL_CSV).name, has_outcomes=False)

    vt = build_views(train, view_names=('snapshot', 'clinical_pro'))
    vv = build_views(val,   view_names=('snapshot', 'clinical_pro'))
    X_tr = np.concatenate([vt[v][0] for v in ('snapshot', 'clinical_pro')], axis=1)
    X_va = np.concatenate([vv[v][0] for v in ('snapshot', 'clinical_pro')], axis=1)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)

    val_meta = pd.read_csv(VAL_CSV)[['trustii_id', 'patient_id_anon']]
    g2i = dict(zip(val_meta['patient_id_anon'].astype(str),
                    val_meta['trustii_id'].astype(int)))
    val_int = np.array([g2i[str(g)] for g in val['patient_ids']])

    out_pred = {}
    for endpoint in ('hepatic', 'death'):
        print(f'\n=== {endpoint} failure-focused ===')
        oof_pool = load_pool(endpoint, 'oof')
        val_pool = load_pool(endpoint, 'val')
        # Filter to models that have BOTH oof and val
        keep = [k for k in oof_pool if k in val_pool
                 and oof_pool[k].shape[0] == X_tr.shape[0]
                 and val_pool[k].shape[0] == X_va.shape[0]]
        print(f'  pool: {len(keep)} models')

        valid = train[endpoint]['valid_mask']
        y = train[endpoint]['y_struct']

        # K-NN restricted to valid training patients (so C-index works)
        valid_idx = np.where(valid)[0]
        nn = NearestNeighbors(n_neighbors=min(K_NN, valid_idx.size)).fit(X_tr_s[valid_idx])

        # Per-val-patient: get K-NN training neighbours (valid only)
        _, nb_local = nn.kneighbors(X_va_s)
        nb_global = valid_idx[nb_local]  # n_val × K

        # For each model, compute local C per val patient on those neighbours
        # Stack ranks of all models for the val predictions
        weights = np.zeros((X_va.shape[0], len(keep)))
        # Precompute per-model OOF rank01 and per-model val rank01
        oof_arr = np.full((X_tr.shape[0], len(keep)), np.nan)
        val_arr = np.full((X_va.shape[0], len(keep)), np.nan)
        for j, k in enumerate(keep):
            oof_arr[:, j] = oof_pool[k]
            val_arr[:, j] = val_pool[k]

        for i in range(X_va.shape[0]):
            nb = nb_global[i]
            # need ≥1 event in nb to compute C
            n_ev = y[nb]['event'].sum()
            if n_ev < 1:
                # fall back to consensus weights (uniform)
                weights[i] = 1.0
                continue
            for j, k in enumerate(keep):
                col = oof_arr[nb, j]
                if not np.isfinite(col).all(): continue
                try:
                    c = concordance(y[nb], col)
                except Exception:
                    c = float('nan')
                if np.isfinite(c) and c > 0.5:
                    weights[i, j] = (c - 0.5) ** 2  # square so good models dominate
                else:
                    weights[i, j] = 0.0
            if weights[i].sum() < 1e-9:
                weights[i] = 1.0  # if no model has signal locally, equal weight
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Weighted rank of val predictions
        val_ranks = np.full_like(val_arr, np.nan)
        for j in range(len(keep)):
            ok = np.isfinite(val_arr[:, j])
            if ok.sum():
                rk = np.full(val_arr.shape[0], np.nan)
                rk[ok] = rankdata(val_arr[ok, j]) / ok.sum()
                val_ranks[:, j] = rk

        # weighted mean rank — handle NaNs by zeroing weight
        ok_mask = np.isfinite(val_ranks)
        w_eff = weights * ok_mask
        w_eff = w_eff / np.maximum(w_eff.sum(axis=1, keepdims=True), 1e-9)
        out = (np.where(ok_mask, val_ranks, 0.0) * w_eff).sum(axis=1)
        out_pred[endpoint] = r01(out)

        print(f'  unique weights per patient: avg n_active_models = '
              f'{(weights > 0.01).sum(axis=1).mean():.1f} / {len(keep)}')

    out_df = pd.DataFrame({
        'trustii_id': val_int,
        'risk_hepatic_event': out_pred['hepatic'],
        'risk_death': out_pred['death'],
    }).sort_values('trustii_id').reset_index(drop=True)
    out_path = FULL / 'sub_master_failure_focused.csv'
    out_df.to_csv(out_path, index=False)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
