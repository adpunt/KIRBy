"""NNLS-stacked fusion: constrained-positive learned weights.

Failure mode of unconstrained Coxnet stack_meta on round 3 was negative
weights overfitting. NNLS forces non-negative weights — can't subtract
one model from another, so much harder to overfit on small samples.

Per-fold nested CV: for each outer fold, fit NNLS on the 4 training
folds' OOFs against the binary event, predict on the held-out fold.
At inference, fit NNLS on full training OOFs, apply to val ranks.

Output: tests/results/annitia/full/sub_master_nnls_stack.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import nnls
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))
FULL = REPO / 'tests/results/annitia/full'
BRANCHES = REPO / 'tests/results/annitia/branches'

from kirby.datasets.annitia_maslg import load_annitia              # noqa: E402
from experiments.annitia.survival import concordance               # noqa: E402


def rank01_nan(x):
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x)
    if m.sum():
        out[m] = rankdata(x[m]) / max(m.sum(), 1)
    return out


def collect_recs(endpoint: str):
    """Returns list of dicts {label, oof, val} for all branches with both files."""
    files = list(BRANCHES.glob(f'oof_{endpoint}_*.npy'))
    recs = []
    for of in files:
        cid = of.name.replace(f'oof_{endpoint}_', '').replace('.npy', '')
        vf = BRANCHES / f'val_{endpoint}_{cid}.npy'
        if not vf.exists():
            continue
        try:
            oof = np.load(of)
            val = np.load(vf)
            if np.isfinite(oof).sum() < 50:
                continue
            recs.append({'label': cid, 'oof': oof, 'val': val})
        except Exception:
            continue
    return recs


def nnls_stack_endpoint(endpoint: str, y_event: np.ndarray,
                       y_struct: np.ndarray, valid: np.ndarray):
    recs = collect_recs(endpoint)
    if len(recs) < 3:
        print(f'  {endpoint}: too few branches ({len(recs)}); skipping')
        return None, None

    n_train = len(y_event)
    n_val = recs[0]['val'].shape[0]
    R_oof = np.stack([rank01_nan(r['oof'])[:n_train] for r in recs], axis=1)
    R_val = np.stack([rank01_nan(r['val'])[:n_val] for r in recs], axis=1)
    # impute NaN (from invalid rows) with column median
    col_med = np.nanmedian(R_oof, axis=0)
    inds = np.where(np.isnan(R_oof))
    R_oof[inds] = np.take(col_med, inds[1])
    val_col_med = np.nanmedian(R_val, axis=0)
    inds_v = np.where(np.isnan(R_val))
    R_val[inds_v] = np.take(val_col_med, inds_v[1])

    # Nested 5-fold to get honest OOF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    nested_oof = np.full(n_train, np.nan)
    valid_int = np.where(valid)[0]
    for fold_id, (tr, te) in enumerate(kf.split(valid_int)):
        tr_idx = valid_int[tr]
        te_idx = valid_int[te]
        w, _ = nnls(R_oof[tr_idx], y_event[tr_idx].astype(float))
        if w.sum() > 0:
            w = w / w.sum()
        nested_oof[te_idx] = R_oof[te_idx] @ w

    # Concordance on the nested OOF
    finite = valid & np.isfinite(nested_oof)
    if finite.sum() >= 5:
        oof_c = float(concordance(y_struct[finite], nested_oof[finite]))
    else:
        oof_c = float('nan')
    print(f'  {endpoint}: nested OOF C = {oof_c:.4f}, n_branches = {len(recs)}')

    # Final fit on all valid training rows
    w_final, _ = nnls(R_oof[valid], y_event[valid].astype(float))
    if w_final.sum() > 0:
        w_final = w_final / w_final.sum()
    val_pred = R_val @ w_final

    return val_pred, oof_c


def main():
    train = load_annitia()
    print('NNLS stack: fitting per endpoint with nested 5-fold CV')

    # Prefer trustii_id from existing CSV (val ordering)
    anchor_csv = FULL / 'sub_master_meta_avg_aware_weighted.csv'
    if not anchor_csv.exists():
        print(f'FAIL: anchor {anchor_csv} not found'); sys.exit(1)
    anchor = pd.read_csv(anchor_csv)
    trustii_id = anchor['trustii_id'].values

    h_pred, h_c = nnls_stack_endpoint(
        'hepatic',
        train['hepatic']['event'].astype(int),
        train['hepatic']['y_struct'],
        train['hepatic']['valid_mask'],
    )
    d_pred, d_c = nnls_stack_endpoint(
        'death',
        train['death']['event'].astype(int),
        train['death']['y_struct'],
        train['death']['valid_mask'],
    )

    if h_pred is None or d_pred is None:
        print('FAIL: missing predictions; aborting')
        sys.exit(1)

    out = pd.DataFrame({
        'trustii_id': trustii_id,
        'risk_hepatic_event': h_pred,
        'risk_death': d_pred,
    })
    dest = FULL / 'sub_master_nnls_stack.csv'
    out.to_csv(dest, index=False)
    print(f'wrote {dest}')

    summary = {'hepatic_oof_c': h_c, 'death_oof_c': d_c}
    (FULL / 'refuse_nnls_summary.json').write_text(json.dumps(summary, indent=2))

    # sanity vs anchor
    from scipy.stats import spearmanr
    rho_h, _ = spearmanr(out['risk_hepatic_event'], anchor['risk_hepatic_event'])
    rho_d, _ = spearmanr(out['risk_death'], anchor['risk_death'])
    print(f'  spearman vs 0.9302 anchor: hep={rho_h:.4f} death={rho_d:.4f}')


if __name__ == '__main__':
    main()
