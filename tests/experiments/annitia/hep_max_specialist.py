"""Hep specialist: a tiny Cox PH on max-ever NIT scores.

Per AUDIT_GAP_ANALYSIS, fibroscan_MAX alone gives C_hep=0.815, while our
226-feature RSF gets 0.864. Sparse model on the strong signals should
outperform a 200+ dim model that can't isolate them.

This script trains a 7-feature Cox PH on:
  - fibs_stiffness_med_BM_1 max-ever (FibroScan kPa max)
  - fibrotest_BM_2 max-ever (FibroTest max)
  - aixp_aix_result_BM_3 max-ever (Aixplorer max)
  - baseline_age (static)
  - ast max-ever
  - ggt max-ever
  - T2DM (static)

5-fold StratifiedKFold (seed=42) on hep event; saves per-fold OOF + refit val_test.
Same for death endpoint (using death event flag).

Output:
  tests/results/annitia/branches/oof_{hepatic,death}_hepmax__sparse.npy
  tests/results/annitia/branches/val_{hepatic,death}_hepmax__sparse.npy
  Plus a sub_master_hepmax_specialist.csv that stacks the hep specialist 60% with
  anchor 40%, death = anchor (since the specialist is hep-targeted).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia        # noqa: E402
from experiments.annitia.survival import concordance         # noqa: E402

BRANCHES = REPO / 'tests/results/annitia/branches'
FULL = REPO / 'tests/results/annitia/full'
VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
ANCHOR = FULL / 'sub_master_meta_avg_aware_weighted.csv'


def _max_ever(longitudinal_for_lab):
    """For each patient, return max-ever observed value (np.nan if no obs)."""
    out = np.full(len(longitudinal_for_lab), np.nan, dtype=float)
    for i, obs in enumerate(longitudinal_for_lab):
        if obs is None:
            continue
        # obs is a list of (age, value) tuples (or array)
        try:
            vals = [v for _, v in obs if np.isfinite(v)]
        except Exception:
            try:
                vals = [v for v in obs if np.isfinite(v)]
            except Exception:
                vals = []
        if vals:
            out[i] = max(vals)
    return out


def build_features(d):
    long = d['longitudinal']
    static = d['static']
    n = len(d['patient_ids'])

    fibroscan_max = _max_ever(long['fibs_stiffness_med_BM_1'])
    fibrotest_max = _max_ever(long['fibrotest_BM_2'])
    aixp_max = _max_ever(long['aixp_aix_result_BM_3'])
    ast_max = _max_ever(long['ast'])
    ggt_max = _max_ever(long['ggt'])
    age = np.asarray(d['baseline_age'], dtype=float)
    t2dm = np.asarray(static['T2DM'], dtype=float)

    X = np.column_stack([fibroscan_max, fibrotest_max, aixp_max, age, ast_max, ggt_max, t2dm])
    cols = ['fibroscan_max', 'fibrotest_max', 'aixp_max', 'age',
            'ast_max', 'ggt_max', 't2dm']
    # impute missing with column mean (Cox needs no nan)
    for j in range(X.shape[1]):
        m = ~np.isfinite(X[:, j])
        if m.any():
            mean = np.nanmean(X[~m, j]) if (~m).any() else 0.0
            X[m, j] = mean
    return X, cols


def fit_cox(X, y, n_folds=5, seed=42):
    """5-fold OOF Cox; returns oof rank predictions."""
    oof = np.full(X.shape[0], np.nan)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_cs = []
    for k, (tr, te) in enumerate(skf.split(X, y['event'].astype(int))):
        # Standardise inside fold
        mu = X[tr].mean(axis=0); sd = X[tr].std(axis=0) + 1e-9
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        model = CoxPHSurvivalAnalysis(alpha=1.0)
        try:
            model.fit(Xtr, y[tr])
        except Exception:
            model = CoxPHSurvivalAnalysis(alpha=10.0)
            model.fit(Xtr, y[tr])
        risk = model.predict(Xte)
        oof[te] = risk
        c = concordance(y[te], risk)
        fold_cs.append(c)
        print(f'  fold {k}: n_te={len(te)} C={c:.4f}')
    print(f'  per-fold mean: {np.mean(fold_cs):.4f} ± {np.std(fold_cs):.4f}')
    print(f'  pooled OOF C: {concordance(y, oof):.4f}')
    return oof, fold_cs, mu, sd


def main():
    print('Loading train + val...')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                        filename=Path(VAL_CSV).name, has_outcomes=False)
    print(f'  train n={len(train["patient_ids"])}, val n={len(val["patient_ids"])}')

    X_tr, cols = build_features(train)
    X_va, _ = build_features(val)
    print(f'  X_tr shape: {X_tr.shape}, columns: {cols}')

    BRANCHES.mkdir(exist_ok=True, parents=True)

    for endpoint in ('hepatic', 'death'):
        print(f'\n=== {endpoint} hep_max specialist ===')
        ep = train[endpoint]
        valid = ep['valid_mask']
        y = ep['y_struct'][valid]
        Xv = X_tr[valid]
        print(f'  valid n={valid.sum()}, events={y["event"].sum()}')

        oof, fold_cs, mu, sd = fit_cox(Xv, y)

        # Pad OOF back to full-train shape (NaN where invalid)
        full_oof = np.full(X_tr.shape[0], np.nan)
        full_oof[valid] = oof

        oof_path = BRANCHES / f'oof_{endpoint}_hepmax__sparse.npy'
        np.save(oof_path, full_oof)
        print(f'  saved {oof_path}')

        # Refit on all valid, predict val_test
        Xv_std = (Xv - mu) / sd
        m = CoxPHSurvivalAnalysis(alpha=1.0)
        try:
            m.fit(Xv_std, y)
        except Exception:
            m = CoxPHSurvivalAnalysis(alpha=10.0); m.fit(Xv_std, y)
        Xva_std = (X_va - mu) / sd
        val_pred = m.predict(Xva_std)
        val_path = BRANCHES / f'val_{endpoint}_hepmax__sparse.npy'
        np.save(val_path, val_pred)
        print(f'  saved {val_path}')

    # Build a candidate that stacks the hep specialist with the anchor.
    print('\n=== Building stacked candidate ===')
    anchor = pd.read_csv(ANCHOR)
    val_meta = pd.read_csv(VAL_CSV)[['trustii_id', 'patient_id_anon']]
    g2i = dict(zip(val_meta['patient_id_anon'].astype(str),
                    val_meta['trustii_id'].astype(int)))
    val_int_ids = np.array([g2i[str(g)] for g in val['patient_ids']])

    hep_pred = np.load(BRANCHES / 'val_hepatic_hepmax__sparse.npy')
    death_pred = np.load(BRANCHES / 'val_death_hepmax__sparse.npy')

    from scipy.stats import rankdata
    def r01(x): return rankdata(x) / len(x)
    # Map anchor's predictions onto the same trustii_id ordering
    anchor_sorted = anchor.set_index('trustii_id').reindex(val_int_ids).reset_index()
    hep_anchor = r01(anchor_sorted['risk_hepatic_event'].values)
    death_anchor = r01(anchor_sorted['risk_death'].values)
    hep_spec = r01(hep_pred)
    death_spec = r01(death_pred)

    # 60% specialist + 40% anchor for hep (specialist dominant);
    # death = anchor (specialist on death is irrelevant for the gap-analysis path)
    hep_blend = 0.6 * hep_spec + 0.4 * hep_anchor
    out = pd.DataFrame({
        'trustii_id': val_int_ids,
        'risk_hepatic_event': hep_blend,
        'risk_death': death_anchor,
    }).sort_values('trustii_id').reset_index(drop=True)
    out_path = FULL / 'sub_master_hepmax_specialist.csv'
    out.to_csv(out_path, index=False)
    print(f'  wrote {out_path}')

    # Also a pure-specialist version (no anchor blend)
    out2 = pd.DataFrame({
        'trustii_id': val_int_ids,
        'risk_hepatic_event': hep_spec,
        'risk_death': death_anchor,
    }).sort_values('trustii_id').reset_index(drop=True)
    out2_path = FULL / 'sub_master_hepmax_pure.csv'
    out2.to_csv(out2_path, index=False)
    print(f'  wrote {out2_path}')


if __name__ == '__main__':
    main()
