"""Trend / ratio feature engineering for hep — never engineered.

Build features:
  - AST/ALT ratio (last + max + slope of ratio)
  - GGT trajectory slope
  - FibroScan max-ever × age interaction
  - PLT × age (low PLT + high age → bad)
  - bilirubin trend slope

Train Cox + RSF on these specifically. Save OOF + val_test predictions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'tests'))
from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.survival import concordance

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
BRANCHES = REPO / 'tests/results/annitia/branches'


def _slope(obs):
    """Linear slope of (age, value) tuples; nan if <2 points."""
    try:
        valid = [(a, v) for a, v in obs if np.isfinite(a) and np.isfinite(v)]
    except Exception:
        valid = []
    if len(valid) < 2:
        return float('nan')
    a = np.array([x for x, _ in valid]); v = np.array([y for _, y in valid])
    if a.std() < 1e-9:
        return float('nan')
    return np.polyfit(a, v, 1)[0]


def _max(obs):
    try: vals = [v for _, v in obs if np.isfinite(v)]
    except:
        try: vals = [v for v in obs if np.isfinite(v)]
        except: vals = []
    return max(vals) if vals else float('nan')


def _last(obs):
    try:
        valid = [(a, v) for a, v in obs if np.isfinite(a) and np.isfinite(v)]
    except Exception:
        valid = []
    if not valid: return float('nan')
    valid.sort()
    return valid[-1][1]


def build_features(d):
    long = d['longitudinal']; static = d['static']
    n = len(d['patient_ids'])
    feats = []
    cols = []

    ast_max = np.array([_max(o) for o in long['ast']])
    alt_max = np.array([_max(o) for o in long['alt']])
    ast_last = np.array([_last(o) for o in long['ast']])
    alt_last = np.array([_last(o) for o in long['alt']])
    ast_slope = np.array([_slope(o) for o in long['ast']])
    alt_slope = np.array([_slope(o) for o in long['alt']])
    ggt_max = np.array([_max(o) for o in long['ggt']])
    ggt_slope = np.array([_slope(o) for o in long['ggt']])
    bili_slope = np.array([_slope(o) for o in long['bilirubin']])
    plt_min = np.array([min([v for _, v in o if np.isfinite(v)] or [np.nan]) for o in long['plt']])
    plt_last = np.array([_last(o) for o in long['plt']])
    fib_max = np.array([_max(o) for o in long['fibs_stiffness_med_BM_1']])
    age = np.asarray(d['baseline_age'], dtype=float)

    aar_max = ast_max / np.where(np.isfinite(alt_max) & (alt_max > 0), alt_max, np.nan)
    aar_last = ast_last / np.where(np.isfinite(alt_last) & (alt_last > 0), alt_last, np.nan)
    aar_slope = ast_slope - alt_slope  # proxy
    fib_age = fib_max * age / 50.0
    plt_age = plt_min * (1 + age / 50.0)

    feats = [aar_max, aar_last, aar_slope, ggt_slope, ggt_max, bili_slope,
             plt_min, plt_last, fib_max, fib_age, plt_age, age,
             ast_max, alt_max, ast_slope, alt_slope]
    cols = ['aar_max', 'aar_last', 'aar_slope', 'ggt_slope', 'ggt_max',
            'bili_slope', 'plt_min', 'plt_last', 'fib_max', 'fib_age',
            'plt_age', 'age', 'ast_max', 'alt_max', 'ast_slope', 'alt_slope']

    X = np.column_stack(feats)
    for j in range(X.shape[1]):
        m = ~np.isfinite(X[:, j])
        if m.any():
            mn = np.nanmean(X[~m, j]) if (~m).any() else 0.0
            X[m, j] = mn
    return X, cols


def main():
    print('Loading...')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                        filename=Path(VAL_CSV).name, has_outcomes=False)
    X_tr, cols = build_features(train)
    X_va, _ = build_features(val)
    print(f'  features: {len(cols)} = {cols}')

    BRANCHES.mkdir(exist_ok=True, parents=True)

    for endpoint in ('hepatic', 'death'):
        print(f'\n=== {endpoint} trend features ===')
        valid = train[endpoint]['valid_mask']
        y = train[endpoint]['y_struct'][valid]
        Xv = X_tr[valid]
        for model_name, ctor in (
            ('cox', lambda: CoxPHSurvivalAnalysis(alpha=1.0)),
            ('rsf-300', lambda: RandomSurvivalForest(n_estimators=300, max_depth=6,
                                                       min_samples_leaf=10, random_state=42, n_jobs=-1)),
        ):
            print(f'  {model_name}...')
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_local = np.full(Xv.shape[0], np.nan)
            for k, (tr, te) in enumerate(skf.split(Xv, y['event'].astype(int))):
                if model_name == 'cox':
                    mu = Xv[tr].mean(axis=0); sd = Xv[tr].std(axis=0) + 1e-9
                    Xtr_s = (Xv[tr] - mu) / sd; Xte_s = (Xv[te] - mu) / sd
                else:
                    Xtr_s, Xte_s = Xv[tr], Xv[te]
                m = ctor()
                try:
                    m.fit(Xtr_s, y[tr])
                    oof_local[te] = m.predict(Xte_s)
                except Exception as e:
                    print(f'    fold {k} failed: {e}'); continue
            ok = np.isfinite(oof_local)
            if ok.sum():
                C = concordance(y[ok], oof_local[ok])
                print(f'    OOF C={C:.4f}')
            full_oof = np.full(X_tr.shape[0], np.nan)
            full_oof[valid] = oof_local
            cid = f'trends_{model_name}__trend16'
            np.save(BRANCHES / f'oof_{endpoint}_{cid}.npy', full_oof)
            # refit + val
            if model_name == 'cox':
                mu = Xv.mean(axis=0); sd = Xv.std(axis=0) + 1e-9
                Xv_s = (Xv - mu) / sd; Xva_s = (X_va - mu) / sd
            else:
                Xv_s, Xva_s = Xv, X_va
            m = ctor()
            try:
                m.fit(Xv_s, y)
                vp = m.predict(Xva_s)
                np.save(BRANCHES / f'val_{endpoint}_{cid}.npy', vp)
                print(f'    saved branches/oof_{endpoint}_{cid}.npy + val')
            except Exception as e:
                print(f'    refit failed: {e}')


if __name__ == '__main__':
    main()
