"""V2: rigorous test of the no-leak hypothesis.

The V1 result (hep RSF OOF 0.957 at delta=0.25) is partially fake:
truncation removes ~19 obs/event-patient but 0 obs/censored, creating
a visit-count leak. Need to verify the underlying signal is real.

Two cleaner experiments here:

1) **Asymmetric eval** (fit-on-truncated, predict-on-untruncated):
   train Cox/RSF on truncated features (so the model never sees post-event
   leak), but compute OOF predictions using each held-out patient's
   UNTRUNCATED features. Mirrors what happens at val_test, where we
   never truncate. If the +0.09 lift survives, the signal is real.

2) **Landmarking** (uniform observation window):
   Pick tau in {1.0, 2.0} years. Use only labs at age < baseline_age + tau.
   Drop training patients with event_time < tau. Predict events AFTER tau.
   Apply same truncation to val_test (universal age < baseline + tau).
   This is the rigorous causal-survival landmark setup.

Outputs land in tests/results/annitia/research_v2/ as
  oof_<ep>_v2{asym,land}_<delta-or-tau>_<view>__<model>.npy
  val_<ep>_v2land_<tau>_<view>__<model>.npy   (only landmark gets val output;
                                                 asym is a sanity check, not a
                                                 fusion-pool input)
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
import numpy as np
import copy

warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia  # noqa
from experiments.annitia.survival import concordance  # noqa

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
OUT = REPO / 'tests/results/annitia/research_v2'
OUT.mkdir(exist_ok=True, parents=True)


def _load_research():
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        'r', str(Path(__file__).parent / 'experiment_research.py'))
    m = iu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def landmark_truncate(d, tau_years: float):
    """Apply a UNIFORM truncation: keep only labs at age < baseline + tau.

    For training (with outcomes): also returns mask of patients whose
    event_time > tau (must come from caller — we don't know endpoint here).
    """
    out = copy.deepcopy(d)
    base = np.asarray(d['baseline_age'], dtype=float)
    n = len(d['patient_ids'])
    cutoff = base + float(tau_years)
    for var, recs in d['longitudinal'].items():
        new = []
        for i, ser in enumerate(recs):
            new.append([(a, v) for a, v in ser if a < cutoff[i]])
        out['longitudinal'][var] = new
    if 'wide' in d:
        age_arr = d['wide']['Age']
        mask = age_arr < cutoff[:, None]
        new_wide = {}
        for var, arr in d['wide'].items():
            if arr.shape[1] == age_arr.shape[1]:
                new_wide[var] = np.where(mask, arr, np.nan)
            else:
                a2 = np.full((n, age_arr.shape[1]), np.nan)
                k = min(arr.shape[1], age_arr.shape[1])
                a2[:, :k] = arr[:, :k]
                new_wide[var] = np.where(mask, a2, np.nan)
        out['wide'] = new_wide
    return out


def asymmetric_eval():
    from sklearn.model_selection import StratifiedKFold
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest

    r = _load_research()
    train = load_annitia()

    print('=== ASYMMETRIC EVAL (fit on truncated, OOF on untruncated) ===')
    for endpoint in ('hepatic',):
        ep = train[endpoint]
        valid = ep['valid_mask']
        y = ep['y_struct']
        for dlt in (0.0, 0.25, 0.5):
            tr_d = r.truncate_for_endpoint(train, endpoint, dlt) if dlt > 0 else train
            for view in ('biomarker', 'summary', 'both'):
                X_tr_trunc = r.build_X(tr_d, view)
                X_tr_full = r.build_X(train, view)  # untruncated features
                Xv_trunc = X_tr_trunc[valid]
                Xv_full = X_tr_full[valid]
                yv = y[valid]
                t = yv['time'].astype(float); e = yv['event'].astype(int)

                for mn in ('cox', 'rsf-300'):
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    oof = np.full(Xv_trunc.shape[0], np.nan)
                    for k, (tr, te) in enumerate(skf.split(Xv_trunc, e)):
                        if mn == 'cox':
                            mu = np.nanmean(Xv_trunc[tr], axis=0)
                            sd = np.nanstd(Xv_trunc[tr], axis=0) + 1e-9
                            Xtr_s = np.clip(np.where(np.isfinite(Xv_trunc[tr]),
                                                      (Xv_trunc[tr] - mu) / sd, 0.0), -5, 5)
                            Xte_s = np.clip(np.where(np.isfinite(Xv_full[te]),
                                                      (Xv_full[te] - mu) / sd, 0.0), -5, 5)
                            m = CoxPHSurvivalAnalysis(alpha=1.0)
                        else:
                            Xtr_s = np.nan_to_num(Xv_trunc[tr])
                            Xte_s = np.nan_to_num(Xv_full[te])
                            m = RandomSurvivalForest(n_estimators=300, max_depth=6,
                                                      min_samples_leaf=10,
                                                      random_state=42, n_jobs=-1)
                        try:
                            m.fit(Xtr_s, yv[tr])
                            oof[te] = m.predict(Xte_s)
                        except Exception as ex:
                            print(f'  fold {k} fail: {ex}')

                    ok = np.isfinite(oof)
                    if ok.sum() < 100: continue
                    C = concordance(yv[ok], oof[ok])
                    print(f'  ASYM {endpoint} d={dlt:.2f} {view:9s} {mn:8s}  '
                          f'OOF C = {C:.4f}  (n={ok.sum()})')
                    full = np.full(X_tr_trunc.shape[0], np.nan)
                    full[valid] = oof
                    cid = f'v2asym_d{int(dlt*1000)}_{view}__{mn}'
                    np.save(OUT / f'oof_{endpoint}_{cid}.npy', full)


def landmark_eval():
    from sklearn.model_selection import StratifiedKFold
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest

    r = _load_research()
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                       filename=Path(VAL_CSV).name, has_outcomes=False)

    print('\n=== LANDMARK EVAL (uniform truncation at baseline+tau) ===')
    for tau in (1.0, 2.0):
        train_lm = landmark_truncate(train, tau)
        val_lm = landmark_truncate(val, tau)
        for endpoint in ('hepatic', 'death'):
            ep_orig = train[endpoint]
            valid_orig = ep_orig['valid_mask']
            t_orig = ep_orig['time'].astype(float)
            # Drop patients whose event time < tau (cannot predict from t<tau window)
            valid = valid_orig & (t_orig > tau)
            n_drop = int(valid_orig.sum() - valid.sum())
            n_drop_ev = int(((t_orig <= tau) & valid_orig & ep_orig['event'].astype(bool)).sum())
            print(f'  tau={tau:.1f}y {endpoint}: dropped {n_drop} patients ({n_drop_ev} pre-tau events)')
            # Build y_struct for restricted set with t' = t - tau
            y = ep_orig['y_struct'].copy()
            y['time'] = y['time'] - tau

            for view in ('biomarker', 'summary', 'both'):
                X_tr = r.build_X(train_lm, view)
                X_va = r.build_X(val_lm, view)
                Xv = X_tr[valid]; yv = y[valid]
                e = yv['event'].astype(int)

                for mn in ('cox', 'rsf-300'):
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    oof = np.full(Xv.shape[0], np.nan)
                    for k, (tr, te) in enumerate(skf.split(Xv, e)):
                        if mn == 'cox':
                            mu = np.nanmean(Xv[tr], axis=0)
                            sd = np.nanstd(Xv[tr], axis=0) + 1e-9
                            Xtr_s = np.clip(np.where(np.isfinite(Xv[tr]),
                                                      (Xv[tr] - mu) / sd, 0.0), -5, 5)
                            Xte_s = np.clip(np.where(np.isfinite(Xv[te]),
                                                      (Xv[te] - mu) / sd, 0.0), -5, 5)
                            m = CoxPHSurvivalAnalysis(alpha=1.0)
                        else:
                            Xtr_s = np.nan_to_num(Xv[tr])
                            Xte_s = np.nan_to_num(Xv[te])
                            m = RandomSurvivalForest(n_estimators=300, max_depth=6,
                                                      min_samples_leaf=10,
                                                      random_state=42, n_jobs=-1)
                        try:
                            m.fit(Xtr_s, yv[tr])
                            oof[te] = m.predict(Xte_s)
                        except Exception as ex:
                            print(f'    fold {k} fail: {ex}')

                    ok = np.isfinite(oof)
                    if ok.sum() < 100: continue
                    C = concordance(yv[ok], oof[ok])
                    print(f'  LAND tau={tau:.1f} {endpoint} {view:9s} {mn:8s}  '
                          f'OOF C = {C:.4f}  (n={ok.sum()}, n_ev={int(yv["event"][ok].sum())})')
                    full = np.full(X_tr.shape[0], np.nan)
                    full[valid] = oof
                    cid = f'v2land_tau{int(tau*10)}_{view}__{mn}'
                    np.save(OUT / f'oof_{endpoint}_{cid}.npy', full)

                    # Refit on all valid + predict val
                    if mn == 'cox':
                        mu = np.nanmean(Xv, axis=0); sd = np.nanstd(Xv, axis=0) + 1e-9
                        Xv_s = np.clip(np.where(np.isfinite(Xv), (Xv - mu) / sd, 0.0), -5, 5)
                        Xva_s = np.clip(np.where(np.isfinite(X_va), (X_va - mu) / sd, 0.0), -5, 5)
                        mfull = CoxPHSurvivalAnalysis(alpha=1.0)
                    else:
                        Xv_s = np.nan_to_num(Xv); Xva_s = np.nan_to_num(X_va)
                        mfull = RandomSurvivalForest(n_estimators=300, max_depth=6,
                                                      min_samples_leaf=10,
                                                      random_state=42, n_jobs=-1)
                    try:
                        mfull.fit(Xv_s, yv)
                        vp = mfull.predict(Xva_s)
                        np.save(OUT / f'val_{endpoint}_{cid}.npy', vp)
                    except Exception as ex:
                        print(f'    refit fail: {ex}')


def main():
    asymmetric_eval()
    landmark_eval()


if __name__ == '__main__':
    main()
