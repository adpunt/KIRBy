"""KM-survival-prob and Nelson-Aalen-cumulative-hazard target-transform XGBoost.

CIBMTR-Equity 1st-place trick: train ordinary regression on transformed
targets that pre-bake the censored-survival structure, then rank-average.
Each transformation converts the censored survival problem into a
different regression problem that ordinary GBDT solves, and the ensemble
of these on rank space outperforms any single sksurv model.

Per row: target = KM(t_i) where KM is the Kaplan-Meier survival curve at
time t_i (so events get a low S(t), late censorings get a high S(t)),
or target = NA(t_i) where NA is Nelson-Aalen cumulative hazard.

Plus xgboost-survival-AFT and xgboost-survival-Cox heads for diversity.
Then rank-average the four heads per endpoint.

Outputs to tests/results/annitia/research/ as oof_<ep>_kmtarget_<head>.npy
and val_<ep>_kmtarget_<head>.npy. Rank-average head saved as kmtarget_blend.
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia  # noqa
from experiments.annitia.survival import concordance  # noqa

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
OUT = REPO / 'tests/results/annitia/research'
OUT.mkdir(exist_ok=True, parents=True)


def km_curve(times, events):
    """Returns (sorted unique times, survival_at_each, n_at_risk_at_each)."""
    order = np.argsort(times)
    t = times[order]; e = events[order]
    uniq = []
    surv = 1.0
    survs = []
    n_total = len(t)
    n = n_total
    j = 0
    for i, ti in enumerate(t):
        if i + 1 < len(t) and t[i+1] == ti:
            continue
        # Number of events and at-risk at time ti
        # All ties processed at the same step: scan from j to i
        d = int(e[j:i+1].sum())
        if d > 0 and n > 0:
            surv = surv * (1.0 - d / n)
        uniq.append(ti); survs.append(surv)
        # decrement at-risk
        n -= (i + 1 - j)
        j = i + 1
    return np.array(uniq), np.array(survs)


def nelson_aalen(times, events):
    """Cumulative hazard estimator."""
    order = np.argsort(times)
    t = times[order]; e = events[order]
    uniq = []
    H = 0.0
    Hs = []
    n_total = len(t)
    n = n_total
    j = 0
    for i, ti in enumerate(t):
        if i + 1 < len(t) and t[i+1] == ti:
            continue
        d = int(e[j:i+1].sum())
        if n > 0 and d > 0:
            H += d / n
        uniq.append(ti); Hs.append(H)
        n -= (i + 1 - j)
        j = i + 1
    return np.array(uniq), np.array(Hs)


def km_target(times_eval, train_times, train_events):
    uniq, surv = km_curve(train_times, train_events)
    # For each eval time, look up S at the largest training time <= eval
    idx = np.searchsorted(uniq, times_eval, side='right') - 1
    idx = np.clip(idx, 0, len(uniq) - 1)
    out = surv[idx]
    out[times_eval < uniq[0]] = 1.0
    return out


def na_target(times_eval, train_times, train_events):
    uniq, H = nelson_aalen(train_times, train_events)
    idx = np.searchsorted(uniq, times_eval, side='right') - 1
    idx = np.clip(idx, 0, len(uniq) - 1)
    out = H[idx]
    out[times_eval < uniq[0]] = 0.0
    return out


def build_features():
    """Reuse the build_X 'both' view from experiment_research."""
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        'exp_research', str(Path(__file__).parent / 'experiment_research.py'))
    mod = iu.module_from_spec(spec); spec.loader.exec_module(mod)
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                       filename=Path(VAL_CSV).name, has_outcomes=False)
    # Use no-leak truncated train for hep, normal train for death
    X_va = mod.build_X(val, 'both')
    return train, val, X_va, mod


def fit_xgb(X_tr, y_tr_target, X_va, X_te=None, objective='reg:squarederror',
            n_estimators=600, max_depth=4, lr=0.05, params_extra=None,
            seed=42, weight=None):
    try:
        import xgboost as xgb
    except ImportError:
        return None, None
    params = dict(
        objective=objective, n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    if params_extra:
        params.update(params_extra)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tr, y_tr_target, sample_weight=weight)
    pv = m.predict(X_va)
    pte = m.predict(X_te) if X_te is not None else None
    return pv, pte


def fit_xgb_aft(X_tr, t, e, X_va, X_te=None, n_estimators=600, max_depth=4,
                 lr=0.05, sigma=1.0):
    try:
        import xgboost as xgb
    except ImportError:
        return None, None
    # AFT requires DMatrix with set_float_info('label_lower_bound', ...)
    # and 'label_upper_bound'
    y_low = t.copy()
    y_up = np.where(e == 1, t, np.inf)
    dtr = xgb.DMatrix(X_tr)
    dtr.set_float_info('label_lower_bound', y_low)
    dtr.set_float_info('label_upper_bound', y_up)
    params = {
        'objective': 'survival:aft',
        'aft_loss_distribution': 'normal',
        'aft_loss_distribution_scale': sigma,
        'tree_method': 'hist', 'eta': lr, 'max_depth': max_depth,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'verbosity': 0,
    }
    bst = xgb.train(params, dtr, num_boost_round=n_estimators)
    pv = bst.predict(xgb.DMatrix(X_va))
    pte = bst.predict(xgb.DMatrix(X_te)) if X_te is not None else None
    # AFT predicts log-time → invert sign so HIGH risk corresponds to early event
    return -pv, (-pte if pte is not None else None)


def fit_xgb_cox(X_tr, t, e, X_va, X_te=None, n_estimators=600, max_depth=4,
                 lr=0.05):
    try:
        import xgboost as xgb
    except ImportError:
        return None, None
    # XGBoost survival:cox uses signed targets: positive=event-time, negative=censor-time
    y = np.where(e == 1, t, -t)
    dtr = xgb.DMatrix(X_tr, label=y)
    params = {
        'objective': 'survival:cox',
        'tree_method': 'hist', 'eta': lr, 'max_depth': max_depth,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'verbosity': 0,
    }
    bst = xgb.train(params, dtr, num_boost_round=n_estimators)
    pv = bst.predict(xgb.DMatrix(X_va))  # log-partial-hazard, larger = more risk
    pte = bst.predict(xgb.DMatrix(X_te)) if X_te is not None else None
    return pv, pte


def main():
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import rankdata

    train, val, X_va, mod = build_features()

    for endpoint in ('hepatic', 'death'):
        print(f'\n=== {endpoint} KM/NA target XGB ===')
        # Use no-leak train for hep
        if endpoint == 'hepatic':
            tr_d = mod.truncate_for_endpoint(train, 'hepatic', 0.5)
        else:
            tr_d = train
        X_tr = mod.build_X(tr_d, 'both')
        valid = train[endpoint]['valid_mask']
        y = train[endpoint]['y_struct']
        Xv = X_tr[valid]; yv = y[valid]
        t = yv['time'].astype(float); e = yv['event'].astype(int)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        n_v = Xv.shape[0]
        oof_km = np.full(n_v, np.nan)
        oof_na = np.full(n_v, np.nan)
        oof_aft = np.full(n_v, np.nan)
        oof_cox = np.full(n_v, np.nan)

        for k, (tr, te) in enumerate(skf.split(Xv, e)):
            t_tr, t_te = t[tr], t[te]; e_tr, e_te = e[tr], e[te]
            # KM/NA targets fit on training fold's KM curve
            km_tar_tr = km_target(t_tr, t_tr, e_tr)
            # KM target at TIME t_i — high = will survive, low = will die
            # We invert sign so larger = higher hazard (consistent C-index direction)
            pv_km, _ = fit_xgb(Xv[tr], -km_tar_tr, Xv[te])
            if pv_km is not None:
                oof_km[te] = pv_km
            na_tar_tr = na_target(t_tr, t_tr, e_tr)
            pv_na, _ = fit_xgb(Xv[tr], na_tar_tr, Xv[te])
            if pv_na is not None:
                oof_na[te] = pv_na
            pv_aft, _ = fit_xgb_aft(Xv[tr], t_tr, e_tr, Xv[te])
            if pv_aft is not None:
                oof_aft[te] = pv_aft
            pv_cox, _ = fit_xgb_cox(Xv[tr], t_tr, e_tr, Xv[te])
            if pv_cox is not None:
                oof_cox[te] = pv_cox

        for name, oof in (('km', oof_km), ('na', oof_na),
                          ('aft', oof_aft), ('cox', oof_cox)):
            ok = np.isfinite(oof)
            if ok.sum() < 10: continue
            C = concordance(yv[ok], oof[ok])
            print(f'  {name}: OOF C = {C:.4f} (n={ok.sum()})')
            full = np.full(X_tr.shape[0], np.nan)
            full[valid] = oof
            np.save(OUT / f'oof_{endpoint}_kmtarget_{name}.npy', full)

        # Rank-average blend
        ranks = []
        for o in (oof_km, oof_na, oof_aft, oof_cox):
            ok = np.isfinite(o)
            if ok.sum() < 10: continue
            r = np.full_like(o, np.nan)
            r[ok] = rankdata(o[ok]) / ok.sum()
            ranks.append(r)
        if ranks:
            blend = np.nanmean(np.column_stack(ranks), axis=1)
            ok = np.isfinite(blend)
            C = concordance(yv[ok], blend[ok])
            print(f'  blend: OOF C = {C:.4f}')
            full = np.full(X_tr.shape[0], np.nan)
            full[valid] = blend
            np.save(OUT / f'oof_{endpoint}_kmtarget_blend.npy', full)

        # Refit each head on all valid data + predict val
        # KM/NA: target is computed on full training KM curve
        km_tar_all = km_target(t, t, e)
        for name, fn, args in (
            ('km', lambda Xtr, Xva, w=None: fit_xgb(Xtr, -km_tar_all, Xva, weight=w),
                None),
            ('na', lambda Xtr, Xva, w=None: fit_xgb(Xtr, na_target(t, t, e), Xva, weight=w),
                None),
            ('aft', lambda Xtr, Xva: fit_xgb_aft(Xtr, t, e, Xva), None),
            ('cox', lambda Xtr, Xva: fit_xgb_cox(Xtr, t, e, Xva), None),
        ):
            try:
                if name in ('km', 'na'):
                    pv, _ = fn(Xv, X_va)
                else:
                    pv, _ = fn(Xv, X_va)
                if pv is not None:
                    np.save(OUT / f'val_{endpoint}_kmtarget_{name}.npy', pv)
            except Exception as ex:
                print(f'  refit-val {name}: {ex}')

        # Blend val: rank-average the four val arrays
        try:
            vs = []
            for name in ('km', 'na', 'aft', 'cox'):
                p = OUT / f'val_{endpoint}_kmtarget_{name}.npy'
                if p.exists():
                    a = np.load(p)
                    if np.std(a) > 1e-10:
                        vs.append(rankdata(a) / len(a))
            if vs:
                blend_v = np.mean(np.column_stack(vs), axis=1)
                np.save(OUT / f'val_{endpoint}_kmtarget_blend.npy', blend_v)
        except Exception as ex:
            print(f'  val blend: {ex}')


if __name__ == '__main__':
    main()
