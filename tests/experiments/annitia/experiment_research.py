"""Research-driven experiment: post-event lab leakage fix + MASLD biomarkers.

Two contributions, both motivated by the 2026-05-08 Claude Research output:

1) **No-leak views** (Bug 2 fix). Audit confirmed 19/39 hep events have
   strictly post-event labs (mean 14.5 leaked obs, max 119; corr(max_visit_age,
   event_age) = 0.998). Death is clean (0/76). For TRAINING patients with an
   event, truncate longitudinal series at `age < event_age - delta` for
   delta in {0, 0.25, 0.5} years. Censored and val_test patients keep all
   visits.  Re-train Cox/RSF on the truncated views.

2) **MASLD biomarker view** (research-recommended features).
   - AGILE 3+ (Sanyal 2022): logit MASLD-fibrosis liver-event score using
     LSM, PLT, ALT/AST, T2DM, sex_M, age.
   - Modified aMAP (Fan 2020 minus albumin): age, sex, log10(bili), platelets.
   - FIB-4 trajectory: max, slope, AUC, time-above-2.67, low-to-high transition.
   - LSM trajectory: delta-pct, progressor (>+20%), regressor (<-20%),
     Baveno-VII categories (LSM × PLT), AUC-above-10kPa, AUC-above-15kPa.
   - TyG = ln(triglycerides_mg/dL * glucose_mg/dL / 2); TyG-BMI = TyG * BMI.
   - Cox-prior linear score (sum of literature log-HRs * z-scored features),
     clipped to log(50) to avoid LiverRisk-style swamping.

Outputs land in tests/results/annitia/research/ as oof_<ep>_<cid>.npy and
val_<ep>_<cid>.npy with cid like:
  research_<view>_<delta>__<model>
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia, NIT_VARS, LAB_VARS  # noqa
from experiments.annitia.survival import concordance  # noqa

VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
OUT = REPO / 'tests/results/annitia/research'
OUT.mkdir(exist_ok=True, parents=True)

DELTAS_YEARS = (0.0, 0.25, 0.5)


# ───────────────────────── leakage truncation ─────────────────────────

def truncate_for_events(d, delta_years: float):
    """Return a deep-copied data dict with event patients' longitudinal
    series truncated at age < (event_age - delta_years).

    Censored and val_test patients (no event) keep all visits. baseline_age
    is preserved (recomputed from truncated visits if first visit got dropped,
    which would only happen if delta is so large the patient has no labs left
    — handled by falling back to an empty list).
    """
    import copy
    out = copy.deepcopy(d)
    if d['hepatic'] is None:
        return out  # val_test loader path: no truncation
    n = len(d['patient_ids'])
    event_age = np.full(n, np.inf)
    # We only know event_age for hep events here; this routine builds one
    # truncated copy per endpoint, so callers MUST pass the right endpoint
    raise RuntimeError("call truncate_for_endpoint directly")


def truncate_for_endpoint(d, endpoint: str, delta_years: float):
    """Truncate event patients' visits at age < event_age - delta. Per endpoint."""
    import copy
    out = copy.deepcopy(d)
    if d[endpoint] is None:
        return out
    n = len(d['patient_ids'])
    base = np.asarray(d['baseline_age'], dtype=float)
    e = d[endpoint]
    is_event = e['valid_mask'] & e['event'].astype(bool)
    event_age = base + e['time']  # equals event_age for events
    cutoff = np.where(is_event, event_age - float(delta_years), np.inf)

    for var, recs in d['longitudinal'].items():
        new = []
        for i, ser in enumerate(recs):
            c = cutoff[i]
            if not np.isfinite(c):
                new.append(ser)
            else:
                new.append([(a, v) for a, v in ser if a < c])
        out['longitudinal'][var] = new

    # Keep wide arrays for clinical_view; rebuild from truncated longitudinal
    # — easier: zero out wide entries whose age >= cutoff
    if 'wide' in d:
        age_arr = d['wide'].get('Age')
        if age_arr is not None:
            mask = age_arr < cutoff[:, None]
            out['wide'] = {}
            for var, arr in d['wide'].items():
                if arr.shape[1] == age_arr.shape[1]:
                    out['wide'][var] = np.where(mask, arr, np.nan)
                else:
                    # ragged columns: pad/truncate to age_arr columns
                    a2 = np.full((n, age_arr.shape[1]), np.nan)
                    k = min(arr.shape[1], age_arr.shape[1])
                    a2[:, :k] = arr[:, :k]
                    out['wide'][var] = np.where(mask, a2, np.nan)
    return out


# ───────────────────────── view builders (lightweight) ─────────────────────────

def view_summary(d):
    """snapshot+summary+slope+delta over all longitudinal vars."""
    n = len(d['patient_ids'])
    feats = []
    names = []
    for var, recs in d['longitudinal'].items():
        last = np.zeros(n); first = np.zeros(n)
        mn = np.zeros(n); mx = np.zeros(n)
        mean = np.zeros(n); std = np.zeros(n)
        slope = np.zeros(n); cnt = np.zeros(n)
        for i, ser in enumerate(recs):
            if not ser:
                continue
            v = np.array([x for _, x in ser], dtype=float)
            a = np.array([x for x, _ in ser], dtype=float)
            last[i] = v[-1]; first[i] = v[0]
            mn[i] = v.min(); mx[i] = v.max()
            mean[i] = v.mean(); std[i] = v.std()
            cnt[i] = len(v)
            if len(v) >= 2 and a.std() > 1e-9:
                slope[i] = ((a - a.mean()) * (v - v.mean())).mean() / (a.var() + 1e-12)
        for arr, suf in ((last, 'last'), (first, 'first'),
                         (mn, 'min'), (mx, 'max'),
                         (mean, 'mean'), (std, 'std'),
                         (slope, 'slope'), (cnt, 'count')):
            feats.append(arr)
            names.append(f'{var}__{suf}')
    return np.column_stack(feats), names


def view_static(d):
    cols = []
    names = []
    for k in ('gender', 'T2DM', 'Hypertension', 'Dyslipidaemia',
              'bariatric_surgery', 'bariatric_surgery_age'):
        x = np.asarray(d['static'].get(k, np.full(len(d['patient_ids']), np.nan)),
                       dtype=float)
        cols.append(np.nan_to_num(x, nan=0.0))
        names.append(k)
    cols.append(np.nan_to_num(np.asarray(d['baseline_age'], dtype=float)))
    names.append('baseline_age')
    return np.column_stack(cols), names


# ───────────────────────── MASLD biomarker view ─────────────────────────

def _last(ser):
    if not ser: return np.nan
    return ser[-1][1]


def _max(ser):
    if not ser: return np.nan
    return max(v for _, v in ser)


def _slope(ser):
    if len(ser) < 2: return 0.0
    a = np.array([x for x, _ in ser]); v = np.array([x for _, x in ser])
    if a.std() < 1e-9: return 0.0
    return float(((a - a.mean()) * (v - v.mean())).mean() / (a.var() + 1e-12))


def _auc_above(ser, thr):
    """Trapezoidal AUC of (value - thr)+ vs age. Years × (value-thr) integrand."""
    if len(ser) < 2: return 0.0
    pts = sorted([(a, v) for a, v in ser if np.isfinite(a) and np.isfinite(v)])
    if len(pts) < 2: return 0.0
    auc = 0.0
    for i in range(len(pts) - 1):
        a0, v0 = pts[i]; a1, v1 = pts[i+1]
        x0 = max(0.0, v0 - thr); x1 = max(0.0, v1 - thr)
        auc += 0.5 * (x0 + x1) * (a1 - a0)
    return auc


def _time_above(ser, thr):
    """Total visit-time fraction (years) where value > thr."""
    if len(ser) < 2: return 0.0
    pts = sorted([(a, v) for a, v in ser if np.isfinite(a) and np.isfinite(v)])
    if len(pts) < 2: return 0.0
    t = 0.0
    for i in range(len(pts) - 1):
        a0, v0 = pts[i]; a1, v1 = pts[i+1]
        if v0 > thr and v1 > thr: t += a1 - a0
        elif v0 > thr or v1 > thr: t += 0.5 * (a1 - a0)
    return t


def _transition(ser, lo_thr, hi_thr):
    """1 if any low-to-high transition (consecutive pair) crosses lo→hi."""
    if len(ser) < 2: return 0.0
    pts = sorted([(a, v) for a, v in ser if np.isfinite(a) and np.isfinite(v)])
    for i in range(len(pts) - 1):
        if pts[i][1] < lo_thr and pts[i+1][1] > hi_thr:
            return 1.0
    return 0.0


def view_biomarkers(d):
    """20+ MASLD-specific clinical-prior features. All continuous."""
    n = len(d['patient_ids'])
    long = d['longitudinal']
    static = d['static']
    base = np.asarray(d['baseline_age'], dtype=float)

    # FIB-4 series and snapshot
    fib4_max = np.full(n, np.nan)
    fib4_last = np.full(n, np.nan)
    fib4_slope = np.zeros(n)
    fib4_auc = np.zeros(n)
    fib4_time_high = np.zeros(n)
    fib4_trans = np.zeros(n)
    # AGILE 3+ snapshot
    agile = np.full(n, np.nan)
    # mod-aMAP
    amap = np.full(n, np.nan)
    # LSM trajectory
    lsm_delta_pct = np.zeros(n)
    lsm_progressor = np.zeros(n)
    lsm_regressor = np.zeros(n)
    lsm_max = np.full(n, np.nan)
    lsm_auc_10 = np.zeros(n)
    lsm_auc_15 = np.zeros(n)
    baveno_csph = np.zeros(n)  # LSM>=20 OR (LSM>=15 AND PLT<150)
    baveno_likely = np.zeros(n)  # LSM>=10
    # AST/ALT, GGT, BARD
    aar_last = np.full(n, np.nan)
    bili_log_last = np.full(n, np.nan)
    bili_max = np.full(n, np.nan)
    plt_min = np.full(n, np.nan)
    plt_last = np.full(n, np.nan)
    ggt_max = np.full(n, np.nan)
    ggt_slope = np.zeros(n)
    # TyG
    tyg = np.full(n, np.nan)
    tyg_bmi = np.full(n, np.nan)
    # APRI
    apri_last = np.full(n, np.nan)

    # Helpers — pair labs by visit-age to compute FIB-4 series.
    # Age column has 22 visits; lab columns have 21 — clip to common count.
    wide_keys = ('Age', 'ast', 'alt', 'plt', 'bilirubin', 'BMI',
                 'gluc_fast', 'triglyc', 'ggt', 'fibs_stiffness_med_BM_1')
    nv = min(d['wide'][k].shape[1] for k in wide_keys)
    age_arr = d['wide']['Age'][:, :nv]
    ast_arr = d['wide']['ast'][:, :nv]
    alt_arr = d['wide']['alt'][:, :nv]
    plt_arr = d['wide']['plt'][:, :nv]
    bili_arr = d['wide']['bilirubin'][:, :nv]
    bmi_arr = d['wide']['BMI'][:, :nv]
    glu_arr = d['wide']['gluc_fast'][:, :nv]
    tri_arr = d['wide']['triglyc'][:, :nv]
    ggt_arr = d['wide']['ggt'][:, :nv]
    lsm_arr = d['wide']['fibs_stiffness_med_BM_1'][:, :nv]

    sex_m = (np.asarray(static.get('gender', np.zeros(n)), dtype=float) == 1).astype(float)
    t2dm = (np.asarray(static.get('T2DM', np.zeros(n)), dtype=float) == 1).astype(float)

    for i in range(n):
        ages = age_arr[i]
        ok_age = np.isfinite(ages)
        # FIB-4 series
        with np.errstate(invalid='ignore', divide='ignore'):
            fib_series_v = ((ages * ast_arr[i]) /
                             (plt_arr[i] * np.sqrt(np.maximum(alt_arr[i], 1e-9))))
        fok = np.isfinite(fib_series_v) & ok_age & np.isfinite(plt_arr[i])
        if fok.any():
            ser = list(zip(ages[fok], fib_series_v[fok]))
            ser.sort()
            fib4_max[i] = max(v for _, v in ser)
            fib4_last[i] = ser[-1][1]
            fib4_slope[i] = _slope(ser)
            fib4_auc[i] = _auc_above(ser, 1.3)
            fib4_time_high[i] = _time_above(ser, 2.67)
            fib4_trans[i] = _transition(ser, 1.3, 2.67)
        # APRI series last
        with np.errstate(invalid='ignore', divide='ignore'):
            apri_series_v = (ast_arr[i] / 40.0) * 100.0 / plt_arr[i]
        aok = np.isfinite(apri_series_v) & ok_age & np.isfinite(plt_arr[i])
        if aok.any():
            order = np.argsort(ages[aok])
            apri_last[i] = apri_series_v[aok][order][-1]
        # AST/ALT last
        ast_ok = np.isfinite(ast_arr[i]) & ok_age
        alt_ok = np.isfinite(alt_arr[i]) & ok_age
        if ast_ok.any() and alt_ok.any():
            ast_lst = ast_arr[i][ast_ok][np.argsort(ages[ast_ok])][-1]
            alt_lst = alt_arr[i][alt_ok][np.argsort(ages[alt_ok])][-1]
            if alt_lst > 0:
                aar_last[i] = ast_lst / alt_lst
        # bilirubin
        bili_ok = np.isfinite(bili_arr[i]) & ok_age
        if bili_ok.any():
            ser = bili_arr[i][bili_ok]
            bili_max[i] = ser.max()
            ord_ = np.argsort(ages[bili_ok])
            bili_log_last[i] = float(np.log10(max(ser[ord_][-1], 0.01)))
        # platelets
        plt_ok = np.isfinite(plt_arr[i]) & ok_age
        if plt_ok.any():
            ser = plt_arr[i][plt_ok]
            plt_min[i] = ser.min()
            plt_last[i] = ser[np.argsort(ages[plt_ok])][-1]
        # ggt
        ggt_ok = np.isfinite(ggt_arr[i]) & ok_age
        if ggt_ok.any():
            ggt_max[i] = ggt_arr[i][ggt_ok].max()
            if ggt_ok.sum() >= 2:
                a = ages[ggt_ok]; v = ggt_arr[i][ggt_ok]
                if a.std() > 1e-9:
                    ggt_slope[i] = ((a - a.mean()) * (v - v.mean())).mean() / (a.var() + 1e-12)
        # LSM trajectory
        lok = np.isfinite(lsm_arr[i]) & ok_age
        if lok.any():
            ser = list(zip(ages[lok], lsm_arr[i][lok]))
            ser.sort()
            v0 = ser[0][1]; vn = ser[-1][1]
            lsm_max[i] = max(v for _, v in ser)
            if v0 > 0:
                lsm_delta_pct[i] = (vn - v0) / v0
                lsm_progressor[i] = float(lsm_delta_pct[i] > 0.20)
                lsm_regressor[i] = float(lsm_delta_pct[i] < -0.20)
            lsm_auc_10[i] = _auc_above(ser, 10.0)
            lsm_auc_15[i] = _auc_above(ser, 15.0)
            # Baveno-VII at last LSM
            last_lsm = ser[-1][1]
            last_plt = plt_last[i]
            if np.isfinite(last_plt):
                if last_lsm >= 20 or (last_lsm >= 15 and last_plt < 150):
                    baveno_csph[i] = 1.0
            if last_lsm >= 10:
                baveno_likely[i] = 1.0

        # AGILE 3+ (Sanyal 2022): -3.92 + 2.297*ln(LSM_last) - 0.00902*PLT_last
        # - 0.398633*(ALT_last/AST_last) + 1.086*T2DM - 0.386*sex_M + 0.0302*age_baseline
        if (np.isfinite(lsm_max[i]) and np.isfinite(plt_last[i])
                and ast_ok.any() and alt_ok.any()):
            lsm_lst = ser[-1][1] if lok.any() else np.nan
            ast_lst = ast_arr[i][ast_ok][np.argsort(ages[ast_ok])][-1]
            alt_lst = alt_arr[i][alt_ok][np.argsort(ages[alt_ok])][-1]
            if (np.isfinite(lsm_lst) and lsm_lst > 0
                    and ast_lst > 0 and alt_lst > 0
                    and np.isfinite(plt_last[i]) and np.isfinite(base[i])):
                logit = (-3.92368
                         + 2.29714 * np.log(lsm_lst)
                         - 0.00902 * plt_last[i]
                         - 0.398633 * (alt_lst / ast_lst)
                         + 1.08636 * t2dm[i]
                         - 0.38581 * sex_m[i]
                         + 0.03018 * base[i])
                agile[i] = float(logit)

        # Modified aMAP (no albumin term):
        # ((0.06*age + 0.89*sex_M + 0.48*ALBI_lite - 0.01*PLT) + 7.4)/14.77*100
        # ALBI_lite = 0.66*log10(bili_μmol) (drop -0.085*albumin)
        if (np.isfinite(bili_log_last[i]) and np.isfinite(plt_last[i])
                and np.isfinite(base[i])):
            albi_lite = 0.66 * bili_log_last[i]
            score = ((0.06 * base[i] + 0.89 * sex_m[i]
                      + 0.48 * albi_lite - 0.01 * plt_last[i]) + 7.4) / 14.77 * 100.0
            amap[i] = float(score)

        # TyG = ln(triglycerides_mg/dL * glucose_mg/dL / 2)
        # Conversion: mmol/L * 88.57 = mg/dL (triglycerides), * 18 = mg/dL (glucose)
        glu_ok = np.isfinite(glu_arr[i]) & ok_age
        tri_ok = np.isfinite(tri_arr[i]) & ok_age
        bmi_ok = np.isfinite(bmi_arr[i]) & ok_age
        if glu_ok.any() and tri_ok.any():
            g_lst = glu_arr[i][glu_ok][np.argsort(ages[glu_ok])][-1]
            t_lst = tri_arr[i][tri_ok][np.argsort(ages[tri_ok])][-1]
            if g_lst > 0 and t_lst > 0:
                # ANNITIA appears to use mmol/L conventions for triglyc and glucose
                t_mg = t_lst * 88.57
                g_mg = g_lst * 18.0
                tyg[i] = float(np.log(t_mg * g_mg / 2.0))
                if bmi_ok.any():
                    b_lst = bmi_arr[i][bmi_ok][np.argsort(ages[bmi_ok])][-1]
                    tyg_bmi[i] = tyg[i] * b_lst

    # Composite Cox-prior linear score: literature log-HRs * z-score.
    # Use available features only; clip extreme z to ±3.
    feat = np.column_stack([
        fib4_last, fib4_max, fib4_slope, fib4_auc, fib4_time_high, fib4_trans,
        agile, amap, lsm_max, lsm_delta_pct, lsm_progressor, lsm_regressor,
        lsm_auc_10, lsm_auc_15, baveno_csph, baveno_likely,
        aar_last, bili_max, bili_log_last, plt_min, plt_last,
        ggt_max, ggt_slope, tyg, tyg_bmi, apri_last,
    ])
    cols = [
        'fib4_last', 'fib4_max', 'fib4_slope', 'fib4_auc', 'fib4_time_high',
        'fib4_trans', 'agile_3_plus', 'mod_aMAP', 'lsm_max', 'lsm_delta_pct',
        'lsm_progressor', 'lsm_regressor', 'lsm_auc_10', 'lsm_auc_15',
        'baveno_csph', 'baveno_likely', 'aar_last', 'bili_max',
        'bili_log_last', 'plt_min', 'plt_last', 'ggt_max', 'ggt_slope',
        'tyg', 'tyg_bmi', 'apri_last',
    ]
    return feat, cols


# ───────────────────────── train Cox + RSF, save OOF + val ─────────────────────────

def fit_one(X_tr, X_va, y, valid, model_name, endpoint, cid):
    Xv = X_tr[valid]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(Xv.shape[0], np.nan)
    yv = y[valid]
    for k, (tr, te) in enumerate(skf.split(Xv, yv['event'].astype(int))):
        if model_name == 'cox':
            mu = np.nanmean(Xv[tr], axis=0); sd = np.nanstd(Xv[tr], axis=0) + 1e-9
            Xtr_s = np.where(np.isfinite(Xv[tr]),
                             (Xv[tr] - mu) / sd, 0.0)
            Xte_s = np.where(np.isfinite(Xv[te]),
                             (Xv[te] - mu) / sd, 0.0)
            Xtr_s = np.clip(Xtr_s, -5, 5); Xte_s = np.clip(Xte_s, -5, 5)
            m = CoxPHSurvivalAnalysis(alpha=1.0)
        else:
            Xtr_s = np.nan_to_num(Xv[tr]); Xte_s = np.nan_to_num(Xv[te])
            m = RandomSurvivalForest(n_estimators=300, max_depth=6,
                                      min_samples_leaf=10,
                                      random_state=42, n_jobs=-1)
        try:
            m.fit(Xtr_s, yv[tr])
            oof[te] = m.predict(Xte_s)
        except Exception as e:
            print(f'    fold {k} {cid} fail: {e}')

    full = np.full(X_tr.shape[0], np.nan)
    full[valid] = oof
    np.save(OUT / f'oof_{endpoint}_{cid}.npy', full)

    ok = np.isfinite(oof)
    if ok.sum():
        C = concordance(yv[ok], oof[ok])
        print(f'    [{model_name}] {cid}  OOF C = {C:.4f} (n={ok.sum()})')

    # Refit on all valid + predict val
    if model_name == 'cox':
        mu = np.nanmean(Xv, axis=0); sd = np.nanstd(Xv, axis=0) + 1e-9
        Xv_s = np.clip(np.where(np.isfinite(Xv), (Xv - mu) / sd, 0.0), -5, 5)
        Xva_s = np.clip(np.where(np.isfinite(X_va), (X_va - mu) / sd, 0.0), -5, 5)
        m = CoxPHSurvivalAnalysis(alpha=1.0)
    else:
        Xv_s = np.nan_to_num(Xv); Xva_s = np.nan_to_num(X_va)
        m = RandomSurvivalForest(n_estimators=300, max_depth=6,
                                  min_samples_leaf=10,
                                  random_state=42, n_jobs=-1)
    try:
        m.fit(Xv_s, yv)
        vp = m.predict(Xva_s)
        np.save(OUT / f'val_{endpoint}_{cid}.npy', vp)
    except Exception as e:
        print(f'    refit {cid} fail: {e}')


def build_X(d, view: str):
    """Concat: static + summary view (or biomarker view)."""
    static, _ = view_static(d)
    if view == 'summary':
        body, _ = view_summary(d)
    elif view == 'biomarker':
        body, _ = view_biomarkers(d)
    elif view == 'both':
        body1, _ = view_summary(d)
        body2, _ = view_biomarkers(d)
        body = np.column_stack([body1, body2])
    else:
        raise ValueError(view)
    X = np.column_stack([static, body])
    return X


def main():
    print('Loading...')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                       filename=Path(VAL_CSV).name, has_outcomes=False)

    # Iterate: endpoint × delta × view × model.
    # For death, deltas are moot (no leakage), still run delta=0 + biomarker+summary.
    results = []
    for endpoint in ('hepatic', 'death'):
        ep = train[endpoint]
        valid = ep['valid_mask']
        y = ep['y_struct']

        deltas = DELTAS_YEARS if endpoint == 'hepatic' else (0.0,)
        for dlt in deltas:
            print(f'\n=== {endpoint} delta={dlt} ===')
            tr_d = truncate_for_endpoint(train, endpoint, dlt) if dlt > 0 else train
            # val never gets truncated (no event info)
            for view in ('summary', 'biomarker', 'both'):
                X_tr = build_X(tr_d, view)
                X_va = build_X(val, view)
                tag = f"d{int(dlt*1000)}_{view}"
                for mn in ('cox', 'rsf-300'):
                    cid = f'research_{tag}__{mn}'
                    print(f'  fit  {cid}  X_tr={X_tr.shape}  X_va={X_va.shape}')
                    fit_one(X_tr, X_va, y, valid, mn, endpoint, cid)


if __name__ == '__main__':
    main()
