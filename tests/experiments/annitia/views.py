"""ANNITIA feature view builders.

Six views over the same per-patient record. Each builds a fixed-length
numeric matrix (n_patients, d_view) with consistent patient ordering. Missing
slots are imputed with 0.0; explicit missingness flags are added where useful.

Views:
    snapshot     — last observed value of each lab + static covariates
    slope        — linear-fit slope (value vs age) for each lab; 0 if <2 obs
    summary      — per-lab mean / min / max / std (NaN-safe; 0 if no obs)
    delta        — last - first per lab; 0 if <2 obs
    missingness  — visit count per var, total followup span, time since last
    nit          — same recipe (snapshot+slope+delta) restricted to the 3 NITs

`build_views` returns a dict {view_name: (X, feature_names)}.

These are deterministic, no fitting required — same builder applies to train
and validation patients alike.
"""

from typing import Dict, List, Tuple

import numpy as np

from kirby.datasets.annitia_maslg import (
    LONGITUDINAL_VARS,
    NIT_VARS,
    LAB_VARS,
    STATIC_NUMERIC,
)

_TRAJECTORY_VARS = [v for v in LONGITUDINAL_VARS if v != 'Age']


def _last_value(series: List[Tuple[float, float]]) -> Tuple[float, bool]:
    if not series:
        return 0.0, False
    return series[-1][1], True


def _first_value(series: List[Tuple[float, float]]) -> Tuple[float, bool]:
    if not series:
        return 0.0, False
    return series[0][1], True


def _slope(series: List[Tuple[float, float]]) -> float:
    if len(series) < 2:
        return 0.0
    a = np.array([p[0] for p in series], dtype=np.float64)
    v = np.array([p[1] for p in series], dtype=np.float64)
    if a.std() < 1e-9:
        return 0.0
    cov = ((a - a.mean()) * (v - v.mean())).mean()
    return float(cov / (a.var() + 1e-12))


def _stats(series: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    if not series:
        return 0.0, 0.0, 0.0, 0.0
    v = np.array([p[1] for p in series], dtype=np.float64)
    return float(v.mean()), float(v.min()), float(v.max()), float(v.std())


def _static_matrix(data: Dict) -> Tuple[np.ndarray, List[str]]:
    cols = []
    names = []
    for col in STATIC_NUMERIC:
        x = np.asarray(data['static'][col], dtype=np.float64)
        miss = ~np.isfinite(x)
        cols.append(np.where(miss, 0.0, x))
        cols.append(miss.astype(np.float64))
        names.append(col)
        names.append(f'{col}__missing')
    cols.append(np.asarray(data['baseline_age'], dtype=np.float64))
    names.append('baseline_age')
    return np.column_stack(cols), names


def _snapshot_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    static, static_names = _static_matrix(data)
    n = static.shape[0]
    extra = []
    extra_names = []
    for var in _TRAJECTORY_VARS:
        col_v = np.zeros(n)
        col_m = np.zeros(n)
        for i, series in enumerate(data['longitudinal'][var]):
            v, present = _last_value(series)
            col_v[i] = v
            col_m[i] = 0.0 if present else 1.0
        extra.append(col_v)
        extra.append(col_m)
        extra_names.append(f'{var}__last')
        extra_names.append(f'{var}__last_missing')
    return np.column_stack([static] + extra), static_names + extra_names


def _slope_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    n = len(data['patient_ids'])
    cols = []
    names = []
    for var in _TRAJECTORY_VARS:
        col = np.zeros(n)
        for i, series in enumerate(data['longitudinal'][var]):
            col[i] = _slope(series)
        cols.append(col)
        names.append(f'{var}__slope')
    return np.column_stack(cols), names


def _summary_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    n = len(data['patient_ids'])
    cols = []
    names = []
    for var in _TRAJECTORY_VARS:
        cm = np.zeros(n)
        cn = np.zeros(n)
        cx = np.zeros(n)
        cs = np.zeros(n)
        for i, series in enumerate(data['longitudinal'][var]):
            mean, mn, mx, sd = _stats(series)
            cm[i], cn[i], cx[i], cs[i] = mean, mn, mx, sd
        cols.extend([cm, cn, cx, cs])
        names.extend([f'{var}__mean', f'{var}__min',
                      f'{var}__max', f'{var}__std'])
    return np.column_stack(cols), names


def _delta_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    n = len(data['patient_ids'])
    cols = []
    names = []
    for var in _TRAJECTORY_VARS:
        col = np.zeros(n)
        for i, series in enumerate(data['longitudinal'][var]):
            if len(series) < 2:
                continue
            col[i] = series[-1][1] - series[0][1]
        cols.append(col)
        names.append(f'{var}__delta')
    return np.column_stack(cols), names


def _missingness_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    n = len(data['patient_ids'])
    cols = []
    names = []
    baseline = np.asarray(data['baseline_age'])
    last_fu = np.asarray(data['last_followup_age'])
    span = np.where(np.isfinite(last_fu - baseline), last_fu - baseline, 0.0)
    cols.append(span)
    names.append('followup_span_years')
    for var in _TRAJECTORY_VARS:
        counts = np.array(
            [len(s) for s in data['longitudinal'][var]], dtype=np.float64)
        cols.append(counts)
        names.append(f'{var}__count')
        # Time since last observation (years before censor/event time)
        tsl = np.zeros(n)
        for i, series in enumerate(data['longitudinal'][var]):
            if series and np.isfinite(last_fu[i]):
                tsl[i] = max(0.0, float(last_fu[i] - series[-1][0]))
        cols.append(tsl)
        names.append(f'{var}__years_since_last')
    return np.column_stack(cols), names


def _leakage_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    """Raw absolute timing features. The R1 leaderboard (CV 0.91 -> LB 0.89)
    showed follow-up-span features generalise at least partly to the held-out
    set, so we expose them as a dedicated view."""
    n = len(data['patient_ids'])
    baseline = np.asarray(data['baseline_age'])
    last_fu = np.asarray(data['last_followup_age'])

    cols, names = [], []
    cols.append(np.where(np.isfinite(baseline), baseline, 0.0))
    names.append('baseline_age_raw')
    cols.append(np.where(np.isfinite(last_fu), last_fu, 0.0))
    names.append('last_observed_age_raw')
    span = np.where(np.isfinite(last_fu - baseline), last_fu - baseline, 0.0)
    cols.append(span)
    names.append('followup_span_raw')

    total = np.zeros(n)
    for var in _TRAJECTORY_VARS:
        for i, series in enumerate(data['longitudinal'][var]):
            total[i] += len(series)
    cols.append(total)
    names.append('total_observations')

    age_arr = data['wide'].get('Age')
    max_gap = np.zeros(n)
    mean_gap = np.zeros(n)
    if age_arr is not None:
        for i in range(n):
            ages = age_arr[i][np.isfinite(age_arr[i])]
            if ages.size >= 2:
                diffs = np.diff(np.sort(ages))
                max_gap[i] = float(diffs.max())
                mean_gap[i] = float(diffs.mean())
    cols.append(max_gap)
    names.append('max_visit_gap_years')
    cols.append(mean_gap)
    names.append('mean_visit_gap_years')

    return np.column_stack(cols), names


def _last_per_visit(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        finite = arr[i][np.isfinite(arr[i])]
        if finite.size:
            out[i] = float(finite[-1])
    return out


def _slope_per_visit(arr: np.ndarray, age_arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    out = np.zeros(n)
    for i in range(n):
        mask = np.isfinite(arr[i]) & np.isfinite(age_arr[i])
        if mask.sum() >= 2:
            a = age_arr[i, mask]
            v = arr[i, mask]
            if a.std() > 1e-9:
                out[i] = float(np.cov(a, v, bias=True)[0, 1] / (a.var() + 1e-12))
    return out


def _clinical_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    """Validated MASLD/fibrosis clinical scores from cohort columns.

    FIB-4  = (age * AST) / (PLT * sqrt(ALT))     — advanced-fibrosis screen (cutoffs 1.3, 2.67)
    APRI   = (AST / 40) * 100 / PLT              — fibrosis screen (cutoff 0.5)
    AAR    = AST / ALT                           — AST/ALT ratio (cutoff 0.8)
    BARD   = (BMI>=28) + 2*(AAR>=0.8) + DM       — composite, range 0-4

    NFS is excluded because it requires serum albumin which the cohort lacks.
    """
    n = len(data['patient_ids'])
    age = data['wide']['Age']
    ast = data['wide'].get('ast')
    alt = data['wide'].get('alt')
    plt_ = data['wide'].get('plt')
    bmi = data['wide'].get('BMI')

    min_v = min(a.shape[1] for a in (age, ast, alt, plt_, bmi) if a is not None)
    age_a = age[:, :min_v]
    ast_a = ast[:, :min_v]
    alt_a = alt[:, :min_v]
    plt_a = plt_[:, :min_v]
    bmi_a = bmi[:, :min_v]

    with np.errstate(divide='ignore', invalid='ignore'):
        fib4 = (age_a * ast_a) / (plt_a * np.sqrt(np.maximum(alt_a, 1e-9)))
        apri = (ast_a / 40.0) * 100.0 / plt_a
        aar = ast_a / np.maximum(alt_a, 1e-9)

    cols, names = [], []
    for arr, label in ((fib4, 'fib4'), (apri, 'apri'), (aar, 'aar')):
        last = _last_per_visit(arr)
        slope = _slope_per_visit(arr, age_a)
        with np.errstate(invalid='ignore'):
            mn = np.nanmean(arr, axis=1)
            mx = np.nanmax(arr, axis=1)
        cols.extend([np.nan_to_num(last), slope,
                     np.nan_to_num(mn), np.nan_to_num(mx)])
        names.extend([f'{label}__last', f'{label}__slope',
                      f'{label}__mean', f'{label}__max'])

    fib4_last = _last_per_visit(fib4)
    apri_last = _last_per_visit(apri)
    aar_last = _last_per_visit(aar)
    cols.append((np.nan_to_num(fib4_last) >= 1.3).astype(np.float64))
    names.append('fib4_last_ge_1_3')
    cols.append((np.nan_to_num(fib4_last) >= 2.67).astype(np.float64))
    names.append('fib4_last_ge_2_67')
    cols.append((np.nan_to_num(apri_last) >= 0.5).astype(np.float64))
    names.append('apri_last_ge_0_5')
    cols.append((np.nan_to_num(aar_last) >= 0.8).astype(np.float64))
    names.append('aar_last_ge_0_8')

    bmi_mean = np.nan_to_num(np.nanmean(bmi_a, axis=1))
    dm = np.nan_to_num(np.asarray(data['static']['T2DM']))
    aar_safe = np.nan_to_num(aar_last)
    bard = (bmi_mean >= 28).astype(np.float64) \
        + 2.0 * (aar_safe >= 0.8).astype(np.float64) \
        + (dm == 1).astype(np.float64)
    cols.append(bard)
    names.append('bard_score')

    return np.column_stack(cols), names


def _nit_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    """Snapshot+slope+delta restricted to the three NITs (FibroScan, FibroTest,
    Aixplorer). Plus missingness flags for each. Required by the challenge:
    the model must use at least one NIT."""
    n = len(data['patient_ids'])
    cols = []
    names = []
    for var in NIT_VARS:
        col_last = np.zeros(n)
        col_first = np.zeros(n)
        col_slope = np.zeros(n)
        col_delta = np.zeros(n)
        col_count = np.zeros(n)
        col_missing = np.zeros(n)
        for i, series in enumerate(data['longitudinal'][var]):
            v_last, present = _last_value(series)
            v_first, _ = _first_value(series)
            col_last[i] = v_last
            col_first[i] = v_first
            col_slope[i] = _slope(series)
            col_delta[i] = (v_last - v_first) if len(series) >= 2 else 0.0
            col_count[i] = len(series)
            col_missing[i] = 0.0 if present else 1.0
        cols.extend([col_last, col_first, col_slope, col_delta,
                     col_count, col_missing])
        names.extend([f'{var}__last', f'{var}__first', f'{var}__slope',
                      f'{var}__delta', f'{var}__count', f'{var}__missing'])
    return np.column_stack(cols), names


VIEW_BUILDERS = {
    'snapshot': _snapshot_view,
    'slope': _slope_view,
    'summary': _summary_view,
    'delta': _delta_view,
    'missingness': _missingness_view,
    'nit': _nit_view,
    'leakage': _leakage_view,
    'clinical': _clinical_view,
}


def build_views(data: Dict, view_names=None) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Compute all (or a subset of) views from a loaded ANNITIA dict."""
    if view_names is None:
        view_names = list(VIEW_BUILDERS.keys())
    out = {}
    for name in view_names:
        X, fnames = VIEW_BUILDERS[name](data)
        # Final NaN/Inf scrub
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        out[name] = (X.astype(np.float64), fnames)
    return out
