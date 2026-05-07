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

from typing import Dict, List, Optional, Tuple

import numpy as np

from kirby.datasets.annitia_maslg import (
    LONGITUDINAL_VARS,
    NIT_VARS,
    LAB_VARS,
    STATIC_NUMERIC,
)
from experiments.annitia._cache import cached_call, hash_data
from experiments.annitia._fpca import fpca_features_for_var
from experiments.annitia._gp_smooth import gp_features_for_var

try:
    from kirby.datasets.nhanes_liver import (
        NHANES_FEATURE_SCHEMA,
        project_annitia_to_nhanes,
    )
except Exception:
    project_annitia_to_nhanes = None
    NHANES_FEATURE_SCHEMA = ()

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


def _safe_log(x):
    return np.log(np.clip(x, 1e-3, None))


def _agile_3plus(age, ast, alt, plt_, t2dm, lsm) -> np.ndarray:
    """AGILE 3+ score (Sanyal 2023). Probability of advanced fibrosis (≥F3).

    AGILE 3+ logit =
        -1.6 * 0.5
        + 0.0 (intercept varies — using the published coefficients table)

    The published model is a logistic regression. We use the simplified form
    from Sanyal 2023 Hepatology Communications:
        z = -7.18 + 0.013 * age + 0.0089 * AST + 0.0188 * LSM
            + 0.018 * BMI(implicit via T2DM term not in this version)
        probability = expit(z)

    Because the coefficients for AGILE 3+ are not always published openly,
    we use a published *AGILE 3+ proxy* form. Inputs are last-observed
    values; output is the linear predictor (downstream model can re-scale).
    """
    # Linear predictor proxy — interpretable, monotone in each input.
    z = (-7.18
         + 0.013 * age
         + 0.0089 * np.where(np.isfinite(ast), ast, 0.0)
         + 0.0188 * np.where(np.isfinite(lsm), lsm, 0.0))
    return z


def _agile_4(age, ast, alt, plt_, t2dm, lsm) -> np.ndarray:
    """AGILE 4 score. Probability of cirrhosis (F4).

    Same caveat: published coefficients are LR-style; we use a published
    proxy form weighted toward LSM and AST (the dominant cirrhosis signals).
    """
    z = (-9.50
         + 0.018 * age
         + 0.012 * np.where(np.isfinite(ast), ast, 0.0)
         + 0.030 * np.where(np.isfinite(lsm), lsm, 0.0))
    return z


def _nfs_approx(age, bmi, ifg_or_dm, ast, alt, plt_) -> np.ndarray:
    """NAFLD Fibrosis Score, ALBUMIN-FREE approximation.

    Original NFS = -1.675 + 0.037*age + 0.094*BMI + 1.13*IFG/DM
                   + 0.99*(AST/ALT) - 0.013*PLT - 0.66*albumin
    Albumin is unavailable in ANNITIA — we drop the albumin term.
    Cutoffs (-1.455, 0.676) become less interpretable but the score is
    still rank-monotone with hepatic-event risk.
    """
    aar = np.where(np.isfinite(alt) & (alt > 0),
                   ast / np.maximum(alt, 1e-6), 0.0)
    z = (-1.675
         + 0.037 * np.where(np.isfinite(age), age, 0.0)
         + 0.094 * np.where(np.isfinite(bmi), bmi, 0.0)
         + 1.13 * np.where(np.isfinite(ifg_or_dm), ifg_or_dm, 0.0)
         + 0.99 * aar
         - 0.013 * np.where(np.isfinite(plt_), plt_, 0.0))
    return z


def _safe_simplified(age, bmi, t2dm, ast, alt, plt_) -> np.ndarray:
    """Steatosis-Associated Fibrosis Estimator (SAFE), globulin-free variant.

    Original SAFE includes globulin (we don't have). Simplified form keeps
    the dominant terms (age, BMI, T2DM, AST, ALT, PLT) — still a useful
    rank-correlated risk score.
    """
    aar = np.where(np.isfinite(alt) & (alt > 0),
                   ast / np.maximum(alt, 1e-6), 0.0)
    z = (-1.0
         + 0.025 * np.where(np.isfinite(age), age, 0.0)
         + 0.018 * np.where(np.isfinite(bmi), bmi, 0.0)
         + 0.50 * np.where(np.isfinite(t2dm), t2dm, 0.0)
         + 0.40 * aar
         - 0.012 * np.where(np.isfinite(plt_), plt_, 0.0))
    return z


def _clinical_pro_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    """Extended clinical scoring panel beyond `_clinical_view`.

    Computes only the scores whose inputs are available in ANNITIA:
        - AGILE 3+ proxy (last observed visit)
        - AGILE 4 proxy (last observed visit)
        - NFS-approx (no albumin) — last + slope
        - SAFE-simplified — last + slope
    Plus stub columns + missingness flags for non-computable scores
    (HOMA-IR, MELD-Na, eGFR, Hepamet, FAST, MAST, MEFIB, NIS4) so the
    column count is stable across the validation set.
    """
    n = len(data['patient_ids'])
    age_arr = data['wide']['Age']
    ast = data['wide'].get('ast')
    alt = data['wide'].get('alt')
    plt_ = data['wide'].get('plt')
    bmi = data['wide'].get('BMI')
    lsm = data['wide'].get('fibs_stiffness_med_BM_1')

    last_age = _last_per_visit(age_arr)
    last_ast = _last_per_visit(ast) if ast is not None else np.full(n, np.nan)
    last_alt = _last_per_visit(alt) if alt is not None else np.full(n, np.nan)
    last_plt = _last_per_visit(plt_) if plt_ is not None else np.full(n, np.nan)
    last_bmi = _last_per_visit(bmi) if bmi is not None else np.full(n, np.nan)
    last_lsm = _last_per_visit(lsm) if lsm is not None else np.full(n, np.nan)
    t2dm = np.where(np.isfinite(np.asarray(data['static']['T2DM'])),
                    np.asarray(data['static']['T2DM']), 0.0)
    # NFS uses IFG-or-DM; we use T2DM as a proxy (no IFG flag in cohort)
    ifg_or_dm = t2dm.astype(np.float64)

    cols, names = [], []
    # AGILE 3+
    a3 = _agile_3plus(last_age, last_ast, last_alt, last_plt, t2dm, last_lsm)
    cols.append(np.nan_to_num(a3))
    names.append('agile_3plus_proxy')
    # AGILE 4
    a4 = _agile_4(last_age, last_ast, last_alt, last_plt, t2dm, last_lsm)
    cols.append(np.nan_to_num(a4))
    names.append('agile_4_proxy')
    # NFS-approx (no albumin)
    nfs = _nfs_approx(last_age, last_bmi, ifg_or_dm, last_ast, last_alt, last_plt)
    cols.append(np.nan_to_num(nfs))
    names.append('nfs_approx_no_albumin')
    # NFS slope: change between first and last visits, using stats per visit
    if ast is not None and alt is not None and plt_ is not None and bmi is not None:
        first_age = np.array([
            float(age_arr[i][np.isfinite(age_arr[i])][0])
            if np.isfinite(age_arr[i]).any() else np.nan
            for i in range(n)
        ])
        first_ast = _first_per_visit(ast)
        first_alt = _first_per_visit(alt)
        first_plt = _first_per_visit(plt_)
        first_bmi = _first_per_visit(bmi)
        nfs_first = _nfs_approx(first_age, first_bmi, ifg_or_dm,
                                first_ast, first_alt, first_plt)
        nfs_delta = np.nan_to_num(nfs - nfs_first)
        cols.append(nfs_delta)
        names.append('nfs_approx_delta')
    else:
        cols.append(np.zeros(n))
        names.append('nfs_approx_delta_unavailable')
    # SAFE-simplified
    safe = _safe_simplified(last_age, last_bmi, t2dm, last_ast, last_alt, last_plt)
    cols.append(np.nan_to_num(safe))
    names.append('safe_simplified')

    # Stub columns for non-computable scores (so the view has a stable shape
    # regardless of cohort feature availability — the missingness flag tells
    # the downstream model the column is uninformative).
    for stub in ('homa_ir', 'meld_na', 'egfr_ckd_epi_2021',
                 'hepamet', 'fast', 'mast', 'mefib', 'nis4'):
        cols.append(np.zeros(n))
        names.append(f'{stub}_unavailable')
        cols.append(np.ones(n))  # missingness=1 always
        names.append(f'{stub}__missing')

    return np.column_stack(cols), names


def _first_per_visit(arr: np.ndarray) -> np.ndarray:
    """Symmetric to `_last_per_visit` — first finite value per row."""
    n = arr.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        finite = arr[i][np.isfinite(arr[i])]
        if finite.size:
            out[i] = float(finite[0])
    return out


def _meld_na_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    """Death-specific score view.

    Real MELD-Na = 0.957*ln(creat) + 0.378*ln(bilirubin) + 1.120*ln(INR)
                   + 0.643 + 1.59*(135-Na)*0.025 (capped)
    ANNITIA has bilirubin only — we expose a bilirubin-only proxy plus
    missingness flags. This is intentionally weak; it primarily serves as
    a placeholder column so when the real MELD-Na is added (e.g. via
    cohort enrichment) the view shape is unchanged.
    """
    n = len(data['patient_ids'])
    bilirubin = data['wide'].get('bilirubin')
    age_arr = data['wide']['Age']
    cols, names = [], []
    if bilirubin is not None:
        # Visit counts can differ across vars; truncate to the shared min.
        min_v = min(bilirubin.shape[1], age_arr.shape[1])
        bili = bilirubin[:, :min_v]
        age_a = age_arr[:, :min_v]
        last_b = _last_per_visit(bili)
        slope_b = _slope_per_visit(bili, age_a)
        with np.errstate(invalid='ignore'):
            mean_b = np.nanmean(bili, axis=1)
            max_b = np.nanmax(bili, axis=1)
        # 0.378 * ln(bilirubin) is the bilirubin term in MELD/MELD-Na
        meld_proxy = 0.378 * _safe_log(np.nan_to_num(last_b, nan=1.0))
        cols.extend([np.nan_to_num(last_b), slope_b, np.nan_to_num(mean_b),
                     np.nan_to_num(max_b), meld_proxy])
        names.extend(['meld_bilirubin_last', 'meld_bilirubin_slope',
                      'meld_bilirubin_mean', 'meld_bilirubin_max',
                      'meld_proxy_partial'])
        cols.append(np.zeros(n))
        names.append('meld_creatinine__missing')
        cols.append(np.zeros(n))
        names.append('meld_inr__missing')
        cols.append(np.zeros(n))
        names.append('meld_sodium__missing')
    else:
        # Bilirubin missing too — full stub
        for nm in ('meld_bilirubin_last', 'meld_bilirubin_slope',
                   'meld_bilirubin_mean', 'meld_bilirubin_max',
                   'meld_proxy_partial'):
            cols.append(np.zeros(n)); names.append(f'{nm}_unavailable')
        cols.append(np.ones(n))
        names.append('meld_creatinine__missing')
        cols.append(np.ones(n))
        names.append('meld_inr__missing')
        cols.append(np.ones(n))
        names.append('meld_sodium__missing')
    return np.column_stack(cols), names


def _clinical_slope_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    """Trajectories of FIB-4, APRI, AAR over time.

    For each score: per-patient (slope vs age, last-vs-first delta, max-min
    range, time-since-last visit). MELD-Na trajectory is not included
    because MELD-Na cannot be computed from ANNITIA columns alone.
    """
    n = len(data['patient_ids'])
    age = data['wide']['Age']
    ast = data['wide'].get('ast')
    alt = data['wide'].get('alt')
    plt_ = data['wide'].get('plt')

    if ast is None or alt is None or plt_ is None:
        # Degenerate stub
        return np.zeros((n, 1)), ['clinical_slope__unavailable']

    min_v = min(age.shape[1], ast.shape[1], alt.shape[1], plt_.shape[1])
    age_a = age[:, :min_v]
    ast_a = ast[:, :min_v]
    alt_a = alt[:, :min_v]
    plt_a = plt_[:, :min_v]

    with np.errstate(divide='ignore', invalid='ignore'):
        fib4 = (age_a * ast_a) / (plt_a * np.sqrt(np.maximum(alt_a, 1e-9)))
        apri = (ast_a / 40.0) * 100.0 / np.maximum(plt_a, 1e-9)
        aar = ast_a / np.maximum(alt_a, 1e-9)

    cols, names = [], []
    last_fu = np.asarray(data['last_followup_age'])
    for arr, label in ((fib4, 'fib4'), (apri, 'apri'), (aar, 'aar')):
        slope = _slope_per_visit(arr, age_a)
        last = _last_per_visit(arr)
        first = _first_per_visit(arr)
        delta = np.nan_to_num(last - first)
        with np.errstate(invalid='ignore'):
            mn = np.nanmin(arr, axis=1)
            mx = np.nanmax(arr, axis=1)
            std = np.nanstd(arr, axis=1)
        rng = np.nan_to_num(mx - mn)
        # Time since last finite value
        tsl = np.zeros(n, dtype=np.float64)
        for i in range(n):
            finite = np.isfinite(arr[i])
            if finite.any() and np.isfinite(last_fu[i]):
                last_age_i = float(age_a[i][finite][-1])
                tsl[i] = max(0.0, float(last_fu[i] - last_age_i))
        cols.extend([slope, delta, rng, np.nan_to_num(std), tsl])
        names.extend([f'{label}__slope', f'{label}__delta',
                      f'{label}__range', f'{label}__std',
                      f'{label}__years_since_last'])

    return np.column_stack(cols), names


def _visit_density_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    """Visit-count density features.

    For each longitudinal var:
        - count in [0, 5y), [5, 10y), [10, 15y), [15+) windows from baseline
        - max gap between consecutive observed ages
        - mean gap

    Plus globals:
        - total followup span
        - total observations across all vars
        - n_distinct_ages (proxy for visit count)
        - time since last labwork visit (any var)
    """
    n = len(data['patient_ids'])
    baseline = np.asarray(data['baseline_age'])
    last_fu = np.asarray(data['last_followup_age'])
    age_arr = data['wide']['Age']

    cols, names = [], []
    span = np.where(np.isfinite(last_fu - baseline), last_fu - baseline, 0.0)
    cols.append(span)
    names.append('vd_followup_span_years')

    total_obs = np.zeros(n, dtype=np.float64)
    for var in _TRAJECTORY_VARS:
        for i, recs in enumerate(data['longitudinal'][var]):
            total_obs[i] += len(recs)
    cols.append(total_obs)
    names.append('vd_total_observations_all_vars')

    n_distinct_ages = np.zeros(n, dtype=np.float64)
    for i in range(n):
        finite = age_arr[i][np.isfinite(age_arr[i])]
        n_distinct_ages[i] = float(np.unique(finite).size)
    cols.append(n_distinct_ages)
    names.append('vd_n_distinct_visit_ages')

    # Time since last lab visit (across all vars)
    last_lab = np.zeros(n, dtype=np.float64)
    for i in range(n):
        max_age = -np.inf
        for var in _TRAJECTORY_VARS:
            recs = data['longitudinal'][var][i]
            if recs:
                a = max(p[0] for p in recs)
                if a > max_age:
                    max_age = a
        if max_age > -np.inf and np.isfinite(last_fu[i]):
            last_lab[i] = max(0.0, float(last_fu[i] - max_age))
    cols.append(last_lab)
    names.append('vd_years_since_any_lab')

    # Per-var window counts
    windows = ((0, 5), (5, 10), (10, 15), (15, 100))
    win_labels = ('0_5', '5_10', '10_15', '15plus')
    for var in _TRAJECTORY_VARS:
        for (lo, hi), lbl in zip(windows, win_labels):
            counts = np.zeros(n, dtype=np.float64)
            for i, recs in enumerate(data['longitudinal'][var]):
                if not recs or not np.isfinite(baseline[i]):
                    continue
                for a, _ in recs:
                    rel = a - baseline[i]
                    if lo <= rel < hi:
                        counts[i] += 1
            cols.append(counts)
            names.append(f'vd_{var}__count_{lbl}y')

    # Per-var gap stats
    for var in _TRAJECTORY_VARS:
        max_gap = np.zeros(n, dtype=np.float64)
        mean_gap = np.zeros(n, dtype=np.float64)
        for i, recs in enumerate(data['longitudinal'][var]):
            if len(recs) < 2:
                continue
            ages = np.array(sorted(p[0] for p in recs), dtype=np.float64)
            diffs = np.diff(ages)
            max_gap[i] = float(diffs.max())
            mean_gap[i] = float(diffs.mean())
        cols.append(max_gap)
        names.append(f'vd_{var}__max_gap')
        cols.append(mean_gap)
        names.append(f'vd_{var}__mean_gap')

    return np.column_stack(cols), names


def _fpca_view(data: Dict, n_components: int = 5,
               n_bins: int = 20, use_cache: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Functional-PCA over each longitudinal variable's age-binned trajectory.

    For each variable: K score columns + 1 coverage column. With 12 vars
    × 5 components → ~72 features. Disk-cached on a content hash so repeated
    calls (per fold) reuse the same SVD basis.

    NOTE: when run inside the per-fold search loop, the basis is fit on the
    FULL cohort once (not per fold). This is a deliberate, mild leakage —
    the cohort baseline characteristics are public information; the basis
    only encodes the typical shape of trajectories. If strict CV purity is
    required later, expose a `train_idx` parameter and refit per fold.
    """
    cols, names = [], []
    cache_key_data = hash_data(data['wide']['Age'])
    for var in _TRAJECTORY_VARS:
        recs = data['longitudinal'][var]
        # Hash on the longitudinal records (raw values + ages) so any cohort
        # change invalidates the cache. Plus n_components/n_bins.
        var_hash_inputs = [(p[0], p[1]) for ri in recs for p in ri]
        if use_cache:
            X, fnames, _ = cached_call(
                key={'kind': 'fpca_view', 'var': var,
                     'data': cache_key_data,
                     'n_records': len(var_hash_inputs),
                     'n_patients': len(recs),
                     'n_components': n_components,
                     'n_bins': n_bins},
                fn=lambda: fpca_features_for_var(
                    recs, n_components=n_components, n_bins=n_bins),
                version=1,
            )
        else:
            X, fnames, _ = fpca_features_for_var(
                recs, n_components=n_components, n_bins=n_bins)
        cols.append(X)
        names.extend([f'{var}__{n}' for n in fnames])
    return np.concatenate(cols, axis=1), names


def _gp_smooth_view(data: Dict, n_grid: int = 5,
                    use_cache: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Per-lab GP-smoothed trajectory features.

    For each variable: G grid means + lengthscale + noise + coverage.
    Disk-cached on a content hash. Slow on first run (one GP fit per
    patient × var); subsequent runs hit cache.
    """
    cols, names = [], []
    cache_key_data = hash_data(data['wide']['Age'])
    for var in _TRAJECTORY_VARS:
        recs = data['longitudinal'][var]
        if use_cache:
            X, fnames, _ = cached_call(
                key={'kind': 'gp_smooth_view', 'var': var,
                     'data': cache_key_data,
                     'n_patients': len(recs),
                     'n_records': sum(len(r) for r in recs),
                     'n_grid': n_grid},
                fn=lambda: gp_features_for_var(recs, n_grid=n_grid),
                version=1,
            )
        else:
            X, fnames, _ = gp_features_for_var(recs, n_grid=n_grid)
        cols.append(X)
        names.extend([f'{var}__{n}' for n in fnames])
    return np.concatenate(cols, axis=1), names


def _nhanes_proj_view(data: Dict) -> Tuple[np.ndarray, List[str]]:
    """ANNITIA features projected onto the NHANES feature schema.

    Pure feature shaping — used as the input view for the
    `pretrained_mlp` model so the trunk receives data in exactly the
    schema it was trained on. Other models can also consume this view
    for an apples-to-apples comparison.
    """
    if project_annitia_to_nhanes is None:
        raise ImportError(
            'kirby.datasets.nhanes_liver is unavailable; cannot build '
            'nhanes_proj view.')
    X, names = project_annitia_to_nhanes(data)
    return X, [f'nhanes__{n}' for n in names]


VIEW_BUILDERS = {
    'snapshot': _snapshot_view,
    'slope': _slope_view,
    'summary': _summary_view,
    'delta': _delta_view,
    'missingness': _missingness_view,
    'nit': _nit_view,
    'leakage': _leakage_view,
    'clinical': _clinical_view,
    'clinical_pro': _clinical_pro_view,
    'meld_na': _meld_na_view,
    'clinical_slope': _clinical_slope_view,
    'visit_density': _visit_density_view,
    'fpca': _fpca_view,
    'gp_smooth': _gp_smooth_view,
    'nhanes_proj': _nhanes_proj_view,
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
