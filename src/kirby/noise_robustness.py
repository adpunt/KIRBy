"""Noise-robustness summary metrics.

These summarise a degradation curve — model performance measured across a grid
of increasing noise levels — into scalars, WITHOUT assuming the curve is a line.
This replaces the old NSI/NDS slope, which fit a straight line to a curve and,
because it was fit to raw performance, partly re-encoded baseline skill.

Everything here operates on the *retention* curve ret(x) = perf(x) / perf(0),
so every config starts at 1.0 and the summaries describe degradation SHAPE,
decoupled from how good the model was to begin with. Only meaningful when the
baseline clears a skill threshold (see ROBUSTNESS_BASELINE_THRESHOLD); below
that the curve is noise on noise.

Ported verbatim from qsar_qm_models so the two repos compute identical numbers.
`x` is the noise axis (sigma for regression additive noise, flip_prob for
classification label noise); `perf` is the matched performance (R2, accuracy,
AUROC, ...). Requires >= 4 points spanning a nonzero range including x=0.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

# Baseline perf(0) must clear this for the retention curve to mean anything.
ROBUSTNESS_BASELINE_THRESHOLD = 0.3


def _retention(perf, base, lower_is_better):
    """perf(x) as a fraction of baseline, oriented so 1.0 = no degradation and
    it decreases as the model gets worse.

    For higher-is-better metrics (R2, accuracy) that is perf/base. For
    lower-is-better error metrics (RMSE, MAE) error GROWS with noise, so the
    correct retention is base/perf -- otherwise the ratio would climb above 1
    and auc_norm/Weibull would be meaningless.
    """
    perf = np.asarray(perf, dtype=float)
    return base / perf if lower_is_better else perf / base


def auc_norm(x, perf, base, lower_is_better=False):
    """Normalised area under the retention curve.

    Primary robustness scalar. HIGHER = more robust (~[0, 1]); decoupled from
    baseline performance (unlike raw AUC or the old NDS slope). No shape
    assumption -- just the mean fraction of baseline perf retained across x.
    """
    x = np.asarray(x, dtype=float)
    ret = _retention(perf, base, lower_is_better)
    srange = x.max() - x.min()
    return float(trapezoid(ret, x) / srange) if srange > 0 else np.nan


def weibull(x, perf, base, lower_is_better=False):
    """Fit retention ~ exp(-(x/tau)^beta); return (tau, beta).

    Supplementary shape descriptor. beta>1 = delayed cliff (holds then
    collapses), beta<1 = early collapse (fast then plateaus). tau is the noise
    scale of decay (the x where retention hits ~0.37).
    """
    x = np.asarray(x, dtype=float)
    ret = np.clip(_retention(perf, base, lower_is_better), 1e-3, None)
    try:
        p, _ = curve_fit(
            lambda s, tau, beta: np.exp(-np.power(np.clip(s, 0, None) / tau, beta)),
            x, ret, p0=[0.5, 1.5], maxfev=10000,
            bounds=([1e-3, 0.1], [np.inf, 10.0]))
        return float(p[0]), float(p[1])
    except Exception:
        return np.nan, np.nan


def robustness_metrics(x, perf, baseline_threshold=ROBUSTNESS_BASELINE_THRESHOLD,
                       lower_is_better=False):
    """Compute all robustness scalars from a degradation curve.

    Returns a dict with auc_norm (primary), weibull_tau, weibull_beta. NaNs if
    the curve is too short (<4 points), lacks an x=0 baseline, or the baseline
    fails the skill gate.

    `lower_is_better` orients the curve for error metrics (RMSE/MAE): the
    retention is inverted (base/perf) so 1.0 still means "no degradation".

    The `baseline_threshold` skill gate is a floor on a higher-is-better
    baseline (R2>0.3, or accuracy/AUROC floors passed by the caller). For
    lower-is-better metrics there is no universal skill floor, so the gate is
    skipped and only base>0 (finite) is required. NOTE: whether/what floor to
    apply for error-metric baselines is an open methodology question.
    """
    x = np.asarray(x, dtype=float)
    perf = np.asarray(perf, dtype=float)
    order = np.argsort(x)
    x, perf = x[order], perf[order]

    out = {'auc_norm': np.nan, 'weibull_tau': np.nan, 'weibull_beta': np.nan}

    at_zero = perf[np.isclose(x, 0.0)]
    if len(at_zero) == 0 or len(x) < 4:
        return out
    base = float(at_zero[0])
    if not np.isfinite(base):
        return out
    # baseline_threshold=None turns the skill gate off for every metric, which
    # is what c10 passes since 2026-09-10: applying a floor on one side of the
    # direction and not the other meant the rmse datasets admitted every
    # representation while the spearman and auc ones dropped any representation
    # under 0.3, so the two were ranked under different admission rules.
    if lower_is_better:
        if base <= 0:
            return out
    elif baseline_threshold is not None and base <= baseline_threshold:
        return out

    out['auc_norm'] = auc_norm(x, perf, base, lower_is_better)
    out['weibull_tau'], out['weibull_beta'] = weibull(x, perf, base, lower_is_better)
    return out
