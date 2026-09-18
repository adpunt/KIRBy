"""Stage-0 noise diagnostics — characterize the noise before choosing how to
fight it.

Ported from `noisy_housing/experiments.py::stage0_diagnostics`, threaded for
group-aware OOF. Everything except the single OOF call is split-free.

In the chemistry mapping this runs in standardized-pActivity space (feed it the
`from_linear`-transformed labels): the log-residual scale is then directly
comparable to the Kramer et al. 2012 assay noise floor (SD ~= 0.54 pKi).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import stats

from .core import oof_log_predict


def stage0_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    splitter: Optional[Callable] = None,
    params: Optional[dict] = None,
) -> dict:
    """Characterize the noise before choosing how to fight it.

    y must be strictly positive (log-space). In KIRBy pass the `from_linear`
    output so log(y) is the standardized activity.

    `params` reaches the surrogate LightGBM, the same dict `oof_predict` has
    always taken. It was the one link in that chain that dropped it, so a
    caller could not say how many threads its own matrix wants -- and
    `make_model` deliberately does not choose (see `core._lgb_threads`: one
    thread is 346x faster on 4 features and 139.6s slower on 200).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    ly = np.log(y)
    oof = oof_log_predict(X, y, groups, n_splits=n_splits, seed=seed,
                          splitter=splitter, params=params)
    resid = ly - oof

    # 1. Tail heaviness of log-residuals (Gaussian => ~0 excess kurtosis).
    kurt = float(stats.kurtosis(resid, fisher=True))
    frac_beyond_3sd = float(
        np.mean(np.abs(resid - resid.mean()) > 3 * resid.std()))

    # 2. Heteroscedasticity across the activity range (U-shape check).
    q = np.quantile(oof, np.linspace(0, 1, 6))
    q[0] -= 1e-9
    tier_sd, tier_mean = {}, {}
    for i in range(5):
        m = (oof > q[i]) & (oof <= q[i + 1])
        tier_sd[f"q{i + 1}"] = float(resid[m].std()) if m.any() else float("nan")
        tier_mean[f"q{i + 1}"] = float(resid[m].mean()) if m.any() else float("nan")
    u_shape = (tier_sd["q1"] > tier_sd["q3"]) and (tier_sd["q5"] > tier_sd["q3"])

    # 3. Directional bias (systematic mean shift => residual means != 0).
    overall_bias = float(resid.mean())
    bias_t = float(stats.ttest_1samp(resid, 0.0).pvalue)

    # 4. Berkson-vs-classical: NOT identifiable from single noisy labels.
    #    resid = ly - oof is mechanically positively correlated with ly and
    #    negatively with oof for ANY model (the residual contains the label),
    #    so a correlation-based verdict always says "classical" regardless of
    #    the true error structure. Reported for reference only.
    corr_resid_y = float(np.corrcoef(resid, ly)[0, 1])
    corr_resid_oof = float(np.corrcoef(resid, oof)[0, 1])

    return {
        "n": int(len(y)),
        "log_resid_sd": float(resid.std()),
        "excess_kurtosis": kurt,
        "frac_beyond_3sd": frac_beyond_3sd,
        "heavy_tails_flag": bool(kurt > 1.0 or frac_beyond_3sd > 0.05),
        "tier_sd": tier_sd,
        "tier_mean_bias": tier_mean,
        "u_shaped_heteroscedasticity": bool(u_shape),
        "overall_log_bias": overall_bias,
        "bias_pvalue": bias_t,
        "bias_flag": bool(bias_t < 0.01 and abs(overall_bias) > 0.005),
        "corr_resid_with_label": corr_resid_y,
        "corr_resid_with_prediction": corr_resid_oof,
        "error_type_lean": "not_identifiable_from_single_noisy_labels",
    }
