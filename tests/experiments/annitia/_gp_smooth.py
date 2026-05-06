"""Per-patient Gaussian-Process smoothed trajectory features.

For each (patient, longitudinal variable):
    - Fit a small GP regression on (age, value) observations.
    - Evaluate posterior mean at a fixed age-grid.
    - Expose the fitted length-scale and noise level as features
      (interpretable: length-scale ≈ how variable the trajectory is over time,
       noise ≈ measurement error).

Patients with <2 observations get the per-variable population mean grid
(the GP would be uninformative). Patients with no observations get zeros.

Why we don't use the entire posterior or sample paths:
    A handful of grid evaluations + 2 hyperparams per (patient, var) is
    enough signal for downstream Cox/RSF, doesn't blow up the feature count,
    and is robust under missing data.

Implementation: sklearn GaussianProcessRegressor (already a hard dep via the
QSAR extras). One GP fit per (patient, var). With n=1250 patients × 12 vars
× <=21 obs per patient, this is O(few seconds) — small enough not to need a
fancy sparse approximation. The whole step is disk-cached anyway.

Output per variable: G grid columns + length-scale + noise + coverage flag
= G+3 features. For 12 longitudinal vars × G=5 grid points → 96 features.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _patient_grid_features(recs: List[Tuple[float, float]],
                           age_grid: np.ndarray,
                           default_lengthscale: float = 5.0,
                           default_noise: float = 0.1) -> Tuple[np.ndarray, float, float, float]:
    """Compute (grid_means, lengthscale, noise, coverage) for one patient.

    Coverage: 1.0 if GP fit succeeded, 0.5 if fallback to per-patient mean,
    0.0 if zero observations.
    """
    G = age_grid.shape[0]
    if not recs:
        return np.zeros(G), default_lengthscale, default_noise, 0.0
    ages = np.array([p[0] for p in recs], dtype=np.float64)
    vals = np.array([p[1] for p in recs], dtype=np.float64)
    finite = np.isfinite(ages) & np.isfinite(vals)
    if finite.sum() < 2:
        # Single observation: posterior mean = the value (constant).
        # No length-scale meaning.
        if finite.any():
            return np.full(G, vals[finite][0], dtype=np.float64), default_lengthscale, default_noise, 0.5
        return np.zeros(G), default_lengthscale, default_noise, 0.0

    a = ages[finite]
    v = vals[finite]
    # Closed-form GP posterior with FIXED kernel hyperparams (no optimizer).
    # We chose this over sklearn's GaussianProcessRegressor with full
    # log-marginal-likelihood optimisation because: 1253 patients × 12 vars
    # × an LBFGS run per fit dominates wall time (>15 min on a Mac) without
    # commensurate downstream signal. The fixed-kernel features still
    # capture trajectory shape; if a later experiment shows more adaptive
    # smoothing matters, swap in the sklearn version (kept in the git
    # history) behind a flag.
    #
    # RBF kernel: k(a, a') = sigma_f^2 * exp(-0.5 * (a-a')^2 / lengthscale^2)
    # Posterior mean at grid x*:
    #     mu(x*) = K(x*, a) * (K(a, a) + sigma_n^2 I)^-1 * v
    try:
        sigma_f2 = max(v.var() + 1e-6, 1e-6)
        ls = max(np.ptp(a) / 2.0, 1.0)
        sigma_n2 = max(0.01 * v.var(), 1e-4)
        # Centre v
        mean_v = float(v.mean())
        v_c = v - mean_v
        K_aa = sigma_f2 * np.exp(-0.5 * (a[:, None] - a[None, :]) ** 2 / ls ** 2)
        K_ga = sigma_f2 * np.exp(
            -0.5 * (age_grid[:, None] - a[None, :]) ** 2 / ls ** 2)
        # Solve (K_aa + sigma_n^2 I) alpha = v_c
        K_reg = K_aa + sigma_n2 * np.eye(len(a))
        try:
            L = np.linalg.cholesky(K_reg)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, v_c))
        except np.linalg.LinAlgError:
            # Fallback: ridge-style direct solve
            alpha = np.linalg.solve(K_reg + 1e-3 * np.eye(len(a)), v_c)
        mu = K_ga @ alpha + mean_v
        return mu.astype(np.float64), float(ls), float(sigma_n2), 1.0
    except Exception:
        return (np.full(G, v.mean(), dtype=np.float64),
                default_lengthscale, default_noise, 0.5)


def fit_gp_grid(longitudinal: List[List[Tuple[float, float]]],
                n_grid: int = 5) -> np.ndarray:
    """Determine the shared age grid used at .transform time."""
    all_ages = [a for recs in longitudinal for (a, _) in recs]
    if not all_ages:
        return np.linspace(0.0, 1.0, n_grid)
    arr = np.array(all_ages, dtype=np.float64)
    return np.quantile(arr, np.linspace(0.05, 0.95, n_grid))


def gp_features_for_var(longitudinal: List[List[Tuple[float, float]]],
                        n_grid: int = 5) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Compute GP-smoothed trajectory features for one variable.

    Returns (X, names, age_grid):
        X has shape (n_patients, n_grid + 3) — grid means, lengthscale, noise, coverage.
        age_grid has shape (n_grid,) and is the shared evaluation grid.
    """
    age_grid = fit_gp_grid(longitudinal, n_grid=n_grid)
    n = len(longitudinal)
    G = age_grid.shape[0]
    grid_arr = np.zeros((n, G), dtype=np.float64)
    ls_arr = np.zeros(n, dtype=np.float64)
    nl_arr = np.zeros(n, dtype=np.float64)
    cov_arr = np.zeros(n, dtype=np.float64)
    for i, recs in enumerate(longitudinal):
        mu, ls, nl, cov = _patient_grid_features(recs, age_grid)
        grid_arr[i] = mu
        ls_arr[i] = ls
        nl_arr[i] = nl
        cov_arr[i] = cov

    X = np.concatenate([grid_arr,
                        ls_arr.reshape(-1, 1),
                        nl_arr.reshape(-1, 1),
                        cov_arr.reshape(-1, 1)], axis=1)
    names = ([f'gp_g{i}' for i in range(G)] +
             ['gp_lengthscale', 'gp_noise', 'gp_coverage'])
    return X, names, age_grid
