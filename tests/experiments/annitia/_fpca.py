"""Functional PCA features for ANNITIA longitudinal labs.

Why hand-rolled and not `scikit-fda`:
    - scikit-fda has a heavy dependency tree and is unavailable on ARC.
    - For our regime (n≈1250, ≤21 visits per patient, ≤13 longitudinal vars)
      a rank-r truncated SVD over an age-binned imputed matrix captures the
      same signal with no install hassle.

Algorithm (per longitudinal variable):
    1. Build age bins shared across all patients (default: 20 quantile bins
       over the union of observed ages for that variable).
    2. For each patient, average the observations falling in each bin. Bins
       with no observation get NaN.
    3. Impute missing bins by the per-bin mean across patients (fast, zeroes
       out an unobserved bin's contribution to the patient score).
    4. Subtract the per-bin mean (centred matrix M).
    5. Truncated SVD on M, keep top-K components. Score = M @ V_K.
    6. Score columns are appended; per-patient missingness flags expose how
       much imputation a patient required (0 = fully observed, 1 = nothing).

The fitted basis (bin edges, V matrix, per-bin means) is cacheable and is
what the `_cache.cached_call` layer stores. At predict time the same basis
is reapplied — the validation data does NOT refit the basis.

Output per variable: K score columns + 1 coverage column = K+1 features.
For 12 longitudinal vars (excluding Age) × K=5 components → 72 features.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _build_age_bins(ages: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin the union of observed ages.

    `ages` is a 1-D array of all finite ages across all patients × visits.
    Returns bin edges of length `n_bins + 1`. If too few unique values,
    falls back to linspace.
    """
    finite = ages[np.isfinite(ages)]
    if finite.size < n_bins + 1:
        if finite.size < 2:
            # Degenerate — single bin.
            return np.array([0.0, 1.0])
        return np.linspace(finite.min(), finite.max(), n_bins + 1)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(finite, qs)
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(finite.min(), finite.max(), n_bins + 1)
    return edges.astype(np.float64)


def _bin_average(values: np.ndarray, ages: np.ndarray,
                 bin_edges: np.ndarray) -> np.ndarray:
    """Average `values` per age-bin for one patient.

    `values` and `ages` are equal-length 1-D arrays of one patient's
    longitudinal observations. Returns shape (len(bin_edges)-1,) with NaN
    for empty bins.
    """
    n_bins = len(bin_edges) - 1
    out = np.full(n_bins, np.nan, dtype=np.float64)
    if values.size == 0:
        return out
    finite = np.isfinite(values) & np.isfinite(ages)
    if not finite.any():
        return out
    v = values[finite]
    a = ages[finite]
    # Bin assignment: index i means a in [edges[i], edges[i+1])
    # For the last bin we use side='right' to include the right edge.
    bin_idx = np.searchsorted(bin_edges, a, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    sums = np.zeros(n_bins, dtype=np.float64)
    for j in range(len(v)):
        counts[bin_idx[j]] += 1
        sums[bin_idx[j]] += v[j]
    nonempty = counts > 0
    out[nonempty] = sums[nonempty] / counts[nonempty]
    return out


def _patient_matrix(longitudinal: List[List[Tuple[float, float]]],
                    bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build a (n_patients, n_bins) matrix M of bin-averaged values.

    Returns (M, observed_mask) where observed_mask[i, b] is 1 if patient i
    had at least one observation in bin b, else 0. M has NaN where the mask
    is 0 — caller imputes.
    """
    n = len(longitudinal)
    n_bins = len(bin_edges) - 1
    M = np.full((n, n_bins), np.nan, dtype=np.float64)
    for i, recs in enumerate(longitudinal):
        if not recs:
            continue
        ages = np.array([p[0] for p in recs], dtype=np.float64)
        vals = np.array([p[1] for p in recs], dtype=np.float64)
        M[i] = _bin_average(vals, ages, bin_edges)
    obs = np.isfinite(M).astype(np.float64)
    return M, obs


def fit_fpca(longitudinal: List[List[Tuple[float, float]]],
             n_components: int = 5,
             n_bins: int = 20) -> Dict:
    """Fit FPCA on a single variable's longitudinal records.

    Returns a dict with:
      - 'bin_edges': (n_bins+1,)
      - 'bin_mean':  (n_bins,) — per-bin grand mean used for imputation
      - 'V':         (n_bins, K) — top-K right singular vectors
      - 'singular_values': (K,)
      - 'n_components': K
      - 'training_M':  the imputed-centred matrix used for SVD (cached so
                        _transform on training data is cheap; not strictly
                        needed at predict time, dropped from the cache by
                        callers if memory matters)
    """
    # 1. Build age bins from the union of observed ages
    all_ages = []
    for recs in longitudinal:
        for a, _ in recs:
            all_ages.append(a)
    if not all_ages:
        # Nothing observed — return a degenerate transform that emits zeros.
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        return {
            'bin_edges': edges,
            'bin_mean': np.zeros(n_bins),
            'V': np.zeros((n_bins, max(1, n_components))),
            'singular_values': np.zeros(max(1, n_components)),
            'n_components': max(1, n_components),
            'training_M': None,
            'degenerate': True,
        }

    edges = _build_age_bins(np.array(all_ages, dtype=np.float64), n_bins)
    n_bins_actual = len(edges) - 1

    M, obs = _patient_matrix(longitudinal, edges)

    # 2. Per-bin grand mean for imputation
    with np.errstate(invalid='ignore'):
        bin_mean = np.nanmean(M, axis=0)
    bin_mean = np.where(np.isfinite(bin_mean), bin_mean, 0.0)
    Mi = np.where(np.isfinite(M), M, bin_mean[None, :])

    # 3. Centre and SVD
    centred = Mi - bin_mean[None, :]
    K = max(1, min(n_components, n_bins_actual, centred.shape[0]))
    # Use truncated SVD via numpy (sufficient for n=1250, n_bins≈20)
    try:
        U, s, Vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback: trivial basis
        return {
            'bin_edges': edges,
            'bin_mean': bin_mean,
            'V': np.zeros((n_bins_actual, K)),
            'singular_values': np.zeros(K),
            'n_components': K,
            'training_M': None,
            'degenerate': True,
        }
    V = Vt[:K].T  # (n_bins, K)
    return {
        'bin_edges': edges,
        'bin_mean': bin_mean,
        'V': V,
        'singular_values': s[:K],
        'n_components': K,
        'training_M': Mi,
        'degenerate': False,
    }


def transform_fpca(longitudinal: List[List[Tuple[float, float]]],
                   basis: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a fitted FPCA basis to new (or training) longitudinal records.

    Returns (scores, coverage) where:
      - scores: (n_patients, K) — top-K functional scores
      - coverage: (n_patients,) — fraction of bins with ≥1 observation
                  per patient (0 = fully imputed, 1 = fully observed)
    """
    edges = basis['bin_edges']
    bin_mean = basis['bin_mean']
    V = basis['V']
    K = basis['n_components']
    n = len(longitudinal)
    scores = np.zeros((n, K), dtype=np.float64)
    coverage = np.zeros(n, dtype=np.float64)
    if basis.get('degenerate'):
        return scores, coverage
    M, obs = _patient_matrix(longitudinal, edges)
    Mi = np.where(np.isfinite(M), M, bin_mean[None, :])
    centred = Mi - bin_mean[None, :]
    scores = centred @ V
    coverage = obs.mean(axis=1) if obs.size else np.zeros(n)
    return scores, coverage


def fpca_features_for_var(longitudinal: List[List[Tuple[float, float]]],
                          n_components: int = 5,
                          n_bins: int = 20) -> Tuple[np.ndarray, List[str], Dict]:
    """One-shot fit-and-transform for a single variable's records.

    Returns (X, names, basis) where X is (n_patients, K+1): K score columns
    + 1 coverage column.
    """
    basis = fit_fpca(longitudinal, n_components=n_components, n_bins=n_bins)
    scores, coverage = transform_fpca(longitudinal, basis)
    K = scores.shape[1]
    X = np.concatenate([scores, coverage.reshape(-1, 1)], axis=1)
    names = [f'fpca_{i}' for i in range(K)] + ['fpca_coverage']
    return X, names, basis
