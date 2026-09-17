"""Core containers, group-aware OOF machinery, metrics, and the linear adapter.

Ported from `noisy_housing/core.py`. The math is unchanged; the ONE substantive
adaptation is that every out-of-fold prediction is now GROUP-AWARE. The source
used KFold(shuffle=True), which is forbidden in KIRBy: on scaffold-correlated
data a random fold leaks a near-duplicate analogue's label into its own
"held-out" prediction, deflating the residual — and the residual is the exact
quantity every correction/diagnostic method keys on.

A `src/` library must not import from `tests/`, where KIRBy's `cv_split` lives.
So the default splitter here is sklearn's deterministic, group-respecting
GroupKFold (no randomness → honors the no-random-split rule and is fully
self-contained). Callers inside `tests/` (e.g. the n14 experiment) may inject
KIRBy's own `cv_split` via the `splitter=` argument for exact production parity.

No fallback logic anywhere: failures raise with clear messages.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold


# --------------------------------------------------------------------------- #
# Baseline surrogate model
# --------------------------------------------------------------------------- #

DEFAULT_LGB_PARAMS = dict(
    objective="regression",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    verbose=-1,
)


def _lgb_threads() -> int:
    """How many threads this surrogate's LightGBM gets.

    KIRBY_LGBM_THREADS is the repository's existing switch for exactly this --
    `kirby/hybrid.py:78` and `:3741` read it the same way, defaulting to -1.
    This file did not read it and passed no n_jobs at all, so LightGBM took
    every core.

    That matters here more than anywhere else, because `oof_predict` below is
    documented as running on the small partitions inside an outer fold. MEASURED
    2026-09-16 on this machine, one fit of DEFAULT_LGB_PARAMS:

        250 rows x 4 features    one thread 0.41s    every core 138.5s
        2000 rows x 4 features   one thread 2.1s     every core (see below)

    Every core is 346 times slower on the first of those. LightGBM splits its
    work across features, so with four features and a dozen threads the threads
    contend over nothing. On a wide representation the arithmetic reverses:
    5,000 rows by 200 features takes 139.6s on one thread.

    So the count is NOT chosen automatically here. -1 stays the default, which
    is byte-identical to what this file did before, and a caller on narrow
    features sets KIRBY_LGBM_THREADS=1 -- which is what the noise experiments
    and their smokes do.
    """
    import os
    return int(os.environ.get('KIRBY_LGBM_THREADS', '-1'))


def make_model(params: Optional[dict] = None) -> lgb.LGBMRegressor:
    p = dict(DEFAULT_LGB_PARAMS)
    p['n_jobs'] = _lgb_threads()
    if params:
        p.update(params)
    return lgb.LGBMRegressor(**p)


# --------------------------------------------------------------------------- #
# Group-aware split machinery
# --------------------------------------------------------------------------- #

def _n_distinct_groups(groups: np.ndarray) -> int:
    return int(len(np.unique(np.asarray(groups))))


def _safe_n_splits(groups: np.ndarray, requested: int) -> int:
    """Clamp the requested fold count to the number of distinct groups.

    Corrections/diagnostics run OOF *inside* an outer CV fold, so a nested OOF
    can be handed a partition with few groups. Rather than silently degrade
    (KIRBy's cv_split refuses to), clamp to the available groups and raise if
    fewer than 2 remain — an OOF needs at least a train/val separation.
    """
    if groups is None:
        raise ValueError(
            "group-aware OOF requires `groups` (random splitting is banned in "
            "KIRBy). Pass the dataset's scaffold/site grouping."
        )
    n_groups = _n_distinct_groups(groups)
    if n_groups < 2:
        raise ValueError(
            f"Need >=2 distinct groups for a group-aware OOF; got {n_groups}."
        )
    return int(min(requested, n_groups))


def group_kfold_splitter(groups, n_splits, X, y, seed=0, **_ignored):
    """Default splitter: deterministic, group-respecting GroupKFold.

    Signature matches KIRBy's `cv_split(groups, n_splits, X, labels, ...)` so
    either can be passed as `splitter=` to `oof_predict`. GroupKFold is
    deterministic (no seed needed) and never places a group in both train and
    val, so full OOF coverage holds: each row is a validation row exactly once.
    """
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(X, y, groups=np.asarray(groups))


# --------------------------------------------------------------------------- #
# Out-of-fold predictors (group-aware)
# --------------------------------------------------------------------------- #

def oof_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    sample_weight: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
    splitter: Optional[Callable] = None,
) -> np.ndarray:
    """Group-aware out-of-fold predictions of a LightGBM model on (X, y).

    Out-of-fold is mandatory for every correction/diagnostic method: in-fold
    predictions have memorized the very noise we are trying to characterize.

    `splitter` defaults to group_kfold_splitter; inject KIRBy's `cv_split` for
    production parity. It must yield (train_idx, val_idx) and cover every row as
    a validation row exactly once (GroupKFold and cv_split both do).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    k = _safe_n_splits(groups, n_splits)
    split = splitter or group_kfold_splitter

    oof = np.full(len(y), np.nan)
    for tr, te in split(groups, k, X, y, seed=seed):
        m = make_model(params)
        w = sample_weight[tr] if sample_weight is not None else None
        m.fit(X[tr], y[tr], sample_weight=w)
        oof[te] = m.predict(X[te])
    if np.any(np.isnan(oof)):
        raise RuntimeError(
            "OOF prediction left NaNs; the splitter did not cover all rows as "
            "validation exactly once."
        )
    return oof


def oof_log_predict(X, y, groups, **kw) -> np.ndarray:
    """OOF prediction of log(y). All correction happens in log space: housing
    estimate error (and multiplicative assay error) is approximately log-normal.
    In KIRBy this is fed y_pos from `from_linear`, so log(y) is the standardized
    activity — exactly the additive scale on which assay noise lives."""
    y = np.asarray(y, dtype=float).ravel()
    return oof_predict(X, np.log(y), groups, **kw)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def log_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log(y_true) - np.log(y_pred)) ** 2)))


def median_ape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_pred - y_true) / y_true))


def noise_recovery_corr(applied_correction: np.ndarray,
                        injected_eps: np.ndarray) -> float:
    """Correlation between the log-correction a method applied and the noise
    actually injected. The most direct validity check available without clean
    labels. A good method applies correction ~= -eps, so report corr(correction,
    -eps)."""
    if np.std(applied_correction) == 0:
        return 0.0
    return float(np.corrcoef(applied_correction, -injected_eps)[0, 1])


def segment_report(
    y_true: np.ndarray, y_pred: np.ndarray, y_ref: np.ndarray, n_bins: int = 5
) -> dict:
    """Per-quantile log-RMSE. Overall MSE flatters methods that sacrifice the
    tails; per-segment error is where selection-bias failure modes (and, in
    chemistry, thin cliff-rich series) show up."""
    q = np.quantile(y_ref, np.linspace(0, 1, n_bins + 1))
    q[0] -= 1e-9
    out = {}
    for i in range(n_bins):
        m = (y_ref > q[i]) & (y_ref <= q[i + 1])
        if m.sum() == 0:
            raise RuntimeError(f"Empty bin {i}; reduce n_bins")
        out[f"q{i + 1}"] = log_rmse(y_true[m], y_pred[m])
    return out


# --------------------------------------------------------------------------- #
# Linear-scale adapter (for additive-noise / non-positive targets)
# --------------------------------------------------------------------------- #

def from_linear(y_raw: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Adapter for targets where noise is ADDITIVE on the raw scale (or the
    target can be <= 0) — which is every chemistry activity label (logD, dG,
    standardized pActivity all go negative). Every method here works through
    log(y); mapping y_pos = exp((y_raw - mu)/s) makes log(y_pos) =
    (y_raw - mu)/s, so the whole pipeline operates additively on the
    standardized raw scale with no internal changes. Standardization keeps
    exp() in a safe numeric range.

        y_pos, params = from_linear(y_raw)
        # ... run any log-space method on y_pos ...
        y_corrected_raw = to_linear(res_y_pos, params)
    """
    y_raw = np.asarray(y_raw, dtype=float).ravel()
    if np.any(~np.isfinite(y_raw)):
        raise ValueError("y_raw contains non-finite values")
    mu = float(y_raw.mean())
    s_ = float(y_raw.std())
    if s_ <= 0:
        raise ValueError("y_raw is constant; nothing to denoise")
    z = (y_raw - mu) / s_
    if np.max(np.abs(z)) > 30:
        raise ValueError(
            "Standardized target exceeds 30 SDs; exp() would overflow. "
            "Clip or investigate extreme rows first."
        )
    return np.exp(z), (mu, s_)


def to_linear(y_pos: np.ndarray, params: tuple[float, float]) -> np.ndarray:
    """Inverse of from_linear for corrected labels or model predictions."""
    mu, s_ = params
    return mu + s_ * np.log(np.asarray(y_pos, dtype=float))
