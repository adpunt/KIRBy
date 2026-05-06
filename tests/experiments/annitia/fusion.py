"""Layer 5 fusion algorithms (Phase E).

Five pure-function fusion modes for combining per-branch survival risk
vectors into a single ranking. All operate on rank-transformed inputs so
that branches with incompatible scales (Cox linear predictor vs RSF
cumulative hazard vs XGBoost cox-LL vs landmark-binary probability) can be
combined without bias toward whichever branch happens to have the largest
output magnitude.

    equal       — uniform mean of ranks (legacy `combine_branches.py` default).
    cv_weighted — rank-mean with manual per-branch weights.
    single      — pick the branch with the highest CV C-index; ignore others.
    rrf         — reciprocal-rank fusion (k=60), score_i = Σ_b 1/(k+rank_b(i)).
    stack_meta  — fit Coxnet on the rank-transformed OOF branch risks against
                  (event, time), then apply learned coefficients to the
                  rank-transformed test branch risks. Effectively a
                  scale-invariant linear model on top of branch ranks.

For modes that need labels:
    `stack_meta` consumes a length-N OOF rank matrix (n_train × n_branches)
    paired with the training y_struct, and outputs a length-M test ranking
    obtained by applying the learned linear weights to the test rank matrix
    (m_test × n_branches).

These are intended as building blocks; `combine_branches.py` exposes the
CLI surface that ties OOF inputs + CV JSONs + branch submission CSVs
together.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _ordinal_rank(x: np.ndarray) -> np.ndarray:
    """Plain ordinal ranks. Higher input → higher rank (rank 0 = lowest).

    Ties broken arbitrarily but stably (argsort+arange). For C-index
    purposes ordinal ranks are sufficient — the rare exact-tie case is
    handled by the metric, not the fusion.
    """
    n = x.size
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n)
    return ranks


def rank_matrix(branch_scores: Sequence[np.ndarray]) -> np.ndarray:
    """Stack a list of length-n score vectors into an (n × B) rank matrix.

    Each column is independently ordinal-rank-transformed. NaN rows are
    treated as the smallest possible value (i.e. assigned rank 0); callers
    should drop / mask those rows upstream when meaningful.
    """
    cols = []
    for s in branch_scores:
        s = np.asarray(s, dtype=np.float64)
        # Replace NaN with -inf so they sort to the bottom and get rank 0.
        s_safe = np.where(np.isfinite(s), s, -np.inf)
        cols.append(_ordinal_rank(s_safe))
    return np.stack(cols, axis=1)


# ----------------------------------------------------------------------
# E.0 / legacy — equal & cv_weighted
# ----------------------------------------------------------------------
def fuse_equal(scores: Sequence[np.ndarray]) -> np.ndarray:
    """Uniform mean of within-branch ranks. Output: length-n ranking."""
    R = rank_matrix(scores)
    return R.mean(axis=1)


def fuse_cv_weighted(scores: Sequence[np.ndarray],
                     weights: Sequence[float]) -> np.ndarray:
    """Weighted mean of ranks. Weights are normalised internally."""
    R = rank_matrix(scores)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != R.shape[1]:
        raise ValueError(
            f'weights size {w.size} != n_branches {R.shape[1]}')
    if (w < 0).any():
        raise ValueError('weights must be non-negative')
    s = w.sum()
    if s <= 0:
        raise ValueError('weights sum to zero')
    w = w / s
    return R @ w


# ----------------------------------------------------------------------
# E.1 — single (CV-driven branch pick)
# ----------------------------------------------------------------------
def fuse_single(scores: Sequence[np.ndarray],
                cv_scores: Sequence[float]) -> np.ndarray:
    """Pick the branch with the highest CV score and return its ranking
    verbatim (rank-transformed for cross-mode comparability).

    Args:
        scores: B branch vectors of length n.
        cv_scores: length-B vector of CV C-indices.

    Returns:
        length-n rank vector from the winning branch.
    """
    cv = np.asarray(cv_scores, dtype=np.float64).ravel()
    if cv.size == 0:
        raise ValueError('cv_scores is empty')
    R = rank_matrix(scores)
    if cv.size != R.shape[1]:
        raise ValueError(
            f'cv_scores size {cv.size} != n_branches {R.shape[1]}')
    if not np.any(np.isfinite(cv)):
        raise ValueError('all cv_scores are non-finite')
    finite = np.where(np.isfinite(cv))[0]
    best = finite[np.argmax(cv[finite])]
    return R[:, best]


# ----------------------------------------------------------------------
# E.2 — reciprocal-rank fusion
# ----------------------------------------------------------------------
def fuse_rrf(scores: Sequence[np.ndarray], k: float = 60.0) -> np.ndarray:
    """Reciprocal-rank fusion. score_i = Σ_b 1 / (k + rank_b(i)).

    Note: in the original information-retrieval definition, lower rank =
    better (rank 1 = top). For survival risk we have higher rank = higher
    risk, so we use rank ascending from 0 directly: a patient at rank n-1
    (highest predicted risk) gets the smallest 1/(k+rank) contribution
    while a patient at rank 0 (lowest) gets the largest. We invert the
    convention by using `(n-1) - rank` so HIGH rank (high risk) → LOW
    transformed_rank → LARGE 1/(k+r) contribution → high RRF score.

    Args:
        scores: B branch risk vectors, length n.
        k: smoothing constant (default 60 from the IR literature).

    Returns:
        length-n RRF score (higher = higher fused risk).
    """
    R = rank_matrix(scores)
    n = R.shape[0]
    inverted = (n - 1) - R   # high-risk rows become small "rank"
    return np.sum(1.0 / (k + inverted), axis=1)


# ----------------------------------------------------------------------
# E.3 — stack_meta (Coxnet on OOF branch ranks)
# ----------------------------------------------------------------------
class StackMetaFusion:
    """Fit a Coxnet meta-learner on rank-transformed OOF branch risks.

    Args:
        l1_ratio: elastic-net mix; 0.5 by default (compromise between
            ridge stability and lasso sparsity over the small branch set).
        alpha_min_ratio: regularisation path lower bound.
        max_iter: Coxnet inner solver iterations.

    Notes:
        - Inputs are NOT centred per-branch — Coxnet handles its own
          standardisation internally. We rank-transform first to remove
          scale incompatibilities; standardisation on top of that is fine.
        - If n_branches == 1, falls back to returning the lone column.
        - If Coxnet fails to fit (extremely rare with rank inputs), falls
          back to mean of ranks (equivalent to `fuse_equal`).
    """

    def __init__(self, l1_ratio: float = 0.5,
                 alpha_min_ratio: float = 0.01,
                 max_iter: int = 1000):
        self.l1_ratio = l1_ratio
        self.alpha_min_ratio = alpha_min_ratio
        self.max_iter = max_iter
        self._coefs: Optional[np.ndarray] = None
        self._fallback: bool = False

    def fit(self, oof_scores: Sequence[np.ndarray],
            y_struct: np.ndarray,
            valid_mask: Optional[np.ndarray] = None):
        """Train the Coxnet meta-learner.

        Args:
            oof_scores: B vectors of length N (training OOF risks per branch).
                Patients with NaN in any branch are masked out for fitting.
            y_struct: sksurv structured array of length N.
            valid_mask: optional length-N boolean mask of usable rows.
        """
        R = rank_matrix(oof_scores)
        n, B = R.shape
        if B == 1:
            self._coefs = np.array([1.0])
            return self
        # Drop rows where any OOF was NaN OR time is NaN OR valid_mask is False.
        finite_rows = np.all(np.isfinite(np.stack(
            [np.where(np.isfinite(s), s, np.nan) for s in oof_scores], axis=1)), axis=1)
        time_ok = np.isfinite(y_struct['time'])
        keep = finite_rows & time_ok
        if valid_mask is not None:
            keep &= np.asarray(valid_mask, dtype=bool)
        if keep.sum() < B + 5:
            # Not enough rows — fall back.
            self._fallback = True
            self._coefs = np.ones(B) / B
            return self
        Xfit = R[keep]
        yfit = y_struct[keep]
        # If no events in the kept rows the fit can't proceed.
        if not yfit['event'].any():
            self._fallback = True
            self._coefs = np.ones(B) / B
            return self
        try:
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            model = CoxnetSurvivalAnalysis(
                l1_ratio=self.l1_ratio,
                alpha_min_ratio=self.alpha_min_ratio,
                max_iter=self.max_iter,
                fit_baseline_model=False,
            )
            model.fit(Xfit, yfit)
            # Coefficients along the alpha path: take the smallest-alpha
            # (least-shrinkage) column, matching SurvivalCoxnet convention.
            coefs = np.asarray(model.coef_)
            if coefs.ndim == 2:
                coefs = coefs[:, -1]
            self._coefs = coefs.astype(np.float64)
            # Sanity: if all coefs are zero, fall back to uniform.
            if not np.any(np.abs(self._coefs) > 1e-12):
                self._fallback = True
                self._coefs = np.ones(B) / B
        except Exception:
            self._fallback = True
            self._coefs = np.ones(B) / B
        return self

    def transform(self, branch_scores: Sequence[np.ndarray]) -> np.ndarray:
        """Apply the fitted coefficients to a (length-m) test rank matrix.

        Returns a length-m fused risk vector (higher = higher risk).
        """
        if self._coefs is None:
            raise RuntimeError('StackMetaFusion not fitted')
        R = rank_matrix(branch_scores)
        if R.shape[1] != self._coefs.size:
            raise ValueError(
                f'test rank matrix has {R.shape[1]} branches but model '
                f'was fit with {self._coefs.size}')
        return R @ self._coefs


def fuse_stack_meta(oof_scores: Sequence[np.ndarray],
                    y_struct: np.ndarray,
                    test_scores: Sequence[np.ndarray],
                    valid_mask: Optional[np.ndarray] = None,
                    l1_ratio: float = 0.5) -> np.ndarray:
    """One-shot helper: fit on OOF, predict on test, return fused vector."""
    meta = StackMetaFusion(l1_ratio=l1_ratio)
    meta.fit(oof_scores, y_struct, valid_mask=valid_mask)
    return meta.transform(test_scores)


# ----------------------------------------------------------------------
# Top-level dispatcher
# ----------------------------------------------------------------------
SUPPORTED_MODES = ('equal', 'cv_weighted', 'single', 'rrf', 'stack_meta')


def fuse(mode: str,
         scores: Sequence[np.ndarray],
         *,
         weights: Optional[Sequence[float]] = None,
         cv_scores: Optional[Sequence[float]] = None,
         oof_scores: Optional[Sequence[np.ndarray]] = None,
         y_struct: Optional[np.ndarray] = None,
         valid_mask: Optional[np.ndarray] = None,
         k: float = 60.0,
         l1_ratio: float = 0.5) -> np.ndarray:
    """Single entry-point for the 5 fusion modes.

    Args:
        mode: one of equal | cv_weighted | single | rrf | stack_meta.
        scores: per-branch test/submission risk vectors (B × n).
        weights: required for `cv_weighted`.
        cv_scores: required for `single`.
        oof_scores: required for `stack_meta` (B × N OOF risk vectors).
        y_struct: required for `stack_meta` (training labels).
        valid_mask: optional for `stack_meta`.
        k: rrf smoothing constant.
        l1_ratio: stack_meta elastic-net mix.

    Returns: length-n fused risk vector.
    """
    if mode == 'equal':
        return fuse_equal(scores)
    if mode == 'cv_weighted':
        if weights is None:
            raise ValueError('cv_weighted requires `weights`')
        return fuse_cv_weighted(scores, weights)
    if mode == 'single':
        if cv_scores is None:
            raise ValueError('single requires `cv_scores`')
        return fuse_single(scores, cv_scores)
    if mode == 'rrf':
        return fuse_rrf(scores, k=k)
    if mode == 'stack_meta':
        if oof_scores is None or y_struct is None:
            raise ValueError('stack_meta requires `oof_scores` and `y_struct`')
        return fuse_stack_meta(oof_scores, y_struct, scores,
                               valid_mask=valid_mask, l1_ratio=l1_ratio)
    raise ValueError(
        f'unknown fusion mode {mode!r}; '
        f'supported: {", ".join(SUPPORTED_MODES)}')
