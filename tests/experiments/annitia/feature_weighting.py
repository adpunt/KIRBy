"""Importance-based feature WEIGHTING (Phase D.2).

Given a vector of per-feature importance scores, scale each column of X by
some monotone transform of its score before fitting the downstream model.
This is distinct from feature SELECTION (top-K prune) — weighting keeps all
features but reweights them, letting the model still see weak features at
reduced magnitude.

Five transforms (all map score ≥ 0 → weight ∈ [0, 1] (or score-rescaled)):
    none        w_j = 1
    linear      w_j = score_j / max(score)
    sqrt        w_j = sqrt(score_j / max(score))     — milder shrinkage
    rank        w_j = rank(score_j) / (D - 1)        — scale-invariant
    softmax_T   w_j ∝ exp(score_j / T) / max         — temperature-controlled

Transforms `none`, `linear`, `sqrt` and `softmax_T` are scale-sensitive
(monotone in raw score). `rank` is scale-invariant — useful when scores from
two different methods are on incomparable scales.

Usage:
    from experiments.annitia.feature_weighting import apply_feature_weights
    from experiments.annitia.importance import compute_survival_importance

    scores = compute_survival_importance(Xtr, ytr, method='treeshap')
    Xtr_w, weights = apply_feature_weights(Xtr, scores, transform='linear')
    Xva_w = Xva * weights[None, :]    # validation gets the SAME weights
    model.fit(Xtr_w, ytr)
    risk = model.predict_risk(Xva_w)

The wrapper class `WeightedSurvivalModel` packages this so the model
interface stays a drop-in replacement for the unweighted model.
"""

from __future__ import annotations

from typing import Optional, Type

import numpy as np


def _transform_scores(scores: np.ndarray, transform: str = 'linear',
                      temperature: float = 1.0) -> np.ndarray:
    """Map raw importance scores → multiplicative column weights ∈ [0, 1].

    Negative or NaN scores are floored at 0; if all scores are non-positive,
    weights default to all-ones (so the model still sees the data).
    """
    s = np.asarray(scores, dtype=np.float64)
    s = np.where(np.isfinite(s), s, 0.0)
    s = np.clip(s, 0.0, None)
    if s.size == 0:
        return np.ones(0, dtype=np.float64)
    if transform == 'none':
        return np.ones_like(s)
    if transform == 'linear':
        m = s.max()
        if m <= 0:
            return np.ones_like(s)
        return s / m
    if transform == 'sqrt':
        m = s.max()
        if m <= 0:
            return np.ones_like(s)
        return np.sqrt(s / m)
    if transform == 'rank':
        if s.size == 1:
            return np.ones_like(s)
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(s.size)
        # Highest-importance feature gets weight 1; lowest gets ~0.
        return ranks / max(s.size - 1, 1)
    if transform == 'softmax_T':
        T = max(temperature, 1e-6)
        m = s.max()
        if m <= 0:
            return np.ones_like(s)
        # Subtract max for numerical stability
        z = (s - m) / T
        e = np.exp(z)
        # Normalise so max weight = 1 (multiplicative scaling, not summing
        # to 1: we don't want softmax to shrink the entire vector).
        return e / e.max()
    raise ValueError(
        f'unknown transform {transform!r}; '
        f'choose from: none, linear, sqrt, rank, softmax_T')


def apply_feature_weights(X: np.ndarray, scores: np.ndarray,
                          transform: str = 'linear',
                          temperature: float = 1.0):
    """Return (X_weighted, weights). `weights` has shape (D,)."""
    X = np.asarray(X, dtype=np.float64)
    weights = _transform_scores(scores, transform=transform,
                                temperature=temperature)
    if weights.size != X.shape[1]:
        raise ValueError(
            f'weights size {weights.size} != n_features {X.shape[1]}')
    return X * weights[None, :], weights


class WeightedSurvivalModel:
    """Drop-in survival model wrapper that applies importance weighting.

    Args:
        base_cls: any class with .fit(X, y_struct) / .predict_risk(X)
            (e.g. SurvivalCoxPH, SurvivalRSF, SurvivalGBSA).
        importance_method: passed to compute_survival_importance.
        transform: weighting transform (one of: none/linear/sqrt/rank/softmax_T).
        temperature: softmax temperature, used only when transform='softmax_T'.
        importance_kwargs: extra kwargs for compute_survival_importance.
        base_kwargs: passed to base_cls(**base_kwargs).
        ensemble_methods: if not None, use the ensemble importance (D.3)
            with these methods instead of the single `importance_method`.
    """

    def __init__(self,
                 base_cls: Type,
                 importance_method: str = 'treeshap',
                 transform: str = 'linear',
                 temperature: float = 1.0,
                 importance_kwargs: Optional[dict] = None,
                 base_kwargs: Optional[dict] = None,
                 ensemble_methods: Optional[tuple] = None):
        self.base_cls = base_cls
        self.importance_method = importance_method
        self.transform = transform
        self.temperature = temperature
        self.importance_kwargs = dict(importance_kwargs or {})
        self.base_kwargs = dict(base_kwargs or {})
        self.ensemble_methods = ensemble_methods
        self._weights: Optional[np.ndarray] = None
        self._inner = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        from experiments.annitia.importance import (
            compute_survival_importance,
            compute_survival_importance_ensemble,
        )
        if self.ensemble_methods is not None:
            scores = compute_survival_importance_ensemble(
                X, y_struct, methods=self.ensemble_methods,
                **self.importance_kwargs)
        else:
            scores = compute_survival_importance(
                X, y_struct, method=self.importance_method,
                **self.importance_kwargs)
        Xw, weights = apply_feature_weights(
            X, scores, transform=self.transform, temperature=self.temperature)
        self._weights = weights
        self._inner = self.base_cls(**self.base_kwargs)
        self._inner.fit(Xw, y_struct)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self._inner is None or self._weights is None:
            raise RuntimeError('WeightedSurvivalModel not fitted')
        Xw = np.asarray(X, dtype=np.float64) * self._weights[None, :]
        return self._inner.predict_risk(Xw)
