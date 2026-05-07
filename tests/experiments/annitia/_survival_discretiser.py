"""Shared time-discretisation utility for tabular foundation-model survival
heads. Reformulates (X, time, event) into a per-bin classification problem so
classifiers like MITRA, TabPFN-2.5, LightGBM, CatBoost can be turned into
survival predictors via the standard discretised-classifier reduction
(arxiv 2601.22259, "Tabular Foundation Models Can Do Survival Analysis").

For patient i and bin k:
    label[i,k] = 1 if event observed by upper edge of bin k
    label[i,k] = 0 if known event-free past upper edge of bin k
    label[i,k] = NaN if censored before upper edge of bin k (skip)

S_hat(t_k) = prod_{j<=k} (1 - p_hat(event in bin j | survived to bin j))
risk(x) = 1 - S_hat(t_K)  (cumulative incidence by max horizon)
"""
from __future__ import annotations

import numpy as np


class SurvivalDiscretiser:
    """Builds bin edges + per-bin labels.

    Parameters
    ----------
    K : int
        Number of bins.
    strategy : 'equal_event' | 'quantile' | 'linear'
        equal_event:  bin upper edges = event-time quantiles (recommended for
                      rare-event regimes; keeps each bin populated with events).
        quantile:     edges = follow-up quantiles (events + censoring).
        linear:       evenly spaced edges over [0, max_time].
    """

    def __init__(self, K: int = 10, strategy: str = 'equal_event'):
        if K < 2:
            raise ValueError('K must be >= 2')
        if strategy not in ('equal_event', 'quantile', 'linear'):
            raise ValueError(f'unknown strategy: {strategy}')
        self.K = int(K)
        self.strategy = strategy
        self.edges_: np.ndarray | None = None  # length K+1

    def fit(self, time: np.ndarray, event: np.ndarray) -> 'SurvivalDiscretiser':
        time = np.asarray(time, dtype=np.float64)
        event = np.asarray(event, dtype=bool)
        finite = np.isfinite(time)
        if self.strategy == 'equal_event':
            ev = time[event & finite]
            if ev.size < 3:
                ev = time[finite]
            qs = np.linspace(0.0, 1.0, self.K + 1)
            edges = np.unique(np.quantile(ev, qs))
        elif self.strategy == 'quantile':
            qs = np.linspace(0.0, 1.0, self.K + 1)
            edges = np.unique(np.quantile(time[finite], qs))
        else:  # linear
            tmin = float(np.nanmin(time[finite])) if finite.any() else 0.0
            tmax = float(np.nanmax(time[finite])) if finite.any() else 1.0
            edges = np.linspace(tmin, tmax, self.K + 1)
        if edges.size < 3:
            tmin = float(np.nanmin(time[finite])) if finite.any() else 0.0
            tmax = float(np.nanmax(time[finite])) if finite.any() else 1.0
            edges = np.linspace(tmin, tmax, self.K + 1)
        # Force monotonicity + small jitter on duplicates
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-6
        self.edges_ = edges
        return self

    @property
    def n_bins(self) -> int:
        if self.edges_ is None:
            raise RuntimeError('not fit')
        return self.edges_.size - 1

    def labels(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        """Per-bin discrete-time hazard labels of shape (n, K).

        Y[i, k] =
            1   if patient i had an observed event in bin k (edge_k-1, edge_k]
            0   if patient i was at risk throughout bin k AND did not have an
                event in it (i.e. survived past upper edge_k, OR was censored
                strictly past upper edge_k)
            NaN if patient i was no longer at risk at the start of bin k
                (already had event or was censored in an earlier bin)

        With these labels, P(event in bin k | at risk at start of bin k) is
        exactly the per-bin hazard h_k, and
            S(t_K) = prod_k (1 - h_k),  risk = 1 - S(t_K)
        recovers the cumulative incidence by max horizon.
        """
        if self.edges_ is None:
            raise RuntimeError('not fit')
        time = np.asarray(time, dtype=np.float64)
        event = np.asarray(event, dtype=bool)
        n, K = time.size, self.n_bins
        Y = np.full((n, K), np.nan, dtype=np.float64)
        lower = self.edges_[:-1]   # length K
        upper = self.edges_[1:]    # length K
        for k in range(K):
            lo, up = lower[k], upper[k]
            in_bin   = (time > lo) & (time <= up)
            past_bin = time > up
            # Event in this bin -> 1
            Y[event & in_bin, k] = 1.0
            # At risk past this bin -> 0
            Y[past_bin, k] = 0.0
            # Censored strictly within bin -> NaN (already default)
            # Patients who already had event/censor in an earlier bin -> NaN
        return Y

    def survival_to_risk(self, p_per_bin: np.ndarray) -> np.ndarray:
        """Convert per-bin event-probabilities (n, K) into a single risk score
        (cumulative incidence by max horizon — higher = more risk)."""
        p = np.clip(np.asarray(p_per_bin, dtype=np.float64), 0.0, 1.0)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        # P(no event by t_K) = prod_k (1 - p_k); risk = 1 - that.
        surv = np.prod(1.0 - p, axis=1)
        return 1.0 - surv

    def per_bin_train_mask(self, Y: np.ndarray, k: int) -> np.ndarray:
        """Boolean mask of patients usable as training rows for bin k."""
        return ~np.isnan(Y[:, k])
