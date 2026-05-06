"""Pairwise ranking loss (Phase D.5).

Two related applications:

    1. PyTorch RankNet-style concordance loss for NN survival models. For a
       batch of patients with risks r_i and times t_i, the loss is
           L = mean_{(i,j) : t_i < t_j AND event_i=1} log(1 + exp(-(r_i - r_j)))
       That is, for every (case_i, anyone_j) pair where i had the event before
       j either (a) had the event or (b) was last seen, push r_i above r_j.
       Censoring gives every uncensored case a comparable set; censored
       observations only act as "at-risk" partners for earlier events.

    2. XGBoost ranking objective on landmark-binary labels. XGBoost natively
       supports `rank:pairwise`, `rank:ndcg`, `rank:map`. We expose a thin
       wrapper that:
         - converts (event, time) → landmark binary at horizon τ
         - uses one query group containing all patients (single ranking
           "session")
         - returns a model with `.predict_risk` that yields probabilities/
           ranking scores.

API:
    pairwise_concordance_loss(risk, time, event)
        — torch tensor input, torch tensor output. Returns 0 if no
        comparable pairs exist in the batch.
    XGBoostPairwiseRanker
        — class with .fit(X, y_struct, horizon=None) / .predict_risk(X).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def pairwise_concordance_loss(risk, time, event):
    """RankNet-style concordance loss in torch.

    Args:
        risk: shape (n,) tensor of predicted risks (higher = worse).
        time: shape (n,) tensor of observed times.
        event: shape (n,) tensor of event indicators (1 = event, 0 = censored).

    Returns: scalar tensor; 0 if no comparable pairs.

    Comparable pair (i, j): event_i = 1 AND time_i < time_j. Loss is then
        BCE-with-logits on (risk_i - risk_j) being positive.

    Vectorised over the full n×n grid; only valid pairs contribute. For a
    batch of size 256 this is fine; if you ever scale to large batches,
    sample comparable pairs instead.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(f'torch missing: {e}')
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1).float()
    n = risk.shape[0]
    # comparable[i,j] = 1 if event_i=1 AND time_i < time_j
    t_i = time.unsqueeze(1)
    t_j = time.unsqueeze(0)
    e_i = event.unsqueeze(1)
    comparable = (e_i.bool() & (t_i < t_j)).float()
    n_pairs = comparable.sum()
    if n_pairs.item() == 0:
        return risk.sum() * 0.0  # zero with grad
    diff = risk.unsqueeze(0) - risk.unsqueeze(1)  # diff[i,j] = r_j - r_i
    # We want r_i > r_j when (i, j) comparable. Loss per pair:
    # log(1 + exp(diff[i,j])) = softplus(r_j - r_i)
    losses = torch.nn.functional.softplus(diff) * comparable
    return losses.sum() / n_pairs


class XGBoostPairwiseRanker:
    """XGBoost rank:pairwise on landmark-binary labels.

    Single query group of all patients. Trained to rank cases (event by τ)
    above controls (event-free past τ). Censored-before-τ patients are
    dropped.

    `.predict_risk(X)` returns the raw margin (higher = more case-like =
    higher risk by τ).
    """

    def __init__(self,
                 horizon: Optional[float] = None,
                 horizon_quantile: float = 0.5,
                 n_estimators: int = 300,
                 learning_rate: float = 0.05,
                 max_depth: int = 4,
                 random_state: int = 42,
                 verbosity: int = 0):
        self.horizon = horizon
        self.horizon_quantile = horizon_quantile
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbosity = verbosity
        self._model = None

    def _resolve_horizon(self, time: np.ndarray, event: np.ndarray) -> float:
        if self.horizon is not None:
            return float(self.horizon)
        ev_times = time[event]
        if ev_times.size == 0:
            ev_times = time
        return float(np.quantile(ev_times, self.horizon_quantile))

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(f'xgboost missing: {e}')
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(y_struct['time'], dtype=np.float64)
        event = np.asarray(y_struct['event'], dtype=bool)
        finite = np.isfinite(time)
        time = time[finite]
        event = event[finite]
        Xk = X[finite]
        tau = self._resolve_horizon(time, event)
        by = (time <= tau) & event
        past = time > tau
        keep = by | past
        if keep.sum() < 4 or by.sum() == 0 or past.sum() == 0:
            raise ValueError(
                'XGBoostPairwiseRanker: degenerate landmark — '
                f'cases={int(by.sum())}, controls={int(past.sum())}')
        y = np.zeros(time.size, dtype=int)
        y[by] = 1
        Xkk = Xk[keep]
        yk = y[keep]
        # Single query group of size keep.sum().
        self._model = xgb.XGBRanker(
            objective='rank:pairwise',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbosity=self.verbosity,
        )
        self._model.fit(Xkk, yk, group=[Xkk.shape[0]])
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('XGBoostPairwiseRanker not fitted')
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)),
                          dtype=np.float64)
