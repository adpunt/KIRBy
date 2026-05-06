"""Snapshot ensembles for NN survival models (Phase D.7).

Classic snapshot ensembles (Huang et al. 2017) train one network with a
cyclical learning-rate schedule, saving the weights at the end of each
cycle, then average predictions at test time. We don't have direct hooks
into the pycox / DeepHit training loops, so we approximate with the
"poor man's snapshot": train N independent models with different seeds
(and optionally different epoch budgets), and average the survival curves
at test time.

This is essentially the same as bootstrap bagging without resampling — the
diversity comes from the random initialisation of the NN weights and the
shuffling of mini-batches under different seeds. For NN models on small
data (n=1253), seed-diversity is non-trivial: each fit lands in a slightly
different local optimum, and averaging the risk vectors halves the
variance attributable to that.

When to use vs `BaggedSurvivalModel`:
    - Bagging: tabular boosters with low fit variance per seed.
    - Snapshot: NN models with high seed-to-seed prediction variance.
    - Both: redundant and not recommended (compute waste).

API: `SnapshotEnsembleSurvival(base_cls, n_snapshots=5, base_kwargs={...})`
— drop-in for any class with .fit(X, y_struct) / .predict_risk(X) AND a
`random_state` constructor argument.
"""

from __future__ import annotations

from typing import List, Optional, Type

import numpy as np


class SnapshotEnsembleSurvival:
    """Train N independent fits with seeds [base, base+1, ..., base+N-1].

    Args:
        base_cls: NN survival class (e.g. SurvivalPCHazard, SurvivalLogisticHazard,
            DeepHitMT). Must accept a `random_state` keyword.
        n_snapshots: number of models to fit.
        base_kwargs: forwarded to base_cls; should NOT contain `random_state`
            (we provide it).
        random_state: starting seed; snapshot k uses seed = base + k.
        verbose: per-snapshot logs.
        skip_failed: if True, drop snapshots that fail to fit.
    """

    def __init__(self,
                 base_cls: Type,
                 n_snapshots: int = 5,
                 base_kwargs: Optional[dict] = None,
                 random_state: int = 42,
                 verbose: bool = False,
                 skip_failed: bool = True):
        self.base_cls = base_cls
        self.n_snapshots = int(n_snapshots)
        self.base_kwargs = {k: v for k, v in (base_kwargs or {}).items()
                            if k != 'random_state'}
        self.random_state = int(random_state)
        self.verbose = verbose
        self.skip_failed = skip_failed
        self._models: List = []

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        self._models = []
        n_failed = 0
        for k in range(self.n_snapshots):
            seed = self.random_state + k
            try:
                mdl = self.base_cls(random_state=seed, **self.base_kwargs)
            except TypeError:
                # Some classes don't accept random_state by name; try seed=
                try:
                    mdl = self.base_cls(seed=seed, **self.base_kwargs)
                except TypeError:
                    mdl = self.base_cls(**self.base_kwargs)
            try:
                mdl.fit(X, y_struct)
                self._models.append(mdl)
                if self.verbose:
                    print(f'  [snapshot {k + 1}/{self.n_snapshots}] '
                          f'seed={seed} fit')
            except Exception as exc:
                n_failed += 1
                if not self.skip_failed:
                    raise
                if self.verbose:
                    print(f'  [snapshot {k + 1}/{self.n_snapshots}] FAIL: {exc!r}')
        if not self._models:
            raise RuntimeError('All snapshot fits failed')
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError('SnapshotEnsembleSurvival not fitted')
        risks = []
        for mdl in self._models:
            try:
                risks.append(np.asarray(mdl.predict_risk(X), dtype=np.float64))
            except Exception:
                continue
        if not risks:
            raise RuntimeError('All snapshot predict_risk calls failed')
        return np.mean(np.stack(risks, axis=0), axis=0)
