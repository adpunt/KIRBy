"""Synthetic-positive augmentation for the rare hep endpoint (47 events).

Wraps any base survival ctor (Cox/RSF/GBSA/...). At fit time:
    1. Subset the training fold rows to event=1.
    2. Train a generator (TabDDPM / TabSyn / TabEBM / Gaussian-mixture
       fallback) on the (X, time) of those positives.
    3. Sample n_synth synthetic positives, concatenate to the training X
       and y_struct, and fit the base model on the augmented set.

Synthetic rows are NEVER added to validation or test folds — that's the
caller's responsibility (we only touch what's passed into .fit()).

Calibration concern (Goorbergh JAMIA 2022): augmentation can wreck
probability calibration even while improving rank C-index. The branches
runner reports Brier + IBS alongside C-index when this wrapper is used,
so calibration regressions are visible.

Generator backends:
    'tabddpm' : tabddpm pip package (preferred for ≥30 positives)
    'tabsyn'  : tabsyn pip package
    'tabebm'  : tabebm pip package (designed for low-N class-conditional)
    'gmm'     : sklearn GaussianMixture (always-available fallback)

`safe_fallback` = True (default): if the requested backend isn't
installed, we silently downgrade to GMM rather than failing the whole
condition.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _gmm_sample(X_pos: np.ndarray, t_pos: np.ndarray, n_synth: int,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Always-available Gaussian-mixture fallback. Models (X, t) jointly."""
    from sklearn.mixture import GaussianMixture
    n, d = X_pos.shape
    Xt = np.column_stack([X_pos, t_pos.reshape(-1, 1)])
    n_components = max(1, min(5, n // 8))
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag',
                          random_state=seed, reg_covar=1e-3, max_iter=200)
    gmm.fit(Xt)
    Z, _ = gmm.sample(n_synth)
    return Z[:, :d], np.clip(Z[:, d], a_min=t_pos.min() / 2, a_max=t_pos.max() * 1.2)


def _tabddpm_sample(X_pos, t_pos, n_synth, seed):
    """TabDDPM via synthcity (`pip install synthcity`)."""
    try:
        from synthcity.plugins import Plugins
    except ImportError as e:
        raise ImportError('synthcity not installed (provides TabDDPM)') from e
    import pandas as pd
    d = X_pos.shape[1]
    df = pd.DataFrame(np.column_stack([X_pos, t_pos.reshape(-1, 1)]),
                      columns=[f'f{i}' for i in range(d)] + ['__t'])
    plugin = Plugins().get('ddpm', random_state=seed)
    plugin.fit(df)
    Z = plugin.generate(count=n_synth).dataframe().to_numpy()
    return Z[:, :d], Z[:, d]


def _tabsyn_sample(X_pos, t_pos, n_synth, seed):
    """TabSyn via synthcity (`pip install synthcity`). Falls back to ARF
    plugin if 'tabsyn' specifically isn't registered (synthcity ships it
    as 'tvae' for the latent-diffusion family)."""
    try:
        from synthcity.plugins import Plugins
    except ImportError as e:
        raise ImportError('synthcity not installed (provides TVAE/ARF)') from e
    import pandas as pd
    d = X_pos.shape[1]
    df = pd.DataFrame(np.column_stack([X_pos, t_pos.reshape(-1, 1)]),
                      columns=[f'f{i}' for i in range(d)] + ['__t'])
    candidates = ('tabsyn', 'tvae', 'arf')
    plugin = None
    for name in candidates:
        try:
            plugin = Plugins().get(name, random_state=seed)
            break
        except Exception:
            continue
    if plugin is None:
        raise ImportError('No TabSyn/TVAE/ARF synthcity plugin available')
    plugin.fit(df)
    Z = plugin.generate(count=n_synth).dataframe().to_numpy()
    return Z[:, :d], Z[:, d]


def _tabebm_sample(X_pos, t_pos, n_synth, seed):
    """TabEBM via tab-ebm (`pip install tab-ebm`)."""
    try:
        from tab_ebm import TabEBM
    except ImportError:
        try:
            from tabebm import TabEBM  # alternate package name
        except ImportError as e:
            raise ImportError(
                'tab-ebm not installed — pip install tab-ebm'
            ) from e
    Xt = np.column_stack([X_pos, t_pos.reshape(-1, 1)])
    y = np.ones(len(Xt), dtype=np.int64)  # class-conditional with single class
    model = TabEBM()
    model.fit(Xt, y)
    Z = model.generate(num_samples=n_synth, y_target=1, seed=seed)
    Z = np.asarray(Z, dtype=np.float64)
    d = X_pos.shape[1]
    return Z[:, :d], Z[:, d]


_BACKENDS = {
    'tabddpm': _tabddpm_sample,
    'tabsyn':  _tabsyn_sample,
    'tabebm':  _tabebm_sample,
    'gmm':     _gmm_sample,
}


class SyntheticAugmentedSurvival:
    """Survival model wrapped with synthetic-positive augmentation at fit.

    Parameters
    ----------
    base_ctor : Callable[[], object]
        Returns a fresh survival model implementing fit(X, y_struct) /
        predict_risk(X). The augmenter does not touch the base model;
        any failure inside the generator raises with backend info.
    backend : str
    n_synth : int
    seed : int
    safe_fallback : bool
        If True and the requested backend isn't installed, silently
        downgrade to 'gmm'.
    """

    def __init__(self, base_ctor: Callable[[], object],
                 backend: str = 'gmm', n_synth: int = 100, seed: int = 0,
                 safe_fallback: bool = True):
        if backend not in _BACKENDS:
            raise ValueError(f'unknown backend: {backend}')
        self.base_ctor = base_ctor
        self.backend = backend
        self.n_synth = int(n_synth)
        self.seed = int(seed)
        self.safe_fallback = bool(safe_fallback)
        self._inner = None
        self.actual_backend_: str | None = None
        self.n_synth_actual_: int = 0

    def _augment(self, X: np.ndarray, y_struct: np.ndarray):
        time = np.asarray(y_struct['time'], dtype=np.float64)
        event = np.asarray(y_struct['event'], dtype=bool)
        finite = np.isfinite(time)
        pos_mask = finite & event
        n_pos = int(pos_mask.sum())
        if n_pos < 5:
            # Not enough positives to model — return originals
            self.actual_backend_ = 'none'
            self.n_synth_actual_ = 0
            return X, y_struct
        X_pos = X[pos_mask]
        t_pos = time[pos_mask]
        sampler = _BACKENDS[self.backend]
        try:
            X_syn, t_syn = sampler(X_pos, t_pos, self.n_synth, self.seed)
            self.actual_backend_ = self.backend
        except ImportError:
            if not self.safe_fallback:
                raise
            X_syn, t_syn = _gmm_sample(X_pos, t_pos, self.n_synth, self.seed)
            self.actual_backend_ = 'gmm-fallback'
        self.n_synth_actual_ = int(X_syn.shape[0])
        # Build new y_struct
        time_new = np.concatenate([time, np.asarray(t_syn, dtype=np.float64)])
        event_new = np.concatenate([event, np.ones(self.n_synth_actual_, dtype=bool)])
        y_new = np.zeros(time_new.size, dtype=y_struct.dtype)
        y_new['time'] = time_new
        y_new['event'] = event_new
        X_new = np.vstack([X, X_syn])
        return X_new, y_new

    def fit(self, X, y_struct):
        X = np.asarray(X, dtype=np.float64)
        X_aug, y_aug = self._augment(X, y_struct)
        self._inner = self.base_ctor()
        self._inner.fit(X_aug, y_aug)
        return self

    def predict_risk(self, X):
        if self._inner is None:
            raise RuntimeError('not fit')
        return self._inner.predict_risk(np.asarray(X, dtype=np.float64))


def brier_score_at_horizon(y_struct_train, y_struct_test, risk_test,
                           horizon: float) -> float:
    """Crude single-horizon Brier (no IPCW). Returns NaN if unusable.

    For calibration regression detection in the augmentation conditions —
    a 10% degradation vs no-aug is the abort threshold.
    """
    time = np.asarray(y_struct_test['time'], dtype=np.float64)
    event = np.asarray(y_struct_test['event'], dtype=bool)
    risk = np.asarray(risk_test, dtype=np.float64)
    finite = np.isfinite(time) & np.isfinite(risk)
    if finite.sum() < 5:
        return float('nan')
    # P(event by horizon) — approximate via rank-normalised risk
    p = (risk - np.nanmin(risk[finite])) / (np.nanmax(risk[finite]) - np.nanmin(risk[finite]) + 1e-9)
    label = ((event & (time <= horizon)) | (~event & (time <= horizon))).astype(int)
    # Restrict to patients with status known by horizon
    known = (event & (time <= horizon)) | (time >= horizon)
    if known.sum() < 5:
        return float('nan')
    return float(np.mean((p[known] - label[known].astype(float)) ** 2))
