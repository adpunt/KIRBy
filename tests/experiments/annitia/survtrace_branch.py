"""SurvTRACE wrapper for joint hep+death competing-risk modelling.

SurvTRACE (RyanWangZf/SurvTRACE) is a transformer-based competing-risk
model. We pre-format both endpoints into the SurvTRACE-expected schema,
fit on the union, then expose per-cause `predict_risk` for each endpoint.

Falls back to ImportError if `survtrace` isn't installed; the runner
catches it and skips the condition.

Two cause-of-interest framings are supported per call:
    cause_of_interest = 'hepatic' or 'death'
The model is trained ONCE on both causes; each instance just exposes the
predict_risk for one cause.
"""
from __future__ import annotations

import numpy as np


class SurvTRACEBranch:
    """SurvTRACE wrapper around joint (hepatic, death) competing-risk fit.

    Usage:
        m = SurvTRACEBranch(cause_of_interest='hepatic')
        m.fit(X, y_hep, y_death)         # competing-risk joint fit
        risk = m.predict_risk(X_test)
    """

    def __init__(self, cause_of_interest: str = 'hepatic',
                 epochs: int = 50, batch_size: int = 64, lr: float = 1e-3,
                 hidden_size: int = 64, n_heads: int = 2, n_layers: int = 2,
                 horizon_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)):
        if cause_of_interest not in ('hepatic', 'death'):
            raise ValueError("cause_of_interest must be 'hepatic' or 'death'")
        self.cause_of_interest = cause_of_interest
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.hidden_size = int(hidden_size)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.horizon_quantiles = tuple(horizon_quantiles)
        self._model = None
        self._cuts = None

    def _import(self):
        try:
            from survtrace.model import SurvTraceMulti  # noqa: F401
            from survtrace.config import STConfig  # noqa: F401
            from survtrace.utils import set_random_seed  # noqa: F401
        except ImportError as e:
            raise ImportError(
                'survtrace not installed — `pip install survtrace`'
            ) from e

    def fit(self, X, y_hep, y_death):
        self._import()
        from survtrace.model import SurvTraceMulti
        from survtrace.config import STConfig

        import pandas as pd
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        # Build SurvTRACE-style competing-risk labels: cause label per row
        # (1=hep first, 2=death first, 0=censored)
        time_hep = np.asarray(y_hep['time'], dtype=np.float64)
        ev_hep   = np.asarray(y_hep['event'], dtype=bool)
        time_de  = np.asarray(y_death['time'], dtype=np.float64)
        ev_de    = np.asarray(y_death['event'], dtype=bool)
        # First-event time + cause
        any_event = ev_hep | ev_de
        time_first = np.where(ev_hep & ev_de, np.minimum(time_hep, time_de),
                              np.where(ev_hep, time_hep,
                                       np.where(ev_de, time_de,
                                                np.fmax(time_hep, time_de))))
        cause = np.where(ev_hep & ev_de,
                         np.where(time_hep <= time_de, 1, 2),
                         np.where(ev_hep, 1, np.where(ev_de, 2, 0)))
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(d)])
        df['duration'] = time_first
        df['event'] = cause
        # Discretise time
        ev_pos = time_first[any_event & np.isfinite(time_first)]
        if ev_pos.size < 3:
            raise ValueError('SurvTRACE: too few events for time discretisation')
        cuts = np.quantile(ev_pos, self.horizon_quantiles)
        self._cuts = cuts
        STConfig['data'] = 'annitia'
        STConfig['num_categorical_feature'] = 0
        STConfig['num_numerical_feature'] = d
        STConfig['num_feature'] = d
        STConfig['num_event'] = 2
        STConfig['hidden_size'] = self.hidden_size
        STConfig['intermediate_size'] = self.hidden_size * 2
        STConfig['num_hidden_layers'] = self.n_layers
        STConfig['num_attention_heads'] = self.n_heads
        STConfig['duration_index'] = list(map(float, cuts))
        STConfig['out_feature'] = len(cuts)
        try:
            self._model = SurvTraceMulti(STConfig)
            from survtrace.train_utils import Trainer
            tr = Trainer(self._model)
            tr.fit((df.drop(columns=['duration', 'event']),
                    df[['duration', 'event']]),
                   batch_size=self.batch_size, epochs=self.epochs, learning_rate=self.lr)
        except Exception as e:
            raise ImportError(f'SurvTRACE fit failed: {e}') from e
        return self

    def predict_risk(self, X):
        if self._model is None:
            raise RuntimeError('not fit')
        import pandas as pd
        X = np.asarray(X, dtype=np.float64)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        cif = self._model.predict_hazard(df)  # dict {1: array(n, T), 2: ...}
        cause_idx = 1 if self.cause_of_interest == 'hepatic' else 2
        haz = np.asarray(cif[cause_idx], dtype=np.float64)
        # Cumulative hazard at last horizon → risk score
        return haz[:, -1]
