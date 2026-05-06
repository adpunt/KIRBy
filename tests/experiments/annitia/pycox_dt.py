"""Discrete-time hazard NN wrappers (PCHazard, LogisticHazard) via pycox.

Both targets the same single-event survival problem with discrete-time
formulations:

    PCHazard
        — Predict piecewise-constant hazard. Best for longer follow-ups
          where the constant-hazard assumption per bin is reasonable.

    LogisticHazard
        — Predict bin-wise hazard via independent logistic regression
          targets per bin. Most flexible; handles tied events well.

Both wrappers share the same fit / predict_risk interface as the existing
sksurv-style classes (`SurvivalCoxPH`, etc.) so they're drop-in for
`SURVIVAL_MODELS`. The risk score is the CIF integrated up to the median
training-time horizon.

Lazy-import: pycox + torchtuples are heavy; install on the server only.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _drop_invalid_xy(X: np.ndarray, y_struct: np.ndarray):
    keep = ~np.isnan(y_struct['time'])
    return np.asarray(X, dtype=np.float64)[keep], y_struct[keep], keep


class _BasePycoxDT:
    """Shared fit / predict / risk machinery."""

    def __init__(self,
                 num_durations: int = 20,
                 hidden: Tuple[int, ...] = (64, 64),
                 dropout: float = 0.3,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 200,
                 patience: int = 20,
                 horizon_quantile: float = 0.5,
                 device: str = 'cpu',
                 random_state: int = 42,
                 verbose: bool = False):
        self.num_durations = num_durations
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.horizon_quantile = horizon_quantile
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self._scaler = None
        self._labtrans = None
        self._model = None
        self._horizon_idx = None

    def _build_net(self, in_features: int, num_outputs: int):
        try:
            import torchtuples as tt
        except ImportError as e:
            raise ImportError(
                'pycox / torchtuples missing — `pip install pycox==0.3.0 '
                f'torchtuples==0.2.2` ({e})')
        return tt.practical.MLPVanilla(
            in_features=in_features,
            num_nodes=list(self.hidden),
            out_features=num_outputs,
            batch_norm=True,
            dropout=self.dropout,
            output_bias=False,
        )

    def _label_transform(self, time, event):
        raise NotImplementedError

    def _model_class(self):
        raise NotImplementedError

    def _num_outputs(self):
        return self.num_durations

    def fit(self, X: np.ndarray, y_struct: np.ndarray):
        try:
            import torch  # noqa: F401
            import torchtuples as tt
        except ImportError as e:
            raise ImportError(f'pycox / torchtuples missing: {e}')
        from sklearn.preprocessing import StandardScaler

        X, y_struct, _ = _drop_invalid_xy(X, y_struct)
        time = y_struct['time'].astype(np.float64)
        event = y_struct['event'].astype(int)

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X).astype(np.float32)

        # Label transform (subclass)
        self._labtrans = self._label_transform(time, event)
        durations, events = self._labtrans.transform(time, event)

        # Train/val split for early stopping
        rng = np.random.default_rng(self.random_state)
        n = len(Xs)
        perm = rng.permutation(n)
        n_val = max(int(0.2 * n), 32)
        idx_val, idx_tr = perm[:n_val], perm[n_val:]

        net = self._build_net(in_features=Xs.shape[1],
                              num_outputs=self._num_outputs())
        ModelCls = self._model_class()
        self._model = ModelCls(net, tt.optim.Adam(self.lr),
                               duration_index=self._labtrans.cuts)
        cb = [tt.callbacks.EarlyStopping(patience=self.patience)]
        y_tr = (durations[idx_tr], events[idx_tr])
        y_va = (durations[idx_val], events[idx_val])
        self._model.fit(Xs[idx_tr], y_tr,
                        batch_size=self.batch_size, epochs=self.epochs,
                        callbacks=cb, val_data=(Xs[idx_val], y_va),
                        verbose=self.verbose)

        # Compute horizon index for risk integration
        horizon_t = float(np.quantile(time, self.horizon_quantile))
        cuts = np.asarray(self._labtrans.cuts, dtype=np.float64)
        self._horizon_idx = int(np.searchsorted(cuts, horizon_t, side='right') - 1)
        self._horizon_idx = max(0, min(self._horizon_idx, len(cuts) - 1))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
        # surv: shape (n_durations, n)
        surv = np.asarray(self._model.predict_surv(Xs))
        if surv.shape[0] != len(self._labtrans.cuts):
            # pycox sometimes returns transposed — handle either
            surv = surv.T
        # CIF at horizon = 1 - surv[horizon_idx]
        return (1.0 - surv[self._horizon_idx]).astype(np.float64)


class SurvivalPCHazard(_BasePycoxDT):
    """Piecewise-constant hazard NN."""

    def _label_transform(self, time, event):
        from pycox.models import PCHazard
        lt = PCHazard.label_transform(self.num_durations)
        # PCHazard needs (time, event) → (durations, events, intervals)
        # but the label_transform.fit_transform returns 3 arrays internally;
        # our `transform` wrapper above expects 2 — call fit_transform here
        # and stash the third intra-bin proportion via `self._frac`.
        durations, events, frac = lt.fit_transform(time, event)
        self._frac = frac
        return lt

    def _model_class(self):
        from pycox.models import PCHazard
        return PCHazard

    def fit(self, X, y_struct):
        # PCHazard's `fit` wants (durations, events, frac) as the y tuple.
        # Override to thread frac through.
        try:
            import torch  # noqa: F401
            import torchtuples as tt
        except ImportError as e:
            raise ImportError(f'pycox / torchtuples missing: {e}')
        from sklearn.preprocessing import StandardScaler

        X, y_struct, _ = _drop_invalid_xy(X, y_struct)
        time = y_struct['time'].astype(np.float64)
        event = y_struct['event'].astype(int)

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X).astype(np.float32)

        from pycox.models import PCHazard
        lt = PCHazard.label_transform(self.num_durations)
        durations, events, frac = lt.fit_transform(time, event)
        self._labtrans = lt

        rng = np.random.default_rng(self.random_state)
        n = len(Xs)
        perm = rng.permutation(n)
        n_val = max(int(0.2 * n), 32)
        idx_val, idx_tr = perm[:n_val], perm[n_val:]

        net = self._build_net(in_features=Xs.shape[1], num_outputs=self.num_durations)
        self._model = PCHazard(net, tt.optim.Adam(self.lr),
                               duration_index=lt.cuts)
        cb = [tt.callbacks.EarlyStopping(patience=self.patience)]
        y_tr = (durations[idx_tr], events[idx_tr], frac[idx_tr])
        y_va = (durations[idx_val], events[idx_val], frac[idx_val])
        self._model.fit(Xs[idx_tr], y_tr, batch_size=self.batch_size,
                        epochs=self.epochs, callbacks=cb,
                        val_data=(Xs[idx_val], y_va), verbose=self.verbose)

        horizon_t = float(np.quantile(time, self.horizon_quantile))
        cuts = np.asarray(lt.cuts, dtype=np.float64)
        self._horizon_idx = int(np.searchsorted(cuts, horizon_t, side='right') - 1)
        self._horizon_idx = max(0, min(self._horizon_idx, len(cuts) - 1))
        return self


class SurvivalLogisticHazard(_BasePycoxDT):
    """Logistic hazard NN (independent logistic per bin)."""

    def _label_transform(self, time, event):
        from pycox.models import LogisticHazard
        return LogisticHazard.label_transform(self.num_durations)

    def _model_class(self):
        from pycox.models import LogisticHazard
        return LogisticHazard

    def fit(self, X, y_struct):
        try:
            import torch  # noqa: F401
            import torchtuples as tt
        except ImportError as e:
            raise ImportError(f'pycox / torchtuples missing: {e}')
        from sklearn.preprocessing import StandardScaler

        X, y_struct, _ = _drop_invalid_xy(X, y_struct)
        time = y_struct['time'].astype(np.float64)
        event = y_struct['event'].astype(int)

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X).astype(np.float32)

        from pycox.models import LogisticHazard
        lt = LogisticHazard.label_transform(self.num_durations)
        durations, events = lt.fit_transform(time, event)
        self._labtrans = lt

        rng = np.random.default_rng(self.random_state)
        n = len(Xs)
        perm = rng.permutation(n)
        n_val = max(int(0.2 * n), 32)
        idx_val, idx_tr = perm[:n_val], perm[n_val:]

        net = self._build_net(in_features=Xs.shape[1], num_outputs=self.num_durations)
        self._model = LogisticHazard(net, tt.optim.Adam(self.lr),
                                     duration_index=lt.cuts)
        cb = [tt.callbacks.EarlyStopping(patience=self.patience)]
        y_tr = (durations[idx_tr], events[idx_tr])
        y_va = (durations[idx_val], events[idx_val])
        self._model.fit(Xs[idx_tr], y_tr, batch_size=self.batch_size,
                        epochs=self.epochs, callbacks=cb,
                        val_data=(Xs[idx_val], y_va), verbose=self.verbose)

        horizon_t = float(np.quantile(time, self.horizon_quantile))
        cuts = np.asarray(lt.cuts, dtype=np.float64)
        self._horizon_idx = int(np.searchsorted(cuts, horizon_t, side='right') - 1)
        self._horizon_idx = max(0, min(self._horizon_idx, len(cuts) - 1))
        return self


PYCOX_DT_MODELS = {
    'pchazard': SurvivalPCHazard,
    'logistic_hazard': SurvivalLogisticHazard,
}
