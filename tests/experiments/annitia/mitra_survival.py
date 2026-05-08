"""MITRA tabular foundation-model classifier wrapper.

MITRA (Amazon AWS, NeurIPS 2025) is shipped via AutoGluon as
`autogluon.tabular.models.mitra`. We expose a thin sklearn-style
classifier over it so DiscretisedSurvival can plug it into the per-bin
classification reduction.

Falls back to ImportError at construction time if AutoGluon's MITRA
weights aren't installed; the surrounding runner skips this branch
gracefully (the existing `experiment_branches._run()` handles ImportError).
"""
from __future__ import annotations

import numpy as np


class MitraClassifier:
    """Sklearn-style binary classifier wrapping AutoGluon MITRA in-context.

    Parameters
    ----------
    n_estimators : int
        Number of in-context ensemble members. MITRA's default is 1; we
        bump to 4 for noisier rare-event bins.
    fine_tune : bool
        Whether to allow MITRA's lightweight fine-tune step. False keeps
        it strictly in-context.
    seed : int
    """

    def __init__(self, n_estimators: int = 4, fine_tune: bool = False,
                 seed: int = 0):
        self.n_estimators = int(n_estimators)
        self.fine_tune = bool(fine_tune)
        self.seed = int(seed)
        self._predictor = None
        self._cols = None

    def _import(self):
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as e:
            raise ImportError(
                'autogluon.tabular not installed — pip install '
                '"autogluon.tabular[mitra]"') from e
        return TabularPredictor

    def fit(self, X, y):
        # If the compute node lacks libcudnn (cudnn9), force CPU before any
        # autogluon/torch import so MITRA doesn't crash during init.
        # Detect by trying to load the library; if it fails, hide CUDA.
        import os, ctypes
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            try:
                ctypes.CDLL('libcudnn.so.9')
            except OSError:
                # Try cudnn 8 fallback before giving up
                try:
                    ctypes.CDLL('libcudnn.so.8')
                except OSError:
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
        TabularPredictor = self._import()
        import pandas as pd
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self._cols = [f'f{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=self._cols)
        df['__y'] = y
        # silenced-tmp dir per call so concurrent folds don't collide
        import tempfile
        path = tempfile.mkdtemp(prefix='mitra_')
        self._predictor = TabularPredictor(
            label='__y', problem_type='binary', path=path, verbosity=0)
        # If CUDA was disabled above, also tell autogluon to use 0 GPUs so
        # MITRA's import doesn't try to allocate cuda memory.
        force_cpu = os.environ.get('CUDA_VISIBLE_DEVICES', None) == ''
        ag_args = {'ag.max_memory_usage_ratio': 2.0}
        if force_cpu:
            ag_args['num_gpus'] = 0
        try:
            self._predictor.fit(
                df,
                hyperparameters={'MITRA': {
                    'n_estimators': self.n_estimators,
                    'fine_tune': self.fine_tune,
                    'seed': self.seed,
                    'ag_args_fit': ag_args,
                }},
                fit_weighted_ensemble=False,
                presets=None,
                verbosity=0,
                raise_on_no_models_fitted=False)
        except Exception as e:  # noqa: BLE001
            # Surface as ImportError so the runner skips MITRA cleanly if
            # weights aren't available for download.
            raise ImportError(f'MITRA fit failed: {e}') from e
        # If autogluon didn't actually train any model (e.g. still OOM,
        # weights download failed silently, …) raise so the runner skips.
        if not self._predictor.model_names():
            raise ImportError(
                'MITRA fit produced no models; check autogluon log + '
                'memory headroom (ag.max_memory_usage_ratio).')
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._predictor is None:
            raise RuntimeError('not fit')
        import pandas as pd
        X = np.asarray(X, dtype=np.float64)
        df = pd.DataFrame(X, columns=self._cols)
        proba = self._predictor.predict_proba(df)
        # AutoGluon returns DataFrame with column per class
        if hasattr(proba, 'values'):
            cols = list(proba.columns)
            try:
                idx = cols.index(1) if 1 in cols else cols.index('1')
            except ValueError:
                idx = -1
            P = proba.values[:, idx]
        else:
            P = np.asarray(proba)[:, -1]
        return np.column_stack([1.0 - P, P])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
