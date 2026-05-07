"""TabTune-based PEFT fine-tuning of TabPFN-v2.6 / MITRA / TabICL.

TabTune (github.com/Lexsi-Labs/TabTune) is a unified library that wraps
the major tabular foundation models with sklearn-style fit/predict +
PEFT and full-SFT APIs. We expose two binary-classifier wrappers
compatible with DiscretisedSurvival's per-bin reduction.

Variants:
    TabTuneTabPFN(strategy='peft' | 'sft')
    TabTuneMITRA  (strategy='peft' | 'sft')

Strategy semantics:
    'peft' — LoRA-style adapters (cheap, ~30 min/fold/endpoint on V100)
    'sft'  — full supervised fine-tune (heavier, server-only)

Falls back to ImportError at construction time if `tabtune` package
isn't installed; the runner skips the condition gracefully.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class _TabTuneBase:
    """Shared sklearn-style binary-classifier wrapper around TabTune."""

    MODEL_NAME = ''  # subclass sets this

    def __init__(self, strategy: str = 'peft', epochs: int = 8,
                 lr: float = 1e-3, batch_size: int = 32,
                 device: str = 'cpu', seed: int = 0):
        if strategy not in ('peft', 'sft'):
            raise ValueError("strategy must be 'peft' or 'sft'")
        self.strategy = strategy
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = int(seed)
        self._pipeline = None

    def _import(self):
        try:
            from tabtune.api import TabularPipeline  # type: ignore[import]
        except ImportError:
            try:
                from tabtune import TabularPipeline  # type: ignore[import]
            except ImportError as e:
                raise ImportError(
                    'tabtune not installed — '
                    'pip install git+https://github.com/Lexsi-Labs/TabTune.git'
                ) from e
        return TabularPipeline

    def fit(self, X, y):
        TabularPipeline = self._import()
        import pandas as pd
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        df['__y'] = y
        try:
            self._pipeline = TabularPipeline(
                model=self.MODEL_NAME,
                task='classification',
                strategy=self.strategy,
                epochs=self.epochs,
                lr=self.lr,
                batch_size=self.batch_size,
                device=self.device,
                seed=self.seed)
            self._pipeline.fit(df.drop(columns=['__y']), df['__y'])
        except Exception as e:  # noqa: BLE001
            # Surface as ImportError so the runner skips cleanly
            raise ImportError(f'TabTune {self.MODEL_NAME} fit failed: {e}') from e
        return self

    def predict_proba(self, X):
        if self._pipeline is None:
            raise RuntimeError('not fit')
        import pandas as pd
        X = np.asarray(X, dtype=np.float64)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        proba = self._pipeline.predict_proba(df)
        if hasattr(proba, 'values'):
            P = proba.values
        else:
            P = np.asarray(proba)
        if P.ndim == 1:
            return np.column_stack([1.0 - P, P])
        if P.shape[1] >= 2:
            return P[:, -2:]
        return np.column_stack([1.0 - P.ravel(), P.ravel()])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


class TabTuneTabPFN(_TabTuneBase):
    """TabPFN-v2.6 PEFT/SFT via TabTune."""
    MODEL_NAME = 'tabpfn-v2.6'


class TabTuneMITRA(_TabTuneBase):
    """MITRA PEFT/SFT via TabTune."""
    MODEL_NAME = 'mitra'
