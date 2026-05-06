"""Endpoint cross-pollination for ANNITIA (Phase D.9).

Hepatic and death are correlated. The risk of dying within 10 years is
informative about whether a hepatic event will occur (and vice versa) —
people who die early may not have time to develop hepatic events; people
flagged as high hepatic risk are likely sicker overall and at higher death
risk too.

Trick:
    1. Train a "donor" model on endpoint A using only original features.
    2. Compute its OOF risk vector r_A on the FULL cohort (n elements with
       NaN for invalid rows in endpoint A).
    3. Append r_A as an extra column to the feature matrix.
    4. Train a "recipient" model on endpoint B using the augmented features.

Symmetric: hepatic → death, death → hepatic, or both ways (yielding two
separate recipient models).

Why OOF:
    Using in-fold predictions would leak label information from endpoint A
    into endpoint B's training and pollute the C-index. The OOF protocol
    ensures that for every patient i, the donor's contribution to feature[i]
    was made by a model that did not see patient i's endpoint-A outcome.

Implementation:
    `attach_oof_feature(X, oof_path, fill=...)`
        — loads .npy from `oof_path`, returns X with one extra column.
        — NaNs in the OOF (invalid donor patients) are filled with the
          training-set median of the OOF column.

    `CrossPollinateSurvival(base_cls, donor_oof_path, ...)`
        — wraps a base survival model. At fit/predict time, attaches the
          donor OOF column to X. The donor OOF must be precomputed by Phase
          X.4 (search.py writes <output_dir>/oof/<endpoint>/<config_id>.npy).

The OOF vector covers ALL n patients (NaN where the donor's valid_mask was
False). At fit time we rely on the same global indexing — caller must pass
the FULL row indexing to recover OOF[i] for each row in X.

API: pass `row_index` (the original patient indices for the X matrix) so
the wrapper can pick OOF[row_index] correctly.

This trick assumes the search runner has already produced an OOF .npy for
endpoint A. If not, fit will raise FileNotFoundError.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

import numpy as np


def _load_oof_vector(oof_path: str) -> np.ndarray:
    p = Path(oof_path)
    if not p.exists():
        raise FileNotFoundError(f'donor OOF not found: {oof_path}')
    return np.asarray(np.load(p), dtype=np.float64)


def _fill_nan_with_median(col: np.ndarray, ref_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Fill NaN in `col` with median of `col[ref_idx]`. If ref_idx None, use
    `col` itself."""
    col = np.asarray(col, dtype=np.float64)
    ref = col if ref_idx is None else col[ref_idx]
    finite = np.isfinite(ref)
    fill = float(np.median(ref[finite])) if finite.any() else 0.0
    out = np.where(np.isfinite(col), col, fill)
    return out


def attach_oof_feature(X: np.ndarray, oof_full: np.ndarray,
                       row_index: np.ndarray,
                       train_row_index: Optional[np.ndarray] = None) -> np.ndarray:
    """Append OOF[row_index] as a column to X.

    Args:
        X: shape (m, d) feature matrix for the rows we're augmenting.
        oof_full: shape (N,) OOF vector indexed by GLOBAL patient id.
        row_index: shape (m,) global patient ids matching X's rows.
        train_row_index: optional global ids used to compute the median fill;
            when training, set to TRAINING fold ids so test-time rows don't
            leak their own statistics.
    """
    X = np.asarray(X, dtype=np.float64)
    oof_full = np.asarray(oof_full, dtype=np.float64)
    if oof_full.size <= row_index.max():
        raise ValueError(
            f'oof vector length {oof_full.size} smaller than max '
            f'row_index {row_index.max()}')
    col = oof_full[row_index]
    if train_row_index is not None:
        # Compute fill from training rows only.
        fill_ref = oof_full[train_row_index]
        finite = np.isfinite(fill_ref)
        fill = float(np.median(fill_ref[finite])) if finite.any() else 0.0
    else:
        finite = np.isfinite(col)
        fill = float(np.median(col[finite])) if finite.any() else 0.0
    col = np.where(np.isfinite(col), col, fill)
    return np.column_stack([X, col])


class CrossPollinateSurvival:
    """Wraps a base survival model and prepends a donor-OOF column to X.

    Args:
        base_cls: any class with .fit(X, y_struct) / .predict_risk(X).
        donor_oof_path: path to the donor's OOF .npy (length-N).
        base_kwargs: forwarded to base_cls.

    The constructor accepts donor_oof_path eagerly — useful so callers see
    FileNotFoundError immediately if the donor hasn't been run yet.
    """

    def __init__(self,
                 base_cls: Type,
                 donor_oof_path: str,
                 base_kwargs: Optional[dict] = None):
        self.base_cls = base_cls
        self.donor_oof_path = donor_oof_path
        self.base_kwargs = dict(base_kwargs or {})
        self._oof = _load_oof_vector(donor_oof_path)
        self._inner = None
        self._fill_value: Optional[float] = None
        self._train_row_index: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y_struct: np.ndarray,
            row_index: np.ndarray):
        row_index = np.asarray(row_index, dtype=np.int64)
        X_aug = attach_oof_feature(X, self._oof, row_index, train_row_index=row_index)
        # Stash the median fill from training so predict-time augmentation
        # can use the same value (no leakage from val rows' own OOF distrib).
        train_oof = self._oof[row_index]
        finite = np.isfinite(train_oof)
        self._fill_value = float(np.median(train_oof[finite])) if finite.any() else 0.0
        self._train_row_index = row_index
        self._inner = self.base_cls(**self.base_kwargs)
        self._inner.fit(X_aug, y_struct)
        return self

    def predict_risk(self, X: np.ndarray, row_index: np.ndarray) -> np.ndarray:
        if self._inner is None:
            raise RuntimeError('CrossPollinateSurvival not fitted')
        row_index = np.asarray(row_index, dtype=np.int64)
        col = self._oof[row_index]
        col = np.where(np.isfinite(col), col, self._fill_value or 0.0)
        X_aug = np.column_stack([np.asarray(X, dtype=np.float64), col])
        return self._inner.predict_risk(X_aug)
