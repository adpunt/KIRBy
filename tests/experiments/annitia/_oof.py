"""Out-of-fold (OOF) risk vector storage for ANNITIA experiments (Phase X.4).

For every condition the search evaluates, we want a length-N vector of OOF
risk predictions per endpoint, where:

    - position i is the risk score predicted on patient i WHEN they were in
      the validation fold (so it's the model that did NOT see them in train).
    - patients that never appeared in any validation fold get NaN.
    - patients filtered out by valid_mask (NaN time) get NaN.

This vector is required by Layer 5 `stack_meta` (Coxnet over rank-transformed
OOF risks) and by importance-method ensembling. Storing one .npy per
(endpoint, config_id) keeps it cheap to load per fusion mode.

File layout:
    <output_dir>/oof/<endpoint>/<config_id>.npy

Filenames use config_id (16-char sha) to avoid collisions across
(views, framing, algorithm, tricks) combinations.

Vectors are float64 (NaN-safe). Read with `np.load(...)` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


def init_oof_buffer(n_patients: int) -> np.ndarray:
    """Create a NaN-filled length-n vector ready to receive per-fold risks."""
    out = np.full(n_patients, np.nan, dtype=np.float64)
    return out


def fill_oof_fold(buffer: np.ndarray, val_idx: np.ndarray,
                  fold_risk: np.ndarray) -> None:
    """In-place: write `fold_risk` (length=len(val_idx)) into `buffer[val_idx]`.

    If a position in `val_idx` already has a non-NaN value (shouldn't happen
    in a clean k-fold split, but defensive), the new value overwrites it
    and a warning is *not* printed — caller should validate folds upstream.
    """
    val_idx = np.asarray(val_idx, dtype=np.int64)
    fold_risk = np.asarray(fold_risk, dtype=np.float64)
    if val_idx.shape[0] != fold_risk.shape[0]:
        raise ValueError(
            f'val_idx length {val_idx.shape[0]} does not match risk length {fold_risk.shape[0]}')
    buffer[val_idx] = fold_risk


def save_oof(buffer: np.ndarray, output_dir: Path | str,
             endpoint: str, config_id: str) -> str:
    """Save the OOF buffer under <output_dir>/oof/<endpoint>/<config_id>.npy.
    Returns the absolute path as a string."""
    out_dir = Path(output_dir) / 'oof' / endpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{config_id}.npy'
    # Atomic write: np.save force-appends `.npy` to its filename argument if
    # absent, so we hand it a stem and rename the resulting file.
    tmp_stem = out_dir / f'{config_id}.tmp'
    np.save(str(tmp_stem), buffer)
    tmp_path = Path(str(tmp_stem) + '.npy')
    tmp_path.replace(path)
    return str(path)


def load_oof(output_dir: Path | str, endpoint: str, config_id: str) -> np.ndarray:
    path = Path(output_dir) / 'oof' / endpoint / f'{config_id}.npy'
    return np.load(path)


def coverage(buffer: np.ndarray) -> Tuple[int, int]:
    """Return (n_filled, n_total) for diagnostics."""
    return int(np.isfinite(buffer).sum()), int(buffer.size)


def assert_full_coverage(buffer: np.ndarray, valid_mask: np.ndarray,
                         endpoint: str = '') -> None:
    """Verify every valid patient has an OOF prediction. Raises on mismatch.

    Patients with valid_mask=False are expected to be NaN; patients with
    valid_mask=True must be finite.
    """
    expected_finite = valid_mask
    actual_finite = np.isfinite(buffer)
    missing = expected_finite & ~actual_finite
    extra = ~expected_finite & actual_finite
    if missing.any() or extra.any():
        raise AssertionError(
            f'[{endpoint}] OOF coverage mismatch: '
            f'{missing.sum()} valid patients missing OOF, '
            f'{extra.sum()} invalid patients have OOF')
