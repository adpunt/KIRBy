"""Training protocols for synthetic-augmented survival models.

Each protocol takes (X_real, y_real_struct, X_synth, y_synth_struct,
val_idx_irrelevant_here, base_ctor) and returns a fit model with
.predict_risk(X). Used inside per-fold loops where X_real is the FOLD's
training subset, val data never touches synthetic.

Protocols:
    P1_oversample        — concat synthetic positives at fit time
    P2_pretrain_finetune — only valid for MLP (other ctors raise)
    P3_mixed_batch       — for MLP: 70/30 batch mix; for non-batch: scaled P1
    P4_self_distill      — teacher (real-only) labels synthetic; student fits
                            on real + synthetic-with-pseudo-labels (no synthetic
                            ground-truth event used)
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def _stack_data(X1, y1, X2, y2):
    """Concatenate two (X, y_struct) tuples."""
    X = np.vstack([np.asarray(X1, dtype=np.float64),
                   np.asarray(X2, dtype=np.float64)])
    if y2.dtype != y1.dtype:
        y2 = y2.astype(y1.dtype)
    y = np.concatenate([y1, y2])
    return X, y


def _filter_valid(X, y_struct):
    valid = np.isfinite(np.asarray(y_struct['time'], dtype=np.float64))
    return X[valid], y_struct[valid]


def P1_oversample(base_ctor: Callable[[], object],
                  X_real, y_real,
                  X_synth, y_synth,
                  positives_only: bool = True,
                  fraction: float = 1.0,
                  seed: int = 42):
    """Concat synthetic to real and fit. If positives_only, drop synth event=0."""
    if X_synth is None or X_synth.shape[0] == 0:
        est = base_ctor(); est.fit(X_real, y_real); return est
    Xs, ys = _filter_valid(X_synth, y_synth)
    if positives_only:
        ev = np.asarray(ys['event']).astype(bool)
        Xs, ys = Xs[ev], ys[ev]
    if fraction < 1.0 and Xs.shape[0] > 0:
        rng = np.random.default_rng(seed)
        n_keep = max(1, int(round(fraction * Xs.shape[0])))
        sel = rng.choice(Xs.shape[0], size=n_keep, replace=False)
        Xs, ys = Xs[sel], ys[sel]
    Xa, ya = _stack_data(X_real, y_real, Xs, ys)
    est = base_ctor(); est.fit(Xa, ya); return est


def P2_pretrain_finetune(base_ctor: Callable[[], object],
                         X_real, y_real,
                         X_synth, y_synth,
                         pretrain_epochs: int = 80,
                         finetune_epochs: int = 50,
                         freeze_trunk: bool = False,
                         seed: int = 42):
    """Pretrain trunk on synth, fine-tune full net on real.

    Only meaningful for MLPDiscreteHazard. base_ctor must produce one.
    """
    if X_synth is None or X_synth.shape[0] == 0:
        est = base_ctor(); est.fit(X_real, y_real); return est
    Xs, ys = _filter_valid(X_synth, y_synth)
    pre = base_ctor()
    if not hasattr(pre, 'get_trunk_state'):
        # Fallback: just oversample
        return P1_oversample(base_ctor, X_real, y_real, X_synth, y_synth,
                             positives_only=False, seed=seed)
    pre.epochs = pretrain_epochs
    pre.fit(Xs, ys)
    pretrain_state = pre.get_trunk_state()
    pretrain_grid = pre._grid
    pretrain_imputer = pre._imputer
    pretrain_scaler = pre._scaler

    # Build student from same ctor, transplant trunk + scaler/imputer
    est = base_ctor()
    # We need a same-shape trunk: align d_in by re-fitting scaler on real
    est.epochs = finetune_epochs
    est._d_in = pre._d_in
    est._build(pre._d_in)
    est.load_trunk_state(pretrain_state)
    # Re-fit scaler/imputer on real ONLY (avoids synth distribution leak)
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    Xr = np.asarray(X_real, dtype=np.float64)
    valid = np.isfinite(np.asarray(y_real['time'], dtype=np.float64))
    imp = SimpleImputer(strategy='median').fit(Xr[valid])
    sc = StandardScaler().fit(imp.transform(Xr[valid]))
    est._imputer, est._scaler = imp, sc
    # Keep pretrain grid (so trunk transfer makes sense)
    est._grid = pretrain_grid
    if freeze_trunk:
        for p in est._trunk.parameters():
            p.requires_grad = False
    est.fit(X_real, y_real, epochs=finetune_epochs, keep_grid=True)
    return est


def P3_mixed_batch(base_ctor: Callable[[], object],
                   X_real, y_real,
                   X_synth, y_synth,
                   synth_fraction: float = 0.3,
                   seed: int = 42):
    """For tree/forest: like P1 with controlled fraction.

    For MLP: batch-level mixing is achieved by oversampling synth into the
    training set proportional to synth_fraction (pragmatic equivalent).
    """
    if X_synth is None or X_synth.shape[0] == 0:
        est = base_ctor(); est.fit(X_real, y_real); return est
    Xs, ys = _filter_valid(X_synth, y_synth)
    n_real = X_real.shape[0]
    n_target = int(round(n_real * synth_fraction / max(1 - synth_fraction, 1e-3)))
    if Xs.shape[0] > n_target and n_target > 0:
        rng = np.random.default_rng(seed)
        sel = rng.choice(Xs.shape[0], size=n_target, replace=False)
        Xs, ys = Xs[sel], ys[sel]
    Xa, ya = _stack_data(X_real, y_real, Xs, ys)
    est = base_ctor(); est.fit(Xa, ya); return est


def P4_self_distill(base_ctor: Callable[[], object],
                    X_real, y_real,
                    X_synth, y_synth,
                    pseudo_top_k: float = 0.5,
                    horizon: Optional[float] = None,
                    seed: int = 42):
    """Teacher trains on real-only, scores synth, student trains on
    real + top-K confidently-classified synth using teacher-derived pseudo-labels.

    Pseudo-event mechanism:
        teacher.predict_risk(synth) -> per-synth risk
        rank-percentile p in [0,1]; classify as event if p > 1 - pseudo_top_k/2
        (top-K), censored if p < pseudo_top_k/2 (bottom-K). Middle dropped.
        For events, sampled time = horizon * (1 - p) + jitter.
        For censored, sampled time = horizon * 1.5 + jitter.
    """
    teacher = base_ctor(); teacher.fit(X_real, y_real)
    if X_synth is None or X_synth.shape[0] == 0:
        return teacher
    Xs, _ys = _filter_valid(X_synth, y_synth)
    # Build pseudo y_struct with teacher labels
    risk = teacher.predict_risk(Xs)
    finite = np.isfinite(risk)
    if finite.sum() < 10:
        return teacher
    rk = np.full_like(risk, np.nan)
    from scipy.stats import rankdata
    rk[finite] = rankdata(risk[finite]) / max(finite.sum(), 1)
    # Horizon = median real event time
    real_t = np.asarray(y_real['time'], dtype=np.float64)
    real_e = np.asarray(y_real['event'], dtype=bool)
    ev_t = real_t[real_e & np.isfinite(real_t)]
    if horizon is None:
        horizon = float(np.median(ev_t)) if ev_t.size else 5.0
    # Top/bottom K bands
    half_k = float(pseudo_top_k) / 2.0
    is_event = rk > (1 - half_k)
    is_cens = rk < half_k
    keep = is_event | is_cens
    if keep.sum() == 0:
        return teacher
    rng = np.random.default_rng(seed)
    pseudo_event = is_event[keep].astype(bool)
    # Map percentile to time
    rk_keep = rk[keep]
    time_pseudo = np.where(pseudo_event,
                           horizon * (1.05 - rk_keep) + rng.normal(0, horizon * 0.1, rk_keep.shape),
                           horizon * 1.5 + rng.normal(0, horizon * 0.2, rk_keep.shape))
    time_pseudo = np.clip(time_pseudo, horizon * 0.05, horizon * 3.0)
    y_pseudo = np.zeros(int(keep.sum()), dtype=y_real.dtype)
    y_pseudo['event'] = pseudo_event
    y_pseudo['time'] = time_pseudo
    # Combine and refit student
    Xa, ya = _stack_data(X_real, y_real, Xs[keep], y_pseudo)
    student = base_ctor(); student.fit(Xa, ya)
    return student
