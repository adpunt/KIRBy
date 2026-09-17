#!/usr/bin/env python3
"""Smoke for the per-sample weight on DNNRegressor.fit and DNNClassifier.fit.

A weighted fit is only worth anything if the weight is read and if it means what
the other four models in the roster mean by it. Four things are checked, on both
the regressor and the classifier:

  1. every weight equal to one predicts what no weight at all predicts.
  2. a weight of zero on a block of rows predicts what deleting those rows
     predicts. Both hold with the fold's groups passed and without them.
  3. a weight that is NOT uniform moves the predictions. Without this, 1 and 2
     would also pass on a fit() that accepted the argument and threw it away.
  4. a weight that is the wrong length, negative, all zero, or NaN raises.

The tolerance for 1 and 2 is measured, not assumed: the same unweighted fit is
repeated at several values of random_state and the largest gap between any two
of those runs is what 1 and 2 have to stay under.

That spread is NOT the yardstick for 3. Changing random_state changes the weight
initialisation and the early-stopping carve together, which moves a prediction
by more than a reweighting does, so a real reweighting can sit under it. 3 holds
random_state fixed instead, where the fit is deterministic and 1 has already
shown a uniform weight changes nothing at all, so any movement is the weight
being read. It is called against the spread of the training labels, so a gap
that is floating-point dust cannot pass.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')

import warnings

import numpy as np

warnings.filterwarnings('ignore')

from kirby.models.dnn import DNNRegressor, DNNClassifier

FAIL = []

# Small enough to run in a minute, long enough to reach early stopping.
EPOCHS = 60
N = 240


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=N, seed=0, clf=False):
    """Two blocks of features, a scaffold-like grouping, 12 rows per group."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype(np.float32)
    signal = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2]
    if clf:
        y = (signal + rng.normal(0, 0.5, n) > 0).astype(np.int64)
    else:
        y = (signal + rng.normal(0, 0.2, n)).astype(np.float32)
    X_val = rng.normal(size=(60, 12)).astype(np.float32)
    groups = np.arange(n) // 12
    return X, y, groups, X_val


def fit_predict(kind, X, y, X_val, groups=None, sample_weight=None, seed=0):
    """One fit, one predict. Regression returns the prediction, classification
    returns P(y = the second class), which is the column the noise experiments
    hand to AUROC."""
    cls = DNNRegressor if kind == 'reg' else DNNClassifier
    m = cls(epochs=EPOCHS, random_state=seed)
    m.fit(X, y, groups=groups, sample_weight=sample_weight)
    if kind == 'reg':
        return m.predict(X_val)
    return m.predict_proba(X_val)[:, 1]


def run_to_run_spread(kind, X, y, X_val, groups, seeds=(0, 1, 2, 3)):
    """Largest gap on any validation row between two unweighted fits that differ
    only in random_state."""
    runs = [fit_predict(kind, X, y, X_val, groups=groups, seed=s) for s in seeds]
    worst = 0.0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            worst = max(worst, float(np.max(np.abs(runs[i] - runs[j]))))
    return worst


def test_weights_and_deletion(kind):
    label = 'DNNRegressor' if kind == 'reg' else 'DNNClassifier'
    print(f"{label}:")
    X, y, groups, X_val = make_data(clf=(kind == 'clf'))

    spread = run_to_run_spread(kind, X, y, X_val, groups)
    unit = 'label units' if kind == 'reg' else 'probability'
    print(f"  run-to-run spread over 4 values of random_state: "
          f"{spread:.6f} ({unit}), largest gap on any of "
          f"{len(X_val)} validation rows")

    for groups_arg, how in ((groups, 'with groups'), (None, 'without groups')):
        base = fit_predict(kind, X, y, X_val, groups=groups_arg)

        ones = fit_predict(kind, X, y, X_val, groups=groups_arg,
                           sample_weight=np.ones(len(y)))
        gap = float(np.max(np.abs(ones - base)))
        check(f'{label} {how}: every weight one matches no weight',
              gap <= spread,
              f"largest gap {gap:.6f}, spread {spread:.6f}")

        # Zero out 40 of the 240 rows, keeping whole groups together so the
        # group-aware early-stopping carve has the same groups either way.
        drop = np.zeros(len(y), dtype=bool)
        drop[groups < 4] = True
        w = np.ones(len(y))
        w[drop] = 0.0
        keep = ~drop

        zeroed = fit_predict(kind, X, y, X_val, groups=groups_arg,
                             sample_weight=w)
        deleted = fit_predict(
            kind, X[keep], y[keep], X_val,
            groups=(groups[keep] if groups_arg is not None else None))
        gap = float(np.max(np.abs(zeroed - deleted)))
        check(f'{label} {how}: a weight of zero on {int(drop.sum())} of '
              f'{len(y)} rows matches deleting them',
              gap <= spread,
              f"largest gap {gap:.6f}, spread {spread:.6f}")

    # A weight that is not uniform has to move something. Five times the weight
    # on half the rows, at the same random_state as the unweighted fit it is
    # compared against.
    base = fit_predict(kind, X, y, X_val, groups=groups)
    w = np.ones(len(y))
    w[groups % 2 == 0] = 5.0
    tilted = fit_predict(kind, X, y, X_val, groups=groups, sample_weight=w)
    gap = float(np.max(np.abs(tilted - base)))
    if kind == 'reg':
        scale = float(np.std(y))
        floor = 0.01 * scale
        detail = (f"largest gap {gap:.6f} label units, {gap / scale:.3f} of "
                  f"the training label spread ({scale:.4f})")
    else:
        floor = 0.01
        detail = f"largest gap {gap:.6f} probability"
    check(f'{label}: a weight of 5 on half the rows moves the predictions',
          gap > floor, detail)


def test_rejects_bad_weights(kind):
    label = 'DNNRegressor' if kind == 'reg' else 'DNNClassifier'
    X, y, groups, _ = make_data(n=60, clf=(kind == 'clf'))
    cases = {
        'wrong length': np.ones(len(y) - 1),
        'a negative weight': np.r_[-1.0, np.ones(len(y) - 1)],
        'every weight zero': np.zeros(len(y)),
        'a NaN weight': np.r_[np.nan, np.ones(len(y) - 1)],
    }
    for name, w in cases.items():
        cls = DNNRegressor if kind == 'reg' else DNNClassifier
        try:
            cls(epochs=2).fit(X, y, groups=groups, sample_weight=w)
            check(f'{label}: {name} raises', False, 'it was accepted')
        except ValueError as exc:
            check(f'{label}: {name} raises', True, str(exc).split('\n')[0])
        except Exception as exc:
            check(f'{label}: {name} raises ValueError', False,
                  f'{type(exc).__name__}: {exc}')


if __name__ == '__main__':
    for kind in ('reg', 'clf'):
        test_weights_and_deletion(kind)
        test_rejects_bad_weights(kind)
        print()
    print("=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
