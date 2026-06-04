#!/usr/bin/env python3
"""Fast unit smoke tests for the load-bearing helpers of n08 / n11 / n12.

These are deliberately CHEAP — no find_best_hybrid, no real datasets, no
foundation models. They lock in the bugs fixed during the 2026-06 noise audit
so a future edit (or a stale-library regression) trips here immediately instead
of silently producing empty result JSONs again.

Run:
    python tests/experiments/noise/_smoke_n08_n12.py
Exit 0 = all green.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import inspect
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from experiments.noise.n12_importance_weighted_distance import (  # noqa: E402
    _weight_features, _inject_noise as n12_inject, _run_condition,
)
from experiments.noise.n08_estimation_survey import _make_estimator  # noqa: E402
from experiments.noise.n11_estimation_remediation_grid import (  # noqa: E402
    _run_estimator_grid,
)
from kirby.utils.noise_estimation import (  # noqa: E402
    get_noise_estimator, NeighborResidualEstimator, weights_from_scores,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ''))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# n12 _weight_features: standardize-then-weight must CHANGE the geometry
# (the previous weight-then-standardize version was a no-op). All-zero
# importance must return the input unchanged.
# ---------------------------------------------------------------------------
def test_weight_features_changes_geometry():
    print("n12._weight_features:")
    rng = np.random.RandomState(0)
    Xs = StandardScaler().fit_transform(rng.randn(60, 8))
    imp = np.array([5., 4., 0.1, 0.05, 3., 0.0, 0.2, 1.0])
    Xw = _weight_features(Xs, imp)
    du, dw = pdist(Xs), pdist(Xw)
    check('weighted distances differ from unweighted',
          not np.allclose(du, dw),
          f'mean dist {du.mean():.3f} -> {dw.mean():.3f}')
    check('weighting is not a trivial global rescale (rank corr < 0.999)',
          abs(np.corrcoef(du, dw)[0, 1]) < 0.999)
    Xz = _weight_features(Xs, np.zeros(8))
    check('all-zero importance returns input unchanged',
          np.allclose(Xz, Xs))


# ---------------------------------------------------------------------------
# n12 _inject_noise: regression adds Gaussian (changes values, same length);
# classification flips a fraction of labels to OTHER classes only.
# ---------------------------------------------------------------------------
def test_inject_noise():
    print("n12._inject_noise:")
    y = np.linspace(0, 10, 100)
    yn = n12_inject(y, 0.3, seed=1, is_clf=False)
    check('regression noise changes values', not np.allclose(y, yn))
    check('regression noise preserves length', len(yn) == len(y))

    yc = np.array([0, 1] * 50)
    ycn = n12_inject(yc, 0.3, seed=1, is_clf=True)
    n_changed = int((yc != ycn).sum())
    check('clf noise flips ~sigma fraction',
          5 <= n_changed <= 45, f'{n_changed} flipped of 100')
    check('clf noise keeps labels within original class set',
          set(np.unique(ycn)).issubset(set(np.unique(yc))))


# ---------------------------------------------------------------------------
# n08 _make_estimator: is_clf=True must reach kNN estimators that accept it,
# and must NOT be passed to model-based estimators that don't (no TypeError).
# ---------------------------------------------------------------------------
def test_make_estimator_is_clf_routing():
    print("n08._make_estimator:")
    knn = _make_estimator('neighbor_residual', is_clf=True)
    check('kNN estimator receives is_clf=True',
          getattr(knn, 'is_clf', None) is True)
    knn_reg = _make_estimator('neighbor_residual', is_clf=False)
    check('kNN estimator defaults is_clf=False',
          getattr(knn_reg, 'is_clf', True) is False)
    # Model-based estimator without an is_clf arg must still construct cleanly
    # under is_clf=True (the bug would have raised TypeError).
    try:
        mb = _make_estimator('cross_val_residual', is_clf=True)
        ok = mb is not None and not hasattr(mb, 'is_clf')
    except TypeError as e:
        ok = False
        print(f"      unexpected TypeError: {e}")
    check('model-based estimator constructs under is_clf without TypeError', ok)


# ---------------------------------------------------------------------------
# Library contract the experiments depend on: estimators expose BOTH .fit()
# (+ noise_scores_) AND .estimate(). The stale-library era that produced the
# empty n08 TCGA / n12 JSONs lacked these — guard against regressing to it.
# ---------------------------------------------------------------------------
def test_library_api_contract():
    print("noise_estimation API contract:")
    est = get_noise_estimator('neighbor_residual')
    check('estimator has .fit', hasattr(est, 'fit'))
    check('estimator has .estimate', hasattr(est, 'estimate'))
    X = np.random.RandomState(0).randn(40, 6)
    y = np.random.RandomState(1).randn(40)
    est.fit(X, y)
    check('estimator exposes noise_scores_ after fit',
          getattr(est, 'noise_scores_', None) is not None)
    s2 = get_noise_estimator('neighbor_residual').estimate(X, y)
    check('.estimate returns one score per sample', len(s2) == len(y))
    for alias in ('distance_weighted_residual', 'cross_val_residual',
                  'ensemble_disagreement', 'hybrid_noise'):
        try:
            get_noise_estimator(alias)
            ok = True
        except Exception:
            ok = False
        check(f'estimator name "{alias}" resolves', ok)


# ---------------------------------------------------------------------------
# n12 _run_condition end-to-end on tiny arrays: must populate every
# (estimator, k) cell with a real float and record NO _errors. This is the
# exact path whose silent failure produced empty per_importance_method.
# ---------------------------------------------------------------------------
def test_run_condition_populates(is_clf):
    label = 'clf' if is_clf else 'reg'
    print(f"n12._run_condition ({label}):")
    rng = np.random.RandomState(0)
    n = 80
    X = rng.randn(n, 5)
    if is_clf:
        y = (X[:, 0] > 0).astype(int)
        metric_fn = lambda a, b: float((np.asarray(a) == np.asarray(b)).mean())
        factory = lambda: RandomForestClassifier(n_estimators=20, random_state=0)
    else:
        y = X[:, 0] + 0.1 * rng.randn(n)
        metric_fn = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
        factory = lambda: RandomForestRegressor(n_estimators=20, random_state=0)
    tr, va = np.arange(0, 60), np.arange(60, 80)
    Xs = StandardScaler().fit_transform(X)
    res = _run_condition(Xs[tr], y[tr], Xs[tr], Xs[va], y[va],
                         factory, metric_fn, is_clf=is_clf)
    errs = res.get('_errors', {})
    cells = {k: v for k, v in res.items() if not k.startswith('_')}
    check(f'{label}: no _errors recorded', not errs,
          str(errs)[:120] if errs else '')
    check(f'{label}: all 9 (estimator,k) cells produced a float',
          len(cells) == 9 and all(isinstance(v, float) for v in cells.values()),
          f'{len(cells)} cells')


# ---------------------------------------------------------------------------
# n11 _run_estimator_grid: regression must fill BOTH weighting (12) and
# smoothing (16) cells; classification must SKIP smoothing (RF classifier can't
# consume continuous smoothed floats) and pass is_clf to the kNN estimator so
# it produces class-disagreement scores instead of erroring.
# ---------------------------------------------------------------------------
def test_n11_run_estimator_grid():
    print("n11._run_estimator_grid:")
    rng = np.random.RandomState(0)
    n = 100
    X = rng.randn(n, 6)
    Xs = StandardScaler().fit_transform(X)
    tr, va = np.arange(0, 75), np.arange(75, 100)

    yr = X[:, 0] + 0.1 * rng.randn(n)
    rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
    facr = lambda: RandomForestRegressor(n_estimators=15, random_state=0)
    r = _run_estimator_grid('neighbor_residual', 'euclidean',
                            Xs[tr], yr[tr], Xs[tr], Xs[va], yr[va],
                            facr, rmse, n, is_clf=False)
    cells = [k for k in r if not k.startswith('_')]
    check('reg: 28 grid cells (12 weighting + 16 smoothing)',
          len(cells) == 28, f'{len(cells)} cells')
    check('reg: no _error / _failures', '_error' not in r and '_failures' not in r)

    yc = (X[:, 0] > 0).astype(int)
    acc = lambda a, b: float((np.asarray(a) == np.asarray(b)).mean())
    facc = lambda: RandomForestClassifier(n_estimators=15, random_state=0)
    rc = _run_estimator_grid('neighbor_residual', 'euclidean',
                             Xs[tr], yc[tr], Xs[tr], Xs[va], yc[va],
                             facc, acc, n, is_clf=True)
    cells_c = [k for k in rc if not k.startswith('_')]
    check('clf: 12 grid cells (weighting only, smoothing skipped)',
          len(cells_c) == 12, f'{len(cells_c)} cells')
    check('clf: _smoothing_skipped_clf marker present',
          rc.get('_smoothing_skipped_clf') is True)
    check('clf: kNN estimator ran (no _error from missing is_clf branch)',
          '_error' not in rc)


def main():
    print("=" * 60)
    print("FAST SMOKE: n08 / n11 / n12 helpers")
    print("=" * 60)
    test_weight_features_changes_geometry()
    test_inject_noise()
    test_make_estimator_is_clf_routing()
    test_library_api_contract()
    test_run_condition_populates(is_clf=False)
    test_run_condition_populates(is_clf=True)
    test_n11_run_estimator_grid()
    print("=" * 60)
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAILURES: {FAILS}")
        sys.exit(1)
    print("RESULT: ALL GREEN")


if __name__ == '__main__':
    main()
