"""Unit tests for kirby.uq.conformal (T-UQ.1).

Acceptance criterion: the MAPIE wrapper produces near-nominal coverage on a
held-out set for both regression and classification, across all three
calibration modes (cross-conformal, split, prefit), and forwards `groups` to
group-aware CV splitters.

Run: pytest tests/test_conformal.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

mapie = pytest.importorskip("mapie")  # skip cleanly if the extra isn't installed

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold

from kirby.uq.conformal import (
    ConformalRegressor, ConformalClassifier,
    RegressionInterval, ClassificationSet,
    regression_coverage, mean_interval_width,
    classification_coverage, mean_set_size,
)

ALPHA = 0.1
TARGET = 1.0 - ALPHA
# Conformal guarantees are marginal and finite-sample noisy; allow a band.
COVERAGE_TOL = 0.08


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

def _make_regression(n=900, d=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    w = rng.randn(d)
    y = X @ w + rng.randn(n) * 0.5
    return X, y


def _make_classification(n=900, d=8, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    centers = rng.randn(n_classes, d) * 2.0
    logits = X @ centers.T + rng.randn(n, n_classes) * 0.5
    y = logits.argmax(axis=1)
    return X, y


def _three_way_split(n, seed=0):
    """train / calib / test index split (40/30/30)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    a, b = int(0.4 * n), int(0.7 * n)
    return idx[:a], idx[a:b], idx[b:]


# --------------------------------------------------------------------------
# Plain-numpy metric functions
# --------------------------------------------------------------------------

def test_regression_metrics():
    y = np.array([1.0, 2, 3, 4, 5])
    lo = np.array([0.0, 1, 2, 5, 5])
    hi = np.array([2.0, 3, 4, 6, 6])
    assert regression_coverage(y, lo, hi) == pytest.approx(0.8)
    assert mean_interval_width(lo, hi) == pytest.approx(1.6)


def test_classification_metrics():
    sets = np.array([[True, False], [True, True], [False, False]])
    classes = np.array([0, 1])
    y = np.array([0, 1, 0])
    assert classification_coverage(y, sets, classes) == pytest.approx(2 / 3)
    assert mean_set_size(sets) == pytest.approx(1.0)


def test_metric_shape_guards():
    with pytest.raises(ValueError):
        regression_coverage([1, 2], [0], [3, 4])
    with pytest.raises(ValueError):
        classification_coverage([0, 1], np.zeros((2, 3), bool), [0, 1])


def test_alpha_validation():
    with pytest.raises(ValueError):
        ConformalRegressor(RandomForestRegressor(), alpha=0.0)
    with pytest.raises(ValueError):
        ConformalClassifier(RandomForestClassifier(), alpha=1.0)


# --------------------------------------------------------------------------
# Regression — three calibration modes
# --------------------------------------------------------------------------

def test_regression_cross_conformal_coverage():
    X, y = _make_regression()
    tr, _, te = _three_way_split(len(y))
    # cross-conformal uses all of (X[tr], y[tr]) for fit + calibration
    cr = ConformalRegressor(RandomForestRegressor(n_estimators=60,
                                                  random_state=0),
                            alpha=ALPHA, cv=5)
    cr.fit(X[tr], y[tr])
    iv = cr.predict(X[te])
    assert isinstance(iv, RegressionInterval)
    assert iv.point.shape == iv.lower.shape == iv.upper.shape == (len(te),)
    assert np.all(iv.upper >= iv.lower)
    cov = regression_coverage(y[te], iv.lower, iv.upper)
    assert abs(cov - TARGET) < COVERAGE_TOL, f"coverage {cov:.3f}"
    sc = cr.score(X[te], y[te])
    assert sc['target_coverage'] == pytest.approx(TARGET)
    assert sc['mean_interval_width'] > 0


def test_regression_split_coverage():
    X, y = _make_regression()
    tr, ca, te = _three_way_split(len(y))
    cr = ConformalRegressor(RandomForestRegressor(n_estimators=60,
                                                  random_state=0),
                            alpha=ALPHA)
    cr.fit_split(X[tr], y[tr], X[ca], y[ca])
    iv = cr.predict(X[te])
    cov = regression_coverage(y[te], iv.lower, iv.upper)
    assert abs(cov - TARGET) < COVERAGE_TOL, f"coverage {cov:.3f}"


def test_regression_prefit_coverage():
    X, y = _make_regression()
    tr, ca, te = _three_way_split(len(y))
    est = RandomForestRegressor(n_estimators=60, random_state=0)
    est.fit(X[tr], y[tr])  # estimator fitted by the caller
    cr = ConformalRegressor(est, alpha=ALPHA)
    cr.fit_prefit(X[ca], y[ca])
    iv = cr.predict(X[te])
    cov = regression_coverage(y[te], iv.lower, iv.upper)
    assert abs(cov - TARGET) < COVERAGE_TOL, f"coverage {cov:.3f}"


def test_regression_predict_before_fit_raises():
    cr = ConformalRegressor(RandomForestRegressor(), alpha=ALPHA)
    with pytest.raises(RuntimeError):
        cr.predict(np.zeros((3, 8)))


# --------------------------------------------------------------------------
# Regression — group-aware CV splitter
# --------------------------------------------------------------------------

def test_regression_cross_conformal_with_groups():
    """A GroupKFold splitter passed as `cv` must work, with `groups`
    forwarded through fit — no group leakage error from MAPIE."""
    X, y = _make_regression(n=600)
    tr, _, te = _three_way_split(len(y))
    rng = np.random.RandomState(1)
    groups_all = rng.randint(0, 40, size=len(y))  # 40 synthetic groups
    cr = ConformalRegressor(RandomForestRegressor(n_estimators=40,
                                                  random_state=0),
                            alpha=ALPHA, cv=GroupKFold(n_splits=5))
    cr.fit(X[tr], y[tr], groups=groups_all[tr])
    iv = cr.predict(X[te])
    cov = regression_coverage(y[te], iv.lower, iv.upper)
    assert abs(cov - TARGET) < COVERAGE_TOL + 0.04, f"coverage {cov:.3f}"


# --------------------------------------------------------------------------
# Classification — three calibration modes
# --------------------------------------------------------------------------

def test_classification_cross_conformal_coverage():
    X, y = _make_classification()
    tr, _, te = _three_way_split(len(y))
    cc = ConformalClassifier(RandomForestClassifier(n_estimators=60,
                                                    random_state=0),
                             alpha=ALPHA, cv=5)
    cc.fit(X[tr], y[tr])
    cs = cc.predict(X[te])
    assert isinstance(cs, ClassificationSet)
    assert cs.sets.shape == (len(te), len(cc.classes_))
    assert cs.sets.dtype == bool
    # classes captured in sorted order, matching predict_set columns
    assert list(cs.classes) == sorted(np.unique(y).tolist())
    cov = classification_coverage(y[te], cs.sets, cs.classes)
    assert abs(cov - TARGET) < COVERAGE_TOL, f"coverage {cov:.3f}"
    sc = cc.score(X[te], y[te])
    assert 1.0 <= sc['mean_set_size'] <= len(cc.classes_)


def test_classification_split_coverage():
    X, y = _make_classification()
    tr, ca, te = _three_way_split(len(y))
    cc = ConformalClassifier(RandomForestClassifier(n_estimators=60,
                                                    random_state=0),
                             alpha=ALPHA)
    cc.fit_split(X[tr], y[tr], X[ca], y[ca])
    cs = cc.predict(X[te])
    cov = classification_coverage(y[te], cs.sets, cs.classes)
    assert abs(cov - TARGET) < COVERAGE_TOL, f"coverage {cov:.3f}"


def test_classification_prefit_coverage():
    X, y = _make_classification()
    tr, ca, te = _three_way_split(len(y))
    est = RandomForestClassifier(n_estimators=60, random_state=0)
    est.fit(X[tr], y[tr])
    cc = ConformalClassifier(est, alpha=ALPHA)
    cc.fit_prefit(X[ca], y[ca])
    cs = cc.predict(X[te])
    cov = classification_coverage(y[te], cs.sets, cs.classes)
    assert abs(cov - TARGET) < COVERAGE_TOL, f"coverage {cov:.3f}"


def test_classification_with_groups():
    X, y = _make_classification(n=600)
    tr, _, te = _three_way_split(len(y))
    rng = np.random.RandomState(2)
    groups_all = rng.randint(0, 40, size=len(y))
    cc = ConformalClassifier(RandomForestClassifier(n_estimators=40,
                                                    random_state=0),
                             alpha=ALPHA, cv=GroupKFold(n_splits=5))
    cc.fit(X[tr], y[tr], groups=groups_all[tr])
    cs = cc.predict(X[te])
    cov = classification_coverage(y[te], cs.sets, cs.classes)
    assert abs(cov - TARGET) < COVERAGE_TOL + 0.04, f"coverage {cov:.3f}"
