"""Conformal prediction wrapper (T-UQ.1).

A thin wrapper over MAPIE that gives any sklearn-compatible estimator
calibrated, distribution-free uncertainty: prediction intervals for
regression, prediction sets for classification. It exists so the
cross-species PT-* experiments (and any downstream-analysis step) can attach
coverage-guaranteed intervals without each script re-deriving the MAPIE call.

Why a wrapper rather than calling MAPIE directly:
  - it honors KIRBy's group-aware CV splitters — pass a `GroupKFold` (or any
    splitter) as `cv` and the `groups` array straight through to `fit`, so
    conformal calibration respects scaffold / cluster groups exactly the way
    the rest of the pipeline does;
  - it pins one stable call shape against the MAPIE 1.x API, so a future
    MAPIE bump only touches this file;
  - coverage / width / set-size metrics are implemented in plain numpy here
    (not imported from MAPIE), so they are version-proof.

Three calibration modes, all reached through the same classes:
  - `fit(X, y, groups=)` — cross-conformal. Hand it an *unfitted* estimator;
    MAPIE cross-fits over the `cv` splitter for calibration, no data wasted.
    With a group-aware splitter, `groups` is forwarded so calibration folds
    respect the groups.
  - `fit_split(X_train, y_train, X_calib, y_calib)` — split-conformal. The
    estimator is fitted on the train half and calibrated on the explicit
    calibration half. This is the design-doc §4 "fold-out calibration"
    policy (n>=500: one of five folds is the permanent calibration set).
  - `fit_prefit(X_calib, y_calib)` — split-conformal with an *already-fitted*
    estimator; (X_calib, y_calib) is a pure calibration holdout. This is the
    §4 "within-fold calibration" policy (150<=n<500: each training fold is
    split 80/20 train/calib).

MAPIE is an optional dependency (the `uncertainty` extra). It is imported
lazily so importing this module never fails; the informative ImportError
fires only when you actually construct a wrapper.

Usage:
    from kirby.uq.conformal import ConformalRegressor, regression_coverage
    from sklearn.model_selection import GroupKFold

    cr = ConformalRegressor(RandomForestRegressor(), alpha=0.1,
                            cv=GroupKFold(n_splits=5))
    cr.fit(X_train, y_train, groups=scaffold_groups)
    iv = cr.predict(X_test)            # RegressionInterval(point, lower, upper)
    cov = regression_coverage(y_test, iv.lower, iv.upper)   # ~0.90

Reference:
    Vovk et al. — conformal prediction.
    MAPIE 1.x — https://github.com/scikit-learn-contrib/MAPIE
"""

from collections import namedtuple

import numpy as np


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------

# point: (n,) point predictions; lower/upper: (n,) interval bounds.
RegressionInterval = namedtuple('RegressionInterval', ['point', 'lower', 'upper'])

# pred: (n,) point predictions; sets: (n, n_classes) bool membership matrix;
# classes: (n_classes,) the class labels indexing `sets`' columns.
ClassificationSet = namedtuple('ClassificationSet', ['pred', 'sets', 'classes'])


# --------------------------------------------------------------------------
# Metrics (plain numpy — independent of the installed MAPIE version)
# --------------------------------------------------------------------------

def regression_coverage(y_true, lower, upper):
    """Fraction of true values that fall inside [lower, upper]."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    if not (len(y_true) == len(lower) == len(upper)):
        raise ValueError("y_true, lower, upper must have the same length")
    if len(y_true) == 0:
        return float('nan')
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mean_interval_width(lower, upper):
    """Mean width of the prediction intervals."""
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    if len(lower) != len(upper):
        raise ValueError("lower and upper must have the same length")
    if len(lower) == 0:
        return float('nan')
    return float(np.mean(upper - lower))


def classification_coverage(y_true, prediction_sets, classes):
    """Fraction of samples whose true label is in the predicted set.

    Args:
        y_true: (n,) true labels.
        prediction_sets: (n, n_classes) bool membership matrix.
        classes: (n_classes,) labels indexing the columns of prediction_sets.
    """
    y_true = np.asarray(y_true).ravel()
    prediction_sets = np.asarray(prediction_sets, dtype=bool)
    classes = np.asarray(classes)
    if prediction_sets.shape != (len(y_true), len(classes)):
        raise ValueError(
            f"prediction_sets shape {prediction_sets.shape} != "
            f"(n_samples={len(y_true)}, n_classes={len(classes)})")
    if len(y_true) == 0:
        return float('nan')
    class_to_col = {c: i for i, c in enumerate(classes)}
    hits = 0
    for i, yt in enumerate(y_true):
        col = class_to_col.get(yt)
        if col is not None and prediction_sets[i, col]:
            hits += 1
    return float(hits / len(y_true))


def mean_set_size(prediction_sets):
    """Mean number of classes per predicted set (efficiency metric)."""
    prediction_sets = np.asarray(prediction_sets, dtype=bool)
    if prediction_sets.size == 0:
        return float('nan')
    return float(np.mean(prediction_sets.sum(axis=1)))


# --------------------------------------------------------------------------
# Lazy MAPIE import
# --------------------------------------------------------------------------

def _require_mapie():
    """Import MAPIE 1.x estimators, with an install hint on failure."""
    try:
        from mapie.regression import (
            CrossConformalRegressor, SplitConformalRegressor)
        from mapie.classification import (
            CrossConformalClassifier, SplitConformalClassifier)
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "kirby.uq.conformal requires MAPIE >= 1.0. Install the "
            "uncertainty extra (`pip install -e \".[uncertainty]\"`) or "
            "`pip install 'mapie>=1.0'`. See SETUP.md."
        ) from e
    return (CrossConformalRegressor, SplitConformalRegressor,
            CrossConformalClassifier, SplitConformalClassifier)


# --------------------------------------------------------------------------
# Wrappers
# --------------------------------------------------------------------------

class ConformalRegressor:
    """Calibrated prediction intervals for a regression estimator.

    Args:
        estimator: an sklearn-compatible regressor. Unfitted for `fit` /
            `fit_split`; already fitted for `fit_prefit`.
        alpha: miscoverage rate. alpha=0.1 -> 90% target coverage.
        method: cross-conformal method ('plus', 'base', 'minmax'). Used by
            `fit` only; `fit_split` / `fit_prefit` ignore it.
        cv: int n_folds or an sklearn CV splitter (e.g. GroupKFold — pass the
            matching `groups` to `fit`). Used by `fit` only.
        conformity_score: MAPIE regression conformity score ('absolute',
            'gamma', ...).
        n_jobs: forwarded to MAPIE.
        random_state: forwarded to MAPIE (cross-conformal only).
    """

    def __init__(self, estimator, alpha=0.1, method='plus', cv=5,
                 conformity_score='absolute', n_jobs=None, random_state=42):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.estimator = estimator
        self.alpha = alpha
        self.method = method
        self.cv = cv
        self.conformity_score = conformity_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.mapie_ = None

    @property
    def confidence_level(self):
        return 1.0 - self.alpha

    def fit(self, X, y, groups=None):
        """Cross-conformal fit: MAPIE cross-fits the unfitted estimator over
        `self.cv` for calibration. Pass `groups` with a group-aware splitter."""
        CrossConformalRegressor = _require_mapie()[0]
        self.mapie_ = CrossConformalRegressor(
            estimator=self.estimator, confidence_level=self.confidence_level,
            conformity_score=self.conformity_score, method=self.method,
            cv=self.cv, n_jobs=self.n_jobs, random_state=self.random_state)
        self.mapie_.fit_conformalize(
            np.asarray(X), np.asarray(y).ravel(), groups=groups)
        return self

    def fit_split(self, X_train, y_train, X_calib, y_calib):
        """Split-conformal fit: the estimator is fitted on (X_train, y_train)
        and calibrated on the explicit holdout (X_calib, y_calib)."""
        SplitConformalRegressor = _require_mapie()[1]
        self.mapie_ = SplitConformalRegressor(
            estimator=self.estimator, confidence_level=self.confidence_level,
            conformity_score=self.conformity_score, prefit=False,
            n_jobs=self.n_jobs)
        self.mapie_.fit(np.asarray(X_train), np.asarray(y_train).ravel())
        self.mapie_.conformalize(np.asarray(X_calib),
                                 np.asarray(y_calib).ravel())
        return self

    def fit_prefit(self, X_calib, y_calib):
        """Split-conformal calibration of an ALREADY-FITTED estimator on the
        calibration holdout (X_calib, y_calib)."""
        SplitConformalRegressor = _require_mapie()[1]
        self.mapie_ = SplitConformalRegressor(
            estimator=self.estimator, confidence_level=self.confidence_level,
            conformity_score=self.conformity_score, prefit=True,
            n_jobs=self.n_jobs)
        self.mapie_.conformalize(np.asarray(X_calib),
                                 np.asarray(y_calib).ravel())
        return self

    def predict(self, X):
        """Return a RegressionInterval(point, lower, upper) at `self.alpha`."""
        if self.mapie_ is None:
            raise RuntimeError("ConformalRegressor.predict called before a fit")
        point, intervals = self.mapie_.predict_interval(np.asarray(X))
        # intervals: (n, 2, n_confidence_levels); single level -> column 0.
        intervals = np.asarray(intervals)
        lower = intervals[:, 0, 0]
        upper = intervals[:, 1, 0]
        return RegressionInterval(point=np.asarray(point).ravel(),
                                  lower=lower, upper=upper)

    def score(self, X, y_true):
        """Empirical coverage + mean interval width on (X, y_true)."""
        iv = self.predict(X)
        return {
            'coverage': regression_coverage(y_true, iv.lower, iv.upper),
            'mean_interval_width': mean_interval_width(iv.lower, iv.upper),
            'target_coverage': self.confidence_level,
        }


class ConformalClassifier:
    """Calibrated prediction sets for a classification estimator.

    Args:
        estimator: an sklearn-compatible classifier (must implement
            predict_proba). Unfitted for `fit` / `fit_split`; already fitted
            for `fit_prefit`.
        alpha: miscoverage rate. alpha=0.1 -> 90% target coverage.
        conformity_score: MAPIE classification conformity score ('lac',
            'aps', 'raps', 'top_k').
        cv: int n_folds or an sklearn CV splitter. Used by `fit` only.
        n_jobs: forwarded to MAPIE.
        random_state: forwarded to MAPIE (cross-conformal only).
    """

    def __init__(self, estimator, alpha=0.1, conformity_score='lac', cv=5,
                 n_jobs=None, random_state=42):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.estimator = estimator
        self.alpha = alpha
        self.conformity_score = conformity_score
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.mapie_ = None
        self.classes_ = None

    @property
    def confidence_level(self):
        return 1.0 - self.alpha

    def _capture_classes(self, y):
        """Resolve the class labels indexing predict_set's columns. MAPIE
        encodes classes with a sorted LabelEncoder, so np.unique(y) matches
        the column order; prefer MAPIE's own attribute when present."""
        cls = getattr(self.mapie_, 'classes_', None)
        if cls is None:
            est = getattr(self.mapie_, 'estimator_', None) or self.estimator
            cls = getattr(est, 'classes_', None)
        if cls is None:
            cls = np.unique(np.asarray(y).ravel())
        self.classes_ = np.asarray(cls)

    def fit(self, X, y, groups=None):
        """Cross-conformal fit. Pass `groups` with a group-aware splitter."""
        CrossConformalClassifier = _require_mapie()[2]
        y = np.asarray(y).ravel()
        self.mapie_ = CrossConformalClassifier(
            estimator=self.estimator, confidence_level=self.confidence_level,
            conformity_score=self.conformity_score, cv=self.cv,
            n_jobs=self.n_jobs, random_state=self.random_state)
        self.mapie_.fit_conformalize(np.asarray(X), y, groups=groups)
        self._capture_classes(y)
        return self

    def fit_split(self, X_train, y_train, X_calib, y_calib):
        """Split-conformal fit: estimator fitted on the train half, calibrated
        on the explicit (X_calib, y_calib) holdout."""
        SplitConformalClassifier = _require_mapie()[3]
        y_train = np.asarray(y_train).ravel()
        self.mapie_ = SplitConformalClassifier(
            estimator=self.estimator, confidence_level=self.confidence_level,
            conformity_score=self.conformity_score, prefit=False,
            n_jobs=self.n_jobs, random_state=self.random_state)
        self.mapie_.fit(np.asarray(X_train), y_train)
        self.mapie_.conformalize(np.asarray(X_calib),
                                 np.asarray(y_calib).ravel())
        self._capture_classes(y_train)
        return self

    def fit_prefit(self, X_calib, y_calib):
        """Split-conformal calibration of an ALREADY-FITTED classifier on the
        calibration holdout (X_calib, y_calib)."""
        SplitConformalClassifier = _require_mapie()[3]
        y_calib = np.asarray(y_calib).ravel()
        self.mapie_ = SplitConformalClassifier(
            estimator=self.estimator, confidence_level=self.confidence_level,
            conformity_score=self.conformity_score, prefit=True,
            n_jobs=self.n_jobs, random_state=self.random_state)
        self.mapie_.conformalize(np.asarray(X_calib), y_calib)
        self._capture_classes(y_calib)
        return self

    def predict(self, X):
        """Return a ClassificationSet(pred, sets, classes) at `self.alpha`.

        `sets` is an (n, n_classes) bool matrix; `classes` indexes its columns.
        """
        if self.mapie_ is None:
            raise RuntimeError("ConformalClassifier.predict called before a fit")
        pred, sets = self.mapie_.predict_set(np.asarray(X))
        # sets: (n, n_classes, n_confidence_levels); single level -> column 0.
        sets = np.asarray(sets, dtype=bool)[:, :, 0]
        return ClassificationSet(pred=np.asarray(pred).ravel(), sets=sets,
                                 classes=np.asarray(self.classes_))

    def score(self, X, y_true):
        """Empirical coverage + mean set size on (X, y_true)."""
        cs = self.predict(X)
        return {
            'coverage': classification_coverage(y_true, cs.sets, cs.classes),
            'mean_set_size': mean_set_size(cs.sets),
            'target_coverage': self.confidence_level,
        }
