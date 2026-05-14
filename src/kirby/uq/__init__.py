"""Uncertainty-quantification utilities for KIRBy.

`conformal` — MAPIE-backed conformal prediction wrapper (T-UQ.1): calibrated
prediction intervals (regression) and prediction sets (classification) for any
sklearn-compatible estimator, honoring KIRBy's group-aware CV splitters.
"""

from kirby.uq.conformal import (
    ConformalRegressor,
    ConformalClassifier,
    RegressionInterval,
    ClassificationSet,
    regression_coverage,
    mean_interval_width,
    classification_coverage,
    mean_set_size,
)

__all__ = [
    'ConformalRegressor',
    'ConformalClassifier',
    'RegressionInterval',
    'ClassificationSet',
    'regression_coverage',
    'mean_interval_width',
    'classification_coverage',
    'mean_set_size',
]
