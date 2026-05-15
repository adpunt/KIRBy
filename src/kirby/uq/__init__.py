"""Uncertainty-quantification utilities for KIRBy.

`conformal` — MAPIE-backed conformal prediction wrapper (T-UQ.1): calibrated
prediction intervals (regression) and prediction sets (classification) for any
sklearn-compatible estimator, honoring KIRBy's group-aware CV splitters.

`conformal_chem` — chemistry-aware variant (T-UQ.2): Mondrian split-conformal
intervals binned by distance to the nearest training compound, so intervals
widen out-of-domain under scaffold splits.
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
from kirby.uq.conformal_chem import ChemBinnedConformalRegressor

__all__ = [
    'ConformalRegressor',
    'ConformalClassifier',
    'ChemBinnedConformalRegressor',
    'RegressionInterval',
    'ClassificationSet',
    'regression_coverage',
    'mean_interval_width',
    'classification_coverage',
    'mean_set_size',
]
