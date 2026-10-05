"""One feature scaler for the whole repo: 0/1 features untouched, the rest standardised.

Adelaide's rule, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.3 item 17): 0/1
features are never scaled; continuous features are standardised; everywhere,
including the 0/1 features inside a hybrid. Before this, four places treated 0/1
features four ways (dirty-run bug 3): find_best_hybrid._evaluate_single left a
whole matrix alone when every value in it was 0 or 1, while the hybrid scorer,
the nested-CV held-out score, the tuning pipeline and every experiment that
scaled for itself standardised every feature. Measured on herg_ki, rmse: svm on
ecfp4 0.6841 standardised against 0.6122 left as 0/1; knn on atom_pair 0.6452
against 0.6193.

A feature is 0/1 when every one of its TRAINING values is 0 or 1 (missing values
ignored). The decision is made per feature at fit time and then applied as-is
to validation rows, so a validation value of 2 in a 0/1 feature passes through
unscaled rather than changing the rule.

FeatureScaler is sklearn's StandardScaler with the 0/1 features' centre set to 0
and their spread to 1 after fitting, so a matrix with no 0/1 feature is scaled
bit for bit as StandardScaler scales it, and it drops into a Pipeline or a
clone() exactly where StandardScaler did.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def binary_feature_mask(X):
    """True for each feature whose every non-missing value is 0 or 1."""
    A = np.asarray(X)
    if A.ndim != 2:
        raise ValueError(f'expected a 2-D matrix, got shape {A.shape}')
    if A.shape[0] == 0:
        return np.zeros(A.shape[1], dtype=bool)
    if A.dtype.kind in 'biu':
        return np.all((A == 0) | (A == 1), axis=0)
    return np.all(np.isnan(A) | (A == 0) | (A == 1), axis=0)


class FeatureScaler(StandardScaler):
    """StandardScaler that leaves 0/1 features as they are.

    After fit, `binary_mask_` holds the 0/1 features; their `mean_` is 0 and
    their `scale_` is 1, so transform and inverse_transform pass them through.
    """

    def fit(self, X, y=None, sample_weight=None):
        self.binary_mask_ = None
        super().fit(X, y, sample_weight=sample_weight)
        self._mark_binary(X)
        return self

    def partial_fit(self, X, y=None, sample_weight=None):
        # StandardScaler.fit calls partial_fit after a reset, so this must work
        # inside fit. Called on its own, on a scaler already fitted, it would
        # update the 0/1 features' zeroed centre as if it were real: refused.
        if getattr(self, 'binary_mask_', None) is not None:
            raise NotImplementedError(
                'FeatureScaler cannot be updated batch by batch; fit it once '
                'on the training rows.')
        return super().partial_fit(X, y, sample_weight=sample_weight)

    def _mark_binary(self, X):
        mask = binary_feature_mask(X)
        self.binary_mask_ = mask
        if self.with_mean and self.mean_ is not None:
            self.mean_ = np.array(self.mean_, dtype=np.float64, copy=True)
            self.mean_[mask] = 0.0
        if self.with_std and self.scale_ is not None:
            self.scale_ = np.array(self.scale_, dtype=np.float64, copy=True)
            self.scale_[mask] = 1.0
