#!/usr/bin/env python3
"""
Noise Estimation Methods for QSAR
==================================

Estimate the MAGNITUDE of label noise for each sample.
Outputs continuous noise scores, not binary flags.

Use cases:
1. Sample weighting: weight = 1 / (1 + noise_score)
2. Label correction: y_corrected = y + alpha * (target - y) * noise_score
3. Uncertainty quantification: noise_score as aleatoric uncertainty proxy

All methods return:
    noise_scores: np.ndarray of shape (n_samples,), higher = more noise
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BASE CLASS
# ============================================================================

class NoiseEstimator(ABC):
    """Base class for noise estimation methods."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.noise_scores_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Sklearn-style fit interface. Calls estimate() and stores results."""
        self.noise_scores_ = self.estimate(X, y)
        self._fitted_X = X
        self._fitted_y = y
        return self

    @abstractmethod
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Estimate noise magnitude for each sample.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            noise_scores: (n_samples,) - higher values = more noise
        """
        pass

    def get_weights(self, X: np.ndarray = None, y: np.ndarray = None,
                    strength: float = 1.0) -> np.ndarray:
        """
        Convert noise scores to sample weights.

        If X/y are None, uses scores from prior fit() or estimate() call.

        Args:
            strength: How aggressively to downweight noisy samples

        Returns:
            weights: (n_samples,) - lower for noisy samples
        """
        if X is not None and y is not None:
            scores = self.estimate(X, y)
        elif self.noise_scores_ is not None:
            scores = self.noise_scores_
        else:
            raise ValueError("No scores available. Call fit() or pass X, y.")
        weights = 1 / (1 + strength * scores)
        return weights / weights.mean()  # Normalize to mean 1
    
    def get_corrections(self, X: np.ndarray, y: np.ndarray,
                        targets: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Get label corrections proportional to noise scores.
        
        Args:
            targets: Where to correct toward (e.g., neighbor means)
            strength: Maximum correction factor
        
        Returns:
            y_corrected: (n_samples,)
        """
        scores = self.estimate(X, y)
        # Normalize scores to [0, 1]
        scores_norm = scores / (scores.max() + 1e-10)
        # Correct proportionally
        correction = strength * scores_norm * (targets - y)
        return y + correction

    def _oof_apply(self, X: np.ndarray, y: np.ndarray, fold_fn,
                   n_folds: int = 5, groups=None) -> Dict[str, np.ndarray]:
        """Out-of-fold application — the honest replacement for fit-then-predict
        on the SAME rows (which grades a model on data it already memorised).

        The rows are split into folds. For each fold, ``fold_fn(X_tr, y_tr,
        X_ev)`` is trained ONLY on the other folds and returns a dict of
        per-EVAL-row arrays (e.g. predictions, uncertainties). Each array is
        scattered back to the held-out rows' original positions, so every row
        is scored by a model that never saw it.

        Splitting is scaffold-GROUP-disjoint (``GroupKFold``) when ``groups``
        is supplied — no scaffold appears on both sides of a fold — and a
        DETERMINISTIC ``KFold(shuffle=False)`` otherwise. Neither path shuffles
        randomly, honouring the no-random-splits rule.

        Returns ``{name: np.ndarray of shape (n_samples,)}`` keyed by whatever
        ``fold_fn`` returned.
        """
        n = len(y)
        X = np.asarray(X)
        y = np.asarray(y)

        use_groups = False
        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) == n and len(np.unique(groups)) >= 2:
                use_groups = True

        if use_groups:
            # Clamp to >=2 (GroupKFold requires it) exactly as the KFold path
            # does, so n_folds=1 doesn't crash one branch while the other runs.
            n_splits = max(2, min(n_folds, len(np.unique(groups))))
            splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
        else:
            n_splits = max(2, min(n_folds, n))
            splits = list(KFold(n_splits=n_splits, shuffle=False).split(X))

        out: Dict[str, np.ndarray] = {}
        for train_idx, eval_idx in splits:
            if len(train_idx) == 0 or len(eval_idx) == 0:
                continue
            res = fold_fn(X[train_idx], y[train_idx], X[eval_idx])
            for key, val in res.items():
                arr = out.get(key)
                if arr is None:
                    # NaN sentinel, not 0.0: an uncovered row must never
                    # masquerade as the cleanest sample. KFold/GroupKFold give
                    # full disjoint coverage, so no NaN should survive normally;
                    # any that does is filled with the column mean (neutral).
                    arr = np.full(n, np.nan, dtype=float)
                    out[key] = arr
                arr[eval_idx] = np.asarray(val, dtype=float).ravel()
        for key, arr in out.items():
            if np.isnan(arr).any():
                arr[np.isnan(arr)] = np.nanmean(arr) if not np.isnan(arr).all() else 0.0
        return out


# ============================================================================
# NEIGHBOR-BASED ESTIMATORS
# ============================================================================

def _knn_indices(X, k, metric, exclude_groups=None, return_distance=False):
    """Each row's k nearest OTHER rows, optionally skipping rows that share its
    ``exclude_groups`` label.

    Every neighbour-based estimator below scores a row against the labels of its
    k nearest rows. When some rows are repeat measurements of the SAME thing —
    BindingDB holds the same (drug, target) pair measured in different labs —
    those repeats are identical in feature space, so they are the row's own
    nearest neighbours. The row's score is then computed from the very labels it
    is being compared against, and part of any correlation with replicate spread
    is arithmetic rather than signal.

    Pass one label per row in ``exclude_groups`` — the (drug, target) pair on
    BindingDB — and no row scores itself from its own repeats. Leave it None and
    only the row itself is skipped, which is what indices[:, 1:] did before.

    Returns ``indices`` of shape (n, k_eff), or ``(distances, indices)`` when
    ``return_distance``. k_eff falls below k only when a dataset's largest group
    leaves fewer than k rows outside it.
    """
    X = np.asarray(X)
    n = len(X)
    if exclude_groups is None:
        n_query = int(min(k + 1, n))
        nn = NearestNeighbors(n_neighbors=n_query, metric=metric)
        nn.fit(X)
        d, idx = nn.kneighbors(X)
        d, idx = d[:, 1:k + 1], idx[:, 1:k + 1]
        return (d, idx) if return_distance else idx

    g = np.asarray(exclude_groups)
    if len(g) != n:
        raise ValueError(
            f"exclude_groups has {len(g)} entries but X has {n} rows.")
    max_g = int(np.unique(g, return_counts=True)[1].max())
    k_eff = int(min(k, n - max_g))
    if k_eff < 1:
        raise ValueError(
            f"exclude_groups leaves no neighbours: the largest group holds "
            f"{max_g} of {n} rows.")
    # Among the n_query nearest rows at most max_g share this row's group
    # (itself included), so at least k_eff survive the mask for every row.
    n_query = int(min(n, k_eff + max_g + 1))
    nn = NearestNeighbors(n_neighbors=n_query, metric=metric)
    nn.fit(X)
    d, idx = nn.kneighbors(X)
    keep = g[idx] != g[:, None]          # drops self and every same-group row

    out_i = np.empty((n, k_eff), dtype=int)
    out_d = np.empty((n, k_eff), dtype=float)
    for i in range(n):
        sel = np.flatnonzero(keep[i])[:k_eff]
        out_i[i] = idx[i, sel]
        out_d[i] = d[i, sel]
    return (out_d, out_i) if return_distance else out_i


class NeighborResidualEstimator(NoiseEstimator):
    """
    Estimate noise as deviation from k-NN consensus.

    Regression: noise_score = |y - neighbor_mean| / scale
    Classification (is_clf=True): noise_score = fraction of neighbors with a
        different class than y. The regression formulation (|y - 0.4|) is
        meaningless for integer class labels — averaging "no"s and "yes"es
        and asking distance-from-fractional-average doesn't measure
        anything about how mislabeled a compound looks.
    """

    def __init__(self, k: int = 10, metric: str = 'cosine',
                 normalize: bool = True, is_clf: bool = False,
                 exclude_groups=None, random_state: int = 42):
        super().__init__(random_state)
        self.k = k
        self.metric = metric
        self.normalize = normalize
        self.is_clf = is_clf
        # One label per row; a row never takes a neighbour sharing it. See
        # _knn_indices for why repeat measurements need this.
        self.exclude_groups = exclude_groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = min(self.k, len(y) - 1)

        neighbor_indices = _knn_indices(X, k, self.metric,
                                        exclude_groups=self.exclude_groups)
        neighbor_labels = y[neighbor_indices]

        if self.is_clf:
            # Classification: class-disagreement score = fraction of
            # neighbors whose class differs from this sample's class.
            disagreement = (neighbor_labels != y[:, None]).mean(axis=1)
            self.noise_scores_ = disagreement
            # neighbor_means_ for classification = majority-vote class
            # (used by downstream label-correction code that expects an
            # array shaped like y)
            from scipy.stats import mode as _scipy_mode
            self.neighbor_means_ = _scipy_mode(
                neighbor_labels, axis=1, keepdims=False).mode
            return disagreement

        # Regression: residual from neighbor mean
        neighbor_mean = np.mean(neighbor_labels, axis=1)
        residuals = np.abs(y - neighbor_mean)
        if self.normalize:
            residuals = residuals / (np.std(y) + 1e-10)
        self.noise_scores_ = residuals
        self.neighbor_means_ = neighbor_mean
        return residuals
    
    def get_neighbor_means(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return neighbor means (useful for label correction)."""
        if self.neighbor_means_ is None:
            self.estimate(X, y)
        return self.neighbor_means_


class LocalVarianceRatioEstimator(NoiseEstimator):
    """
    Estimate noise relative to local label variance.

    Regression: noise_score = |y - neighbor_mean| / neighbor_std
        High score means: you deviate from neighbors more than neighbors
        deviate from each other.
    Classification (is_clf=True): noise_score = disagreement_with_me /
        (1 - max_class_purity). Numerator is the fraction of neighbors
        with a different class than y; denominator is one minus the
        purity of the dominant class among neighbors. So a sample whose
        class differs from neighbors AND whose neighbors are themselves
        uniform (high purity, low denominator) scores higher than one
        whose neighbors are already heterogeneous.
    """

    def __init__(self, k: int = 10, metric: str = 'cosine',
                 is_clf: bool = False, exclude_groups=None,
                 random_state: int = 42):
        super().__init__(random_state)
        self.k = k
        self.metric = metric
        self.is_clf = is_clf
        self.exclude_groups = exclude_groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = min(self.k, len(y) - 1)

        neighbor_indices = _knn_indices(X, k, self.metric,
                                        exclude_groups=self.exclude_groups)
        neighbor_labels = y[neighbor_indices]
        k = neighbor_indices.shape[1]

        if self.is_clf:
            from scipy.stats import mode as _scipy_mode
            disagreement = (neighbor_labels != y[:, None]).mean(axis=1)
            modes = _scipy_mode(neighbor_labels, axis=1, keepdims=False)
            dominant_class = modes.mode
            # Per-row class purity: fraction of neighbors that share the
            # dominant class within those neighbors.
            purity = (neighbor_labels == dominant_class[:, None]).mean(axis=1)
            local_heterogeneity = 1.0 - purity
            # Floor the denominator at 1/k (the minimum non-zero
            # heterogeneity a k-neighbor sample can have). Without this,
            # disagreement / 1e-10 yields ~1e10 for samples in perfectly-
            # pure neighborhoods, which dominates min-max normalisation
            # downstream and collapses every other score to ~0.
            heterogeneity_floor = max(1.0 / k, 1e-3)
            scores = disagreement / np.maximum(local_heterogeneity, heterogeneity_floor)
            self.noise_scores_ = scores
            self.neighbor_means_ = dominant_class
            self.neighbor_stds_ = local_heterogeneity
            return scores

        neighbor_mean = np.mean(neighbor_labels, axis=1)
        neighbor_std = np.std(neighbor_labels, axis=1)
        deviation = np.abs(y - neighbor_mean)
        # Floor the denominator at a fraction of the global label spread.
        # Without this, deviation / 1e-10 yields ~1e10 for samples in near-
        # constant neighborhoods, which dominates min-max normalisation
        # downstream and collapses every other score to ~0 (mirrors the
        # heterogeneity_floor applied in the classification branch above).
        std_floor = max(0.01 * np.std(y), 1e-3)
        scores = deviation / np.maximum(neighbor_std, std_floor)
        self.noise_scores_ = scores
        self.neighbor_means_ = neighbor_mean
        self.neighbor_stds_ = neighbor_std
        return scores


class DistanceWeightedResidualEstimator(NoiseEstimator):
    """
    Like NeighborResidual, but weight closer neighbors more heavily.

    Regression: noise_score = |y - weighted_neighbor_mean|, weights ∝ 1/distance
    Classification (is_clf=True): noise_score = weighted fraction of
        neighbors with a different class than y, weights ∝ 1/distance.
    """

    def __init__(self, k: int = 10, metric: str = 'cosine',
                 normalize: bool = True, is_clf: bool = False,
                 exclude_groups=None, random_state: int = 42):
        super().__init__(random_state)
        self.k = k
        self.metric = metric
        self.normalize = normalize
        self.is_clf = is_clf
        self.exclude_groups = exclude_groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = min(self.k, len(y) - 1)

        neighbor_distances, neighbor_indices = _knn_indices(
            X, k, self.metric, exclude_groups=self.exclude_groups,
            return_distance=True)
        neighbor_labels = y[neighbor_indices]

        weights = 1 / (neighbor_distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        if self.is_clf:
            from scipy.stats import mode as _scipy_mode
            different = (neighbor_labels != y[:, None]).astype(float)
            scores = np.sum(different * weights, axis=1)
            self.noise_scores_ = scores
            _modes = _scipy_mode(neighbor_labels, axis=1, keepdims=False).mode
            # Set BOTH attributes so downstream code reading either name works.
            self.neighbor_means_ = _modes
            self.weighted_means_ = _modes
            return scores

        weighted_mean = np.sum(neighbor_labels * weights, axis=1)
        residuals = np.abs(y - weighted_mean)

        if self.normalize:
            residuals = residuals / (np.std(y) + 1e-10)

        self.noise_scores_ = residuals
        # Set BOTH so consumers reading either name work in both modes.
        self.neighbor_means_ = weighted_mean
        self.weighted_means_ = weighted_mean
        return residuals


class ActivityCliffAwareEstimator(NoiseEstimator):
    """
    Distinguish noise from activity cliffs.

    Regression key insight:
    - NOISE: Sample deviates from neighbors, but neighbors agree with each other
    - ACTIVITY CLIFF: Sample deviates from neighbors, and neighbors also disagree

    noise_score = deviation * (1 - local_variance_normalized)

    Classification (is_clf=True) analog:
    - NOISE: y differs from neighbors AND neighbors are class-pure
    - CLIFF-LIKE: y differs from neighbors AND neighbors are class-heterogeneous

    noise_score = class_disagreement * class_purity_of_neighbors

    Class-disagreement is the fraction of neighbors with a different class
    than this sample; class-purity is the fraction of neighbors sharing
    their own dominant class. A noisy sample is one that disagrees with
    neighbors who otherwise all agree with each other.
    """

    def __init__(self, k: int = 10, metric: str = 'cosine',
                 is_clf: bool = False, exclude_groups=None,
                 random_state: int = 42):
        super().__init__(random_state)
        self.k = k
        self.metric = metric
        self.is_clf = is_clf
        self.exclude_groups = exclude_groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = min(self.k, len(y) - 1)

        neighbor_indices = _knn_indices(X, k, self.metric,
                                        exclude_groups=self.exclude_groups)
        neighbor_labels = y[neighbor_indices]

        if self.is_clf:
            from scipy.stats import mode as _scipy_mode
            disagreement = (neighbor_labels != y[:, None]).mean(axis=1)
            modes = _scipy_mode(neighbor_labels, axis=1, keepdims=False)
            dominant_class = modes.mode
            purity = (neighbor_labels == dominant_class[:, None]).mean(axis=1)
            # Noise: high disagreement + high neighbor purity (neighbors all
            # agree but this sample disagrees with them)
            scores = disagreement * purity
            self.noise_scores_ = scores
            self.neighbor_means_ = dominant_class
            self.consistency_scores_ = purity
            self.is_likely_cliff_ = (disagreement > 0.5) & (purity < 0.5)
            return scores

        global_std = np.std(y) + 1e-10
        neighbor_mean = np.mean(neighbor_labels, axis=1)
        neighbor_std = np.std(neighbor_labels, axis=1)

        deviation = np.abs(y - neighbor_mean) / global_std
        local_variance_norm = neighbor_std / global_std
        consistency = 1 / (1 + local_variance_norm)

        scores = deviation * consistency
        self.noise_scores_ = scores
        self.neighbor_means_ = neighbor_mean
        self.consistency_scores_ = consistency
        self.is_likely_cliff_ = (deviation > 1.0) & (consistency < 0.5)
        return scores


# ============================================================================
# MODEL-BASED ESTIMATORS
# ============================================================================

class CrossValResidualEstimator(NoiseEstimator):
    """
    Estimate noise using cross-validated predictions.
    
    noise_score = |y - cv_prediction|
    
    Idea: If a model trained on OTHER samples can't predict your label,
    your label might be noisy.
    """
    
    def __init__(self, n_folds: int = 5, normalize: bool = True,
                 groups=None, random_state: int = 42):
        super().__init__(random_state)
        self.n_folds = n_folds
        self.normalize = normalize
        self.groups = groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Out-of-fold predictions on the same scaffold-disjoint folds the other
        # estimators use. Without groups a molecule's own scaffold-mates train
        # the model that predicts it, so its residual is small for a reason
        # that has nothing to do with its label being noisy.
        def fold_fn(X_tr, y_tr, X_te):
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            return {'pred': model.predict(X_te)}

        oof = self._oof_apply(X, y, fold_fn, n_folds=self.n_folds,
                              groups=self.groups)
        cv_pred = oof['pred']
        residuals = np.abs(y - cv_pred)
        
        if self.normalize:
            residuals = residuals / (np.std(y) + 1e-10)
        
        self.noise_scores_ = residuals
        self.cv_predictions_ = cv_pred
        return residuals


class EnsembleDisagreementEstimator(NoiseEstimator):
    """
    Estimate noise using ensemble prediction variance.
    
    Train multiple models, measure how much they disagree on each sample.
    High disagreement suggests the label might be inconsistent with features.
    
    noise_score = std(predictions across models) + |y - mean_prediction|
    """
    
    def __init__(self, n_estimators: int = 10, normalize: bool = True,
                 n_folds: int = 5, groups=None, random_state: int = 42):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.normalize = normalize
        self.n_folds = n_folds
        self.groups = groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        def fold_fn(X_tr, y_tr, X_ev):
            preds = np.zeros((len(X_ev), self.n_estimators))
            for i in range(self.n_estimators):
                # Different random states for diversity
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_features='sqrt',
                    random_state=self.random_state + i,
                    n_jobs=-1
                )
                model.fit(X_tr, y_tr)
                preds[:, i] = model.predict(X_ev)
            return {'pred_mean': preds.mean(axis=1), 'pred_std': preds.std(axis=1)}

        # Out-of-fold: each row's prediction comes from models that never saw it,
        # so the residual term is honest rather than a memorised zero.
        oof = self._oof_apply(X, y, fold_fn, n_folds=self.n_folds, groups=self.groups)
        pred_mean = oof['pred_mean']
        pred_std = oof['pred_std']

        # Combine disagreement with residual
        residual = np.abs(y - pred_mean)
        scores = pred_std + residual

        if self.normalize:
            scores = scores / (np.std(y) + 1e-10)

        self.noise_scores_ = scores
        self.prediction_means_ = pred_mean
        self.prediction_stds_ = pred_std
        return scores


class LeaveOneOutEstimator(NoiseEstimator):
    """
    Estimate noise using leave-one-out predictions.

    For each sample, train on all others and predict.
    High residual = label inconsistent with the rest of the data.

    Computationally expensive but most direct estimate.
    Uses RF's OOB predictions as efficient approximation.

    exclude_groups: one label per row. Out-of-bag leaves a row out of some
        trees, but never leaves out the OTHER rows carrying the same label. On
        BindingDB those are repeat measurements of the same (drug, target)
        pair, so a row's residual is predicted from its own repeats — the very
        labels whose spread the estimator is then scored against. With
        exclude_groups the out-of-bag path is replaced by a group-disjoint
        cross-validated residual: whole pairs are held out together, so a row's
        prediction never sees its own repeats. n_group_folds sets how many
        folds that split uses; the more folds, the closer to leaving one pair
        out at a time and the more model fits it costs.
    """

    def __init__(self, normalize: bool = True, exclude_groups=None,
                 n_group_folds: int = 20, random_state: int = 42):
        super().__init__(random_state)
        self.normalize = normalize
        self.exclude_groups = exclude_groups
        self.n_group_folds = n_group_folds
        self.n_group_folds_used_ = None

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.exclude_groups is not None:
            g = np.asarray(self.exclude_groups)
            n_groups = len(np.unique(g))
            self.n_group_folds_used_ = max(2, min(self.n_group_folds, n_groups))

            def _fold(X_tr, y_tr, X_ev):
                m = RandomForestRegressor(
                    n_estimators=200, random_state=self.random_state, n_jobs=-1)
                m.fit(X_tr, y_tr)
                return {'pred': m.predict(X_ev)}

            pred = self._oof_apply(X, y, _fold,
                                   n_folds=self.n_group_folds_used_,
                                   groups=g)['pred']
        else:
            # Use OOB predictions as LOO approximation
            model = RandomForestRegressor(
                n_estimators=200,
                oob_score=True,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X, y)
            pred = model.oob_prediction_

        residuals = np.abs(y - pred)

        if self.normalize:
            residuals = residuals / (np.std(y) + 1e-10)

        self.noise_scores_ = residuals
        self.oob_predictions_ = pred
        return residuals


# ============================================================================
# HYBRID ESTIMATORS
# ============================================================================

class HybridNoiseEstimator(NoiseEstimator):
    """
    Combine multiple estimators for robust noise estimation.
    
    Averages normalized scores from:
    - Neighbor residual (structure-based)
    - Cross-val residual (model-based)
    - Optionally: local variance ratio (activity-cliff aware)
    """
    
    def __init__(self, k: int = 10, metric: str = 'cosine',
                 use_cv: bool = True, use_variance_ratio: bool = True,
                 groups=None, exclude_groups=None, random_state: int = 42):
        super().__init__(random_state)
        self.k = k
        self.metric = metric
        self.use_cv = use_cv
        self.use_variance_ratio = use_variance_ratio
        self.groups = groups
        # Forwarded to the two neighbour-based parts, so a row's repeat
        # measurements stay out of its own score here too.
        self.exclude_groups = exclude_groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        estimators = []

        # Always use neighbor residual
        nr = NeighborResidualEstimator(k=self.k, metric=self.metric,
                                        normalize=True,
                                        exclude_groups=self.exclude_groups,
                                        random_state=self.random_state)
        estimators.append(nr)

        if self.use_variance_ratio:
            lvr = LocalVarianceRatioEstimator(k=self.k, metric=self.metric,
                                               exclude_groups=self.exclude_groups,
                                               random_state=self.random_state)
            estimators.append(lvr)
        
        if self.use_cv:
            cv = CrossValResidualEstimator(normalize=True, groups=self.groups,
                                           random_state=self.random_state)
            estimators.append(cv)
        
        # Get scores from each
        all_scores = []
        for est in estimators:
            scores = est.estimate(X, y)
            # Normalize to [0, 1] range
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            all_scores.append(scores_norm)
        
        # Average
        combined = np.mean(all_scores, axis=0)
        
        self.noise_scores_ = combined
        self.neighbor_means_ = estimators[0].neighbor_means_
        return combined


# ============================================================================
# SOTA MODEL-BASED ESTIMATORS
# ============================================================================

class GaussianProcessEstimator(NoiseEstimator):
    """
    Gaussian Process regression provides natural noise estimation.
    
    GP models y = f(x) + ε where ε ~ N(0, σ²).
    The learned noise variance σ² indicates data noise level.
    Per-sample uncertainty comes from posterior variance.
    
    For heteroscedastic noise, we use a separate GP for log-variance.
    """
    
    def __init__(self, kernel: str = 'rbf', n_restarts: int = 3,
                 normalize: bool = True, n_folds: int = 5, groups=None,
                 random_state: int = 42):
        super().__init__(random_state)
        self.kernel = kernel
        self.n_restarts = n_restarts
        self.normalize = normalize
        self.n_folds = n_folds
        self.groups = groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
        from sklearn.preprocessing import StandardScaler

        noise_levels = []

        def fold_fn(X_tr, y_tr, X_ev):
            # Standardise on the TRAINING fold only, then apply to eval rows.
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_tr_s = scaler_X.fit_transform(X_tr)
            y_tr_s = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
            X_ev_s = scaler_X.transform(X_ev)

            if self.kernel == 'matern':
                base_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            else:
                base_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            kernel = base_kernel + WhiteKernel(noise_level=0.1,
                                               noise_level_bounds=(1e-5, 1.0))

            # Cap the training-fold size for tractability (deterministic subset).
            n_max = min(1000, len(y_tr_s))
            if len(y_tr_s) > n_max:
                idx = np.random.RandomState(self.random_state).choice(
                    len(y_tr_s), n_max, replace=False)
                X_fit, y_fit = X_tr_s[idx], y_tr_s[idx]
            else:
                X_fit, y_fit = X_tr_s, y_tr_s

            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=self.n_restarts,
                random_state=self.random_state, normalize_y=False)
            gp.fit(X_fit, y_fit)

            y_pred_s, y_std_s = gp.predict(X_ev_s, return_std=True)
            y_scale = scaler_y.scale_[0]
            noise_levels.append(
                np.sqrt(gp.kernel_.get_params().get('k2__noise_level', 0.1)) * y_scale)
            # Return everything in ORIGINAL y units so folds are comparable.
            return {
                'y_pred': y_pred_s * y_scale + scaler_y.mean_[0],
                'y_std': y_std_s * y_scale,
            }

        oof = self._oof_apply(X, y, fold_fn, n_folds=self.n_folds, groups=self.groups)
        y_pred = oof['y_pred']
        y_std = oof['y_std']

        # Noise score = out-of-fold predictive uncertainty + out-of-fold residual
        residuals = np.abs(y - y_pred)
        scores = y_std + residuals

        if self.normalize:
            scores = scores / (scores.std() + 1e-10)

        self.noise_scores_ = scores
        self.predictions_ = y_pred
        self.uncertainties_ = y_std
        self.learned_noise_level_ = float(np.mean(noise_levels)) if noise_levels else 0.1

        return scores


class HeteroscedasticNNEstimator(NoiseEstimator):
    """
    Heteroscedastic Neural Network with β-NLL loss.
    
    Network predicts both μ(x) and σ²(x).
    β-NLL prevents variance from hiding mean errors.
    
    loss = 0.5 * ((y - μ)² / σ² + log(σ²)) * σ^(2β)
    
    β=0: standard NLL (can be unstable)
    β=0.5: balanced (recommended)
    β=1: equal weighting
    
    Reference: Seitzer et al. ICLR 2022
    """
    
    def __init__(self, hidden_dims: Tuple[int, ...] = (64, 32),
                 beta: float = 0.5, epochs: int = 100, lr: float = 0.001,
                 batch_size: int = 64, normalize: bool = True,
                 n_folds: int = 5, groups=None, random_state: int = 42):
        super().__init__(random_state)
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.normalize = normalize
        self.n_folds = n_folds
        self.groups = groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = X.shape[1]

        class HeteroscedasticNet(nn.Module):
            def __init__(self, input_dim, hidden_dims):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for h_dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.Dropout(0.1)])
                    prev_dim = h_dim
                self.shared = nn.Sequential(*layers)
                self.mean_head = nn.Linear(prev_dim, 1)
                self.var_head = nn.Linear(prev_dim, 1)
                
            def forward(self, x):
                h = self.shared(x)
                mean = self.mean_head(h)
                log_var = self.var_head(h)
                # Clamp for stability
                log_var = torch.clamp(log_var, min=-10, max=10)
                return mean.squeeze(), log_var.squeeze()
        
        # β-NLL loss
        def beta_nll_loss(mean, log_var, target, beta=0.5):
            var = torch.exp(log_var)
            loss = 0.5 * ((target - mean) ** 2 / var + log_var)
            if beta > 0:
                loss = loss * (var.detach() ** beta)
            return loss.mean()

        def fold_fn(X_tr, y_tr, X_ev):
            # Standardise on the training fold only, apply to held-out rows.
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_tr_s = scaler_X.fit_transform(X_tr).astype(np.float32)
            y_tr_s = scaler_y.fit_transform(y_tr.reshape(-1, 1)).astype(np.float32).ravel()
            X_ev_s = scaler_X.transform(X_ev).astype(np.float32)

            model = HeteroscedasticNet(input_dim, self.hidden_dims)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                         weight_decay=1e-4)
            dataset = TensorDataset(torch.from_numpy(X_tr_s),
                                    torch.from_numpy(y_tr_s))
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            model.train()
            for epoch in range(self.epochs):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    mean, log_var = model(X_batch)
                    loss = beta_nll_loss(mean, log_var, y_batch, self.beta)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                mean_pred, log_var_pred = model(torch.from_numpy(X_ev_s))
                mean_pred = np.atleast_1d(mean_pred.numpy())
                std_pred = np.sqrt(np.atleast_1d(np.exp(log_var_pred.numpy())))
            y_scale = scaler_y.scale_[0]
            # Return in ORIGINAL y units so folds are comparable.
            return {
                'pred': mean_pred * y_scale + scaler_y.mean_[0],
                'std': std_pred * y_scale,
            }

        oof = self._oof_apply(X, y, fold_fn, n_folds=self.n_folds, groups=self.groups)
        std_pred = oof['std']

        # Noise score = out-of-fold predicted uncertainty
        scores = std_pred

        if self.normalize:
            scores = scores / (scores.std() + 1e-10)

        self.noise_scores_ = scores
        self.predictions_ = oof['pred']
        self.uncertainties_ = std_pred

        return scores


class ConformalEstimator(NoiseEstimator):
    """
    Conformal Prediction for noise estimation.
    
    Uses split conformal prediction to get prediction intervals.
    Wider intervals indicate higher noise/uncertainty.
    
    Can use any base model; we use RF or Quantile Regression.
    
    Reference: ACS Omega 2024 (QSAR-specific)
    """
    
    def __init__(self, base_model: str = 'rf', alpha: float = 0.1,
                 n_cal: float = 0.2, normalize: bool = True,
                 n_folds: int = 5, groups=None, random_state: int = 42):
        super().__init__(random_state)
        self.base_model = base_model
        self.alpha = alpha  # miscoverage rate
        self.n_cal = n_cal  # fraction of each training fold used for calibration
        self.normalize = normalize
        self.n_folds = n_folds
        # Optional scaffold-group labels (one per row of X). When present the
        # out-of-fold split is scaffold-DISJOINT, so a row's interval never
        # comes from a model that trained on its scaffold.
        self.groups = groups

    def _fold_conformity(self, X_tr, y_tr, X_ev):
        """Fit the base model + conformal quantile on the training fold, then
        return interval widths / predictions / uncertainties for the held-out
        eval rows. The inner fit/calibration split is a deterministic
        contiguous split of the training fold (disjoint fit vs calibration);
        the OUTER out-of-fold split already guarantees X_ev was never trained
        on, which is what removes the in-sample leakage."""
        n_cal = max(1, int(len(y_tr) * self.n_cal))
        idx = np.arange(len(y_tr))
        cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
        if len(fit_idx) == 0:  # degenerate tiny fold
            fit_idx, cal_idx = idx, idx
        X_fit, y_fit = X_tr[fit_idx], y_tr[fit_idx]
        X_cal, y_cal = X_tr[cal_idx], y_tr[cal_idx]

        if self.base_model == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100,
                                          random_state=self.random_state, n_jobs=-1)
            model.fit(X_fit, y_fit)
            y_pred_cal = model.predict(X_cal)
            cal_std = np.array([t.predict(X_cal) for t in model.estimators_]).std(axis=0)
            ev_pred = model.predict(X_ev)
            ev_unc = np.array([t.predict(X_ev) for t in model.estimators_]).std(axis=0)
        elif self.base_model == 'quantile':
            from sklearn.ensemble import GradientBoostingRegressor
            model_lo = GradientBoostingRegressor(loss='quantile', alpha=self.alpha / 2,
                                                 n_estimators=100, random_state=self.random_state)
            model_hi = GradientBoostingRegressor(loss='quantile', alpha=1 - self.alpha / 2,
                                                 n_estimators=100, random_state=self.random_state)
            model_med = GradientBoostingRegressor(loss='quantile', alpha=0.5,
                                                  n_estimators=100, random_state=self.random_state)
            model_lo.fit(X_fit, y_fit)
            model_hi.fit(X_fit, y_fit)
            model_med.fit(X_fit, y_fit)
            y_pred_cal = model_med.predict(X_cal)
            cal_std = (model_hi.predict(X_cal) - model_lo.predict(X_cal)) / 2
            ev_pred = model_med.predict(X_ev)
            ev_unc = (model_hi.predict(X_ev) - model_lo.predict(X_ev)) / 2
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")

        residuals_cal = np.abs(y_cal - y_pred_cal)
        conformity_scores = residuals_cal / (cal_std + 1e-10)
        q = np.quantile(conformity_scores, 1 - self.alpha)
        return {
            'interval_width': 2 * q * ev_unc,
            'pred': ev_pred,
            'uncertainty': ev_unc,
        }

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        oof = self._oof_apply(X, y, self._fold_conformity,
                              n_folds=self.n_folds, groups=self.groups)
        interval_width = oof['interval_width']

        scores = interval_width
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)

        self.noise_scores_ = scores
        self.predictions_ = oof['pred']
        self.uncertainties_ = oof['uncertainty']
        self.interval_widths_ = interval_width

        return scores


class DeepEnsembleEstimator(NoiseEstimator):
    """
    Deep Ensemble for uncertainty estimation.
    
    Trains M independent neural networks and uses:
    - Mean of means as prediction
    - Variance of means as epistemic uncertainty
    - Mean of variances as aleatoric uncertainty (if heteroscedastic)
    
    Total uncertainty = epistemic + aleatoric
    
    Reference: Lakshminarayanan et al. NeurIPS 2017
    """
    
    def __init__(self, n_ensemble: int = 5, hidden_dims: Tuple[int, ...] = (64, 32),
                 heteroscedastic: bool = True, epochs: int = 100,
                 lr: float = 0.001, normalize: bool = True,
                 n_folds: int = 5, groups=None, random_state: int = 42):
        super().__init__(random_state)
        self.n_ensemble = n_ensemble
        self.hidden_dims = hidden_dims
        self.heteroscedastic = heteroscedastic
        self.epochs = epochs
        self.lr = lr
        self.normalize = normalize
        self.n_folds = n_folds
        self.groups = groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        input_dim = X.shape[1]

        # Network definition
        class EnsembleNet(nn.Module):
            def __init__(self, input_dim, hidden_dims, heteroscedastic):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for h_dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
                    prev_dim = h_dim
                self.shared = nn.Sequential(*layers)
                self.mean_head = nn.Linear(prev_dim, 1)
                self.heteroscedastic = heteroscedastic
                if heteroscedastic:
                    self.var_head = nn.Linear(prev_dim, 1)
                
            def forward(self, x):
                h = self.shared(x)
                mean = self.mean_head(h).squeeze()
                if self.heteroscedastic:
                    log_var = torch.clamp(self.var_head(h).squeeze(), -10, 10)
                    return mean, log_var
                return mean, None
        
        def fold_fn(X_tr, y_tr, X_ev):
            # Standardise on the training fold only, apply to held-out rows.
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_tr_s = scaler_X.fit_transform(X_tr).astype(np.float32)
            y_tr_s = scaler_y.fit_transform(y_tr.reshape(-1, 1)).astype(np.float32).ravel()
            X_ev_s = scaler_X.transform(X_ev).astype(np.float32)

            ensemble_means = []
            ensemble_vars = []
            for m in range(self.n_ensemble):
                torch.manual_seed(self.random_state + m)
                model = EnsembleNet(input_dim, self.hidden_dims, self.heteroscedastic)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                dataset = TensorDataset(torch.from_numpy(X_tr_s),
                                        torch.from_numpy(y_tr_s))
                loader = DataLoader(dataset, batch_size=64, shuffle=True)

                model.train()
                for epoch in range(self.epochs):
                    for X_batch, y_batch in loader:
                        optimizer.zero_grad()
                        mean, log_var = model(X_batch)
                        if self.heteroscedastic:
                            var = torch.exp(log_var)
                            loss = 0.5 * ((y_batch - mean) ** 2 / var + log_var).mean()
                        else:
                            loss = ((y_batch - mean) ** 2).mean()
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    mean_pred, log_var_pred = model(torch.from_numpy(X_ev_s))
                    ensemble_means.append(np.atleast_1d(mean_pred.numpy()))
                    if self.heteroscedastic:
                        ensemble_vars.append(np.atleast_1d(np.exp(log_var_pred.numpy())))

            ensemble_means = np.array(ensemble_means)
            epistemic = ensemble_means.std(axis=0)
            if self.heteroscedastic:
                aleatoric = np.sqrt(np.array(ensemble_vars).mean(axis=0))
            else:
                aleatoric = np.zeros_like(epistemic)
            total = np.sqrt(epistemic ** 2 + aleatoric ** 2)
            y_scale = scaler_y.scale_[0]
            # Return in ORIGINAL y units so folds are comparable.
            return {
                'pred': ensemble_means.mean(axis=0) * y_scale + scaler_y.mean_[0],
                'epistemic': epistemic * y_scale,
                'aleatoric': aleatoric * y_scale,
                'total': total * y_scale,
            }

        oof = self._oof_apply(X, y, fold_fn, n_folds=self.n_folds, groups=self.groups)
        total_uncertainty = oof['total']

        scores = total_uncertainty
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)

        self.noise_scores_ = scores
        self.predictions_ = oof['pred']
        self.epistemic_uncertainty_ = oof['epistemic']
        self.aleatoric_uncertainty_ = oof['aleatoric']
        self.total_uncertainty_ = total_uncertainty

        return scores


class BayesianRFEstimator(NoiseEstimator):
    """
    Bayesian interpretation of Random Forest uncertainty.
    
    Uses:
    1. Prediction variance across trees (epistemic)
    2. OOB residual variance (aleatoric proxy)
    3. Jackknife+ variance estimate
    
    This is computationally cheaper than deep ensembles.
    """
    
    def __init__(self, n_estimators: int = 200, method: str = 'jackknife',
                 normalize: bool = True, n_folds: int = 5, groups=None,
                 random_state: int = 42):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.method = method
        self.normalize = normalize
        self.n_folds = n_folds
        self.groups = groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.ensemble import RandomForestRegressor

        def fold_fn(X_tr, y_tr, X_ev):
            rf = RandomForestRegressor(n_estimators=self.n_estimators,
                                       random_state=self.random_state, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            tree_preds = np.array([t.predict(X_ev) for t in rf.estimators_])
            pred = rf.predict(X_ev)

            if self.method in ('tree_variance', 'jackknife'):
                # Tree spread on held-out rows. For jackknife the honest
                # out-of-fold residual term is added after assembly (it needs
                # the full y vector), so both start from the tree spread here.
                uncertainty = tree_preds.std(axis=0)
            elif self.method == 'quantile':
                q_lo = np.percentile(tree_preds, 10, axis=0)
                q_hi = np.percentile(tree_preds, 90, axis=0)
                uncertainty = (q_hi - q_lo) / 2
            else:
                raise ValueError(f"Unknown method: {self.method}")
            return {'uncertainty': uncertainty, 'pred': pred}

        oof = self._oof_apply(X, y, fold_fn, n_folds=self.n_folds, groups=self.groups)
        uncertainty = oof['uncertainty']
        pred = oof['pred']

        if self.method == 'jackknife':
            # Add the out-of-fold residual term (Wager-style variance proxy).
            uncertainty = uncertainty + np.abs(y - pred) * 0.5

        scores = uncertainty
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)

        self.noise_scores_ = scores
        self.predictions_ = pred
        self.uncertainties_ = uncertainty
        self.oob_predictions_ = pred  # out-of-fold predictions (OOB analogue)

        return scores


class QuantileRegressionEstimator(NoiseEstimator):
    """
    Quantile Regression for heteroscedastic uncertainty.
    
    Fits models for multiple quantiles and uses interval width
    as noise estimate. Wider intervals = more uncertainty.
    
    Can use Gradient Boosting or Neural Network quantile regression.
    """
    
    def __init__(self, quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
                 model_type: str = 'gbr', normalize: bool = True,
                 n_folds: int = 5, groups=None, random_state: int = 42):
        super().__init__(random_state)
        self.quantiles = quantiles
        self.model_type = model_type
        self.normalize = normalize
        self.n_folds = n_folds
        self.groups = groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        q_lo = min(self.quantiles)
        q_hi = max(self.quantiles)
        q_med = sorted(self.quantiles)[len(self.quantiles) // 2]

        def fold_fn(X_tr, y_tr, X_ev):
            preds = {}
            if self.model_type == 'gbr':
                from sklearn.ensemble import GradientBoostingRegressor
                for q in self.quantiles:
                    model = GradientBoostingRegressor(
                        loss='quantile', alpha=q, n_estimators=100,
                        max_depth=5, random_state=self.random_state)
                    model.fit(X_tr, y_tr)
                    preds[q] = model.predict(X_ev)
            elif self.model_type == 'nn':
                import torch
                import torch.nn as nn
                from torch.utils.data import DataLoader, TensorDataset
                from sklearn.preprocessing import StandardScaler

                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_tr_s = scaler_X.fit_transform(X_tr).astype(np.float32)
                y_tr_s = scaler_y.fit_transform(y_tr.reshape(-1, 1)).astype(np.float32).ravel()
                X_ev_s = scaler_X.transform(X_ev).astype(np.float32)

                class QuantileNet(nn.Module):
                    def __init__(self, input_dim, n_quantiles):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(input_dim, 64), nn.ReLU(),
                            nn.Linear(64, 32), nn.ReLU(),
                            nn.Linear(32, n_quantiles))

                    def forward(self, x):
                        return self.net(x)

                def pinball_loss(pred, target, quantiles):
                    target = target.unsqueeze(1).expand_as(pred)
                    quantiles_t = torch.tensor(quantiles).float().unsqueeze(0).expand_as(pred)
                    errors = target - pred
                    return torch.max(quantiles_t * errors, (quantiles_t - 1) * errors).mean()

                torch.manual_seed(self.random_state)
                model = QuantileNet(X_tr.shape[1], len(self.quantiles))
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                dataset = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_s))
                loader = DataLoader(dataset, batch_size=64, shuffle=True)
                model.train()
                for epoch in range(100):
                    for X_batch, y_batch in loader:
                        optimizer.zero_grad()
                        loss = pinball_loss(model(X_batch), y_batch, self.quantiles)
                        loss.backward()
                        optimizer.step()
                model.eval()
                with torch.no_grad():
                    ev = np.atleast_2d(model(torch.from_numpy(X_ev_s)).numpy())
                y_scale, y_mean = scaler_y.scale_[0], scaler_y.mean_[0]
                preds = {q: ev[:, i] * y_scale + y_mean
                         for i, q in enumerate(self.quantiles)}
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            return {
                'interval_width': preds[q_hi] - preds[q_lo],
                'pred_median': preds[q_med],
            }

        oof = self._oof_apply(X, y, fold_fn, n_folds=self.n_folds, groups=self.groups)
        interval_width = oof['interval_width']

        scores = interval_width
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)

        self.noise_scores_ = scores
        self.predictions_ = oof['pred_median']
        self.interval_widths_ = interval_width

        return scores


class NGBoostEstimator(NoiseEstimator):
    """
    NGBoost for probabilistic predictions with uncertainty.
    
    Returns:
    - noise_scores: sigma (predicted std) - correlates with label noise
    - predictions_: mu (predicted mean) - denoised estimate
    
    Use with method='confidence' smoothing:
        correction = (alpha * sigma) / (1 + alpha * sigma)
        y_denoised = (1 - correction) * y + correction * mu
    
    Reference: Duan et al. 2020 "NGBoost: Natural Gradient Boosting"
    """
    
    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.01,
                 minibatch_frac: float = 1.0, normalize: bool = False,
                 n_folds: int = 5, groups=None, random_state: int = 42):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.normalize = normalize  # Default False - use raw sigma for confidence method
        self.n_folds = n_folds
        self.groups = groups

    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        def fold_fn(X_tr, y_tr, X_ev):
            model = NGBRegressor(
                Dist=Normal, n_estimators=self.n_estimators,
                learning_rate=self.learning_rate, minibatch_frac=self.minibatch_frac,
                random_state=self.random_state, verbose=False)
            model.fit(X_tr, y_tr)
            dist = model.pred_dist(X_ev)
            return {'mu': dist.loc, 'sigma': dist.scale}

        oof = self._oof_apply(X, y, fold_fn, n_folds=self.n_folds, groups=self.groups)
        mu = oof['mu']
        sigma = oof['sigma']

        scores = sigma
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)

        self.noise_scores_ = scores
        self.predictions_ = mu  # Use as correction target
        self.sigma_ = sigma
        self.mu_ = mu

        return scores


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

ESTIMATOR_REGISTRY = {
    # Neighbor-based (use molecular representation distances)
    'neighbor_residual': NeighborResidualEstimator,
    'local_variance_ratio': LocalVarianceRatioEstimator,
    'distance_weighted': DistanceWeightedResidualEstimator,
    'distance_weighted_residual': DistanceWeightedResidualEstimator,  # alias
    'activity_cliff_aware': ActivityCliffAwareEstimator,

    # Model-based (use prediction disagreement)
    'cross_val': CrossValResidualEstimator,
    'cross_val_residual': CrossValResidualEstimator,  # alias
    'ensemble': EnsembleDisagreementEstimator,
    'ensemble_disagreement': EnsembleDisagreementEstimator,  # alias
    'leave_one_out': LeaveOneOutEstimator,
    'hybrid': HybridNoiseEstimator,
    'hybrid_noise': HybridNoiseEstimator,  # alias

    # SOTA methods (estimate noise magnitude directly)
    'gaussian_process': GaussianProcessEstimator,
    'heteroscedastic_nn': HeteroscedasticNNEstimator,
    'conformal': ConformalEstimator,
    'deep_ensemble': DeepEnsembleEstimator,
    'bayesian_rf': BayesianRFEstimator,
    'quantile_regression': QuantileRegressionEstimator,
    'ngboost': NGBoostEstimator,
}


def get_noise_estimator(method: str, **kwargs) -> NoiseEstimator:
    """
    Get noise estimator by name.
    
    Args:
        method: One of 'neighbor_residual', 'local_variance_ratio', 
                'distance_weighted', 'activity_cliff_aware', 'cross_val',
                'ensemble', 'leave_one_out', 'hybrid'
        **kwargs: Estimator-specific parameters
    
    Returns:
        NoiseEstimator instance
    """
    if method not in ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown method: {method}. Available: {list(ESTIMATOR_REGISTRY.keys())}")
    
    return ESTIMATOR_REGISTRY[method](**kwargs)


# ============================================================================
# SMOOTHING FUNCTIONS (use estimator output)
# ============================================================================

def smooth_by_scores(y: np.ndarray, 
                     noise_scores: np.ndarray,
                     neighbor_means: np.ndarray,
                     alpha: float = 0.5,
                     method: str = 'proportional') -> np.ndarray:
    """
    Smooth labels using noise scores.
    
    Args:
        y: Original labels
        noise_scores: From estimator (higher = noisier)
        neighbor_means: Target values to smooth toward
        alpha: Smoothing strength
        method: 'proportional', 'threshold', 'soft_threshold', or 'confidence'
    
    Returns:
        y_smoothed: Corrected labels
    """
    # Normalize scores to [0, 1] for most methods
    scores_norm = (noise_scores - noise_scores.min()) / (noise_scores.max() - noise_scores.min() + 1e-10)
    
    if method == 'proportional':
        # Correct proportionally to noise score
        # High score = more correction toward neighbor mean
        correction_strength = alpha * scores_norm
        y_smooth = y + correction_strength * (neighbor_means - y)
        
    elif method == 'threshold':
        # Only correct samples above threshold
        threshold = np.percentile(scores_norm, 100 * (1 - alpha))
        correction_mask = scores_norm > threshold
        y_smooth = y.copy()
        y_smooth[correction_mask] = (1 - alpha) * y[correction_mask] + alpha * neighbor_means[correction_mask]
        
    elif method == 'soft_threshold':
        # Soft threshold: minimal correction for low scores, strong for high
        # Using sigmoid-like function
        correction_strength = alpha * (1 / (1 + np.exp(-5 * (scores_norm - 0.5))))
        y_smooth = y + correction_strength * (neighbor_means - y)
    
    elif method == 'confidence':
        # Use raw noise_scores (e.g., sigma from NGBoost), not normalized
        # High noise_score = more correction toward target (mu)
        # alpha scales the noise_scores
        scaled_scores = alpha * noise_scores
        correction = scaled_scores / (1 + scaled_scores)
        y_smooth = (1 - correction) * y + correction * neighbor_means
        
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return y_smooth


def weights_from_scores(noise_scores: np.ndarray, 
                        strength: float = 1.0,
                        method: str = 'inverse') -> np.ndarray:
    """
    Convert noise scores to sample weights.
    
    Args:
        noise_scores: From estimator
        strength: How aggressively to downweight
        method: 'inverse', 'exponential', or 'soft'
    
    Returns:
        weights: Sample weights (mean-normalized to 1)
    """
    # Normalize scores
    scores_norm = (noise_scores - noise_scores.min()) / (noise_scores.max() - noise_scores.min() + 1e-10)
    
    if method == 'inverse':
        weights = 1 / (1 + strength * scores_norm)
    elif method == 'exponential':
        weights = np.exp(-strength * scores_norm)
    elif method == 'soft':
        # Soft weighting: mostly 1, drops for high scores
        weights = 1 - strength * scores_norm * scores_norm
        weights = np.clip(weights, 0.1, 1.0)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights / weights.mean()