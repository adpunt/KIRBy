#!/usr/bin/env python3
"""
Noise Mitigation Methods for QSAR
==================================

Implements techniques to identify and remove noisy labels from training data.

Categories:
1. Distance-based: k-NN, Activity Cliffs, Mahalanobis, LOF, DistanceWeighted, NeighborhoodConsensus
2. Ensemble-based: CV disagreement, Bootstrap ensemble
3. Model-based: Co-Teaching, DivideMix (adapted)
4. Simple baselines: Z-score, Prediction error
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist, mahalanobis
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BASE CLASS
# ============================================================================

class NoiseMitigationMethod(ABC):
    """Base class for noise mitigation methods"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.noisy_indices_ = None
        
    @abstractmethod
    def identify_noise(self, X, y):
        """
        Identify noisy samples.
        
        Returns:
            noisy_indices: array of indices flagged as noisy
        """
        pass
    
    def clean_data(self, X, y):
        """
        Remove noisy samples.
        
        Returns:
            X_clean, y_clean, removed_indices
        """
        self.noisy_indices_ = self.identify_noise(X, y)
        clean_mask = np.ones(len(y), dtype=bool)
        clean_mask[self.noisy_indices_] = False
        
        return X[clean_mask], y[clean_mask], self.noisy_indices_


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_distance_matrix(X, metric=None, feature_weights=None,
                            standardize=True, weight_convention='mean1',
                            p=None):
    """
    Compute pairwise distance matrix using scipy.cdist.

    Args:
        X: Feature matrix (n_samples, n_features)
        metric: Distance metric (euclidean, manhattan, cosine, chebyshev,
                canberra, correlation, braycurtis, mahalanobis), or the
                canonical name 'minkowski_p0.5'. None (the default) reads
                kirby.similarity.distance_config.get_distance_metric(), which
                is manhattan unless --distance-metric or KIRBY_DISTANCE_METRIC
                says otherwise. Before 2026-09-04 the default was the literal
                'euclidean' and no caller in the experimental suite passed
                anything, so all 22 of them were Euclidean by omission.
        p: Minkowski exponent. Required when metric='minkowski' and ignored
                otherwise. scipy's cdist takes p=2 when nobody passes one, so
                metric='minkowski' with no p returned the Euclidean number --
                measured 3.319733 for both on the same input, against 13.014929
                at p=0.5. Passing 'minkowski_p0.5' as the metric fills p in.
        feature_weights: Optional per-feature weights, applied AFTER
                standardising. Negatives are clipped to 0 and the vector is
                rescaled to average 1 (see weight_convention).
        standardize: Whether to standardize features first
        weight_convention: 'mean1' (default, f17's measured choice), 'sum1' or
                'raw'. Rankings are identical under all three; absolute cutoffs
                and kernel widths are not.

    Returns:
        D: Distance matrix (n_samples, n_samples)

    FIXED 2026-08-26. Before that date every metric got the same
    `X * np.sqrt(feature_weights)`, which is correct for euclidean and cosine
    and wrong for the other six:

      manhattan, chebyshev, braycurtis  got sqrt(w) where they need w. The
          resulting ranking is not even the same one -- Spearman 0.976 against
          the intended weighting.
      canberra    got NOTHING. Canberra divides each term by the same features
          it differences, so any per-feature rescale cancels exactly: the
          weighted and unweighted calls differed by 4.4e-16, while the same
          weights moved euclidean by 6.79.
      mahalanobis got NOTHING. Its covariance is re-estimated from the rescaled
          features, so the rescale cancels by affine invariance: 3.6e-15.
      correlation got neither the weighted nor the unweighted answer, because
          the weight belongs inside the centring mean as well as the three sums.

    Negative weights are now clipped at the door. np.sqrt of a negative returns
    nan with only a RuntimeWarning, this module suppresses warnings at import,
    and the result was an all-nan distance matrix with no error. The seven
    `np.maximum(w, 0)` guards scattered across the s- and n- experiments were
    working around exactly that.
    """
    from kirby.similarity.preprocessing import (
        check_input, coefficient_family, fold_weights, NO_WEIGHT,
    )
    from kirby.similarity.distance_config import base_and_p, DISTANCE_METRICS

    if metric is None or str(metric).lower() in {m.lower()
                                                 for m in DISTANCE_METRICS}:
        # One of the four the suite sweeps, or nothing at all. base_and_p
        # resolves the process-wide setting and splits 'minkowski_p0.5' into a
        # family name plus its exponent.
        metric, _p = base_and_p(metric)
        p = _p if p is None else p
    if metric == 'minkowski' and p is None:
        raise ValueError(
            "minkowski needs its exponent p. Without one scipy uses p=2 and "
            "returns the Euclidean number under a different name. Pass p=, or "
            "pass metric='minkowski_p0.5'.")

    X_proc = np.asarray(X, dtype=np.float64).copy()

    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    # braycurtis needs features >= 0. On z-scored features its median is exactly
    # 1.00, half the pairs score above 1, and it stops being monotone in how far
    # apart two rows are: x = [1] against y = [-0.99] scores 199, while the same
    # x against y = [-100] scores 1.02.
    check_input(X_proc, metric)

    scipy_kw = {}
    if feature_weights is not None:
        if coefficient_family(metric) == NO_WEIGHT:
            # mahalanobis. Silently ignoring the weights is what happened
            # before; refuse instead, so a weighted mahalanobis arm cannot be
            # reported as a measured null.
            raise ValueError(
                "mahalanobis has no per-feature weighting that survives its own "
                "covariance re-estimation -- the weighted and unweighted matrices "
                "differ by 3.6e-15. Drop mahalanobis from the weighted arm, or "
                "build the fixed-precision form where the covariance is estimated "
                "once on unweighted features and held fixed.")
        # The exponent that folds a weight into the features is 1/p, so it has
        # to be the SAME p scipy is about to raise the differences to. It was
        # hard-coded to 3 here while scipy silently used 2.
        X_proc, scipy_kw = fold_weights(
            X_proc, feature_weights, metric,
            p=(p if metric == 'minkowski' else None),
            convention=weight_convention)

    # Handle special metrics
    if metric == 'mahalanobis':
        cov = EmpiricalCovariance().fit(X_proc)
        # rank, not d < n. With column centring rank(cov) <= min(d, n-1), so
        # d == n-1 passes a `d >= n` guard and returns a CONSTANT matrix --
        # every off-diagonal exactly sqrt(2n), std across them 3.7e-15.
        d = X_proc.shape[1]
        if np.linalg.matrix_rank(cov.covariance_) < d:
            raise ValueError(
                f"mahalanobis needs a full-rank covariance and got rank "
                f"{np.linalg.matrix_rank(cov.covariance_)} for {d} features on "
                f"{X_proc.shape[0]} rows. sklearn's precision_ is a "
                f"pseudo-inverse, so this would have returned a plausible "
                f"matrix with no contrast rather than failing.")
        VI = np.linalg.inv(cov.covariance_)
        D = cdist(X_proc, X_proc, metric='mahalanobis', VI=VI)
    elif metric == 'manhattan':
        D = cdist(X_proc, X_proc, metric='cityblock', **scipy_kw)
    elif metric == 'chebyshev' and scipy_kw:
        # scipy's weighted chebyshev is the p->inf limit of weighted minkowski,
        # where w**(1/p) -> 1, so it reduces to max((w > 0) * |u - v|) -- a 0/1
        # mask with the magnitudes thrown away. fold_weights already folded w
        # into the features, so never pass w= here.
        raise AssertionError("chebyshev must not receive scipy w=; see fold_weights")
    elif metric == 'minkowski':
        D = cdist(X_proc, X_proc, metric='minkowski', p=p, **scipy_kw)
    else:
        D = cdist(X_proc, X_proc, metric=metric, **scipy_kw)

    return D


def get_k_neighbors_from_distances(D, k, higher_is_similar=False):
    """
    Get k nearest neighbor indices and distances from precomputed distance matrix.

    Args:
        D: Square matrix (n_samples, n_samples). By default interpreted as a
           DISTANCE matrix (smaller = more similar). If ``higher_is_similar``
           is True, D is a SIMILARITY matrix (larger = more similar, e.g.
           Tanimoto/cosine similarity) and is converted to a distance via
           ``1 - D`` before ranking so neighbour selection is not inverted.
        k: Number of neighbors (excluding self)
        higher_is_similar: Set True when D holds similarities rather than
           distances.

    Returns:
        indices: (n_samples, k) neighbor indices
        distances: (n_samples, k) neighbor distances (distance-like even when
           a similarity matrix was passed, so downstream 1/distance weighting
           stays valid)
    """
    n = D.shape[0]
    D_copy = D.copy()
    if higher_is_similar:
        # Convert similarity -> distance so smaller = more similar.
        D_copy = 1.0 - D_copy
    np.fill_diagonal(D_copy, np.inf)

    indices = np.argsort(D_copy, axis=1)[:, :k]
    distances = np.take_along_axis(D_copy, indices, axis=1)

    return indices, distances


# ============================================================================
# SIMILARITY-PRESERVATION OBJECTIVE (SAC)
# ============================================================================
# "Two similar molecules have similar properties": the Spearman correlation
# between pairwise rep-space distance and pairwise |delta_y|. Higher = the
# representation geometry preserves the structure-activity relationship.
#
# This is the objective used by find_best_hybrid's similarity search
# (objective='similarity_sac'). It is a numerically-faithful mirror of
# tests/experiments/similarity/similarity_helpers.similarity_activity_correlation_scalable
# — kept here (in src) so the engine does not import test code. Same
# standardized-Euclidean metric, same sampling, same seed. If you change one,
# change the other.

def _sample_offdiag_pairs(n, n_sample=5000, seed=42):
    """Random distinct off-diagonal (i != j) index pairs WITHOUT enumerating
    all O(n^2) of them (np.triu_indices itself OOMs above ~15k)."""
    rng = np.random.RandomState(seed)
    pi = rng.randint(0, n, size=n_sample * 2)
    pj = rng.randint(0, n, size=n_sample * 2)
    mask = pi != pj
    return pi[mask][:n_sample], pj[mask][:n_sample]


def similarity_activity_correlation_scalable(X, y, n_sample=5000, is_clf=False,
                                             seed=42):
    """SAC over a random sample of pairs (no full n x n matrix).

    Standardizes X, samples ~n_sample off-diagonal pairs, and returns the
    Spearman rho between their Euclidean distance and activity difference
    (|delta_y| for regression/binary, class-mismatch indicator for is_clf).
    Higher rho = better similarity preservation. Returns 0.0 when the sample
    is degenerate (all-equal distances or labels make rho NaN).
    """
    from scipy.stats import spearmanr
    n = len(y)
    if n < 3:
        return 0.0
    pi, pj = _sample_offdiag_pairs(n, n_sample, seed)
    if len(pi) == 0:
        return 0.0
    Xs = StandardScaler().fit_transform(X)
    dists = np.linalg.norm(Xs[pi] - Xs[pj], axis=1)
    if is_clf:
        delta_y = (y[pi] != y[pj]).astype(float)
    else:
        delta_y = np.abs(y[pi] - y[pj])
    rho, _ = spearmanr(dists, delta_y)
    if rho is None or not np.isfinite(rho):
        return 0.0
    return float(rho)


def activity_readacross(X, y, groups, k=5, seed=42, missing_groups='raise',
                        degenerate=0.0, is_clf=False):
    """Activity read-across: how well do near neighbours predict activity?

    Builds the pairwise Euclidean distance matrix on the (already standardised)
    block, predicts each sample's activity from its k nearest neighbours under a
    scaffold-grouped split, and returns the Spearman rho between predicted and
    true activity. Higher is better. No model is fit on the features.

    ★ CLASSIFICATION (is_clf=True) returns 2*(AUROC - 0.5) instead. On a binary
    label the neighbour prediction is the distance-weighted fraction of the k
    neighbours in class 1 -- a graded number, so AUROC reads it with no cut-off
    invented. Spearman between a graded score and a 0/1 truth IS AUROC up to a
    monotone rescale, so this REORDERS NOTHING; measured on herg over 45
    candidate spaces, rank agreement was rho=+1.000 and 2 of 990 pairwise
    orderings differed, both ties. What it fixes is the zero point: a
    correlation sits at 0 at chance and AUROC at 0.5, and these scores are what
    the budget split divides by and what a blend mixes with a prediction score.
    The prediction branch of hybrid._score_component already uses this exact
    form. Decided 2026-09-09 over MODI, which is the published measurement
    (Golbraikh/Muratov/Tropsha, JCIM 2014) but discards the graded fraction for
    a hard 1-nearest-neighbour class match and reordered ~9% of pairs for no
    measured gain.

    `groups` are the scaffold labels, one per row. Read-across is only honest
    when neighbours come from a different scaffold, so groups are REQUIRED.

    missing_groups controls what happens when groups is None:
      'raise' (default) -- raise ValueError. Use this anywhere the score feeds a
          decision. Returning a number when the split could not be made silently
          scores every candidate identically, which looks like a tie rather than
          a failure.
      'nan' -- return NaN, for a caller that reports the score and can show a
          gap in the table.

    Returns 0.0 for a genuinely degenerate slice (under 10 rows, fewer than 2
    scaffolds, or constant activity) -- those are real answers, not failures.

    degenerate controls what a degenerate slice returns. 0.0 (the default, and
    every existing caller's behaviour) reports it as a real zero correlation.
    'nan' reports it as undefined -- use that wherever the score RANKS
    representations against each other: 0.0 for all of them is a tie, ties give
    every representation weight zero, and the budget then splits equally under
    whatever allocation name the run was launched with.
    """
    _degen = float('nan') if degenerate == 'nan' else float(degenerate)
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsRegressor

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if groups is None:
        if missing_groups == 'raise':
            raise ValueError(
                "activity_readacross needs scaffold groups. Without them the "
                "neighbours come from anywhere, which is not read-across. "
                "Pass groups, or pass missing_groups='nan' if the caller only "
                "reports the number and never decides on it.")
        return float('nan')
    if n < 10 or X.shape[0] != n:
        return _degen
    g = np.asarray(groups).ravel()
    n_groups = len(np.unique(g))
    if n_groups < 2:
        return _degen
    try:
        G = X @ X.T
        sq = np.diag(G)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0))
        preds = np.full(n, np.nan)
        for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(
                np.arange(n), y, g):
            kk = min(k, len(tr))
            if kk < 1:
                continue
            m = KNeighborsRegressor(n_neighbors=kk, weights='distance',
                                    metric='precomputed')
            m.fit(D[np.ix_(tr, tr)], y[tr])
            preds[te] = m.predict(D[np.ix_(te, tr)])
        ok = np.isfinite(preds)
        if ok.sum() < 3 or np.std(y[ok]) == 0 or np.std(preds[ok]) == 0:
            return _degen
        if is_clf:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y[ok])) != 2:
                # Multiclass: AUROC needs a one-vs-rest scheme this scorer does
                # not define, and the neighbour prediction is a mean of class
                # indices, which is not a score for any class. Degenerate is the
                # honest answer, not a number.
                return _degen
            auc = roc_auc_score(y[ok].astype(int), preds[ok])
            return _degen if not np.isfinite(auc) else float(2.0 * (auc - 0.5))
        rho, _ = spearmanr(preds[ok], y[ok])
        return _degen if not np.isfinite(rho) else float(rho)
    except Exception:
        return _degen


# ============================================================================
# DISTANCE-BASED METHODS
# ============================================================================

class KNNConsistencyFilter(NoiseMitigationMethod):
    """
    Flag samples where k-nearest neighbors have inconsistent labels.
    
    Parameters:
    -----------
    k : int
        Number of neighbors to consider
    threshold : float
        Disagreement threshold (std dev of neighbor labels)
    precomputed_distances : np.ndarray or None
        Precomputed distance matrix (n_samples, n_samples)
    distance_metric : str
        'euclidean', 'manhattan', 'cosine' (only used if precomputed_distances is None)
    feature_weights : array-like or None
        Weights for features in distance calculation
    """
    
    def __init__(self, k=5, threshold=None, precomputed_distances=None,
                 distance_metric='euclidean', feature_weights=None,
                 higher_is_similar=False, random_state=42):
        super().__init__(random_state)
        self.k = k
        self.threshold = threshold
        self.precomputed_distances = precomputed_distances
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        self.higher_is_similar = higher_is_similar

    def identify_noise(self, X, y):
        k = min(self.k, len(y) - 1)

        if self.precomputed_distances is not None:
            # Use precomputed distances
            indices, _ = get_k_neighbors_from_distances(
                self.precomputed_distances, k,
                higher_is_similar=self.higher_is_similar)
        else:
            # Compute distances on the fly
            if self.feature_weights is not None:
                X_weighted = X * self.feature_weights
            else:
                X_weighted = X
            
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric=self.distance_metric)
            nbrs.fit(X_weighted)
            _, indices = nbrs.kneighbors(X_weighted)
            indices = indices[:, 1:]  # Exclude self
        
        # For each sample, check label consistency with neighbors
        neighbor_labels = y[indices]

        # A two-class label needs a different statistic. The spread of the
        # neighbours' labels answers 'is this neighbourhood mixed', and on a
        # 0/1 label that is not the question: a molecule sitting in a pure
        # neighbourhood of the OTHER class — the clearest flipped label there
        # is — has neighbour spread 0 and was never flagged. The spread also
        # takes at most k/2 + 1 distinct values on a two-class label, so its
        # 90th percentile lands on the largest value that occurs and the
        # strict `>` below selected nothing at all. MEASURED 2026-09-18 on 200
        # rows with 20 labels flipped: this filter flagged 0 rows at k=5 and 0
        # at k=10, while prediction_error flagged 19 and bootstrap_ensemble 20.
        #
        # For two classes the statistic is the row's disagreement with its own
        # neighbourhood: the fraction of its k nearest neighbours carrying the
        # other label. Flagging above one half is the edited-nearest-neighbour
        # rule. A continuous label keeps the neighbour spread and the 90th
        # percentile exactly as before — nothing about the regression path
        # changes.
        _classes = np.unique(y)
        if len(_classes) == 2:
            disagreement = (neighbor_labels != y[:, None]).mean(axis=1)
            threshold = 0.5 if self.threshold is None else self.threshold
            return np.where(disagreement > threshold)[0]

        neighbor_std = np.std(neighbor_labels, axis=1)
        
        # Auto-threshold if not provided (per-call local so a reused instance
        # does not silently carry a prior dataset's threshold)
        threshold = self.threshold
        if threshold is None:
            threshold = np.percentile(neighbor_std, 90)

        # Flag high-disagreement samples
        noisy_indices = np.where(neighbor_std > threshold)[0]

        return noisy_indices


class LOFFilter(NoiseMitigationMethod):
    """
    Local Outlier Factor for noise detection.
    
    Parameters:
    -----------
    n_neighbors : int
        Number of neighbors for LOF
    contamination : float
        Expected proportion of outliers
    precomputed_distances : np.ndarray or None
        Precomputed distance matrix
    distance_metric : str
        Distance metric (only used if precomputed_distances is None)
    feature_weights : array-like or None
        Weights for features
    """
    
    def __init__(self, n_neighbors=20, contamination=0.1, precomputed_distances=None,
                 distance_metric='euclidean', feature_weights=None,
                 higher_is_similar=False, random_state=42):
        super().__init__(random_state)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.precomputed_distances = precomputed_distances
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        self.higher_is_similar = higher_is_similar

    def identify_noise(self, X, y):
        k = min(self.n_neighbors, len(y) - 1)

        if self.precomputed_distances is not None:
            # LOF with metric='precomputed' assumes DISTANCES (smaller = closer);
            # convert a similarity matrix so neighbours are not inverted.
            precomputed = self.precomputed_distances
            if self.higher_is_similar:
                precomputed = 1.0 - precomputed
            lof = LocalOutlierFactor(
                n_neighbors=k,
                metric='precomputed',
                contamination=self.contamination,
                novelty=False
            )
            labels = lof.fit_predict(precomputed)
        else:
            if self.feature_weights is not None:
                X_weighted = X * self.feature_weights
            else:
                X_weighted = X
            
            lof = LocalOutlierFactor(
                n_neighbors=k,
                metric=self.distance_metric,
                contamination=self.contamination,
                novelty=False
            )
            labels = lof.fit_predict(X_weighted)
        
        noisy_indices = np.where(labels == -1)[0]
        return noisy_indices


class DistanceWeightedPredictionFilter(NoiseMitigationMethod):
    """
    Predict each sample as distance-weighted average of neighbors.
    Flag samples with high prediction error.
    
    Parameters:
    -----------
    k : int
        Number of neighbors
    threshold : float
        Error threshold in std units
    precomputed_distances : np.ndarray or None
        Precomputed distance matrix
    distance_metric : str
        Distance metric (only used if precomputed_distances is None)
    feature_weights : array-like or None
        Weights for features
    """
    
    def __init__(self, k=15, threshold=2.0, precomputed_distances=None,
                 distance_metric='euclidean', feature_weights=None,
                 higher_is_similar=False, random_state=42):
        super().__init__(random_state)
        self.k = k
        self.threshold = threshold
        self.precomputed_distances = precomputed_distances
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        self.higher_is_similar = higher_is_similar

    def identify_noise(self, X, y):
        k = min(self.k, len(y) - 1)

        if self.precomputed_distances is not None:
            indices, distances = get_k_neighbors_from_distances(
                self.precomputed_distances, k,
                higher_is_similar=self.higher_is_similar)
        else:
            if self.feature_weights is not None:
                X_weighted = X * self.feature_weights
            else:
                X_weighted = X
            
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric=self.distance_metric)
            nbrs.fit(X_weighted)
            distances, indices = nbrs.kneighbors(X_weighted)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        
        # Distance-weighted prediction
        epsilon = 1e-10
        weights = 1.0 / (distances + epsilon)
        neighbor_labels = y[indices]
        
        y_pred = np.sum(weights * neighbor_labels, axis=1) / np.sum(weights, axis=1)
        
        # Flag high error
        errors = np.abs(y - y_pred)
        threshold_value = self.threshold * np.std(y)
        
        noisy_indices = np.where(errors > threshold_value)[0]
        return noisy_indices


class NeighborhoodConsensusFilter(NoiseMitigationMethod):
    """
    Activity-Cliff-Aware noise detection.
    
    Key insight:
    - NOISE: Sample disagrees with neighbors, BUT neighbors agree with each other
    - ACTIVITY CLIFF: Sample disagrees with neighbors, AND neighbors also disagree
    
    Only flags as noise if sample disagrees AND neighborhood is consistent.
    
    Parameters:
    -----------
    k : int
        Number of neighbors
    disagreement_threshold : float
        Threshold for sample-to-neighborhood disagreement (in std units)
    consistency_threshold : float
        Threshold for neighborhood consistency (in std units)
    precomputed_distances : np.ndarray or None
        Precomputed distance matrix
    distance_metric : str
        Distance metric (only used if precomputed_distances is None)
    feature_weights : array-like or None
        Weights for features
    """
    
    def __init__(self, k=10, disagreement_threshold=2.0, consistency_threshold=1.0,
                 precomputed_distances=None, distance_metric='euclidean',
                 feature_weights=None, higher_is_similar=False, random_state=42):
        super().__init__(random_state)
        self.k = k
        self.disagreement_threshold = disagreement_threshold
        self.consistency_threshold = consistency_threshold
        self.precomputed_distances = precomputed_distances
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        self.higher_is_similar = higher_is_similar
        self._is_activity_cliff = None

    def identify_noise(self, X, y):
        k = min(self.k, len(y) - 1)
        y_std = np.std(y) + 1e-10

        if self.precomputed_distances is not None:
            indices, _ = get_k_neighbors_from_distances(
                self.precomputed_distances, k,
                higher_is_similar=self.higher_is_similar)
        else:
            if self.feature_weights is not None:
                X_weighted = X * self.feature_weights
            else:
                X_weighted = X
            
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric=self.distance_metric)
            nbrs.fit(X_weighted)
            _, indices = nbrs.kneighbors(X_weighted)
            indices = indices[:, 1:]
        
        neighbor_labels = y[indices]
        
        # Sample-to-neighborhood disagreement
        y_pred = np.mean(neighbor_labels, axis=1)
        sample_disagreement = np.abs(y - y_pred) / y_std
        
        # Neighborhood internal consistency
        neighbor_variance = np.std(neighbor_labels, axis=1) / y_std
        neighborhood_consistent = neighbor_variance < self.consistency_threshold
        
        # Flag as noise only if: high disagreement AND consistent neighborhood
        high_disagreement = sample_disagreement > self.disagreement_threshold
        noise_mask = high_disagreement & neighborhood_consistent
        
        # Track activity cliffs
        self._is_activity_cliff = high_disagreement & ~neighborhood_consistent
        
        noisy_indices = np.where(noise_mask)[0]
        return noisy_indices
    
    def get_activity_cliff_indices(self):
        """Returns indices of samples that are likely activity cliffs (not noise)."""
        if self._is_activity_cliff is None:
            raise ValueError("Must call identify_noise first")
        return np.where(self._is_activity_cliff)[0]


class ActivityCliffDetector(NoiseMitigationMethod):
    """
    Detect activity cliffs: structurally similar pairs with large activity differences.
    
    Parameters:
    -----------
    similarity_threshold : float
        Maximum distance to consider as "similar"
    activity_threshold : float
        Minimum activity difference to consider as "cliff"
    precomputed_distances : np.ndarray or None
        Precomputed distance matrix
    distance_metric : str
        Distance metric for structural similarity
    feature_weights : array-like or None
        Weights for features in distance calculation
    """
    
    def __init__(self, similarity_threshold=None, activity_threshold=None,
                 precomputed_distances=None, distance_metric='euclidean',
                 feature_weights=None, higher_is_similar=False, random_state=42):
        super().__init__(random_state)
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.precomputed_distances = precomputed_distances
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        self.higher_is_similar = higher_is_similar

    def identify_noise(self, X, y):
        if self.precomputed_distances is not None:
            # `distances < similarity_threshold` assumes DISTANCES; convert a
            # similarity matrix so "structurally similar" is not inverted.
            if self.higher_is_similar:
                distances = 1.0 - self.precomputed_distances
            else:
                distances = self.precomputed_distances
        else:
            # Apply feature weighting
            if self.feature_weights is not None:
                X_weighted = X * self.feature_weights
            else:
                X_weighted = X
            
            # Compute pairwise distances
            if self.distance_metric == 'euclidean':
                distances = cdist(X_weighted, X_weighted, metric='euclidean')
            elif self.distance_metric == 'manhattan':
                distances = cdist(X_weighted, X_weighted, metric='cityblock')
            elif self.distance_metric == 'cosine':
                distances = cdist(X_weighted, X_weighted, metric='cosine')
            else:
                distances = cdist(X_weighted, X_weighted, metric=self.distance_metric)
        
        # Auto-threshold if not provided (per-call locals so a reused instance
        # does not silently carry a prior dataset's thresholds)
        similarity_threshold = self.similarity_threshold
        if similarity_threshold is None:
            non_diag = distances[np.triu_indices_from(distances, k=1)]
            similarity_threshold = np.percentile(non_diag, 10)
        
        # Multi-class: use binary indicator (same/different class) instead of |delta_y|
        n_classes = len(np.unique(y))
        nominal = n_classes > 2 and np.allclose(y, y.astype(int))

        activity_threshold = self.activity_threshold
        if activity_threshold is None:
            if nominal:
                activity_threshold = 0.5  # indicator is 0 or 1
            else:
                activity_diffs = np.abs(y[:, None] - y[None, :])
                activity_threshold = np.percentile(activity_diffs, 90)

        # Find activity cliffs
        if nominal:
            delta_y = (y[:, None] != y[None, :]).astype(float)
        else:
            delta_y = np.abs(y[:, None] - y[None, :])
        cliff_mask = (distances < similarity_threshold) & \
                     (delta_y > activity_threshold)
        
        # Count cliff involvement per sample
        cliff_counts = cliff_mask.sum(axis=1)
        
        # Flag samples involved in cliffs
        noisy_indices = np.where(cliff_counts > 0)[0]
        
        return noisy_indices


class MahalanobisOutlierDetector(NoiseMitigationMethod):
    """
    Detect outliers using Mahalanobis distance.
    
    Parameters:
    -----------
    contamination : float
        Proportion of outliers expected
    feature_weights : array-like or None
        Weights for features (applied before covariance estimation)
    """
    
    def __init__(self, contamination=0.1, feature_weights=None, random_state=42):
        super().__init__(random_state)
        self.contamination = contamination
        self.feature_weights = feature_weights
        
    def identify_noise(self, X, y):
        # feature_weights has no effect here and cannot be given one without
        # changing the method. Multiplying feature j by w_j and then running
        # StandardScaler divides it by w_j again, and EmpiricalCovariance
        # re-estimates on what is left, so the Mahalanobis distance is invariant
        # to any per-feature scaling. Measured on 300 rows and 12 features with a
        # non-uniform weight vector: the weighted and unweighted arms flag the
        # identical 30 rows. Kept so the argument stays accepted, and the sweep
        # that used to pair it with weighted=True now drops that cell and says so
        # (tests/noise_mitigation_advanced.py). Noted 2026-09-10 (metric-direction
        # audit).
        if self.feature_weights is not None:
            X_weighted = X * self.feature_weights
        else:
            X_weighted = X
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_weighted)
        
        # Fit covariance
        cov = EmpiricalCovariance().fit(X_scaled)
        
        # Compute Mahalanobis distances
        mahal_dist = cov.mahalanobis(X_scaled)
        
        # Flag top contamination% as outliers
        threshold = np.percentile(mahal_dist, (1 - self.contamination) * 100)
        noisy_indices = np.where(mahal_dist > threshold)[0]
        
        return noisy_indices


# ============================================================================
# ENSEMBLE-BASED METHODS
# ============================================================================

class CVDisagreementFilter(NoiseMitigationMethod):
    """
    Cross-validation disagreement: train on fold A, flag disagreements in fold B.
    
    Parameters:
    -----------
    n_folds : int
        Number of CV folds
    disagreement_threshold : float
        Absolute error threshold to flag as noisy
    """
    
    def __init__(self, n_folds=5, disagreement_threshold=None, groups=None,
                 random_state=42):
        super().__init__(random_state)
        self.n_folds = n_folds
        self.disagreement_threshold = disagreement_threshold
        self.groups = groups

    def identify_noise(self, X, y):
        # NO random splits: use scaffold-GROUP-aware folds when groups are
        # available, otherwise a DETERMINISTIC (unshuffled) KFold.
        # TODO(group-plumbing): thread `groups` (scaffold labels) from the
        # caller so this falls into the GroupKFold branch — the deterministic
        # KFold fallback is contiguous, not scaffold-aware.
        if self.groups is not None:
            n_splits = min(self.n_folds, len(np.unique(self.groups)))
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(X, y, groups=self.groups)
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=False)
            splits = kf.split(X)

        prediction_errors = np.zeros(len(y))

        for train_idx, val_idx in splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model on fold
            model = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Predict on validation fold
            y_pred = model.predict(X_val)
            
            # Store errors
            prediction_errors[val_idx] = np.abs(y_val - y_pred)
        
        # Auto-threshold if not provided (per-call local so a reused instance
        # does not silently carry a prior dataset's threshold)
        disagreement_threshold = self.disagreement_threshold
        if disagreement_threshold is None:
            disagreement_threshold = np.percentile(prediction_errors, 90)

        # Flag high-error samples
        noisy_indices = np.where(prediction_errors > disagreement_threshold)[0]

        return noisy_indices


class BootstrapEnsembleFilter(NoiseMitigationMethod):
    """
    Bootstrap ensemble voting: flag samples where ensemble disagrees with label.
    
    Parameters:
    -----------
    n_estimators : int
        Number of bootstrap models
    disagreement_criterion : str
        'mean_error' or 'vote_variance'
    threshold : float
        Threshold for flagging
    """
    
    def __init__(self, n_estimators=10, disagreement_criterion='mean_error',
                 threshold=None, random_state=42):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.disagreement_criterion = disagreement_criterion
        self.threshold = threshold
        
    def identify_noise(self, X, y):
        rng = np.random.RandomState(self.random_state)

        # Bootstrap predictions
        predictions = np.zeros((len(y), self.n_estimators))
        has_prediction = np.zeros((len(y), self.n_estimators), dtype=bool)

        for i in range(self.n_estimators):
            # Bootstrap sample
            bootstrap_idx = rng.choice(len(y), size=len(y), replace=True)
            oob_idx = np.setdiff1d(np.arange(len(y)), bootstrap_idx)

            if len(oob_idx) == 0:
                continue

            X_boot, y_boot = X[bootstrap_idx], y[bootstrap_idx]

            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=i, n_jobs=-1)
            model.fit(X_boot, y_boot)

            # Predict on OOB samples
            predictions[oob_idx, i] = model.predict(X[oob_idx])
            has_prediction[oob_idx, i] = True

        # Calculate disagreement
        if self.disagreement_criterion == 'mean_error':
            # Average prediction error across ensemble
            n_preds = has_prediction.sum(axis=1)
            masked_preds = np.where(has_prediction, predictions, 0.0)
            mean_pred = np.where(n_preds > 0,
                                masked_preds.sum(axis=1) / n_preds,
                                y)  # Use true label if no predictions
            disagreement = np.abs(y - mean_pred)
        elif self.disagreement_criterion == 'vote_variance':
            # Variance of predictions (only over OOB predictions)
            n_preds = has_prediction.sum(axis=1)
            masked_preds = np.where(has_prediction, predictions, 0.0)
            mean_pred = np.where(n_preds > 0, masked_preds.sum(axis=1) / n_preds, 0.0)
            sq_diff = np.where(has_prediction, (predictions - mean_pred[:, None]) ** 2, 0.0)
            disagreement = np.where(n_preds > 1,
                                   sq_diff.sum(axis=1) / n_preds,
                                   0)
        else:
            raise ValueError(f"Unknown criterion: {self.disagreement_criterion}")
        
        # Auto-threshold (per-call local so a reused instance does not
        # silently carry a prior dataset's threshold)
        threshold = self.threshold
        if threshold is None:
            threshold = np.percentile(disagreement, 90)

        # Flag high-disagreement samples
        noisy_indices = np.where(disagreement > threshold)[0]

        return noisy_indices


# ============================================================================
# MODEL-BASED METHODS (ADAPTED FROM NOISY LABEL LEARNING)
# ============================================================================

class CoTeachingFilter(NoiseMitigationMethod):
    """
    Co-Teaching adapted to mitigation: extract consistently rejected samples.
    
    Parameters:
    -----------
    n_epochs : int
        Number of training epochs
    selection_rate : float
        Fraction of small-loss samples to select per epoch
    """
    
    def __init__(self, n_epochs=20, selection_rate=0.7, random_state=42):
        super().__init__(random_state)
        self.n_epochs = n_epochs
        self.selection_rate = selection_rate
        
    def identify_noise(self, X, y):
        rng = np.random.RandomState(self.random_state)

        # Track rejection counts
        rejection_counts = np.zeros(len(y))
        n_select = int(len(y) * self.selection_rate)

        # Initial training sets: full data for both models
        idx1 = np.arange(len(y))
        idx2 = np.arange(len(y))

        for epoch in range(self.n_epochs):
            # Train each model on the subset selected by the OTHER model
            model1 = RandomForestRegressor(n_estimators=50, random_state=self.random_state + epoch, n_jobs=-1)
            model2 = RandomForestRegressor(n_estimators=50, random_state=self.random_state + epoch + 1000, n_jobs=-1)

            model1.fit(X[idx2], y[idx2])  # model1 trained on model2's selection
            model2.fit(X[idx1], y[idx1])  # model2 trained on model1's selection

            # Each model scores ALL samples, then selects small-loss for the other
            loss1 = np.abs(y - model1.predict(X))
            loss2 = np.abs(y - model2.predict(X))

            # Model1's small-loss selection feeds model2, and vice versa
            idx2 = np.argsort(loss1)[:n_select]  # model1 selects for model2
            idx1 = np.argsort(loss2)[:n_select]  # model2 selects for model1

            # Track which samples were rejected by both
            rejected1 = np.setdiff1d(np.arange(len(y)), idx1)
            rejected2 = np.setdiff1d(np.arange(len(y)), idx2)

            rejection_counts[rejected1] += 1
            rejection_counts[rejected2] += 1

        # Flag samples rejected in majority of epochs
        threshold = self.n_epochs * 0.6  # Rejected in >60% of epochs
        noisy_indices = np.where(rejection_counts >= threshold)[0]

        return noisy_indices


class DivideMixFilter(NoiseMitigationMethod):
    """
    DivideMix adapted to mitigation: use GMM on losses to identify noisy samples.
    
    Parameters:
    -----------
    n_components : int
        Number of GMM components (2 for clean/noisy modes)
    """
    
    def __init__(self, n_components=2, random_state=42):
        super().__init__(random_state)
        self.n_components = n_components
        
    def identify_noise(self, X, y):
        # Train model with OOB to get unbiased loss estimates
        model = RandomForestRegressor(n_estimators=100, oob_score=True,
                                      random_state=self.random_state, n_jobs=-1)
        model.fit(X, y)

        # Use OOB predictions to avoid in-sample bias
        pred = model.oob_prediction_
        losses = np.abs(y - pred)
        
        # Fit GMM to loss distribution
        gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        gmm.fit(losses.reshape(-1, 1))
        
        # Predict components
        components = gmm.predict(losses.reshape(-1, 1))
        
        # Identify noisy mode (higher mean loss)
        mean_losses = [losses[components == i].mean() for i in range(self.n_components)]
        noisy_component = np.argmax(mean_losses)
        
        # Flag samples in noisy mode
        noisy_indices = np.where(components == noisy_component)[0]
        
        return noisy_indices


# ============================================================================
# SIMPLE BASELINES
# ============================================================================

class ZScoreFilter(NoiseMitigationMethod):
    """
    Remove samples with extreme label values (outliers in label space).
    
    Parameters:
    -----------
    threshold : float
        Z-score threshold (typically 3.0)
    """
    
    def __init__(self, threshold=3.0, random_state=42):
        super().__init__(random_state)
        self.threshold = threshold
        
    def identify_noise(self, X, y):
        # Calculate z-scores (handle constant labels gracefully)
        std = y.std()
        if std < 1e-10:
            return np.array([], dtype=int)
        z_scores = np.abs((y - y.mean()) / std)

        # Flag extreme values
        noisy_indices = np.where(z_scores > self.threshold)[0]

        return noisy_indices


class PredictionErrorFilter(NoiseMitigationMethod):
    """
    Train model on noisy data, flag samples with highest prediction errors.
    
    Parameters:
    -----------
    contamination : float
        Fraction of data to flag as noisy
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        super().__init__(random_state)
        self.contamination = contamination
        
    def identify_noise(self, X, y):
        # Train model with OOB to get unbiased error estimates
        model = RandomForestRegressor(n_estimators=100, oob_score=True,
                                      random_state=self.random_state, n_jobs=-1)
        model.fit(X, y)

        # Use OOB predictions to avoid in-sample bias
        pred = model.oob_prediction_
        errors = np.abs(y - pred)
        
        # Flag top contamination% errors
        threshold = np.percentile(errors, (1 - self.contamination) * 100)
        noisy_indices = np.where(errors > threshold)[0]
        
        return noisy_indices


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

# Methods that accept precomputed_distances
DISTANCE_AWARE_METHODS = {
    'knn_k5', 'knn_k10', 'knn_k20',
    'lof', 'lof_k10', 'lof_k20',
    'distance_weighted',
    'neighborhood_consensus',
    'activity_cliffs',
}


def get_mitigation_method(method_name, **kwargs):
    """
    Factory function to get mitigation method by name.
    
    Parameters:
    -----------
    method_name : str
        Name of method (e.g., 'knn_k5', 'activity_cliffs', etc.)
    **kwargs : dict
        Method-specific parameters
    
    Returns:
    --------
    method : NoiseMitigationMethod instance
    """
    
    methods = {
        # Distance-based (accept precomputed_distances)
        'knn_k5': lambda **kw: KNNConsistencyFilter(k=5, **kw),
        'knn_k10': lambda **kw: KNNConsistencyFilter(k=10, **kw),
        'knn_k20': lambda **kw: KNNConsistencyFilter(k=20, **kw),
        'lof': lambda **kw: LOFFilter(**kw),
        'lof_k10': lambda **kw: LOFFilter(n_neighbors=10, **kw),
        'lof_k20': lambda **kw: LOFFilter(n_neighbors=20, **kw),
        'distance_weighted': lambda **kw: DistanceWeightedPredictionFilter(**kw),
        'neighborhood_consensus': lambda **kw: NeighborhoodConsensusFilter(**kw),
        'activity_cliffs': lambda **kw: ActivityCliffDetector(**kw),
        
        # Other distance methods (don't use precomputed)
        'mahalanobis': lambda **kw: MahalanobisOutlierDetector(**kw),
        
        # Ensemble-based
        'cv_disagreement': lambda **kw: CVDisagreementFilter(**kw),
        'bootstrap_ensemble': lambda **kw: BootstrapEnsembleFilter(**kw),
        
        # Model-based
        'co_teaching': lambda **kw: CoTeachingFilter(**kw),
        'dividemix': lambda **kw: DivideMixFilter(**kw),
        
        # Baselines
        'zscore': lambda **kw: ZScoreFilter(**kw),
        'prediction_error': lambda **kw: PredictionErrorFilter(**kw),
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(methods.keys())}")
    
    # Filter kwargs based on what the method accepts
    if method_name in DISTANCE_AWARE_METHODS:
        return methods[method_name](**kwargs)
    else:
        # Filter out distance-specific kwargs for non-distance methods
        filtered_kwargs = {k: v for k, v in kwargs.items()
                         if k not in ['precomputed_distances', 'distance_metric',
                                      'higher_is_similar']}
        return methods[method_name](**filtered_kwargs)