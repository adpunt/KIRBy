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
from sklearn.model_selection import KFold
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

def compute_distance_matrix(X, metric='euclidean', feature_weights=None, standardize=True):
    """
    Compute pairwise distance matrix using scipy.cdist.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        metric: Distance metric (euclidean, manhattan, cosine, chebyshev, 
                canberra, correlation, braycurtis, mahalanobis)
        feature_weights: Optional weights for features
        standardize: Whether to standardize features first
    
    Returns:
        D: Distance matrix (n_samples, n_samples)
    """
    X_proc = X.copy()
    
    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)
    
    if feature_weights is not None:
        X_proc = X_proc * np.sqrt(feature_weights)
    
    # Handle special metrics
    if metric == 'mahalanobis':
        cov = EmpiricalCovariance().fit(X_proc)
        VI = np.linalg.inv(cov.covariance_)
        D = cdist(X_proc, X_proc, metric='mahalanobis', VI=VI)
    elif metric == 'manhattan':
        D = cdist(X_proc, X_proc, metric='cityblock')
    else:
        D = cdist(X_proc, X_proc, metric=metric)
    
    return D


def get_k_neighbors_from_distances(D, k):
    """
    Get k nearest neighbor indices and distances from precomputed distance matrix.
    
    Args:
        D: Square distance matrix (n_samples, n_samples)
        k: Number of neighbors (excluding self)
    
    Returns:
        indices: (n_samples, k) neighbor indices
        distances: (n_samples, k) neighbor distances
    """
    n = D.shape[0]
    D_copy = D.copy()
    np.fill_diagonal(D_copy, np.inf)
    
    indices = np.argsort(D_copy, axis=1)[:, :k]
    distances = np.take_along_axis(D_copy, indices, axis=1)
    
    return indices, distances


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
                 distance_metric='euclidean', feature_weights=None, random_state=42):
        super().__init__(random_state)
        self.k = k
        self.threshold = threshold
        self.precomputed_distances = precomputed_distances
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        
    def identify_noise(self, X, y):
        k = min(self.k, len(y) - 1)
        
        if self.precomputed_distances is not None:
            # Use precomputed distances
            indices, _ = get_k_neighbors_from_distances(self.precomputed_distances, k)
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
        neighbor_std = np.std(neighbor_labels, axis=1)
        
        # Auto-threshold if not provided
        if self.threshold is None:
            self.threshold = np.percentile(neighbor_std, 90)
        
        # Flag high-disagreement samples
        noisy_indices = np.where(neighbor_std > self.threshold)[0]
        
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
                 distance_metric='euclidean', feature_weights=None, random_state=42):
        super().__init__(random_state)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.precomputed_distances = precomputed_distances
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        
    def identify_noise(self, X, y):
        k = min(self.n_neighbors, len(y) - 1)
        
        if self.precomputed_distances is not None:
            lof = LocalOutlierFactor(
                n_neighbors=k,
                metric='precomputed',
                contamination=self.contamination,
                novelty=False
            )
            labels = lof.fit_predict(self.precomputed_distances)
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
                 distance_metric='euclidean', feature_weights=None, random_state=42):
        super().__init__(random_state)
        self.k = k
        self.threshold = threshold
        self.precomputed_distances = precomputed_distances
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        
    def identify_noise(self, X, y):
        k = min(self.k, len(y) - 1)
        
        if self.precomputed_distances is not None:
            indices, distances = get_k_neighbors_from_distances(self.precomputed_distances, k)
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
                 feature_weights=None, random_state=42):
        super().__init__(random_state)
        self.k = k
        self.disagreement_threshold = disagreement_threshold
        self.consistency_threshold = consistency_threshold
        self.precomputed_distances = precomputed_distances
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        self._is_activity_cliff = None
        
    def identify_noise(self, X, y):
        k = min(self.k, len(y) - 1)
        y_std = np.std(y) + 1e-10
        
        if self.precomputed_distances is not None:
            indices, _ = get_k_neighbors_from_distances(self.precomputed_distances, k)
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
                 feature_weights=None, random_state=42):
        super().__init__(random_state)
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.precomputed_distances = precomputed_distances
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        
    def identify_noise(self, X, y):
        if self.precomputed_distances is not None:
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
        
        # Auto-threshold if not provided
        if self.similarity_threshold is None:
            non_diag = distances[np.triu_indices_from(distances, k=1)]
            self.similarity_threshold = np.percentile(non_diag, 10)
        
        if self.activity_threshold is None:
            activity_diffs = np.abs(y[:, None] - y[None, :])
            self.activity_threshold = np.percentile(activity_diffs, 90)
        
        # Find activity cliffs
        cliff_mask = (distances < self.similarity_threshold) & \
                     (np.abs(y[:, None] - y[None, :]) > self.activity_threshold)
        
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
        # Apply feature weighting
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
    
    def __init__(self, n_folds=5, disagreement_threshold=None, random_state=42):
        super().__init__(random_state)
        self.n_folds = n_folds
        self.disagreement_threshold = disagreement_threshold
        
    def identify_noise(self, X, y):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        prediction_errors = np.zeros(len(y))
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model on fold
            model = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Predict on validation fold
            y_pred = model.predict(X_val)
            
            # Store errors
            prediction_errors[val_idx] = np.abs(y_val - y_pred)
        
        # Auto-threshold if not provided
        if self.disagreement_threshold is None:
            self.disagreement_threshold = np.percentile(prediction_errors, 90)
        
        # Flag high-error samples
        noisy_indices = np.where(prediction_errors > self.disagreement_threshold)[0]
        
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
        np.random.seed(self.random_state)
        
        # Bootstrap predictions
        predictions = np.zeros((len(y), self.n_estimators))
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            bootstrap_idx = np.random.choice(len(y), size=len(y), replace=True)
            oob_idx = np.setdiff1d(np.arange(len(y)), bootstrap_idx)
            
            if len(oob_idx) == 0:
                continue
            
            X_boot, y_boot = X[bootstrap_idx], y[bootstrap_idx]
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=i, n_jobs=-1)
            model.fit(X_boot, y_boot)
            
            # Predict on OOB samples
            predictions[oob_idx, i] = model.predict(X[oob_idx])
        
        # Calculate disagreement
        if self.disagreement_criterion == 'mean_error':
            # Average prediction error across ensemble
            mask = predictions != 0  # Only count predictions that were made
            mean_pred = np.where(mask.sum(axis=1) > 0,
                                predictions.sum(axis=1) / mask.sum(axis=1),
                                y)  # Use true label if no predictions
            disagreement = np.abs(y - mean_pred)
        elif self.disagreement_criterion == 'vote_variance':
            # Variance of predictions
            mask = predictions != 0
            disagreement = np.where(mask.sum(axis=1) > 1,
                                   np.var(predictions, axis=1),
                                   0)
        else:
            raise ValueError(f"Unknown criterion: {self.disagreement_criterion}")
        
        # Auto-threshold
        if self.threshold is None:
            self.threshold = np.percentile(disagreement, 90)
        
        # Flag high-disagreement samples
        noisy_indices = np.where(disagreement > self.threshold)[0]
        
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
        np.random.seed(self.random_state)
        
        # Track rejection counts
        rejection_counts = np.zeros(len(y))
        
        for epoch in range(self.n_epochs):
            # Train two models
            model1 = RandomForestRegressor(n_estimators=50, random_state=self.random_state + epoch, n_jobs=-1)
            model2 = RandomForestRegressor(n_estimators=50, random_state=self.random_state + epoch + 1000, n_jobs=-1)
            
            model1.fit(X, y)
            model2.fit(X, y)
            
            # Get losses
            pred1 = model1.predict(X)
            pred2 = model2.predict(X)
            
            loss1 = np.abs(y - pred1)
            loss2 = np.abs(y - pred2)
            
            # Select small-loss samples
            n_select = int(len(y) * self.selection_rate)
            
            idx1 = np.argsort(loss1)[:n_select]
            idx2 = np.argsort(loss2)[:n_select]
            
            # Track which samples were rejected
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
        # Train model to get loss estimates
        model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        model.fit(X, y)
        
        # Get prediction errors (proxy for loss)
        pred = model.predict(X)
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
        # Calculate z-scores
        z_scores = np.abs((y - y.mean()) / y.std())
        
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
        # Train model on all data
        model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        model.fit(X, y)
        
        # Get prediction errors
        pred = model.predict(X)
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
                         if k not in ['precomputed_distances', 'distance_metric']}
        return methods[method_name](**filtered_kwargs)