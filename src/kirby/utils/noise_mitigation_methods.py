#!/usr/bin/env python3
"""
Noise Mitigation Methods for QSAR
==================================

Implements techniques to identify and remove noisy labels from training data.

Categories:
1. Distance-based: k-NN, Activity Cliffs, Mahalanobis
2. Ensemble-based: CV disagreement, Bootstrap ensemble
3. Model-based: Co-Teaching, DivideMix (adapted)
4. Simple baselines: Z-score, Prediction error
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
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
    distance_metric : str
        'euclidean', 'manhattan', 'cosine'
    feature_weights : array-like or None
        Weights for features in distance calculation
    """
    
    def __init__(self, k=5, threshold=None, distance_metric='euclidean', 
                 feature_weights=None, random_state=42):
        super().__init__(random_state)
        self.k = k
        self.threshold = threshold
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        
    def identify_noise(self, X, y):
        # Apply feature weighting if provided
        if self.feature_weights is not None:
            X_weighted = X * self.feature_weights
        else:
            X_weighted = X
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric=self.distance_metric)
        nbrs.fit(X_weighted)
        distances, indices = nbrs.kneighbors(X_weighted)
        
        # For each sample, check label consistency with neighbors
        neighbor_std = np.zeros(len(y))
        for i in range(len(y)):
            neighbor_labels = y[indices[i, 1:]]  # Exclude self
            neighbor_std[i] = np.std(neighbor_labels)
        
        # Auto-threshold if not provided
        if self.threshold is None:
            self.threshold = np.percentile(neighbor_std, 90)
        
        # Flag high-disagreement samples
        noisy_indices = np.where(neighbor_std > self.threshold)[0]
        
        return noisy_indices


class ActivityCliffDetector(NoiseMitigationMethod):
    """
    Detect activity cliffs: structurally similar pairs with large activity differences.
    
    Parameters:
    -----------
    similarity_threshold : float
        Maximum distance to consider as "similar"
    activity_threshold : float
        Minimum activity difference to consider as "cliff"
    distance_metric : str
        Distance metric for structural similarity
    feature_weights : array-like or None
        Weights for features in distance calculation
    """
    
    def __init__(self, similarity_threshold=None, activity_threshold=None,
                 distance_metric='euclidean', feature_weights=None, random_state=42):
        super().__init__(random_state)
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights
        
    def identify_noise(self, X, y):
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
            raise ValueError(f"Unknown metric: {self.distance_metric}")
        
        # Auto-threshold if not provided
        if self.similarity_threshold is None:
            # Use 10th percentile of non-zero distances
            non_diag = distances[np.triu_indices_from(distances, k=1)]
            self.similarity_threshold = np.percentile(non_diag, 10)
        
        if self.activity_threshold is None:
            # Use 90th percentile of activity differences
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
        'knn_k5': lambda: KNNConsistencyFilter(k=5, **kwargs),
        'knn_k10': lambda: KNNConsistencyFilter(k=10, **kwargs),
        'knn_k20': lambda: KNNConsistencyFilter(k=20, **kwargs),
        'activity_cliffs': lambda: ActivityCliffDetector(**kwargs),
        'mahalanobis': lambda: MahalanobisOutlierDetector(**kwargs),
        'cv_disagreement': lambda: CVDisagreementFilter(**kwargs),
        'bootstrap_ensemble': lambda: BootstrapEnsembleFilter(**kwargs),
        'co_teaching': lambda: CoTeachingFilter(**kwargs),
        'dividemix': lambda: DivideMixFilter(**kwargs),
        'zscore': lambda: ZScoreFilter(**kwargs),
        'prediction_error': lambda: PredictionErrorFilter(**kwargs),
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}")
    
    return methods[method_name]()