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
from sklearn.model_selection import KFold, cross_val_predict
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
    
    def get_weights(self, X: np.ndarray, y: np.ndarray, 
                    strength: float = 1.0) -> np.ndarray:
        """
        Convert noise scores to sample weights.
        
        Args:
            strength: How aggressively to downweight noisy samples
        
        Returns:
            weights: (n_samples,) - lower for noisy samples
        """
        scores = self.estimate(X, y)
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


# ============================================================================
# NEIGHBOR-BASED ESTIMATORS
# ============================================================================

class NeighborResidualEstimator(NoiseEstimator):
    """
    Estimate noise as deviation from k-NN mean.
    
    Intuition: If your label differs from similar molecules, 
    you might have more measurement error.
    
    noise_score = |y - neighbor_mean| / scale
    """
    
    def __init__(self, k: int = 10, metric: str = 'cosine', 
                 normalize: bool = True, random_state: int = 42):
        super().__init__(random_state)
        self.k = k
        self.metric = metric
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = min(self.k, len(y) - 1)
        
        nn = NearestNeighbors(n_neighbors=k + 1, metric=self.metric)
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        
        # Exclude self
        neighbor_indices = indices[:, 1:]
        neighbor_labels = y[neighbor_indices]
        neighbor_mean = np.mean(neighbor_labels, axis=1)
        
        # Raw residual
        residuals = np.abs(y - neighbor_mean)
        
        if self.normalize:
            # Normalize by global std
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
    
    noise_score = |y - neighbor_mean| / neighbor_std
    
    High score means: you deviate from neighbors more than 
    neighbors deviate from each other.
    """
    
    def __init__(self, k: int = 10, metric: str = 'cosine', random_state: int = 42):
        super().__init__(random_state)
        self.k = k
        self.metric = metric
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = min(self.k, len(y) - 1)
        
        nn = NearestNeighbors(n_neighbors=k + 1, metric=self.metric)
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        
        neighbor_indices = indices[:, 1:]
        neighbor_labels = y[neighbor_indices]
        
        neighbor_mean = np.mean(neighbor_labels, axis=1)
        neighbor_std = np.std(neighbor_labels, axis=1)
        
        # Ratio of deviation to local spread
        deviation = np.abs(y - neighbor_mean)
        scores = deviation / (neighbor_std + 1e-10)
        
        self.noise_scores_ = scores
        self.neighbor_means_ = neighbor_mean
        self.neighbor_stds_ = neighbor_std
        return scores


class DistanceWeightedResidualEstimator(NoiseEstimator):
    """
    Like NeighborResidual, but weight closer neighbors more heavily.
    
    noise_score = |y - weighted_neighbor_mean|
    where weights ∝ 1/distance
    """
    
    def __init__(self, k: int = 10, metric: str = 'cosine', 
                 normalize: bool = True, random_state: int = 42):
        super().__init__(random_state)
        self.k = k
        self.metric = metric
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = min(self.k, len(y) - 1)
        
        nn = NearestNeighbors(n_neighbors=k + 1, metric=self.metric)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        # Exclude self
        neighbor_distances = distances[:, 1:]
        neighbor_indices = indices[:, 1:]
        neighbor_labels = y[neighbor_indices]
        
        # Inverse distance weighting
        weights = 1 / (neighbor_distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        weighted_mean = np.sum(neighbor_labels * weights, axis=1)
        residuals = np.abs(y - weighted_mean)
        
        if self.normalize:
            residuals = residuals / (np.std(y) + 1e-10)
        
        self.noise_scores_ = residuals
        self.weighted_means_ = weighted_mean
        return residuals


class ActivityCliffAwareEstimator(NoiseEstimator):
    """
    Distinguish noise from activity cliffs.
    
    Key insight:
    - NOISE: Sample deviates from neighbors, but neighbors agree with each other
    - ACTIVITY CLIFF: Sample deviates from neighbors, and neighbors also disagree
    
    noise_score = deviation * (1 - local_variance_normalized)
    
    This downweights deviations in high-variance regions (cliffs).
    """
    
    def __init__(self, k: int = 10, metric: str = 'cosine', random_state: int = 42):
        super().__init__(random_state)
        self.k = k
        self.metric = metric
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = min(self.k, len(y) - 1)
        global_std = np.std(y) + 1e-10
        
        nn = NearestNeighbors(n_neighbors=k + 1, metric=self.metric)
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        
        neighbor_indices = indices[:, 1:]
        neighbor_labels = y[neighbor_indices]
        
        neighbor_mean = np.mean(neighbor_labels, axis=1)
        neighbor_std = np.std(neighbor_labels, axis=1)
        
        # Sample-to-neighborhood deviation
        deviation = np.abs(y - neighbor_mean) / global_std
        
        # Neighborhood consistency (high = neighbors agree = likely noise not cliff)
        # Normalize neighbor_std by global_std
        local_variance_norm = neighbor_std / global_std
        consistency = 1 / (1 + local_variance_norm)  # High when neighbors agree
        
        # Noise score: deviation weighted by consistency
        # High deviation + high consistency = likely noise
        # High deviation + low consistency = likely activity cliff
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
                 random_state: int = 42):
        super().__init__(random_state)
        self.n_folds = n_folds
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Get CV predictions
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=self.random_state,
            n_jobs=-1
        )
        
        cv_pred = cross_val_predict(model, X, y, cv=self.n_folds)
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
                 random_state: int = 42):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        predictions = np.zeros((len(y), self.n_estimators))
        
        for i in range(self.n_estimators):
            # Different random states for diversity
            model = RandomForestRegressor(
                n_estimators=50,
                max_features='sqrt',
                random_state=self.random_state + i,
                n_jobs=-1
            )
            model.fit(X, y)
            predictions[:, i] = model.predict(X)
        
        # Prediction statistics
        pred_mean = predictions.mean(axis=1)
        pred_std = predictions.std(axis=1)
        
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
    """
    
    def __init__(self, normalize: bool = True, random_state: int = 42):
        super().__init__(random_state)
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Use OOB predictions as LOO approximation
        model = RandomForestRegressor(
            n_estimators=200,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X, y)
        
        # OOB predictions
        oob_pred = model.oob_prediction_
        residuals = np.abs(y - oob_pred)
        
        if self.normalize:
            residuals = residuals / (np.std(y) + 1e-10)
        
        self.noise_scores_ = residuals
        self.oob_predictions_ = oob_pred
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
                 random_state: int = 42):
        super().__init__(random_state)
        self.k = k
        self.metric = metric
        self.use_cv = use_cv
        self.use_variance_ratio = use_variance_ratio
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        estimators = []
        
        # Always use neighbor residual
        nr = NeighborResidualEstimator(k=self.k, metric=self.metric, 
                                        normalize=True, random_state=self.random_state)
        estimators.append(nr)
        
        if self.use_variance_ratio:
            lvr = LocalVarianceRatioEstimator(k=self.k, metric=self.metric,
                                               random_state=self.random_state)
            estimators.append(lvr)
        
        if self.use_cv:
            cv = CrossValResidualEstimator(normalize=True, random_state=self.random_state)
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
                 normalize: bool = True, random_state: int = 42):
        super().__init__(random_state)
        self.kernel = kernel
        self.n_restarts = n_restarts
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
        from sklearn.preprocessing import StandardScaler
        
        # Standardize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Build kernel with noise term
        if self.kernel == 'rbf':
            base_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        elif self.kernel == 'matern':
            base_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        else:
            base_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        
        kernel = base_kernel + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
        
        # Fit GP (use subset if too large)
        n_max = min(1000, len(y))
        if len(y) > n_max:
            idx = np.random.RandomState(self.random_state).choice(len(y), n_max, replace=False)
            X_fit, y_fit = X_scaled[idx], y_scaled[idx]
        else:
            X_fit, y_fit = X_scaled, y_scaled
            
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
            normalize_y=False
        )
        gp.fit(X_fit, y_fit)
        
        # Get predictions with uncertainty on full data
        y_pred, y_std = gp.predict(X_scaled, return_std=True)
        
        # Noise score = prediction uncertainty + residual
        residuals = np.abs(y_scaled - y_pred)
        scores = y_std + residuals
        
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)
        
        self.noise_scores_ = scores
        self.gp_model_ = gp
        self.predictions_ = y_pred * scaler_y.scale_[0] + scaler_y.mean_[0]
        self.uncertainties_ = y_std * scaler_y.scale_[0]
        
        # Extract learned noise level
        self.learned_noise_level_ = np.sqrt(gp.kernel_.get_params().get('k2__noise_level', 0.1))
        
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
                 random_state: int = 42):
        super().__init__(random_state)
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Standardize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X).astype(np.float32)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32).ravel()
        
        # Build network
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
        
        model = HeteroscedasticNet(input_dim, self.hidden_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Data loader
        dataset = TensorDataset(
            torch.from_numpy(X_scaled),
            torch.from_numpy(y_scaled)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # β-NLL loss
        def beta_nll_loss(mean, log_var, target, beta=0.5):
            var = torch.exp(log_var)
            loss = 0.5 * ((target - mean) ** 2 / var + log_var)
            if beta > 0:
                loss = loss * (var.detach() ** beta)
            return loss.mean()
        
        # Train
        model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                mean, log_var = model(X_batch)
                loss = beta_nll_loss(mean, log_var, y_batch, self.beta)
                loss.backward()
                optimizer.step()
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_scaled)
            mean_pred, log_var_pred = model(X_tensor)
            mean_pred = mean_pred.numpy()
            var_pred = np.exp(log_var_pred.numpy())
            std_pred = np.sqrt(var_pred)
        
        # Noise score = predicted uncertainty
        scores = std_pred
        
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)
        
        self.noise_scores_ = scores
        self.predictions_ = mean_pred * scaler_y.scale_[0] + scaler_y.mean_[0]
        self.uncertainties_ = std_pred * scaler_y.scale_[0]
        self.model_ = model
        
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
                 random_state: int = 42):
        super().__init__(random_state)
        self.base_model = base_model
        self.alpha = alpha  # miscoverage rate
        self.n_cal = n_cal  # fraction for calibration
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.model_selection import train_test_split
        
        # Split into train and calibration
        n_cal = int(len(y) * self.n_cal)
        idx = np.arange(len(y))
        np.random.RandomState(self.random_state).shuffle(idx)
        cal_idx, train_idx = idx[:n_cal], idx[n_cal:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]
        
        # Train base model
        if self.base_model == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # For RF, use tree variance as uncertainty estimate
            y_pred_cal = model.predict(X_cal)
            
            # Get predictions from all trees for variance
            tree_preds = np.array([tree.predict(X_cal) for tree in model.estimators_])
            cal_std = tree_preds.std(axis=0)
            
        elif self.base_model == 'quantile':
            from sklearn.ensemble import GradientBoostingRegressor
            # Train quantile models
            model_lo = GradientBoostingRegressor(
                loss='quantile', alpha=self.alpha/2,
                n_estimators=100, random_state=self.random_state
            )
            model_hi = GradientBoostingRegressor(
                loss='quantile', alpha=1-self.alpha/2,
                n_estimators=100, random_state=self.random_state
            )
            model_med = GradientBoostingRegressor(
                loss='quantile', alpha=0.5,
                n_estimators=100, random_state=self.random_state
            )
            
            model_lo.fit(X_train, y_train)
            model_hi.fit(X_train, y_train)
            model_med.fit(X_train, y_train)
            
            y_pred_cal = model_med.predict(X_cal)
            y_lo_cal = model_lo.predict(X_cal)
            y_hi_cal = model_hi.predict(X_cal)
            cal_std = (y_hi_cal - y_lo_cal) / 2
            
            model = (model_lo, model_med, model_hi)
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")
        
        # Compute conformity scores on calibration set
        residuals_cal = np.abs(y_cal - y_pred_cal)
        
        # Normalize by local uncertainty if available
        conformity_scores = residuals_cal / (cal_std + 1e-10)
        
        # Get quantile for prediction interval
        q = np.quantile(conformity_scores, 1 - self.alpha)
        
        # Now get uncertainty for all samples
        if self.base_model == 'rf':
            tree_preds_all = np.array([tree.predict(X) for tree in model.estimators_])
            uncertainty = tree_preds_all.std(axis=0)
            y_pred_all = model.predict(X)
        else:
            model_lo, model_med, model_hi = model
            y_lo_all = model_lo.predict(X)
            y_hi_all = model_hi.predict(X)
            uncertainty = (y_hi_all - y_lo_all) / 2
            y_pred_all = model_med.predict(X)
        
        # Conformal interval width as noise score
        interval_width = 2 * q * uncertainty
        
        scores = interval_width
        
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)
        
        self.noise_scores_ = scores
        self.predictions_ = y_pred_all
        self.uncertainties_ = uncertainty
        self.interval_widths_ = interval_width
        self.conformal_quantile_ = q
        
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
                 random_state: int = 42):
        super().__init__(random_state)
        self.n_ensemble = n_ensemble
        self.hidden_dims = hidden_dims
        self.heteroscedastic = heteroscedastic
        self.epochs = epochs
        self.lr = lr
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        
        # Standardize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X).astype(np.float32)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32).ravel()
        
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
        
        # Train ensemble
        ensemble_means = []
        ensemble_vars = []
        
        for m in range(self.n_ensemble):
            torch.manual_seed(self.random_state + m)
            
            model = EnsembleNet(input_dim, self.hidden_dims, self.heteroscedastic)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            dataset = TensorDataset(
                torch.from_numpy(X_scaled),
                torch.from_numpy(y_scaled)
            )
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            # Train
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
            
            # Predict
            model.eval()
            with torch.no_grad():
                X_tensor = torch.from_numpy(X_scaled)
                mean_pred, log_var_pred = model(X_tensor)
                ensemble_means.append(mean_pred.numpy())
                if self.heteroscedastic:
                    ensemble_vars.append(np.exp(log_var_pred.numpy()))
        
        ensemble_means = np.array(ensemble_means)
        
        # Epistemic uncertainty (disagreement between models)
        epistemic = ensemble_means.std(axis=0)
        
        # Aleatoric uncertainty (average predicted variance)
        if self.heteroscedastic:
            ensemble_vars = np.array(ensemble_vars)
            aleatoric = np.sqrt(ensemble_vars.mean(axis=0))
        else:
            aleatoric = np.zeros_like(epistemic)
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic**2 + aleatoric**2)
        
        scores = total_uncertainty
        
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)
        
        self.noise_scores_ = scores
        self.predictions_ = ensemble_means.mean(axis=0) * scaler_y.scale_[0] + scaler_y.mean_[0]
        self.epistemic_uncertainty_ = epistemic * scaler_y.scale_[0]
        self.aleatoric_uncertainty_ = aleatoric * scaler_y.scale_[0]
        self.total_uncertainty_ = total_uncertainty * scaler_y.scale_[0]
        
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
                 normalize: bool = True, random_state: int = 42):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.method = method
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.ensemble import RandomForestRegressor
        
        # Train RF with OOB
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        if self.method == 'tree_variance':
            # Simple: variance across tree predictions
            tree_preds = np.array([tree.predict(X) for tree in rf.estimators_])
            uncertainty = tree_preds.std(axis=0)
            
        elif self.method == 'jackknife':
            # Jackknife+ variance estimate (Wager et al. 2014)
            n_samples = len(y)
            
            # Get OOB predictions
            oob_preds = rf.oob_prediction_
            oob_residuals = y - oob_preds
            
            # Tree predictions
            tree_preds = np.array([tree.predict(X) for tree in rf.estimators_])
            
            # Compute jackknife variance
            # V_IJ = variance of (mean prediction when sample i is OOB)
            pred_mean = tree_preds.mean(axis=0)
            
            # For each sample, estimate variance from trees where it was OOB
            uncertainty = np.zeros(n_samples)
            for i in range(n_samples):
                # Trees where sample i was OOB
                # We approximate by using tree variance + residual term
                uncertainty[i] = tree_preds[:, i].std() + np.abs(oob_residuals[i]) * 0.5
                
        elif self.method == 'quantile':
            # Use quantile predictions from trees
            tree_preds = np.array([tree.predict(X) for tree in rf.estimators_])
            q_lo = np.percentile(tree_preds, 10, axis=0)
            q_hi = np.percentile(tree_preds, 90, axis=0)
            uncertainty = (q_hi - q_lo) / 2
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        scores = uncertainty
        
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)
        
        self.noise_scores_ = scores
        self.predictions_ = rf.predict(X)
        self.uncertainties_ = uncertainty
        self.oob_predictions_ = rf.oob_prediction_
        self.model_ = rf
        
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
                 random_state: int = 42):
        super().__init__(random_state)
        self.quantiles = quantiles
        self.model_type = model_type
        self.normalize = normalize
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.model_type == 'gbr':
            from sklearn.ensemble import GradientBoostingRegressor
            
            predictions = {}
            for q in self.quantiles:
                model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=q,
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state
                )
                model.fit(X, y)
                predictions[q] = model.predict(X)
                
        elif self.model_type == 'nn':
            # Neural network quantile regression
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.preprocessing import StandardScaler
            
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X).astype(np.float32)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32).ravel()
            
            class QuantileNet(nn.Module):
                def __init__(self, input_dim, n_quantiles):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, n_quantiles)
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            def pinball_loss(pred, target, quantiles):
                target = target.unsqueeze(1).expand_as(pred)
                quantiles_t = torch.tensor(quantiles).float().unsqueeze(0).expand_as(pred)
                errors = target - pred
                loss = torch.max(quantiles_t * errors, (quantiles_t - 1) * errors)
                return loss.mean()
            
            torch.manual_seed(self.random_state)
            model = QuantileNet(X.shape[1], len(self.quantiles))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_scaled))
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            model.train()
            for epoch in range(100):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = pinball_loss(pred, y_batch, self.quantiles)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                preds = model(torch.from_numpy(X_scaled)).numpy()
            
            predictions = {q: preds[:, i] * scaler_y.scale_[0] + scaler_y.mean_[0] 
                          for i, q in enumerate(self.quantiles)}
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Interval width as uncertainty
        q_lo = min(self.quantiles)
        q_hi = max(self.quantiles)
        interval_width = predictions[q_hi] - predictions[q_lo]
        
        scores = interval_width
        
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)
        
        self.noise_scores_ = scores
        self.predictions_ = predictions.get(0.5, predictions[list(predictions.keys())[len(predictions)//2]])
        self.quantile_predictions_ = predictions
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
                 random_state: int = 42):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.normalize = normalize  # Default False - use raw sigma for confidence method
        
    def estimate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        
        model = NGBRegressor(
            Dist=Normal,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            random_state=self.random_state,
            verbose=False
        )
        model.fit(X, y)
        
        # Get distribution parameters
        dist = model.pred_dist(X)
        mu = dist.loc      # predicted mean
        sigma = dist.scale  # predicted std (uncertainty)
        
        scores = sigma
        
        if self.normalize:
            scores = scores / (scores.std() + 1e-10)
        
        self.noise_scores_ = scores
        self.predictions_ = mu  # Use as correction target
        self.sigma_ = sigma
        self.mu_ = mu
        self.model_ = model
        
        return scores


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

ESTIMATOR_REGISTRY = {
    # Neighbor-based (use molecular representation distances)
    'neighbor_residual': NeighborResidualEstimator,
    'local_variance_ratio': LocalVarianceRatioEstimator,
    'distance_weighted': DistanceWeightedResidualEstimator,
    'activity_cliff_aware': ActivityCliffAwareEstimator,
    
    # Model-based (use prediction disagreement)
    'cross_val': CrossValResidualEstimator,
    'ensemble': EnsembleDisagreementEstimator,
    'leave_one_out': LeaveOneOutEstimator,
    'hybrid': HybridNoiseEstimator,
    
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