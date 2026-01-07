"""
Feature Filtering - Quality Filters Only

Simplified filtering module that removes low-quality features:
- Sparsity filter: Remove features that are mostly zeros
- Variance filter: Remove features with near-zero variance

Based on testing, cross-rep correlation filters hurt performance, so they're excluded.
"""

import numpy as np


def apply_quality_filters(X, sparsity_threshold=0.95, variance_threshold=0.01):
    """
    Apply data quality filters to remove low-information features.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        sparsity_threshold: Maximum allowed sparsity (default: 0.95)
                          Features with >95% zeros are removed
        variance_threshold: Minimum required variance (default: 0.01)
                          Features with variance < 0.01 are removed
        
    Returns:
        tuple: (X_filtered, keep_mask)
            - X_filtered: Filtered feature matrix
            - keep_mask: Boolean mask of kept features
    """
    n_samples, n_features = X.shape
    keep_mask = np.ones(n_features, dtype=bool)
    
    # Sparsity filter
    if sparsity_threshold is not None:
        sparsity = (X == 0).sum(axis=0) / n_samples
        keep_mask &= (sparsity <= sparsity_threshold)
    
    # Variance filter
    if variance_threshold is not None:
        variances = np.var(X, axis=0)
        keep_mask &= (variances > variance_threshold)
    
    X_filtered = X[:, keep_mask]
    
    n_removed = n_features - keep_mask.sum()
    if n_removed > 0:
        print(f"  Removed {n_removed}/{n_features} features "
              f"({n_removed/n_features*100:.1f}%) via quality filters")
    
    return X_filtered, keep_mask


def apply_filters(rep_dict_train, rep_dict_test, train_labels,
                 sparsity_threshold=0.95, variance_threshold=0.01):
    """
    Apply quality filters to multiple representations.
    
    Convenience function for filtering multiple representations at once.
    Filters are fit on training data and applied to both train and test.
    
    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_test: Dict of {rep_name: test_features}
        train_labels: Training labels (not used, kept for API compatibility)
        sparsity_threshold: Sparsity threshold (default: 0.95)
        variance_threshold: Variance threshold (default: 0.01)
        
    Returns:
        tuple: (filtered_train, filtered_test)
            - filtered_train: Dict of {rep_name: filtered_train_features}
            - filtered_test: Dict of {rep_name: filtered_test_features}
    """
    filtered_train = {}
    filtered_test = {}
    
    for rep_name in rep_dict_train.keys():
        X_train = rep_dict_train[rep_name]
        X_test = rep_dict_test[rep_name]
        
        # Apply filters on train, get mask
        X_train_filt, keep_mask = apply_quality_filters(
            X_train, sparsity_threshold, variance_threshold
        )
        
        # Apply same mask to test
        X_test_filt = X_test[:, keep_mask]
        
        filtered_train[rep_name] = X_train_filt
        filtered_test[rep_name] = X_test_filt
    
    return filtered_train, filtered_test


# Predefined filter configurations for easy use
FILTER_CONFIGS = {
    'none': {
        'sparsity_threshold': None,
        'variance_threshold': None
    },
    'quality_only': {
        'sparsity_threshold': 0.95,
        'variance_threshold': 0.01
    },
    'quality_strict': {
        'sparsity_threshold': 0.90,
        'variance_threshold': 0.05
    }
}