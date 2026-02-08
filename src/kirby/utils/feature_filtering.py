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


# =============================================================================
# MULTICOLLINEARITY DIAGNOSTICS
# =============================================================================

def compute_vif(X, feature_names=None):
    """
    Compute Variance Inflation Factor for each feature.

    VIF measures how much the variance of a regression coefficient is inflated
    due to multicollinearity. VIF > 5 suggests moderate collinearity, VIF > 10
    suggests high collinearity.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        dict: {
            'vif_scores': array of VIF for each feature,
            'feature_names': list of names,
            'high_vif_indices': indices with VIF > 10,
            'moderate_vif_indices': indices with VIF > 5,
            'mean_vif': mean VIF across features,
            'max_vif': maximum VIF
        }
    """
    from sklearn.linear_model import LinearRegression

    n_samples, n_features = X.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Standardize features
    X_centered = X - X.mean(axis=0)
    X_std = X_centered / (X_centered.std(axis=0) + 1e-10)

    vif_scores = np.zeros(n_features)

    for i in range(n_features):
        # Regress feature i on all other features
        y = X_std[:, i]
        X_others = np.delete(X_std, i, axis=1)

        if X_others.shape[1] == 0:
            vif_scores[i] = 1.0
            continue

        model = LinearRegression()
        model.fit(X_others, y)
        r_squared = model.score(X_others, y)

        # VIF = 1 / (1 - R²)
        if r_squared >= 1.0:
            vif_scores[i] = np.inf
        else:
            vif_scores[i] = 1.0 / (1.0 - r_squared)

    return {
        'vif_scores': vif_scores,
        'feature_names': feature_names,
        'high_vif_indices': np.where(vif_scores > 10)[0],
        'moderate_vif_indices': np.where((vif_scores > 5) & (vif_scores <= 10))[0],
        'mean_vif': np.mean(vif_scores[np.isfinite(vif_scores)]),
        'max_vif': np.max(vif_scores[np.isfinite(vif_scores)]) if np.any(np.isfinite(vif_scores)) else np.inf,
    }


def effective_dimensionality(X, variance_threshold=0.95):
    """
    Compute effective dimensionality using PCA.

    The effective dimensionality is the number of principal components needed
    to explain the specified fraction of total variance.

    Args:
        X: Feature matrix (n_samples, n_features)
        variance_threshold: Fraction of variance to explain (default: 0.95)

    Returns:
        dict: {
            'nominal_dim': original number of features,
            'effective_dim': number of PCs for threshold,
            'explained_variance_ratio': cumulative variance explained,
            'compression_ratio': nominal_dim / effective_dim
        }
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_samples, n_features = X.shape

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA
    n_components = min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # Find effective dim
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    effective_dim = np.searchsorted(cumulative_var, variance_threshold) + 1
    effective_dim = min(effective_dim, n_features)

    return {
        'nominal_dim': n_features,
        'effective_dim': int(effective_dim),
        'explained_variance_ratio': cumulative_var.tolist(),
        'compression_ratio': n_features / effective_dim if effective_dim > 0 else np.inf,
    }


def diagnose_multicollinearity(X, feature_names=None, verbose=True):
    """
    Run full multicollinearity diagnostics on a feature matrix.

    Combines VIF analysis and effective dimensionality to assess
    redundancy in the feature set.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: Optional list of feature names
        verbose: Print summary if True

    Returns:
        dict: Combined diagnostics from VIF and effective_dim
    """
    vif_results = compute_vif(X, feature_names)
    dim_results = effective_dimensionality(X)

    results = {
        **vif_results,
        **dim_results,
    }

    if verbose:
        print(f"\n=== MULTICOLLINEARITY DIAGNOSTICS ===")
        print(f"Nominal dimension: {dim_results['nominal_dim']}")
        print(f"Effective dimension (95% var): {dim_results['effective_dim']}")
        print(f"Compression ratio: {dim_results['compression_ratio']:.2f}x")
        print(f"Mean VIF: {vif_results['mean_vif']:.2f}")
        print(f"Max VIF: {vif_results['max_vif']:.2f}")
        print(f"Features with VIF > 10: {len(vif_results['high_vif_indices'])}")
        print(f"Features with VIF > 5: {len(vif_results['moderate_vif_indices'])}")

        if len(vif_results['high_vif_indices']) > 0:
            print(f"\nHigh-VIF features (consider removing):")
            for idx in vif_results['high_vif_indices'][:10]:
                name = vif_results['feature_names'][idx]
                vif = vif_results['vif_scores'][idx]
                print(f"  {name}: VIF={vif:.1f}")

    return results