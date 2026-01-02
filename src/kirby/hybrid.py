"""
Core hybrid representation creation functions - ENHANCED

Supports multiple feature importance methods:
- random_forest (default)
- shap
- fasttreeshap  
- permutation
- mutual_info
- interpretml
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def compute_feature_importance(base_reps, labels, method='random_forest', **kwargs):
    """
    Compute feature importance scores for each representation.
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        labels: Target values (n_samples,)
        method: Method for computing importance:
            - 'random_forest' (default)
            - 'shap'
            - 'fasttreeshap'
            - 'permutation'
            - 'mutual_info'
            - 'interpretml'
        **kwargs: Additional arguments for the importance method
        
    Returns:
        dict: Dictionary with representation names as keys and importance score arrays as values
    """
    importance_scores = {}
    is_classification = len(np.unique(labels)) < 10
    
    # =========================================================================
    # RANDOM FOREST (DEFAULT - YOUR ORIGINAL METHOD)
    # =========================================================================
    if method == 'random_forest':
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        
        for rep_name, X in base_reps.items():
            print(f"Computing RF importance for {rep_name}...")
            
            if is_classification:
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
            
            rf.fit(X, labels)
            importance_scores[rep_name] = rf.feature_importances_
    
    # =========================================================================
    # SHAP
    # =========================================================================
    elif method == 'shap':
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Run: pip install shap")
        
        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        
        for rep_name, X in base_reps.items():
            print(f"Computing SHAP importance for {rep_name}...")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                              random_state=random_state, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                             random_state=random_state, n_jobs=-1)
            
            model.fit(X_scaled, labels)
            explainer = shap.TreeExplainer(model)
            
            n_samples = min(500, X_scaled.shape[0])
            sample_idx = np.random.choice(X_scaled.shape[0], n_samples, replace=False)
            shap_values = explainer.shap_values(X_scaled[sample_idx])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) == 2 else np.mean(shap_values, axis=0)
            
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            importance_scores[rep_name] = np.nan_to_num(mean_abs_shap, nan=0.0)
    
    # =========================================================================
    # FASTTREESHAP
    # =========================================================================
    elif method == 'fasttreeshap':
        try:
            import fasttreeshap
        except ImportError:
            raise ImportError("fasttreeshap not installed. Run: pip install fasttreeshap")
        
        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        
        for rep_name, X in base_reps.items():
            print(f"Computing FastTreeSHAP importance for {rep_name}...")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                              random_state=random_state, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                             random_state=random_state, n_jobs=-1)
            
            model.fit(X_scaled, labels)
            explainer = fasttreeshap.TreeExplainer(model, algorithm='auto', n_jobs=-1)
            
            n_samples = min(500, X_scaled.shape[0])
            sample_idx = np.random.choice(X_scaled.shape[0], n_samples, replace=False)
            shap_values = explainer.shap_values(X_scaled[sample_idx])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) == 2 else np.mean(shap_values, axis=0)
            
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            importance_scores[rep_name] = np.nan_to_num(mean_abs_shap, nan=0.0)
    
    # =========================================================================
    # PERMUTATION IMPORTANCE
    # =========================================================================
    elif method == 'permutation':
        from sklearn.inspection import permutation_importance
        
        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        n_repeats = kwargs.get('n_repeats', 5)
        
        for rep_name, X in base_reps.items():
            print(f"Computing permutation importance for {rep_name}...")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                              random_state=random_state, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                             random_state=random_state, n_jobs=-1)
            
            model.fit(X_scaled, labels)
            perm = permutation_importance(model, X_scaled, labels, n_repeats=n_repeats,
                                         random_state=random_state, n_jobs=-1)
            importance_scores[rep_name] = np.nan_to_num(perm.importances_mean, nan=0.0)
    
    # =========================================================================
    # MUTUAL INFORMATION
    # =========================================================================
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        random_state = kwargs.get('random_state', 42)
        
        for rep_name, X in base_reps.items():
            print(f"Computing mutual information for {rep_name}...")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if is_classification:
                mi_scores = mutual_info_classif(X_scaled, labels, random_state=random_state)
            else:
                mi_scores = mutual_info_regression(X_scaled, labels, random_state=random_state)
            
            importance_scores[rep_name] = np.nan_to_num(mi_scores, nan=0.0)
    
    # =========================================================================
    # INTERPRETML
    # =========================================================================
    elif method == 'interpretml':
        try:
            from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
        except ImportError:
            raise ImportError("InterpretML not installed. Run: pip install interpret")
        
        random_state = kwargs.get('random_state', 42)
        
        for rep_name, X in base_reps.items():
            print(f"Computing InterpretML importance for {rep_name}...")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if is_classification:
                model = ExplainableBoostingClassifier(random_state=random_state)
            else:
                model = ExplainableBoostingRegressor(random_state=random_state)
            
            model.fit(X_scaled, labels)
            importance_scores[rep_name] = np.nan_to_num(model.term_importances(), nan=0.0)
    
    else:
        raise NotImplementedError(f"Method '{method}' not yet implemented. "
                                 f"Choose from: random_forest, shap, fasttreeshap, permutation, mutual_info, interpretml")
    
    return importance_scores


def concatenate_features(base_reps, importance_scores, n_per_rep=100):
    """
    Concatenate top features from each representation based on importance scores.
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        importance_scores: Dict with representation names as keys and importance arrays as values
        n_per_rep: Number of top features to select from each representation
        
    Returns:
        tuple: (hybrid_features, feature_info)
            - hybrid_features: Concatenated array of selected features
            - feature_info: Dict with metadata about selected features INCLUDING indices for reuse
    """
    selected_parts = []
    feature_info = {}
    
    for rep_name in base_reps.keys():
        X = base_reps[rep_name]
        scores = importance_scores[rep_name]
        
        # Select top N features
        n_select = min(n_per_rep, X.shape[1])
        top_indices = np.argsort(scores)[-n_select:][::-1]  # Descending order
        
        # Extract selected features
        X_selected = X[:, top_indices]
        selected_parts.append(X_selected)
        
        # Store metadata INCLUDING indices for applying to test set
        feature_info[rep_name] = {
            'selected_indices': top_indices,
            'importance_scores': scores[top_indices],
            'n_features': n_select
        }
        
        print(f"{rep_name}: Selected {n_select} features (top importance: {scores[top_indices[0]]:.6f})")
    
    # Concatenate all selected features
    hybrid_features = np.hstack(selected_parts).astype(np.float32)
    
    return hybrid_features, feature_info


def apply_feature_selection(base_reps, feature_info):
    """
    Apply pre-computed feature selection to new data (e.g., test set).
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        feature_info: Dict from create_hybrid containing selected_indices for each rep
        
    Returns:
        np.ndarray: Concatenated selected features
    """
    selected_parts = []
    
    for rep_name in base_reps.keys():
        X = base_reps[rep_name]
        indices = feature_info[rep_name]['selected_indices']
        
        # Apply same selection as training
        X_selected = X[:, indices]
        selected_parts.append(X_selected)
    
    return np.hstack(selected_parts).astype(np.float32)


def create_hybrid(base_reps, labels, n_per_rep=100, importance_method='random_forest', **kwargs):
    """
    Create hybrid representation by combining top features from multiple base representations.
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        labels: Target values for computing importance
        n_per_rep: Number of features to select from each representation
        importance_method: Method for computing feature importance:
            - 'random_forest' (default)
            - 'shap'
            - 'fasttreeshap'
            - 'permutation'
            - 'mutual_info'
            - 'interpretml'
        **kwargs: Additional arguments for importance computation
        
    Returns:
        tuple: (hybrid_features, feature_info)
            - hybrid_features: Combined feature array
            - feature_info: Metadata about selected features
    """
    print(f"\nCreating hybrid representation from {len(base_reps)} base representations...")
    print(f"Method: {importance_method}, Features per rep: {n_per_rep}")
    
    # Step 1: Compute feature importance
    importance_scores = compute_feature_importance(
        base_reps, 
        labels, 
        method=importance_method,
        **kwargs
    )
    
    # Step 2: Concatenate top features
    hybrid_features, feature_info = concatenate_features(
        base_reps,
        importance_scores,
        n_per_rep=n_per_rep
    )
    
    print(f"\nHybrid representation created: {hybrid_features.shape[1]} total features")
    
    return hybrid_features, feature_info