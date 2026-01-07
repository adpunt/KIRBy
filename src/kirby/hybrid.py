"""
Core hybrid representation creation functions - ENHANCED with Greedy Allocation

Supports multiple feature importance methods and allocation strategies:
- Allocation methods: 'fixed' (equal per rep) or 'greedy' (adaptive)
- Importance methods: random_forest, shap, permutation, mutual_info, etc.

Default configuration (from testing):
- allocation_method='greedy'
- budget=100
- step_size=10
- patience=3
- apply_filters=True (quality filters: sparsity + variance)
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================================
# GREEDY ALLOCATION
# ============================================================================

def allocate_greedy_forward(rep_dict_train, rep_dict_val, train_labels, val_labels,
                           total_budget=100, step_size=10, patience=3, 
                           importance_method='random_forest'):
    """
    Greedy forward feature allocation with plateau detection.
    
    Iteratively adds features from the representation that improves validation
    performance most, stopping when performance plateaus or budget is exhausted.
    
    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        step_size: Features to add per iteration (default: 10)
        patience: Early stopping patience, None to disable (default: 3)
        importance_method: Method for feature selection within reps (default: 'random_forest')
        
    Returns:
        tuple: (allocation, history)
            - allocation: Dict of {rep_name: n_features}
            - history: List of iteration details
    """
    allocation = {rep: 0 for rep in rep_dict_train.keys()}
    remaining_budget = total_budget
    
    best_val_r2 = -np.inf
    steps_without_improvement = 0
    history = []
    
    iteration = 0
    while remaining_budget >= step_size:
        iteration += 1
        best_gain = -np.inf
        best_rep = None
        
        # Try adding step_size features from each rep
        for rep_name in rep_dict_train.keys():
            trial_allocation = allocation.copy()
            trial_allocation[rep_name] += step_size
            
            # Check if rep has enough features
            max_features = rep_dict_train[rep_name].shape[1]
            if trial_allocation[rep_name] > max_features:
                continue
            
            # Create hybrid on train and apply SAME features to val
            X_train_trial, feat_info = _create_hybrid_with_allocation(
                rep_dict_train, train_labels, trial_allocation, importance_method
            )
            X_val_trial = apply_feature_selection(rep_dict_val, feat_info)
            
            # Evaluate on validation
            r2 = _evaluate_allocation(X_train_trial, X_val_trial, train_labels, val_labels)
            
            if r2 > best_gain:
                best_gain = r2
                best_rep = rep_name
        
        # Stopping conditions
        if best_rep is None:
            break
        
        # Patience-based early stopping
        if patience is not None:
            if best_gain <= best_val_r2:
                steps_without_improvement += 1
                if steps_without_improvement >= patience:
                    break
            else:
                steps_without_improvement = 0
                best_val_r2 = best_gain
        else:
            best_val_r2 = max(best_val_r2, best_gain)
        
        # Add features
        allocation[best_rep] += step_size
        remaining_budget -= step_size
        
        history.append({
            'iteration': iteration,
            'rep': best_rep,
            'allocation': allocation.copy(),
            'val_r2': best_gain
        })
    
    return allocation, history


def _create_hybrid_with_allocation(rep_dict, labels, allocation, importance_method='random_forest'):
    """Helper: Create hybrid with specific allocation"""
    selected_features = []
    feature_info = {}
    
    for name, X in rep_dict.items():
        n_select = allocation[name]
        
        if n_select == 0:
            continue
        
        # Clean data
        X_clipped = np.clip(X, -1e10, 1e10)
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if n_select >= X_clipped.shape[1]:
            selected_features.append(X_clipped)
            feature_info[name] = {
                'selected_indices': np.arange(X_clipped.shape[1]),
                'n_features': X_clipped.shape[1]
            }
            continue
        
        # Compute importance and select top features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clipped)
        
        # Use RF for importance (fast and reliable)
        is_classification = len(np.unique(labels)) < 10
        if is_classification:
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        
        model.fit(X_scaled, labels)
        importances = model.feature_importances_
        importances = np.nan_to_num(importances, nan=0.0)
        
        n_select = min(n_select, X_clipped.shape[1])
        top_idx = np.argsort(importances)[-n_select:][::-1]
        
        X_selected = X_clipped[:, top_idx]
        selected_features.append(X_selected)
        
        feature_info[name] = {
            'selected_indices': top_idx,
            'importance_scores': importances[top_idx],
            'n_features': n_select
        }
    
    if len(selected_features) == 0:
        return np.zeros((len(labels), 0)), feature_info
    
    return np.hstack(selected_features).astype(np.float32), feature_info


def _evaluate_allocation(X_train, X_val, y_train, y_val):
    """Helper: Evaluate allocation with RF"""
    if X_train.shape[1] == 0:
        return -np.inf
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    is_classification = len(np.unique(y_train)) < 10
    if is_classification:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    model.fit(X_train_scaled, y_train)
    
    if is_classification:
        return model.score(X_val_scaled, y_val)
    else:
        from sklearn.metrics import r2_score
        y_pred = model.predict(X_val_scaled)
        return r2_score(y_val, y_pred)


# ============================================================================
# FEATURE IMPORTANCE COMPUTATION
# ============================================================================

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
    
    # RANDOM FOREST (DEFAULT)
    if method == 'random_forest':
        n_estimators = kwargs.get('n_estimators', 50)  # Match original implementation
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        
        for rep_name, X in base_reps.items():
            # Clean data (match original implementation)
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Scale data (match original implementation)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)
            
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
            
            rf.fit(X_scaled, labels)
            importance_scores[rep_name] = rf.feature_importances_
    
    # PERMUTATION IMPORTANCE
    elif method == 'permutation':
        from sklearn.inspection import permutation_importance
        
        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        n_repeats = kwargs.get('n_repeats', 5)
        
        for rep_name, X in base_reps.items():
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
    
    # MUTUAL INFORMATION
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        random_state = kwargs.get('random_state', 42)
        
        for rep_name, X in base_reps.items():
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if is_classification:
                mi_scores = mutual_info_classif(X_scaled, labels, random_state=random_state)
            else:
                mi_scores = mutual_info_regression(X_scaled, labels, random_state=random_state)
            
            importance_scores[rep_name] = np.nan_to_num(mi_scores, nan=0.0)
    
    else:
        raise NotImplementedError(f"Method '{method}' not yet implemented. "
                                 f"Choose from: random_forest, permutation, mutual_info")
    
    return importance_scores


# ============================================================================
# FEATURE CONCATENATION
# ============================================================================

def concatenate_features(base_reps, importance_scores, allocation):
    """
    Concatenate top features from each representation based on allocation.
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        importance_scores: Dict with representation names as keys and importance arrays as values
        allocation: Dict with representation names as keys and n_features as values
                   OR int for same allocation across all reps
        
    Returns:
        tuple: (hybrid_features, feature_info)
            - hybrid_features: Concatenated array of selected features
            - feature_info: Dict with metadata about selected features
    """
    # Handle int allocation (backwards compatibility)
    if isinstance(allocation, int):
        allocation = {rep: allocation for rep in base_reps.keys()}
    
    selected_parts = []
    feature_info = {}
    
    for rep_name in base_reps.keys():
        X = base_reps[rep_name]
        scores = importance_scores[rep_name]
        n_select = allocation.get(rep_name, 0)
        
        if n_select == 0:
            continue
        
        # Select top N features
        n_select = min(n_select, X.shape[1])
        top_indices = np.argsort(scores)[-n_select:][::-1]
        
        # Extract selected features
        X_selected = X[:, top_indices]
        selected_parts.append(X_selected)
        
        # Store metadata
        feature_info[rep_name] = {
            'selected_indices': top_indices,
            'importance_scores': scores[top_indices],
            'n_features': n_select
        }
    
    # Concatenate all selected features
    if len(selected_parts) == 0:
        return np.zeros((list(base_reps.values())[0].shape[0], 0)), feature_info
    
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
        if rep_name not in feature_info:
            continue
        
        X = base_reps[rep_name]
        indices = feature_info[rep_name]['selected_indices']
        
        # Apply same selection as training
        X_selected = X[:, indices]
        selected_parts.append(X_selected)
    
    if len(selected_parts) == 0:
        return np.zeros((list(base_reps.values())[0].shape[0], 0))
    
    return np.hstack(selected_parts).astype(np.float32)


# ============================================================================
# MAIN HYBRID CREATION
# ============================================================================

def create_hybrid(base_reps, labels, 
                 allocation_method='greedy',
                 n_per_rep=100,
                 budget=100,
                 step_size=10,
                 patience=3,
                 validation_split=0.2,
                 apply_filters=False,
                 importance_method='random_forest',
                 **kwargs):
    """
    Create hybrid representation by combining features from multiple base representations.
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        labels: Target values for computing importance
        
        allocation_method: Method for allocating features across representations
            - 'greedy': Adaptive allocation via greedy forward selection (default)
            - 'fixed': Equal allocation (n_per_rep for all)
            
        n_per_rep: Number of features per rep (for 'fixed' method, default: 100)
        budget: Total feature budget (for 'greedy' method, default: 100)
        step_size: Features to add per greedy iteration (default: 10)
        patience: Early stopping patience for greedy, None to disable (default: 3)
        validation_split: Fraction for validation in greedy (default: 0.2)
        
        apply_filters: Apply quality filters (sparsity + variance) before allocation (default: False)
        importance_method: Method for feature importance (default: 'random_forest')
        
        **kwargs: Additional arguments for importance computation
        
    Returns:
        tuple: (hybrid_features, feature_info)
            - hybrid_features: Combined feature array
            - feature_info: Metadata about selected features (includes 'allocation' for greedy)
    """
    # Apply quality filters if requested
    if apply_filters:
        try:
            from kirby.utils.feature_filtering import apply_quality_filters
            
            filtered_reps = {}
            for rep_name, X in base_reps.items():
                X_filtered, keep_mask = apply_quality_filters(
                    X, 
                    sparsity_threshold=kwargs.get('sparsity_threshold', 0.95),
                    variance_threshold=kwargs.get('variance_threshold', 0.01)
                )
                filtered_reps[rep_name] = X_filtered
            base_reps = filtered_reps
        except ImportError:
            print("Warning: Could not import feature_filtering, skipping filters")
    
    # GREEDY ALLOCATION
    if allocation_method == 'greedy':
        # Split into train/val for greedy allocation
        n_samples = len(labels)
        n_val = int(n_samples * validation_split)
        
        # Shuffle indices
        indices = np.random.RandomState(42).permutation(n_samples)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        # Split data
        rep_dict_train = {name: X[train_idx] for name, X in base_reps.items()}
        rep_dict_val = {name: X[val_idx] for name, X in base_reps.items()}
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        # Run greedy allocation
        allocation, history = allocate_greedy_forward(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget, step_size=step_size, patience=patience,
            importance_method=importance_method
        )
        
        # Compute importance on full data
        importance_scores = compute_feature_importance(
            base_reps, labels, method=importance_method, **kwargs
        )
        
        # Create final hybrid with optimal allocation
        hybrid_features, feature_info = concatenate_features(
            base_reps, importance_scores, allocation
        )
        
        # Add allocation to feature_info for reference
        feature_info['allocation'] = allocation
        feature_info['greedy_history'] = history
        
    # FIXED ALLOCATION (BACKWARDS COMPATIBLE)
    elif allocation_method == 'fixed':
        # Compute importance
        importance_scores = compute_feature_importance(
            base_reps, labels, method=importance_method, **kwargs
        )
        
        # Use same n_per_rep for all reps
        allocation = {rep: n_per_rep for rep in base_reps.keys()}
        
        hybrid_features, feature_info = concatenate_features(
            base_reps, importance_scores, allocation
        )
        
        feature_info['allocation'] = allocation
    
    else:
        raise ValueError(f"Unknown allocation_method: {allocation_method}. "
                        f"Choose 'greedy' or 'fixed'")
    
    return hybrid_features, feature_info