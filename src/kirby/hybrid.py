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
from sklearn.metrics import r2_score


# ============================================================================
# GREEDY ALLOCATION
# ============================================================================

def allocate_greedy_forward(rep_dict_train, rep_dict_val, train_labels, val_labels,
                           total_budget=100, step_size=10, patience=3):
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
        
    Returns:
        tuple: (best_allocation, history, final_feature_info)
            - best_allocation: Dict of {rep_name: n_features} for best-performing iteration
            - history: List of iteration details
            - final_feature_info: Feature selection info matching best_allocation
    """
    allocation = {rep: 0 for rep in rep_dict_train.keys()}
    remaining_budget = total_budget
    
    best_val_r2 = -np.inf
    best_allocation = None
    best_feature_info = {}
    steps_without_improvement = 0
    history = []
    
    iteration = 0
    while remaining_budget > 0:
        iteration += 1
        iter_best_gain = -np.inf
        iter_best_rep = None
        iter_best_feat_info = None
        iter_best_trial_allocation = None
        
        # Try adding step_size features from each rep
        for rep_name in rep_dict_train.keys():
            # Check if rep is already maxed out
            max_features = rep_dict_train[rep_name].shape[1]
            if allocation[rep_name] >= max_features:
                continue  # Already using all features from this rep
            
            trial_allocation = allocation.copy()
            # Add step_size, but cap at max_features AND remaining_budget
            features_to_add = min(step_size, max_features - allocation[rep_name], remaining_budget)
            if features_to_add <= 0:
                continue
            trial_allocation[rep_name] = allocation[rep_name] + features_to_add
            
            # Create hybrid on train and apply SAME features to val
            X_train_trial, feat_info = _create_hybrid_with_allocation(
                rep_dict_train, train_labels, trial_allocation
            )
            X_val_trial = apply_feature_selection(rep_dict_val, feat_info)
            
            # Evaluate on validation
            r2 = _evaluate_allocation(X_train_trial, X_val_trial, train_labels, val_labels)
            
            if r2 > iter_best_gain:
                iter_best_gain = r2
                iter_best_rep = rep_name
                iter_best_feat_info = feat_info
                iter_best_trial_allocation = trial_allocation
        
        # Stopping conditions
        if iter_best_rep is None:
            break
        
        # Commit this iteration's best choice
        features_added = sum(iter_best_trial_allocation.values()) - sum(allocation.values())
        allocation = iter_best_trial_allocation
        remaining_budget -= features_added
        
        # Track global best (for patience-based early stopping)
        if iter_best_gain > best_val_r2:
            best_val_r2 = iter_best_gain
            best_allocation = allocation.copy()
            best_feature_info = iter_best_feat_info
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        
        history.append({
            'iteration': iteration,
            'rep': iter_best_rep,
            'allocation': allocation.copy(),
            'val_r2': iter_best_gain
        })
        
        # Patience-based early stopping
        if patience is not None and steps_without_improvement >= patience:
            break
    
    # If we never found anything, raise an error
    if best_allocation is None:
        raise ValueError(
            "Greedy allocation found no valid feature combinations. "
            "Check that representations have features and data is valid."
        )
    
    return best_allocation, history, best_feature_info


def allocate_performance_weighted(rep_dict_train, rep_dict_val, train_labels, val_labels,
                                  total_budget=100):
    """
    Allocate features proportional to each representation's baseline performance.
    
    Tests each representation individually on validation set, then allocates budget
    proportional to their R² scores (regression) or accuracy (classification).
    Better performing reps get more features.
    
    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        
    Returns:
        dict: Allocation of {rep_name: n_features}
    """
    is_classification = _is_classification_task(train_labels)
    
    baseline_scores = {}
    
    # Test each representation individually
    for rep_name, X_train in rep_dict_train.items():
        X_val = rep_dict_val[rep_name]
        
        # Clean and scale
        X_train_clean = np.clip(X_train, -1e10, 1e10)
        X_train_clean = np.nan_to_num(X_train_clean, nan=0.0, posinf=1e10, neginf=-1e10)
        X_val_clean = np.clip(X_val, -1e10, 1e10)
        X_val_clean = np.nan_to_num(X_val_clean, nan=0.0, posinf=1e10, neginf=-1e10)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_val_scaled = scaler.transform(X_val_clean)
        
        # Train and evaluate with appropriate model
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, train_labels)
            score = model.score(X_val_scaled, val_labels)  # accuracy
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, train_labels)
            y_pred = model.predict(X_val_scaled)
            score = r2_score(val_labels, y_pred)
        
        baseline_scores[rep_name] = max(score, 0.0)  # Clip negative scores to 0
    
    # Allocate proportional to performance
    total_score = sum(baseline_scores.values())
    
    if total_score == 0:
        raise ValueError(
            "All representations have zero or negative validation scores. "
            "Cannot compute performance-weighted allocation. Use allocation_method='greedy' or 'fixed' instead."
        )
    
    allocation = {}
    for rep_name, score in baseline_scores.items():
        # Allocate proportional to score
        n_features = int(round((score / total_score) * total_budget))
        # Ensure we don't exceed available features
        max_features = rep_dict_train[rep_name].shape[1]
        allocation[rep_name] = min(n_features, max_features)
    
    # Adjust to hit budget exactly
    current_total = sum(allocation.values())
    if current_total < total_budget:
        # Distribute remainder to reps with capacity, prioritizing best performers
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1], reverse=True)
        remaining = total_budget - current_total
        for rep_name, _ in sorted_reps:
            if remaining <= 0:
                break
            max_features = rep_dict_train[rep_name].shape[1]
            can_add = max_features - allocation[rep_name]
            add_amount = min(can_add, remaining)
            allocation[rep_name] += add_amount
            remaining -= add_amount
    elif current_total > total_budget:
        # Remove excess from worst performer
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1])
        excess = current_total - total_budget
        for rep_name, _ in sorted_reps:
            if excess <= 0:
                break
            reduce_by = min(allocation[rep_name], excess)
            allocation[rep_name] -= reduce_by
            excess -= reduce_by
    
    return allocation


def _is_classification_task(labels):
    """Determine if this is a classification or regression task.
    
    Uses dtype check: integer types are classification, float types are regression.
    """
    labels = np.asarray(labels)
    # Check if labels are integer type (classification)
    if np.issubdtype(labels.dtype, np.integer):
        return True
    # Check if float labels are actually integers (e.g., 0.0, 1.0, 2.0)
    if np.issubdtype(labels.dtype, np.floating):
        return np.allclose(labels, labels.astype(int)) and len(np.unique(labels)) < 20
    return False


def _create_hybrid_with_allocation(rep_dict, labels, allocation):
    """Helper: Create hybrid with specific allocation.
    
    Note: Always uses RandomForest for importance (fast and reliable for greedy inner loop).
    """
    selected_features = []
    feature_info = {}
    
    is_classification = _is_classification_task(labels)
    
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
                'importance_scores': np.ones(X_clipped.shape[1]),  # All features selected, importance not computed
                'n_features': X_clipped.shape[1]
            }
            continue
        
        # Compute importance and select top features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clipped)
        
        # Use RF for importance (fast and reliable)
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
        n_samples = len(labels)
        return np.zeros((n_samples, 0), dtype=np.float32), feature_info
    
    return np.hstack(selected_features).astype(np.float32), feature_info


def _evaluate_allocation(X_train, X_val, y_train, y_val):
    """Helper: Evaluate allocation with RF"""
    if X_train.shape[1] == 0:
        return -np.inf
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    is_classification = _is_classification_task(y_train)
    if is_classification:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    model.fit(X_train_scaled, y_train)
    
    if is_classification:
        return model.score(X_val_scaled, y_val)
    else:
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
    is_classification = _is_classification_task(labels)
    
    # RANDOM FOREST (DEFAULT)
    if method == 'random_forest':
        n_estimators = kwargs.get('n_estimators', 50)  # Match original implementation
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        
        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")
            
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
            importance_scores[rep_name] = np.nan_to_num(rf.feature_importances_, nan=0.0)
    
    # PERMUTATION IMPORTANCE
    elif method == 'permutation':
        from sklearn.inspection import permutation_importance
        
        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        n_repeats = kwargs.get('n_repeats', 5)
        
        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")
            
            # Clean data (match random_forest implementation)
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)
            
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
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")
            
            # Clean data (match random_forest implementation)
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)
            
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
            - hybrid_features: Concatenated array of selected features (clipped and cleaned)
            - feature_info: Dict with metadata about selected features
    """
    if not isinstance(allocation, dict):
        raise TypeError(f"allocation must be a dict, got {type(allocation).__name__}")
    
    selected_parts = []
    feature_info = {}
    
    for rep_name in base_reps.keys():
        if rep_name not in allocation:
            raise KeyError(f"allocation missing key '{rep_name}'")
        
        X = base_reps[rep_name]
        scores = importance_scores[rep_name]
        n_select = allocation[rep_name]
        
        if n_select == 0:
            continue
        
        # Verify dimension match
        if len(scores) != X.shape[1]:
            raise ValueError(
                f"Dimension mismatch for '{rep_name}': importance_scores has {len(scores)} values "
                f"but features has {X.shape[1]} columns. Ensure importance was computed on same data."
            )
        
        # Clean data (consistent with all other functions)
        X_clipped = np.clip(X, -1e10, 1e10)
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Select top N features
        n_select = min(n_select, X_clipped.shape[1])
        top_indices = np.argsort(scores)[-n_select:][::-1]
        
        # Extract selected features
        X_selected = X_clipped[:, top_indices]
        selected_parts.append(X_selected)
        
        # Store metadata
        feature_info[rep_name] = {
            'selected_indices': top_indices,
            'importance_scores': scores[top_indices],
            'n_features': n_select
        }
    
    # Concatenate all selected features
    if len(selected_parts) == 0:
        raise ValueError(
            "No features selected. Check that allocation has non-zero values "
            "for at least one representation."
        )
    
    hybrid_features = np.hstack(selected_parts).astype(np.float32)
    
    return hybrid_features, feature_info


def apply_feature_selection(base_reps, feature_info):
    """
    Apply pre-computed feature selection to new data (e.g., test set).
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        feature_info: Dict from create_hybrid containing selected_indices for each rep
        
    Returns:
        np.ndarray: Concatenated selected features (clipped and cleaned)
    """
    selected_parts = []
    
    # Iterate over feature_info keys to maintain same order as training
    # (feature_info preserves the order from create_hybrid)
    for rep_name in feature_info.keys():
        # Skip metadata keys like 'allocation', 'greedy_history'
        if not isinstance(feature_info[rep_name], dict) or 'selected_indices' not in feature_info[rep_name]:
            continue
            
        if rep_name not in base_reps:
            raise ValueError(f"Representation '{rep_name}' found in feature_info but not in base_reps")
        
        X = base_reps[rep_name]
        indices = feature_info[rep_name]['selected_indices']
        
        # Check that indices are valid for this data
        max_idx = np.max(indices) if len(indices) > 0 else -1
        if max_idx >= X.shape[1]:
            raise ValueError(
                f"Feature index {max_idx} out of bounds for representation '{rep_name}' "
                f"with {X.shape[1]} features. Ensure test data has same number of features as training data."
            )
        
        # Apply same selection as training
        X_selected = X[:, indices]
        
        # Apply same clipping as training (CRITICAL for consistency)
        X_selected = np.clip(X_selected, -1e10, 1e10)
        X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=1e10, neginf=-1e10)
        
        selected_parts.append(X_selected)
    
    if len(selected_parts) == 0:
        raise ValueError(
            "No features to select. feature_info contains no valid representation entries."
        )
    
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
            - 'performance_weighted': Allocate proportional to baseline R² scores
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
    # Convert labels to numpy array if needed
    labels = np.asarray(labels)
    
    # Input validation
    if not base_reps:
        raise ValueError("base_reps cannot be empty")
    
    # Check all reps have same number of samples and at least 1 feature
    n_samples_list = [X.shape[0] for X in base_reps.values()]
    if len(set(n_samples_list)) > 1:
        raise ValueError(f"All representations must have same number of samples. Got: {n_samples_list}")
    
    n_samples = n_samples_list[0]
    if len(labels) != n_samples:
        raise ValueError(f"labels length ({len(labels)}) must match number of samples ({n_samples})")
    
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples, got {n_samples}")
    
    # Check for reps with 0 features
    for rep_name, X in base_reps.items():
        if X.shape[1] == 0:
            raise ValueError(f"Representation '{rep_name}' has 0 features")
    
    # Check for NaN/inf in labels
    if np.any(np.isnan(labels)):
        raise ValueError("labels contains NaN values")
    if np.any(np.isinf(labels)):
        raise ValueError("labels contains infinite values")
    
    # Validate numeric parameters
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if n_per_rep <= 0:
        raise ValueError(f"n_per_rep must be positive, got {n_per_rep}")
    if not (0 < validation_split < 1):
        raise ValueError(f"validation_split must be between 0 and 1, got {validation_split}")
    if patience is not None and patience < 1:
        raise ValueError(f"patience must be at least 1 or None, got {patience}")
    
    # Apply quality filters if requested
    # We track the mapping from filtered indices back to original indices
    filter_index_maps = {}  # {rep_name: array mapping filtered_idx -> original_idx}
    original_base_reps = base_reps  # Keep reference to original data
    
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
                if X_filtered.shape[1] == 0:
                    raise ValueError(
                        f"All features filtered out for '{rep_name}'. "
                        f"Adjust sparsity_threshold or variance_threshold, or set apply_filters=False."
                    )
                filtered_reps[rep_name] = X_filtered
                filter_index_maps[rep_name] = np.where(keep_mask)[0]
            base_reps = filtered_reps
        except ImportError as e:
            raise ImportError(
                f"apply_filters=True requires kirby.utils.feature_filtering module: {e}"
            )
    
    # GREEDY ALLOCATION
    if allocation_method == 'greedy':
        # Split into train/val for greedy allocation
        n_samples = len(labels)
        n_val = int(n_samples * validation_split)
        
        # Ensure minimum val size to avoid degenerate cases
        n_val = max(n_val, 1)
        n_train = n_samples - n_val
        
        if n_train < 2:
            raise ValueError(f"Need at least 2 training samples for greedy allocation. "
                           f"Got {n_train} with validation_split={validation_split} and {n_samples} total samples.")
        if n_val >= n_samples:
            raise ValueError(f"validation_split={validation_split} leaves no training data "
                           f"for {n_samples} samples")
        
        # Shuffle indices
        indices = np.random.RandomState(42).permutation(n_samples)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        # Split data
        rep_dict_train = {name: X[train_idx] for name, X in base_reps.items()}
        rep_dict_val = {name: X[val_idx] for name, X in base_reps.items()}
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        # Run greedy allocation - now returns feature_info directly
        allocation, history, greedy_feature_info = allocate_greedy_forward(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget, step_size=step_size, patience=patience
        )
        
        # If filters were applied, remap indices from filtered space to original space
        if filter_index_maps:
            for rep_name in list(greedy_feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in greedy_feature_info[rep_name]:
                    filtered_indices = greedy_feature_info[rep_name]['selected_indices']
                    original_indices = filter_index_maps[rep_name][filtered_indices]
                    greedy_feature_info[rep_name]['selected_indices'] = original_indices
        
        # Apply the SAME feature selection from greedy to full data
        # Use original_base_reps so indices (now in original space) match
        hybrid_features = apply_feature_selection(original_base_reps, greedy_feature_info)
        
        # Build feature_info with allocation metadata
        feature_info = greedy_feature_info.copy()
        feature_info['allocation'] = allocation
        feature_info['greedy_history'] = history
    
    # PERFORMANCE-WEIGHTED ALLOCATION
    elif allocation_method == 'performance_weighted':
        # Split into train/val to test baselines
        n_samples = len(labels)
        n_val = int(n_samples * validation_split)
        
        # Ensure minimum val size to avoid degenerate cases
        n_val = max(n_val, 1)
        n_train = n_samples - n_val
        
        if n_train < 2:
            raise ValueError(f"Need at least 2 training samples for performance_weighted allocation. "
                           f"Got {n_train} with validation_split={validation_split} and {n_samples} total samples.")
        if n_val >= n_samples:
            raise ValueError(f"validation_split={validation_split} leaves no training data "
                           f"for {n_samples} samples")
        
        # Shuffle indices
        indices = np.random.RandomState(42).permutation(n_samples)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        # Split data
        rep_dict_train = {name: X[train_idx] for name, X in base_reps.items()}
        rep_dict_val = {name: X[val_idx] for name, X in base_reps.items()}
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        # Run performance-weighted allocation
        allocation = allocate_performance_weighted(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget
        )
        
        # Compute importance on TRAIN split (consistent with allocation decision)
        importance_scores = compute_feature_importance(
            rep_dict_train, train_labels, method=importance_method, **kwargs
        )
        
        # Select features based on train importance, then apply to full data
        _, feature_info = concatenate_features(
            rep_dict_train, importance_scores, allocation
        )
        
        # If filters were applied, remap indices from filtered space to original space
        if filter_index_maps:
            for rep_name in list(feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in feature_info[rep_name]:
                    filtered_indices = feature_info[rep_name]['selected_indices']
                    original_indices = filter_index_maps[rep_name][filtered_indices]
                    feature_info[rep_name]['selected_indices'] = original_indices
        
        # Apply same feature selection to full data (use original if filters applied)
        hybrid_features = apply_feature_selection(original_base_reps, feature_info)
        
        # Add allocation to feature_info for reference
        feature_info['allocation'] = allocation
        
    # FIXED ALLOCATION (BACKWARDS COMPATIBLE)
    elif allocation_method == 'fixed':
        # Split data for consistent importance computation (avoid data leakage)
        n_samples = len(labels)
        n_val = int(n_samples * validation_split)
        n_val = max(n_val, 1)
        n_train = n_samples - n_val
        
        if n_train < 2:
            raise ValueError(f"Need at least 2 training samples for fixed allocation. "
                           f"Got {n_train} with validation_split={validation_split} and {n_samples} total samples.")
        if n_val >= n_samples:
            raise ValueError(f"validation_split={validation_split} leaves no training data "
                           f"for {n_samples} samples")
        
        indices = np.random.RandomState(42).permutation(n_samples)
        train_idx = indices[n_val:]
        
        # Split reps for importance computation
        rep_dict_train = {name: X[train_idx] for name, X in base_reps.items()}
        train_labels = labels[train_idx]
        
        # Compute importance on TRAIN split (consistent with other methods)
        importance_scores = compute_feature_importance(
            rep_dict_train, train_labels, method=importance_method, **kwargs
        )
        
        # Use same n_per_rep for all reps
        allocation = {rep: n_per_rep for rep in base_reps.keys()}
        
        # Select features based on train importance
        _, feature_info = concatenate_features(
            rep_dict_train, importance_scores, allocation
        )
        
        # If filters were applied, remap indices from filtered space to original space
        if filter_index_maps:
            for rep_name in list(feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in feature_info[rep_name]:
                    filtered_indices = feature_info[rep_name]['selected_indices']
                    original_indices = filter_index_maps[rep_name][filtered_indices]
                    feature_info[rep_name]['selected_indices'] = original_indices
        
        # Apply same feature selection to original data
        hybrid_features = apply_feature_selection(original_base_reps, feature_info)
        
        feature_info['allocation'] = allocation
    
    else:
        raise ValueError(f"Unknown allocation_method: {allocation_method}. "
                        f"Choose 'greedy', 'performance_weighted', or 'fixed'")
    
    return hybrid_features, feature_info