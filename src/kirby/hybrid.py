"""
Core hybrid representation creation functions - ENHANCED with Greedy Allocation

Supports multiple feature importance methods and allocation strategies:
- Allocation methods: 'fixed' (equal per rep), 'greedy' (adaptive), 'performance_weighted'
- Importance methods: random_forest, permutation, treeshap, integrated_gradients, kernelshap,
  drop_column, xgboost_gain/weight/cover, lightgbm_gain/split, deeplift, deepliftshap,
  gradientshap, lime, boruta
- Augmentation strategies: 'all', 'none', 'greedy_ablation'

Default configuration (from testing):
- allocation_method='greedy'
- budget=100
- step_size=10
- patience=3
- apply_filters=True (quality filters: sparsity + variance)

Note on removed methods:
- mutual_info was removed (2024-02) - univariate filter method that achieved only
  60-68% recall vs 92-96% for RF/permutation in synthetic benchmarks. Cannot capture
  feature interactions and has high variance with the k-NN estimator.
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


def allocate_greedy_feature(rep_dict_train, rep_dict_val, train_labels, val_labels,
                            total_budget=100, patience=5, importance_method='random_forest'):
    """
    Feature-level greedy allocation.

    Unlike rep-based greedy which adds chunks of features from the best rep,
    this adds one feature at a time from whichever rep has the best next feature.
    More precise but slower.

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        patience: Early stopping patience (default: 5)
        importance_method: Method for ranking features ('random_forest', 'permutation', 'shap', etc.)

    Returns:
        tuple: (allocation, history, feature_info)
    """
    # Use configurable importance method for ranking
    all_importance = compute_feature_importance(
        rep_dict_train, train_labels, method=importance_method
    )

    # Get importance ranking for each rep
    rep_importance = {}
    for rep_name, importances in all_importance.items():
        # Indices sorted by importance (descending)
        ranked_indices = np.argsort(importances)[::-1]
        rep_importance[rep_name] = {
            'ranked_indices': ranked_indices,
            'scores': importances,
            'next_idx': 0,  # Pointer to next feature to consider
        }

    # Track selected features per rep
    selected = {rep: [] for rep in rep_dict_train.keys()}

    best_val_score = -np.inf
    best_selected = None
    best_feature_info = None
    steps_without_improvement = 0
    history = []

    for iteration in range(total_budget):
        iter_best_score = -np.inf
        iter_best_rep = None
        iter_best_feature_idx = None

        # Try adding the next best feature from each rep
        for rep_name in rep_dict_train.keys():
            info = rep_importance[rep_name]

            # Skip if we've used all features from this rep
            if info['next_idx'] >= len(info['ranked_indices']):
                continue

            # Get next feature to try
            feature_idx = info['ranked_indices'][info['next_idx']]

            # Build trial feature set
            trial_selected = {r: list(s) for r, s in selected.items()}
            trial_selected[rep_name].append(feature_idx)

            # Create hybrid with trial selection
            X_train_trial, X_val_trial = _build_from_selection(
                rep_dict_train, rep_dict_val, trial_selected
            )

            if X_train_trial.shape[1] == 0:
                continue

            # Evaluate
            score = _evaluate_allocation(X_train_trial, X_val_trial, train_labels, val_labels)

            if score > iter_best_score:
                iter_best_score = score
                iter_best_rep = rep_name
                iter_best_feature_idx = feature_idx

        # No valid feature found
        if iter_best_rep is None:
            break

        # Commit the best feature
        selected[iter_best_rep].append(iter_best_feature_idx)
        rep_importance[iter_best_rep]['next_idx'] += 1

        # Track best overall
        if iter_best_score > best_val_score:
            best_val_score = iter_best_score
            best_selected = {r: list(s) for r, s in selected.items()}
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        history.append({
            'iteration': iteration + 1,
            'rep': iter_best_rep,
            'feature_idx': iter_best_feature_idx,
            'val_score': iter_best_score,
            'allocation': {r: len(s) for r, s in selected.items()},
        })

        # Early stopping
        if patience and steps_without_improvement >= patience:
            break

    if best_selected is None:
        raise ValueError("Greedy feature allocation found no valid features.")

    # Build final feature_info from best_selected
    feature_info = {}
    for rep_name, indices in best_selected.items():
        if len(indices) == 0:
            continue
        indices_arr = np.array(indices)
        scores = rep_importance[rep_name]['scores'][indices_arr]
        feature_info[rep_name] = {
            'selected_indices': indices_arr,
            'importance_scores': scores,
            'n_features': len(indices),
        }

    allocation = {r: len(s) for r, s in best_selected.items()}
    feature_info['allocation'] = allocation

    return allocation, history, feature_info


def _build_from_selection(rep_dict_train, rep_dict_val, selected):
    """Helper: Build train/val arrays from explicit feature selection."""
    train_parts = []
    val_parts = []

    for rep_name, indices in selected.items():
        if len(indices) == 0:
            continue
        train_parts.append(rep_dict_train[rep_name][:, indices])
        val_parts.append(rep_dict_val[rep_name][:, indices])

    if not train_parts:
        return np.array([]).reshape(rep_dict_train[list(rep_dict_train.keys())[0]].shape[0], 0), \
               np.array([]).reshape(rep_dict_val[list(rep_dict_val.keys())[0]].shape[0], 0)

    return np.hstack(train_parts), np.hstack(val_parts)


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


def allocate_mrmr(rep_dict_train, rep_dict_val, train_labels, val_labels,
                  total_budget=100):
    """
    Allocate features using minimum Redundancy Maximum Relevance (mRMR).

    mRMR balances two objectives:
    1. Maximum relevance: features should be highly correlated with the target
    2. Minimum redundancy: features should be uncorrelated with each other

    This helps select diverse, informative features and works well with
    linear models that suffer from multicollinearity.

    Reference: Peng et al. (2005) "Feature Selection Based on Mutual Information"

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)

    Returns:
        dict: Allocation of {rep_name: n_features}
    """
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

    is_classification = _is_classification_task(train_labels)

    # Concatenate all features to compute global mRMR
    all_features = []
    feature_to_rep = []  # Maps feature index to (rep_name, local_idx)

    for rep_name, X_train in rep_dict_train.items():
        X_clean = np.clip(X_train, -1e10, 1e10)
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=1e10, neginf=-1e10)

        for local_idx in range(X_clean.shape[1]):
            all_features.append(X_clean[:, local_idx])
            feature_to_rep.append((rep_name, local_idx))

    X_all = np.column_stack(all_features)
    n_total_features = X_all.shape[1]

    # Compute relevance (MI with target)
    if is_classification:
        relevance = mutual_info_classif(X_all, train_labels, random_state=42)
    else:
        relevance = mutual_info_regression(X_all, train_labels, random_state=42)

    # Greedy mRMR selection
    selected_indices = []
    remaining_indices = list(range(n_total_features))

    for _ in range(min(total_budget, n_total_features)):
        if not remaining_indices:
            break

        best_score = -np.inf
        best_idx = None

        for idx in remaining_indices:
            # Relevance term
            rel = relevance[idx]

            # Redundancy term (mean correlation with already selected)
            if selected_indices:
                redundancy = 0.0
                for sel_idx in selected_indices:
                    # Use absolute Pearson correlation as redundancy measure
                    corr = np.corrcoef(X_all[:, idx], X_all[:, sel_idx])[0, 1]
                    redundancy += np.abs(corr) if np.isfinite(corr) else 0.0
                redundancy /= len(selected_indices)
            else:
                redundancy = 0.0

            # mRMR score = relevance - redundancy
            score = rel - redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    # Convert selected indices to per-rep allocation
    allocation = {rep_name: 0 for rep_name in rep_dict_train.keys()}
    for idx in selected_indices:
        rep_name, _ = feature_to_rep[idx]
        allocation[rep_name] += 1

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
# AUGMENTATION HANDLING
# ============================================================================

def _select_augmentation_features(X_aug, labels, budget, is_classification):
    """
    Select top features from an augmentation array based on RF importance.
    
    Args:
        X_aug: Augmentation features (n_samples, n_features)
        labels: Target labels
        budget: Max features to select
        is_classification: Whether this is a classification task
        
    Returns:
        tuple: (selected_features, selected_indices, importance_scores)
    """
    # Clean data
    X_clean = np.clip(X_aug, -1e10, 1e10)
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=1e10, neginf=-1e10)
    
    n_features = X_clean.shape[1]
    n_select = min(budget, n_features)
    
    if n_select >= n_features:
        # Use all features
        return X_clean, np.arange(n_features), np.ones(n_features)
    
    # Compute importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    if is_classification:
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    model.fit(X_scaled, labels)
    importances = np.nan_to_num(model.feature_importances_, nan=0.0)
    
    # Select top features
    top_idx = np.argsort(importances)[-n_select:][::-1]
    
    return X_clean[:, top_idx], top_idx, importances[top_idx]


def _evaluate_augmentation_addition(
    X_base_train, X_base_val,
    X_aug_train, X_aug_val,
    y_train, y_val,
    baseline_score
):
    """
    Evaluate whether adding an augmentation improves performance.
    
    Args:
        X_base_train, X_base_val: Base hybrid features
        X_aug_train, X_aug_val: Augmentation features to test
        y_train, y_val: Labels
        baseline_score: Current validation score without augmentation
        
    Returns:
        tuple: (new_score, improvement)
    """
    # Combine base + augmentation
    X_train_combined = np.hstack([X_base_train, X_aug_train])
    X_val_combined = np.hstack([X_base_val, X_aug_val])
    
    new_score = _evaluate_allocation(X_train_combined, X_val_combined, y_train, y_val)
    improvement = new_score - baseline_score
    
    return new_score, improvement


def greedy_augmentation_ablation(
    X_base_train, X_base_val,
    augmentations_train, augmentations_val,
    y_train, y_val,
    augmentation_budget=20,
    threshold=None,
    return_all_scores=False
):
    """
    Greedily select augmentations that improve validation performance.
    
    For each augmentation:
    1. Select top `augmentation_budget` features by importance
    2. Test if adding improves validation score
    3. Keep if improvement exceeds threshold (or all if threshold=None)
    
    Args:
        X_base_train, X_base_val: Base hybrid features
        augmentations_train: Dict of {aug_name: train_features}
        augmentations_val: Dict of {aug_name: val_features}
        y_train, y_val: Labels
        augmentation_budget: Max features per augmentation (default: 20)
        threshold: Minimum improvement required to keep (default: None = keep all improving)
        return_all_scores: Return scores for all augmentations, not just kept ones
        
    Returns:
        dict: {
            'kept_augmentations': List of augmentation names that were kept,
            'augmentation_info': Dict of {aug_name: {'indices': ..., 'improvement': ...}},
            'all_scores': Dict of {aug_name: improvement} (if return_all_scores=True),
            'final_score': Final validation score with all kept augmentations
        }
    """
    is_classification = _is_classification_task(y_train)
    
    # Get baseline score (base hybrid only)
    baseline_score = _evaluate_allocation(X_base_train, X_base_val, y_train, y_val)
    
    kept_augmentations = []
    augmentation_info = {}
    all_scores = {}
    
    # Current best features (accumulates as we add augmentations)
    X_current_train = X_base_train.copy()
    X_current_val = X_base_val.copy()
    current_score = baseline_score
    
    # Test each augmentation
    for aug_name, X_aug_train in augmentations_train.items():
        X_aug_val = augmentations_val[aug_name]
        
        # Select top features from this augmentation
        X_aug_train_selected, indices, importances = _select_augmentation_features(
            X_aug_train, y_train, augmentation_budget, is_classification
        )
        X_aug_val_selected = X_aug_val[:, indices]
        X_aug_val_selected = np.clip(X_aug_val_selected, -1e10, 1e10)
        X_aug_val_selected = np.nan_to_num(X_aug_val_selected, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Test improvement
        new_score, improvement = _evaluate_augmentation_addition(
            X_current_train, X_current_val,
            X_aug_train_selected, X_aug_val_selected,
            y_train, y_val,
            current_score
        )
        
        all_scores[aug_name] = {
            'improvement': improvement,
            'new_score': new_score,
            'n_features': len(indices)
        }
        
        # Decide whether to keep
        keep = False
        if threshold is None:
            # Keep if any improvement
            keep = improvement > 0
        else:
            # Keep if improvement exceeds threshold
            keep = improvement >= threshold
        
        if keep:
            kept_augmentations.append(aug_name)
            augmentation_info[aug_name] = {
                'selected_indices': indices,
                'importance_scores': importances,
                'n_features': len(indices),
                'improvement': improvement
            }
            # Update current features
            X_current_train = np.hstack([X_current_train, X_aug_train_selected])
            X_current_val = np.hstack([X_current_val, X_aug_val_selected])
            current_score = new_score
    
    result = {
        'kept_augmentations': kept_augmentations,
        'augmentation_info': augmentation_info,
        'baseline_score': baseline_score,
        'final_score': current_score,
        'total_improvement': current_score - baseline_score
    }
    
    if return_all_scores:
        result['all_scores'] = all_scores
    
    return result


# ============================================================================
# FEATURE IMPORTANCE COMPUTATION
# ============================================================================

def compute_feature_importance(base_reps, labels, method='random_forest', model=None, **kwargs):
    """
    Compute feature importance scores for each representation.

    Can either train a model internally (for feature selection) or explain a
    pre-trained model (for model interpretation).

    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        labels: Target values (n_samples,) - used for training if model=None
        method: Method for computing importance:
            Tree-based methods:
            - 'random_forest' (default): MDI/Gini importance from RandomForest
            - 'treeshap': TreeSHAP values (handles correlated features better)
            - 'xgboost_gain': XGBoost gain (avg improvement per split)
            - 'xgboost_weight': XGBoost weight (split count)
            - 'xgboost_cover': XGBoost cover (samples covered)
            - 'lightgbm_gain': LightGBM gain importance
            - 'lightgbm_split': LightGBM split count
            Neural network methods:
            - 'integrated_gradients': Path integral attribution (Captum)
            - 'deeplift': DeepLIFT attribution (Captum)
            - 'deepliftshap': DeepLIFT + SHAP (Captum)
            - 'gradientshap': Gradient + SHAP sampling (Captum)
            Model-agnostic methods:
            - 'permutation': Permutation importance (stable, slower)
            - 'kernelshap': KernelSHAP (slow, any model)
            - 'lime': LIME local approximation
            - 'drop_column': Retrain without feature (gold standard, very slow)
            - 'boruta': Shadow feature selection (statistically sound, slow)
        model: Pre-trained model to explain (optional). If None, trains a new model.
            - For tree methods: sklearn RF/XGBoost/LightGBM model
            - For NN methods: PyTorch nn.Module
            - For model-agnostic: any model with predict/predict_proba
            Note: drop_column and boruta always train internally (cannot use pre-trained)
        **kwargs: Additional arguments for the importance method
            - n_estimators: Number of trees (default: 50)
            - max_depth: Max tree depth (default: 10)
            - random_state: Random seed (default: 42)
            - n_repeats: Permutation repeats, for 'permutation' only (default: 5)
            - hidden_dims: MLP hidden layers, for NN methods (default: [64, 32])
            - epochs: Training epochs, for NN methods (default: 100)
            - n_steps: Integration steps, for 'integrated_gradients' (default: 50)
            - background_samples: Background data size, for SHAP methods (default: 100)
            - nsamples: SHAP samples per instance, for 'kernelshap' (default: 'auto')
            - max_eval_samples: Max samples to evaluate (default: 100)
            - cv_folds: Cross-validation folds, for 'drop_column' (default: 3)
            - boruta_max_iter: Max Boruta iterations (default: 100)
            - boruta_perc: Boruta percentile threshold (default: 100)

    Returns:
        dict: Dictionary with representation names as keys and importance score arrays as values

    Example:
        # Feature selection (trains internally):
        scores = compute_feature_importance(reps, labels, method='treeshap')

        # Explain pre-trained model:
        my_model = XGBRegressor().fit(X, y)
        scores = compute_feature_importance(reps, labels, method='treeshap', model=my_model)

    Note:
        mutual_info was removed - it's a univariate filter method that performed
        poorly in testing (60-68% recall vs 92-96% for RF/permutation). It cannot
        capture feature interactions and has high variance with limited samples.
    """
    importance_scores = {}
    is_classification = _is_classification_task(labels)

    # RANDOM FOREST (DEFAULT)
    if method == 'random_forest':
        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            # Clean and scale data
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                rf = model
                # Validate model has feature_importances_
                if not hasattr(rf, 'feature_importances_'):
                    raise ValueError("Provided model must have feature_importances_ attribute for 'random_forest' method")
            else:
                if is_classification:
                    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               random_state=random_state, n_jobs=-1)
                else:
                    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                              random_state=random_state, n_jobs=-1)
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

            # Use provided model or train new one
            if model is not None:
                perm_model = model
            else:
                if is_classification:
                    perm_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                       random_state=random_state, n_jobs=-1)
                else:
                    perm_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                      random_state=random_state, n_jobs=-1)
                perm_model.fit(X_scaled, labels)

            perm = permutation_importance(perm_model, X_scaled, labels, n_repeats=n_repeats,
                                         random_state=random_state, n_jobs=-1)
            importance_scores[rep_name] = np.nan_to_num(perm.importances_mean, nan=0.0)

    # TREESHAP - SHAP values for tree-based models
    elif method == 'treeshap':
        try:
            import shap
        except ImportError:
            raise ImportError(
                "TreeSHAP requires the 'shap' package. "
                "Install with: conda install shap (recommended) or pip install shap"
            )

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            # Clean data
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                tree_model = model
            else:
                if is_classification:
                    tree_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                       random_state=random_state, n_jobs=-1)
                else:
                    tree_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                      random_state=random_state, n_jobs=-1)
                tree_model.fit(X_scaled, labels)

            # Compute SHAP values using TreeExplainer (fast, exact for trees)
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_scaled)

            # Handle different SHAP output formats:
            # - Older SHAP: list of arrays [class0_values, class1_values, ...]
            # - Newer SHAP (0.40+): 3D array (n_samples, n_features, n_classes)
            # - Regression: 2D array (n_samples, n_features)
            if is_classification:
                if isinstance(shap_values, list):
                    # Older SHAP format: list of arrays
                    if len(shap_values) == 2:
                        # Binary: use positive class
                        shap_values = shap_values[1]
                    else:
                        # Multiclass: average absolute values across classes
                        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                elif shap_values.ndim == 3:
                    # Newer SHAP format: (n_samples, n_features, n_classes)
                    if shap_values.shape[2] == 2:
                        # Binary: use positive class
                        shap_values = shap_values[:, :, 1]
                    else:
                        # Multiclass: average absolute values across classes
                        shap_values = np.mean(np.abs(shap_values), axis=2)

            # Feature importance = mean absolute SHAP value per feature
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # INTEGRATED GRADIENTS - Neural network attribution
    elif method == 'integrated_gradients':
        try:
            import torch
            import torch.nn as nn
            from captum.attr import IntegratedGradients
        except ImportError:
            raise ImportError(
                "Integrated Gradients requires 'torch' and 'captum' packages. "
                "Install with: pip install torch captum"
            )

        hidden_dims = kwargs.get('hidden_dims', [64, 32])
        epochs = kwargs.get('epochs', 100)
        lr = kwargs.get('lr', 0.001)
        n_steps = kwargs.get('n_steps', 50)
        random_state = kwargs.get('random_state', 42)
        n_classes_hint = kwargs.get('n_classes', None)  # For multiclass with pre-trained model

        torch.manual_seed(random_state)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            # Clean data
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Convert to tensors
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # Use provided model or train new one
            if model is not None:
                nn_model = model
                if not isinstance(nn_model, nn.Module):
                    raise ValueError("Provided model for integrated_gradients must be a PyTorch nn.Module")
                nn_model.eval()
                # Infer n_classes from model output
                with torch.no_grad():
                    sample_output = nn_model(X_tensor[:1])
                    if sample_output.dim() > 1 and sample_output.shape[1] > 1:
                        n_classes = sample_output.shape[1]
                    else:
                        n_classes = 2 if is_classification else 1
            else:
                # Build MLP
                input_dim = X_scaled.shape[1]
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
                    prev_dim = hidden_dim

                if is_classification:
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        # Binary classification: single output with sigmoid
                        layers.append(nn.Linear(prev_dim, 1))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                        criterion = nn.BCEWithLogitsLoss()
                    else:
                        # Multiclass: softmax output
                        layers.append(nn.Linear(prev_dim, n_classes))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.long)
                        criterion = nn.CrossEntropyLoss()
                else:
                    n_classes = 1
                    # Regression: single output
                    layers.append(nn.Linear(prev_dim, 1))
                    nn_model = nn.Sequential(*layers)
                    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                    criterion = nn.MSELoss()

                # Train the model
                optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
                nn_model.train()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = nn_model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()

                nn_model.eval()

            # Compute Integrated Gradients
            # For multi-output models, we need a wrapper that returns scalar
            if is_classification and n_classes > 2:
                # For multiclass, compute IG for each class and average
                all_attributions = []
                for class_idx in range(n_classes):
                    def forward_func(x, idx=class_idx):
                        return nn_model(x)[:, idx]

                    ig = IntegratedGradients(forward_func)
                    baseline = torch.zeros_like(X_tensor)
                    attr = ig.attribute(X_tensor, baselines=baseline, n_steps=n_steps)
                    all_attributions.append(attr.detach().numpy())

                # Average absolute attributions across classes
                attributions = np.mean([np.abs(a) for a in all_attributions], axis=0)
            else:
                # Binary classification or regression: single output
                def forward_func(x):
                    out = nn_model(x)
                    return out if out.dim() == 1 else out.squeeze(-1)

                ig = IntegratedGradients(forward_func)
                baseline = torch.zeros_like(X_tensor)
                attributions = ig.attribute(X_tensor, baselines=baseline, n_steps=n_steps)
                attributions = np.abs(attributions.detach().numpy())

            # Feature importance = mean absolute attribution per feature
            feature_importance = np.mean(attributions, axis=0)
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # KERNELSHAP - Model-agnostic SHAP (slow but works with any model)
    elif method == 'kernelshap':
        try:
            import shap
        except ImportError:
            raise ImportError(
                "KernelSHAP requires the 'shap' package. "
                "Install with: conda install shap (recommended) or pip install shap"
            )

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        background_samples = kwargs.get('background_samples', 100)
        nsamples = kwargs.get('nsamples', 'auto')
        max_eval_samples = kwargs.get('max_eval_samples', 100)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            # Clean data
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                kernel_model = model
                # Determine predict function based on model type
                if hasattr(kernel_model, 'predict_proba') and is_classification:
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        predict_fn = lambda x, m=kernel_model: m.predict_proba(x)[:, 1]
                    else:
                        predict_fn = kernel_model.predict_proba
                else:
                    predict_fn = kernel_model.predict
            else:
                # Train model (using RF as the underlying model)
                if is_classification:
                    kernel_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                         random_state=random_state, n_jobs=-1)
                    kernel_model.fit(X_scaled, labels)

                    # For classification, use predict_proba
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        # Binary: use probability of positive class
                        predict_fn = lambda x, m=kernel_model: m.predict_proba(x)[:, 1]
                    else:
                        # Multiclass: use full probability matrix
                        predict_fn = kernel_model.predict_proba
                else:
                    kernel_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                        random_state=random_state, n_jobs=-1)
                    kernel_model.fit(X_scaled, labels)
                    predict_fn = kernel_model.predict

            # Create background dataset (subsample for speed)
            n_background = min(background_samples, X_scaled.shape[0])
            background = shap.sample(X_scaled, n_background, random_state=random_state)

            # Create KernelExplainer
            explainer = shap.KernelExplainer(predict_fn, background)

            # Compute SHAP values (subsample evaluation data for speed)
            n_eval = min(max_eval_samples, X_scaled.shape[0])
            np.random.seed(random_state)
            eval_idx = np.random.choice(X_scaled.shape[0], n_eval, replace=False)
            X_eval = X_scaled[eval_idx]

            # Suppress progress bar for cleaner output
            shap_values = explainer.shap_values(X_eval, nsamples=nsamples, silent=True)

            # Handle multiclass output (list of arrays)
            if isinstance(shap_values, list):
                # Average absolute values across classes
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)

            # Feature importance = mean absolute SHAP value per feature
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # DROP-COLUMN IMPORTANCE - Retrain without each feature (gold standard)
    elif method == 'drop_column':
        from sklearn.model_selection import cross_val_score

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        cv_folds = kwargs.get('cv_folds', 3)

        for rep_name, X in base_reps.items():
            n_features = X.shape[1]
            if n_features == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            # Clean data
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Select model and scoring
            if is_classification:
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                              random_state=random_state, n_jobs=-1)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                             random_state=random_state, n_jobs=-1)
                scoring = 'r2'

            # Baseline score with all features
            baseline_score = cross_val_score(
                model, X_scaled, labels, cv=cv_folds, scoring=scoring, n_jobs=-1
            ).mean()

            # Drop each feature and measure performance drop
            feature_importance = np.zeros(n_features)

            for i in range(n_features):
                # Remove feature i
                X_dropped = np.delete(X_scaled, i, axis=1)

                # Handle edge case: single feature
                if X_dropped.shape[1] == 0:
                    # Dropping the only feature - importance is the baseline score
                    feature_importance[i] = baseline_score
                    continue

                # Score without feature i
                dropped_score = cross_val_score(
                    model, X_dropped, labels, cv=cv_folds, scoring=scoring, n_jobs=-1
                ).mean()

                # Importance = drop in performance (higher = more important)
                # Can be negative if removing feature improves performance
                feature_importance[i] = baseline_score - dropped_score

            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # XGBOOST GAIN - Average improvement per split
    elif method == 'xgboost_gain':
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost methods require 'xgboost'. Install with: pip install xgboost")

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 6)
        random_state = kwargs.get('random_state', 42)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                xgb_model = model
                if not hasattr(xgb_model, 'get_booster'):
                    raise ValueError("Provided model for xgboost_gain must be an XGBoost model with get_booster()")
            else:
                if is_classification:
                    xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                 random_state=random_state, n_jobs=-1, verbosity=0)
                else:
                    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                random_state=random_state, n_jobs=-1, verbosity=0)
                xgb_model.fit(X_scaled, labels)

            booster = xgb_model.get_booster()
            scores = booster.get_score(importance_type='gain')

            # Convert to array (XGBoost uses f0, f1, ... naming)
            feature_importance = np.zeros(X_scaled.shape[1])
            for fname, score in scores.items():
                idx = int(fname[1:])  # 'f0' -> 0
                feature_importance[idx] = score

            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # XGBOOST WEIGHT - Split count
    elif method == 'xgboost_weight':
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost methods require 'xgboost'. Install with: pip install xgboost")

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 6)
        random_state = kwargs.get('random_state', 42)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                xgb_model = model
                if not hasattr(xgb_model, 'get_booster'):
                    raise ValueError("Provided model for xgboost_weight must be an XGBoost model with get_booster()")
            else:
                if is_classification:
                    xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                 random_state=random_state, n_jobs=-1, verbosity=0)
                else:
                    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                random_state=random_state, n_jobs=-1, verbosity=0)
                xgb_model.fit(X_scaled, labels)

            booster = xgb_model.get_booster()
            scores = booster.get_score(importance_type='weight')

            feature_importance = np.zeros(X_scaled.shape[1])
            for fname, score in scores.items():
                idx = int(fname[1:])
                feature_importance[idx] = score

            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # XGBOOST COVER - Samples covered
    elif method == 'xgboost_cover':
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost methods require 'xgboost'. Install with: pip install xgboost")

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 6)
        random_state = kwargs.get('random_state', 42)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                xgb_model = model
                if not hasattr(xgb_model, 'get_booster'):
                    raise ValueError("Provided model for xgboost_cover must be an XGBoost model with get_booster()")
            else:
                if is_classification:
                    xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                 random_state=random_state, n_jobs=-1, verbosity=0)
                else:
                    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                random_state=random_state, n_jobs=-1, verbosity=0)
                xgb_model.fit(X_scaled, labels)

            booster = xgb_model.get_booster()
            scores = booster.get_score(importance_type='cover')

            feature_importance = np.zeros(X_scaled.shape[1])
            for fname, score in scores.items():
                idx = int(fname[1:])
                feature_importance[idx] = score

            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # LIGHTGBM GAIN
    elif method == 'lightgbm_gain':
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM methods require 'lightgbm'. Install with: pip install lightgbm")

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 6)
        random_state = kwargs.get('random_state', 42)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                lgb_model = model
                if not hasattr(lgb_model, 'booster_'):
                    raise ValueError("Provided model for lightgbm_gain must be a fitted LightGBM model with booster_")
            else:
                if is_classification:
                    lgb_model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                  random_state=random_state, n_jobs=-1, verbosity=-1)
                else:
                    lgb_model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                 random_state=random_state, n_jobs=-1, verbosity=-1)
                lgb_model.fit(X_scaled, labels)

            feature_importance = lgb_model.booster_.feature_importance(importance_type='gain')
            importance_scores[rep_name] = np.nan_to_num(feature_importance.astype(float), nan=0.0)

    # LIGHTGBM SPLIT
    elif method == 'lightgbm_split':
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM methods require 'lightgbm'. Install with: pip install lightgbm")

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 6)
        random_state = kwargs.get('random_state', 42)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                lgb_model = model
                if not hasattr(lgb_model, 'booster_'):
                    raise ValueError("Provided model for lightgbm_split must be a fitted LightGBM model with booster_")
            else:
                if is_classification:
                    lgb_model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                  random_state=random_state, n_jobs=-1, verbosity=-1)
                else:
                    lgb_model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                 random_state=random_state, n_jobs=-1, verbosity=-1)
                lgb_model.fit(X_scaled, labels)

            feature_importance = lgb_model.booster_.feature_importance(importance_type='split')
            importance_scores[rep_name] = np.nan_to_num(feature_importance.astype(float), nan=0.0)

    # DEEPLIFT - DeepLIFT attribution
    elif method == 'deeplift':
        try:
            import torch
            import torch.nn as nn
            from captum.attr import DeepLift
        except ImportError:
            raise ImportError("DeepLIFT requires 'torch' and 'captum'. Install with: pip install torch captum")

        hidden_dims = kwargs.get('hidden_dims', [64, 32])
        epochs = kwargs.get('epochs', 100)
        lr = kwargs.get('lr', 0.001)
        random_state = kwargs.get('random_state', 42)

        torch.manual_seed(random_state)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # Use provided model or train new one
            if model is not None:
                nn_model = model
                if not isinstance(nn_model, nn.Module):
                    raise ValueError("Provided model for deeplift must be a PyTorch nn.Module")
                nn_model.eval()
                # Infer n_classes from model output
                with torch.no_grad():
                    sample_output = nn_model(X_tensor[:1])
                    if sample_output.dim() > 1 and sample_output.shape[1] > 1:
                        n_classes = sample_output.shape[1]
                    else:
                        n_classes = 2 if is_classification else 1
            else:
                # Build MLP
                input_dim = X_scaled.shape[1]
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
                    prev_dim = hidden_dim

                if is_classification:
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        layers.append(nn.Linear(prev_dim, 1))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                        criterion = nn.BCEWithLogitsLoss()
                    else:
                        layers.append(nn.Linear(prev_dim, n_classes))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.long)
                        criterion = nn.CrossEntropyLoss()
                else:
                    n_classes = 1
                    layers.append(nn.Linear(prev_dim, 1))
                    nn_model = nn.Sequential(*layers)
                    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                    criterion = nn.MSELoss()

                optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
                nn_model.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = criterion(nn_model(X_tensor), y_tensor)
                    loss.backward()
                    optimizer.step()

                nn_model.eval()

            # DeepLIFT attribution - need to use model directly, not lambda
            baseline = torch.zeros_like(X_tensor)
            dl = DeepLift(nn_model)

            if is_classification and n_classes > 2:
                # Multiclass: compute attribution for each class and average
                all_attributions = []
                for class_idx in range(n_classes):
                    attr = dl.attribute(X_tensor, baselines=baseline, target=class_idx)
                    all_attributions.append(np.abs(attr.detach().numpy()))
                attributions = np.mean(all_attributions, axis=0)
            else:
                # Binary classification or regression
                attributions = dl.attribute(X_tensor, baselines=baseline)
                attributions = np.abs(attributions.detach().numpy())

            feature_importance = np.mean(attributions, axis=0)
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # DEEPLIFTSHAP - DeepLIFT + SHAP
    elif method == 'deepliftshap':
        try:
            import torch
            import torch.nn as nn
            from captum.attr import DeepLiftShap
        except ImportError:
            raise ImportError("DeepLiftSHAP requires 'torch' and 'captum'. Install with: pip install torch captum")

        hidden_dims = kwargs.get('hidden_dims', [64, 32])
        epochs = kwargs.get('epochs', 100)
        lr = kwargs.get('lr', 0.001)
        random_state = kwargs.get('random_state', 42)
        background_samples = kwargs.get('background_samples', 50)

        torch.manual_seed(random_state)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # Use provided model or train new one
            if model is not None:
                nn_model = model
                if not isinstance(nn_model, nn.Module):
                    raise ValueError("Provided model for deepliftshap must be a PyTorch nn.Module")
                nn_model.eval()
                # Infer n_classes from model output
                with torch.no_grad():
                    sample_output = nn_model(X_tensor[:1])
                    if sample_output.dim() > 1 and sample_output.shape[1] > 1:
                        n_classes = sample_output.shape[1]
                    else:
                        n_classes = 2 if is_classification else 1
            else:
                # Build MLP (same as deeplift)
                input_dim = X_scaled.shape[1]
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
                    prev_dim = hidden_dim

                if is_classification:
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        layers.append(nn.Linear(prev_dim, 1))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                        criterion = nn.BCEWithLogitsLoss()
                    else:
                        layers.append(nn.Linear(prev_dim, n_classes))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.long)
                        criterion = nn.CrossEntropyLoss()
                else:
                    n_classes = 1
                    layers.append(nn.Linear(prev_dim, 1))
                    nn_model = nn.Sequential(*layers)
                    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                    criterion = nn.MSELoss()

                optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
                nn_model.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = criterion(nn_model(X_tensor), y_tensor)
                    loss.backward()
                    optimizer.step()

                nn_model.eval()

            # Background samples for SHAP
            n_bg = min(background_samples, X_tensor.shape[0])
            np.random.seed(random_state)
            bg_idx = np.random.choice(X_tensor.shape[0], n_bg, replace=False)
            baseline = X_tensor[bg_idx]

            dls = DeepLiftShap(nn_model)

            if is_classification and n_classes > 2:
                all_attributions = []
                for class_idx in range(n_classes):
                    attr = dls.attribute(X_tensor, baselines=baseline, target=class_idx)
                    all_attributions.append(np.abs(attr.detach().numpy()))
                attributions = np.mean(all_attributions, axis=0)
            else:
                # Binary classification or regression
                attributions = dls.attribute(X_tensor, baselines=baseline)
                attributions = np.abs(attributions.detach().numpy())

            feature_importance = np.mean(attributions, axis=0)
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # GRADIENTSHAP - Gradient + SHAP sampling
    elif method == 'gradientshap':
        try:
            import torch
            import torch.nn as nn
            from captum.attr import GradientShap
        except ImportError:
            raise ImportError("GradientSHAP requires 'torch' and 'captum'. Install with: pip install torch captum")

        hidden_dims = kwargs.get('hidden_dims', [64, 32])
        epochs = kwargs.get('epochs', 100)
        lr = kwargs.get('lr', 0.001)
        random_state = kwargs.get('random_state', 42)
        background_samples = kwargs.get('background_samples', 50)
        n_samples = kwargs.get('n_samples', 50)

        torch.manual_seed(random_state)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # Use provided model or train new one
            if model is not None:
                nn_model = model
                if not isinstance(nn_model, nn.Module):
                    raise ValueError("Provided model for gradientshap must be a PyTorch nn.Module")
                nn_model.eval()
                # Infer n_classes from model output
                with torch.no_grad():
                    sample_output = nn_model(X_tensor[:1])
                    if sample_output.dim() > 1 and sample_output.shape[1] > 1:
                        n_classes = sample_output.shape[1]
                    else:
                        n_classes = 2 if is_classification else 1
            else:
                # Build MLP
                input_dim = X_scaled.shape[1]
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
                    prev_dim = hidden_dim

                if is_classification:
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        layers.append(nn.Linear(prev_dim, 1))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                        criterion = nn.BCEWithLogitsLoss()
                    else:
                        layers.append(nn.Linear(prev_dim, n_classes))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.long)
                        criterion = nn.CrossEntropyLoss()
                else:
                    n_classes = 1
                    layers.append(nn.Linear(prev_dim, 1))
                    nn_model = nn.Sequential(*layers)
                    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                    criterion = nn.MSELoss()

                optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
                nn_model.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = criterion(nn_model(X_tensor), y_tensor)
                    loss.backward()
                    optimizer.step()

                nn_model.eval()

            # Background samples
            n_bg = min(background_samples, X_tensor.shape[0])
            np.random.seed(random_state)
            bg_idx = np.random.choice(X_tensor.shape[0], n_bg, replace=False)
            baseline = X_tensor[bg_idx]

            if is_classification and n_classes > 2:
                all_attributions = []
                for class_idx in range(n_classes):
                    gs = GradientShap(lambda x, idx=class_idx, m=nn_model: m(x)[:, idx])
                    attr = gs.attribute(X_tensor, baselines=baseline, n_samples=n_samples,
                                       stdevs=0.0)
                    all_attributions.append(attr.detach().numpy())
                attributions = np.mean([np.abs(a) for a in all_attributions], axis=0)
            else:
                gs = GradientShap(lambda x, m=nn_model: m(x).squeeze(-1) if m(x).dim() > 1 else m(x))
                attributions = gs.attribute(X_tensor, baselines=baseline, n_samples=n_samples,
                                           stdevs=0.0)
                attributions = np.abs(attributions.detach().numpy())

            feature_importance = np.mean(attributions, axis=0)
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # LIME - Local Interpretable Model-agnostic Explanations
    elif method == 'lime':
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            raise ImportError("LIME requires 'lime'. Install with: pip install lime")

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        max_eval_samples = kwargs.get('max_eval_samples', 100)
        num_features = kwargs.get('num_features', None)  # None = all features

        for rep_name, X in base_reps.items():
            n_features = X.shape[1]
            if n_features == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                lime_model = model
                # Determine predict function and mode
                if hasattr(lime_model, 'predict_proba') and is_classification:
                    predict_fn = lime_model.predict_proba
                    mode = 'classification'
                else:
                    predict_fn = lime_model.predict
                    mode = 'regression'
            else:
                # Train model
                if is_classification:
                    lime_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                       random_state=random_state, n_jobs=-1)
                    lime_model.fit(X_scaled, labels)
                    predict_fn = lime_model.predict_proba
                    mode = 'classification'
                else:
                    lime_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                      random_state=random_state, n_jobs=-1)
                    lime_model.fit(X_scaled, labels)
                    predict_fn = lime_model.predict
                    mode = 'regression'

            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_scaled,
                mode=mode,
                random_state=random_state
            )

            # Compute explanations for subset of samples
            n_eval = min(max_eval_samples, X_scaled.shape[0])
            np.random.seed(random_state)
            eval_idx = np.random.choice(X_scaled.shape[0], n_eval, replace=False)

            # Aggregate feature importance across samples
            feature_importance = np.zeros(n_features)
            n_feats = num_features if num_features else n_features

            for idx in eval_idx:
                exp = explainer.explain_instance(
                    X_scaled[idx], predict_fn,
                    num_features=n_feats
                )
                # exp.as_list() returns [(feature_idx, weight), ...]
                for feat_idx, weight in exp.local_exp[1 if is_classification else 0]:
                    feature_importance[feat_idx] += abs(weight)

            feature_importance /= n_eval  # Average
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # BORUTA - Shadow feature selection
    elif method == 'boruta':
        try:
            from boruta import BorutaPy
        except ImportError:
            raise ImportError("Boruta requires 'boruta'. Install with: pip install boruta")

        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 5)
        random_state = kwargs.get('random_state', 42)
        max_iter = kwargs.get('boruta_max_iter', 100)
        perc = kwargs.get('boruta_perc', 100)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Boruta uses RF internally
            if is_classification:
                rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           random_state=random_state, n_jobs=-1)
            else:
                rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                          random_state=random_state, n_jobs=-1)

            # Run Boruta
            boruta = BorutaPy(rf, n_estimators='auto', max_iter=max_iter, perc=perc,
                             random_state=random_state, verbose=0)
            boruta.fit(X_scaled, np.asarray(labels))

            # ranking_: 1 = confirmed important, 2+ = tentative/rejected (lower = better)
            # Convert to importance: importance = max_rank - rank + 1
            max_rank = boruta.ranking_.max()
            feature_importance = (max_rank - boruta.ranking_ + 1).astype(float)

            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    else:
        raise NotImplementedError(
            f"Method '{method}' not implemented. Choose from: "
            "random_forest, permutation, treeshap, integrated_gradients, kernelshap, drop_column, "
            "xgboost_gain, xgboost_weight, xgboost_cover, lightgbm_gain, lightgbm_split, "
            "deeplift, deepliftshap, gradientshap, lime, boruta"
        )
    
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
        # Skip metadata keys like 'allocation', 'greedy_history', 'augmentation_info'
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


def apply_augmentation_selection(augmentations, augmentation_info):
    """
    Apply pre-computed augmentation selection to new data (e.g., test set).
    
    Args:
        augmentations: Dict of {aug_name: features}
        augmentation_info: Dict from create_hybrid containing selected_indices for each aug
        
    Returns:
        np.ndarray: Concatenated selected augmentation features
    """
    if not augmentation_info:
        return None
    
    selected_parts = []
    
    for aug_name, info in augmentation_info.items():
        if aug_name not in augmentations:
            raise ValueError(f"Augmentation '{aug_name}' found in augmentation_info but not in augmentations")
        
        X = augmentations[aug_name]
        indices = info['selected_indices']
        
        # Check bounds
        max_idx = np.max(indices) if len(indices) > 0 else -1
        if max_idx >= X.shape[1]:
            raise ValueError(
                f"Feature index {max_idx} out of bounds for augmentation '{aug_name}' "
                f"with {X.shape[1]} features."
            )
        
        X_selected = X[:, indices]
        X_selected = np.clip(X_selected, -1e10, 1e10)
        X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=1e10, neginf=-1e10)
        
        selected_parts.append(X_selected)
    
    if len(selected_parts) == 0:
        return None
    
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
                 # Augmentation parameters
                 augmentations=None,
                 augmentation_strategy='greedy_ablation',
                 augmentation_threshold=None,
                 augmentation_budget=20,
                 **kwargs):
    """
    Create hybrid representation by combining features from multiple base representations.
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        labels: Target values for computing importance
        
        allocation_method: Method for allocating features across representations
            - 'greedy': Adaptive allocation via greedy forward selection (default)
            - 'greedy_feature': Feature-level greedy (slower but more precise)
            - 'performance_weighted': Allocate proportional to baseline R² scores
            - 'mrmr': Minimum Redundancy Maximum Relevance allocation
            - 'fixed': Equal allocation (n_per_rep for all)
            
        n_per_rep: Number of features per rep (for 'fixed' method, default: 100)
        budget: Total feature budget (for 'greedy' method, default: 100)
        step_size: Features to add per greedy iteration (default: 10)
        patience: Early stopping patience for greedy, None to disable (default: 3)
        validation_split: Fraction for validation in greedy (default: 0.2)
        
        apply_filters: Apply quality filters (sparsity + variance) before allocation (default: False)
        importance_method: Method for feature importance (default: 'random_forest')
        
        augmentations: Dict of {aug_name: feature_array} for supplementary features (default: None)
            These are evaluated separately from base_reps via greedy ablation
        augmentation_strategy: How to handle augmentations
            - 'greedy_ablation': Test each, keep only those that improve (default)
            - 'all': Include all augmentations
            - 'none': Ignore augmentations
        augmentation_threshold: Minimum improvement to keep augmentation (default: None = any improvement)
            Can be float (e.g., 0.01 for 1% R² improvement) or None
        augmentation_budget: Max features per augmentation to select (default: 20)
        
        **kwargs: Additional arguments for importance computation
        
    Returns:
        tuple: (hybrid_features, feature_info)
            - hybrid_features: Combined feature array
            - feature_info: Metadata about selected features (includes 'allocation' for greedy,
                           'augmentation_info' if augmentations used)
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
    
    # Validate augmentations if provided
    if augmentations is not None:
        for aug_name, X_aug in augmentations.items():
            if X_aug.shape[0] != n_samples:
                raise ValueError(f"Augmentation '{aug_name}' has {X_aug.shape[0]} samples, expected {n_samples}")
            if X_aug.shape[1] == 0:
                raise ValueError(f"Augmentation '{aug_name}' has 0 features")
    
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
    if augmentation_budget <= 0:
        raise ValueError(f"augmentation_budget must be positive, got {augmentation_budget}")
    
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
    
    # Common setup: create train/val split
    n_val = int(n_samples * validation_split)
    n_val = max(n_val, 1)
    n_train = n_samples - n_val
    
    if n_train < 2:
        raise ValueError(f"Need at least 2 training samples. "
                       f"Got {n_train} with validation_split={validation_split} and {n_samples} total samples.")
    
    indices = np.random.RandomState(42).permutation(n_samples)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    
    # Split base reps
    rep_dict_train = {name: X[train_idx] for name, X in base_reps.items()}
    rep_dict_val = {name: X[val_idx] for name, X in base_reps.items()}
    
    # Split augmentations if provided
    aug_dict_train = None
    aug_dict_val = None
    if augmentations is not None and augmentation_strategy != 'none':
        aug_dict_train = {name: X[train_idx] for name, X in augmentations.items()}
        aug_dict_val = {name: X[val_idx] for name, X in augmentations.items()}
    
    # ========================================================================
    # BASE HYBRID CREATION
    # ========================================================================
    
    # GREEDY ALLOCATION
    if allocation_method == 'greedy':
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

    # MRMR ALLOCATION
    elif allocation_method == 'mrmr':
        # Run mRMR allocation
        allocation = allocate_mrmr(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget
        )

        # Compute importance on TRAIN split (using RF for feature selection within allocation)
        importance_scores = compute_feature_importance(
            rep_dict_train, train_labels, method=importance_method, **kwargs
        )

        # Select features based on train importance, then apply to full data
        _, feature_info = concatenate_features(
            rep_dict_train, importance_scores, allocation
        )

        # If filters were applied, remap indices
        if filter_index_maps:
            for rep_name in list(feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in feature_info[rep_name]:
                    filtered_indices = feature_info[rep_name]['selected_indices']
                    original_indices = filter_index_maps[rep_name][filtered_indices]
                    feature_info[rep_name]['selected_indices'] = original_indices

        # Apply same feature selection to original data
        hybrid_features = apply_feature_selection(original_base_reps, feature_info)

        feature_info['allocation'] = allocation

    # GREEDY FEATURE ALLOCATION (feature-level, high computation)
    elif allocation_method == 'greedy_feature':
        # Run feature-level greedy allocation
        allocation, history, greedy_feature_info = allocate_greedy_feature(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget, patience=patience,
            importance_method=importance_method
        )

        # If filters were applied, remap indices
        if filter_index_maps:
            for rep_name in list(greedy_feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in greedy_feature_info[rep_name]:
                    filtered_indices = greedy_feature_info[rep_name]['selected_indices']
                    original_indices = filter_index_maps[rep_name][filtered_indices]
                    greedy_feature_info[rep_name]['selected_indices'] = original_indices

        # Apply the SAME feature selection from greedy to full data
        hybrid_features = apply_feature_selection(original_base_reps, greedy_feature_info)

        # Build feature_info with allocation metadata
        feature_info = greedy_feature_info.copy()
        feature_info['allocation'] = allocation
        feature_info['greedy_feature_history'] = history

    else:
        raise ValueError(f"Unknown allocation_method: {allocation_method}. "
                        f"Choose 'greedy', 'greedy_feature', 'performance_weighted', 'fixed', or 'mrmr'")
    
    # ========================================================================
    # AUGMENTATION HANDLING
    # ========================================================================
    
    if aug_dict_train is not None and augmentation_strategy != 'none':
        # Get base hybrid for train/val
        X_base_train = apply_feature_selection(
            {name: X[train_idx] for name, X in original_base_reps.items()},
            feature_info
        )
        X_base_val = apply_feature_selection(
            {name: X[val_idx] for name, X in original_base_reps.items()},
            feature_info
        )
        
        if augmentation_strategy == 'greedy_ablation':
            # Run greedy ablation
            aug_result = greedy_augmentation_ablation(
                X_base_train, X_base_val,
                aug_dict_train, aug_dict_val,
                train_labels, val_labels,
                augmentation_budget=augmentation_budget,
                threshold=augmentation_threshold,
                return_all_scores=True
            )
            
            feature_info['augmentation_info'] = aug_result['augmentation_info']
            feature_info['augmentation_scores'] = aug_result['all_scores']
            feature_info['augmentation_baseline'] = aug_result['baseline_score']
            feature_info['augmentation_final'] = aug_result['final_score']
            feature_info['kept_augmentations'] = aug_result['kept_augmentations']
            
            # Apply kept augmentations to full data
            if aug_result['kept_augmentations']:
                aug_features = apply_augmentation_selection(
                    augmentations, aug_result['augmentation_info']
                )
                if aug_features is not None:
                    hybrid_features = np.hstack([hybrid_features, aug_features])
        
        elif augmentation_strategy == 'all':
            # Include all augmentations (select top features from each)
            is_classification = _is_classification_task(labels)
            augmentation_info = {}
            aug_parts = []
            
            for aug_name, X_aug in augmentations.items():
                X_selected, indices, importances = _select_augmentation_features(
                    X_aug, labels, augmentation_budget, is_classification
                )
                augmentation_info[aug_name] = {
                    'selected_indices': indices,
                    'importance_scores': importances,
                    'n_features': len(indices)
                }
                aug_parts.append(X_selected)
            
            feature_info['augmentation_info'] = augmentation_info
            feature_info['kept_augmentations'] = list(augmentations.keys())
            
            if aug_parts:
                hybrid_features = np.hstack([hybrid_features] + aug_parts)
    
    return hybrid_features, feature_info