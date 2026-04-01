"""
Core hybrid representation creation functions - ENHANCED with Greedy Allocation

Supports multiple feature importance methods and allocation strategies:
- Allocation methods: 'greedy', 'performance_weighted', 'dominant_greedy', 'fixed',
  plus named variants — see NAMED_ALLOCATIONS for full list.
- Importance methods (8 approved after benchmark correlation analysis):
  Fast: treeshap, xgboost_gain, lightgbm_gain, deeplift
  Slower: permutation, kernelshap
  Long search only: integrated_gradients, gradientshap
- Augmentation strategies: 'all', 'none', 'greedy_ablation'

Importance-sensitive vs insensitive allocations:
  'greedy' and 'dominant_greedy' use internal RF feature_importances_ for
  feature selection, so they produce identical results regardless of the
  importance_method parameter. All other allocation methods (performance_weighted,
  and all named performance_weighted variants) use the specified importance_method
  and will produce different results per method.
  find_best_hybrid accounts for this — insensitive allocations are only evaluated
  once per (budget, model) rather than N times for N importance methods.

Default configuration (from F12 top-2 analysis, 2026-03-29):
- allocation_method='top_3'
- importance_method='lightgbm_gain'
- internal_evaluator='svm'
- budget=100
"""

from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================================
# NAMED ALLOCATION REGISTRY
# ============================================================================

NAMED_ALLOCATIONS = {
    # --- Base methods ---
    'greedy': ('greedy', {}),
    'performance_weighted': ('performance_weighted', {}),
    'fixed': ('fixed', {}),

    # --- Softmax temperature variants ---
    'softmax_T0.5': ('performance_weighted', {'weighting_scheme': 'softmax', 'weighting_params': {'temperature': 0.5}}),
    'softmax_T1': ('performance_weighted', {'weighting_scheme': 'softmax', 'weighting_params': {'temperature': 1.0}}),
    'softmax_T2': ('performance_weighted', {'weighting_scheme': 'softmax', 'weighting_params': {'temperature': 2.0}}),
    'softmax_T5': ('performance_weighted', {'weighting_scheme': 'softmax', 'weighting_params': {'temperature': 5.0}}),
    'softmax_T10': ('performance_weighted', {'weighting_scheme': 'softmax', 'weighting_params': {'temperature': 10.0}}),

    # --- Power law variants ---
    'power_2': ('performance_weighted', {'weighting_scheme': 'power_law', 'weighting_params': {'power': 2.0}}),
    'power_3': ('performance_weighted', {'weighting_scheme': 'power_law', 'weighting_params': {'power': 3.0}}),

    # --- Thresholded variants ---
    'thresh_0.3': ('performance_weighted', {'weighting_scheme': 'thresholded', 'weighting_params': {'alpha': 0.3}}),
    'thresh_0.5': ('performance_weighted', {'weighting_scheme': 'thresholded', 'weighting_params': {'alpha': 0.5}}),
    'thresh_0.7': ('performance_weighted', {'weighting_scheme': 'thresholded', 'weighting_params': {'alpha': 0.7}}),

    # --- Top-k variants ---
    'top_2': ('performance_weighted', {'weighting_scheme': 'top_k', 'weighting_params': {'k': 2}}),
    'top_3': ('performance_weighted', {'weighting_scheme': 'top_k', 'weighting_params': {'k': 3}}),
    'top_5': ('performance_weighted', {'weighting_scheme': 'top_k', 'weighting_params': {'k': 5}}),

    # --- Rank-based variants ---
    'rank_linear': ('performance_weighted', {'weighting_scheme': 'rank_linear'}),
    'rank_exp_d0.1': ('performance_weighted', {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.1}}),
    'rank_exp_d0.3': ('performance_weighted', {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.3}}),
    'rank_exp_d0.5': ('performance_weighted', {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.5}}),

    # --- Score gap variants ---
    'score_gap_0.2': ('performance_weighted', {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.2}}),
    'score_gap_0.3': ('performance_weighted', {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.3}}),
    'score_gap_0.5': ('performance_weighted', {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.5}}),
    'score_gap_0.7': ('performance_weighted', {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.7}}),

    # --- Entropy variants ---
    'entropy_0.5': ('performance_weighted', {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 0.5}}),
    'entropy_1.0': ('performance_weighted', {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 1.0}}),
    'entropy_2.0': ('performance_weighted', {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 2.0}}),

    # --- Greedy with early stopping (forced = no patience, fills budget) ---
    'greedy_forced_s5': ('greedy', {'step_size': 5, 'patience': None}),
    'greedy_forced_s10': ('greedy', {'step_size': 10, 'patience': None}),
    'greedy_forced_s25': ('greedy', {'step_size': 25, 'patience': None}),

    # --- Greedy with budget fill (patience + fills remaining budget from best reps) ---
    'greedy_fill_s5': ('greedy', {'step_size': 5, 'budget_fill': True}),
    'greedy_fill_s10': ('greedy', {'step_size': 10, 'budget_fill': True}),
    'greedy_fill_s25': ('greedy', {'step_size': 25, 'budget_fill': True}),

    # --- Dominant greedy (F12: dom_greedy_f0.7 and f0.9 retained, f0.5 cut) ---
    'dominant_greedy': ('dominant_greedy', {}),
    'dom_greedy_f0.7': ('dominant_greedy', {'dominant_fraction': 0.7}),
    'dom_greedy_f0.9': ('dominant_greedy', {'dominant_fraction': 0.9}),

    # Legacy aliases for backward compatibility with old experiment configs
    'rank_exp': ('performance_weighted', {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.3}}),
    'score_gap': ('performance_weighted', {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.3}}),
    'entropy': ('performance_weighted', {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 1.0}}),
}


# ============================================================================
# GREEDY ALLOCATION
# ============================================================================

def allocate_greedy_forward(rep_dict_train, rep_dict_val, train_labels, val_labels,
                           total_budget=100, step_size=10, patience=3,
                           internal_evaluator='ridge',
                           model=None, metric_fn=None, lower_is_better=False,
                           greedy_tolerance=0.0):
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
        model: Optional sklearn estimator for metric-aware evaluation.
        metric_fn: Optional callable(y_true, y_pred) -> float.
        lower_is_better: Whether lower metric values are better (default: False).
        greedy_tolerance: Absolute tolerance for patience tracking (default: 0.0).
            When > 0, iterations that score within tolerance of the best are
            treated as non-degrading and reset the patience counter.

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

    # Cache RF importance scores per rep — computed once, reused every iteration.
    # Importance depends only on (rep features, labels) which are constant across
    # the greedy loop. The cache is local to this call and discarded on return.
    _importance_cache = {}

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
                rep_dict_train, train_labels, trial_allocation,
                importance_cache=_importance_cache,
            )
            X_val_trial = apply_feature_selection(rep_dict_val, feat_info)
            
            # Evaluate on validation
            r2 = _evaluate_allocation(X_train_trial, X_val_trial, train_labels, val_labels,
                                     model=model, metric_fn=metric_fn,
                                     lower_is_better=lower_is_better,
                                     internal_evaluator=internal_evaluator)

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
        elif greedy_tolerance > 0 and iter_best_gain >= best_val_r2 - greedy_tolerance:
            # Within tolerance of best — don't count as degradation
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
                            total_budget=100, patience=5, importance_method='treeshap',
                            internal_evaluator='ridge',
                            model=None, metric_fn=None, lower_is_better=False):
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
        importance_method: Method for ranking features ('treeshap', 'permutation', etc.)

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
            score = _evaluate_allocation(X_train_trial, X_val_trial, train_labels, val_labels,
                                         model=model, metric_fn=metric_fn,
                                         lower_is_better=lower_is_better,
                                         internal_evaluator=internal_evaluator)

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


def _compute_allocation_weights(baseline_scores, scheme='linear', params=None,
                                rep_dict_train=None):
    """Compute allocation weights from baseline scores using various schemes.

    Args:
        baseline_scores: Dict of {rep_name: score} (already clipped to >= 0).
        scheme: Weighting scheme name.
        params: Dict of scheme-specific parameters.
        rep_dict_train: Optional dict of {rep_name: train_features}. Required
            only for schemes that need the raw features (e.g. 'entropy').

    Returns:
        Dict of {rep_name: weight} (non-negative, not necessarily normalised).
    """
    params = params or {}

    if scheme == 'linear':
        return dict(baseline_scores)

    elif scheme == 'softmax':
        temperature = params.get('temperature', 1.0)
        scores = np.array(list(baseline_scores.values()))
        # Subtract max for numerical stability
        exp_scores = np.exp((scores - scores.max()) / max(temperature, 1e-10))
        return {k: float(w) for k, w in zip(baseline_scores.keys(), exp_scores)}

    elif scheme == 'power_law':
        power = params.get('power', 2.0)
        return {k: float(v ** power) for k, v in baseline_scores.items()}

    elif scheme == 'thresholded':
        alpha = params.get('alpha', 0.5)
        max_score = max(baseline_scores.values()) if baseline_scores else 0
        threshold = alpha * max_score
        return {k: (v if v >= threshold else 0.0) for k, v in baseline_scores.items()}

    elif scheme == 'top_k':
        k = params.get('k', 3)
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1], reverse=True)
        top_names = {name for name, _ in sorted_reps[:k]}
        return {name: (score if name in top_names else 0.0)
                for name, score in baseline_scores.items()}

    elif scheme == 'min_floor':
        # Special: handled in allocate_performance_weighted directly
        return dict(baseline_scores)

    elif scheme == 'rank_linear':
        # Weights by rank position: best rep gets n, worst gets 1.
        # Score magnitudes are ignored — prevents clusters of similar scores
        # from diluting budget across mediocre reps.
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1])
        return {name: float(rank) for rank, (name, _)
                in enumerate(sorted_reps, start=1)}

    elif scheme == 'rank_exp':
        # Exponential rank decay: exp(-rank * decay).  Concentrates budget on
        # top few reps with smooth falloff — no hard cutoff like top_k but
        # more aggressive than rank_linear.  decay controls steepness.
        decay = params.get('decay', 0.3)
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1],
                             reverse=True)
        return {name: float(np.exp(-rank * decay))
                for rank, (name, _) in enumerate(sorted_reps)}

    elif scheme == 'score_gap':
        # Zero out any rep whose score is more than `gap` below the best.
        # gap is a fraction of best_score (e.g. 0.3 = within 30%).
        gap_frac = params.get('gap', 0.3)
        best_score = max(baseline_scores.values()) if baseline_scores else 0
        gap = gap_frac * best_score if best_score > 0 else 0
        return {k: (v if best_score - v <= gap else 0.0)
                for k, v in baseline_scores.items()}

    elif scheme == 'entropy':
        # Penalize reps that are redundant with higher-scoring reps.
        # Walk reps in descending score order. Each rep's weight is its
        # baseline score multiplied by (1 - max_correlation_with_any_
        # already-accepted_rep).  This down-weights reps that duplicate
        # information already captured by a better rep.
        #
        # Redundancy is measured via correlation of sample-wise L2 norms
        # (always n_samples-length, regardless of feature dimensionality).
        if rep_dict_train is None:
            return dict(baseline_scores)
        diversity_penalty = params.get('diversity_penalty', 1.0)
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1],
                             reverse=True)
        accepted_norms = []  # list of (name, sample_norms_array)
        weights = {}
        for name, score in sorted_reps:
            X = rep_dict_train[name]
            X_clean = np.nan_to_num(np.clip(X, -1e10, 1e10), nan=0.0)
            sample_norms = np.linalg.norm(X_clean, axis=1)
            std = np.std(sample_norms)
            if std > 0:
                sample_norms = (sample_norms - sample_norms.mean()) / std

            if not accepted_norms:
                max_corr = 0.0
            else:
                correlations = []
                for _, prev_norms in accepted_norms:
                    r = np.corrcoef(sample_norms, prev_norms)[0, 1]
                    correlations.append(abs(r) if not np.isnan(r) else 0.0)
                max_corr = max(correlations)

            penalty = max(0.0, 1.0 - diversity_penalty * max_corr)
            weights[name] = score * penalty
            accepted_norms.append((name, sample_norms))
        return weights

    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")


def allocate_performance_weighted(rep_dict_train, rep_dict_val, train_labels, val_labels,
                                  total_budget=100, weighting_scheme='linear',
                                  weighting_params=None, internal_evaluator='ridge',
                                  precomputed_scores=None,
                                  model=None, metric_fn=None, lower_is_better=False):
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
        weighting_scheme: How to transform baseline scores into weights.
            'linear' (default, identical to original), 'softmax', 'power_law',
            'thresholded', 'top_k', 'min_floor', 'rank_linear', 'exponential',
            'score_gap', 'entropy'.
        weighting_params: Dict of scheme-specific parameters.
        precomputed_scores: Pre-computed {rep_name: score} dict (optional).
            If provided, skips per-rep model fitting entirely.

    Returns:
        dict: Allocation of {rep_name: n_features}
    """
    if precomputed_scores is not None:
        baseline_scores = {rn: precomputed_scores.get(rn, 0.0)
                           for rn in rep_dict_train}
    else:
        baseline_scores = {}

        # Test each representation individually using _evaluate_allocation
        # so that monkey-patching the evaluator (e.g. F11 experiments) affects
        # performance_weighted uniformly.
        for rep_name, X_train in rep_dict_train.items():
            X_val = rep_dict_val[rep_name]

            # Clean
            X_train_clean = np.clip(X_train, -1e10, 1e10)
            X_train_clean = np.nan_to_num(X_train_clean, nan=0.0, posinf=1e10, neginf=-1e10)
            X_val_clean = np.clip(X_val, -1e10, 1e10)
            X_val_clean = np.nan_to_num(X_val_clean, nan=0.0, posinf=1e10, neginf=-1e10)

            score = _evaluate_allocation(X_train_clean, X_val_clean,
                                         train_labels, val_labels,
                                         model=model, metric_fn=metric_fn,
                                         lower_is_better=lower_is_better,
                                         internal_evaluator=internal_evaluator)

            baseline_scores[rep_name] = max(score, 0.0)  # Clip negative scores to 0
    
    # Apply weighting scheme to transform baseline scores into weights
    weights = _compute_allocation_weights(baseline_scores, weighting_scheme,
                                          weighting_params,
                                          rep_dict_train=rep_dict_train)

    # Handle min_floor scheme: guarantee minimum allocation per rep
    if weighting_scheme == 'min_floor':
        floor = (weighting_params or {}).get('floor', 0.05)
        n_reps = len(weights)
        floor_per_rep = max(1, int(floor * total_budget))
        floor_total = floor_per_rep * n_reps
        remainder_budget = max(0, total_budget - floor_total)

        total_w = sum(weights.values())
        allocation = {}
        for rep_name, w in weights.items():
            max_features = rep_dict_train[rep_name].shape[1]
            prop = int(round((w / total_w) * remainder_budget)) if total_w > 0 else 0
            allocation[rep_name] = min(floor_per_rep + prop, max_features)

        # Adjust to hit budget
        current_total = sum(allocation.values())
        if current_total < total_budget:
            sorted_reps = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            remaining = total_budget - current_total
            for rep_name, _ in sorted_reps:
                if remaining <= 0:
                    break
                can_add = rep_dict_train[rep_name].shape[1] - allocation[rep_name]
                add_amount = min(can_add, remaining)
                allocation[rep_name] += add_amount
                remaining -= add_amount
        elif current_total > total_budget:
            sorted_reps = sorted(weights.items(), key=lambda x: x[1])
            excess = current_total - total_budget
            for rep_name, _ in sorted_reps:
                if excess <= 0:
                    break
                reduce_by = min(allocation[rep_name] - floor_per_rep, excess)
                if reduce_by > 0:
                    allocation[rep_name] -= reduce_by
                    excess -= reduce_by
        return allocation

    # Standard path: proportional allocation from weights
    total_w = sum(weights.values())

    if total_w == 0:
        # All weights zero — fall back to equal allocation
        n_reps = len(weights)
        base_alloc = total_budget // n_reps
        remainder = total_budget - base_alloc * n_reps
        allocation = {}
        for i, rep_name in enumerate(sorted(weights.keys())):
            alloc = base_alloc + (1 if i < remainder else 0)
            max_features = rep_dict_train[rep_name].shape[1]
            allocation[rep_name] = min(alloc, max_features)
        return allocation

    allocation = {}
    for rep_name, w in weights.items():
        n_features = int(round((w / total_w) * total_budget))
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


def allocate_dominant_greedy(rep_dict_train, rep_dict_val, train_labels, val_labels,
                             total_budget=100, step_size=10, patience=3,
                             dominant_fraction=0.7, internal_evaluator='ridge',
                             model=None, metric_fn=None, lower_is_better=False,
                             greedy_tolerance=0.0):
    """
    Dominant-greedy allocation: give the best rep a large share, greedy-fill the rest.

    1. Evaluate each rep individually (reuses _evaluate_allocation helper).
    2. Give the best rep int(total_budget * dominant_fraction) features.
    3. Greedy-fill remaining budget: iteratively try adding step_size features from
       each other rep, keep whichever improves validation most.
    4. Patience-based early stopping on greedy phase.

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        step_size: Features to add per greedy iteration (default: 10)
        patience: Early stopping patience, None to disable (default: 3)
        dominant_fraction: Fraction of budget for best rep (default: 0.7)

    Returns:
        tuple: (best_allocation, history, final_feature_info)
    """
    # --- Phase 1: evaluate each rep individually ---
    rep_scores = {}
    for rep_name in rep_dict_train:
        X_tr = rep_dict_train[rep_name]
        X_va = rep_dict_val[rep_name]
        score = _evaluate_allocation(X_tr, X_va, train_labels, val_labels,
                                     model=model, metric_fn=metric_fn,
                                     lower_is_better=lower_is_better,
                                     internal_evaluator=internal_evaluator)
        rep_scores[rep_name] = score

    best_rep = max(rep_scores, key=rep_scores.get)
    dominant_budget = int(total_budget * dominant_fraction)
    max_feats_best = rep_dict_train[best_rep].shape[1]
    dominant_budget = min(dominant_budget, max_feats_best)

    # --- Phase 2: initialise allocation with dominant rep ---
    allocation = {rep: 0 for rep in rep_dict_train}
    allocation[best_rep] = dominant_budget
    remaining_budget = total_budget - dominant_budget

    # Cache RF importance scores — local to this call, discarded on return.
    _importance_cache = {}

    # Build initial hybrid and score
    X_train_init, init_feat_info = _create_hybrid_with_allocation(
        rep_dict_train, train_labels, allocation,
        importance_cache=_importance_cache,
    )
    X_val_init = apply_feature_selection(rep_dict_val, init_feat_info)
    best_val_score = _evaluate_allocation(X_train_init, X_val_init, train_labels, val_labels,
                                         model=model, metric_fn=metric_fn,
                                         lower_is_better=lower_is_better,
                                         internal_evaluator=internal_evaluator)

    best_allocation = allocation.copy()
    best_feature_info = init_feat_info
    steps_without_improvement = 0
    history = [{
        'iteration': 0, 'phase': 'dominant', 'rep': best_rep,
        'allocation': allocation.copy(), 'val_score': best_val_score,
    }]

    # --- Phase 3: greedy fill remaining budget ---
    iteration = 0
    while remaining_budget > 0:
        iteration += 1
        iter_best_gain = -np.inf
        iter_best_rep = None
        iter_best_feat_info = None
        iter_best_trial_allocation = None

        for rep_name in rep_dict_train:
            max_features = rep_dict_train[rep_name].shape[1]
            if allocation[rep_name] >= max_features:
                continue

            trial_allocation = allocation.copy()
            features_to_add = min(step_size, max_features - allocation[rep_name],
                                  remaining_budget)
            if features_to_add <= 0:
                continue
            trial_allocation[rep_name] = allocation[rep_name] + features_to_add

            X_train_trial, feat_info = _create_hybrid_with_allocation(
                rep_dict_train, train_labels, trial_allocation,
                importance_cache=_importance_cache,
            )
            X_val_trial = apply_feature_selection(rep_dict_val, feat_info)
            score = _evaluate_allocation(X_train_trial, X_val_trial, train_labels, val_labels,
                                         model=model, metric_fn=metric_fn,
                                         lower_is_better=lower_is_better,
                                         internal_evaluator=internal_evaluator)

            if score > iter_best_gain:
                iter_best_gain = score
                iter_best_rep = rep_name
                iter_best_feat_info = feat_info
                iter_best_trial_allocation = trial_allocation

        if iter_best_rep is None:
            break

        features_added = sum(iter_best_trial_allocation.values()) - sum(allocation.values())
        allocation = iter_best_trial_allocation
        remaining_budget -= features_added

        if iter_best_gain > best_val_score:
            best_val_score = iter_best_gain
            best_allocation = allocation.copy()
            best_feature_info = iter_best_feat_info
            steps_without_improvement = 0
        elif greedy_tolerance > 0 and iter_best_gain >= best_val_score - greedy_tolerance:
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        history.append({
            'iteration': iteration, 'phase': 'greedy', 'rep': iter_best_rep,
            'allocation': allocation.copy(), 'val_score': iter_best_gain,
        })

        if patience is not None and steps_without_improvement >= patience:
            break

    return best_allocation, history, best_feature_info


def _concat_rep_dicts(rep_dict):
    """Concatenate rep dicts into flat array with offset tracking.

    Returns:
        (X_concat, rep_names, rep_offsets, rep_sizes)
    """
    rep_names = sorted(rep_dict.keys())
    rep_offsets = {}
    rep_sizes = {}
    offset = 0
    for name in rep_names:
        rep_offsets[name] = offset
        rep_sizes[name] = rep_dict[name].shape[1]
        offset += rep_sizes[name]
    X_concat = np.hstack([rep_dict[name] for name in rep_names])
    return X_concat, rep_names, rep_offsets, rep_sizes


def _global_to_per_rep_feature_info(selected_global, rep_names, rep_offsets,
                                     rep_sizes, importance_scores):
    """Convert global selected indices to per-rep feature_info dict."""
    feature_info = {}
    for name in rep_names:
        offset = rep_offsets[name]
        size = rep_sizes[name]
        mask = (selected_global >= offset) & (selected_global < offset + size)
        local_indices = selected_global[mask] - offset
        if len(local_indices) > 0:
            scores = importance_scores[selected_global[mask]]
            feature_info[name] = {
                'selected_indices': local_indices,
                'importance_scores': scores,
                'n_features': len(local_indices),
            }
    return feature_info


def allocate_two_stage(rep_dict_train, rep_dict_val, train_labels, val_labels,
                       total_budget=100, stage1_method='treeshap',
                       stage2_method='permutation', stage1_multiplier=3):
    """
    Two-stage allocation: fast screening then precise selection.

    Stage 1: Use fast method to reduce to stage1_multiplier * budget candidates.
    Stage 2: Use slower method on candidates for final selection.

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        stage1_method: Fast importance method for screening (default: 'treeshap')
        stage2_method: Precise importance method for final selection (default: 'permutation')
        stage1_multiplier: Candidates = multiplier * budget (default: 3)

    Returns:
        tuple: (allocation, feature_info)
    """
    X_train, rep_names, rep_offsets, rep_sizes = _concat_rep_dicts(rep_dict_train)
    n_features = X_train.shape[1]
    stage1_budget = min(stage1_multiplier * total_budget, n_features)

    # Stage 1: Fast screening
    stage1_scores = compute_feature_importance(
        {'features': X_train}, train_labels, method=stage1_method
    )['features']
    stage1_candidates = np.argsort(stage1_scores)[-stage1_budget:]

    # Stage 2: Precise selection on candidates
    X_train_subset = X_train[:, stage1_candidates]
    try:
        stage2_scores = compute_feature_importance(
            {'features': X_train_subset}, train_labels, method=stage2_method
        )['features']
        selected_local = np.argsort(stage2_scores)[-total_budget:]
    except Exception:
        selected_local = np.argsort(stage1_scores[stage1_candidates])[-total_budget:]

    selected_global = stage1_candidates[selected_local]

    feature_info = _global_to_per_rep_feature_info(
        selected_global, rep_names, rep_offsets, rep_sizes, stage1_scores
    )
    allocation = {name: feature_info[name]['n_features']
                  for name in rep_names if name in feature_info}
    for name in rep_names:
        if name not in allocation:
            allocation[name] = 0

    return allocation, feature_info


def allocate_stability_selection(rep_dict_train, rep_dict_val, train_labels, val_labels,
                                  total_budget=100, n_bootstrap=20,
                                  importance_method='treeshap',
                                  selection_threshold=0.6):
    """
    Stability selection via bootstrapping.

    Runs importance on B bootstrap samples, keeps features that appear
    in top-K for >= threshold fraction of samples.

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        n_bootstrap: Number of bootstrap samples (default: 20)
        importance_method: Importance method to use (default: 'treeshap')
        selection_threshold: Minimum selection frequency to keep (default: 0.6)

    Returns:
        tuple: (allocation, feature_info)
    """
    X_train, rep_names, rep_offsets, rep_sizes = _concat_rep_dicts(rep_dict_train)
    n_samples, n_features = X_train.shape
    top_k = min(total_budget * 2, n_features)

    selection_counts = np.zeros(n_features)

    for b in range(n_bootstrap):
        boot_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_train[boot_idx]
        y_boot = train_labels[boot_idx]

        try:
            scores = compute_feature_importance(
                {'features': X_boot}, y_boot, method=importance_method
            )['features']
            top_indices = np.argsort(scores)[-top_k:]
            selection_counts[top_indices] += 1
        except Exception:
            continue

    selection_prob = selection_counts / n_bootstrap
    above_threshold = np.where(selection_prob >= selection_threshold)[0]

    if len(above_threshold) > total_budget:
        prob_ranking = np.argsort(selection_prob[above_threshold])[::-1]
        selected = above_threshold[prob_ranking[:total_budget]]
    elif len(above_threshold) > 0:
        selected = above_threshold
    else:
        selected = np.argsort(selection_prob)[-total_budget:]

    # Use selection probability as importance scores
    feature_info = _global_to_per_rep_feature_info(
        selected, rep_names, rep_offsets, rep_sizes, selection_prob
    )
    allocation = {name: feature_info[name]['n_features']
                  for name in rep_names if name in feature_info}
    for name in rep_names:
        if name not in allocation:
            allocation[name] = 0

    return allocation, feature_info


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


def _create_hybrid_with_allocation(rep_dict, labels, allocation,
                                    importance_cache=None):
    """Helper: Create hybrid with a pre-computed allocation dict {rep_name: n_features}.

    Always uses RandomForest feature_importances_ for within-rep feature selection.
    This is intentional: the importance_method parameter in create_hybrid() controls
    how allocations are *computed* (for methods like performance_weighted), but once
    we have an allocation, feature selection within each rep uses RF for speed.
    Greedy/dominant_greedy bypass importance_method entirely — they evaluate
    allocations by training RF models and using feature_importances_ directly.

    Args:
        importance_cache: Optional dict {rep_name: importance_scores}. When
            provided, cached importance scores are reused instead of re-fitting
            an RF.  New scores are written back into the dict so callers in
            iterative loops (greedy, dominant_greedy) pay the RF cost only once
            per rep.
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
                'importance_scores': np.ones(X_clipped.shape[1]),
                'n_features': X_clipped.shape[1]
            }
            continue

        # Check cache before fitting RF
        if importance_cache is not None and name in importance_cache:
            importances = importance_cache[name]
        else:
            # Compute importance via RF
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            if is_classification:
                model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

            model.fit(X_scaled, labels)
            importances = model.feature_importances_
            importances = np.nan_to_num(importances, nan=0.0)

            # Store in cache for reuse
            if importance_cache is not None:
                importance_cache[name] = importances

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


def _evaluate_allocation(X_train, X_val, y_train, y_val,
                         model=None, metric_fn=None, lower_is_better=False,
                         internal_evaluator='ridge'):
    """Helper: Evaluate allocation quality on a train/val split.

    If *model* is provided it is deep-copied, fitted on (X_train, y_train),
    and scored with *metric_fn(y_val, y_pred)*.  The returned value is
    negated when *lower_is_better* so that "higher return = better" always.

    If *model* is None (default), uses the *internal_evaluator* ('ridge',
    'rf', or 'xgboost') with R²/accuracy as the score.
    """
    if X_train.shape[1] == 0:
        return -np.inf

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if model is not None:
        m = deepcopy(model)
        m.fit(X_train_scaled, y_train)
        y_pred = m.predict(X_val_scaled)
        score = metric_fn(y_val, y_pred)
        return -score if lower_is_better else score

    # Internal evaluator — used during greedy allocation loops
    is_classification = _is_classification_task(y_train)

    if internal_evaluator == 'ridge':
        if is_classification:
            from sklearn.linear_model import RidgeClassifierCV
            m = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        else:
            from sklearn.linear_model import RidgeCV
            m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        m.fit(X_train_scaled, y_train)
        if is_classification:
            return m.score(X_val_scaled, y_val)
        return r2_score(y_val, m.predict(X_val_scaled))

    elif internal_evaluator == 'rf':
        if is_classification:
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=50, max_depth=10,
                                       random_state=42, n_jobs=-1)
        else:
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=50, max_depth=10,
                                      random_state=42, n_jobs=-1)
        m.fit(X_train_scaled, y_train)
        if is_classification:
            return m.score(X_val_scaled, y_val)
        return r2_score(y_val, m.predict(X_val_scaled))

    elif internal_evaluator == 'xgboost':
        try:
            from xgboost import XGBRegressor, XGBClassifier
            if is_classification:
                m = XGBClassifier(n_estimators=50, max_depth=4,
                                  random_state=42, verbosity=0, n_jobs=-1)
            else:
                m = XGBRegressor(n_estimators=50, max_depth=4,
                                 random_state=42, verbosity=0, n_jobs=-1)
            m.fit(X_train_scaled, y_train)
            if is_classification:
                return m.score(X_val_scaled, y_val)
            return r2_score(y_val, m.predict(X_val_scaled))
        except ImportError:
            # Fallback to ridge if xgboost not available
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='ridge')

    elif internal_evaluator == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
            if is_classification:
                m = LGBMClassifier(n_estimators=50, max_depth=4,
                                   random_state=42, verbosity=-1, n_jobs=-1)
            else:
                m = LGBMRegressor(n_estimators=50, max_depth=4,
                                  random_state=42, verbosity=-1, n_jobs=-1)
            m.fit(X_train_scaled, y_train)
            if is_classification:
                return m.score(X_val_scaled, y_val)
            return r2_score(y_val, m.predict(X_val_scaled))
        except ImportError:
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='xgboost')

    elif internal_evaluator == 'mlp':
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        if is_classification:
            m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                              random_state=42, early_stopping=True)
        else:
            m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200,
                             random_state=42, early_stopping=True)
        m.fit(X_train_scaled, y_train)
        if is_classification:
            return m.score(X_val_scaled, y_val)
        return r2_score(y_val, m.predict(X_val_scaled))

    elif internal_evaluator == 'svm':
        from sklearn.svm import SVR, SVC
        if is_classification:
            m = SVC(kernel='rbf', C=1.0, random_state=42)
        else:
            m = SVR(kernel='rbf', C=1.0)
        m.fit(X_train_scaled, y_train)
        if is_classification:
            return m.score(X_val_scaled, y_val)
        return r2_score(y_val, m.predict(X_val_scaled))

    elif internal_evaluator == 'gp':
        from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        if is_classification:
            m = GaussianProcessClassifier(kernel=kernel, random_state=42,
                                          max_iter_predict=50, n_restarts_optimizer=0)
        else:
            m = GaussianProcessRegressor(kernel=kernel, random_state=42,
                                         n_restarts_optimizer=0, alpha=1e-2)
        # GP scales O(n³) — limit training samples for large datasets
        max_gp_samples = 500
        if X_train_scaled.shape[0] > max_gp_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(X_train_scaled.shape[0], max_gp_samples, replace=False)
            m.fit(X_train_scaled[idx], y_train[idx])
        else:
            m.fit(X_train_scaled, y_train)
        if is_classification:
            return m.score(X_val_scaled, y_val)
        return r2_score(y_val, m.predict(X_val_scaled))

    elif internal_evaluator == 'knn':
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        # Adaptive k matching find_best_hybrid's model factory
        k = min(5, max(1, int(np.sqrt(X_train_scaled.shape[0]))))
        k = min(k, X_train_scaled.shape[0] - 1)
        if k < 1:
            return 0.0
        if is_classification:
            m = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        else:
            m = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        m.fit(X_train_scaled, y_train)
        if is_classification:
            return m.score(X_val_scaled, y_val)
        return r2_score(y_val, m.predict(X_val_scaled))

    elif internal_evaluator == 'catboost':
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
            if is_classification:
                m = CatBoostClassifier(iterations=50, depth=4, verbose=0,
                                       random_seed=42)
            else:
                m = CatBoostRegressor(iterations=50, depth=4, verbose=0,
                                      random_seed=42)
            m.fit(X_train_scaled, y_train)
            if is_classification:
                return m.score(X_val_scaled, y_val)
            return r2_score(y_val, m.predict(X_val_scaled))
        except ImportError:
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='lightgbm')

    elif internal_evaluator == 'dnn':
        try:
            from kirby.models.dnn import DNNRegressor, DNNClassifier
            if is_classification:
                m = DNNClassifier(epochs=50, patience=10)
            else:
                m = DNNRegressor(epochs=50, patience=10)
            m.fit(X_train_scaled, y_train)
            if is_classification:
                return m.score(X_val_scaled, y_val)
            return r2_score(y_val, m.predict(X_val_scaled))
        except Exception:
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='mlp')

    elif internal_evaluator == 'cnn':
        try:
            from kirby.models.cnn import CNNRegressor, CNNClassifier
            if is_classification:
                m = CNNClassifier(epochs=50, patience=10)
            else:
                m = CNNRegressor(epochs=50, patience=10)
            m.fit(X_train_scaled, y_train)
            if is_classification:
                return m.score(X_val_scaled, y_val)
            return r2_score(y_val, m.predict(X_val_scaled))
        except Exception:
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='mlp')

    elif internal_evaluator == 'ngboost':
        try:
            from ngboost import NGBRegressor, NGBClassifier
            from ngboost.distns import Normal, Bernoulli
            if is_classification:
                m = NGBClassifier(Dist=Bernoulli, n_estimators=50,
                                  learning_rate=0.05, random_state=42, verbose=False)
            else:
                m = NGBRegressor(Dist=Normal, n_estimators=50,
                                 learning_rate=0.05, random_state=42, verbose=False)
            m.fit(X_train_scaled, y_train)
            if is_classification:
                return m.score(X_val_scaled, y_val)
            return r2_score(y_val, m.predict(X_val_scaled))
        except ImportError:
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='lightgbm')

    elif internal_evaluator == 'bnn':
        try:
            from kirby.models.bnn import BNNRegressor, BNNClassifier
            if is_classification:
                m = BNNClassifier(epochs=50, patience=10, mc_samples=5)
            else:
                m = BNNRegressor(epochs=50, patience=10, mc_samples=5)
            m.fit(X_train_scaled, y_train)
            if is_classification:
                return m.score(X_val_scaled, y_val)
            return r2_score(y_val, m.predict(X_val_scaled))
        except Exception:
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='mlp')

    elif callable(internal_evaluator):
        # Custom evaluator: must be a callable that returns a fresh sklearn-compatible model
        m = internal_evaluator()
        m.fit(X_train_scaled, y_train)
        if is_classification:
            return m.score(X_val_scaled, y_val)
        return r2_score(y_val, m.predict(X_val_scaled))

    else:
        raise ValueError(f"Unknown internal_evaluator: {internal_evaluator}. "
                         f"Use 'ridge', 'rf', 'xgboost', 'lightgbm', 'mlp', 'svm', 'gp', "
                         f"'knn', 'catboost', 'dnn', 'cnn', 'ngboost', 'bnn', "
                         f"or a callable that returns an sklearn-compatible model.")


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
    baseline_score,
    model=None, metric_fn=None, lower_is_better=False,
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

    new_score = _evaluate_allocation(X_train_combined, X_val_combined, y_train, y_val,
                                     model=model, metric_fn=metric_fn,
                                     lower_is_better=lower_is_better)
    improvement = new_score - baseline_score

    return new_score, improvement


def greedy_augmentation_ablation(
    X_base_train, X_base_val,
    augmentations_train, augmentations_val,
    y_train, y_val,
    augmentation_budget=20,
    threshold=None,
    return_all_scores=False,
    model=None, metric_fn=None, lower_is_better=False,
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
    baseline_score = _evaluate_allocation(X_base_train, X_base_val, y_train, y_val,
                                          model=model, metric_fn=metric_fn,
                                          lower_is_better=lower_is_better)
    
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
            current_score,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
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


def iterative_augmentation(
    X_base_train, X_base_val,
    base_feature_info,
    augmentations_train, augmentations_val,
    y_train, y_val,
    model=None,
    total_budget=100,
    metric_fn=None,
    lower_is_better=False,
    max_candidates=200,
    patience=10,
):
    """Iterative per-feature augmentation with fill-then-replace.

    Phase A — fill remaining budget slots by individually A/B-testing
    augmentation features (best first, ranked by a single RF importance pass).

    Phase B — replace lowest-importance base features with better
    augmentation features, verified by A/B testing with the downstream model.

    Parameters
    ----------
    X_base_train, X_base_val : ndarray
        Base hybrid features (already selected).
    base_feature_info : dict
        Feature info from base hybrid creation. Keys are rep names, values
        have ``'selected_indices'``.  Modified **in-place** during Phase B
        to reflect replacements.
    augmentations_train, augmentations_val : dict
        ``{aug_name: ndarray}`` of candidate augmentation features.
    y_train, y_val : ndarray
        Target labels.
    model : estimator or None
        Downstream sklearn-compatible model.  If None, internal RF is used.
    total_budget : int
        Maximum number of features in the final hybrid (base + aug).
    metric_fn : callable or None
        ``metric_fn(y_true, y_pred) -> float``.  Required when *model* is
        provided.
    lower_is_better : bool
        If True, lower metric values are better.
    max_candidates : int
        Pre-ranking cutoff — only this many top aug features are considered.
    patience : int
        Stop after this many consecutive non-improving candidates.

    Returns
    -------
    dict
        ``augmentation_info`` (compatible with ``apply_augmentation_selection``),
        plus ``replacements``, ``phase_a_added``, ``phase_b_replaced``,
        ``baseline_score``, ``final_score``.
    """
    is_classification = _is_classification_task(y_train)
    n_base = X_base_train.shape[1]
    remaining_budget = max(0, total_budget - n_base)

    # ------------------------------------------------------------------
    # Pre-rank: pool all aug features, fit one RF, rank by importance
    # ------------------------------------------------------------------
    aug_names_flat = []   # (aug_name, col_idx) per pooled column
    pool_parts_train = []
    pool_parts_val = []

    for aug_name, X_aug_tr in augmentations_train.items():
        X_aug_vl = augmentations_val[aug_name]
        X_aug_tr = np.clip(X_aug_tr, -1e10, 1e10)
        X_aug_tr = np.nan_to_num(X_aug_tr, nan=0.0, posinf=1e10, neginf=-1e10)
        X_aug_vl = np.clip(X_aug_vl, -1e10, 1e10)
        X_aug_vl = np.nan_to_num(X_aug_vl, nan=0.0, posinf=1e10, neginf=-1e10)
        for col in range(X_aug_tr.shape[1]):
            aug_names_flat.append((aug_name, col))
        pool_parts_train.append(X_aug_tr)
        pool_parts_val.append(X_aug_vl)

    if not pool_parts_train:
        return {
            'augmentation_info': {},
            'replacements': [],
            'phase_a_added': 0,
            'phase_b_replaced': 0,
            'baseline_score': float('nan'),
            'final_score': float('nan'),
        }

    X_pool_train = np.hstack(pool_parts_train)
    X_pool_val = np.hstack(pool_parts_val)

    scaler_pool = StandardScaler()
    X_pool_scaled = scaler_pool.fit_transform(X_pool_train)

    if is_classification:
        rf_ranker = RandomForestClassifier(n_estimators=50, max_depth=10,
                                           random_state=42, n_jobs=-1)
    else:
        rf_ranker = RandomForestRegressor(n_estimators=50, max_depth=10,
                                          random_state=42, n_jobs=-1)
    rf_ranker.fit(X_pool_scaled, y_train)
    importances = np.nan_to_num(rf_ranker.feature_importances_, nan=0.0)

    # Rank and take top max_candidates
    ranked_idx = np.argsort(importances)[::-1][:max_candidates]

    # ------------------------------------------------------------------
    # Baseline score
    # ------------------------------------------------------------------
    baseline_score = _evaluate_allocation(
        X_base_train, X_base_val, y_train, y_val,
        model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
    )

    # Current hybrid state (columns appended during Phase A)
    X_cur_train = X_base_train.copy()
    X_cur_val = X_base_val.copy()
    current_score = baseline_score

    # Track which aug features we've added: {aug_name: [col_indices]}
    aug_selected = {}
    phase_a_added = 0
    consecutive_failures = 0

    # ------------------------------------------------------------------
    # Phase A — fill remaining budget
    # ------------------------------------------------------------------
    for rank_pos in ranked_idx:
        if remaining_budget <= 0:
            break
        if consecutive_failures >= patience:
            break

        aug_name, col_idx = aug_names_flat[rank_pos]
        feat_train = X_pool_train[:, rank_pos:rank_pos + 1]
        feat_val = X_pool_val[:, rank_pos:rank_pos + 1]

        # A/B test: current + this one feature
        X_test_train = np.hstack([X_cur_train, feat_train])
        X_test_val = np.hstack([X_cur_val, feat_val])

        new_score = _evaluate_allocation(
            X_test_train, X_test_val, y_train, y_val,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
        )

        if new_score > current_score:
            X_cur_train = X_test_train
            X_cur_val = X_test_val
            current_score = new_score
            remaining_budget -= 1
            phase_a_added += 1
            consecutive_failures = 0
            aug_selected.setdefault(aug_name, []).append(col_idx)
        else:
            consecutive_failures += 1

    # ------------------------------------------------------------------
    # Phase B — replace lowest-importance base features
    # ------------------------------------------------------------------
    replacements = []
    phase_b_replaced = 0
    consecutive_failures = 0

    # Collect remaining candidates not yet added
    added_pool_indices = set()
    for aug_name, cols in aug_selected.items():
        for col in cols:
            for i, (an, ci) in enumerate(aug_names_flat):
                if an == aug_name and ci == col:
                    added_pool_indices.add(i)
                    break
    remaining_candidates = [i for i in ranked_idx if i not in added_pool_indices]

    if remaining_candidates and n_base > 0:
        for rank_pos in remaining_candidates:
            if consecutive_failures >= patience:
                break

            aug_name, col_idx = aug_names_flat[rank_pos]
            feat_train = X_pool_train[:, rank_pos:rank_pos + 1]
            feat_val = X_pool_val[:, rank_pos:rank_pos + 1]

            # Compute permutation importance of current hybrid to find worst feature
            scaler_cur = StandardScaler()
            X_cur_train_s = scaler_cur.fit_transform(X_cur_train)
            X_cur_val_s = scaler_cur.transform(X_cur_val)

            if model is not None:
                m_perm = deepcopy(model)
            elif is_classification:
                m_perm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                m_perm = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            m_perm.fit(X_cur_train_s, y_train)

            if metric_fn is not None:
                scoring = make_scorer(metric_fn,
                                      greater_is_better=not lower_is_better)
            else:
                scoring = None
            perm_result = permutation_importance(
                m_perm, X_cur_val_s, y_val,
                n_repeats=3, random_state=42,
                scoring=scoring,
            )
            perm_imp = perm_result.importances_mean

            # Only consider base feature columns for replacement (indices 0..n_base-1
            # of the *current* hybrid — augmentation cols are after n_base)
            base_perm_imp = perm_imp[:n_base]
            if lower_is_better:
                # For lower-is-better, permutation_importance with default scorer
                # gives negative importances for helpful features; worst = highest
                worst_base_idx = int(np.argmax(base_perm_imp))
            else:
                worst_base_idx = int(np.argmin(base_perm_imp))

            # A/B test: replace worst base feature with this aug feature
            X_test_train = X_cur_train.copy()
            X_test_val = X_cur_val.copy()
            X_test_train[:, worst_base_idx] = feat_train[:, 0]
            X_test_val[:, worst_base_idx] = feat_val[:, 0]

            new_score = _evaluate_allocation(
                X_test_train, X_test_val, y_train, y_val,
                model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
            )

            if new_score > current_score:
                X_cur_train = X_test_train
                X_cur_val = X_test_val
                current_score = new_score
                phase_b_replaced += 1
                consecutive_failures = 0
                aug_selected.setdefault(aug_name, []).append(col_idx)

                # Track replacement for provenance
                replacements.append({
                    'replaced_base_col': worst_base_idx,
                    'replaced_by_aug': aug_name,
                    'replaced_by_col': col_idx,
                    'score_delta': new_score - current_score,
                })

                # Update base_feature_info: remove the replaced base feature
                # Find which rep owns worst_base_idx
                col_cursor = 0
                for rep_name, info in list(base_feature_info.items()):
                    if 'selected_indices' not in info:
                        continue
                    sel = info['selected_indices']
                    rep_n = len(sel)
                    if col_cursor <= worst_base_idx < col_cursor + rep_n:
                        # Remove this index from the rep's selected_indices
                        local_pos = worst_base_idx - col_cursor
                        info['selected_indices'] = np.delete(sel, local_pos)
                        info['n_features'] = len(info['selected_indices'])
                        if 'importance_scores' in info and len(info['importance_scores']) > local_pos:
                            info['importance_scores'] = np.delete(
                                info['importance_scores'], local_pos)
                        break
                    col_cursor += rep_n
            else:
                consecutive_failures += 1

    # ------------------------------------------------------------------
    # Build augmentation_info (compatible with apply_augmentation_selection)
    # ------------------------------------------------------------------
    augmentation_info = {}
    for aug_name, col_indices in aug_selected.items():
        col_arr = np.array(sorted(set(col_indices)), dtype=int)
        augmentation_info[aug_name] = {
            'selected_indices': col_arr,
            'importance_scores': importances[
                [i for i, (an, _) in enumerate(aug_names_flat)
                 if an == aug_name and _ in set(col_indices)]
            ] if len(col_arr) > 0 else np.array([]),
            'n_features': len(col_arr),
        }

    return {
        'augmentation_info': augmentation_info,
        'kept_augmentations': list(aug_selected.keys()),
        'replacements': replacements,
        'phase_a_added': phase_a_added,
        'phase_b_replaced': phase_b_replaced,
        'baseline_score': baseline_score,
        'final_score': current_score,
        'total_improvement': current_score - baseline_score,
    }


# ============================================================================
# AUGMENTATION v2 — capped and uncapped (from f08 experiments)
# ============================================================================

def _pool_and_rank_aug_features(augmentations_train, augmentations_val,
                                y_train, X_base_train, model,
                                candidate_ranking, is_classification,
                                max_candidates=200):
    """Pool all augmentation features, rank them, return ranked arrays + provenance.

    Parameters
    ----------
    candidate_ranking : str
        'rf_importance' or 'residual_correlation'

    Returns
    -------
    ranked_train, ranked_val : ndarray
        Columns reordered by rank (best first), trimmed to max_candidates.
    provenance : list of (aug_name, col_idx)
        Original identity of each ranked column.
    """
    prov = []
    parts_tr, parts_vl = [], []

    for aug_name, X_tr in augmentations_train.items():
        X_vl = augmentations_val[aug_name]
        X_tr = np.nan_to_num(np.clip(X_tr, -1e10, 1e10), nan=0.0)
        X_vl = np.nan_to_num(np.clip(X_vl, -1e10, 1e10), nan=0.0)
        for col in range(X_tr.shape[1]):
            prov.append((aug_name, col))
        parts_tr.append(X_tr)
        parts_vl.append(X_vl)

    if not parts_tr:
        return None, None, []

    X_pool_tr = np.hstack(parts_tr)
    X_pool_vl = np.hstack(parts_vl)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pool_tr)

    if candidate_ranking == 'rf_importance':
        if is_classification:
            rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                        random_state=42, n_jobs=-1)
        else:
            rf = RandomForestRegressor(n_estimators=50, max_depth=10,
                                       random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y_train)
        scores = np.nan_to_num(rf.feature_importances_, nan=0.0)
    elif candidate_ranking == 'residual_correlation':
        scaler_b = StandardScaler()
        X_base_s = scaler_b.fit_transform(X_base_train)
        m = deepcopy(model)
        m.fit(X_base_s, y_train)
        residuals = y_train - m.predict(X_base_s)
        scores = np.zeros(X_pool_tr.shape[1])
        for j in range(X_pool_tr.shape[1]):
            col = X_pool_tr[:, j]
            if np.std(col) > 1e-12:
                scores[j] = abs(np.corrcoef(col, residuals)[0, 1])
        scores = np.nan_to_num(scores, nan=0.0)
    else:
        raise ValueError(f"Unknown candidate_ranking: {candidate_ranking}")

    order = np.argsort(scores)[::-1][:max_candidates]
    ranked_tr = X_pool_tr[:, order]
    ranked_vl = X_pool_vl[:, order]
    ranked_prov = [prov[i] for i in order]

    return ranked_tr, ranked_vl, ranked_prov


def _ab_test(X_cur_train, X_cur_val, X_candidate_train, X_candidate_val,
             y_train, y_val, model, metric_fn, lower_is_better):
    """A/B test: score current hybrid with and without candidate feature(s).

    Returns (new_score, improves) where new_score is comparable (higher=better).
    """
    X_test_tr = np.hstack([X_cur_train, X_candidate_train])
    X_test_vl = np.hstack([X_cur_val, X_candidate_val])
    new_score = _evaluate_allocation(
        X_test_tr, X_test_vl, y_train, y_val,
        model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
    )
    return X_test_tr, X_test_vl, new_score


def _find_worst_base_feature(X_cur_train, X_cur_val, y_train, y_val,
                              model, metric_fn, lower_is_better,
                              n_base, is_classification):
    """Find the worst base feature by permutation importance.

    Only considers columns 0..n_base-1 (base features, not aug features).
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_cur_train)
    X_vl_s = scaler.transform(X_cur_val)

    m = deepcopy(model) if model is not None else (
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        if is_classification else
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    )
    m.fit(X_tr_s, y_train)

    scoring = (make_scorer(metric_fn, greater_is_better=not lower_is_better)
               if metric_fn is not None else None)
    result = permutation_importance(m, X_vl_s, y_val, n_repeats=3,
                                    random_state=42, scoring=scoring)
    base_imp = result.importances_mean[:n_base]

    if lower_is_better:
        return int(np.argmax(base_imp))
    return int(np.argmin(base_imp))


def augment_capped(
    X_base_train, X_base_val,
    base_feature_info,
    augmentations_train, augmentations_val,
    y_train, y_val,
    model,
    metric_fn,
    lower_is_better,
    total_budget,
    candidate_ranking='rf_importance',
    patience=20,
    max_candidates=200,
):
    """Capped augmentation: fill remaining budget slots, then replace worst base features.

    Phase A — fill: add aug features one at a time into remaining budget slots.
    Phase B — replace: swap worst base feature with next-best aug feature.
    Both phases use A/B testing with the downstream model.

    Parameters
    ----------
    X_base_train, X_base_val : ndarray
        Base hybrid features (already selected).
    base_feature_info : dict
        Feature info from base hybrid creation (modified in-place during replace).
    augmentations_train, augmentations_val : dict
        {aug_name: ndarray} of candidate augmentation features.
    y_train, y_val : ndarray
    model : estimator
        Downstream model for A/B testing.
    metric_fn : callable
        metric_fn(y_true, y_pred) -> float.
    lower_is_better : bool
    total_budget : int
        Max features in final hybrid.
    candidate_ranking : str
        'rf_importance' or 'residual_correlation' (searchable).
    patience : int
        Stop after this many consecutive non-improving candidates.
    max_candidates : int
        Pre-ranking cutoff.

    Returns
    -------
    dict with augmentation_info, kept_augmentations, phase_a_added,
    phase_b_replaced, baseline_score, final_score.
    """
    is_clf = _is_classification_task(y_train)
    n_base = X_base_train.shape[1]
    remaining_budget = max(0, total_budget - n_base)

    ranked_tr, ranked_vl, prov = _pool_and_rank_aug_features(
        augmentations_train, augmentations_val,
        y_train, X_base_train, model,
        candidate_ranking, is_clf, max_candidates,
    )

    if ranked_tr is None:
        return {
            'augmentation_info': {}, 'kept_augmentations': [],
            'phase_a_added': 0, 'phase_b_replaced': 0,
            'baseline_score': float('nan'), 'final_score': float('nan'),
        }

    baseline_score = _evaluate_allocation(
        X_base_train, X_base_val, y_train, y_val,
        model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
    )

    X_cur_tr = X_base_train.copy()
    X_cur_vl = X_base_val.copy()
    current_score = baseline_score
    aug_selected = {}
    used = set()

    # Phase A — fill
    phase_a_added = 0
    consec_fail = 0
    for i in range(ranked_tr.shape[1]):
        if remaining_budget <= 0 or consec_fail >= patience:
            break

        feat_tr = ranked_tr[:, i:i + 1]
        feat_vl = ranked_vl[:, i:i + 1]
        X_test_tr, X_test_vl, new_score = _ab_test(
            X_cur_tr, X_cur_vl, feat_tr, feat_vl,
            y_train, y_val, model, metric_fn, lower_is_better,
        )

        if new_score > current_score:
            X_cur_tr, X_cur_vl = X_test_tr, X_test_vl
            current_score = new_score
            remaining_budget -= 1
            phase_a_added += 1
            consec_fail = 0
            used.add(i)
            aug_name, col_idx = prov[i]
            aug_selected.setdefault(aug_name, []).append(col_idx)
        else:
            consec_fail += 1

    # Phase B — replace worst base features
    replacements = []
    phase_b_replaced = 0
    consec_fail = 0
    for i in range(ranked_tr.shape[1]):
        if consec_fail >= patience:
            break
        if i in used:
            continue

        worst_idx = _find_worst_base_feature(
            X_cur_tr, X_cur_vl, y_train, y_val,
            model, metric_fn, lower_is_better, n_base, is_clf,
        )

        X_test_tr = X_cur_tr.copy()
        X_test_vl = X_cur_vl.copy()
        X_test_tr[:, worst_idx] = ranked_tr[:, i]
        X_test_vl[:, worst_idx] = ranked_vl[:, i]

        new_score = _evaluate_allocation(
            X_test_tr, X_test_vl, y_train, y_val,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
        )

        if new_score > current_score:
            prev_score = current_score
            X_cur_tr, X_cur_vl = X_test_tr, X_test_vl
            current_score = new_score
            phase_b_replaced += 1
            consec_fail = 0
            used.add(i)
            aug_name, col_idx = prov[i]
            aug_selected.setdefault(aug_name, []).append(col_idx)

            # Track which base feature was replaced by which aug feature
            replaced_rep = None
            col_cursor = 0
            for rep_name, info in list(base_feature_info.items()):
                if 'selected_indices' not in info:
                    continue
                sel = info['selected_indices']
                rep_n = len(sel)
                if col_cursor <= worst_idx < col_cursor + rep_n:
                    local_pos = worst_idx - col_cursor
                    replaced_rep = rep_name
                    replaced_feature_idx = int(sel[local_pos])
                    info['selected_indices'] = np.delete(sel, local_pos)
                    info['n_features'] = len(info['selected_indices'])
                    if 'importance_scores' in info and len(info['importance_scores']) > local_pos:
                        info['importance_scores'] = np.delete(
                            info['importance_scores'], local_pos)
                    n_base -= 1
                    break
                col_cursor += rep_n

            replacements.append({
                'replaced_rep': replaced_rep,
                'replaced_feature_idx': replaced_feature_idx if replaced_rep else None,
                'replaced_base_col': worst_idx,
                'replaced_by_aug': aug_name,
                'replaced_by_col': col_idx,
                'score_delta': current_score - prev_score,
            })
        else:
            consec_fail += 1

    # Build augmentation_info
    augmentation_info = {}
    for aug_name, cols in aug_selected.items():
        col_arr = np.array(sorted(set(cols)), dtype=int)
        augmentation_info[aug_name] = {
            'selected_indices': col_arr,
            'n_features': len(col_arr),
        }

    return {
        'augmentation_info': augmentation_info,
        'kept_augmentations': list(aug_selected.keys()),
        'replacements': replacements,
        'phase_a_added': phase_a_added,
        'phase_b_replaced': phase_b_replaced,
        'baseline_score': baseline_score,
        'final_score': current_score,
        'total_improvement': current_score - baseline_score,
    }


def augment_uncapped(
    X_base_train, X_base_val,
    augmentations_train, augmentations_val,
    y_train, y_val,
    model,
    metric_fn,
    lower_is_better,
    candidate_ranking='rf_importance',
    patience=20,
    max_candidates=200,
):
    """Uncapped augmentation: add features beyond the budget if they help.

    Ranks all aug features, A/B tests each one at a time. Keeps it if the
    score improves. No budget limit, no replace phase.

    Parameters
    ----------
    X_base_train, X_base_val : ndarray
        Base hybrid features.
    augmentations_train, augmentations_val : dict
        {aug_name: ndarray} of candidate augmentation features.
    y_train, y_val : ndarray
    model : estimator
        Downstream model for A/B testing.
    metric_fn : callable
    lower_is_better : bool
    candidate_ranking : str
        'rf_importance' or 'residual_correlation' (searchable).
    patience : int
        Stop after this many consecutive non-improving candidates.
    max_candidates : int
        Pre-ranking cutoff.

    Returns
    -------
    dict with augmentation_info, kept_augmentations, n_added,
    baseline_score, final_score.
    """
    is_clf = _is_classification_task(y_train)

    ranked_tr, ranked_vl, prov = _pool_and_rank_aug_features(
        augmentations_train, augmentations_val,
        y_train, X_base_train, model,
        candidate_ranking, is_clf, max_candidates,
    )

    if ranked_tr is None:
        return {
            'augmentation_info': {}, 'kept_augmentations': [],
            'n_added': 0,
            'baseline_score': float('nan'), 'final_score': float('nan'),
        }

    baseline_score = _evaluate_allocation(
        X_base_train, X_base_val, y_train, y_val,
        model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
    )

    X_cur_tr = X_base_train.copy()
    X_cur_vl = X_base_val.copy()
    current_score = baseline_score
    aug_selected = {}
    n_added = 0
    consec_fail = 0

    for i in range(ranked_tr.shape[1]):
        if consec_fail >= patience:
            break

        feat_tr = ranked_tr[:, i:i + 1]
        feat_vl = ranked_vl[:, i:i + 1]
        X_test_tr, X_test_vl, new_score = _ab_test(
            X_cur_tr, X_cur_vl, feat_tr, feat_vl,
            y_train, y_val, model, metric_fn, lower_is_better,
        )

        if new_score > current_score:
            X_cur_tr, X_cur_vl = X_test_tr, X_test_vl
            current_score = new_score
            n_added += 1
            consec_fail = 0
            aug_name, col_idx = prov[i]
            aug_selected.setdefault(aug_name, []).append(col_idx)
        else:
            consec_fail += 1

    augmentation_info = {}
    for aug_name, cols in aug_selected.items():
        col_arr = np.array(sorted(set(cols)), dtype=int)
        augmentation_info[aug_name] = {
            'selected_indices': col_arr,
            'n_features': len(col_arr),
        }

    return {
        'augmentation_info': augmentation_info,
        'kept_augmentations': list(aug_selected.keys()),
        'n_added': n_added,
        'baseline_score': baseline_score,
        'final_score': current_score,
        'total_improvement': current_score - baseline_score,
    }


# ============================================================================
# FEATURE IMPORTANCE COMPUTATION
# ============================================================================

def compute_feature_importance(base_reps, labels, method='treeshap', model=None, **kwargs):
    """
    Compute feature importance scores for each representation.

    Can either train a model internally (for feature selection) or explain a
    pre-trained model (for model interpretation).

    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        labels: Target values (n_samples,) - used for training if model=None
        method: Method for computing importance:
            Fast methods:
            - 'treeshap' (default): TreeSHAP values (MLP-stable, handles correlations)
            - 'xgboost_gain': XGBoost gain (avg improvement per split)
            - 'lightgbm_gain': LightGBM gain importance (#1 weight source)
            - 'deeplift': DeepLIFT attribution (Captum, fast neural method)
            Slower methods:
            - 'permutation': Permutation importance (model-agnostic, stable)
            - 'kernelshap': KernelSHAP (model-agnostic, most independent)
            Long search only (slow, correlated with deeplift):
            - 'integrated_gradients': Path integral attribution (Captum)
            - 'gradientshap': Gradient + SHAP sampling (Captum)
        model: Pre-trained model to explain (optional). If None, trains a new model.
            - For tree methods: sklearn RF/XGBoost/LightGBM model
            - For NN methods: PyTorch nn.Module
            - For model-agnostic: any model with predict/predict_proba
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

    # PERMUTATION IMPORTANCE
    if method == 'permutation':
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
        max_features_shap = kwargs.get('max_features_shap', 2000)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            # Clean data
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Pre-screen high-dimensional reps to avoid slow RF + SHAP
            selected_idx = None
            if X_scaled.shape[1] > max_features_shap:
                variances = np.var(X_scaled, axis=0)
                selected_idx = np.argsort(variances)[-max_features_shap:]
                selected_idx.sort()
                X_fit = X_scaled[:, selected_idx]
            else:
                X_fit = X_scaled

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
                tree_model.fit(X_fit, labels)

            # Compute SHAP values using TreeExplainer (fast, exact for trees)
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_fit)

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
            feature_importance_fit = np.mean(np.abs(shap_values), axis=0)

            # Map back to full feature space if pre-screened
            if selected_idx is not None:
                feature_importance = np.zeros(X_scaled.shape[1])
                feature_importance[selected_idx] = feature_importance_fit
            else:
                feature_importance = feature_importance_fit
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

    # XGBOOST GAIN - Average improvement per split
    elif method == 'xgboost_gain':
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost methods require 'xgboost'. Install with: pip install xgboost")

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 6)
        random_state = kwargs.get('random_state', 42)

        max_features_xgb = kwargs.get('max_features_xgb', 2000)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Pre-screen high-dimensional reps
            selected_idx = None
            if X_scaled.shape[1] > max_features_xgb:
                variances = np.var(X_scaled, axis=0)
                selected_idx = np.argsort(variances)[-max_features_xgb:]
                selected_idx.sort()
                X_fit = X_scaled[:, selected_idx]
            else:
                X_fit = X_scaled

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
                xgb_model.fit(X_fit, labels)

            booster = xgb_model.get_booster()
            scores = booster.get_score(importance_type='gain')

            # Convert to array
            feature_importance_fit = np.zeros(X_fit.shape[1])
            for fname, score in scores.items():
                idx = int(fname[1:])  # 'f0' -> 0
                feature_importance_fit[idx] = score

            # Map back to full feature space if pre-screened
            if selected_idx is not None:
                feature_importance = np.zeros(X_scaled.shape[1])
                feature_importance[selected_idx] = feature_importance_fit
            else:
                feature_importance = feature_importance_fit

            importance_scores[rep_name] = np.nan_to_num(feature_importance.astype(float), nan=0.0)

    # LIGHTGBM GAIN
    elif method == 'lightgbm_gain':
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM methods require 'lightgbm'. Install with: pip install lightgbm")

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 6)
        random_state = kwargs.get('random_state', 42)
        max_features_lgb = kwargs.get('max_features_lgb', 2000)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Pre-screen high-dimensional reps to avoid slow LightGBM fits
            selected_idx = None
            if X_scaled.shape[1] > max_features_lgb:
                # Use variance as a fast pre-filter, keep top max_features_lgb
                variances = np.var(X_scaled, axis=0)
                selected_idx = np.argsort(variances)[-max_features_lgb:]
                selected_idx.sort()
                X_fit = X_scaled[:, selected_idx]
            else:
                X_fit = X_scaled

            # Use provided model or train new one
            if model is not None:
                lgb_model = model
                if not hasattr(lgb_model, 'booster_'):
                    raise ValueError("Provided model for lightgbm_gain must be a fitted LightGBM model with booster_")
            else:
                if is_classification:
                    lgb_model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                  random_state=random_state, n_jobs=1, verbosity=-1)
                else:
                    lgb_model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                 random_state=random_state, n_jobs=1, verbosity=-1)
                lgb_model.fit(X_fit, labels)

            feature_importance_fit = lgb_model.booster_.feature_importance(importance_type='gain')

            # Map back to full feature space if pre-screened
            if selected_idx is not None:
                feature_importance = np.zeros(X_scaled.shape[1])
                feature_importance[selected_idx] = feature_importance_fit
            else:
                feature_importance = feature_importance_fit

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

    else:
        raise NotImplementedError(
            f"Method '{method}' not implemented. Choose from: "
            "treeshap, xgboost_gain, lightgbm_gain, deeplift, permutation, "
            "kernelshap, integrated_gradients, gradientshap"
        )

    return importance_scores


# REMOVED METHODS (after benchmark correlation analysis, 2026-02-10):
# random_forest: ~0.99 corr with treeshap, treeshap is MLP-stable on QM9
# xgboost_weight: ~0.99 corr with xgboost_gain, gain performs better
# xgboost_cover: universally bad
# lightgbm_split: ~1.00 corr with lightgbm_gain, gain is better weight source
# deepliftshap: timed out in testing, correlated with deeplift
# lime: universally bad, actively harmful as weights
# drop_column: extremely slow (hours), bad weighting
# boruta: slow (20+ min), bad weighting
_REMOVED_METHODS_NOTE = None  # placeholder, see comment above


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
                 allocation_method='top_3',
                 n_per_rep=100,
                 budget=100,
                 step_size=5,
                 patience=20,
                 validation_split=0.2,
                 apply_filters=False,
                 importance_method='lightgbm_gain',
                 # Augmentation parameters
                 augmentations=None,
                 augmentation_strategy='uncapped',
                 augmentation_threshold=None,
                 augmentation_budget=20,
                 candidate_ranking='rf_importance',
                 # Model-aware augmentation (capped/uncapped strategies)
                 model=None,
                 metric_fn=None,
                 lower_is_better=False,
                 # Internal evaluator for greedy/performance_weighted allocation
                 internal_evaluator='svm',
                 # Pre-computed scores (deduplication optimisation)
                 precomputed_importance=None,
                 precomputed_alloc_scores=None,
                 **kwargs):
    """
    Create hybrid representation by combining features from multiple base representations.
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        labels: Target values for computing importance
        
        allocation_method: Method for allocating features across representations.
            Base methods:
            - 'performance_weighted': Allocate proportional to baseline scores (default)
            - 'greedy': Adaptive allocation via greedy forward selection
            - 'dominant_greedy': Greedy with dominant rep fraction
            - 'fixed': Equal allocation (n_per_rep for all)
            Named variants (see NAMED_ALLOCATIONS for full list):
            - 'top_2', 'top_3', 'top_5': Top-k reps only
            - 'entropy_0.5', 'entropy_1.0', 'entropy_2.0': Entropy-weighted
            - 'rank_exp_d0.1'...'rank_exp_d0.5': Rank exponential decay
            - 'greedy_fill_s5'...'greedy_fill_s25': Greedy + budget fill
            - 'greedy_forced_s5'...'greedy_forced_s25': Greedy, no early stopping
            
        n_per_rep: Number of features per rep (for 'fixed' method, default: 100)
        budget: Total feature budget (for 'greedy' method, default: 100)
        step_size: Features to add per greedy iteration (default: 5)
        patience: Early stopping patience for greedy, None to disable (default: 20)
        validation_split: Fraction for validation in greedy (default: 0.2)
        
        apply_filters: Apply quality filters (sparsity + variance) before allocation (default: False)
        importance_method: Method for feature importance (default: 'treeshap').
            NOTE: 'greedy' and 'dominant_greedy' allocations ignore this parameter —
            they use internal RF feature_importances_. All performance_weighted
            variants (softmax_T*, top_*, entropy_*, etc.) use the specified
            importance_method.
        
        augmentations: Dict of {aug_name: feature_array} for supplementary features (default: None)
            These are evaluated separately from base_reps after initial hybrid creation
        augmentation_strategy: How to handle augmentations
            - 'uncapped': Feature-level add-if-it-helps, no budget limit (default)
            - 'greedy_ablation': Test each rep, keep only those that improve
            - 'capped': Feature-level fill-then-replace within budget (F08 tested)
            - 'uncapped': Feature-level add-if-it-helps, no budget limit
            - 'all': Include all augmentations
            - 'none': Ignore augmentations
        augmentation_threshold: Minimum improvement to keep augmentation (default: None = any improvement)
            Can be float (e.g., 0.01 for 1% R² improvement) or None
        augmentation_budget: Max features per augmentation to select (default: 20)
        candidate_ranking: How to rank augmentation features for capped/uncapped strategies
            - 'rf_importance': Rank by RF feature importance (default)
            - 'residual_correlation': Rank by correlation with base model residuals

        model: sklearn-compatible estimator for model-aware augmentation (default: None)
            Used by 'capped' and 'uncapped' strategies. Deepcopied for each evaluation.
        metric_fn: callable(y_true, y_pred) -> float for model-aware augmentation
        lower_is_better: Whether lower metric_fn values are better (default: False)
        
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

    # Resolve 'match' internal evaluator — maps downstream model type to evaluator
    # Callables pass through as-is (custom evaluator)
    if callable(internal_evaluator):
        pass  # Use as-is
    elif internal_evaluator == 'match':
        if model is not None:
            model_class = type(model).__name__.lower()
            if 'ridge' in model_class:
                internal_evaluator = 'ridge'
            elif 'randomforest' in model_class or model_class == 'rf':
                internal_evaluator = 'rf'
            elif 'xgb' in model_class or 'xgboost' in model_class:
                internal_evaluator = 'xgboost'
            elif 'lgbm' in model_class or 'lightgbm' in model_class:
                internal_evaluator = 'lightgbm'
            elif 'mlp' in model_class:
                internal_evaluator = 'mlp'
            elif 'sv' in model_class:
                internal_evaluator = 'svm'
            elif 'gaussianprocess' in model_class or model_class == 'gp':
                internal_evaluator = 'gp'
            elif 'kneighbors' in model_class or model_class == 'knn':
                internal_evaluator = 'knn'
            else:
                internal_evaluator = 'ridge'  # safe fallback
        else:
            internal_evaluator = 'ridge'  # no model provided, default

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

    # Resolve named allocation methods
    if allocation_method in NAMED_ALLOCATIONS:
        _base, _extra = NAMED_ALLOCATIONS[allocation_method]
        allocation_method = _base
        for k, v in _extra.items():
            kwargs.setdefault(k, v)

    # GREEDY ALLOCATION
    if allocation_method == 'greedy':
        # Run greedy allocation - now returns feature_info directly
        allocation, history, greedy_feature_info = allocate_greedy_forward(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget, step_size=step_size, patience=patience,
            internal_evaluator=internal_evaluator,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
            greedy_tolerance=kwargs.get('greedy_tolerance', 0.0),
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
        ws = kwargs.pop('weighting_scheme', 'linear')
        wp = kwargs.pop('weighting_params', None)
        allocation = allocate_performance_weighted(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget, weighting_scheme=ws, weighting_params=wp,
            internal_evaluator=internal_evaluator,
            precomputed_scores=precomputed_alloc_scores,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
        )

        # Compute importance on ALL data passed to create_hybrid (the full fold
        # training set). The internal train/val split above is for allocation
        # evaluation only. Importance should use all available labeled data.
        # Use precomputed scores if available (deduplication across search configs).
        if precomputed_importance is not None:
            importance_scores = precomputed_importance
            # If filters were applied, subset precomputed importance to match
            # filtered feature indices (precomputed on unfiltered data)
            if filter_index_maps:
                importance_scores = {
                    rn: scores[filter_index_maps[rn]]
                    if rn in filter_index_maps else scores
                    for rn, scores in importance_scores.items()
                }
        else:
            importance_scores = compute_feature_importance(
                base_reps, labels, method=importance_method, **kwargs
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

        # Store full importance scores for budget floor reuse
        for rep_name, imp_arr in importance_scores.items():
            if rep_name in feature_info and isinstance(feature_info[rep_name], dict):
                feature_info[rep_name]['importance_scores_full'] = imp_arr

        # Add allocation to feature_info for reference
        feature_info['allocation'] = allocation
        
    # FIXED ALLOCATION (BACKWARDS COMPATIBLE)
    elif allocation_method == 'fixed':
        # Compute importance on all data (consistent with performance_weighted)
        if precomputed_importance is not None:
            importance_scores = precomputed_importance
            if filter_index_maps:
                importance_scores = {
                    rn: scores[filter_index_maps[rn]]
                    if rn in filter_index_maps else scores
                    for rn, scores in importance_scores.items()
                }
        else:
            importance_scores = compute_feature_importance(
                base_reps, labels, method=importance_method, **kwargs
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

    # DOMINANT-GREEDY ALLOCATION
    elif allocation_method == 'dominant_greedy':
        dominant_fraction = kwargs.pop('dominant_fraction', 0.7)
        allocation, history, dom_greedy_feature_info = allocate_dominant_greedy(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget, step_size=step_size, patience=patience,
            dominant_fraction=dominant_fraction,
            internal_evaluator=internal_evaluator,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
            greedy_tolerance=kwargs.get('greedy_tolerance', 0.0),
        )

        if filter_index_maps:
            for rep_name in list(dom_greedy_feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in dom_greedy_feature_info[rep_name]:
                    filtered_indices = dom_greedy_feature_info[rep_name]['selected_indices']
                    original_indices = filter_index_maps[rep_name][filtered_indices]
                    dom_greedy_feature_info[rep_name]['selected_indices'] = original_indices

        hybrid_features = apply_feature_selection(original_base_reps, dom_greedy_feature_info)

        feature_info = dom_greedy_feature_info.copy()
        feature_info['allocation'] = allocation
        feature_info['dominant_greedy_history'] = history

    else:
        raise ValueError(f"Unknown allocation_method: {allocation_method}. "
                        f"Choose from: 'greedy', 'performance_weighted', 'dominant_greedy', "
                        f"'fixed', or any named variant in NAMED_ALLOCATIONS "
                        f"(e.g. 'top_3', 'entropy_1.0', 'greedy_fill_s25').")

    # ========================================================================
    # BUDGET FLOOR: fill from best-performing reps (ranked by baseline score)
    # When an allocator (e.g. greedy) converges early, fill remaining budget
    # with top features from the best single reps. Walks reps in order of
    # baseline score, adds their top unselected features by importance.
    # With capped augmentation, this means Phase A (fill) is a no-op and only
    # Phase B (replace worst base with better aug) applies. This is intentional.
    # ========================================================================
    current_n = sum(
        info.get('n_features', len(info.get('selected_indices', [])))
        for info in feature_info.values()
        if isinstance(info, dict) and 'selected_indices' in info
    )
    _do_fill = kwargs.pop('budget_fill', False)
    if _do_fill and current_n < budget and allocation_method != 'fixed':
        deficit = budget - current_n

        # Get baseline scores per rep for ranking
        _fill_baselines = kwargs.pop('precomputed_baselines', None)
        if _fill_baselines is None:
            _fill_baselines = {}
            for rep_name, X_tr in rep_dict_train.items():
                X_vl = rep_dict_val[rep_name]
                X_tr_c = np.nan_to_num(np.clip(X_tr, -1e10, 1e10), nan=0.0)
                X_vl_c = np.nan_to_num(np.clip(X_vl, -1e10, 1e10), nan=0.0)
                _fill_baselines[rep_name] = _evaluate_allocation(
                    X_tr_c, X_vl_c, train_labels, val_labels,
                    model=model, metric_fn=metric_fn,
                    lower_is_better=lower_is_better,
                    internal_evaluator=internal_evaluator,
                )

        # Walk reps best-first, fill with their top unselected features
        ranked_reps = sorted(_fill_baselines.keys(),
                             key=lambda r: _fill_baselines[r], reverse=True)

        filled = 0
        for rep_name in ranked_reps:
            if filled >= deficit:
                break
            X_rep = rep_dict_train[rep_name]
            info = feature_info.get(rep_name, {})
            sel_idx = info.get('selected_indices', np.array([], dtype=int))
            selected = set(sel_idx.tolist() if hasattr(sel_idx, 'tolist')
                          else list(sel_idx))
            n_total = X_rep.shape[1]
            if len(selected) >= n_total:
                continue

            # Get importance for this rep (cached or compute)
            imp = info.get('importance_scores_full', info.get('importance_scores'))
            if imp is None or len(imp) != n_total:
                X_clean = np.nan_to_num(np.clip(X_rep, -1e10, 1e10), nan=0.0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)
                is_clf = _is_classification_task(train_labels)
                if is_clf:
                    rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                                random_state=42, n_jobs=-1)
                else:
                    rf = RandomForestRegressor(n_estimators=50, max_depth=10,
                                               random_state=42, n_jobs=-1)
                rf.fit(X_scaled, train_labels)
                imp = rf.feature_importances_
                imp = np.nan_to_num(imp, nan=0.0)
                del rf

            # Add top unselected features from this rep
            unselected = [(imp[i], i) for i in range(n_total) if i not in selected]
            unselected.sort(key=lambda x: -x[0])

            for _, feat_idx in unselected:
                if filled >= deficit:
                    break
                if 'selected_indices' not in feature_info.get(rep_name, {}):
                    feature_info[rep_name] = {
                        'selected_indices': np.array([feat_idx]),
                        'n_features': 1,
                    }
                else:
                    cur = feature_info[rep_name]['selected_indices']
                    cur = cur.tolist() if hasattr(cur, 'tolist') else list(cur)
                    cur.append(feat_idx)
                    feature_info[rep_name]['selected_indices'] = np.array(
                        sorted(set(cur)), dtype=int)
                    feature_info[rep_name]['n_features'] = len(
                        feature_info[rep_name]['selected_indices'])
                filled += 1

        # Rebuild hybrid_features
        hybrid_features = apply_feature_selection(original_base_reps, feature_info)

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
                return_all_scores=True,
                model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
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
        
        elif augmentation_strategy == 'capped':
            # Feature-level fill-then-replace within budget (F08 tested)
            aug_result = augment_capped(
                X_base_train, X_base_val,
                feature_info,
                aug_dict_train, aug_dict_val,
                train_labels, val_labels,
                model=model,
                metric_fn=metric_fn,
                lower_is_better=lower_is_better,
                total_budget=budget,
                candidate_ranking=candidate_ranking,
                patience=20,
                max_candidates=kwargs.pop('aug_max_candidates', 200),
            )

            feature_info['augmentation_info'] = aug_result['augmentation_info']
            feature_info['augmentation_baseline'] = aug_result['baseline_score']
            feature_info['augmentation_final'] = aug_result['final_score']
            feature_info['kept_augmentations'] = aug_result['kept_augmentations']
            feature_info['augmentation_replacements'] = aug_result['replacements']
            feature_info['phase_a_added'] = aug_result['phase_a_added']
            feature_info['phase_b_replaced'] = aug_result['phase_b_replaced']

            # Rebuild hybrid_features from (possibly modified) base_feature_info
            # + augmentation features.  Phase B may have removed base indices.
            hybrid_features = apply_feature_selection(
                original_base_reps, feature_info
            )
            if aug_result['kept_augmentations']:
                aug_features = apply_augmentation_selection(
                    augmentations, aug_result['augmentation_info']
                )
                if aug_features is not None:
                    hybrid_features = np.hstack([hybrid_features, aug_features])

        elif augmentation_strategy == 'uncapped':
            # Feature-level add-if-it-helps, no budget limit
            aug_result = augment_uncapped(
                X_base_train, X_base_val,
                aug_dict_train, aug_dict_val,
                train_labels, val_labels,
                model=model,
                metric_fn=metric_fn,
                lower_is_better=lower_is_better,
                candidate_ranking=candidate_ranking,
                patience=20,
                max_candidates=kwargs.pop('aug_max_candidates', 200),
            )

            feature_info['augmentation_info'] = aug_result['augmentation_info']
            feature_info['augmentation_baseline'] = aug_result['baseline_score']
            feature_info['augmentation_final'] = aug_result['final_score']
            feature_info['kept_augmentations'] = aug_result['kept_augmentations']
            feature_info['n_added'] = aug_result['n_added']

            # Append augmentation features to hybrid
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