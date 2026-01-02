"""
Evaluation utilities for model assessment

Common metrics and evaluation functions for regression and classification tasks.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def evaluate_regression(y_true, y_pred, verbose=True):
    """
    Calculate standard regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        verbose: Whether to print results
    
    Returns:
        dict: Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    if verbose:
        print("Regression Metrics:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
    
    return metrics


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def evaluate_binary_classification(y_true, y_pred, y_prob=None, verbose=True):
    """
    Calculate metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC-AUC)
        verbose: Whether to print results
    
    Returns:
        dict: Dictionary of metrics
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1': f1
    }
    
    # Add ROC-AUC if probabilities provided
    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        metrics['roc_auc'] = roc_auc
    
    if verbose:
        print("Binary Classification Metrics:")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")
        print(f"  F1 Score:          {f1:.4f}")
        if y_prob is not None:
            print(f"  ROC-AUC:           {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    [[TN={cm[0,0]:3d}, FP={cm[0,1]:3d}]")
        print(f"     [FN={cm[1,0]:3d}, TP={cm[1,1]:3d}]]")
    
    return metrics


def evaluate_multiclass_classification(y_true, y_pred, class_names=None, verbose=True):
    """
    Calculate metrics for multi-class classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names for display
        verbose: Whether to print results
    
    Returns:
        dict: Dictionary of metrics
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    
    if verbose:
        print("Multi-Class Classification Metrics:")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")
        print(f"  F1 (Macro):        {f1_macro:.4f}")
        print(f"  F1 (Weighted):     {f1_weighted:.4f}")
        
        # Per-class metrics
        if class_names is not None:
            print(f"\n  Per-Class Performance:")
            cm = confusion_matrix(y_true, y_pred)
            for i, name in enumerate(class_names):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                support = cm[i, :].sum()
                
                print(f"    {name:<15}: Precision={precision:.3f}, Recall={recall:.3f}, Support={support}")
    
    return metrics


# =============================================================================
# CROSS-VALIDATION UTILITIES
# =============================================================================

def cross_validate_model(model, X, y, cv=5, scoring='r2', verbose=True):
    """
    Perform cross-validation and return scores.
    
    Args:
        model: Scikit-learn model with fit/predict methods
        X: Feature matrix
        y: Target values
        cv: Number of folds
        scoring: Scoring metric
        verbose: Whether to print results
    
    Returns:
        dict: {'mean': float, 'std': float, 'scores': array}
    """
    from sklearn.model_selection import cross_val_score
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    result = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
    
    if verbose:
        print(f"Cross-Validation ({cv}-fold):")
        print(f"  {scoring}: {result['mean']:.4f} ± {result['std']:.4f}")
    
    return result


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_models(results, metric='test_rmse', ascending=True):
    """
    Compare multiple model results and print ranking.
    
    Args:
        results: List of dicts with model results
        metric: Metric to compare on
        ascending: Whether lower is better (True for error metrics)
    
    Returns:
        list: Sorted results
    """
    # Sort results
    sorted_results = sorted(results, key=lambda x: x.get(metric, float('inf')), reverse=not ascending)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<30} {metric:>15}")
    print("-"*70)
    
    for r in sorted_results:
        model_name = r.get('name', 'Unknown')
        value = r.get(metric, float('nan'))
        print(f"{model_name:<30} {value:>15.4f}")
    
    print("="*70)
    
    return sorted_results


def print_summary_table(results, metrics=['RMSE', 'MAE', 'R2']):
    """
    Print formatted summary table of results.
    
    Args:
        results: List of dicts with model results
        metrics: List of metrics to display
    """
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Header
    header = f"{'Model':<25}"
    for metric in metrics:
        header += f" {metric:>10}"
    print(header)
    print("-"*70)
    
    # Rows
    for r in results:
        row = f"{r.get('name', 'Unknown'):<25}"
        for metric in metrics:
            value = r.get(metric.lower(), float('nan'))
            row += f" {value:>10.4f}"
        print(row)
    
    print("="*70)


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def paired_t_test(scores_a, scores_b, alpha=0.05):
    """
    Perform paired t-test to compare two sets of cross-validation scores.
    
    Args:
        scores_a: CV scores for model A
        scores_b: CV scores for model B
        alpha: Significance level
    
    Returns:
        dict: {'t_statistic': float, 'p_value': float, 'significant': bool}
    """
    from scipy import stats
    
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    significant = p_value < alpha
    
    print(f"Paired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4f}")
    print(f"  Significant: {'Yes' if significant else 'No'} (α={alpha})")
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant
    }