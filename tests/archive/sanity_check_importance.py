"""
Sanity check for feature importance methods in hybrid.py
"""
import numpy as np
import sys
sys.path.insert(0, 'src')

from kirby.hybrid import compute_feature_importance, _is_classification_task

def make_synthetic_data(n_samples=500, n_informative=5, n_noise=45, seed=42):
    """
    Create synthetic data where we KNOW which features are important.
    First n_informative features are correlated with target, rest are noise.
    """
    np.random.seed(seed)

    n_features = n_informative + n_noise
    X = np.random.randn(n_samples, n_features)

    # Target is linear combination of first n_informative features
    true_weights = np.zeros(n_features)
    # Create weights that decrease linearly
    true_weights[:n_informative] = np.linspace(1.0, 0.2, n_informative)

    y = X @ true_weights + np.random.randn(n_samples) * 0.1

    return X, y, true_weights

def make_classification_data(n_samples=500, n_informative=5, n_noise=45, seed=42):
    """Classification version."""
    X, y_cont, true_weights = make_synthetic_data(n_samples, n_informative, n_noise, seed)
    y = (y_cont > np.median(y_cont)).astype(int)
    return X, y, true_weights

def check_importance_ranking(importances, true_weights, top_k=5):
    """
    Check if top-k important features according to method match true important features.
    Returns recall@k (fraction of true important features in top-k predictions).
    """
    true_important = set(np.where(true_weights > 0)[0])
    predicted_top_k = set(np.argsort(importances)[-top_k:])

    recall = len(true_important & predicted_top_k) / len(true_important)
    return recall, true_important, predicted_top_k

def test_method(method, X, y, true_weights, task_type="regression"):
    """Test a single importance method."""
    print(f"\n{'='*60}")
    print(f"Testing: {method} ({task_type})")
    print(f"{'='*60}")

    base_reps = {'test_rep': X}

    try:
        importance_scores = compute_feature_importance(base_reps, y, method=method)
        scores = importance_scores['test_rep']

        print(f"  Shape: {scores.shape}")
        print(f"  Min: {scores.min():.6f}, Max: {scores.max():.6f}, Mean: {scores.mean():.6f}")
        print(f"  Any NaN: {np.any(np.isnan(scores))}")
        print(f"  Any Inf: {np.any(np.isinf(scores))}")
        print(f"  Any negative: {np.any(scores < 0)}")

        # Check ranking
        recall, true_imp, pred_imp = check_importance_ranking(scores, true_weights)
        print(f"\n  True important features: {sorted(true_imp)}")
        print(f"  Top-5 predicted:         {sorted(pred_imp)}")
        print(f"  Recall@5: {recall:.1%}")

        # Show actual importance values for informative vs noise
        informative_scores = scores[:5]
        noise_scores = scores[5:]
        print(f"\n  Informative feature scores: {informative_scores}")
        print(f"  Noise feature scores (mean): {noise_scores.mean():.6f} (std: {noise_scores.std():.6f})")
        print(f"  Separation ratio (mean_informative / mean_noise): {informative_scores.mean() / (noise_scores.mean() + 1e-10):.2f}x")

        if recall >= 0.8:
            print(f"\n  ✓ PASS - Method correctly identifies important features")
            return True, scores
        else:
            print(f"\n  ✗ FAIL - Method fails to identify important features")
            return False, scores

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_edge_cases():
    """Test edge cases that might cause issues."""
    print(f"\n{'='*60}")
    print("Testing Edge Cases")
    print(f"{'='*60}")

    results = {}

    # Edge case 1: Very small dataset
    print("\n1. Very small dataset (n=20)...")
    X_small, y_small, _ = make_synthetic_data(n_samples=20, n_informative=5, n_noise=5)
    try:
        scores = compute_feature_importance({'rep': X_small}, y_small, method='random_forest')
        print(f"   ✓ OK - Shape: {scores['rep'].shape}")
        results['small_dataset'] = True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        results['small_dataset'] = False

    # Edge case 2: Single feature
    print("\n2. Single feature...")
    X_single = np.random.randn(100, 1)
    y_single = X_single[:, 0] + np.random.randn(100) * 0.1
    try:
        scores = compute_feature_importance({'rep': X_single}, y_single, method='random_forest')
        print(f"   ✓ OK - Shape: {scores['rep'].shape}, Value: {scores['rep'][0]:.4f}")
        results['single_feature'] = True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        results['single_feature'] = False

    # Edge case 3: Features with NaN
    print("\n3. Features with NaN values...")
    X_nan = np.random.randn(100, 10)
    X_nan[0, 0] = np.nan
    X_nan[50, 5] = np.nan
    y_nan = np.random.randn(100)
    try:
        scores = compute_feature_importance({'rep': X_nan}, y_nan, method='random_forest')
        print(f"   ✓ OK - Handled NaN, any NaN in output: {np.any(np.isnan(scores['rep']))}")
        results['nan_features'] = True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        results['nan_features'] = False

    # Edge case 4: Features with Inf
    print("\n4. Features with Inf values...")
    X_inf = np.random.randn(100, 10)
    X_inf[0, 0] = np.inf
    X_inf[50, 5] = -np.inf
    y_inf = np.random.randn(100)
    try:
        scores = compute_feature_importance({'rep': X_inf}, y_inf, method='random_forest')
        print(f"   ✓ OK - Handled Inf, any Inf in output: {np.any(np.isinf(scores['rep']))}")
        results['inf_features'] = True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        results['inf_features'] = False

    # Edge case 5: Constant feature (zero variance)
    print("\n5. Constant feature (zero variance)...")
    X_const = np.random.randn(100, 10)
    X_const[:, 3] = 5.0  # Constant column
    y_const = np.random.randn(100)
    try:
        scores = compute_feature_importance({'rep': X_const}, y_const, method='random_forest')
        print(f"   ✓ OK - Constant feature importance: {scores['rep'][3]:.6f}")
        results['constant_feature'] = True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        results['constant_feature'] = False

    # Edge case 6: Highly correlated features
    print("\n6. Highly correlated features...")
    X_corr = np.random.randn(100, 10)
    X_corr[:, 1] = X_corr[:, 0] + np.random.randn(100) * 0.01  # Near-duplicate
    y_corr = X_corr[:, 0] + np.random.randn(100) * 0.1
    try:
        scores = compute_feature_importance({'rep': X_corr}, y_corr, method='random_forest')
        print(f"   ✓ OK - Feature 0 importance: {scores['rep'][0]:.4f}, Feature 1 (correlated): {scores['rep'][1]:.4f}")
        results['correlated_features'] = True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        results['correlated_features'] = False

    # Edge case 7: All zero features
    print("\n7. All zero features...")
    X_zero = np.zeros((100, 10))
    y_zero = np.random.randn(100)
    try:
        scores = compute_feature_importance({'rep': X_zero}, y_zero, method='random_forest')
        print(f"   ✓ OK - All-zero importance sum: {scores['rep'].sum():.6f}")
        results['zero_features'] = True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        results['zero_features'] = False

    # Edge case 8: Multiple representations
    print("\n8. Multiple representations...")
    X1 = np.random.randn(100, 5)
    X2 = np.random.randn(100, 15)
    y_multi = np.random.randn(100)
    try:
        scores = compute_feature_importance({'rep1': X1, 'rep2': X2}, y_multi, method='random_forest')
        print(f"   ✓ OK - rep1 shape: {scores['rep1'].shape}, rep2 shape: {scores['rep2'].shape}")
        results['multiple_reps'] = True
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        results['multiple_reps'] = False

    return results

def test_classification_detection():
    """Test the _is_classification_task function."""
    print(f"\n{'='*60}")
    print("Testing Classification Detection")
    print(f"{'='*60}")

    # Integer labels
    y_int = np.array([0, 1, 0, 1, 2])
    print(f"  Integer [0,1,0,1,2]: {_is_classification_task(y_int)} (expected: True)")

    # Float labels that are integers
    y_float_int = np.array([0.0, 1.0, 0.0, 1.0])
    print(f"  Float [0.0,1.0,...]: {_is_classification_task(y_float_int)} (expected: True)")

    # Continuous float labels
    y_cont = np.array([0.1, 0.5, 0.9, 1.2, 1.8])
    print(f"  Continuous floats:  {_is_classification_task(y_cont)} (expected: False)")

    # Many unique integer-like floats (>20)
    y_many = np.arange(25).astype(float)
    print(f"  25 unique int-floats: {_is_classification_task(y_many)} (expected: False)")

def main():
    print("="*60)
    print("FEATURE IMPORTANCE SANITY CHECK")
    print("="*60)

    # Create synthetic data
    X_reg, y_reg, weights_reg = make_synthetic_data()
    X_clf, y_clf, weights_clf = make_classification_data()

    print(f"\nSynthetic data: {X_reg.shape[0]} samples, {X_reg.shape[1]} features")
    print(f"True informative features: indices 0-4 with weights {weights_reg[:5]}")

    # Test classification detection first
    test_classification_detection()

    # Test each method on regression
    # Note: mutual_info was removed - see hybrid.py docstring for rationale
    methods = ['random_forest', 'permutation', 'treeshap', 'integrated_gradients', 'kernelshap', 'drop_column']
    results = {}

    for method in methods:
        passed, scores = test_method(method, X_reg, y_reg, weights_reg, "regression")
        results[f"{method}_regression"] = passed

    # Test each method on classification
    for method in methods:
        passed, scores = test_method(method, X_clf, y_clf, weights_clf, "classification")
        results[f"{method}_classification"] = passed

    # Test edge cases
    edge_results = test_edge_cases()
    results.update(edge_results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for test, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        print("\n⚠️  Some tests failed - investigation needed!")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0

if __name__ == "__main__":
    exit(main())
