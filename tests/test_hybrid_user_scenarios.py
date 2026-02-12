#!/usr/bin/env python3
"""
User-facing test scenarios for find_best_hybrid().

Each scenario demonstrates calling find_best_hybrid() as a real user would,
with different configurations. These validate the library API end-to-end
and serve as reference examples.

Scenarios:
  1. Basic search (minimal args, fast speed)
  2. Custom models (SVM, MLP passed as dict)
  3. Custom metric (RMSE — lower_is_better)
  4. Blueprint reproduction (find best → extract → recreate per-fold)
  5. Feature weights for similarity (return_fitted=True, weight_method)
  6. Medium search (more budgets, allocations, importance methods)
  7. Override search parameters (custom budgets, allocation_methods, importance_methods)
  8. Multi-dataset comparison (logd vs caco2)

Usage:
    python tests/test_hybrid_user_scenarios.py --scenarios 1 2 3 4 5 6 7 8
    python tests/test_hybrid_user_scenarios.py --scenarios 1 4 5 --reps fast
    python tests/test_hybrid_user_scenarios.py --scenarios all --reps all
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from kirby.find_best_hybrid import find_best_hybrid, SearchResult, Blueprint, FittedResult
from kirby.hybrid import create_hybrid, apply_feature_selection
from find_best_hybrid_molecular import load_dataset, get_representations, assign_scaffold_groups

warnings.filterwarnings('ignore')


def scenario_1_basic(reps, labels, groups):
    """Scenario 1: Basic search with minimal arguments."""
    print("\n" + "=" * 80)
    print("SCENARIO 1: Basic search (fast speed, default models, R²)")
    print("=" * 80)

    result = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        speed='fast',
        cache=True,
    )

    assert isinstance(result, SearchResult)
    assert result.best_overall is not None
    assert result.metric_name == 'r2'
    assert result.lower_is_better is False
    assert len(result.all_results) > 0
    assert len(result.baselines) > 0

    best = result.best_overall
    bp = best['blueprint']
    print(f"\nBest: {best['model']}, {bp.allocation_method}/{bp.importance_method}, "
          f"budget={bp.budget}")
    print(f"R²={best['score_mean']:.4f}±{best['score_std']:.3f}, dim={best['n_features']}")

    print(f"\nModels tested: {list(result.best_by_model.keys())}")
    print(f"Dimension tiers: {list(result.best_by_dimension.keys())}")
    print(f"Total configs tested: {len(result.all_results)}")
    print(f"Baselines: {list(result.baselines.keys())}")

    # Verify baselines have expected structure
    for bl_name, bl_data in result.baselines.items():
        assert 'dimension' in bl_data, f"Baseline {bl_name} missing 'dimension'"

    print("\nSCENARIO 1: PASSED")
    return result


def scenario_2_custom_models(reps, labels, groups):
    """Scenario 2: Custom models (SVM, MLP)."""
    print("\n" + "=" * 80)
    print("SCENARIO 2: Custom models (SVM, MLP)")
    print("=" * 80)

    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor

    custom_models = {
        'svm': SVR(kernel='rbf', C=1.0),
        'mlp': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    }

    result = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        models=custom_models,
        speed='fast',
        cache=True,
    )

    assert set(result.best_by_model.keys()) == {'svm', 'mlp'}

    for model_name, best in result.best_by_model.items():
        bp = best['blueprint']
        print(f"\n{model_name}: R²={best['score_mean']:.4f}±{best['score_std']:.3f}, "
              f"dim={best['n_features']}, {bp.allocation_method}/{bp.importance_method}")

    print("\nSCENARIO 2: PASSED")
    return result


def scenario_3_custom_metric(reps, labels, groups):
    """Scenario 3: Custom metric (RMSE — lower is better)."""
    print("\n" + "=" * 80)
    print("SCENARIO 3: Custom metric (RMSE)")
    print("=" * 80)

    result = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        metric='rmse',
        speed='fast',
        cache=True,
    )

    assert result.metric_name == 'rmse'
    assert result.lower_is_better is True

    best = result.best_overall
    bp = best['blueprint']
    print(f"\nBest: {best['model']}, {bp.allocation_method}/{bp.importance_method}")
    print(f"RMSE={best['score_mean']:.4f}±{best['score_std']:.3f}")

    # Also test callable metric
    def custom_mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    result2 = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        metric=(custom_mae, True),  # (callable, lower_is_better)
        speed='fast',
        verbose=False,
        cache=True,
    )

    assert result2.metric_name == 'custom'
    assert result2.lower_is_better is True
    print(f"Custom MAE: {result2.best_overall['score_mean']:.4f}")

    print("\nSCENARIO 3: PASSED")
    return result


def scenario_4_blueprint_reproduction(reps, labels, groups):
    """Scenario 4: Blueprint reproduction.

    1. Run find_best_hybrid to find best config
    2. Extract blueprint
    3. Use blueprint params in per-fold CV to reproduce the result
    4. Verify reproduced score is close to reported score
    """
    print("\n" + "=" * 80)
    print("SCENARIO 4: Blueprint reproduction (find → extract → reproduce)")
    print("=" * 80)

    # Step 1: Find best
    result = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        speed='fast',
        cache=True,
    )

    best = result.best_overall
    bp = best['blueprint']
    reported_score = best['score_mean']

    print(f"\nReported best: {best['model']}, R²={reported_score:.4f}")
    print(f"Blueprint: {bp.to_dict()}")

    # Step 2: Extract blueprint and reproduce per-fold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge

    model_map = {
        'rf': lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'knn': lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        'ridge': lambda: Ridge(alpha=1.0),
    }

    model_name = best['model']
    if model_name not in model_map:
        print(f"  Skipping reproduction for non-default model '{model_name}'")
        print("\nSCENARIO 4: PASSED (blueprint extraction verified)")
        return result

    model_fn = model_map[model_name]
    n_folds = 3  # fast speed uses 3 folds

    if groups is not None:
        splitter = GroupKFold(n_splits=n_folds)
        split_args = (next(iter(reps.values())), labels, groups)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_args = (next(iter(reps.values())), labels)

    fold_scores = []
    for train_idx, val_idx in splitter.split(*split_args):
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]

        X_train, feature_info = create_hybrid(
            train_reps, y_train,
            allocation_method=bp.allocation_method,
            budget=bp.budget,
            importance_method=bp.importance_method,
        )
        X_val = apply_feature_selection(val_reps, feature_info)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_fn()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_val_s)
        fold_scores.append(r2_score(y_val, y_pred))

    reproduced_score = float(np.mean(fold_scores))
    delta = abs(reproduced_score - reported_score)

    print(f"\nReproduced R²={reproduced_score:.4f} (reported={reported_score:.4f}, Δ={delta:.4f})")

    # Scores should be identical since same CV splits and same seed
    assert delta < 0.01, f"Reproduction failed: delta={delta:.4f} > 0.01"

    # Verify blueprint dict round-trips
    bp_dict = bp.to_dict()
    assert bp_dict['allocation_method'] == bp.allocation_method
    assert bp_dict['importance_method'] == bp.importance_method
    assert bp_dict['budget'] == bp.budget
    assert isinstance(bp_dict['reps'], list)

    print("\nSCENARIO 4: PASSED")
    return result


def scenario_5_feature_weights(reps, labels, groups):
    """Scenario 5: Feature weights for similarity.

    Tests return_fitted=True and weight_method to get feature-level importance
    weights suitable for weighted kNN / similarity search.
    """
    print("\n" + "=" * 80)
    print("SCENARIO 5: Feature weights for similarity (return_fitted=True)")
    print("=" * 80)

    result = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        speed='fast',
        return_fitted=True,
        weight_method='lightgbm_gain',
        cache=True,
    )

    assert result.fitted_result is not None
    fitted = result.fitted_result

    assert isinstance(fitted, FittedResult)
    assert isinstance(fitted.blueprint, Blueprint)
    assert isinstance(fitted.selected_indices, dict)
    assert fitted.n_features > 0

    print(f"\nBlueprint: {fitted.blueprint.to_dict()}")
    print(f"Total features: {fitted.n_features}")

    # Check selected_indices
    for rep_name, indices in fitted.selected_indices.items():
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1
        assert len(indices) > 0
        print(f"  {rep_name}: {len(indices)} features selected")

    # Check feature_weights
    if fitted.feature_weights is not None:
        print("\nFeature weights (per rep):")
        for rep_name, weights in fitted.feature_weights.items():
            assert isinstance(weights, np.ndarray)
            assert len(weights) == len(fitted.selected_indices[rep_name])
            assert np.all(weights >= 0)
            print(f"  {rep_name}: {len(weights)} weights, "
                  f"range=[{weights.min():.4f}, {weights.max():.4f}], "
                  f"mean={weights.mean():.4f}")
    else:
        print("\nWARNING: feature_weights is None (weight_method may have failed)")

    # Use weights for weighted distance: apply sqrt(importance) as feature scaling
    X_hybrid = apply_feature_selection(reps, {
        rep_name: {'selected_indices': indices}
        for rep_name, indices in fitted.selected_indices.items()
    })

    if fitted.feature_weights is not None:
        weights_flat = np.concatenate([
            fitted.feature_weights[rep_name]
            for rep_name in fitted.selected_indices
        ])
        X_weighted = X_hybrid * np.sqrt(weights_flat)[np.newaxis, :]
        print(f"\nWeighted hybrid shape: {X_weighted.shape}")
        print(f"Unweighted hybrid shape: {X_hybrid.shape}")
    else:
        print("\nSkipping weighted distance demo (no weights)")

    print("\nSCENARIO 5: PASSED")
    return result


def scenario_6_medium_search(reps, labels, groups):
    """Scenario 6: Medium-speed search (more thorough)."""
    print("\n" + "=" * 80)
    print("SCENARIO 6: Medium-speed search")
    print("=" * 80)

    result = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        speed='medium',
        cache=True,
    )

    # Medium should test more configs than fast
    print(f"\nTotal configs: {len(result.all_results)}")
    print(f"Models: {list(result.best_by_model.keys())}")

    # Check that medium explores more allocation methods
    allocs = set(r['allocation_method'] for r in result.all_results)
    imps = set(r['importance_method'] for r in result.all_results)
    budgets_seen = set(r['budget'] for r in result.all_results)
    print(f"Allocations tested: {sorted(allocs)}")
    print(f"Importance methods: {sorted(imps)}")
    print(f"Budgets tested: {sorted(budgets_seen)}")

    # Medium tier should have at least 2 allocation methods and 5+ importance methods
    assert len(allocs) >= 2, f"Expected >=2 allocations, got {len(allocs)}"
    assert len(imps) >= 5, f"Expected >=5 importance methods, got {len(imps)}"

    best = result.best_overall
    bp = best['blueprint']
    print(f"\nBest: {best['model']}, {bp.allocation_method}/{bp.importance_method}, "
          f"R²={best['score_mean']:.4f}")

    print("\nSCENARIO 6: PASSED")
    return result


def scenario_7_override_params(reps, labels, groups):
    """Scenario 7: Override search parameters."""
    print("\n" + "=" * 80)
    print("SCENARIO 7: Override search parameters")
    print("=" * 80)

    result = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        speed='fast',
        budgets=[25, 50, 100, 200],
        allocation_methods=['greedy', 'performance_weighted', 'greedy_feature'],
        importance_methods=['treeshap', 'lightgbm_gain'],
        n_folds=3,
        cache=True,
    )

    allocs = set(r['allocation_method'] for r in result.all_results)
    imps = set(r['importance_method'] for r in result.all_results)
    budgets_seen = set(r['budget'] for r in result.all_results)

    assert allocs == {'greedy', 'performance_weighted', 'greedy_feature'}, f"Got {allocs}"
    assert imps == {'treeshap', 'lightgbm_gain'}, f"Got {imps}"
    assert budgets_seen == {25, 50, 100, 200}, f"Got {budgets_seen}"

    print(f"Configs tested: {len(result.all_results)}")
    print(f"Allocations: {sorted(allocs)}")
    print(f"Importance: {sorted(imps)}")
    print(f"Budgets: {sorted(budgets_seen)}")

    best = result.best_overall
    bp = best['blueprint']
    print(f"\nBest: {best['model']}, {bp.allocation_method}/{bp.importance_method}, "
          f"budget={bp.budget}, R²={best['score_mean']:.4f}")

    print("\nSCENARIO 7: PASSED")
    return result


def scenario_8_multi_dataset(reps_primary, labels_primary, groups_primary,
                              dataset2_name='caco2', rep_preset='fast'):
    """Scenario 8: Multi-dataset comparison."""
    print("\n" + "=" * 80)
    print(f"SCENARIO 8: Multi-dataset comparison (logd vs {dataset2_name})")
    print("=" * 80)

    # Primary dataset (already loaded)
    print(f"\n--- Dataset 1: logd ---")
    result1 = find_best_hybrid(
        base_reps=reps_primary,
        labels=labels_primary,
        groups=groups_primary,
        speed='fast',
        verbose=False,
        cache=True,
    )
    best1 = result1.best_overall
    bp1 = best1['blueprint']
    print(f"Best: {best1['model']}, {bp1.allocation_method}/{bp1.importance_method}, "
          f"R²={best1['score_mean']:.4f}")

    # Second dataset
    print(f"\n--- Dataset 2: {dataset2_name} ---")
    smiles2, labels2, _ = load_dataset(dataset2_name)
    groups2, n_scaff2 = assign_scaffold_groups(smiles2)
    reps2 = get_representations(smiles2, rep_preset)

    result2 = find_best_hybrid(
        base_reps=reps2,
        labels=labels2,
        groups=groups2,
        speed='fast',
        verbose=False,
        cache=True,
    )
    best2 = result2.best_overall
    bp2 = best2['blueprint']
    print(f"Best: {best2['model']}, {bp2.allocation_method}/{bp2.importance_method}, "
          f"R²={best2['score_mean']:.4f}")

    # Compare
    print(f"\n--- Comparison ---")
    print(f"{'Dataset':<12} {'Model':<8} {'Alloc':<22} {'Importance':<18} {'R²':<10} {'Dim'}")
    print("-" * 80)
    print(f"{'logd':<12} {best1['model']:<8} {bp1.allocation_method:<22} "
          f"{bp1.importance_method:<18} {best1['score_mean']:<10.4f} {best1['n_features']}")
    print(f"{dataset2_name:<12} {best2['model']:<8} {bp2.allocation_method:<22} "
          f"{bp2.importance_method:<18} {best2['score_mean']:<10.4f} {best2['n_features']}")

    # Do different datasets prefer different configs?
    same_alloc = bp1.allocation_method == bp2.allocation_method
    same_imp = bp1.importance_method == bp2.importance_method
    same_model = best1['model'] == best2['model']
    print(f"\nSame allocation: {same_alloc}")
    print(f"Same importance: {same_imp}")
    print(f"Same model: {same_model}")

    print("\nSCENARIO 8: PASSED")
    return result1, result2


def main():
    parser = argparse.ArgumentParser(description='User-facing test scenarios for find_best_hybrid()')
    parser.add_argument('--scenarios', type=str, nargs='+', default=['1', '2', '3', '4', '5', '7'],
                        help="Scenarios to run (1-8 or 'all')")
    parser.add_argument('--dataset', type=str, default='logd')
    parser.add_argument('--reps', type=str, default='fast',
                        help="Rep preset: 'fast', 'fast+hf', 'all'")
    parser.add_argument('--dataset2', type=str, default='caco2',
                        help="Second dataset for scenario 8")

    args = parser.parse_args()

    if 'all' in args.scenarios:
        scenarios = list(range(1, 9))
    else:
        scenarios = [int(s) for s in args.scenarios]

    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Scenarios: {scenarios}")

    # Load data
    smiles, labels, task_type = load_dataset(args.dataset)
    print(f"  {len(smiles)} molecules, task: {task_type}")

    groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  {n_scaffolds} scaffolds")

    reps = get_representations(smiles, args.reps)

    # Run scenarios
    passed = []
    failed = []

    for s in scenarios:
        start = time.time()
        try:
            if s == 1:
                scenario_1_basic(reps, labels, groups)
            elif s == 2:
                scenario_2_custom_models(reps, labels, groups)
            elif s == 3:
                scenario_3_custom_metric(reps, labels, groups)
            elif s == 4:
                scenario_4_blueprint_reproduction(reps, labels, groups)
            elif s == 5:
                scenario_5_feature_weights(reps, labels, groups)
            elif s == 6:
                scenario_6_medium_search(reps, labels, groups)
            elif s == 7:
                scenario_7_override_params(reps, labels, groups)
            elif s == 8:
                scenario_8_multi_dataset(reps, labels, groups,
                                          dataset2_name=args.dataset2,
                                          rep_preset=args.reps)
            else:
                print(f"\nUnknown scenario {s}, skipping")
                continue

            elapsed = time.time() - start
            passed.append((s, elapsed))
        except Exception as e:
            elapsed = time.time() - start
            failed.append((s, str(e)))
            print(f"\nSCENARIO {s}: FAILED — {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for s, t in passed:
        print(f"  Scenario {s}: PASSED ({t:.1f}s)")
    for s, e in failed:
        print(f"  Scenario {s}: FAILED — {e}")
    print(f"\n{len(passed)}/{len(passed) + len(failed)} passed")


if __name__ == '__main__':
    main()
