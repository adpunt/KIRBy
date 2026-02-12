#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User-facing test scenarios for find_best_hybrid() with antibody data.

Mirror of test_hybrid_user_scenarios.py but using antibody representations
instead of molecular. Validates the antibody hybrid pipeline end-to-end.

Scenarios:
  1. Basic search (minimal args, fast speed)
  2. Custom models (SVM, MLP passed as dict)
  3. Custom metric (RMSE - lower_is_better)
  4. Blueprint reproduction (find best -> extract -> recreate per-fold)
  5. Feature weights for similarity (return_fitted=True, weight_method)
  6. Medium search (more budgets, allocations, importance methods)
  7. Override search parameters (custom budgets, allocation_methods, importance_methods)

Usage:
    python tests/test_hybrid_user_scenarios_antibody.py --scenarios 1 2 3 4 5 7
    python tests/test_hybrid_user_scenarios_antibody.py --scenarios all --reps fast+lm
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
from find_best_hybrid_antibody import load_dataset, get_representations, assign_identity_groups

warnings.filterwarnings('ignore')


def scenario_1_basic(reps, labels, groups):
    """Scenario 1: Basic search with minimal arguments."""
    print("\n" + "=" * 80)
    print("SCENARIO 1: Basic search (fast speed, default models, R2)")
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
    print(f"R2={best['score_mean']:.4f}+/-{best['score_std']:.3f}, dim={best['n_features']}")

    print(f"\nModels tested: {list(result.best_by_model.keys())}")
    print(f"Dimension tiers: {list(result.best_by_dimension.keys())}")
    print(f"Total configs tested: {len(result.all_results)}")
    print(f"Baselines: {list(result.baselines.keys())}")

    for bl_name, bl_data in result.baselines.items():
        assert 'dimension' in bl_data, f"Baseline {bl_name} missing 'dimension'"

    # Verify new Blueprint fields are populated
    assert bp.speed == 'fast'
    assert bp.n_folds == 3
    assert bp.seed == 42
    assert bp.metric == 'r2'
    print(f"Blueprint reproducibility fields: speed={bp.speed}, n_folds={bp.n_folds}, "
          f"seed={bp.seed}, metric={bp.metric}")

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
        print(f"\n{model_name}: R2={best['score_mean']:.4f}+/-{best['score_std']:.3f}, "
              f"dim={best['n_features']}, {bp.allocation_method}/{bp.importance_method}")

    print("\nSCENARIO 2: PASSED")
    return result


def scenario_3_custom_metric(reps, labels, groups):
    """Scenario 3: Custom metric (RMSE -- lower is better)."""
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
    print(f"RMSE={best['score_mean']:.4f}+/-{best['score_std']:.3f}")

    # Also test callable metric
    def custom_mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    result2 = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        metric=(custom_mae, True),
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
    print("SCENARIO 4: Blueprint reproduction (find -> extract -> reproduce)")
    print("=" * 80)

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

    print(f"\nReported best: {best['model']}, R2={reported_score:.4f}")
    print(f"Blueprint: {bp.to_dict()}")

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
    n_folds = 3

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

    print(f"\nReproduced R2={reproduced_score:.4f} (reported={reported_score:.4f}, delta={delta:.4f})")
    assert delta < 0.01, f"Reproduction failed: delta={delta:.4f} > 0.01"

    bp_dict = bp.to_dict()
    assert bp_dict['allocation_method'] == bp.allocation_method
    assert bp_dict['importance_method'] == bp.importance_method
    assert bp_dict['budget'] == bp.budget
    assert isinstance(bp_dict['reps'], list)
    assert 'speed' in bp_dict
    assert 'seed' in bp_dict

    print("\nSCENARIO 4: PASSED")
    return result


def scenario_5_feature_weights(reps, labels, groups):
    """Scenario 5: Feature weights for similarity."""
    print("\n" + "=" * 80)
    print("SCENARIO 5: Feature weights for similarity (return_fitted=True)")
    print("=" * 80)

    result = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        speed='fast',
        return_fitted=True,
        weight_method='treeshap',
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

    for rep_name, indices in fitted.selected_indices.items():
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1
        assert len(indices) > 0
        print(f"  {rep_name}: {len(indices)} features selected")

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

    print(f"\nTotal configs: {len(result.all_results)}")
    print(f"Models: {list(result.best_by_model.keys())}")

    allocs = set(r['allocation_method'] for r in result.all_results)
    imps = set(r['importance_method'] for r in result.all_results)
    budgets_seen = set(r['budget'] for r in result.all_results)
    print(f"Allocations tested: {sorted(allocs)}")
    print(f"Importance methods: {sorted(imps)}")
    print(f"Budgets tested: {sorted(budgets_seen)}")

    assert len(allocs) >= 2, f"Expected >=2 allocations, got {len(allocs)}"
    assert len(imps) >= 5, f"Expected >=5 importance methods, got {len(imps)}"

    best = result.best_overall
    bp = best['blueprint']
    print(f"\nBest: {best['model']}, {bp.allocation_method}/{bp.importance_method}, "
          f"R2={best['score_mean']:.4f}")

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
          f"budget={bp.budget}, R2={best['score_mean']:.4f}")

    print("\nSCENARIO 7: PASSED")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Antibody user-facing test scenarios for find_best_hybrid()')
    parser.add_argument('--scenarios', type=str, nargs='+', default=['1', '2', '3', '4', '5', '7'],
                        help="Scenarios to run (1-7 or 'all')")
    parser.add_argument('--dataset', type=str, default='sabdab_affinity')
    parser.add_argument('--reps', type=str, default='fast',
                        help="Rep preset: 'fast', 'fast+lm', 'all_server', 'all'")
    parser.add_argument('--identity-threshold', type=float, default=0.8)

    args = parser.parse_args()

    if 'all' in args.scenarios:
        scenarios = list(range(1, 8))
    else:
        scenarios = [int(s) for s in args.scenarios]

    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Scenarios: {scenarios}")

    # Load data
    heavy_seqs, light_seqs, labels, task_type = load_dataset(args.dataset)
    print(f"  {len(heavy_seqs)} antibodies, task: {task_type}")
    if light_seqs:
        print(f"  Paired (VH+VL)")
    else:
        print(f"  Heavy chain only")

    groups, n_groups = assign_identity_groups(heavy_seqs, threshold=args.identity_threshold)
    print(f"  {n_groups} identity groups (threshold={args.identity_threshold})")

    reps = get_representations(heavy_seqs, light_seqs, args.reps)

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
            else:
                print(f"\nUnknown scenario {s}, skipping")
                continue

            elapsed = time.time() - start
            passed.append((s, elapsed))
        except Exception as e:
            elapsed = time.time() - start
            failed.append((s, str(e)))
            print(f"\nSCENARIO {s}: FAILED -- {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for s, t in passed:
        print(f"  Scenario {s}: PASSED ({t:.1f}s)")
    for s, e in failed:
        print(f"  Scenario {s}: FAILED -- {e}")
    print(f"\n{len(passed)}/{len(passed) + len(failed)} passed")


if __name__ == '__main__':
    main()
