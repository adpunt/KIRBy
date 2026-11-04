"""Ridge as an internal evaluator, and three importance methods read within groups.

All four are opt-in. The first test is the one that matters most: ridge stays out of
every default list. I will not make ridge the default for everything.
"""
import inspect

import numpy as np
import pytest

from kirby import find_best_hybrid as fbh_mod
from kirby import hybrid
from kirby.hybrid import _evaluate_allocation, compute_feature_importance


def _data(n_groups=40, per_group=30, seed=0):
    rng = np.random.RandomState(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    n = len(groups)
    signal = rng.normal(size=n)
    level = rng.normal(size=n_groups)[groups]  # differs between groups, not within
    y = signal + 3 * level + 0.1 * rng.normal(size=n)
    X = np.column_stack([
        signal,                               # 0: real signal within groups
        signal + 0.01 * rng.normal(size=n),   # 1: near copy of 0
        level + 0.01 * rng.normal(size=n),    # 2: only separates groups
        rng.normal(size=n),                   # 3: noise
    ])
    return X, y, groups


def test_ridge_is_in_no_default():
    assert 'ridge' not in fbh_mod.DEFAULT_INTERNAL_EVALUATORS
    assert 'ridge' not in fbh_mod.FAST_INTERNAL_EVALUATORS
    for fn in (hybrid.create_hybrid, _evaluate_allocation, hybrid.allocate_weighted_global):
        assert inspect.signature(fn).parameters['internal_evaluator'].default != 'ridge'


def test_new_importance_methods_are_in_no_default():
    new = {'information_coefficient', 'ic_ratio', 'clustered_permutation'}
    assert not new & set(fbh_mod.DEFAULT_IMPORTANCE_METHODS)
    assert not new & set(fbh_mod.FAST_IMPORTANCE_METHODS)


def test_ridge_evaluator_regression_and_classification():
    X, y, _ = _data()
    score = _evaluate_allocation(X[:800], X[800:], y[:800], y[800:], internal_evaluator='ridge')
    assert 0 < score <= 1
    labels = (X[:, 0] > 0).astype(int)
    score, graded = _evaluate_allocation(
        X[:800], X[800:], labels[:800], labels[800:], internal_evaluator='ridge',
        return_predictions=True, graded_predictions_for_clf=True)
    assert score > 0.9 and len(graded) == len(labels) - 800


def test_information_coefficient_ignores_between_group_level():
    X, y, groups = _data()
    s = compute_feature_importance({'r': X}, y, 'information_coefficient', groups=groups)['r']
    assert s[0] > 0.2 and s[1] > 0.2
    assert s[2] < 0.1 and s[3] < 0.1  # pooled, feature 2 would dominate


def test_ic_ratio_rewards_consistency():
    rng = np.random.RandomState(1)
    groups = np.repeat(np.arange(60), 30)
    y = rng.normal(size=len(groups))
    steady = y + 2.0 * rng.normal(size=len(y))
    flips = y * np.where(groups % 2 == 0, 1.0, 0.0) + 2.0 * rng.normal(size=len(y))
    X = np.column_stack([steady, flips])
    s = compute_feature_importance({'r': X}, y, 'ic_ratio', groups=groups)['r']
    assert s[0] > s[1]


@pytest.mark.parametrize('method', ['information_coefficient', 'ic_ratio'])
def test_refuses_without_groups(method):
    X, y, _ = _data()
    with pytest.raises(ValueError, match='groups'):
        compute_feature_importance({'r': X}, y, method)


def test_clustered_permutation_credits_a_duplicated_feature():
    X, _, groups = _data()
    y = X[:, 0] + 0.1 * np.random.RandomState(2).normal(size=len(X))
    s = compute_feature_importance({'r': X}, y, 'clustered_permutation', groups=groups)['r']
    assert s[0] + s[1] > 0.5                  # the pair keeps its credit
    assert s[0] + s[1] > 20 * max(s[2], s[3])


def test_row_counts_must_match():
    X, y, groups = _data()
    with pytest.raises(ValueError, match='rows'):
        compute_feature_importance({'r': X}, y, 'ic_ratio', groups=groups[:-1])


def test_search_refuses_group_method_without_groups():
    X, y, _ = _data(n_groups=10, per_group=10)
    with pytest.raises(ValueError, match='groups'):
        hybrid_search = __import__('kirby.find_best_hybrid', fromlist=['find_best_hybrid']).find_best_hybrid
        hybrid_search({'a': X[:, :2], 'b': X[:, 2:]}, y, budgets=[2], models=['rf'], metric='r2',
                      task_type='regression', importance_methods=['ic_ratio'], verbose=False)
