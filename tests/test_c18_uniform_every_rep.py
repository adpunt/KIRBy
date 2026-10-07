"""c18 uniform_every_rep (Adelaide, 2026-10-07): the budget is split equally
over every general representation given and filled to the budget."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('src', 'tests', os.path.join('tests', 'experiments')):
    sys.path.insert(0, os.path.join(ROOT, p))

from experiments.complexity import c18_diversity_vs_depth as c18  # noqa: E402


def _fake_reps(widths, n=40, seed=0):
    rng = np.random.default_rng(seed)
    return {name: rng.normal(size=(n, w)) for name, w in widths.items()}


def _cache_for(reps):
    # Pre-filled importance cache: no LightGBM fit is needed. Feature j of a
    # representation scores j, so the top features are the last ones.
    return {('lightgbm_gain', r): np.arange(X.shape[1], dtype=float)
            for r, X in reps.items()}


def test_format_bumped_and_arm_named():
    assert c18.RESULT_FORMAT == 'per_fold_uniform_every_rep_added_2026_10_07'
    src = open(c18.__file__).read()
    assert "arms['uniform_every_rep'] = arm" in src
    assert "arms['uniform_all'] = arm" in src


def test_budget_spread_over_every_representation():
    widths = {'a': 50, 'b': 50, 'c': 50, 'd': 50}
    tr = _fake_reps(widths, seed=1)
    va = _fake_reps(widths, n=10, seed=2)
    y = np.zeros(40)
    X_tr, X_va = c18._uniform_allocation(
        tr, va, y, list(widths), 22, cache=_cache_for(tr))
    # Filled to the budget: 22 features, 6/6/5/5 over the four representations.
    assert X_tr.shape == (40, 22)
    assert X_va.shape == (10, 22)
    # Every representation contributes: each block of features taken matches
    # the top-scored features of that representation.
    col = 0
    for name, n_sel in zip(sorted(widths), (6, 6, 5, 5)):
        want = tr[name][:, ::-1][:, :n_sel]
        np.testing.assert_array_equal(X_tr[:, col:col + n_sel], want)
        col += n_sel


def test_narrow_representation_hands_back_its_share():
    widths = {'a': 2, 'b': 50, 'c': 50}
    tr = _fake_reps(widths, seed=3)
    va = _fake_reps(widths, n=10, seed=4)
    X_tr, _ = c18._uniform_allocation(
        tr, va, np.zeros(40), list(widths), 30, cache=_cache_for(tr))
    # 'a' can give 2 features; the other 28 go 14/14 to 'b' and 'c'.
    assert X_tr.shape[1] == 30
    np.testing.assert_array_equal(X_tr[:, :2], tr['a'][:, ::-1])
