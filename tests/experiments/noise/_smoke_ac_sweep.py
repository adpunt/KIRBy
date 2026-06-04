#!/usr/bin/env python3
"""Fast unit tests for the activity-cliff percentile-knob sweep in n01/n04/n08.

These do NOT train any model or load any dataset. They build a tiny synthetic
distance / feature matrix and assert the core sweep invariant: cliff PREVALENCE
is MONOTONICALLY NON-DECREASING as the percentile knob grows (1% <= 2% <= 5%
<= 10%), because a larger knob counts more pairs as "structurally similar" and
therefore can only ADD cliff compounds, never remove them. Run directly:

    python tests/experiments/noise/_smoke_ac_sweep.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

KNOBS = [1.0, 2.0, 5.0, 10.0]


def _synthetic(n=120, seed=0):
    """Synthetic regression data with planted near-neighbour cliff structure.

    Half the points are tight pairs (a base point + a near-twin) whose labels
    differ a lot; the rest are spread out. This guarantees a non-trivial,
    non-degenerate cliff count at every knob.
    """
    rng = np.random.RandomState(seed)
    base = rng.normal(0, 5, size=(n // 2, 4))
    twins = base + rng.normal(0, 0.05, size=base.shape)  # very close neighbours
    X = np.vstack([base, twins])
    y_base = rng.normal(0, 1, size=n // 2)
    y_twins = y_base + rng.choice([-6, 6], size=n // 2)  # large activity jump
    y = np.concatenate([y_base, y_twins])
    return X.astype(float), y.astype(float)


def _assert_monotonic(prevs, label):
    assert len(prevs) == len(KNOBS), f"{label}: expected {len(KNOBS)} knobs, got {len(prevs)}"
    for a, b in zip(prevs, prevs[1:]):
        assert a <= b + 1e-9, f"{label}: prevalence not monotonic: {prevs}"
    # Non-degenerate: at least the largest knob should find some cliffs.
    assert prevs[-1] > 0, f"{label}: all knobs degenerate (0 cliffs): {prevs}"
    print(f"  OK {label}: prevalence per knob {KNOBS} = {prevs}")


def test_n08_helper_monotonic():
    """n08._cliff_mask_from_distances: distance-percentile near-neighbour rule."""
    from experiments.noise.n08_estimation_survey import _cliff_mask_from_distances
    from kirby.utils.noise_mitigation_methods import compute_distance_matrix
    X, y = _synthetic()
    D = compute_distance_matrix(X, standardize=True)
    prevs = []
    for pct in KNOBS:
        mask, _ = _cliff_mask_from_distances(D, y, pct, is_clf=False)
        prevs.append(int(mask.sum()))
    _assert_monotonic(prevs, 'n08 _cliff_mask_from_distances')


def test_n04_identify_cliffs_monotonic():
    """n04._identify_cliffs_euclidean: similarity-percentile rule."""
    from experiments.noise.n04_gp_loo_detection import _identify_cliffs_euclidean
    X, y = _synthetic()
    prevs = []
    for pct in KNOBS:
        mask = _identify_cliffs_euclidean(X, y, similar_percentile=pct)
        prevs.append(int(mask.sum()))
    _assert_monotonic(prevs, 'n04 _identify_cliffs_euclidean')


def test_n01_identify_cliff_pairs_monotonic():
    """n01.identify_cliff_pairs: similarity-percentile rule on a sim matrix."""
    from experiments.noise.n01_activity_cliff_framework import identify_cliff_pairs
    from kirby.utils.noise_mitigation_methods import compute_distance_matrix
    X, y = _synthetic()
    D = compute_distance_matrix(X, standardize=True)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)
    prevs = []
    for pct in KNOBS:
        mask, info = identify_cliff_pairs(
            sim, y, sim_threshold=0.55, activity_k=2.0,
            is_clf=False, similar_percentile=pct)
        prevs.append(info['n_cliff_compounds'])
    _assert_monotonic(prevs, 'n01 identify_cliff_pairs')


def test_constants_consistent():
    """Primary knob must be in AC_PERCENTILES and equal across the three files."""
    from experiments.noise import n01_activity_cliff_framework as n01
    from experiments.noise import n04_gp_loo_detection as n04
    from experiments.noise import n08_estimation_survey as n08
    for mod in (n01, n04, n08):
        assert mod.AC_PERCENTILES == KNOBS, f"{mod.__name__}: AC_PERCENTILES mismatch"
        assert mod.PRIMARY_AC_PERCENTILE in mod.AC_PERCENTILES
    assert (n01.PRIMARY_AC_PERCENTILE == n04.PRIMARY_AC_PERCENTILE
            == n08.PRIMARY_AC_PERCENTILE == 2.0), "primary knob must be 2.0 in all three"
    # n01's legacy module constant must equal the primary knob (headline keys).
    assert n01.SIMILAR_PERCENTILE == n01.PRIMARY_AC_PERCENTILE
    print(f"  OK constants: AC_PERCENTILES={KNOBS}, PRIMARY=2.0 in all three files")


def main():
    test_constants_consistent()
    test_n08_helper_monotonic()
    test_n04_identify_cliffs_monotonic()
    test_n01_identify_cliff_pairs_monotonic()
    print("\nALL AC-SWEEP UNIT TESTS PASSED")


if __name__ == '__main__':
    main()
