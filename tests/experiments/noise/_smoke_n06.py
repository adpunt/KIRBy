#!/usr/bin/env python3
"""Fast smoke test for N06 (BindingDB ground-truth noise).

No downloads, no torch, no server. Builds tiny synthetic replicate data with
KNOWN disagreement and asserts:
  1. build_replicate_groups groups rows by (drug_id, target_id) correctly.
  2. compute_replicate_stds derives the EXACT population std per group,
     flags only disagreeing groups as ground truth, and zeroes singletons.
  3. The headline Spearman correlation in run() recovers a perfect monotone
     relationship (rho == 1.0) when the estimator scores are, by construction,
     monotone in the true replicate std.

Run:  python tests/experiments/noise/_smoke_n06.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import n06_bindingdb_ground_truth as n06


def test_build_replicate_groups():
    # 6 rows: pair A measured 3x, pair B 2x, pair C 1x.
    data = {
        'labels': np.array([1.0, 2.0, 3.0, 5.0, 7.0, 9.0]),
        'metadata': {
            'drug_ids':    ['d1', 'd1', 'd1', 'd2', 'd2', 'd3'],
            'protein_ids': ['p1', 'p1', 'p1', 'p9', 'p9', 'p3'],
        },
    }
    g = n06.build_replicate_groups(data)
    # rows 0,1,2 share (d1,p1); rows 3,4 share (d2,p9); row 5 alone.
    assert g[0] == g[1] == g[2], g
    assert g[3] == g[4], g
    assert len({g[0], g[3], g[5]}) == 3, "three distinct pairs expected"
    print("PASS build_replicate_groups: groups =", g.tolist())


def test_build_replicate_groups_explicit_wins():
    explicit = np.array([0, 0, 1])
    data = {'labels': np.zeros(3), 'replicate_groups': explicit,
            'metadata': {'drug_ids': ['x', 'y', 'z'],
                         'protein_ids': ['a', 'b', 'c']}}
    g = n06.build_replicate_groups(data)
    assert np.array_equal(g, explicit), g
    print("PASS build_replicate_groups: explicit replicate_groups respected")


def test_compute_replicate_stds_arithmetic():
    # Hand-worked example:
    #   group 0: labels {5.0, 7.0}      -> mean 6.0, population std = 1.0
    #            (deviations -1, +1; mean sq dev = (1+1)/2 = 1; sqrt = 1.0)
    #   group 1: labels {2.0, 2.0, 8.0} -> mean 4.0; sq dev (4,4,16)/3 = 8;
    #            sqrt(8) = 2.8284271...
    #   group 2: labels {3.0}           -> singleton -> std 0.0, no ground truth
    #   group 3: labels {4.0, 4.0}      -> identical replicates -> std 0.0,
    #            NOT flagged as ground truth (carries no ranking info)
    labels = np.array([5.0, 7.0, 2.0, 2.0, 8.0, 3.0, 4.0, 4.0])
    groups = np.array([0,   0,   1,   1,   1,   2,   3,   3])
    stds, has = n06.compute_replicate_stds(labels, groups)

    expected_g0 = 1.0
    expected_g1 = float(np.sqrt(8.0))  # 2.8284271247...

    assert abs(stds[0] - expected_g0) < 1e-12, (stds[0], expected_g0)
    assert abs(stds[1] - expected_g0) < 1e-12
    assert abs(stds[2] - expected_g1) < 1e-12, (stds[2], expected_g1)
    assert abs(stds[4] - expected_g1) < 1e-12
    assert stds[5] == 0.0          # singleton
    assert stds[6] == 0.0 and stds[7] == 0.0  # identical replicates

    # has_replicate: only disagreeing groups (0 and 1) flagged True
    assert has[0] and has[1] and has[2] and has[3] and has[4]
    assert not has[5]              # singleton
    assert not has[6] and not has[7]  # identical -> std 0 -> not ground truth
    assert int(has.sum()) == 5

    print(f"PASS compute_replicate_stds arithmetic: "
          f"g0 std={stds[0]:.6f} (expect 1.0), "
          f"g1 std={stds[2]:.6f} (expect {expected_g1:.6f}), "
          f"singleton={stds[5]}, identical={stds[6]}, "
          f"n_ground_truth={int(has.sum())}")


def test_run_headline_correlation_perfect():
    """End-to-end run() on a tiny dataset with a planted monotone estimator.

    We construct replicate groups whose ground-truth std increases with group
    index, and a single 1-D feature whose value is also monotone in group
    index. The 'neighbor_residual' estimator (and the others) will not give a
    literal copy of the std, so to make the headline metric exactly checkable
    we monkeypatch get_noise_estimator to return a trivial estimator whose
    score == the sample's true replicate std. Spearman(scores, gt) must be
    exactly 1.0 over the replicate samples.
    """
    n06_module = n06

    class PerfectEstimator:
        def fit(self, X, y):
            # Ground-truth std is CONSTANT within a replicate group, so a
            # perfectly-ranking estimator must also be constant within a group.
            # Our groups are consecutive pairs (rows 2g, 2g+1) with strictly
            # increasing labels, so the per-group MIN label is constant within
            # a group and strictly increasing across groups -> a score equal to
            # that per-group value is monotone in the true std -> Spearman = 1.
            y = np.asarray(y, dtype=float)
            scores = np.empty_like(y)
            for g in range(len(y) // 2):
                scores[2 * g] = scores[2 * g + 1] = y[2 * g]
            self.noise_scores_ = scores
            return self

        def get_weights(self, strength=1.0):
            s = self.noise_scores_
            w = 1.0 / (1.0 + strength * s)
            return w / w.mean()

    original = n06_module.get_noise_estimator
    n06_module.get_noise_estimator = lambda name, **kw: PerfectEstimator()
    n06_module.ESTIMATORS_SCREENING_BACKUP = list(n06_module.ESTIMATORS_SCREENING)
    n06_module.ESTIMATORS_SCREENING[:] = ['neighbor_residual']

    try:
        # 4 replicate groups, each measured twice. Group g has labels
        # {g, g + (g+1)} so its std = (g+1)/2 -> strictly increasing in g.
        # Mean label per group also strictly increases in g, so a score equal
        # to the label is monotone in the group's true std -> Spearman == 1.
        n_groups = 6  # 6 groups x 2 reps = 12 replicate samples (>= MIN=10)
        labels = []
        drug_ids = []
        protein_ids = []
        for g in range(n_groups):
            base = 10.0 * g
            labels += [base, base + (g + 1)]
            drug_ids += [f'd{g}', f'd{g}']
            protein_ids += [f'p{g}', f'p{g}']
        labels = np.array(labels, dtype=float)
        # one feature, value == label (kept simple; estimator ignores X anyway)
        reps = {'feat': labels.reshape(-1, 1)}
        data = {
            'labels': labels,
            'reps': reps,
            'groups': np.arange(len(labels)),  # cold-target stand-in (unused for GT)
            'task_type': 'regression',
            'augmentations': None,
            'metadata': {'drug_ids': drug_ids, 'protein_ids': protein_ids},
        }
        config = {'search_speed': 'full'}
        results = n06_module.run(data, config, '/tmp', 'bindingdb_kd_smoke')

        # ground-truth std per group: (g+1)/2 for g=0..n -> 0.5,1.0,1.5,...
        gt_per_group = [(g + 1) / 2.0 for g in range(n_groups)]
        n_rep = 2 * n_groups
        assert results['n_with_replicates'] == n_rep, results['n_with_replicates']
        assert results['has_ground_truth'] is True

        entry = results['estimators']['concat']['neighbor_residual']
        rho = entry.get('ground_truth_correlation')
        assert rho is not None, entry
        assert abs(rho - 1.0) < 1e-9, f"expected rho=1.0, got {rho}"
        assert entry['n_ground_truth'] == n_rep, entry

        summ = results['summary']
        assert summ['best_estimator'] == 'neighbor_residual'
        assert abs(summ['best_correlation'] - 1.0) < 1e-9

        print(f"PASS run() headline correlation: rho={rho:.6f} (expect 1.0), "
              f"n_ground_truth={entry['n_ground_truth']}, "
              f"gt stds per group={gt_per_group}")
    finally:
        n06_module.get_noise_estimator = original
        n06_module.ESTIMATORS_SCREENING[:] = n06_module.ESTIMATORS_SCREENING_BACKUP


def test_run_no_replicates_does_not_crash():
    """All-singleton data: no ground truth, run() should still complete."""
    labels = np.arange(20, dtype=float)
    data = {
        'labels': labels,
        'reps': {'feat': labels.reshape(-1, 1)},
        'groups': np.arange(20),
        'task_type': 'regression',
        'augmentations': None,
        'metadata': {
            'drug_ids': [f'd{i}' for i in range(20)],
            'protein_ids': [f'p{i}' for i in range(20)],
        },
    }
    # use the perfect estimator to avoid heavy real estimators
    class Trivial:
        def fit(self, X, y):
            self.noise_scores_ = np.asarray(y, float); return self
        def get_weights(self, strength=1.0):
            return np.ones_like(self.noise_scores_)
    orig = n06.get_noise_estimator
    n06.get_noise_estimator = lambda name, **kw: Trivial()
    bk = list(n06.ESTIMATORS_SCREENING)
    n06.ESTIMATORS_SCREENING[:] = ['neighbor_residual']
    try:
        res = n06.run(data, {'search_speed': 'full'}, '/tmp', 'bindingdb_kd_smoke')
        assert res['n_with_replicates'] == 0
        assert res['has_ground_truth'] is False
        assert 'summary' not in res or res.get('summary', {}).get('best_estimator') is None
        print("PASS run() with no replicates: completes, has_ground_truth=False")
    finally:
        n06.get_noise_estimator = orig
        n06.ESTIMATORS_SCREENING[:] = bk


if __name__ == '__main__':
    test_build_replicate_groups()
    test_build_replicate_groups_explicit_wins()
    test_compute_replicate_stds_arithmetic()
    test_run_headline_correlation_perfect()
    test_run_no_replicates_does_not_crash()
    print("\nALL N06 SMOKE TESTS PASSED")
