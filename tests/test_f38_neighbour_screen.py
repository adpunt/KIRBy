"""f38's scores on planted inputs, where the right answer is known."""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('KIRBY_CACHE_DIR', '/tmp')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'experiments' / 'framework'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f38_neighbour_screen as f38  # noqa: E402


def test_neighbours_carrying_clean_labels_give_auroc_one():
    # 10 clean molecules at 0, 4 corrupted at +5. Each molecule's neighbours are
    # clean ones, so every corrupted gap (5) beats every clean gap (0).
    y = np.r_[np.zeros(10), np.full(4, 5.0)]
    mask = np.r_[np.zeros(10, bool), np.ones(4, bool)]
    idx = np.array([[j for j in range(10) if j != i][:3] for i in range(14)])
    mean_a, med_a = f38.detection_auroc(y, mask, idx, is_clf=False)
    assert mean_a == 1.0 and med_a == 1.0


def test_classification_gap_is_share_disagreeing():
    y = np.array([0, 0, 0, 1])
    mask = np.array([False, False, False, True])
    idx = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    a, b = f38.detection_auroc(y, mask, idx, is_clf=True)
    assert a == 1.0 and b is None


def test_shared_share():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[3, 2, 9], [7, 8, 9]])
    assert f38.shared_share(a, b, 3) == (2 / 3 + 0) / 2
    assert f38.shared_share(a, b, 1) == 0.0


def test_powers_and_noise_suite_settings():
    assert f38.powers_for('manhattan') == (0.5, 1.0)
    assert f38.powers_for('minkowski_p0.5') == (0.5, 1.0, 2.0)
    assert f38.powers_for('correlation') == (None,)
    # Since 2026-10-05 the noise suite sweeps no importance weighting and the
    # distances manhattan, cosine and minkowski p=0.5 (KNN record axes 2, 3).
    from experiments.noise import similarity_space as ss
    assert ss.WEIGHTINGS == ('none',)
    assert f38.is_noise_suite_setting('none', 'manhattan', None)
    assert f38.is_noise_suite_setting('none', 'cosine', None)
    assert f38.is_noise_suite_setting('none', 'minkowski_p0.5', None)
    assert not f38.is_noise_suite_setting('none', 'euclidean', None)
    assert not f38.is_noise_suite_setting('none', 'correlation', None)


def test_overlap_pairs_differ_in_one_axis():
    d = [f38.describe('base_1', 'none', 'manhattan', None),
         f38.describe('base_2', 'none', 'manhattan', None),
         f38.describe('base_2', 'none', 'cosine', None),
         f38.describe('tanimoto_ecfp4', 'none', 'tanimoto', None)]
    pairs = set(f38.overlap_pairs(d))
    assert (0, 1) in pairs and (1, 2) in pairs and (0, 2) not in pairs
    assert {(0, 3), (1, 3), (2, 3)} <= pairs


def test_recovery_regression_and_classification():
    # Corrupted molecule 3 (clean 0, noisy 5); its neighbours carry 0, 0.
    y_noisy = np.array([0.0, 0.0, 0.0, 5.0])
    y_clean = np.zeros(4)
    mask = np.array([False, False, False, True])
    idx = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    rec, before = f38.recovery(y_noisy, y_clean, mask, idx, is_clf=False)
    assert rec == 0.0 and before == 5.0
    yc = np.array([0, 0, 0, 0])
    yn = np.array([0, 0, 0, 1])
    share, none = f38.recovery(yn, yc, mask, idx, is_clf=True)
    assert share == 1.0 and none is None


def test_scores_at_every_scheme():
    # Molecule 3 is corrupted (+5). Neighbour distances differ, so the schemes
    # weigh them differently but all neighbours carry the clean label 0.
    y = np.array([0.0, 0.0, 0.0, 5.0])
    mask = np.array([False, False, False, True])
    idx = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    dist = np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 3.0], [0.5, 4.0]])
    row = f38.scores_at(y, np.zeros(4), mask, idx, dist, 2, is_clf=False)
    for sc in f38.SCHEMES:
        assert row[f'auroc_{sc}_k2'] == 1.0
        assert row[f'recovery_{sc}_k2'] == 0.0
    assert row['recovery_before_cleaning'] == 5.0 and row['auroc_median_k2'] == 1.0
    assert set(f38.empty_at(2)) == {k for k in row if k.endswith('_k2')}


def test_noise_suite_lists_are_reproduced_exactly():
    # f38 computes every distance the noise suite sweeps on a feature space
    # (manhattan, cosine, minkowski p=0.5 since 2026-10-05; no importance
    # weighting); its lists must be the lists similarity_space.neighbour_lists
    # gives, list for list, with and without the scaffold group excluded.
    from experiments.noise import similarity_space as ss
    rng = np.random.RandomState(0)
    Z = rng.randn(60, 12)
    y = rng.randn(60)
    groups = np.repeat(np.arange(20), 3)
    assert set(ss.FEATURE_METRICS) == {'manhattan', 'cosine', 'minkowski_p0.5'}
    for metric in ss.FEATURE_METRICS:
        for eg in (None, groups):
            setting = ss.SimilaritySetting('base_1', 'none', metric)
            ref = ss.neighbour_lists(setting, spaces_train={'base_1': Z},
                                     spaces_val=None, y_train=y,
                                     exclude_groups=eg)
            D, _ = f38.square_distance(Z, metric, None, None)
            idx, dist = f38.lists_from_square(D, eg)
            assert np.array_equal(idx, ref['idx_train'])
            assert np.allclose(dist, ref['dist_train'])


def test_fast_correlation_matches_scipy():
    from scipy.spatial.distance import cdist
    rng = np.random.RandomState(1)
    Z = rng.randn(25, 9)
    w = np.abs(rng.randn(9)) + 0.1
    assert np.allclose(f38._correlation_square(Z), cdist(Z, Z, 'correlation'),
                       atol=1e-10)
    assert np.allclose(f38._correlation_square(Z, w),
                       cdist(Z, Z, 'correlation', w=w), atol=1e-10)


def test_fast_minkowski_matches_scipy():
    from scipy.spatial.distance import cdist
    Z = np.random.RandomState(2).randn(20, 7)
    assert np.allclose(f38._minkowski_square(Z, 0.5),
                       cdist(Z, Z, 'minkowski', p=0.5), atol=1e-8)


def test_minkowski_in_one_pass_matches_one_at_a_time():
    from kirby.similarity.preprocessing import normalise_weights
    rng = np.random.RandomState(3)
    Z = rng.randn(30, 8)
    w = np.abs(rng.randn(8)) + 0.1
    v = normalise_weights(w, 'mean1')
    cols = {'none': np.ones(8)}
    for pw in f38.powers_for('minkowski_p0.5'):
        cols[pw] = v ** (pw * 0.5)
    got = f38.minkowski_squares(Z, 0.5, cols)
    D0, _ = f38.square_distance(Z, 'minkowski_p0.5', None, None)
    assert np.allclose(got['none'], D0)
    for pw in f38.powers_for('minkowski_p0.5'):
        D, _ = f38.square_distance(Z, 'minkowski_p0.5', w, pw)
        assert np.allclose(got[pw], D)


def test_forest_lists_match_the_noise_suite():
    from experiments.noise import similarity_space as ss
    rng = np.random.RandomState(4)
    X = rng.randn(50, 6)
    y = rng.randn(50)
    groups = np.repeat(np.arange(25), 2)
    for eg in (None, groups):
        ref = ss.neighbour_lists(ss.SimilaritySetting(ss.FOREST_SPACE, 'none', 'forest'),
                                 spaces_train={'model_input': X}, spaces_val=None,
                                 y_train=y, exclude_groups=eg)
        L, _ = ss._forest_leaves(X, y, False)
        D = ss._forest_distance(L, L)
        k_eff = ss._k_eff(50, f38.K_MAX, eg)
        idx, dist = ss._lists_from_distance(lambda a, b: D[a:b].copy(), 50, 50,
                                            k_eff, self_rows=True,
                                            exclude_groups=eg, itemsize=4,
                                            sorter='numpy')
        assert np.array_equal(idx, ref['idx_train'])
