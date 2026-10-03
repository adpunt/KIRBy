"""f38's scores on planted inputs, where the right answer is known."""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('KIRBY_CACHE_DIR', '/tmp')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'experiments' / 'framework'))
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
    assert f38.is_noise_suite_setting('treeshap', 'manhattan', 1.0)
    assert not f38.is_noise_suite_setting('treeshap', 'manhattan', 0.5)
    assert not f38.is_noise_suite_setting('none', 'cosine', None)


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
