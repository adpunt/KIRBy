"""The noise suite's neighbour settings against KNN_SEARCH_SPACE_RECORD.md
(Adelaide, 2026-10-05), plus hybrid__distance1 (handoff 2.3 item 1)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('KIRBY_CACHE_DIR', '/tmp')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments.noise import similarity_space as ss  # noqa: E402


def test_axis_values():
    assert ss.WEIGHTINGS == ('none',)
    assert ss.SWEPT_KS == (5, 10, 20)
    assert ss.SWEPT_NEIGHBOUR_SCHEMES == ('rank', 'inverse_square')
    assert ss.MODEL_INPUT_SPACE not in ss.ALL_SPACES


def test_group_counts():
    full = ss.enumerate_settings(group='full_knn')
    fast = ss.enumerate_settings(group='fast_knn')
    # 8 feature spaces x 3 distances + tanimoto + forest; 6 x 2 + forest.
    assert len(full) == 26 and len(fast) == 13
    assert {s.metric for s in fast if s.space in ss.FEATURE_SPACES} == \
        {'manhattan', 'cosine'}
    assert not {'base_2', 'base_2_aug_uncapped', ss.TANIMOTO_SPACE} & \
        {s.space for s in fast}
    # the record's 138 and 66, before hybrid__distance1 was added
    no_d = [s for s in full if s.space != 'hybrid__distance1']
    assert len(no_d) * len(ss.SWEPT_KS) * len(ss.SWEPT_NEIGHBOUR_SCHEMES) == 138
    no_d = [s for s in fast if s.space != 'hybrid__distance1']
    assert len(no_d) * len(ss.SWEPT_KS) * len(ss.SWEPT_NEIGHBOUR_SCHEMES) == 66


def test_no_smiles_drops_tanimoto():
    sp = ss.group_spaces('full_knn', has_smiles=False)
    assert ss.TANIMOTO_SPACE not in sp
    assert len(ss.enumerate_settings(sp, group='full_knn')) == 25


def test_knn_group_rule():
    assert ss.knn_group('herg_ki', 1412) == 'full_knn'
    assert ss.knn_group('logd', 5028) == 'fast_knn'
    assert ss.knn_group('logd') == 'fast_knn'
    assert ss.knn_group('caco2') == 'full_knn'
    # the count rule beats the list
    assert ss.knn_group('herg_ic50', 9802) == 'fast_knn'


def test_settings_from_env_raises_on_cut_names():
    for name in ('base_1|treeshap|manhattan', 'base_1|none|euclidean',
                 'base_1|none|correlation'):
        with pytest.raises(ValueError):
            ss.settings_from_env(env={'KIRBY_SIMILARITY_SETTINGS': name})
    for name in ('base_2|none|manhattan', 'base_1|none|minkowski_p0.5',
                 'tanimoto_ecfp4|none|tanimoto'):
        with pytest.raises(ValueError):
            ss.settings_from_env(env={'KIRBY_SIMILARITY_SETTINGS': name},
                                 group='fast_knn')
    got = ss.settings_from_env(
        env={'KIRBY_SIMILARITY_SETTINGS': 'model_input,hybrid__distance1|none|cosine'})
    assert [s.name for s in got] == ['model_input|none|process',
                                     'hybrid__distance1|none|cosine']


def test_neighbour_weights_two_schemes():
    d = np.array([[1.0, 2.0, 4.0]])
    assert np.allclose(ss.neighbour_weights(d, 'rank'), [[1, 1 / 2, 1 / 3]])
    assert np.allclose(ss.neighbour_weights(d, 'inverse_square'),
                       [[1, 1 / 4, 1 / 16]])
    with pytest.raises(ValueError):
        ss.neighbour_weights(d, 'uniform')
    y = np.array([0.0, 1.0, 2.0, 4.0])
    idx = np.array([[1, 2, 3]])
    got = ss.weighted_neighbour_label(y, idx, d, 'inverse_square')
    assert np.isclose(got[0], (1 * 1 + 2 / 4 + 4 / 16) / (1 + 1 / 4 + 1 / 16))


def test_cosine_on_a_zero_row_is_nobodys_neighbour():
    Z = np.random.RandomState(0).randn(12, 5)
    Z[3] = 0.0
    out = ss.neighbour_lists(ss.SimilaritySetting('base_1', 'none', 'cosine'),
                             spaces_train={'base_1': Z}, spaces_val=None,
                             y_train=np.zeros(12))
    k = out['idx_train'].shape[1]
    others = np.delete(out['idx_train'], 3, axis=0)
    # row 3 sits at +inf from everyone, so it comes after all 10 finite ones
    assert k == 11 and not np.any(others[:, :10] == 3)


def test_a_row_tied_at_inf_never_lists_itself():
    # an all-zero row under cosine is +inf from everyone, its own column too;
    # it must take the other rows, never itself
    Z = np.random.RandomState(0).randn(12, 5)
    Z[3] = 0.0
    out = ss.neighbour_lists(ss.SimilaritySetting('base_1', 'none', 'cosine'),
                             spaces_train={'base_1': Z}, spaces_val=None,
                             y_train=np.zeros(12))
    for i, row in enumerate(out['idx_train']):
        assert i not in row


def test_cdist_old_name_never_hands_out_a_bare_minkowski():
    # n32 passes _CDIST[name] to cdist; minkowski without p=0.5 would be p=2
    assert ss._CDIST == {'manhattan': 'cityblock', 'cosine': 'cosine'}
