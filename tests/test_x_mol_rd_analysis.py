"""Network-free tests for the T-Mol.5 R&D analysis pipeline.

The cache-loading and selection helpers are exercised against in-memory
``SearchResult`` mocks (no pickle round-trip; no `find_best_hybrid` run).
The liability-flag helpers are exercised against synthetic prediction
dicts. The end-to-end candidate path is NOT exercised here — that lives
on the server.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))

from experiments.cross_species.x_mol_rd_analysis import (  # noqa: E402
    _search_cache_path,
    load_search_result,
    select_blueprint,
    flag_out_of_domain,
    flag_below_potency_floor,
    flag_species_divergence,
    _parse_potency_floors,
    _calibration_split,
    _apply_selection,
    _reps_for,
)


class _SR:
    """Minimal stand-in for find_best_hybrid.SearchResult.

    `best_by_model` is the level at which `fitted_results` is populated
    (return_fitted=True picks the best across budgets per model). The
    cross-budget grid `best_by_model_budget` is also kept on the dataclass
    but the analysis path doesn't read it.
    """

    def __init__(self, best_by_model, lower_is_better=False,
                 best_by_model_budget=None):
        self.best_by_model = best_by_model
        self.best_by_model_budget = best_by_model_budget or {}
        self.lower_is_better = lower_is_better
        self.fitted_results = {}


def _rec(model, budget, score, blueprint='BP'):
    # ``blueprint`` is opaque to the cache loader; we use a string sentinel.
    return {
        'model': model, 'budget': budget, 'score_mean': score,
        'score_std': 0.01, 'n_features': 32, 'blueprint': blueprint,
    }


# --------------------------------------------------------------------------
# Cache path + loader
# --------------------------------------------------------------------------

def test_search_cache_path_matches_t_mol_4_layout(tmp_path, monkeypatch):
    monkeypatch.setenv('KIRBY_CACHE_DIR', str(tmp_path))
    p = _search_cache_path('cox2')
    assert p.name == 'x_mol_species_admet_cox2__general__uncapped__full.pkl'
    assert p.parent.name == 'searches'
    assert p.parent.parent == tmp_path


def test_load_search_result_returns_none_on_missing(tmp_path, monkeypatch):
    monkeypatch.setenv('KIRBY_CACHE_DIR', str(tmp_path))
    assert load_search_result('does_not_exist') is None


def test_load_search_result_returns_none_on_corrupt(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('KIRBY_CACHE_DIR', str(tmp_path))
    p = _search_cache_path('garbage')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b'not a pickle')
    assert load_search_result('garbage') is None
    out = capsys.readouterr().out
    assert 'unpickle failed' in out


def test_load_search_result_returns_none_on_empty_payload(tmp_path, monkeypatch):
    monkeypatch.setenv('KIRBY_CACHE_DIR', str(tmp_path))
    p = _search_cache_path('empty')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps({'_cache_meta': {}, 'payload': _SR({})}))
    assert load_search_result('empty') is None  # no blueprints in the cache


def test_load_search_result_unwraps_envelope(tmp_path, monkeypatch):
    monkeypatch.setenv('KIRBY_CACHE_DIR', str(tmp_path))
    sr = _SR({'rf': _rec('rf', 32, 0.5)})
    p = _search_cache_path('ok')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps({'_cache_meta': {}, 'payload': sr}))
    out = load_search_result('ok')
    # pickle round-trips to a fresh object — check shape, not identity.
    assert out is not None
    assert list(out.best_by_model) == ['rf']
    assert out.best_by_model['rf']['score_mean'] == 0.5


def test_load_search_result_accepts_bare_object(tmp_path, monkeypatch):
    # Some older caches were written without the {'_cache_meta', 'payload'}
    # envelope. Loader should still accept those by sniffing the attr.
    monkeypatch.setenv('KIRBY_CACHE_DIR', str(tmp_path))
    sr = _SR({'rf': _rec('rf', 32, 0.5)})
    p = _search_cache_path('bare')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(sr))
    out = load_search_result('bare')
    assert out is not None
    assert list(out.best_by_model) == ['rf']


# --------------------------------------------------------------------------
# Blueprint selection
# --------------------------------------------------------------------------

def test_select_blueprint_higher_is_better():
    sr = _SR({
        'rf':       _rec('rf',       32, 0.6),
        'lightgbm': _rec('lightgbm', 32, 0.55),
    }, lower_is_better=False)
    best = select_blueprint(sr)
    assert best['model'] == 'rf' and best['budget'] == 32
    assert best['score_mean'] == pytest.approx(0.6)


def test_select_blueprint_lower_is_better_picks_smallest():
    # With RMSE-style metric the smallest score wins.
    sr = _SR({
        'rf':       _rec('rf',       32, 0.6),
        'lightgbm': _rec('lightgbm', 16, 0.5),
    }, lower_is_better=True)
    best = select_blueprint(sr)
    assert best['model'] == 'lightgbm' and best['budget'] == 16


def test_select_blueprint_returns_none_on_empty():
    assert select_blueprint(_SR({})) is None


# --------------------------------------------------------------------------
# Liability flags
# --------------------------------------------------------------------------

def test_flag_out_of_domain_triggers_on_top_bin_only():
    pred = {'n_bins': 4, 'bin': [0, 1, 2]}
    assert flag_out_of_domain(pred) is False
    pred = {'n_bins': 4, 'bin': [0, 3]}
    assert flag_out_of_domain(pred) is True
    pred = {'n_bins': 1, 'bin': [0]}      # single global bin → top == 0
    assert flag_out_of_domain(pred) is True


def test_flag_below_potency_floor_respects_none():
    pred = {'lower': [4.5, 5.5]}
    assert flag_below_potency_floor(pred, None) is False
    assert flag_below_potency_floor(pred, 5.0) is True
    assert flag_below_potency_floor(pred, 4.0) is False


def test_flag_species_divergence_overlap_logic():
    # Overlapping intervals → no divergence.
    pred = {'lower': [4.0, 4.5], 'upper': [5.5, 5.8]}
    assert flag_species_divergence(pred) is False
    # Touching at a single point → still overlap (a<=d and c<=b).
    pred = {'lower': [4.0, 5.0], 'upper': [5.0, 6.0]}
    assert flag_species_divergence(pred) is False
    # Clear gap → divergence.
    pred = {'lower': [4.0, 6.0], 'upper': [5.0, 7.0]}
    assert flag_species_divergence(pred) is True
    # Three species, one outlier pair → divergence.
    pred = {'lower': [4.0, 4.5, 7.0], 'upper': [5.0, 5.5, 8.0]}
    assert flag_species_divergence(pred) is True


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def test_parse_potency_floors_happy_and_empty():
    assert _parse_potency_floors(None) == {}
    assert _parse_potency_floors('') == {}
    assert _parse_potency_floors('cox2=5.0,drd2=6.5') == {
        'cox2': 5.0, 'drd2': 6.5}
    # Whitespace tolerated.
    assert _parse_potency_floors(' cox2 = 5.0 , drd2 = 6.5 ') == {
        'cox2': 5.0, 'drd2': 6.5}


def test_parse_potency_floors_rejects_bad_pair():
    with pytest.raises(ValueError, match='task=value'):
        _parse_potency_floors('cox2')


def test_apply_selection_concats_selected_indices_and_aug(monkeypatch):
    # Stub out the rep generator so we can test the index/aug plumbing.
    import experiments.cross_species.x_mol_rd_analysis as mod

    def fake_reps(smiles, requested):
        n = len(smiles)
        return {
            'foo': np.arange(n * 4, dtype=np.float32).reshape(n, 4),
            'bar': np.full((n, 3), 9.0, dtype=np.float32),
        }
    monkeypatch.setattr(mod, 'get_representations', fake_reps)

    feature_info = {
        'foo': {'selected_indices': np.array([0, 2])},
        'bar': {'selected_indices': np.array([1])},
        'augmentation_info': {
            'species_token': {'selected_indices': np.array([0, 3])}},
    }
    smiles = ['CCO', 'CCN']  # n=2
    organisms = ['human', 'dog']
    X = _apply_selection(_reps_for(smiles, ['foo', 'bar']),
                         organisms, feature_info)
    # foo selects cols 0,2 -> [[0,2],[4,6]]
    # bar selects col 1 -> [[9],[9]]
    # species_token: human=[1,0,0,0,0], dog=[0,1,0,0,0] — selected_indices [0,3] picks
    # ([1,0],[0,0]). So row 0 -> 1,0; row 1 -> 0,0.
    expected = np.array([
        [0, 2, 9, 1, 0],
        [4, 6, 9, 0, 0],
    ], dtype=np.float32)
    np.testing.assert_array_equal(X, expected)


def test_apply_selection_raises_on_missing_aug(monkeypatch):
    import experiments.cross_species.x_mol_rd_analysis as mod
    monkeypatch.setattr(mod, 'get_representations', lambda s, r: {
        'foo': np.zeros((len(s), 4), dtype=np.float32)})
    # An aug index out of bounds is a hard error — propagates from
    # apply_augmentation_selection.
    feature_info = {
        'foo': {'selected_indices': np.array([0])},
        'augmentation_info': {
            'species_token': {'selected_indices': np.array([99])}},
    }
    with pytest.raises(ValueError, match='out of bounds'):
        _apply_selection(_reps_for(['CCO'], ['foo']), ['human'], feature_info)


def test_reps_for_raises_when_a_rep_cannot_be_built(monkeypatch):
    import experiments.cross_species.x_mol_rd_analysis as mod
    monkeypatch.setattr(mod, 'get_representations', lambda s, r: {
        'foo': np.zeros((len(s), 4), dtype=np.float32)})
    with pytest.raises(RuntimeError, match='base reps missing'):
        _reps_for(['CCO'], ['foo', 'gone'])


def test_calibration_split_uses_groupkfold_and_clamps():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 5).astype(int)
    rng.shuffle(groups)
    train_idx, calib_idx, cover_idx, n_folds = _calibration_split(
        groups, n_folds=5)
    assert n_folds == 5
    # Three blocks, no row and no scaffold group shared between any two.
    assert len(set(train_idx) & set(calib_idx)) == 0
    assert len(set(train_idx) & set(cover_idx)) == 0
    assert len(set(calib_idx) & set(cover_idx)) == 0
    assert not (set(groups[train_idx]) & set(groups[calib_idx]))
    assert not (set(groups[train_idx]) & set(groups[cover_idx]))
    assert not (set(groups[calib_idx]) & set(groups[cover_idx]))
    assert len(train_idx) + len(calib_idx) + len(cover_idx) == len(groups)
    # Asking for too many folds is clamped to the number of unique groups.
    groups_few = np.array([0, 0, 1, 1, 2, 2])
    _, _, _, n_folds = _calibration_split(groups_few, n_folds=10)
    assert n_folds == 3
    # Two scaffold groups cannot make three blocks.
    with pytest.raises(ValueError, match='three scaffold-disjoint blocks'):
        _calibration_split(np.array([0, 0, 1, 1]), n_folds=5)
