"""Network-free smoke tests for the T-Mol.4 species-token assembly logic.

Covers the data-plumbing helpers in
``tests/experiments/cross_species/x_mol_species_admet.py``:
species-one-hot encoding, `_assemble_task` concatenation + sparse skipping,
and the canonical `admet_tasks` registry. The actual `find_best_hybrid`
path is exercised on the server (dry-run flag included in the script).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Mirror the experiment script's sys.path setup so the imports work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))

from experiments.cross_species.x_mol_species_admet import (  # noqa: E402
    SPECIES_VOCAB,
    species_one_hot,
    admet_tasks,
    _assemble_task,
)


def test_species_vocab_order_stable():
    # Cache keys depend on this order; pin it explicitly.
    assert SPECIES_VOCAB == ['human', 'dog', 'cat', 'cow', 'horse']


def test_species_one_hot_shape_and_rows():
    oh = species_one_hot(['human', 'dog', 'cow', 'dog'])
    assert oh.shape == (4, 5)
    assert oh.dtype == np.float32
    assert np.all(oh.sum(axis=1) == 1.0)
    # Row 0 = human (index 0), row 1 = dog (index 1), row 2 = cow (index 3).
    assert oh[0].argmax() == 0
    assert oh[1].argmax() == 1
    assert oh[2].argmax() == 3
    assert oh[3].argmax() == 1


def test_species_one_hot_unknown_raises():
    with pytest.raises(ValueError, match='not in SPECIES_VOCAB'):
        species_one_hot(['mouse'])


def test_admet_tasks_canonical_list():
    names = [t[0] for t in admet_tasks()]
    assert names == [
        'cox2', 'drd2',
        'canine_cyp_1a2', 'canine_cyp_2b11', 'canine_cyp_2c21',
        'canine_cyp_2d15', 'canine_cyp_3a12',
        'canine_pgp',
    ]


def test_admet_tasks_canine_isoform_closure_binds_correctly():
    # Lambda + default-arg closure is the classic late-binding bug spot;
    # verify each canine_cyp_* task actually requests its own isoform when
    # the loader callable is invoked.
    captured = {}

    def fake_loader(isoform, organism):
        captured.setdefault(isoform, []).append(organism)
        return [], np.array([], dtype=float)

    # Monkey-patch the underlying loader and walk each task.
    import experiments.cross_species.x_mol_species_admet as mod
    real = mod._load_canine_cyp_slice
    mod._load_canine_cyp_slice = fake_loader
    try:
        for name, organisms, fn in admet_tasks():
            if not name.startswith('canine_cyp_'):
                continue
            for organism in organisms:
                fn(organism)
        # Each isoform should have been called once per organism.
        assert sorted(captured) == ['1A2', '2B11', '2C21', '2D15', '3A12']
    finally:
        mod._load_canine_cyp_slice = real


def test_assemble_task_returns_none_when_single_organism(capsys):
    def loader(organism):
        if organism == 'human':
            return ['CCO'] * 50, np.linspace(4.0, 8.0, 50)
        return [], np.array([], dtype=float)
    result = _assemble_task('demo', ['human', 'dog'], loader,
                            min_rows_per_organism=20)
    out = capsys.readouterr().out
    assert result is None
    assert 'cross-species training is moot' in out


def test_assemble_task_concatenates_and_tracks_per_organism(capsys):
    def loader(organism):
        if organism == 'human':
            return [f'C{i}' for i in range(30)], np.full(30, 5.0)
        if organism == 'dog':
            return [f'CC{i}' for i in range(25)], np.full(25, 6.0)
        return [], np.array([], dtype=float)
    result = _assemble_task('demo', ['human', 'dog', 'cow'], loader,
                            min_rows_per_organism=20)
    capsys.readouterr()
    assert result is not None
    assert len(result['smiles']) == 55
    assert len(result['labels']) == 55
    assert len(result['organisms']) == 55
    assert result['per_organism_counts'] == {'human': 30, 'dog': 25}
    # Per-row organism alignment with labels.
    human_mask = np.array([o == 'human' for o in result['organisms']])
    assert np.all(result['labels'][human_mask] == 5.0)
    assert np.all(result['labels'][~human_mask] == 6.0)


def test_assemble_task_drops_sparse_organism(capsys):
    def loader(organism):
        if organism == 'human':
            return [f'C{i}' for i in range(30)], np.full(30, 5.0)
        if organism == 'dog':
            return [f'CC{i}' for i in range(30)], np.full(30, 6.0)
        if organism == 'cow':
            return ['CCO'] * 5, np.full(5, 7.0)  # below 20-row threshold
        return [], np.array([], dtype=float)
    result = _assemble_task('demo', ['human', 'dog', 'cow'], loader,
                            min_rows_per_organism=20)
    out = capsys.readouterr().out
    assert 'cow' in out and 'sparse' in out
    assert result is not None
    assert set(result['per_organism_counts']) == {'human', 'dog'}


def test_assemble_task_loader_error_is_logged(capsys):
    def loader(organism):
        if organism == 'human':
            raise RuntimeError('synthetic ChEMBL outage')
        return [f'C{i}' for i in range(25)], np.full(25, 6.0)
    result = _assemble_task('demo', ['human', 'dog'], loader,
                            min_rows_per_organism=20)
    out = capsys.readouterr().out
    assert 'loader error' in out and 'synthetic ChEMBL outage' in out
    # Only one organism remains → task skipped.
    assert result is None


def test_assemble_task_mismatched_lengths_raises():
    def loader(organism):
        return ['CCO'] * 30, np.full(25, 5.0)  # 30 vs 25
    with pytest.raises(ValueError, match='length mismatch'):
        _assemble_task('demo', ['human', 'dog'], loader,
                       min_rows_per_organism=20)
