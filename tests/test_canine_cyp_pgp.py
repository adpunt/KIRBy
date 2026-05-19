"""Smoke tests for canine_cyp + canine_pgp loaders (T-Mol.3).

Network-free: only the CSV fallback path and ValueError / sparse-skip
guards are exercised. ChEMBL queries are covered separately when
``docs/cross_species_targets_resolved.json`` is populated for the new
genes (CYP1A2/2B11/2C21/2D15/3A12/ABCB1) on the server.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kirby.datasets import canine_cyp as cyp_mod
from kirby.datasets import canine_pgp as pgp_mod
from kirby.datasets.canine_cyp import (
    CANINE_CYP_ISOFORMS,
    SPARSE_THRESHOLD,
    load_canine_cyp,
    load_all_canine_cyp,
)
from kirby.datasets.canine_pgp import load_canine_pgp


def test_isoform_list():
    assert CANINE_CYP_ISOFORMS == ['1A2', '2B11', '2C21', '2D15', '3A12']


def test_unknown_isoform_raises():
    with pytest.raises(ValueError, match='Unknown canine CYP isoform'):
        load_canine_cyp('9X9')


def test_unknown_organism_raises():
    with pytest.raises(ValueError, match='Unknown species'):
        load_canine_cyp('3A12', organism='gibberish')


def test_bad_prefer_raises():
    with pytest.raises(ValueError, match="prefer must be"):
        load_canine_cyp('3A12', prefer='ZZZ')


def test_documented_skip_when_no_data(tmp_path, capsys):
    # tmp_path has no canine_cyp_*.csv, JSON lookup likely returns None for
    # CYP3A12/dog (not yet resolved) — expect 'none' result + warning.
    result = load_canine_cyp('3A12', organism='dog', csv_dir=tmp_path)
    out = capsys.readouterr().out
    assert result['source'] == 'none'
    assert result['n_compounds'] == 0
    assert result['organism'] == 'dog'
    assert result['isoform'] == '3A12'
    assert 'Documented skip' in out


def test_csv_fallback_works(tmp_path, capsys):
    csv = tmp_path / 'canine_cyp_3a12.csv'
    smiles = [
        'CCO', 'CCCO', 'CCC', 'CC(=O)O', 'CC(=O)C', 'CCN', 'CCS', 'CCBr',
        'CCCl', 'CCF',
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    pd.DataFrame({
        'smiles': smiles,
        'label': labels,
        'reference': ['PMID:test'] * len(smiles),
    }).to_csv(csv, index=False)

    result = load_canine_cyp('3A12', organism='dog', csv_dir=tmp_path)
    out = capsys.readouterr().out
    assert result['source'] == 'csv'
    assert result['n_compounds'] == 10
    assert result['source_path'] == str(csv)
    assert result['smiles'] == smiles
    assert np.array_equal(result['labels'], np.asarray(labels, dtype=float))
    assert result['extras']['reference'] == ['PMID:test'] * 10
    # 10 < SPARSE_THRESHOLD=20 → sparse-skip warning printed.
    assert f'below the {SPARSE_THRESHOLD}-row inclusion threshold' in out


def test_csv_passes_through_when_above_threshold(tmp_path, capsys):
    csv = tmp_path / 'canine_cyp_2d15_dog.csv'
    n = 25
    pd.DataFrame({
        'smiles': [f'C{"C" * (i + 1)}' for i in range(n)],
        'label': np.linspace(4.0, 9.0, n),
    }).to_csv(csv, index=False)

    result = load_canine_cyp('2D15', organism='dog', csv_dir=tmp_path)
    out = capsys.readouterr().out
    assert result['source'] == 'csv'
    assert result['n_compounds'] == n
    assert 'below the' not in out  # no sparse warning


def test_organism_specific_csv_wins(tmp_path):
    # Both organism-agnostic and organism-suffixed CSVs exist;
    # the suffixed one (more specific) should be picked first.
    (tmp_path / 'canine_cyp_3a12.csv').write_text(
        'smiles,label\nCCO,1\n')
    (tmp_path / 'canine_cyp_3a12_dog.csv').write_text(
        'smiles,label\nCCN,0\nCCS,1\n')
    result = load_canine_cyp('3A12', organism='dog', csv_dir=tmp_path)
    assert result['n_compounds'] == 2
    assert result['smiles'] == ['CCN', 'CCS']


def test_csv_missing_columns_raises(tmp_path):
    csv = tmp_path / 'canine_cyp_1a2.csv'
    pd.DataFrame({'smiles': ['CCO'], 'value': [1.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="must contain 'smiles' and 'label'"):
        load_canine_cyp('1A2', csv_dir=tmp_path)


def test_load_all_canine_cyp_returns_all(tmp_path, capsys):
    result = load_all_canine_cyp(organism='dog')
    capsys.readouterr()
    assert set(result.keys()) == set(CANINE_CYP_ISOFORMS)
    for iso in CANINE_CYP_ISOFORMS:
        assert result[iso]['isoform'] == iso


def test_pgp_documented_skip(tmp_path, capsys):
    result = load_canine_pgp(organism='dog', csv_dir=tmp_path)
    out = capsys.readouterr().out
    assert result['source'] == 'none'
    assert result['n_compounds'] == 0
    assert result['gene'] == 'ABCB1'
    assert 'Documented skip' in out


def test_pgp_csv_fallback(tmp_path, capsys):
    csv = tmp_path / 'canine_pgp.csv'
    pd.DataFrame({
        'smiles': ['CCO', 'CCN', 'CCS'],
        'label': [1, 0, 1],
        'assay_type': ['substrate', 'non-substrate', 'inhibitor'],
    }).to_csv(csv, index=False)
    result = load_canine_pgp(organism='dog', csv_dir=tmp_path)
    capsys.readouterr()
    assert result['source'] == 'csv'
    assert result['n_compounds'] == 3
    assert result['source_path'] == str(csv)
    assert result['extras']['assay_type'] == [
        'substrate', 'non-substrate', 'inhibitor']


def test_pgp_unknown_organism_raises(tmp_path):
    with pytest.raises(ValueError, match='Unknown species'):
        load_canine_pgp(organism='gibberish', csv_dir=tmp_path)


def test_pgp_bad_prefer_raises(tmp_path):
    with pytest.raises(ValueError, match="prefer must be"):
        load_canine_pgp(prefer='ZZZ', csv_dir=tmp_path)


def test_pgp_csv_missing_columns_raises(tmp_path):
    csv = tmp_path / 'canine_pgp.csv'
    pd.DataFrame({'smiles': ['CCO'], 'value': [1.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="must contain 'smiles' and 'label'"):
        load_canine_pgp(csv_dir=tmp_path)
