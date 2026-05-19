"""Smoke tests for the organism kwarg on cox2/drd2 loaders (T-Mol.2d).

Network-free: only the organism->target-ID resolution and ValueError
guards are exercised. The actual ChEMBL download is covered on the server.
"""

import pytest

from kirby.datasets.cox2 import COX2_TARGET_IDS, load_cox2_chembl
from kirby.datasets.drd2 import DRD2_TARGET_IDS, load_drd2_chembl


def test_cox2_target_registry():
    # Resolved 2026-05-13; cat/horse excluded (zero activities).
    assert COX2_TARGET_IDS == {
        'human': 'CHEMBL230',
        'dog': 'CHEMBL4033',
        'cow': 'CHEMBL3331',
    }


def test_drd2_target_registry():
    # Resolved 2026-05-13; dog/cat/horse excluded (no usable Ki data).
    assert DRD2_TARGET_IDS == {
        'human': 'CHEMBL217',
        'cow': 'CHEMBL3998',
    }


def test_cox2_unsupported_organism_raises():
    for organism in ('cat', 'horse', 'mouse', 'gibberish'):
        with pytest.raises(ValueError, match='COX-2'):
            load_cox2_chembl(organism=organism)


def test_drd2_unsupported_organism_raises():
    for organism in ('dog', 'cat', 'horse', 'gibberish'):
        with pytest.raises(ValueError, match='DRD2'):
            load_drd2_chembl(organism=organism)


def test_organism_case_insensitive():
    # The loader lowercases organism before the registry lookup, so an
    # uppercase-but-unsupported value still raises the same ValueError
    # (no accidental bypass via case mismatch).
    with pytest.raises(ValueError, match='COX-2'):
        load_cox2_chembl(organism='CAT')


if __name__ == '__main__':
    test_cox2_target_registry()
    test_drd2_target_registry()
    test_cox2_unsupported_organism_raises()
    test_drd2_unsupported_organism_raises()
    test_organism_case_insensitive()
    print("OK")
