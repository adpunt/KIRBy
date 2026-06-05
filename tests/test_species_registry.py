"""Smoke test for the cross-species registry (T-Mol.2a acceptance)."""

from kirby.utils.species import SPECIES_REGISTRY, resolve_species, chembl_organism


def test_registry_acceptance():
    assert SPECIES_REGISTRY['dog']['taxonomy_id'] == 9615
    assert chembl_organism('cat') == 'Felis catus'
    assert resolve_species('HORSE')['taxonomy_id'] == 9796


if __name__ == '__main__':
    test_registry_acceptance()
    print("OK")
