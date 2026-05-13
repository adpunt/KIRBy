"""
Species registry for cross-species KIRBy extensions.

Single source of truth mapping short species keys to ChEMBL organism strings
and NCBI taxonomy IDs. Used by ChEMBL-backed loaders (chembl_bioactivity,
herg, cox2, drd2) and downstream cross-species experiments.

See docs/cross_species_admet_design.md §4.
"""

from typing import Dict


SPECIES_REGISTRY: Dict[str, Dict[str, object]] = {
    'human': {'chembl_organism': 'Homo sapiens',       'taxonomy_id': 9606},
    'rat':   {'chembl_organism': 'Rattus norvegicus',  'taxonomy_id': 10116},
    'mouse': {'chembl_organism': 'Mus musculus',       'taxonomy_id': 10090},
    'dog':   {'chembl_organism': 'Canis familiaris',   'taxonomy_id': 9615},
    'cat':   {'chembl_organism': 'Felis catus',        'taxonomy_id': 9685},
    'cow':   {'chembl_organism': 'Bos taurus',         'taxonomy_id': 9913},
    'horse': {'chembl_organism': 'Equus caballus',     'taxonomy_id': 9796},
}

SPECIES_KEYS = frozenset(SPECIES_REGISTRY.keys())


def resolve_species(name: str) -> Dict[str, object]:
    """Return registry entry for `name` (e.g. 'dog'). Raises ValueError if unknown."""
    key = name.lower()
    if key not in SPECIES_REGISTRY:
        raise ValueError(
            f"Unknown species '{name}'. Known: {sorted(SPECIES_KEYS)}"
        )
    return SPECIES_REGISTRY[key]


def chembl_organism(name: str) -> str:
    """ChEMBL Latin binomial for `name` (e.g. 'dog' → 'Canis familiaris')."""
    return resolve_species(name)['chembl_organism']  # type: ignore[return-value]


def taxonomy_id(name: str) -> int:
    """NCBI taxonomy ID for `name` (e.g. 'dog' → 9615)."""
    return resolve_species(name)['taxonomy_id']  # type: ignore[return-value]
