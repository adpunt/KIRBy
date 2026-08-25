"""Structural-similarity engine (MCS overlap) for the KIRBy framework.

Provides the label-blind maximum-common-substructure ground truth and its
per-dataset pair cache, on which the structure-tuned similarity recipes are
selected and ranked.
"""
from kirby.similarity.mcs import (
    mcs_distance,
    compute_mcs_pairs,
    get_mcs_pairs,
    restrict_pairs_to_rows,
)

__all__ = [
    'mcs_distance',
    'compute_mcs_pairs',
    'get_mcs_pairs',
    'restrict_pairs_to_rows',
]
