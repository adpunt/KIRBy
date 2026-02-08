"""
Feature importance and attribution methods.

This module provides various methods for computing feature importance:
- General methods in kirby.hybrid.compute_feature_importance
- Molecule-specific methods in kirby.importance.molecular
- Hybrid explainability in kirby.importance.hybrid
"""

from kirby.importance.molecular import (
    get_fingerprint_bit_info,
    explain_fingerprint_bits,
    explain_smiles_tokens,
    substructure_shap,
    compute_substructure_features,
    count_substructure_occurrences,
    DEFAULT_FUNCTIONAL_GROUPS,
)

from kirby.importance.hybrid import (
    explain_hybrid,
    interpret_pdv_index,
    interpret_ecfp_bit,
    draw_ecfp_bit,
    aggregate_importance_by_rep,
    format_explanation_summary,
    visualize_molecule_importance,
)

__all__ = [
    # Molecular
    'get_fingerprint_bit_info',
    'explain_fingerprint_bits',
    'explain_smiles_tokens',
    'substructure_shap',
    'compute_substructure_features',
    'count_substructure_occurrences',
    'DEFAULT_FUNCTIONAL_GROUPS',
    # Hybrid
    'explain_hybrid',
    'interpret_pdv_index',
    'interpret_ecfp_bit',
    'draw_ecfp_bit',
    'aggregate_importance_by_rep',
    'format_explanation_summary',
    'visualize_molecule_importance',
]
