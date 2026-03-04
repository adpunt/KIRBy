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
    FEATURE_NAME_REGISTRY,
)

from kirby.importance.feature_names import (
    get_pdv_names,
    get_pca_names,
    get_protein_aac_names,
    get_protein_descriptor_names,
    get_antibody_aac_names,
    get_structural_names,
    get_hallmark_names,
    get_progeny_names,
    get_immune_names,
    get_gene_stats_names,
    get_chromosome_arm_names,
    get_feature_description,
    FEATURE_DESCRIPTIONS,
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
    'FEATURE_NAME_REGISTRY',
    # Feature names (most commonly used)
    'get_pdv_names',
    'get_pca_names',
    'get_protein_aac_names',
    'get_protein_descriptor_names',
    'get_antibody_aac_names',
    'get_structural_names',
    'get_hallmark_names',
    'get_progeny_names',
    'get_immune_names',
    'get_gene_stats_names',
    'get_chromosome_arm_names',
    'get_feature_description',
    'FEATURE_DESCRIPTIONS',
]
