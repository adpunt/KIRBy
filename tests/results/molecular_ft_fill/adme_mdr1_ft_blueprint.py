"""
Auto-generated hybrid blueprints.

Usage:
    from results_blueprint import create_rf_hybrid
    X = create_rf_hybrid(smiles_list, labels)
"""

from kirby.hybrid import create_hybrid, apply_feature_selection
from kirby.representations.molecular import (
    create_ecfp4, create_maccs, create_pdv,
    create_pdv_binary_mean, create_pdv_rank,
    create_mol2vec,
    create_chemberta, create_molformer, create_selformer,
    create_chembert, create_molclr, create_graphmvp, create_smited,
    create_mhg_gnn, create_grover, create_graph_kernel,
    create_unimol, create_schnet,
)


# Representation generators
REP_GENERATORS = {
    'ecfp4': create_ecfp4,
    'maccs': create_maccs,
    'pdv': create_pdv,
    'pdv_binary_mean': create_pdv_binary_mean,
    'pdv_rank': create_pdv_rank,
    'mol2vec': create_mol2vec,
    'chemberta': create_chemberta,
    'molformer': create_molformer,
    'selformer': create_selformer,
    'chembert': create_chembert,
    'molclr': create_molclr,
    'graphmvp': create_graphmvp,
    'smited': create_smited,
    'mhg_gnn': create_mhg_gnn,
    'grover': create_grover,
    'graph_kernel': create_graph_kernel,
    'unimol': create_unimol,
    'schnet': create_schnet,
}


def create_rf_hybrid(smiles_list, labels):
    """
    Create the optimal hybrid for RF model.
    
    Blueprint: method=entropy_1.0, budget=150, importance=lightgbm_gain
    Expected dimension: ~150
    Expected score: 0.4804
    """
    # Generate required representations
    reps = {}
    reps['ecfp4'] = REP_GENERATORS['ecfp4'](smiles_list)
    reps['maccs'] = REP_GENERATORS['maccs'](smiles_list)
    reps['pdv'] = REP_GENERATORS['pdv'](smiles_list)
    reps['lora_molformer'] = REP_GENERATORS['lora_molformer'](smiles_list)
    reps['contrastive_molformer'] = REP_GENERATORS['contrastive_molformer'](smiles_list)
    reps['gnn_graphmvp'] = REP_GENERATORS['gnn_graphmvp'](smiles_list)
    
    # Create hybrid with optimal settings
    X_hybrid, feature_info = create_hybrid(
        reps, labels,
        allocation_method='entropy_1.0',
        budget=150,
        importance_method='lightgbm_gain',
    )
    
    return X_hybrid, feature_info


def create_xgboost_hybrid(smiles_list, labels):
    """
    Create the optimal hybrid for XGBOOST model.
    
    Blueprint: method=entropy_1.0, budget=300, importance=treeshap
    Expected dimension: ~300
    Expected score: 0.4954
    """
    # Generate required representations
    reps = {}
    reps['ecfp4'] = REP_GENERATORS['ecfp4'](smiles_list)
    reps['maccs'] = REP_GENERATORS['maccs'](smiles_list)
    reps['pdv'] = REP_GENERATORS['pdv'](smiles_list)
    reps['lora_molformer'] = REP_GENERATORS['lora_molformer'](smiles_list)
    reps['contrastive_molformer'] = REP_GENERATORS['contrastive_molformer'](smiles_list)
    reps['gnn_graphmvp'] = REP_GENERATORS['gnn_graphmvp'](smiles_list)
    
    # Create hybrid with optimal settings
    X_hybrid, feature_info = create_hybrid(
        reps, labels,
        allocation_method='entropy_1.0',
        budget=300,
        importance_method='treeshap',
    )
    
    return X_hybrid, feature_info

